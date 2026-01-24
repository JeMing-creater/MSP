# gen_hovernet_maps.py
# ------------------------------------------------------------
# TIAToolbox HoVer-Net predict -> tmp/*.dat -> decode -> boundary PNG
#
# Fixes your exact failure mode:
# - Many TIAToolbox versions do NOT store inst_map in .dat;
#   they store per-instance contours (arrays ending with ".contour" / containing "contour").
# - This script decodes:
#     1) inst_map if present
#     2) else contours -> draw polylines -> boundary map
#
# Outputs:
#   out_img_dir/hovernet_boundary_png/<slide>/<patch_stem>.png
#   out_img_dir/hovernet_boundary_png/<slide>/__debug_boundary_preview.png  (first success)
#   out_img_dir/hovernet_boundary_png/<slide>/.HOVERNET_DONE
#   out_root/__debug_artifact_report_<tmp_name>.txt (per chunk)
# ------------------------------------------------------------

import os
import sys
import time
import uuid
import glob
import shutil
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import joblib
from tqdm import tqdm

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.setrecursionlimit(20000)

from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.models.architecture import get_pretrained_model
from tiatoolbox.models.architecture.utils import compile_model


# -------------------------
# FS helpers
# -------------------------
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def atomic_touch(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def list_slide_dirs(hr_root: Path) -> List[Path]:
    slide_dirs = sorted([p for p in hr_root.iterdir() if p.is_dir()])
    return slide_dirs if slide_dirs else [hr_root]


def make_unique_tmp_dir(parent: Path, slide_id: str) -> Path:
    """Return a path that DOES NOT exist (do not mkdir)."""
    pid = os.getpid()
    ts = int(time.time())
    for _ in range(500):
        u = uuid.uuid4().hex[:10]
        p = parent / f"__tmp_{slide_id}_{pid}_{ts}_{u}"
        if not p.exists():
            return p
    raise RuntimeError("Failed to allocate a non-existing tmp dir name.")


def find_prediction_artifacts(tmp_save: Path) -> List[Path]:
    files = sorted(list(tmp_save.rglob("*.dat")))
    files += sorted(list(tmp_save.rglob("*.joblib")))
    # de-dup
    seen = set()
    uniq = []
    for f in files:
        s = f.as_posix()
        if s not in seen:
            uniq.append(f)
            seen.add(s)
    return uniq


# -------------------------
# Boundary ops
# -------------------------
def dilate_binary(img: np.ndarray, k: int) -> np.ndarray:
    if k is None or k <= 1:
        return img
    kk = int(k)
    if kk % 2 == 0:
        kk += 1
    kernel = np.ones((kk, kk), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


def inst_map_to_boundary(inst_map: np.ndarray, dilate: int = 3) -> np.ndarray:
    inst_map = inst_map.astype(np.int32, copy=False)
    H, W = inst_map.shape
    boundary = np.zeros((H, W), dtype=np.uint8)

    ids = np.unique(inst_map)
    ids = ids[ids != 0]
    for _id in ids:
        m = (inst_map == _id).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            cv2.drawContours(boundary, cnts, -1, 255, thickness=1)

    boundary = dilate_binary(boundary, dilate)
    return boundary


def contours_to_boundary(
    contours: List[np.ndarray],
    canvas_hw: Tuple[int, int],
    dilate: int = 3,
) -> np.ndarray:
    H, W = int(canvas_hw[0]), int(canvas_hw[1])
    boundary = np.zeros((H, W), dtype=np.uint8)

    # Each contour: Nx2 (x,y) or Nx1x2 or 2xN...
    for c in contours:
        if c is None:
            continue
        c = np.asarray(c)
        if c.size < 6:
            continue

        # normalize shape to Nx2
        if c.ndim == 3 and c.shape[-1] == 2:
            pts = c.reshape(-1, 2)
        elif c.ndim == 2 and c.shape[1] == 2:
            pts = c
        elif c.ndim == 2 and c.shape[0] == 2:  # 2xN
            pts = c.T
        else:
            continue

        pts = np.round(pts).astype(np.int32)

        # clip
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)

        # OpenCV wants shape Nx1x2
        pts_cv = pts.reshape(-1, 1, 2)
        cv2.polylines(boundary, [pts_cv], isClosed=True, color=255, thickness=1)

    boundary = dilate_binary(boundary, dilate)
    return boundary


# -------------------------
# Artifact mapping
# -------------------------
def artifact_index(artifact_path: Path) -> Optional[int]:
    stem = artifact_path.stem
    return int(stem) if stem.isdigit() else None


def artifact_to_patch_stem_and_src(artifact_path: Path, sub_pngs: List[str]) -> Tuple[Optional[str], Optional[str]]:
    idx = artifact_index(artifact_path)
    if idx is not None and 0 <= idx < len(sub_pngs):
        src = sub_pngs[idx]
        return Path(src).stem, src
    # fallback: try exact stem match
    a = artifact_path.stem
    for p in sub_pngs:
        if Path(p).stem == a:
            return a, p
    return None, None


# -------------------------
# Recursive extractors
# -------------------------
def _iter_items(obj: Any, prefix: str = "", max_depth: int = 8, depth: int = 0):
    """Yield (path, value) recursively."""
    if depth > max_depth:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{prefix}.{k}" if prefix else str(k)
            yield kp, v
            yield from _iter_items(v, kp, max_depth=max_depth, depth=depth + 1)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            kp = f"{prefix}[{i}]" if prefix else f"[{i}]"
            yield kp, v
            yield from _iter_items(v, kp, max_depth=max_depth, depth=depth + 1)
    else:
        return


def try_extract_inst_map(obj: Any) -> Optional[np.ndarray]:
    """
    Conservative: only accept a real 2D integer map with background zeros.
    """
    # fast known keys
    if isinstance(obj, dict):
        for k in ("inst_map", "instance_map"):
            if k in obj and isinstance(obj[k], np.ndarray) and obj[k].ndim == 2:
                a = obj[k]
                if np.issubdtype(a.dtype, np.integer):
                    if (a == 0).mean() > 0.05 and a.max() > 0:
                        return a.astype(np.int32, copy=False)

        for k in ("inst_dict", "pred", "result", "output", "pred_inst"):
            if k in obj and isinstance(obj[k], dict):
                for kk in ("inst_map", "instance_map"):
                    if kk in obj[k] and isinstance(obj[k][kk], np.ndarray) and obj[k][kk].ndim == 2:
                        a = obj[k][kk]
                        if np.issubdtype(a.dtype, np.integer):
                            if (a == 0).mean() > 0.05 and a.max() > 0:
                                return a.astype(np.int32, copy=False)

    # fallback scan: find first 2D integer-ish map with many zeros and max>0
    for p, v in _iter_items(obj):
        if isinstance(v, np.ndarray) and v.ndim == 2:
            if np.issubdtype(v.dtype, np.integer):
                if v.size >= 256 and (v == 0).mean() > 0.05 and int(v.max()) > 0 and int(v.max()) < 50000:
                    return v.astype(np.int32, copy=False)
    return None


def extract_contours(obj: Any) -> List[np.ndarray]:
    """
    Extract contour coordinate arrays.
    We accept arrays that look like Nx2 coordinates and whose path contains 'contour'.
    """
    contours: List[np.ndarray] = []
    for p, v in _iter_items(obj):
        if "contour" not in p.lower():
            continue
        if isinstance(v, np.ndarray):
            a = v
            # accept coordinate-like arrays
            if a.ndim == 2 and (a.shape[1] == 2 or a.shape[0] == 2) and a.size >= 6:
                contours.append(a)
            elif a.ndim == 3 and a.shape[-1] == 2 and a.size >= 6:
                contours.append(a)

        # sometimes contour is stored as list of arrays
        if isinstance(v, (list, tuple)):
            for vv in v:
                if isinstance(vv, np.ndarray):
                    a = vv
                    if a.ndim == 2 and (a.shape[1] == 2 or a.shape[0] == 2) and a.size >= 6:
                        contours.append(a)
                    elif a.ndim == 3 and a.shape[-1] == 2 and a.size >= 6:
                        contours.append(a)
    return contours


def infer_canvas_hw_from_contours(contours: List[np.ndarray], fallback_hw: Tuple[int, int]) -> Tuple[int, int]:
    """
    If contours are in output resolution (e.g., 420x420), infer H/W by max coord.
    Otherwise fallback to source patch size.
    """
    if not contours:
        return fallback_hw
    max_x = 0
    max_y = 0
    for c in contours[: min(len(contours), 2000)]:
        a = np.asarray(c).reshape(-1, 2)
        if a.size == 0:
            continue
        max_x = max(max_x, int(np.nanmax(a[:, 0])))
        max_y = max(max_y, int(np.nanmax(a[:, 1])))
    # if max coords look plausible
    H = max_y + 1
    W = max_x + 1
    if H >= 64 and W >= 64 and H <= 5000 and W <= 5000:
        return (H, W)
    return fallback_hw


# -------------------------
# Decode artifact -> boundary
# -------------------------
def decode_artifact_to_boundary(
    artifact_path: Path,
    src_patch_path: Optional[str],
    dilate: int,
    report_lines: List[str],
) -> Tuple[Optional[np.ndarray], str]:
    """
    Returns (boundary, mode_used) where mode_used is 'inst_map' or 'contour' or 'none'.
    """
    obj = joblib.load(artifact_path)

    # Determine target output size from src patch (final output should match src)
    src_hw = None
    if src_patch_path is not None:
        img = cv2.imread(src_patch_path, cv2.IMREAD_COLOR)
        if img is not None:
            src_hw = (img.shape[0], img.shape[1])
    if src_hw is None:
        src_hw = (512, 512)

    # 1) try inst_map
    inst_map = try_extract_inst_map(obj)
    if inst_map is not None:
        bnd = inst_map_to_boundary(inst_map, dilate=dilate)
        # resize to src if needed
        if bnd.shape[:2] != src_hw:
            bnd = cv2.resize(bnd, (src_hw[1], src_hw[0]), interpolation=cv2.INTER_NEAREST)
        # stats
        uniq = np.unique(inst_map)
        report_lines.append(
            f"  [USE inst_map] shape={inst_map.shape} min={int(inst_map.min())} max={int(inst_map.max())} "
            f"uniq={len(uniq)} bg_ratio={(inst_map==0).mean():.3f}"
        )
        return bnd, "inst_map"

    # 2) contour path (your current .dat looks like this)
    contours = extract_contours(obj)
    report_lines.append(f"  contour_count={len(contours)}")
    if contours:
        canvas_hw = infer_canvas_hw_from_contours(contours, fallback_hw=src_hw)
        bnd = contours_to_boundary(contours, canvas_hw=canvas_hw, dilate=dilate)

        # resize to src size (final alignment)
        if bnd.shape[:2] != src_hw:
            bnd = cv2.resize(bnd, (src_hw[1], src_hw[0]), interpolation=cv2.INTER_NEAREST)

        # boundary density for sanity (avoid all-white)
        density = float((bnd > 0).mean())
        report_lines.append(f"  [USE contour] canvas_hw={canvas_hw} final_hw={src_hw} boundary_density={density:.4f}")
        return bnd, "contour"

    report_lines.append("  [USE none] no inst_map and no contour found")
    return None, "none"


# -------------------------
# Build segmentor
# -------------------------
def build_segmentor(pretrained_model: str, batch_size: int, num_loader_workers: int, num_postproc_workers: int):
    model, ioconfig = get_pretrained_model(pretrained_model=pretrained_model)
    model = compile_model(model, mode="disable")
    seg = NucleusInstanceSegmentor(
        model=model,
        batch_size=batch_size,
        num_loader_workers=num_loader_workers,
        num_postproc_workers=num_postproc_workers,
    )
    return seg, ioconfig


# -------------------------
# Process one chunk
# -------------------------
def process_one_chunk(
    inst_segmentor: NucleusInstanceSegmentor,
    ioconfig,
    sub_pngs: List[str],
    tmp_save: Path,
    out_slide_dir: Path,
    device: str,
    dilate: int,
    overwrite: bool,
    patch_level_skip: bool,
    debug_preview: bool,
) -> Tuple[int, int, int, int, int, int, Path]:
    """
    Returns:
      artifacts_count, mapped_count, instmap_used, contour_used, written, failed, report_path
    """
    # DO NOT create tmp_save dir yourself
    _ = inst_segmentor.predict(
        sub_pngs,
        save_dir=str(tmp_save),
        mode="tile",
        device=device,
        ioconfig=ioconfig,
        crash_on_exception=True,
    )

    artifacts = find_prediction_artifacts(tmp_save)
    artifacts_count = len(artifacts)

    mapped_count = 0
    instmap_used = 0
    contour_used = 0
    written = 0
    failed = 0

    # write a detailed report per chunk
    report_path = tmp_save.parent / f"__debug_artifact_report_{tmp_save.name}.txt"
    report_lines: List[str] = []
    report_lines.append(f"[TMP] {tmp_save}")
    report_lines.append(f"artifacts_count={artifacts_count}")

    # save a preview once per chunk/slide
    did_preview = False

    for art in artifacts:
        patch_stem, src_path = artifact_to_patch_stem_and_src(art, sub_pngs)
        if patch_stem is None:
            # ignore non-index artifacts like file_map.dat
            report_lines.append(f"[SKIP] {art.name} -> cannot map to patch stem")
            continue

        mapped_count += 1

        out_path = out_slide_dir / f"{patch_stem}.png"
        if patch_level_skip and out_path.exists() and (not overwrite):
            continue

        report_lines.append(f"\n[ARTIFACT] {art}")
        bnd, mode_used = decode_artifact_to_boundary(
            artifact_path=art,
            src_patch_path=src_path,
            dilate=dilate,
            report_lines=report_lines,
        )

        if bnd is None:
            failed += 1
            continue

        if mode_used == "inst_map":
            instmap_used += 1
        elif mode_used == "contour":
            contour_used += 1

        ok = cv2.imwrite(str(out_path), bnd)
        if ok:
            written += 1
        else:
            failed += 1

        if debug_preview and (not did_preview):
            try:
                cv2.imwrite(str(out_slide_dir / "__debug_boundary_preview.png"), bnd)
            except Exception:
                pass
            did_preview = True

    report_path.write_text("\n".join(report_lines))
    return artifacts_count, mapped_count, instmap_used, contour_used, written, failed, report_path


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_img_dir", type=str, default="/mnt/liangjm/SpRR_data/")
    ap.add_argument("--pretrained_model", type=str, default="hovernet_fast-pannuke")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--predict_chunk", type=int, default=64)
    ap.add_argument("--dilate", type=int, default=3)
    ap.add_argument("--num_loader_workers", type=int, default=0)
    ap.add_argument("--num_postproc_workers", type=int, default=0)

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_skip_done", action="store_true")
    ap.add_argument("--no_patch_skip", action="store_true")

    ap.add_argument("--keep_tmp", action="store_true")
    ap.add_argument("--keep_tmp_on_failure", action="store_true")
    ap.add_argument("--debug_preview", action="store_true")

    args = ap.parse_args()

    out_img_dir = Path(args.out_img_dir)
    hr_root = out_img_dir / "hr_png"
    out_root = out_img_dir / "hovernet_boundary_pp"
    safe_mkdir(out_root)

    if not hr_root.exists():
        raise FileNotFoundError(f"hr_png not found: {hr_root}")

    slide_dirs = list_slide_dirs(hr_root)
    seg, ioconfig = build_segmentor(
        pretrained_model=args.pretrained_model,
        batch_size=int(args.batch_size),
        num_loader_workers=int(args.num_loader_workers),
        num_postproc_workers=int(args.num_postproc_workers),
    )

    skip_if_done = not bool(args.no_skip_done)
    patch_level_skip = not bool(args.no_patch_skip)

    print(f"[INFO] slides={len(slide_dirs)} model={args.pretrained_model} device={args.device} bs={args.batch_size}")
    print(f"[INFO] out_root={out_root} predict_chunk={args.predict_chunk} dilate={args.dilate}")
    print(f"[INFO] keep_tmp={args.keep_tmp} keep_tmp_on_failure={args.keep_tmp_on_failure} overwrite={args.overwrite}")

    for sdir in slide_dirs:
        slide_id = sdir.name if sdir != hr_root else "hr_png_flat"
        pngs = sorted(glob.glob(str(sdir / "*.png")))
        if not pngs:
            continue

        out_slide_dir = out_root / slide_id
        safe_mkdir(out_slide_dir)
        done_flag = out_slide_dir / ".HOVERNET_DONE"

        if skip_if_done and done_flag.exists() and (not args.overwrite):
            print(f"[SKIP] {slide_id} (DONE exists)")
            continue

        if args.overwrite:
            for f in out_slide_dir.glob("*.png"):
                try:
                    f.unlink()
                except Exception:
                    pass
            if done_flag.exists():
                try:
                    done_flag.unlink()
                except Exception:
                    pass

        print(f"[RUN] {slide_id} patches={len(pngs)}")

        totals = {
            "artifacts": 0,
            "mapped": 0,
            "instmap_used": 0,
            "contour_used": 0,
            "written": 0,
            "failed": 0,
        }
        kept_tmps: List[Path] = []

        for start in tqdm(range(0, len(pngs), max(1, int(args.predict_chunk))), desc=slide_id):
            sub_pngs = pngs[start:start + max(1, int(args.predict_chunk))]

            # fast skip if all outputs exist
            if patch_level_skip and (not args.overwrite):
                all_exist = True
                for p in sub_pngs:
                    if not (out_slide_dir / f"{Path(p).stem}.png").exists():
                        all_exist = False
                        break
                if all_exist:
                    continue

            tmp_save = make_unique_tmp_dir(out_root, slide_id)

            try:
                a, m, im, co, w, f, report = process_one_chunk(
                    inst_segmentor=seg,
                    ioconfig=ioconfig,
                    sub_pngs=sub_pngs,
                    tmp_save=tmp_save,
                    out_slide_dir=out_slide_dir,
                    device=args.device,
                    dilate=int(args.dilate),
                    overwrite=bool(args.overwrite),
                    patch_level_skip=patch_level_skip,
                    debug_preview=bool(args.debug_preview),
                )
                totals["artifacts"] += a
                totals["mapped"] += m
                totals["instmap_used"] += im
                totals["contour_used"] += co
                totals["written"] += w
                totals["failed"] += f

                print(
                    f"[CHUNK] {slide_id} start={start} artifacts={a} mapped={m} "
                    f"instmap_used={im} contour_used={co} written={w} failed={f} report={report}"
                )
            except Exception as e:
                totals["failed"] += 1
                print(f"[ERROR] {slide_id} start={start}: {repr(e)}")

            # keep tmp?
            keep_this = False
            if args.keep_tmp:
                keep_this = True
            if args.keep_tmp_on_failure and (totals["written"] == 0):
                keep_this = True

            if keep_this:
                kept_tmps.append(tmp_save)
                print(f"[KEEP] tmp kept: {tmp_save}")
            else:
                if tmp_save.exists():
                    shutil.rmtree(tmp_save, ignore_errors=True)

        existing_now = len(list(out_slide_dir.glob("*.png")))
        atomic_touch(
            done_flag,
            text=(
                f"done\nexisting_now={existing_now}\n"
                f"artifacts_total={totals['artifacts']}\n"
                f"mapped_total={totals['mapped']}\n"
                f"instmap_used_total={totals['instmap_used']}\n"
                f"contour_used_total={totals['contour_used']}\n"
                f"written_total={totals['written']}\n"
                f"failed_total={totals['failed']}\n"
                f"kept_tmps={len(kept_tmps)}\n"
            ),
        )

        print(
            f"[DONE] {slide_id} existing_now={existing_now} "
            f"instmap_used={totals['instmap_used']} contour_used={totals['contour_used']} "
            f"written={totals['written']} failed={totals['failed']}"
        )

    print("[ALL DONE]")


if __name__ == "__main__":
    main()
