import os
import sys
import json
import yaml
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict
from accelerate import Accelerator

# ======================
# CLAM / slicing imports
# ======================
CLAM_ROOT = "CLAM"
sys.path.append(CLAM_ROOT)
import src.preprocess_tcga_luad_with_clam as slic_tool

from src.loader import build_case_split_dataloaders
from src.utils import Logger, same_seeds
from src.sr_metrics import SRMetrics
from src.model import SRModel, SRModelConfig


# ----------------------
# helpers: config access
# ----------------------
def cfg_get(cfg: EasyDict, path: str, default=None):
    cur = cfg
    for k in path.split("."):
        if not hasattr(cur, k):
            return default
        cur = getattr(cur, k)
    return cur


def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ----------------------
# model / optim / sched
# ----------------------
def build_model_and_optim(config: EasyDict, device: torch.device):
    """
    Build SRModel + optimizer using the NEW model.py:
      - NO DINO
      - HoVer-Net guidance supported
    """
    mcfg = SRModelConfig(
        # scale
        scale=int(cfg_get(config, "data_loader.down_scale", 4)),

        # encoder & kernel predictor
        feat_ch=int(cfg_get(config, "model.feat_ch", 64)),
        pe_bands=int(cfg_get(config, "model.pe_bands", 10)),
        mlp_hidden=int(cfg_get(config, "model.mlp_hidden", 256)),
        mlp_depth=int(cfg_get(config, "model.mlp_depth", 5)),

        # kernel
        kernel_size=int(cfg_get(config, "model.kernel_size", 7)),
        kernel_allow_negative=bool(cfg_get(config, "model.kernel_allow_negative", True)),

        # sampling & weighting
        num_points=int(cfg_get(config, "model.num_points", 4096)),
        saliency_ratio=float(cfg_get(config, "model.saliency_ratio", 0.7)),
        loss_alpha=float(cfg_get(config, "model.loss_alpha", 3.0)),

        # residual SR
        use_residual=bool(cfg_get(config, "model.use_residual", True)),
        use_res_refiner=bool(cfg_get(config, "model.use_res_refiner", False)),

        # gradient consistency
        lambda_grad=float(cfg_get(config, "model.lambda_grad", 0.2)),
        grad_crop=int(cfg_get(config, "model.grad_crop", 128)),

        # inference chunk
        infer_chunk=int(cfg_get(config, "model.infer_chunk", 8192)),

        # HoVer-Net guidance (NEW)
        use_hovernet_guidance=bool(cfg_get(config, "model.use_hovernet_guidance", False)),
        hovernet_pretrained_name=str(cfg_get(config, "model.hovernet_pretrained_name", "cin2_v1_efficientnet_b5")),
        hovernet_use_gray=bool(cfg_get(config, "model.hovernet_use_gray", True)),
        hovernet_mix_beta=float(cfg_get(config, "model.hovernet_mix_beta", 0.5)),
        hovernet_weight_alpha=float(cfg_get(config, "model.hovernet_weight_alpha", 1.0)),

        # temperature annealing
        tau_start=float(cfg_get(config, "model.tau_start", 1.0)),
        tau_end=float(cfg_get(config, "model.tau_end", 0.5)),
        tau_warm_epochs=int(cfg_get(config, "model.tau_warm_epochs", 2)),
        tau_anneal_epochs=int(cfg_get(config, "model.tau_anneal_epochs", 8)),
    )

    model = SRModel(cfg=mcfg).to(device)

    lr = float(cfg_get(config, "trainer.lr", 1e-4))
    wd = float(cfg_get(config, "trainer.weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = None
    return model, optimizer, scheduler

# ----------------------
# checkpoint
# ----------------------
def save_checkpoint_latest(save_dir: Path, accelerator: Accelerator, model, optimizer, scheduler, epoch: int, best_metric):
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "checkpoint_latest.pt"

    unwrap = accelerator.unwrap_model(model)
    obj = {
        "epoch": int(epoch),
        "best_metric": float(best_metric) if best_metric is not None else None,
        "model": unwrap.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    accelerator.save(obj, str(ckpt_path))


def try_resume_from_latest(save_dir: Path, accelerator: Accelerator, model, optimizer, scheduler):
    ckpt_path = save_dir / "checkpoint_latest.pt"
    if not ckpt_path.exists():
        return 0, None

    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    unwrap = accelerator.unwrap_model(model)
    unwrap.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_metric = ckpt.get("best_metric", None)
    return start_epoch, best_metric


def save_best_model(save_dir: Path, accelerator: Accelerator, model, epoch: int, best_metric: float):
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best_model.pt"
    meta_path = save_dir / "best_meta.json"

    unwrap = accelerator.unwrap_model(model)
    accelerator.save({"model": unwrap.state_dict()}, str(best_path))

    if accelerator.is_local_main_process:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"epoch": int(epoch), "best_metric": float(best_metric)}, f, indent=2)


# ----------------------
# visuals
# ----------------------
@torch.no_grad()
def save_triplet_visual(save_path: Path, lr, hr, sr):
    """Save HR | LR_up | SR triplet as a horizontal concatenated PNG."""
    lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)

    def to_uint8(img):
        img = img[0].detach().cpu().permute(1, 2, 0).numpy()
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        return img

    a = to_uint8(hr)
    b = to_uint8(lr_up)
    c = to_uint8(sr)
    cat = np.concatenate([a, b, c], axis=1)

    from PIL import Image
    Image.fromarray(cat).save(str(save_path))


@torch.no_grad()
def save_random_train_visual_to_val_vis(
    accelerator: Accelerator,
    model,
    train_loader,
    epoch: int,
    val_vis_dir: Path,
):
    """
    Save ONE random training sample visualization into the SAME val_vis folder,
    distinguished by filename: train_epoch_XXXX.png
    """
    if not accelerator.is_local_main_process:
        return
    if len(train_loader) <= 0:
        return

    vis_idx = random.randint(0, max(len(train_loader) - 1, 0))

    picked = None
    for i, batch in enumerate(train_loader):
        if i == vis_idx:
            picked = batch
            break
    if picked is None:
        return

    was_training = model.training
    model.eval()

    lr = picked["lr"].to(accelerator.device, non_blocking=True)
    hr = picked["hr"].to(accelerator.device, non_blocking=True)
    sr = model(lr).clamp(0, 1)

    save_triplet_visual(val_vis_dir / f"train_epoch_{epoch:04d}.png", lr, hr, sr)

    if was_training:
        model.train()


# ----------------------
# train / val
# ----------------------
def train_one_epoch(accelerator: Accelerator, model, optimizer, train_loader, epoch: int, config: EasyDict):
    model.train()
    running = 0.0
    step = 0

    log_every = int(cfg_get(config, "trainer.log_every", 50))
    grad_clip = cfg_get(config, "trainer.grad_clip", None)

    for batch in train_loader:
        lr = batch["lr"].to(accelerator.device, non_blocking=True)
        hr = batch["hr"].to(accelerator.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        loss, dbg = model.compute_loss(lr, hr, return_debug=True)
        accelerator.backward(loss)

        if grad_clip is not None:
            accelerator.clip_grad_norm_(model.parameters(), float(grad_clip))

        optimizer.step()

        running += float(loss.detach().item())
        step += 1

        if accelerator.is_local_main_process and (step % log_every == 0):
            accelerator.print(
                f"[Epoch {epoch}][{step}] "
                f"loss={running/step:.4f} "
                f"pix={dbg.get('loss_pix', 0):.4f} grad={dbg.get('loss_grad', 0):.4f} "
                f"mix={dbg.get('mean_mix', 0):.3f} dino={dbg.get('mean_dino', 0):.3f} edge={dbg.get('mean_edge', 0):.3f} "
                f"tau={dbg.get('tau', 1.0):.3f}"
            )

    return running / max(step, 1)


@torch.no_grad()
def validate_one_epoch(accelerator: Accelerator, model, val_loader, epoch: int, val_vis_dir: Path):
    """
    Validation: compute PSNR/SSIM/LPIPS and save ONE random validation visualization
    into val_vis folder: val_epoch_XXXX.png
    """
    model.eval()
    metrics = SRMetrics(device=str(accelerator.device))

    psnr_list, ssim_list, lpips_list = [], [], []

    # choose a batch index for visualization (sync across processes)
    if accelerator.is_local_main_process:
        vis_idx = random.randint(0, max(len(val_loader) - 1, 0))
        t = torch.tensor([vis_idx], device=accelerator.device, dtype=torch.long)
    else:
        t = torch.tensor([0], device=accelerator.device, dtype=torch.long)

    if accelerator.num_processes > 1:
        accelerator.broadcast(t, from_process=0)
    vis_idx = int(t.item())

    for i, batch in enumerate(val_loader):
        lr = batch["lr"].to(accelerator.device, non_blocking=True)
        hr = batch["hr"].to(accelerator.device, non_blocking=True)

        sr = model(lr).clamp(0, 1)

        psnr_list.append(metrics.psnr(sr, hr))
        ssim_list.append(metrics.ssim(sr, hr))
        lpips_list.append(metrics.lpips(sr, hr))

        if accelerator.is_local_main_process and i == vis_idx:
            save_triplet_visual(val_vis_dir / f"val_epoch_{epoch:04d}.png", lr, hr, sr)

    val_metrics = {
        "psnr": float(np.mean(psnr_list)) if psnr_list else 0.0,
        "ssim": float(np.mean(ssim_list)) if ssim_list else 0.0,
        "lpips": float(np.mean(lpips_list)) if lpips_list else 0.0,
    }
    return val_metrics


def is_better(val_metrics: dict, best_metric, config: EasyDict):
    key = str(cfg_get(config, "validator.metric_for_best", "psnr"))
    mode = str(cfg_get(config, "validator.best_mode", "max")).lower()
    val = float(val_metrics.get(key, 0.0))

    if best_metric is None:
        return True
    if mode == "max":
        return val > float(best_metric)
    return val < float(best_metric)


# ----------------------
# main
# ----------------------
if __name__ == "__main__":
    config = EasyDict(yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader))

    # seed
    same_seeds(int(cfg_get(config, "data_loader.seed", 2025)))

    # logging dir
    ckpt_name = str(cfg_get(config, "checkpoint", "exp"))
    logging_dir = Path(os.getcwd()) / "logs" / f"{ckpt_name}_{now_str()}"
    logging_dir.mkdir(parents=True, exist_ok=True)

    # accelerator
    mp = "fp16" if bool(cfg_get(config, "trainer.use_amp", False)) else "no"
    accelerator = Accelerator(
        cpu=False,
        log_with=["tensorboard"],
        project_dir=str(logging_dir),
        mixed_precision=mp,
    )

    Logger(str(logging_dir) if accelerator.is_local_main_process else None)

    # ---------------------------------
    # slicing (optional)
    # ---------------------------------
    image_dict = {
        "TCGA-LUAD": [config.TCGA_LUAD.root, config.TCGA_LUAD.choose_WSI],
        "TCGA-KIRC": [config.TCGA_KIRC.root, config.TCGA_KIRC.choose_WSI],
        "TCGA-LIHC": [config.TCGA_LIHC.root, config.TCGA_LIHC.choose_WSI],
    }

    skip_slicing = bool(cfg_get(config, "data_loader.skip_slicing", False))
    if not skip_slicing:
        slic_tool.run_clam_and_export(
            data_cfg=image_dict,
            out_img_dir=config.data_loader.out_img_dir,
            patch_size=config.data_loader.patch_size,
            step_size=config.data_loader.step_size,
            patch_level=config.data_loader.patch_level,
            down_scale=config.data_loader.down_scale,
            min_tissue_ratio=config.data_loader.min_tissue_ratio,
            seed=config.data_loader.seed,
            limit_samples=config.data_loader.patch_num,
        )
        accelerator.print("[CHECK] slicing done.")
    else:
        accelerator.print("[CHECK] skip slicing (data_loader.skip_slicing=true).")

    # ---------------------------------
    # dataloaders
    # ---------------------------------
    train_loader, val_loader, test_loader = build_case_split_dataloaders(
        out_img_dir=config.data_loader.out_img_dir,
        batch_size=int(cfg_get(config, "trainer.batch_size", 1)),
        patch_num=int(cfg_get(config, "data_loader.patch_num", 200)),
        train_ratio=float(cfg_get(config, "data_loader.train_ratio", 0.8)),
        val_ratio=float(cfg_get(config, "data_loader.val_ratio", 0.1)),
        test_ratio=float(cfg_get(config, "data_loader.test_ratio", 0.1)),
        split_seed=int(cfg_get(config, "data_loader.seed", 2025)),
        num_workers=int(cfg_get(config, "data_loader.num_workers", 4)),
        pin_memory=bool(cfg_get(config, "data_loader.pin_memory", True)),
    )

    # ---------------------------------
    # model / optim
    # ---------------------------------
    model, optimizer, scheduler = build_model_and_optim(config, accelerator.device)

    # prepare
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    save_dir = logging_dir
    start_epoch, best_metric = try_resume_from_latest(save_dir, accelerator, model, optimizer, scheduler)
    num_epochs = int(cfg_get(config, "trainer.epochs", 10))

    # val_vis dir (save BOTH val and train visuals here)
    vis_subdir = str(cfg_get(config, "validator.vis_subdir", "val_vis"))
    val_vis_dir = save_dir / vis_subdir
    val_vis_dir.mkdir(parents=True, exist_ok=True)

    accelerator.print(f"[CHECK] start_epoch={start_epoch} best_metric={best_metric}")

    for epoch in range(start_epoch, num_epochs):
        # enable tau annealing
        unwrap = accelerator.unwrap_model(model)
        if hasattr(unwrap, "set_epoch"):
            unwrap.set_epoch(epoch)

        train_loss = train_one_epoch(accelerator, model, optimizer, train_loader, epoch, config)

        # validate and save val_epoch_XXXX.png
        val_metrics = validate_one_epoch(accelerator, model, val_loader, epoch, val_vis_dir)

        # additionally save train_epoch_XXXX.png in the SAME folder
        save_random_train_visual_to_val_vis(accelerator, model, train_loader, epoch, val_vis_dir)

        metric_key = str(cfg_get(config, "validator.metric_for_best", "psnr"))
        if is_better(val_metrics, best_metric, config):
            best_metric = float(val_metrics.get(metric_key, 0.0))
            save_best_model(save_dir, accelerator, model, epoch, best_metric)

        save_checkpoint_latest(save_dir, accelerator, model, optimizer, scheduler, epoch, best_metric)

        if accelerator.is_local_main_process:
            accelerator.print(
                f"[Epoch {epoch}] train_loss={train_loss:.4f} "
                f"val={val_metrics} best({metric_key})={best_metric}"
            )
