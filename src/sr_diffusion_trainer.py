# src/sr_diffusion_trainer.py
import os
import re
import json
import shutil
from dataclasses import dataclass
from typing import Optional, Dict
from PIL import Image, ImageFilter
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

# 可选：huggingface 镜像加速（如果 cfg.pretrained_sd15 指向本地目录，不影响）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from src.sr_metrics import SRMetrics

# ✅ M1: VFM control (DINOv2)
from src.vfm_control import VFMConfig, VFMControlGenerator


@dataclass
class SRTrainConfig:
    # -------------------------
    # basic / io
    # -------------------------
    output_dir: str = "./outputs"
    checkpoint: str = "TryFirst"         # 用于日志/实验名
    device: str = "cuda"
    seed: int = 42

    # -------------------------
    # pretrained weights
    # -------------------------
    pretrained_sd15: str = "runwayml/stable-diffusion-v1-5"
    token: Optional[str] = None          # HuggingFace token（可选）
    local_dir: str = "./src/models/dinov2_vitb14"                  # DINOv2 本地权重目录（必填：你现在是离线加载）

    # -------------------------
    # training
    # -------------------------
    lr: float = 1e-4
    train_steps: int = 20000
    grad_accum: int = 1
    mixed_precision: str = "fp16"        # "fp16" | "bf16" | "no"
    save_every: int = 1000               # 覆盖式 latest 保存频率
    val_every: int = 2000
    val_batches: int = 10
    sample_steps: int = 20               # 验证/采样的扩散步数（越大越慢）
    val_vis_keep: int = 5                # 验证可视化结果保留数量（按 step，0=不保留）

    # -------------------------
    # checkpoint behavior
    # -------------------------
    save_full_ckpt: bool = False         # True: 额外保存 optimizer/scaler 等 trainer_state.pt
    resume_ckpt: Optional[str] = None    # 手动指定恢复路径（可选）

    # -------------------------
    # VFM / DINOv2 co-training (NEW)
    # -------------------------
    vfm_unfreeze_last_blocks: int = 2    # ✅只解冻最后 N 个 transformer blocks（推荐 1~2 起步）
    # 如果你之后要做 warmup 再解冻，可以加这个（目前我给你的实现没有强制用到，但预留很有用）
    vfm_warmup_steps: int = 0            # 0 表示一开始就按 unfreeze_last_blocks 训练

    # 如果你希望 VFM 用独立学习率（目前我给你的实现是共用 lr；你可后续扩展 optimizer 分组）
    vfm_lr: Optional[float] = None       # None 表示跟 lr 一样；或设置为 lr*0.1 之类

    # -------------------------
    # discrepancy-guided / survival-aware (NEW - 预留)
    # 说明：你目前决定“先用 proxy 占位”，后续接入 VLM 差异图，这些参数就能派上用场
    # -------------------------
    use_proxy_discrepancy: bool = True   # True: 用 proxy 生成差异图（梯度差/SSIM 等）
    disc_alpha: float = 2.0              # 差异图权重放大系数（1 + disc_alpha * D）

    # survival-aware weighting（如果你后续把 time/event 融进去）
    surv_beta: float = 1.0               # event=1 & time短 的样本权重系数
    ssa_disc_scale: float = 1.0
    ssa_w_min: float = 1.0
    ssa_w_max: float = 2.0
    ssa_w_gamma: float = 0.5          # <1 更温和，避免过度强调结构区导致色彩/纹理崩
    ssa_disc_smooth_ksize: float = 7  # 建议 3~7；0 关闭
    ssa_warmup_steps: float = 1000
    ssa_ramp_steps: float = 3000
    ssa_alpha_up: float = 0.5
    


    # -------------------------
    # bookkeeping (optional)
    # -------------------------
    init_best_psnr: float = -1e9         # 用于 resume 后 best tracking
    init_best_step: int = 0

    # -------------------------
    # ✅ Color correction (inference-time, used in validate metrics)
    # -------------------------
    enable_color_correction: bool = True
    color_correction_method: str = "gain"   # currently: reinhard
    color_correction_ref: str = "lr"            # ✅ "lr" (recommended), or "template"
    color_template_path: str = ""               # used when ref="template"
    color_lowpass_ksize: int = 51              # ✅ blur LR_up reference to stabilize stain (odd >=1)

    # 强约束：防止爆色
    color_gain_min: float = 0.9
    color_gain_max: float = 1.1

    # 组织mask
    color_use_tissue_mask: bool = True
    tissue_s_thr: float = 0.10
    tissue_v_thr: float = 0.92
    bg_v_thr: float = 0.97
    bg_s_thr: float = 0.06
    
    
def make_control_image_from_lr_fallback(lr_up: torch.Tensor) -> torch.Tensor:
    """
    兜底控制图（不依赖 SAM，不依赖 VFM）。
    lr_up: [B,3,512,512] in [0,1]
    return: [B,3,512,512] => (gray, edge, gray)
    """
    gray = lr_up.mean(dim=1, keepdim=True)  # [B,1,H,W]
    dx = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
    dy = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    edge = torch.clamp(dx + dy, 0.0, 1.0)
    ctrl = torch.cat([gray, edge, gray], dim=1)
    return ctrl


class DiffusionSRControlNetTrainer:
    def __init__(self, cfg: SRTrainConfig, token: Optional[str] = None):
        self.cfg = cfg
        os.makedirs(cfg.output_dir, exist_ok=True)

        self.device = torch.device(cfg.device)
        self.metrics = SRMetrics(device=self.device)

        # ----------------------------
        # ✅ M1: init DINOv2 control generator (trainable-last-blocks)
        # ----------------------------
        self.vfm_cfg = VFMConfig(
            variant="dinov2_vitb14",
            local_dir=cfg.local_dir,
            image_size=518,
            patch_size=14,
            control_mode="energy_edge_gray",
            normalize=True,
        )
        self.ctrl_gen = VFMControlGenerator(self.vfm_cfg, device=self.device)
        self._warned_control_fallback = False

        # 训练策略：只解冻最后若干层（默认 2）
        self.vfm_unfreeze_last_blocks = int(getattr(cfg, "vfm_unfreeze_last_blocks", 2) or 0)
        self.ctrl_gen.set_trainable(unfreeze_last_n_blocks=self.vfm_unfreeze_last_blocks, train_ln=True)

        print(
            f"[VFM] enabled: {self.vfm_cfg.variant} | local_dir={self.vfm_cfg.local_dir} | "
            f"unfreeze_last_blocks={self.vfm_unfreeze_last_blocks}"
        )

        # best tracking (for best-PSNR ckpt)
        self.best_psnr = float(getattr(cfg, "init_best_psnr", -1e9))
        self.best_step = int(getattr(cfg, "init_best_step", 0))

        # ----------------------------
        # SD1.5 components
        # ----------------------------
        self.tokenizer = CLIPTokenizer.from_pretrained(
            cfg.pretrained_sd15, subfolder="tokenizer", token=token
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            cfg.pretrained_sd15, subfolder="text_encoder", token=token
        ).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            cfg.pretrained_sd15, subfolder="vae", token=token
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            cfg.pretrained_sd15, subfolder="unet", token=token
        ).to(self.device)

        # ControlNet init from UNet
        self.controlnet = ControlNetModel.from_unet(self.unet).to(self.device)

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            cfg.pretrained_sd15, subfolder="scheduler", token=token
        )

        # freeze base
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()

        # train controlnet + (optional) trainable parts of ctrl_gen
        # train controlnet + EnergyHead only
        self.controlnet.train()

        trainable_head_params = list(self.ctrl_gen.energy_head.parameters())
        self.optimizer = AdamW(
            list(self.controlnet.parameters()) + trainable_head_params,
            lr=float(cfg.lr),
        )

        # AMP
        self.use_amp = (cfg.mixed_precision in ("fp16", "bf16"))
        self.autocast_dtype = torch.float16 if cfg.mixed_precision == "fp16" else torch.bfloat16
        self.scaler = torch.amp.GradScaler("cuda", enabled=(cfg.mixed_precision == "fp16"))

        # cache empty prompt
        self.empty_prompt_embeds = self._encode_text([""]) 
        
        
        
    def _proxy_discrepancy_map(self, a_01: torch.Tensor, b_01: torch.Tensor) -> torch.Tensor:
        """
        Proxy discrepancy map D in [0,1], higher = more inconsistent.
        Here we use gradient-magnitude difference as a cheap, stable placeholder.

        a_01, b_01: [B,3,H,W] in [0,1]
        return: D [B,1,H,W] in [0,1]
        """
        # grayscale
        a = a_01.mean(dim=1, keepdim=True)
        b = b_01.mean(dim=1, keepdim=True)

        # Sobel kernels
        kx = torch.tensor([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=a.dtype, device=a.device).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=a.dtype, device=a.device).view(1, 1, 3, 3)

        # grad magnitude
        ax = F.conv2d(a, kx, padding=1)
        ay = F.conv2d(a, ky, padding=1)
        bx = F.conv2d(b, kx, padding=1)
        by = F.conv2d(b, ky, padding=1)

        ga = torch.sqrt(ax * ax + ay * ay + 1e-12)
        gb = torch.sqrt(bx * bx + by * by + 1e-12)

        d = torch.abs(ga - gb)  # [B,1,H,W]

        # normalize per-sample to [0,1]
        B = d.shape[0]
        d_flat = d.view(B, -1)
        d_min = d_flat.min(dim=1)[0].view(B, 1, 1, 1)
        d_max = d_flat.max(dim=1)[0].view(B, 1, 1, 1)
        d = (d - d_min) / (d_max - d_min + 1e-6)
        return d.clamp(0, 1)    
        
    def _survival_weight(self, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        """
        Survival-aware per-sample weight.
        - event=1 gets higher weight
        - shorter time gets higher weight (rank-normalized)
        return: w [B] (>=1)
        """
        # ensure float tensors on device
        t = time.float().to(self.device)
        e = event.float().to(self.device)

        # avoid div by zero; also robust if time already normalized
        inv_t = 1.0 / (t + 1e-6)

        # rank-normalize inv_t to [0,1]
        # (cheap approximation: min-max per batch)
        inv_min = inv_t.min()
        inv_max = inv_t.max()
        inv_norm = (inv_t - inv_min) / (inv_max - inv_min + 1e-6)

        w = 1.0 + float(getattr(self, "surv_beta", 1.0)) * e * inv_norm
        return w.detach()    
    
    def _cox_ph_loss(self, risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        """
        Standard Cox partial likelihood (negative log-likelihood).
        risk: [B] (higher = higher hazard)
        time: [B]
        event: [B] (1=observed, 0=censored)
        """
        r = risk.view(-1).float()
        t = time.view(-1).float()
        e = event.view(-1).float()

        # sort by time descending so that risk set is prefix
        order = torch.argsort(t, descending=True)
        r = r[order]
        e = e[order]

        # log cumulative sum exp for risk set
        log_cumsum = torch.logcumsumexp(r, dim=0)

        # only for observed events
        neg_log_lik = -(r - log_cumsum) * e
        denom = e.sum().clamp(min=1.0)
        return neg_log_lik.sum() / denom
        
    # ----------------------------
    # helpers
    # ----------------------------
    @torch.no_grad()
    def _encode_text(self, prompts):
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)
        return self.text_encoder(**tokens).last_hidden_state  # [B,77,768]

    @torch.no_grad()
    def _vae_encode(self, x_01: torch.Tensor) -> torch.Tensor:
        x = x_01 * 2.0 - 1.0
        latents = self.vae.encode(x).latent_dist.mode()
        return latents * 0.18215

    @torch.no_grad()
    def _vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / 0.18215
        img = self.vae.decode(latents).sample
        img = (img + 1.0) / 2.0
        return img.clamp(0, 1)

    def _build_control(self, lr_up_512: torch.Tensor) -> torch.Tensor:
        """
        lr_up_512: [B,3,512,512] in [0,1]
        return: [B,3,512,512] in [0,1]
        """
        try:
            return self.ctrl_gen(lr_up_512)
        except Exception as e:
            if not self._warned_control_fallback:
                print(f"[WARN] VFM control failed once, fallback to lr-edge control. err={e}")
                self._warned_control_fallback = True
            return make_control_image_from_lr_fallback(lr_up_512)

    def _build_control_train(self, lr_up_512: torch.Tensor) -> torch.Tensor:
        """
        训练用 control：允许梯度（用于同步优化 DINOv2 最后若干层）
        """
        try:
            return self.ctrl_gen.forward_train(lr_up_512)
        except Exception as e:
            if not self._warned_control_fallback:
                print(f"[WARN] VFM control failed once, fallback to lr-edge control. err={e}")
                self._warned_control_fallback = True
            return make_control_image_from_lr_fallback(lr_up_512)

    @torch.no_grad()
    def _build_control_eval(self, lr_up_512: torch.Tensor) -> torch.Tensor:
        """
        验证/采样用 control：no_grad，保证速度与确定性
        """
        try:
            return self.ctrl_gen(lr_up_512)
        except Exception as e:
            if not self._warned_control_fallback:
                print(f"[WARN] VFM control failed once, fallback to lr-edge control. err={e}")
                self._warned_control_fallback = True
            return make_control_image_from_lr_fallback(lr_up_512)
    
    
    def _reinhard_color_match_u8(self, src_rgb_u8: np.ndarray, ref_rgb_u8: np.ndarray) -> np.ndarray:
        """
        Reinhard color transfer in LAB space (8-bit).
        src_rgb_u8/ref_rgb_u8: [H,W,3] uint8 RGB
        return: corrected RGB uint8 [H,W,3]
        """
        src_lab = Image.fromarray(src_rgb_u8, mode="RGB").convert("LAB")
        ref_lab = Image.fromarray(ref_rgb_u8, mode="RGB").convert("LAB")

        src = np.array(src_lab).astype(np.float32)  # [H,W,3]
        ref = np.array(ref_lab).astype(np.float32)

        src_flat = src.reshape(-1, 3)
        ref_flat = ref.reshape(-1, 3)

        src_mu = src_flat.mean(axis=0)
        src_std = src_flat.std(axis=0)
        ref_mu = ref_flat.mean(axis=0)
        ref_std = ref_flat.std(axis=0)

        # avoid divide-by-zero / over-amplification
        src_std = np.maximum(src_std, 1.0)
        ref_std = np.maximum(ref_std, 1.0)

        out = (src - src_mu) / src_std
        out = out * ref_std + ref_mu
        out = np.clip(out, 0, 255).astype(np.uint8)

        out_rgb = Image.fromarray(out, mode="LAB").convert("RGB")
        return np.array(out_rgb, dtype=np.uint8)

    
    def _gaussian_blur_u8(self, rgb_u8: np.ndarray, ksize: int = 51) -> np.ndarray:
        """
        Low-pass filtering (Gaussian blur) for uint8 RGB image.
        ksize: odd int, larger => more global/low-frequency only.
        """
        k = int(ksize)
        if k < 1:
            return rgb_u8
        if k % 2 == 0:
            k += 1

        # radius mapping: ksize≈6*radius (heuristic)
        radius = max(1, k // 6)
        return np.array(
            Image.fromarray(rgb_u8, mode="RGB").filter(ImageFilter.GaussianBlur(radius=radius)),
            dtype=np.uint8,
        )
    
    def _apply_color_correction_batch(self, sr_01: torch.Tensor, lr_up_01: torch.Tensor) -> torch.Tensor:
        """
        Safe color correction:
        - method="gain": bounded per-channel gain matching (LR_up as reference)
        ✅ tissue mask computed from LR_up (more stable than SR)
        ✅ optional low-pass on LR_up to stabilize stain bias (color_lowpass_ksize)
        """
        if not bool(getattr(self.cfg, "enable_color_correction", True)):
            return sr_01

        method = str(getattr(self.cfg, "color_correction_method", "gain")).lower()
        if method not in ["gain", "wb", "white_balance"]:
            # keep other methods as no-op for now
            return sr_01

        gain_min = float(getattr(self.cfg, "color_gain_min", 0.9))
        gain_max = float(getattr(self.cfg, "color_gain_max", 1.1))
        eps = 1e-6

        use_mask = bool(getattr(self.cfg, "color_use_tissue_mask", True))
        s_thr = float(getattr(self.cfg, "tissue_s_thr", 0.10))
        v_thr = float(getattr(self.cfg, "tissue_v_thr", 0.92))
        bg_v_thr = float(getattr(self.cfg, "bg_v_thr", 0.97))
        bg_s_thr = float(getattr(self.cfg, "bg_s_thr", 0.06))

        # -------------------------
        # 1) build tissue mask (✅ from LR_up, not SR)
        # -------------------------
        if use_mask:
            x = lr_up_01.clamp(0, 1)  # ✅ key change: use LR_up
            r, g, b = x[:, 0], x[:, 1], x[:, 2]
            mx = torch.maximum(torch.maximum(r, g), b)
            mn = torch.minimum(torch.minimum(r, g), b)
            diff = mx - mn
            v = mx
            s = torch.zeros_like(v)
            s = torch.where(mx > eps, diff / (mx + eps), s)

            tissue = (s > s_thr) | (v < v_thr)
            bg = (v > bg_v_thr) & (s < bg_s_thr)
            tissue = tissue & (~bg)
            mask = tissue.float().unsqueeze(1)  # [B,1,H,W]
        else:
            mask = torch.ones((sr_01.shape[0], 1, sr_01.shape[2], sr_01.shape[3]),
                            device=sr_01.device, dtype=sr_01.dtype)

        # -------------------------
        # 2) low-pass on LR_up reference (✅ enable color_lowpass_ksize)
        # -------------------------
        k = int(getattr(self.cfg, "color_lowpass_ksize", 0) or 0)
        ref = lr_up_01
        if k and k > 1:
            if k % 2 == 0:
                k += 1
            # avg_pool2d as a fast low-pass approximation
            pad = k // 2
            ref = torch.nn.functional.avg_pool2d(
                torch.nn.functional.pad(ref, (pad, pad, pad, pad), mode="reflect"),
                kernel_size=k,
                stride=1,
            )

        # -------------------------
        # 3) masked per-channel mean + bounded gain
        # -------------------------
        def masked_mean(img, m):
            w = m.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
            return (img * m).sum(dim=(2, 3), keepdim=True) / w  # [B,3,1,1]

        src = sr_01
        mu_src = masked_mean(src, mask)
        mu_ref = masked_mean(ref, mask)

        gain = (mu_ref / (mu_src + eps)).clamp(gain_min, gain_max)  # [B,3,1,1]
        out = src * (1.0 - mask) + (src * gain) * mask
        return out.clamp(0, 1)

    # ----------------------------
    # checkpoint utils
    # ----------------------------
    def _ckpt_root_dir(self) -> str:
        return os.path.join(self.cfg.output_dir, getattr(self.cfg, "ckpt_root", "checkpoints"))

    def _ckpt_dir(self, step: int) -> str:
        return os.path.join(self._ckpt_root_dir(), f"step_{step:08d}")

    def _find_latest_checkpoint(self) -> Optional[str]:
        ckpt_root = self._ckpt_root_dir()
        if not os.path.isdir(ckpt_root):
            return None

        # ✅ New: prefer "latest" dir (single latest checkpoint)
        latest_dir = os.path.join(ckpt_root, "latest")
        if os.path.isdir(os.path.join(latest_dir, "controlnet")):
            # if full ckpt required, check trainer_state.pt
            if getattr(self.cfg, "save_full_ckpt", False):
                if os.path.isfile(os.path.join(latest_dir, "trainer_state.pt")):
                    return latest_dir
            else:
                return latest_dir

        # Fallback: legacy step_XXXXXXXX dirs (keep for backward compatibility)
        step_re = re.compile(r"^step_(\d+)$")
        best_step = -1
        best_dir = None

        for name in os.listdir(ckpt_root):
            m = step_re.match(name)
            if not m:
                continue
            step = int(m.group(1))
            d = os.path.join(ckpt_root, name)

            if not os.path.isdir(os.path.join(d, "controlnet")):
                continue

            if getattr(self.cfg, "save_full_ckpt", False):
                if not os.path.isfile(os.path.join(d, "trainer_state.pt")):
                    continue

            if step > best_step:
                best_step = step
                best_dir = d

        return best_dir


    def _prune_val_vis(self):
        keep = int(getattr(self.cfg, "val_vis_keep", 0) or 0)
        if keep <= 0:
            return

        vis_base = os.path.join(self.cfg.output_dir, "val_vis")
        if not os.path.isdir(vis_base):
            return

        step_re = re.compile(r"^step_(\d+)$")
        items = []
        for name in os.listdir(vis_base):
            m = step_re.match(name)
            if not m:
                continue
            step = int(m.group(1))
            d = os.path.join(vis_base, name)
            if os.path.isdir(d):
                items.append((step, d))

        # 少于等于 keep，无需清理
        if len(items) <= keep:
            return

        # 按 step 从小到大排序，删最旧的
        items.sort(key=lambda x: x[0])
        to_delete = items[: max(0, len(items) - keep)]

        for step, d in to_delete:
            try:
                shutil.rmtree(d)
                print(f"[val_vis] pruned old vis: step={step} dir={d}")
            except Exception as e:
                print(f"[WARN] failed to prune val_vis dir={d}: {e}")
    
    def save_checkpoint(self, step: int, is_best: bool = False):
        """
        Save checkpoint in an overwrite manner:
        - latest  -> <output_dir>/checkpoints/latest
        - best    -> <output_dir>/checkpoints/best

        Contents:
        - controlnet/ (diffusers save_pretrained)
        - trainer_state.pt
        - vfm_control.pt   (EnergyHead etc. in VFMControlGenerator)  [Option A]
        - meta.json
        """
        import os
        import json
        import shutil
        import time

        ckpt_root = self._ckpt_root_dir()
        os.makedirs(ckpt_root, exist_ok=True)

        tag = "best" if is_best else "latest"
        dst_dir = os.path.join(ckpt_root, tag)
        tmp_dir = os.path.join(ckpt_root, f"._tmp_{tag}_{int(time.time())}_{int(step)}")

        # start clean tmp
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        # -------------------------
        # 1) save controlnet weights (diffusers format)
        # -------------------------
        ctrl_dir = os.path.join(tmp_dir, "controlnet")
        os.makedirs(ctrl_dir, exist_ok=True)
        try:
            self.controlnet.save_pretrained(ctrl_dir)
        except Exception as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(f"controlnet.save_pretrained failed: {e}")

        # -------------------------
        # 2) save trainer state  (IMPORTANT: keep both keys)
        # -------------------------
        trainer_state = {
            "step": int(step),
            "global_step": int(step),  # forward/backward compatible
            "best_psnr": float(getattr(self, "best_psnr", 0.0)),
            "best_step": int(getattr(self, "best_step", 0)),
            "optimizer": self.optimizer.state_dict() if getattr(self, "optimizer", None) is not None else None,
            "scaler": self.scaler.state_dict() if getattr(self, "scaler", None) is not None else None,
        }
        torch.save(trainer_state, os.path.join(tmp_dir, "trainer_state.pt"))

        # -------------------------
        # 3) save VFM control (EnergyHead only)  [Option A]
        # -------------------------
        if hasattr(self, "ctrl_gen") and self.ctrl_gen is not None:
            vfm_path = os.path.join(tmp_dir, "vfm_control.pt")
            # requires VFMControlGenerator.save_ckpt()
            self.ctrl_gen.save_ckpt(vfm_path)

        # -------------------------
        # 4) meta.json
        # -------------------------
        meta = {
            "step": int(step),
            "global_step": int(step),
            "is_best": bool(is_best),
            "best_psnr": float(getattr(self, "best_psnr", 0.0)),
            "best_step": int(getattr(self, "best_step", 0)),
            "output_dir": str(getattr(self.cfg, "output_dir", "")),
            "note": "overwrite checkpoint (only keep latest/best)",
        }
        with open(os.path.join(tmp_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # -------------------------
        # 5) atomic replace dst_dir
        # -------------------------
        if os.path.isdir(dst_dir):
            shutil.rmtree(dst_dir, ignore_errors=True)

        os.replace(tmp_dir, dst_dir)

        print(f"[ckpt] saved {tag} -> {dst_dir} (step={step})", flush=True)


    def load_checkpoint(self, ckpt_dir: str, strict_full: bool = True) -> int:
        """
        ckpt_dir:
        - recommended: .../checkpoints/latest  or .../checkpoints/best
        - also supports legacy: .../step_XXXXXXXX

        strict_full:
        - True: if cfg.save_full_ckpt=True, require trainer_state.pt
        - False: weights-only load allowed

        return: global_step
        """
        import os
        import json
        import re

        if not ckpt_dir or (not os.path.isdir(ckpt_dir)):
            raise FileNotFoundError(f"ckpt_dir not found: {ckpt_dir}")

        controlnet_dir = os.path.join(ckpt_dir, "controlnet")
        if not os.path.isdir(controlnet_dir):
            raise FileNotFoundError(f"controlnet dir missing: {controlnet_dir}")

        # -------------------------
        # 1) load controlnet weights (replace module)
        # -------------------------
        self.controlnet = ControlNetModel.from_pretrained(controlnet_dir).to(self.device)
        self.controlnet.train()

        # -------------------------
        # 2) load ctrl_gen EnergyHead  [Option A]
        # -------------------------
        vfm_path = os.path.join(ckpt_dir, "vfm_control.pt")
        if os.path.isfile(vfm_path):
            if hasattr(self, "ctrl_gen") and self.ctrl_gen is not None:
                missing, unexpected = self.ctrl_gen.load_ckpt(vfm_path, strict=False)
                if missing or unexpected:
                    print(f"[ckpt] vfm_control loaded strict=False | missing={len(missing)} unexpected={len(unexpected)}")
            else:
                print("[WARN] ctrl_gen is None; skip loading vfm_control.pt")
        else:
            print("[ckpt] vfm_control.pt not found; EnergyHead remains random-init (NOT recommended).")

        # -------------------------
        # 3) rebuild optimizer to bind new params (controlnet + energy_head)
        # -------------------------
        head_params = []
        if hasattr(self, "ctrl_gen") and self.ctrl_gen is not None and hasattr(self.ctrl_gen, "energy_head"):
            head_params = list(self.ctrl_gen.energy_head.parameters())

        self.optimizer = AdamW(
            list(self.controlnet.parameters()) + head_params,
            lr=float(self.cfg.lr),
        )

        # -------------------------
        # 4) determine global_step (priority: trainer_state.pt > meta.json > folder name)
        # -------------------------
        global_step = 0

        state_path = os.path.join(ckpt_dir, "trainer_state.pt")
        if os.path.exists(state_path):
            payload = torch.load(state_path, map_location="cpu")
            if isinstance(payload, dict):
                if "global_step" in payload:
                    global_step = int(payload.get("global_step", 0))
                elif "step" in payload:
                    global_step = int(payload.get("step", 0))

                # restore best tracking
                if "best_psnr" in payload:
                    self.best_psnr = float(payload.get("best_psnr", 0.0))
                if "best_step" in payload:
                    self.best_step = int(payload.get("best_step", 0))

                # optimizer/scaler (safe load)
                if payload.get("optimizer", None) is not None:
                    try:
                        self.optimizer.load_state_dict(payload["optimizer"])
                    except Exception as e:
                        print(f"[WARN] optimizer state load failed (param groups changed). Reset optimizer. err={e}")

                if payload.get("scaler", None) is not None and getattr(self, "scaler", None) is not None:
                    try:
                        self.scaler.load_state_dict(payload["scaler"])
                    except Exception as e:
                        print(f"[WARN] scaler state load failed, reset scaler. err={e}")
        else:
            if getattr(self.cfg, "save_full_ckpt", False) and strict_full:
                raise FileNotFoundError(f"trainer_state.pt missing: {state_path}")
            print("[ckpt] trainer_state.pt not found, loaded weights only (optimizer/scaler reset).")

        # fallback: meta.json
        if global_step <= 0:
            meta_path = os.path.join(ckpt_dir, "meta.json")
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if "global_step" in meta:
                        global_step = int(meta.get("global_step", 0))
                    elif "step" in meta:
                        global_step = int(meta.get("step", 0))
                except Exception:
                    pass

        # legacy: folder name step_XXXXXXXX
        if global_step <= 0:
            m = re.search(r"step_(\d+)", os.path.basename(ckpt_dir))
            global_step = int(m.group(1)) if m else 0

        print(f"[ckpt] loaded from {ckpt_dir}, resume global_step={global_step}")
        return global_step

    # ----------------------------
    # training
    # ----------------------------
    def train(self, train_loader, val_loader=None, resume_ckpt: Optional[str] = None):
        cfg = self.cfg

        # -------------------------
        # resume (user-specified > auto-latest > none)
        # -------------------------
        global_step = 0
        ckpt_used = None

        if resume_ckpt:
            ckpt_used = resume_ckpt
        else:
            ckpt_root = self._ckpt_root_dir()
            latest_dir = os.path.join(ckpt_root, "latest")
            ckpt_used = latest_dir if os.path.isdir(latest_dir) else self._find_latest_checkpoint()

        if ckpt_used:
            try:
                strict_full = bool(getattr(cfg, "save_full_ckpt", False))
                global_step = int(self.load_checkpoint(ckpt_used, strict_full=strict_full))
            except Exception as e:
                print(f"[WARN] resume failed from {ckpt_used}. Start from scratch. err={e}")
                global_step = 0
        else:
            print("[ckpt] no checkpoint found. Start from scratch.")

        if global_step >= int(cfg.train_steps):
            print(f"[WARN] global_step({global_step}) >= train_steps({cfg.train_steps}). Nothing to train.")
            return

        # -------------------------
        # validation schedule
        # -------------------------
        val_every = int(getattr(cfg, "val_every", 0) or 0)
        val_batches = int(getattr(cfg, "val_batches", 10) or 10)
        do_val = (val_loader is not None) and (val_every > 0)

        # -------------------------
        # Option B: disc_map gated diffusion loss
        # -------------------------
        # weight map: w = 1 + disc_scale * D, then clamp
        disc_scale = float(getattr(cfg, "ssa_disc_scale", 1.0))   # e.g., 0.5~2.0
        w_min = float(getattr(cfg, "ssa_w_min", 1.0))             # e.g., 1.0
        w_max = float(getattr(cfg, "ssa_w_max", 3.0))             # e.g., 2.0~4.0
        # optional: soften weights (gamma < 1 reduces aggressiveness)
        w_gamma = float(getattr(cfg, "ssa_w_gamma", 1.0))         # e.g., 0.5 or 1.0
        # optional: smooth disc_map to avoid noisy over-weighting
        disc_smooth_ks = int(getattr(cfg, "ssa_disc_smooth_ksize", 0))  # 0 disables, else 3/5/7
        
        ssa_alpha_up = float(getattr(cfg, "ssa_alpha_up", 0.5))  # optional alpha upweighting

        def _smooth_disc(d: torch.Tensor) -> torch.Tensor:
            # d: [B,1,H,W] in [0,1]
            ks = int(disc_smooth_ks)
            if ks is None or ks <= 1:
                return d
            if ks % 2 == 0:
                ks += 1
            pad = ks // 2
            # avg blur (simple + stable)
            return F.avg_pool2d(d, kernel_size=ks, stride=1, padding=pad)

        # -------------------------
        # progress bar
        # -------------------------
        pbar = tqdm(
            total=int(cfg.train_steps),
            desc="train",
            unit="step",
            initial=int(global_step),
        )

        data_iter = iter(train_loader)

        # -------------------------
        # main loop
        # -------------------------
        while global_step < int(cfg.train_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            lr = batch["lr"].to(self.device, non_blocking=True)  # [B,3,128,128]
            hr = batch["hr"].to(self.device, non_blocking=True)  # [B,3,512,512]

            disc_map = batch.get("disc_map", None)
            if disc_map is None:
                disc_map = torch.zeros((lr.shape[0], 1, 512, 512), dtype=torch.float32)
            disc_map = disc_map.to(self.device, non_blocking=True).float().clamp(0, 1)  # [B,1,512,512]

            # optional smoothing to reduce noisy weight spikes
            disc_map = _smooth_disc(disc_map).clamp(0, 1)

            # LR_up to 512
            lr_up = F.interpolate(lr, size=(512, 512), mode="bicubic", align_corners=False)

            # build control (DINO backbone frozen, EnergyHead trainable)
            # Use your existing safe fallback in _build_control_train if you prefer.
            try:
                # if you kept forward_train_with_energy, you can call it; energy is not used here
                control_image, _ = self.ctrl_gen.forward_train_with_energy(lr_up)
            except Exception as e:
                if not getattr(self, "_warned_control_fallback_optb", False):
                    print(f"[WARN] VFM control failed once, fallback to lr-edge control. err={e}")
                    self._warned_control_fallback_optb = True
                control_image = make_control_image_from_lr_fallback(lr_up)

            # target latents
            with torch.no_grad():
                latents = self._vae_encode(hr)

            bsz = latents.shape[0]
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=self.device,
                dtype=torch.long,
            )
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = self.empty_prompt_embeds.repeat(bsz, 1, 1)

            # --------- forward & loss ---------
            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.autocast_dtype):
                down_samples, mid_sample = self.controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=control_image,
                    return_dict=False,
                )

                noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample,
                    return_dict=False,
                )[0]

                # ---- Option B: weighted diffusion loss ----
                # noise_pred/noise: [B,4,h,w] where h=w=512/8=64 (SD VAE)
                # disc_map: [B,1,512,512] -> downsample to [B,1,h,w]
                h, w = noise_pred.shape[-2], noise_pred.shape[-1]
                disc_lat = F.interpolate(disc_map, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)

                
                weight = (1.0 + disc_scale * disc_lat).clamp(w_min, w_max)
                if w_gamma != 1.0:
                    weight = torch.pow(weight, w_gamma)

                # --- warmup + ramp ---
                warm = int(getattr(cfg, "ssa_warmup_steps", 3000))
                ramp = int(getattr(cfg, "ssa_ramp_steps", 3000))

                if global_step < warm:
                    alpha = 0.0
                elif global_step < warm + ramp:
                    alpha = float(global_step - warm) / float(max(ramp, 1)) * ssa_alpha_up
                else:
                    alpha = ssa_alpha_up

                # blend with uniform weight=1 (prevents early collapse)
                weight = (1.0 - alpha) + alpha * weight

                # expand to 4 latent channels
                weight = weight.expand(-1, noise_pred.shape[1], -1, -1)

                # --- mean normalize (critical!) ---
                w_mean = weight.mean().detach()
                weight = weight / torch.clamp(w_mean, min=1e-6)

                diff2 = (noise_pred.float() - noise.float()) ** 2
                loss_diff = (weight * diff2).mean()

                loss = loss_diff / int(cfg.grad_accum)

            
                
            
            # backward
            if cfg.mixed_precision == "fp16":
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # optimizer step (grad accum)
            if (global_step + 1) % int(cfg.grad_accum) == 0:
                if cfg.mixed_precision == "fp16":
                    self.scaler.unscale_(self.optimizer)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

            # step update
            global_step += 1
            pbar.update(1)

            # postfix
            if global_step % 1 == 0:
                pbar.set_postfix({
                    "loss": float(loss.detach().cpu()),
                    "diff_w": float(loss_diff.detach().cpu()),
                    "disc_m": float(disc_lat.mean().detach().cpu()),
                    "w_mean": float(w_mean.detach().cpu()),
                    "alpha": float(alpha),
                    "disc_s": float(disc_scale),
                    "w": f"[{w_min:.1f},{w_max:.1f}]",
                    "g": float(w_gamma),
                })

            # validation + best ckpt
            if do_val and (global_step % val_every == 0):
                try:
                    metrics = self.validate(val_loader, step=global_step, max_batches=val_batches)
                    if isinstance(metrics, dict) and metrics:
                        show = {f"val_{k}": float(v) for k, v in metrics.items()}

                        cur = pbar.postfix if isinstance(pbar.postfix, dict) else {}
                        pbar.set_postfix({**cur, **show})

                        cur_psnr = float(metrics.get("PSNR", 0.0))
                        if cur_psnr > float(self.best_psnr):
                            self.best_psnr = cur_psnr
                            self.best_step = int(global_step)
                            print(f"[best] PSNR improved to {self.best_psnr:.6f} @ step {self.best_step}, saving best...")
                            self.save_checkpoint(global_step, is_best=True)
                except Exception as e:
                    print(f"[WARN] validate failed at step={global_step}: {e}")

            
            

            # latest checkpoint
            if (global_step % int(cfg.save_every) == 0):
                self.save_checkpoint(global_step, is_best=False)

        pbar.close()
        self.save_checkpoint(global_step, is_best=False)


    # ----------------------------
    # sampling
    # ----------------------------
    @torch.no_grad()
    def sample_sr(self, lr: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        self.controlnet.eval()
        self.ctrl_gen.eval()

        num_steps = int(num_steps or getattr(self.cfg, "sample_steps", 20))
        lr_up = F.interpolate(lr, size=(512, 512), mode="bicubic", align_corners=False)

        # ✅ use eval control path
        control = self._build_control_eval(lr_up)

        B = lr.shape[0]
        latents = torch.randn((B, 4, 64, 64), device=self.device)

        self.noise_scheduler.set_timesteps(num_steps, device=self.device)

        for t in self.noise_scheduler.timesteps:
            encoder_hidden_states = self.empty_prompt_embeds.repeat(B, 1, 1)
            down, mid = self.controlnet(
                latents,
                t,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=control,
                return_dict=False,
            )
            noise_pred = self.unet(
                latents,
                t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down,
                mid_block_additional_residual=mid,
                return_dict=False,
            )[0]
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        sr = self._vae_decode(latents)

        # back to train
        self.controlnet.train()
        self.ctrl_gen.train()
        return sr


    # ----------------------------
    # validation + save SR/GT/control
    # ----------------------------
    @torch.no_grad()
    def validate(self, val_loader, step: int = 0, max_batches: int = 10) -> Dict[str, float]:
        """
        Validation protocol:
        - Generate SR from LR
        - Apply color correction to SR using ONLY LR_up as reference (no HR used in correction)
        - Compute metrics against HR GT (evaluation only)
        - Save a small set of visuals (raw SR, corrected SR, GT, LR_up, CONTROL)
        """
        self.controlnet.eval()
        self.ctrl_gen.eval()
        self.unet.eval()

        # raw metrics (before color correction)
        psnr_raw_list, ssim_raw_list = [], []
        lpips_raw_list, stlpips_raw_list = [], []
        # corrected metrics (after color correction)
        psnr_list, ssim_list = [], []
        lpips_list, stlpips_list = [], []

        num_steps = int(getattr(self.cfg, "sample_steps", 20))

        # save one random example per validate() call
        save_vis = True
        vis_saved = False
        vis_batch_idx = int(np.random.randint(0, max(1, int(max_batches))))

        vis_root = os.path.join(self.cfg.output_dir, "val_vis", f"step_{step:07d}")
        os.makedirs(vis_root, exist_ok=True)

        def to_u8(img_3chw: torch.Tensor) -> np.ndarray:
            x = img_3chw.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            return (x * 255.0 + 0.5).astype(np.uint8)

        for i, batch in enumerate(val_loader):
            if i >= int(max_batches):
                break

            lr = batch["lr"].to(self.device, non_blocking=True)  # [B,3,128,128]
            hr = batch["hr"].to(self.device, non_blocking=True)  # [B,3,512,512]

            # build control from LR_up
            lr_up = F.interpolate(lr, size=(512, 512), mode="bicubic", align_corners=False)  # [B,3,512,512]
            control = self._build_control_eval(lr_up)  # [B,3,512,512] (or your control channels)

            # diffusion sampling
            B = lr.shape[0]
            latents = torch.randn((B, 4, 64, 64), device=self.device)
            self.noise_scheduler.set_timesteps(num_steps, device=self.device)

            encoder_hidden_states = self.empty_prompt_embeds.repeat(B, 1, 1)

            for t in self.noise_scheduler.timesteps:
                down, mid = self.controlnet(
                    latents,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=control,
                    return_dict=False,
                )
                noise_pred = self.unet(
                    latents,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down,
                    mid_block_additional_residual=mid,
                    return_dict=False,
                )[0]
                latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

            sr = self._vae_decode(latents)  # [B,3,512,512] in [0,1]

            # ✅ color correction using ONLY LR (LR_up as reference), no HR involved in correction
            sr_cc = self._apply_color_correction_batch(sr, lr_up)

            # ---- metrics: raw & corrected (computed against HR GT for evaluation) ----
            psnr_raw_list.append(float(self.metrics.psnr(sr, hr)))
            ssim_raw_list.append(float(self.metrics.ssim(sr, hr)))
            lpips_raw_list.append(float(self.metrics.lpips(sr, hr)))
            stlpips_raw_list.append(float(self.metrics.shift_tolerant_lpips(sr, hr)))

            psnr_list.append(float(self.metrics.psnr(sr_cc, hr)))
            ssim_list.append(float(self.metrics.ssim(sr_cc, hr)))
            lpips_list.append(float(self.metrics.lpips(sr_cc, hr)))
            stlpips_list.append(float(self.metrics.shift_tolerant_lpips(sr_cc, hr)))

            # ---- save one example ----
            if save_vis and (not vis_saved) and (i == vis_batch_idx):
                j = int(np.random.randint(0, B))

                Image.fromarray(to_u8(sr[j])).save(os.path.join(vis_root, "SR_raw.png"))
                Image.fromarray(to_u8(sr_cc[j])).save(os.path.join(vis_root, "SR_cc.png"))
                Image.fromarray(to_u8(hr[j])).save(os.path.join(vis_root, "GT.png"))
                Image.fromarray(to_u8(lr_up[j])).save(os.path.join(vis_root, "LR_up.png"))

                # CONTROL might not be RGB; try best-effort save if it is 3ch
                try:
                    c = control[j]
                    if c.ndim == 3 and c.shape[0] == 3:
                        Image.fromarray(to_u8(c)).save(os.path.join(vis_root, "CONTROL.png"))
                except Exception:
                    pass

                with open(os.path.join(vis_root, "metrics.txt"), "w") as f:
                    f.write(f"step: {step}\n")
                    f.write(f"batch_idx: {i}\n")
                    f.write(f"sample_idx: {j}\n")
                    f.write(f"sample_steps: {num_steps}\n")
                    f.write("\n# Color correction\n")
                    f.write(f"enable_color_correction: {bool(getattr(self.cfg, 'enable_color_correction', True))}\n")
                    f.write(f"color_correction_method: {getattr(self.cfg, 'color_correction_method', 'reinhard')}\n")
                    f.write(f"color_correction_ref: {getattr(self.cfg, 'color_correction_ref', 'lr')} (LR_up)\n")
                    f.write(f"color_lowpass_ksize: {getattr(self.cfg, 'color_lowpass_ksize', 0)}\n")

                    f.write("\n# RAW (before color correction)\n")
                    f.write(f"PSNR_raw: {psnr_raw_list[-1]:.6f}\n")
                    f.write(f"SSIM_raw: {ssim_raw_list[-1]:.6f}\n")
                    f.write(f"LPIPS_raw: {lpips_raw_list[-1]:.6f}\n")
                    f.write(f"ST-LPIPS_raw: {stlpips_raw_list[-1]:.6f}\n")

                    f.write("\n# CC (after color correction; ref=LR_up)\n")
                    f.write(f"PSNR: {psnr_list[-1]:.6f}\n")
                    f.write(f"SSIM: {ssim_list[-1]:.6f}\n")
                    f.write(f"LPIPS: {lpips_list[-1]:.6f}\n")
                    f.write(f"ST-LPIPS: {stlpips_list[-1]:.6f}\n")

                vis_saved = True
                print(f"[val] saved SR_raw/SR_cc/GT/LR_up to: {vis_root}")
                self._prune_val_vis()

        # restore training modes (keep unet state consistent with training loop)
        self.controlnet.train()
        self.ctrl_gen.train()
        self.unet.train()

        results = {
            # ✅ main metrics use corrected SR (after color correction w/ LR reference)
            "PSNR": float(np.mean(psnr_list)) if psnr_list else 0.0,
            "SSIM": float(np.mean(ssim_list)) if ssim_list else 0.0,
            "LPIPS": float(np.mean(lpips_list)) if lpips_list else 0.0,
            "ST-LPIPS": float(np.mean(stlpips_list)) if stlpips_list else 0.0,

            # extra: raw metrics
            "PSNR_raw": float(np.mean(psnr_raw_list)) if psnr_raw_list else 0.0,
            "SSIM_raw": float(np.mean(ssim_raw_list)) if ssim_raw_list else 0.0,
            "LPIPS_raw": float(np.mean(lpips_raw_list)) if lpips_raw_list else 0.0,
            "ST-LPIPS_raw": float(np.mean(stlpips_raw_list)) if stlpips_raw_list else 0.0,
        }

        print(
            f"[val @ step {step}] "
            f"PSNR(raw)={results['PSNR_raw']:.3f} -> PSNR(cc|ref=LR)={results['PSNR']:.3f} | "
            f"SSIM(raw)={results['SSIM_raw']:.4f} -> SSIM(cc|ref=LR)={results['SSIM']:.4f}"
        )
        return results

