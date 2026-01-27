# model.py
# -*- coding: utf-8 -*-
"""
SR core: Dynamic kernel prediction (non-generative) + optional residual.
Boundary/detail constraint: FFT high-pass consistency loss (Scheme 2).

External interface:
  - model(lr) -> sr
  - model.compute_loss(lr, hr, return_debug=False) -> loss (and debug dict)
  - model.set_epoch(epoch) (optional)

Notes:
- Inference/validation without HR: just call model(lr).
- FFT loss is computed ONLY inside compute_loss(lr, hr).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Utils
# =============================================================================
def _minmax_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample min-max normalize to [0,1]. x: [B,1,H,W]."""
    B = x.shape[0]
    v = x.view(B, -1)
    mn = v.min(dim=1)[0].view(B, 1, 1, 1)
    mx = v.max(dim=1)[0].view(B, 1, 1, 1)
    return (x - mn) / (mx - mn + eps)


def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """x: [B,3,H,W] -> [B,1,H,W]"""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def _grad_mag_gray(x_gray: torch.Tensor) -> torch.Tensor:
    gx = x_gray[:, :, :, 1:] - x_gray[:, :, :, :-1]
    gy = x_gray[:, :, 1:, :] - x_gray[:, :, :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def _make_hr_int_grid(H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.arange(H, device=device)
    xs = torch.arange(W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(dtype=dtype)


def _hr_int_to_hr_norm(hr_xy_int: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """(x,y) int -> normalized [-1,1] at pixel centers."""
    x = hr_xy_int[:, 0].to(torch.float32)
    y = hr_xy_int[:, 1].to(torch.float32)
    x_norm = (x + 0.5) / float(W) * 2.0 - 1.0
    y_norm = (y + 0.5) / float(H) * 2.0 - 1.0
    return torch.stack([x_norm, y_norm], dim=1).to(dtype=torch.float32, device=hr_xy_int.device)


def _image_grad_l1(sr: torch.Tensor, hr: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Gradient L1 between sr and hr (grayscale).
    sr/hr: [B,3,H,W] in [0,1]
    weight: [B,1,H,W] optional
    """
    sr_g = _rgb_to_gray(sr)
    hr_g = _rgb_to_gray(hr)

    sr_dx = sr_g[:, :, :, 1:] - sr_g[:, :, :, :-1]
    hr_dx = hr_g[:, :, :, 1:] - hr_g[:, :, :, :-1]
    sr_dy = sr_g[:, :, 1:, :] - sr_g[:, :, :-1, :]
    hr_dy = hr_g[:, :, 1:, :] - hr_g[:, :, :-1, :]

    dx_l1 = (sr_dx - hr_dx).abs()
    dy_l1 = (sr_dy - hr_dy).abs()

    if weight is not None:
        dx_l1 = dx_l1 * weight[:, :, :, :-1]
        dy_l1 = dy_l1 * weight[:, :, :-1, :]

    return dx_l1.mean() + dy_l1.mean()


def _get_points_rgb(img: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """img: [3,H,W] -> [N,3] by flattened idx"""
    return img.permute(1, 2, 0).reshape(-1, 3)[idx]


def _get_points_map(map1: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """map1: [1,H,W] -> [N]"""
    return map1.reshape(-1)[idx]


def _fuse_keep_color(lr_up: torch.Tensor, sr_pred: torch.Tensor) -> torch.Tensor:
    """
    Hard color-preserving fusion (Scheme B).
    Keep chroma/color from lr_up, inject only luminance/detail from sr_pred.

    Implementation in RGB space using grayscale residual:
      sr_final = lr_up + (gray(sr_pred) - gray(lr_up))  (broadcast to 3 channels)

    lr_up, sr_pred: [B,3,H,W] in [0,1]
    return: [B,3,H,W] in [0,1]
    """
    lr_g = _rgb_to_gray(lr_up)      # [B,1,H,W]
    sr_g = _rgb_to_gray(sr_pred)    # [B,1,H,W]
    delta = sr_g - lr_g             # luminance residual
    sr_final = (lr_up + delta.repeat(1, 3, 1, 1)).clamp(0, 1)
    return sr_final




# =============================================================================
# Positional Encoding
# =============================================================================
class FourierPositionalEncoding(nn.Module):
    def __init__(self, num_bands: int = 10):
        super().__init__()
        self.num_bands = int(num_bands)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        if xy.ndim != 2 or xy.shape[-1] != 2:
            raise ValueError(f"xy must be [N,2], got {tuple(xy.shape)}")
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        outs = []
        for k in range(self.num_bands):
            freq = (2.0 ** k) * math.pi
            outs.append(torch.sin(freq * x))
            outs.append(torch.cos(freq * x))
            outs.append(torch.sin(freq * y))
            outs.append(torch.cos(freq * y))
        return torch.cat(outs, dim=1)


# =============================================================================
# Local Encoder (LR -> feature map)
# =============================================================================
class LocalEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, feat_ch: int = 64):
        super().__init__()
        c = int(feat_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.GELU(),
        )

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        return self.net(lr)


# =============================================================================
# Kernel Predictor MLP
# =============================================================================
class KernelMLP(nn.Module):
    """
    Multi-head kernel predictor.

    input = [local_feat(center), PE(hr_xy_norm), lr_frac(dx,dy)]
    output:
      - kernel_logits: [*, heads, K*K]
      - gate_logits:   [*, heads]  (mixture weights over heads)
    """
    def __init__(
        self,
        feat_dim: int,
        pe_dim: int,
        kernel_size: int = 7,
        hidden: int = 256,
        depth: int = 5,
        num_heads: int = 4,
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.num_heads = int(num_heads)
        K2 = self.kernel_size * self.kernel_size
        in_dim = int(feat_dim + pe_dim + 2)

        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        self.backbone = nn.Sequential(*layers)

        # heads * K2 kernel logits + heads gate logits
        self.head_kernel = nn.Linear(hidden, self.num_heads * K2)
        self.head_gate = nn.Linear(hidden, self.num_heads)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)  # [*, hidden]
        klogits = self.head_kernel(h)  # [*, heads*K2]
        glogits = self.head_gate(h)    # [*, heads]
        return klogits, glogits

# =============================================================================
# Optional residual refiner (color-safe groups=3)
# =============================================================================
class ResidualRefiner(nn.Module):
    """
    Color-safe refiner (no cross-channel mixing):
    groups=3 so each channel processed independently.
    input channels 6: [lr_up_rgb, res0_rgb] -> output residual RGB
    """
    def __init__(self, ch_per_group: int = 16):
        super().__init__()
        hidden = 3 * int(ch_per_group)
        self.net = nn.Sequential(
            nn.Conv2d(6, hidden, 3, 1, 1, groups=3),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=3),
            nn.GELU(),
            nn.Conv2d(hidden, 3, 3, 1, 1, groups=3),
        )

    def forward(self, lr_up: torch.Tensor, res0: torch.Tensor) -> torch.Tensor:
        x = torch.cat([lr_up, res0], dim=1)
        return self.net(x)


# =============================================================================
# Config
# =============================================================================
@dataclass
class SRModelConfig:
    # ----------------------------
    # Basic SR
    # ----------------------------
    scale: int = 4

    # encoder & kernel predictor
    feat_ch: int = 64
    pe_bands: int = 10
    mlp_hidden: int = 256
    mlp_depth: int = 5

    # kernel prediction
    kernel_size: int = 7  # odd
    kernel_allow_negative: bool = True

    # ----------------------------
    # Multi-head dynamic kernel (S4)
    # ----------------------------
    kernel_heads: int = 4            # number of kernel heads
    kernel_gate_tau: float = 1.0     # softmax temperature for head-gate
    kernel_logit_norm: bool = True   # normalize logits before softmax/tanh to reduce saturation

    # ----------------------------
    # training sampling
    # ----------------------------
    num_points: int = 4096
    saliency_ratio: float = 0.7
    loss_alpha: float = 3.0  # pixel-weight strength using W_edge (and optionally HoVer band)

    # gradient consistency (base)
    lambda_grad: float = 0.1
    grad_crop: int = 128

    # inference chunking
    infer_chunk: int = 8192

    # temperature annealing (tau)
    tau_start: float = 1.0
    tau_end: float = 0.5
    tau_warm_epochs: int = 2
    tau_anneal_epochs: int = 8

    # residual SR
    use_residual: bool = True
    use_res_refiner: bool = True

    # ----------------------------
    # HoVer-Net guidance (maps + losses)
    # ----------------------------
    use_hover: bool = True
    hover_sample_ratio: float = 0.5
    hover_bnd_sigma: float = 2.0
    hover_mask_sigma: float = 1.5   # NOTE: 统一命名（原 yaml 里是 hover_in_sigma）

    hover_bnd_weight: float = 2.0
    hover_in_weight: float = 0.5

    lambda_hover_grad: float = 0.2
    lambda_in_gray: float = 0.05

    in_gray_crop: int = 192

    # ----------------------------
    # HoVer-Net condition injection (train-time teacher)
    # Validation has NO hover maps, so student path is unconditional.
    # ----------------------------
    use_hover_cond: bool = True
    hover_cond_bnd: bool = True
    hover_cond_mask: bool = True
    hover_cond_strength: float = 1.0
    hover_cond_blur_lr: float = 0.0

    # condition dropout (applied to teacher condition during training for robustness)
    hover_cond_dropout: float = 0.5

    # ----------------------------
    # Consistency distillation: student (no hover) learns from teacher (with hover)
    # ----------------------------
    use_ema_teacher: bool = True
    ema_decay: float = 0.999
    ema_update_every: int = 1

    lambda_cond_consistency: float = 0.15
    cond_consistency_crop: int = 192
    cond_consistency_hp_sigma: float = 1.5

    distill_warmup_epochs: int = 10
    distill_ramp_epochs: int = 10

    # ----------------------------
    # Diagnostics (optional)
    # ----------------------------
    diag_kernel_points: int = 256

# =============================================================================
# Main SR Model
# =============================================================================
class SRModel(nn.Module):
    def __init__(self, cfg: Optional[SRModelConfig] = None):
        super().__init__()
        self.cfg = cfg or SRModelConfig()
        if self.cfg.kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd.")

        self.encoder = LocalEncoder(in_ch=3, feat_ch=self.cfg.feat_ch)
        self.pe = FourierPositionalEncoding(num_bands=self.cfg.pe_bands)
        pe_dim = 4 * self.cfg.pe_bands

        # --------- Multi-head kernel predictor (S4) ----------
        self.kernel_heads = int(getattr(self.cfg, "kernel_heads", 4))
        if self.kernel_heads < 1:
            self.kernel_heads = 1

        self.kernel_mlp = KernelMLP(
            feat_dim=self.cfg.feat_ch,
            pe_dim=pe_dim,
            kernel_size=self.cfg.kernel_size,
            hidden=self.cfg.mlp_hidden,
            depth=self.cfg.mlp_depth,
            num_heads=self.kernel_heads,
        )

        self.res_refiner = ResidualRefiner() if self.cfg.use_res_refiner else None

        # --- HoVer condition fuser (teacher path) ---
        cond_ch = 0
        if getattr(self.cfg, "use_hover_cond", True):
            if getattr(self.cfg, "hover_cond_bnd", True):
                cond_ch += 1
            if getattr(self.cfg, "hover_cond_mask", True):
                cond_ch += 1
        self._cond_ch = int(cond_ch)

        if self._cond_ch > 0:
            self.cond_fuser = nn.Sequential(
                nn.Conv2d(self.cfg.feat_ch + self._cond_ch, self.cfg.feat_ch, kernel_size=1, stride=1, padding=0),
                nn.GELU(),
                nn.Conv2d(self.cfg.feat_ch, self.cfg.feat_ch, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.cond_fuser = None

        # kernel offsets (K*K)
        r = self.cfg.kernel_size // 2
        offsets = [(ox, oy) for oy in range(-r, r + 1) for ox in range(-r, r + 1)]
        self.register_buffer("_kernel_offsets", torch.tensor(offsets, dtype=torch.float32), persistent=False)

        # tau state
        self._epoch = 0
        self._tau = float(self.cfg.tau_start)

        # ---------------------------------------------------
        # EMA Teacher: keep EMA shadow on CPU float32 ALWAYS
        # ---------------------------------------------------
        self._use_ema_teacher = bool(getattr(self.cfg, "use_ema_teacher", True))
        self._ema_decay = float(getattr(self.cfg, "ema_decay", 0.999))
        self._ema_update_every = int(getattr(self.cfg, "ema_update_every", 1))
        self._ema_step = 0

        # name -> CPU float32 tensor
        self._ema_shadow = {}
        if self._use_ema_teacher:
            for n, p in self.named_parameters():
                if p.requires_grad:
                    self._ema_shadow[n] = p.detach().to(device="cpu", dtype=torch.float32).clone()

    # ----------------------------
    # temperature schedule
    # ----------------------------
    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def _compute_tau(self, epoch: int) -> float:
        s = float(self.cfg.tau_start)
        e = float(self.cfg.tau_end)
        warm = int(self.cfg.tau_warm_epochs)
        anneal = int(self.cfg.tau_anneal_epochs)
        if anneal <= 0:
            return s
        if epoch <= warm:
            return s
        t = min(1.0, max(0.0, (epoch - warm) / float(anneal)))
        return (1.0 - t) * s + t * e

    # ----------------------------
    # edge saliency (from LR-up)
    # ----------------------------
    @torch.no_grad()
    def compute_edge_saliency(self, lr: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        H_hr, W_hr = out_hw
        lr_up = F.interpolate(lr, size=(H_hr, W_hr), mode="bicubic", align_corners=False).clamp(0, 1)
        g = _rgb_to_gray(lr_up)
        W_edge = _minmax_norm(_grad_mag_gray(g))
        return W_edge


    # ----------------------------
    # HoVer condition (LR space)
    # ----------------------------
    def _prepare_hover_cond_lr(
        self,
        lr: torch.Tensor,
        hover_bnd: Optional[torch.Tensor],
        hover_mask: Optional[torch.Tensor],
        apply_dropout: bool = True,
    ) -> Optional[torch.Tensor]:
        """Build condition maps at LR resolution: [B,Cc,H_lr,W_lr] in [0,1]."""
        if (self._cond_ch <= 0) or (not getattr(self.cfg, "use_hover_cond", True)):
            return None

        B, _, H_lr, W_lr = lr.shape
        outs: List[torch.Tensor] = []

        def _to01(m: torch.Tensor) -> torch.Tensor:
            if m.ndim == 3:
                m_ = m.unsqueeze(1)
            else:
                m_ = m
            if m_.shape[1] != 1:
                m_ = m_[:, :1]
            m_ = m_.to(device=lr.device, dtype=torch.float32)
            if m_.max() > 1.5:
                m_ = m_ / 255.0
            return (m_ > 0.5).float()

        if getattr(self.cfg, "hover_cond_bnd", True) and (hover_bnd is not None):
            hb = _to01(hover_bnd)
            hb_lr = F.interpolate(hb, size=(H_lr, W_lr), mode="nearest")
            outs.append(hb_lr)

        if getattr(self.cfg, "hover_cond_mask", True) and (hover_mask is not None):
            hm = _to01(hover_mask)
            hm_lr = F.interpolate(hm, size=(H_lr, W_lr), mode="nearest")
            outs.append(hm_lr)

        if len(outs) == 0:
            return None

        cond = torch.cat(outs, dim=1)  # [B,Cc,H_lr,W_lr]

        # optional blur at LR (avgpool)
        sigma = float(getattr(self.cfg, "hover_cond_blur_lr", 0.0))
        if sigma > 0:
            k = int(round(sigma * 4 + 1))
            if k % 2 == 0:
                k += 1
            pad = k // 2
            cond = F.pad(cond, (pad, pad, pad, pad), mode="replicate")
            cond = F.avg_pool2d(cond, kernel_size=k, stride=1)

        # dropout condition (robustness)
        p = float(getattr(self.cfg, "hover_cond_dropout", 0.0))
        if self.training and apply_dropout and p > 0:
            keep = (torch.rand(B, 1, 1, 1, device=lr.device) > p).float()
            cond = cond * keep

        s = float(getattr(self.cfg, "hover_cond_strength", 1.0))
        cond = (cond * s).clamp(0, 1)
        return cond

    def _encode_with_condition(
        self,
        lr: torch.Tensor,
        hover_bnd: Optional[torch.Tensor] = None,
        hover_mask: Optional[torch.Tensor] = None,
        apply_dropout: bool = True,
    ) -> torch.Tensor:
        """Encoder feature + optional HoVer condition injection."""
        feat = self.encoder(lr)
        if self.cond_fuser is None:
            return feat
        cond = self._prepare_hover_cond_lr(lr, hover_bnd, hover_mask, apply_dropout=apply_dropout)
        if cond is None:
            return feat
        x = torch.cat([feat, cond], dim=1)
        return self.cond_fuser(x)

    @staticmethod
    def _gray_highpass(x_rgb: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
        """High-pass on grayscale using (x - blur(x))."""
        g = _rgb_to_gray(x_rgb)
        if sigma <= 0:
            return g
        k = int(round(sigma * 4 + 1))
        if k % 2 == 0:
            k += 1
        pad = k // 2
        g_pad = F.pad(g, (pad, pad, pad, pad), mode="replicate")
        blur = F.avg_pool2d(g_pad, kernel_size=k, stride=1)
        return (g - blur)
    
    # ----------------------------
    # coord mapping HR -> LR continuous
    # ----------------------------
    def _lr_continuous_from_hr_int(
        self,
        hr_xy_int: torch.Tensor,
        lr_hw: Tuple[int, int],
        hr_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = float(self.cfg.scale)
        H_lr, W_lr = lr_hw
        H_hr, W_hr = hr_hw

        x_hr = hr_xy_int[:, 0].to(torch.float32)
        y_hr = hr_xy_int[:, 1].to(torch.float32)

        x_lr = (x_hr + 0.5) / scale - 0.5
        y_lr = (y_hr + 0.5) / scale - 0.5
        lr_xy_cont = torch.stack([x_lr, y_lr], dim=1).to(device=hr_xy_int.device, dtype=torch.float32)

        x0 = torch.floor(x_lr)
        y0 = torch.floor(y_lr)
        dx = (x_lr - x0).clamp(0.0, 1.0)
        dy = (y_lr - y0).clamp(0.0, 1.0)
        lr_frac = torch.stack([dx, dy], dim=1).to(device=hr_xy_int.device, dtype=torch.float32)

        x_norm = (x_lr + 0.5) / float(W_lr) * 2.0 - 1.0
        y_norm = (y_lr + 0.5) / float(H_lr) * 2.0 - 1.0
        lr_xy_norm = torch.stack([x_norm, y_norm], dim=1).to(device=hr_xy_int.device, dtype=torch.float32)

        return lr_xy_cont, lr_frac, lr_xy_norm

    def _sample_lr_rgb_neighbors(self, lr: torch.Tensor, lr_xy_cont: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        Returns rgb [B,N,K,3]
        """
        B, _, H_lr, W_lr = lr.shape
        N = lr_xy_cont.shape[0]
        K = offsets.shape[0]

        xy = lr_xy_cont.unsqueeze(1) + offsets.unsqueeze(0)  # [N,K,2]
        x_norm = (xy[..., 0] + 0.5) / float(W_lr) * 2.0 - 1.0
        y_norm = (xy[..., 1] + 0.5) / float(H_lr) * 2.0 - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1)  # [N,K,2]
        grid = grid.view(1, 1, N * K, 2).repeat(B, 1, 1, 1)  # [B,1,NK,2]

        samp = F.grid_sample(lr, grid, mode="bilinear", align_corners=False)  # [B,3,1,NK]
        samp = samp.squeeze(2).transpose(1, 2).contiguous().view(B, N, K, 3)
        return samp

    # ----------------------------
    # kernel weights with tau
    # ----------------------------
    def _kernel_weights_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: [..., K2]
        return: [..., K2]  (sum=1, non-negative if kernel_allow_negative=False)
        """
        # optional: normalize logits per-point to avoid softmax saturation (structural, no extra loss)
        if bool(getattr(self.cfg, "kernel_logit_norm", True)):
            m = logits.mean(dim=-1, keepdim=True)
            s = logits.std(dim=-1, keepdim=True).clamp_min(1e-6)
            logits = (logits - m) / s

        tau = float(self._tau)
        logits = logits / max(tau, 1e-6)

        if not self.cfg.kernel_allow_negative:
            return torch.softmax(logits, dim=-1)

        # allow negative weights branch (keep your original behavior)
        w = torch.tanh(logits)
        K2 = w.shape[-1]
        center = K2 // 2
        oh = F.one_hot(torch.tensor(center, device=w.device), num_classes=K2).to(w.dtype)
        w = w + oh.view(*([1] * (w.ndim - 1)), K2)  # add identity bias (broadcast)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)
        return w

    
    def _predict_kernel_weights(
        self,
        feat_lr: torch.Tensor,
        hr_xy_norm: torch.Tensor,
        lr_xy_norm_center: torch.Tensor,
        lr_frac: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return fused kernel weights: [B, N, K2]
        using multi-head mixture: w = sum_h softmax(gate)_h * w_h
        """
        B = feat_lr.shape[0]
        N = hr_xy_norm.shape[0]
        K2 = self.cfg.kernel_size * self.cfg.kernel_size
        H = int(getattr(self, "kernel_heads", 1))

        # center feature sampling
        grid = lr_xy_norm_center.view(1, 1, -1, 2).to(feat_lr.device).repeat(B, 1, 1, 1)
        cen = F.grid_sample(feat_lr, grid, mode="bilinear", align_corners=False)  # [B,C,1,N]
        cen = cen.squeeze(2).transpose(1, 2).contiguous()  # [B,N,C]

        pe = self.pe(hr_xy_norm.to(device=feat_lr.device, dtype=torch.float32))  # [N,pe_dim]
        pe = pe.unsqueeze(0).expand(B, -1, -1)
        frac = lr_frac.to(device=feat_lr.device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)

        inp = torch.cat([cen, pe, frac], dim=-1)  # [B,N,D]
        inp2 = inp.view(B * N, -1)

        # multi-head outputs
        klogits_flat, glogits_flat = self.kernel_mlp(inp2)  # [B*N, H*K2], [B*N, H]
        klogits = klogits_flat.view(B, N, H, K2)
        glogits = glogits_flat.view(B, N, H)

        # per-head kernel weights
        w_head = self._kernel_weights_from_logits(klogits.view(B, N * H, K2)).view(B, N, H, K2)

        # head mixing weights (softmax over heads)
        gate_tau = float(getattr(self.cfg, "kernel_gate_tau", 1.0))
        gate = torch.softmax(glogits / max(gate_tau, 1e-6), dim=-1)  # [B,N,H]
        gate = gate.unsqueeze(-1)  # [B,N,H,1]

        w = (gate * w_head).sum(dim=2)  # [B,N,K2]
        return w

    
    # ----------------------------
    # point prediction
    # ----------------------------
    def predict_points_base(
        self,
        lr: torch.Tensor,
        hr_xy_int: torch.Tensor,
        hr_hw: Tuple[int, int],
        feat_lr: Optional[torch.Tensor] = None,
        lr_up: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict RESIDUAL at HR points using dynamic kernel.

        Returns:
            residual_pts: [B,N,3]  (can be negative), where
                residual = sr_rgb - lr_up_rgb_at_points
        Notes:
            - lr_up should be bicubic-upsampled LR to HR size for consistency with training target (hr - lr_up).
            - If lr_up is None, it will be computed internally (bicubic).
        """
        B, _, H_lr, W_lr = lr.shape
        H_hr, W_hr = hr_hw

        if feat_lr is None:
            feat_lr = self.encoder(lr)

        # Ensure hr_xy_int is float32 but represents integer coords
        hr_xy_int = hr_xy_int.to(device=lr.device, dtype=torch.float32)

        # --- SR RGB prediction at points (0~1) ---
        hr_xy_norm = _hr_int_to_hr_norm(hr_xy_int, H_hr, W_hr)  # [N,2]
        lr_xy_cont, lr_frac, lr_xy_norm_center = self._lr_continuous_from_hr_int(
            hr_xy_int=hr_xy_int, lr_hw=(H_lr, W_lr), hr_hw=(H_hr, W_hr)
        )

        w = self._predict_kernel_weights(feat_lr, hr_xy_norm, lr_xy_norm_center, lr_frac)  # [B,N,K]
        offsets = self._kernel_offsets.to(device=lr.device, dtype=torch.float32)
        rgb = self._sample_lr_rgb_neighbors(lr, lr_xy_cont, offsets)  # [B,N,K,3]
        sr_rgb_pts = (w.unsqueeze(-1) * rgb).sum(dim=2).clamp(0, 1)    # [B,N,3] in [0,1]

        # --- Bicubic LR-up at HR points (0~1) ---
        if lr_up is None:
            lr_up = F.interpolate(lr, size=(H_hr, W_hr), mode="bicubic", align_corners=False).clamp(0, 1)
        else:
            # safety: ensure correct size and range
            if lr_up.shape[-2:] != (H_hr, W_hr):
                lr_up = F.interpolate(lr_up, size=(H_hr, W_hr), mode="bicubic", align_corners=False).clamp(0, 1)
            else:
                lr_up = lr_up.clamp(0, 1)

        # gather lr_up rgb at those integer points
        # hr_xy_int are integer coords in float; round & clamp for safety
        xy = torch.round(hr_xy_int).to(dtype=torch.long)
        x = xy[:, 0].clamp(0, W_hr - 1)
        y = xy[:, 1].clamp(0, H_hr - 1)
        idx = (y * W_hr + x)  # [N]

        lr_up_flat = lr_up.permute(0, 2, 3, 1).reshape(B, -1, 3)  # [B,HW,3]
        lr_up_pts = lr_up_flat[:, idx, :]  # [B,N,3]

        # --- residual ---
        residual_pts = sr_rgb_pts - lr_up_pts  # can be negative
        return residual_pts

    # ----------------------------
    # full SR image (inference)
    # ----------------------------
    @torch.no_grad()
    def super_resolve(
        self,
        lr: torch.Tensor,
        out_hw: Optional[Tuple[int, int]] = None,
        hover_bnd: Optional[torch.Tensor] = None,
        hover_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inference: SR = lr_up + residual (predicted by dynamic kernel).
        Final output uses hard color-preserving fusion.
        """
        B, _, H_lr, W_lr = lr.shape
        if out_hw is None:
            out_hw = (H_lr * self.cfg.scale, W_lr * self.cfg.scale)
        H_hr, W_hr = out_hw

        # bicubic upsample baseline
        lr_up = F.interpolate(lr, size=(H_hr, W_hr), mode="bicubic", align_corners=False).clamp(0, 1)

        # student feature (no hover condition in inference)
        feat_lr = self.encoder(lr)

        # HR grid
        hr_xy_int_full = _make_hr_int_grid(H_hr, W_hr, device=lr.device, dtype=torch.float32)
        N = hr_xy_int_full.shape[0]
        chunk = int(self.cfg.infer_chunk)

        # predict residual in chunks
        outs = []
        for st in range(0, N, chunk):
            ed = min(N, st + chunk)
            res_pts = self.predict_points_base(
                lr=lr,
                hr_xy_int=hr_xy_int_full[st:ed],
                hr_hw=(H_hr, W_hr),
                feat_lr=feat_lr,
                lr_up=lr_up,  # IMPORTANT: consistent residual definition
            )  # [B, n, 3]
            outs.append(res_pts)

        res_map = torch.cat(outs, dim=1).transpose(1, 2).contiguous().view(B, 3, H_hr, W_hr)  # residual map

        # optional residual refiner
        if self.cfg.use_residual:
            res0 = res_map
            if self.res_refiner is not None:
                res = self.res_refiner(lr_up, res0)
            else:
                res = res0
            sr_pred = (lr_up + res).clamp(0, 1)
        else:
            sr_pred = (lr_up + res_map).clamp(0, 1)

        # Final: keep color from lr_up, inject only luminance residual from sr_pred
        sr_final = _fuse_keep_color(lr_up, sr_pred)
        return sr_final


    def forward(
        self,
        lr: torch.Tensor,
        hover_bnd: Optional[torch.Tensor] = None,
        hover_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.super_resolve(lr, out_hw=None, hover_bnd=hover_bnd, hover_mask=hover_mask)

    
    # ----------------------------
    # training loss
    # ----------------------------
    def compute_loss(
        self,
        lr: torch.Tensor,
        hr: torch.Tensor,
        hover_bnd: Optional[torch.Tensor] = None,
        hover_mask: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ):
        """Compute training loss.

        Validation/inference has no hover maps; this function is only used in training where HR exists.
        We optimize the *student* path (unconditional, no hover condition in encoder) for robustness.
        A *teacher* path (encoder conditioned on hover maps) is used to distill sharper structures
        via a consistency loss, without requiring hover maps at validation.
        """
        B, _, H_lr, W_lr = lr.shape
        Bh, _, H_hr, W_hr = hr.shape
        if B != Bh:
            raise ValueError("lr/hr batch size mismatch")

        
        
        # tau schedule
        self._tau = float(self._compute_tau(self._epoch))

        # ---------------------------------------------------
        # EMA update (CPU shadow) - SAFE across devices
        # ---------------------------------------------------
        if self.training and getattr(self, "_use_ema_teacher", False) and len(getattr(self, "_ema_shadow", {})) > 0:
            self._ema_step += 1
            if (self._ema_step % max(1, int(getattr(self, "_ema_update_every", 1)))) == 0:
                d = float(getattr(self, "_ema_decay", 0.999))
                with torch.no_grad():
                    for n, p in self.named_parameters():
                        if (n in self._ema_shadow) and p.requires_grad:
                            cur_cpu = p.detach().to(device="cpu", dtype=torch.float32)
                            self._ema_shadow[n].mul_(d).add_(cur_cpu, alpha=(1.0 - d))

        # ----------------------------
        # Prepare maps (HR space)
        # ----------------------------
        lr_up = F.interpolate(lr, size=(H_hr, W_hr), mode="bicubic", align_corners=False).clamp(0, 1)
        W_edge = self.compute_edge_saliency(lr, out_hw=(H_hr, W_hr))  # [B,1,H,W] in [0,1]

        def _blur01(m: torch.Tensor, sigma: float) -> torch.Tensor:
            if sigma <= 0:
                return m
            k = int(round(sigma * 4 + 1))
            if k % 2 == 0:
                k += 1
            pad = k // 2
            mp = F.pad(m, (pad, pad, pad, pad), mode="replicate")
            return F.avg_pool2d(mp, kernel_size=k, stride=1)

        use_hover = bool(getattr(self.cfg, "use_hover", False))

        # boundary soft weight (HR)
        if use_hover and (hover_bnd is not None):
            hb = hover_bnd
            if hb.ndim == 3:
                hb = hb.unsqueeze(1)
            hb = hb[:, :1].to(device=lr.device, dtype=torch.float32)
            if hb.max() > 1.5:
                hb = hb / 255.0
            hb = (hb > 0.5).float()
            if hb.shape[-2:] != (H_hr, W_hr):
                hb = F.interpolate(hb, size=(H_hr, W_hr), mode="nearest")
            W_bnd_soft = _blur01(hb, float(getattr(self.cfg, "hover_bnd_sigma", 2.0)))
            W_bnd_soft = _minmax_norm(W_bnd_soft)
        else:
            W_bnd_soft = torch.zeros((B, 1, H_hr, W_hr), device=lr.device, dtype=torch.float32)

        # interior soft weight (HR)
        if use_hover and (hover_mask is not None):
            hm = hover_mask
            if hm.ndim == 3:
                hm = hm.unsqueeze(1)
            hm = hm[:, :1].to(device=lr.device, dtype=torch.float32)
            if hm.max() > 1.5:
                hm = hm / 255.0
            hm = (hm > 0.5).float()
            if hm.shape[-2:] != (H_hr, W_hr):
                hm = F.interpolate(hm, size=(H_hr, W_hr), mode="nearest")
            W_in_soft = _blur01(hm, float(getattr(self.cfg, "hover_mask_sigma", 1.5)))
            W_in_soft = _minmax_norm(W_in_soft)
        else:
            W_in_soft = torch.zeros((B, 1, H_hr, W_hr), device=lr.device, dtype=torch.float32)

        # sampling weight mix
        r = float(getattr(self.cfg, "hover_sample_ratio", 0.5)) if use_hover else 0.0
        W_mix = _minmax_norm(W_edge + r * W_bnd_soft)  # [B,1,H,W]

        # ----------------------------
        # student feature (no hover condition)
        # ----------------------------
        feat_student = self.encoder(lr)

        # ----------------------------
        # helpers: predict residual/sr crop from features
        # ----------------------------
        def _make_crop_xy_int(top: int, left: int, crop: int) -> torch.Tensor:
            ys = torch.arange(top, top + crop, device=lr.device)
            xs = torch.arange(left, left + crop, device=lr.device)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(dtype=torch.float32)

        def _predict_residual_crop(b: int, top: int, left: int, crop: int, feat_lr: torch.Tensor) -> torch.Tensor:
            hr_xy_int = _make_crop_xy_int(top, left, crop)
            res_pts = self.predict_points_base(
                lr=lr[b:b + 1],
                hr_xy_int=hr_xy_int,
                hr_hw=(H_hr, W_hr),
                feat_lr=feat_lr[b:b + 1],
                lr_up=lr_up[b:b + 1],
            )  # [1, crop*crop, 3]
            res_map = res_pts.transpose(1, 2).contiguous().view(1, 3, crop, crop)
            return res_map

        def _predict_sr_crop(b: int, top: int, left: int, crop: int, feat_lr: torch.Tensor) -> torch.Tensor:
            lr_up_crop = lr_up[b:b + 1, :, top:top + crop, left:left + crop]
            res0 = _predict_residual_crop(b, top, left, crop, feat_lr)
            if bool(getattr(self.cfg, "use_residual", True)):
                if self.res_refiner is not None:
                    res = self.res_refiner(lr_up_crop, res0)
                else:
                    res = res0
                sr = (lr_up_crop + res).clamp(0, 1)
            else:
                sr = (lr_up_crop + res0).clamp(0, 1)

            # 注意：训练 loss 仍然基于 residual 的点采样；这里 sr_crop 主要用于 grad/hover/consistency
            return sr

        def _predict_sr_crop_from_residual(b: int, top: int, left: int, crop: int, feat_lr: torch.Tensor) -> torch.Tensor:
            # same as _predict_sr_crop，但保留这个名字以兼容你之前版本里可能用到的调用习惯
            return _predict_sr_crop(b, top, left, crop, feat_lr)

        # ----------------------------
        # Pixel residual loss (sampled points)
        # ----------------------------
        N = int(getattr(self.cfg, "num_points", 4096))
        N_sal = int(round(N * float(getattr(self.cfg, "saliency_ratio", 0.7))))
        N_uni = max(0, N - N_sal)

        loss_pix = torch.tensor(0.0, device=lr.device)
        mean_edge = 0.0

        # diagnostic accumulators
        _acc_res_pred_abs = 0.0
        _acc_res_tgt_abs = 0.0
        _acc_wmix = 0.0

        for b in range(B):
            w_flat = W_mix[b, 0].reshape(-1) + 1e-6
            prob = w_flat / w_flat.sum()

            idx_sal = torch.multinomial(prob, num_samples=N_sal, replacement=True) if N_sal > 0 else None
            idx_uni = torch.randint(0, H_hr * W_hr, (N_uni,), device=lr.device) if N_uni > 0 else None
            idx = idx_uni if idx_sal is None else (idx_sal if idx_uni is None else torch.cat([idx_sal, idx_uni], dim=0))

            y = torch.div(idx, W_hr, rounding_mode="floor")
            x = idx - y * W_hr
            hr_xy_int = torch.stack([x, y], dim=1).to(dtype=torch.float32, device=lr.device)

            res_pred_pts = self.predict_points_base(
                lr=lr[b:b + 1],
                hr_xy_int=hr_xy_int,
                hr_hw=(H_hr, W_hr),
                feat_lr=feat_student[b:b + 1],
                lr_up=lr_up[b:b + 1],
            )[0]  # [N,3]

            lr_up_pts = _get_points_rgb(lr_up[b], idx)
            hr_pts = _get_points_rgb(hr[b], idx)
            res_tgt_pts = hr_pts - lr_up_pts

            ws = _get_points_map(W_mix[b], idx)
            mean_edge += float(ws.mean().detach().cpu())
            _acc_wmix += float(ws.mean().detach().cpu())

            alpha = float(getattr(self.cfg, "loss_alpha", 3.0))
            weight = 1.0 + alpha * ws

            l1 = (res_pred_pts - res_tgt_pts).abs().mean(dim=1)  # [N]
            loss_pix = loss_pix + (weight * l1).mean()

            _acc_res_pred_abs += float(res_pred_pts.abs().mean().detach().cpu())
            _acc_res_tgt_abs += float(res_tgt_pts.abs().mean().detach().cpu())

        loss_pix = loss_pix / B
        mean_edge = mean_edge / max(B, 1)

        # ----------------------------
        # Gradient consistency (student)
        # ----------------------------
        lam_g = float(getattr(self.cfg, "lambda_grad", 0.1))
        loss_grad = torch.tensor(0.0, device=lr.device)
        if lam_g > 0:
            crop = int(getattr(self.cfg, "grad_crop", 128))
            crop = min(max(32, crop), H_hr, W_hr)
            gacc = 0.0
            for b in range(B):
                top = torch.randint(0, H_hr - crop + 1, (1,), device=lr.device).item()
                left = torch.randint(0, W_hr - crop + 1, (1,), device=lr.device).item()
                sr_crop = _predict_sr_crop_from_residual(b, top, left, crop, feat_student)
                hr_crop = hr[b:b + 1, :, top:top + crop, left:left + crop]
                w_crop = 1.0 + float(getattr(self.cfg, "loss_alpha", 3.0)) * W_edge[b:b + 1, :, top:top + crop, left:left + crop]
                gacc = gacc + _image_grad_l1(sr_crop, hr_crop, weight=w_crop)
            loss_grad = gacc / B

        # ----------------------------
        # HoVer boundary-guided gradient (student)
        # ----------------------------
        lam_hg = float(getattr(self.cfg, "lambda_hover_grad", 0.2)) if use_hover else 0.0
        loss_hgrad = torch.tensor(0.0, device=lr.device)
        if lam_hg > 0:
            crop = int(getattr(self.cfg, "grad_crop", 128))
            crop = min(max(32, crop), H_hr, W_hr)
            hgacc = 0.0
            for b in range(B):
                top = torch.randint(0, H_hr - crop + 1, (1,), device=lr.device).item()
                left = torch.randint(0, W_hr - crop + 1, (1,), device=lr.device).item()
                sr_crop = _predict_sr_crop_from_residual(b, top, left, crop, feat_student)
                hr_crop = hr[b:b + 1, :, top:top + crop, left:left + crop]
                wb = W_bnd_soft[b:b + 1, :, top:top + crop, left:left + crop]
                w = 1.0 + float(getattr(self.cfg, "hover_bnd_weight", 2.0)) * wb
                hgacc = hgacc + _image_grad_l1(sr_crop, hr_crop, weight=w)
            loss_hgrad = hgacc / B

        # ----------------------------
        # HoVer interior gray consistency (student)
        # ----------------------------
        lam_in = float(getattr(self.cfg, "lambda_in_gray", 0.05)) if use_hover else 0.0
        loss_in = torch.tensor(0.0, device=lr.device)
        if lam_in > 0:
            crop_i = int(getattr(self.cfg, "in_gray_crop", 192))
            crop_i = min(max(64, crop_i), H_hr, W_hr)
            iacc = 0.0
            for b in range(B):
                top = torch.randint(0, H_hr - crop_i + 1, (1,), device=lr.device).item()
                left = torch.randint(0, W_hr - crop_i + 1, (1,), device=lr.device).item()
                sr_crop = _predict_sr_crop_from_residual(b, top, left, crop_i, feat_student)
                hr_crop = hr[b:b + 1, :, top:top + crop_i, left:left + crop_i]
                w_in = W_in_soft[b:b + 1, :, top:top + crop_i, left:left + crop_i]
                g_sr = _rgb_to_gray(sr_crop)
                g_hr = _rgb_to_gray(hr_crop)
                iacc = iacc + (w_in * (g_sr - g_hr).abs()).mean()
            loss_in = iacc / B

        # ----------------------------
        # Distillation: EMA teacher + residual high-pass (student <- teacher)
        #   - warmup/ramp (A2/C1 style) 用 epoch gating 防止过早把 student 锁死
        # ----------------------------
        lam_max = float(getattr(self.cfg, "lambda_cond_consistency", 0.0))
        warm = int(getattr(self.cfg, "distill_warmup_epochs", 10))
        ramp = int(getattr(self.cfg, "distill_ramp_epochs", 10))
        if self._epoch < warm:
            lam_c = 0.0
        else:
            if ramp <= 0:
                lam_c = lam_max
            else:
                t = (self._epoch - warm + 1) / float(ramp)
                lam_c = lam_max * float(max(0.0, min(1.0, t)))

        loss_cons = torch.tensor(0.0, device=lr.device)
        feat_teacher = None
        dbg_hp_s = 0.0
        dbg_hp_t = 0.0

        if lam_c > 0 and getattr(self, "_use_ema_teacher", False) and use_hover and (len(getattr(self, "_ema_shadow", {})) > 0) and (hover_bnd is not None or hover_mask is not None):
            crop_c = int(getattr(self.cfg, "cond_consistency_crop", 192))
            crop_c = min(max(64, crop_c), H_hr, W_hr)
            sigma_hp = float(getattr(self.cfg, "cond_consistency_hp_sigma", 1.5))

            def _swap_params_to_ema_cpu_shadow():
                backup = {}
                for n, p in self.named_parameters():
                    if (n in self._ema_shadow) and p.requires_grad:
                        backup[n] = p.detach().clone()
                        p.data.copy_(self._ema_shadow[n].to(device=p.device, dtype=p.dtype))
                return backup

            def _restore_params(backup):
                for n, p in self.named_parameters():
                    if (n in backup) and p.requires_grad:
                        p.data.copy_(backup[n].to(device=p.device, dtype=p.dtype))

            with torch.no_grad():
                backup = _swap_params_to_ema_cpu_shadow()
                try:
                    feat_teacher = self._encode_with_condition(lr, hover_bnd=hover_bnd, hover_mask=hover_mask)
                finally:
                    _restore_params(backup)

            cacc = 0.0
            for b in range(B):
                wb = W_bnd_soft[b, 0].reshape(-1)
                if float(wb.sum().detach().cpu()) > 1e-6:
                    prob = (wb + 1e-6) / (wb.sum() + 1e-6)
                    idx0 = int(torch.multinomial(prob, 1, replacement=True).item())
                    cy = idx0 // W_hr
                    cx = idx0 - cy * W_hr
                    top = int(max(0, min(H_hr - crop_c, cy - crop_c // 2)))
                    left = int(max(0, min(W_hr - crop_c, cx - crop_c // 2)))
                else:
                    top = torch.randint(0, H_hr - crop_c + 1, (1,), device=lr.device).item()
                    left = torch.randint(0, W_hr - crop_c + 1, (1,), device=lr.device).item()

                # student residual crop
                res_s = _predict_residual_crop(b, top, left, crop_c, feat_student)
                # teacher residual crop (no grad)
                with torch.no_grad():
                    res_t = _predict_residual_crop(b, top, left, crop_c, feat_teacher)

                hp_s = self._gray_highpass(res_s, sigma=sigma_hp)
                hp_t = self._gray_highpass(res_t, sigma=sigma_hp)

                dbg_hp_s += float(hp_s.abs().mean().detach().cpu())
                dbg_hp_t += float(hp_t.abs().mean().detach().cpu())

                w_crop = 1.0 + W_bnd_soft[b:b + 1, :, top:top + crop_c, left:left + crop_c]
                cacc = cacc + (w_crop * (hp_s - hp_t).abs()).mean()

            loss_cons = cacc / B
            dbg_hp_s /= max(B, 1)
            dbg_hp_t /= max(B, 1)

        # ----------------------------
        # FFT removed completely
        # ----------------------------
        loss_fft = torch.tensor(0.0, device=lr.device)
        lam_f = 0.0

        # ----------------------------
        # Kernel diagnostics (sample a few points)
        # ----------------------------
        dbg_kernel_entropy = 0.0
        dbg_kernel_center_w = 0.0
        dbg_kernel_wsum = 0.0
        dbg_kernel_wmax = 0.0
        dbg_kernel_wmin = 0.0

        try:
            diag_n = int(getattr(self.cfg, "diag_kernel_points", 256))
            diag_n = max(32, min(diag_n, H_hr * W_hr))

            # sample points uniformly for stability
            idx = torch.randint(0, H_hr * W_hr, (diag_n,), device=lr.device)
            y = torch.div(idx, W_hr, rounding_mode="floor")
            x = idx - y * W_hr
            hr_xy_int = torch.stack([x, y], dim=1).to(dtype=torch.float32, device=lr.device)  # [N,2]
            hr_xy_norm = _hr_int_to_hr_norm(hr_xy_int, H_hr, W_hr)

            _, lr_frac, lr_xy_norm_center = self._lr_continuous_from_hr_int(
                hr_xy_int=hr_xy_int, lr_hw=(H_lr, W_lr), hr_hw=(H_hr, W_hr)
            )

            w = self._predict_kernel_weights(
                feat_lr=feat_student[0:1],
                hr_xy_norm=hr_xy_norm,
                lr_xy_norm_center=lr_xy_norm_center,
                lr_frac=lr_frac,
            )  # expected [1,N,K2] but may NOT be a probability distribution
            

            w0 = w[0]  # [N,K2]

            # --- IMPORTANT: make it a valid probability distribution for diagnostics ---
            # 1) ensure non-negative
            w0 = w0.clamp_min(0.0)

            # 2) normalize so sum=1 (avoid cw>1 and invalid entropy)
            wsum = w0.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            wprob = w0 / wsum  # [N,K2], sum=1

            # entropy: max should be log(K2); for 7x7, log(49) ≈ 3.89
            ent = (-wprob * torch.log(wprob.clamp_min(1e-12))).sum(dim=-1).mean()

            center_idx = wprob.shape[-1] // 2
            cw = wprob[:, center_idx].mean()

            dbg_kernel_entropy = float(ent.detach().cpu())
            dbg_kernel_center_w = float(cw.detach().cpu())
            dbg_kernel_wsum = float(wprob.sum(dim=-1).mean().detach().cpu())  # should be ~1.0
            dbg_kernel_wmax = float(wprob.max(dim=-1).values.mean().detach().cpu())
            dbg_kernel_wmin = float(wprob.min(dim=-1).values.mean().detach().cpu())

        except Exception:
            pass


        if return_debug:
            # ----------------------------
    # Total loss (define BEFORE return)
    # ----------------------------
            loss = (
                loss_pix
                + lam_g * loss_grad
                + lam_hg * loss_hgrad
                + lam_in * loss_in
                + lam_f * loss_fft
                + lam_c * loss_cons
            )
            return loss, {
                "loss_pix": float(loss_pix.detach().cpu()),
                "loss_grad": float(loss_grad.detach().cpu()),
                "loss_hgrad": float(loss_hgrad.detach().cpu()),
                "loss_in": float(loss_in.detach().cpu()),
                "loss_fft": float(loss_fft.detach().cpu()),
                "loss_cons": float(loss_cons.detach().cpu()),
                "mean_edge": float(mean_edge),
                "tau": float(self._tau),
                "lambda_grad": float(lam_g),
                "lambda_hover_grad": float(lam_hg),
                "lambda_in_gray": float(lam_in),
                "lambda_fft": float(lam_f),
                "lambda_cond_consistency": float(lam_c),
                "hover_sample_ratio": float(r),
                # --- diagnostics ---
                "dbg_res_pred_abs": float(_acc_res_pred_abs / max(B, 1)),
                "dbg_res_tgt_abs": float(_acc_res_tgt_abs / max(B, 1)),
                "dbg_wmix_mean": float(_acc_wmix / max(B, 1)),
                "dbg_kernel_entropy": float(dbg_kernel_entropy),
                "dbg_kernel_center_w": float(dbg_kernel_center_w),
                "dbg_hp_stu": float(dbg_hp_s),
                "dbg_hp_tch": float(dbg_hp_t),
                "dbg_kernel_wsum": dbg_kernel_wsum,
                "dbg_kernel_wmax": dbg_kernel_wmax,
                "dbg_kernel_wmin": dbg_kernel_wmin,
            }
        return loss


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SRModelConfig()
    model = SRModel(cfg=cfg).to(device)

    lr = torch.rand(2, 3, 128, 128, device=device)
    hr = torch.rand(2, 3, 512, 512, device=device)

    sr = model(lr)
    print("sr:", sr.shape)

    loss, dbg = model.compute_loss(lr, hr, return_debug=True)
    print("loss:", float(loss), dbg)