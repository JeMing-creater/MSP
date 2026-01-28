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
@dataclass
class SRModelConfig:
    scale: int = 4

    # ----------------------------
    # Encoder & kernel predictor
    # ----------------------------
    feat_ch: int = 64
    pe_bands: int = 10
    mlp_hidden: int = 256
    mlp_depth: int = 5

    # ----------------------------
    # Kernel prediction
    # ----------------------------
    kernel_size: int = 7  # odd
    kernel_allow_negative: bool = False

    # ----------------------------
    # Training sampling
    # ----------------------------
    num_points: int = 4096
    saliency_ratio: float = 0.7
    loss_alpha: float = 3.0

    # ----------------------------
    # Gradient consistency
    # ----------------------------
    lambda_grad: float = 0.1
    grad_crop: int = 128

    # ----------------------------
    # Residual SR
    # ----------------------------
    use_residual: bool = True
    use_res_refiner: bool = True   # ★★★ 关键：补上这个

    # ----------------------------
    # Inference chunkin
    # ----------------------------
    infer_chunk: int = 8192

    # ----------------------------
    # Temperature annealing (kernel)
    # ----------------------------
    tau_start: float = 1.0
    tau_end: float = 0.5
    tau_warm_epochs: int = 2
    tau_anneal_epochs: int = 8

    # ----------------------------
    # HoVer-Net guidance (maps + losses)
    # ----------------------------
    use_hover: bool = True
    hover_sample_ratio: float = 0.5
    hover_bnd_sigma: float = 2.0
    hover_mask_sigma: float = 1.5
    hover_bnd_weight: float = 2.0
    hover_in_weight: float = 0.5
    lambda_hover_grad: float = 0.2
    lambda_in_gray: float = 0.05

    # ----------------------------
    # HoVer-Net kernel modulation (STRUCTURE)
    # ----------------------------
    use_hover_kernel_mod: bool = True
    hover_gate_bias: float = 1.0

    # ----------------------------
    # Teacher–Student / Distillation
    # ----------------------------
    lambda_cond_consistency: float = 0.1
    distill_warmup_epochs: int = 1
    distill_ramp_epochs: int = 5
    cond_consistency_crop: int = 192
    cond_consistency_hp_sigma: float = 1.5

    # ----------------------------
    # Diagnostics
    # ----------------------------
    diag_kernel_points: int = 256


# =============================================================================
# Main SR Model
# =============================================================================
class SRModel(nn.Module):
    def __init__(self, cfg: Optional[SRModelConfig] = None):
        super().__init__()
        self.cfg = cfg or SRModelConfig()
        self._last_gate_prob = None   # Tensor [B,N] in [0,1]
        self._last_gate_used = False    
        
        # ----------------------------
        # training states
        # ----------------------------
        self._epoch = 0
        self._tau = float(getattr(cfg, "tau_start", 1.0))

        # NEW: global step for step-based schedules (gate tau etc.)
        self._global_step = 0
        self._gate_tau = float(getattr(cfg, "kernel_gate_tau_start", getattr(cfg, "kernel_gate_tau", 1.0)))

        
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
        hover_bnd_hr: Optional[torch.Tensor] = None,
    ):
        """
        Dynamic kernel prediction with optional HoVer boundary modulation.
        Also caches gate statistics for logging (no effect on optimization).
        """
        B = feat_lr.shape[0]
        N = hr_xy_norm.shape[0]
        K2 = self.cfg.kernel_size ** 2
        H = self.kernel_heads

        # ---- sample LR feature ----
        grid = lr_xy_norm_center.view(1, 1, -1, 2).repeat(B, 1, 1, 1)
        cen = F.grid_sample(feat_lr, grid, mode="bilinear", align_corners=False)
        cen = cen.squeeze(2).transpose(1, 2)  # [B,N,C]

        pe = self.pe(hr_xy_norm).unsqueeze(0).expand(B, -1, -1)     # [B,N,pe]
        frac = lr_frac.unsqueeze(0).expand(B, -1, -1)               # [B,N,2]

        x = torch.cat([cen, pe, frac], dim=-1).contiguous()         # [B,N,dim]
        x = x.view(B * N, -1)

        klogits, glogits = self.kernel_mlp(x)
        klogits = klogits.view(B, N, H, K2)
        glogits = glogits.view(B, N, H)

        # ---- STRUCTURAL GATE MODULATION ----
        if self.training and hover_bnd_hr is not None:
            # hover_bnd_hr: [B,N]
            glogits = glogits + float(getattr(self.cfg, "hover_gate_bias", 1.0)) * hover_bnd_hr.unsqueeze(-1)

        # ---- gate ----
        gate_tau = float(getattr(self, "_gate_tau", 1.0))
        gate = torch.softmax(glogits / max(gate_tau, 1e-6), dim=-1)  # [B,N,H]

        # ---- cache gate for logging (NO grad, NO effect) ----
        # Store a light-weight detached copy; this is only for debug printing.
        self._last_gate_prob = gate.detach()
        self._last_gate_used = True

        # ---- kernel weights ----
        w_head = self._kernel_weights_from_logits(
            klogits.view(B, N * H, K2)
        ).view(B, N, H, K2)

        # mixture over heads
        w = (gate.unsqueeze(-1) * w_head).sum(dim=2)  # [B,N,K2]
        return w


    def _gather_lr_patch(self, lr: torch.Tensor, lr_xy_cont: torch.Tensor) -> torch.Tensor:
        """
        Gather LR KxK neighborhood (flattened) for each HR-point-mapped LR coordinate.

        Args:
            lr:        [B, C, H_lr, W_lr]
            lr_xy_cont:[N, 2] float, (x_lr, y_lr) in LR coordinate system
                    (typically produced by _lr_continuous_from_hr_int)

        Returns:
            patch: [B, N, K*K*C] if C>1, else [B, N, K*K]
                In this project we usually apply kernels per-channel (RGB),
                so the common usage is to gather per-channel and then apply
                kernel to each channel separately in predict_points_base.
        """
        B, C, H, W = lr.shape
        K = int(self.cfg.kernel_size)
        r = K // 2
        device = lr.device

        # base integer anchor at floor(x), floor(y)
        x0 = torch.floor(lr_xy_cont[:, 0]).long()  # [N]
        y0 = torch.floor(lr_xy_cont[:, 1]).long()  # [N]

        # offsets for KxK
        offs = torch.arange(-r, r + 1, device=device, dtype=torch.long)
        oy, ox = torch.meshgrid(offs, offs, indexing="ij")  # [K,K]
        ox = ox.reshape(-1)  # [K2]
        oy = oy.reshape(-1)  # [K2]
        K2 = ox.numel()

        # coords for each point and each offset
        xx = x0[:, None] + ox[None, :]  # [N,K2]
        yy = y0[:, None] + oy[None, :]  # [N,K2]

        # clamp to valid range (replicate-border behavior)
        xx = xx.clamp(0, W - 1)
        yy = yy.clamp(0, H - 1)

        # flatten index for gather: idx = yy*W + xx
        idx = (yy * W + xx).view(-1)  # [N*K2]

        # gather for each batch, each channel
        lr_flat = lr.view(B, C, H * W)              # [B,C,HW]
        gathered = lr_flat[:, :, idx]               # [B,C,N*K2]
        gathered = gathered.view(B, C, -1, K2)      # [B,C,N,K2]
        gathered = gathered.permute(0, 2, 3, 1)     # [B,N,K2,C]

        # If you want per-channel kernel application later, keep C as last dim.
        # Many kernels are shared across channels; you can reshape accordingly.
        return gathered  # [B,N,K2,C]

    
    # ----------------------------
    # point prediction
    # ----------------------------
    def predict_points_base(
        self,
        lr: torch.Tensor,
        hr_xy_int: torch.Tensor,
        hr_hw: Tuple[int, int],
        feat_lr: torch.Tensor,
        lr_up: torch.Tensor,
        hover_bnd: Optional[torch.Tensor] = None,
    ):
        """
        Predict SR residual (RGB) at given HR integer points.

        Returns:
            res: [B, N, 3]  (RGB residual at points)
        """
        B, _, H_lr, W_lr = lr.shape
        H_hr, W_hr = hr_hw
        device = lr.device

        # ---- HR int → HR norm ----
        hr_xy_norm = _hr_int_to_hr_norm(hr_xy_int, H_hr, W_hr)  # [N,2]

        # ---- HR → LR continuous mapping ----
        lr_xy_cont, lr_frac, lr_xy_norm_center = self._lr_continuous_from_hr_int(
            hr_xy_int=hr_xy_int,
            lr_hw=(H_lr, W_lr),
            hr_hw=(H_hr, W_hr),
        )  # lr_xy_cont:[N,2], lr_frac:[N,2], lr_xy_norm_center:[N,2]

        # ---- sample lr_up at queried HR integer points (for residual definition) ----
        xh = hr_xy_int[:, 0].long().clamp(0, W_hr - 1)
        yh = hr_xy_int[:, 1].long().clamp(0, H_hr - 1)
        # lr_up_pts: [B,3,N] -> [B,N,3]
        lr_up_pts = lr_up[:, :, yh, xh].permute(0, 2, 1).contiguous()

        # ---- HoVer boundary sampling at queried points (TRAIN ONLY) ----
        hover_bnd_pts = None
        if (
            self.training
            and bool(getattr(self.cfg, "use_hover_kernel_mod", False))
            and (hover_bnd is not None)
        ):
            hb = hover_bnd
            if hb.ndim == 3:
                hb = hb.unsqueeze(1)
            hb = hb[:, :1].to(device=device, dtype=torch.float32)
            if hb.max() > 1.5:
                hb = hb / 255.0
            hb = (hb > 0.5).float()
            if hb.shape[-2:] != (H_hr, W_hr):
                hb = F.interpolate(hb, size=(H_hr, W_hr), mode="nearest")

            hover_bnd_pts = hb[:, 0, yh, xh]  # [B,N]

        # ---- kernel prediction ----
        # NOTE: 如果你的 _predict_kernel_weights 不支持 hover_bnd_hr 参数，这行会报错。
        # 那你就把 hover_bnd_hr=... 这一项删掉即可（先让 residual 正确）
        w = self._predict_kernel_weights(
            feat_lr=feat_lr,
            hr_xy_norm=hr_xy_norm,
            lr_xy_norm_center=lr_xy_norm_center,
            lr_frac=lr_frac,
            hover_bnd_hr=hover_bnd_pts,   # [B,N] or None
        )  # [B,N,K2]

        # ---- gather LR patches & apply kernel (得到的是“SR-like intensity”) ----
        patch = self._gather_lr_patch(lr, lr_xy_cont)          # [B,N,K2,3]
        sr_like = (patch * w.unsqueeze(-1)).sum(dim=2)         # [B,N,3]

        # ---- convert to residual ----
        res = sr_like - lr_up_pts                              # [B,N,3]
        return res

    
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
        Inference must NOT require HoVer priors, so hover_bnd is forced to None.
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
        hr_xy_int_full = _make_hr_int_grid(H_hr, W_hr, device=lr.device, dtype=torch.float32)  # [N,2]
        N = hr_xy_int_full.shape[0]
        chunk = int(self.cfg.infer_chunk)

        # predict residual in chunks
        outs = []
        for st in range(0, N, chunk):
            ed = min(N, st + chunk)
            res_pts = self.predict_points_base(
                lr=lr,
                feat_lr=feat_lr,
                hr_xy_int=hr_xy_int_full[st:ed],
                hr_hw=(H_hr, W_hr),
                lr_up=lr_up,
                hover_bnd=None,  # IMPORTANT: inference禁止先验
            )  # [B,n,3]
            outs.append(res_pts)

        # [B,N,3] -> [B,3,H,W]
        res_all = torch.cat(outs, dim=1)  # [B,N,3]
        res_map = res_all.view(B, H_hr, W_hr, 3).permute(0, 3, 1, 2).contiguous()  # [B,3,H,W]

        # optional residual refiner
        if bool(getattr(self.cfg, "use_residual", True)):
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
        """
        Training loss for residual SR with optional HoVer-Net priors (TRAIN ONLY).
        - Student path is ALWAYS unconditional (no priors).
        - Teacher/EMA path may use priors for distillation.
        - During training, supervision is applied on sr_pred = (lr_up + residual) (NO keep-color fuse).
        """
        B, _, H_lr, W_lr = lr.shape
        Bh, _, H_hr, W_hr = hr.shape
        if B != Bh:
            raise ValueError("lr/hr batch size mismatch")

        device = lr.device

        # ----------------------------
        # global step (preferred) + epoch tau
        # ----------------------------
        if not hasattr(self, "_global_step"):
            self._global_step = 0
        if self.training:
            self._global_step += 1

        # tau schedule (epoch-based kernel temperature)
        if hasattr(self, "_compute_tau") and hasattr(self, "_epoch"):
            self._tau = float(self._compute_tau(self._epoch))
        else:
            self._tau = float(getattr(self, "_tau", 1.0))

        # gate tau schedule (if configured)
        tau_s = float(getattr(self.cfg, "kernel_gate_tau_start", 1.0))
        tau_e = float(getattr(self.cfg, "kernel_gate_tau_end", 1.0))
        warm = int(getattr(self.cfg, "kernel_gate_tau_warm_steps", 0))
        anneal = int(getattr(self.cfg, "kernel_gate_tau_anneal_steps", 0))
        if warm > 0 and self._global_step < warm:
            self._gate_tau = tau_s
        elif anneal > 0:
            t = (self._global_step - warm) / float(max(1, anneal))
            t = max(0.0, min(1.0, t))
            self._gate_tau = tau_s + (tau_e - tau_s) * t
        else:
            self._gate_tau = tau_e

        # ---------------------------------------------------
        # EMA update (CPU shadow) - SAFE across devices
        # ---------------------------------------------------
        if self.training and getattr(self, "_use_ema_teacher", False) and len(getattr(self, "_ema_shadow", {})) > 0:
            self._ema_step = int(getattr(self, "_ema_step", 0)) + 1
            self._ema_update_every = int(getattr(self, "_ema_update_every", 1))
            self._ema_decay = float(getattr(self, "_ema_decay", 0.999))
            if (self._ema_step % max(1, self._ema_update_every)) == 0:
                d = float(self._ema_decay)
                with torch.no_grad():
                    for n, p in self.named_parameters():
                        if (n in self._ema_shadow) and p.requires_grad:
                            cur_cpu = p.detach().to(device="cpu", dtype=torch.float32)
                            self._ema_shadow[n].mul_(d).add_(cur_cpu, alpha=(1.0 - d))

        # ----------------------------
        # Prepare maps (HR space)
        # ----------------------------
        lr_up = F.interpolate(lr, size=(H_hr, W_hr), mode="bicubic", align_corners=False).clamp(0, 1)
        W_edge = self.compute_edge_saliency(lr, out_hw=(H_hr, W_hr))  # [B,1,H,W]

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

        # boundary soft weight
        if use_hover and (hover_bnd is not None):
            hb = hover_bnd
            if hb.ndim == 3:
                hb = hb.unsqueeze(1)
            hb = hb[:, :1].to(device=device, dtype=torch.float32)
            if hb.max() > 1.5:
                hb = hb / 255.0
            hb = (hb > 0.5).float()
            if hb.shape[-2:] != (H_hr, W_hr):
                hb = F.interpolate(hb, size=(H_hr, W_hr), mode="nearest")
            W_bnd_soft = _blur01(hb, float(getattr(self.cfg, "hover_bnd_sigma", 1.0)))
            W_bnd_soft = _minmax_norm(W_bnd_soft)
        else:
            W_bnd_soft = torch.zeros((B, 1, H_hr, W_hr), device=device, dtype=torch.float32)

        # interior soft weight
        if use_hover and (hover_mask is not None):
            hm = hover_mask
            if hm.ndim == 3:
                hm = hm.unsqueeze(1)
            hm = hm[:, :1].to(device=device, dtype=torch.float32)
            if hm.max() > 1.5:
                hm = hm / 255.0
            hm = (hm > 0.5).float()
            if hm.shape[-2:] != (H_hr, W_hr):
                hm = F.interpolate(hm, size=(H_hr, W_hr), mode="nearest")
            W_in_soft = _blur01(hm, float(getattr(self.cfg, "hover_in_sigma", 1.0)))
            W_in_soft = _minmax_norm(W_in_soft)
        else:
            W_in_soft = torch.zeros((B, 1, H_hr, W_hr), device=device, dtype=torch.float32)

        # sampling mixture
        W_mix = _minmax_norm(W_edge + float(getattr(self.cfg, "hover_sample_ratio", 0.4)) * W_bnd_soft)

        # student feature (unconditional)
        feat_student = self.encoder(lr)

        # ----------------------------
        # Helper: safe predict_points_base call
        # ----------------------------
        _ppb_vars = set(getattr(self.predict_points_base, "__code__", None).co_varnames) if hasattr(self, "predict_points_base") else set()

        def _ppb(
            b: int,
            hr_xy_int: torch.Tensor,
            feat_lr: torch.Tensor,
            hover_bnd_call: Optional[torch.Tensor],
        ) -> torch.Tensor:
            kwargs = dict(
                lr=lr[b:b + 1],
                hr_xy_int=hr_xy_int,
                hr_hw=(H_hr, W_hr),
                feat_lr=feat_lr[b:b + 1],
                lr_up=lr_up[b:b + 1],
            )
            if "hover_bnd" in _ppb_vars:
                kwargs["hover_bnd"] = hover_bnd_call
            return self.predict_points_base(**kwargs)

        # ----------------------------
        # Pixel residual loss (sampled points)
        # ----------------------------
        N = int(getattr(self.cfg, "num_points", 4096))
        N_sal = int(round(N * float(getattr(self.cfg, "saliency_ratio", 0.7))))
        N_uni = max(0, N - N_sal)

        loss_pix = torch.tensor(0.0, device=device)
        mean_edge = 0.0
        dbg_res_pred = 0.0
        dbg_res_tgt = 0.0

        for b in range(B):
            w_flat = W_mix[b, 0].reshape(-1) + 1e-6
            prob = w_flat / w_flat.sum()
            idx_sal = torch.multinomial(prob, num_samples=N_sal, replacement=True) if N_sal > 0 else None
            idx_uni = torch.randint(0, H_hr * W_hr, (N_uni,), device=device) if N_uni > 0 else None
            idx = idx_uni if idx_sal is None else (idx_sal if idx_uni is None else torch.cat([idx_sal, idx_uni], dim=0))

            y = torch.div(idx, W_hr, rounding_mode="floor")
            x = idx - y * W_hr
            hr_xy_int = torch.stack([x, y], dim=1).to(dtype=torch.float32, device=device)

            # student residual at points (unconditional)
            res_pred_pts = _ppb(b, hr_xy_int, feat_student, hover_bnd_call=None)[0]  # [N,3]

            lr_up_pts = _get_points_rgb(lr_up[b], idx)  # [N,3]
            hr_pts = _get_points_rgb(hr[b], idx)        # [N,3]
            res_tgt_pts = hr_pts - lr_up_pts            # [N,3]

            ws = _get_points_map(W_mix[b], idx)         # [N]
            mean_edge += float(ws.mean().detach().cpu())

            alpha = float(getattr(self.cfg, "loss_alpha", 2.0))
            weight = 1.0 + alpha * ws

            l1 = (res_pred_pts - res_tgt_pts).abs().mean(dim=1)  # [N]
            loss_pix = loss_pix + (weight * l1).mean()

            dbg_res_pred += float(res_pred_pts.abs().mean().detach().cpu())
            dbg_res_tgt += float(res_tgt_pts.abs().mean().detach().cpu())

        loss_pix = loss_pix / B
        mean_edge = mean_edge / max(B, 1)
        dbg_res_pred = dbg_res_pred / max(B, 1)
        dbg_res_tgt = dbg_res_tgt / max(B, 1)

        # ----------------------------
        # Helpers: residual & SR crop (TRAIN: NO keep-color fuse)
        # ----------------------------
        def _predict_residual_crop(b: int, top: int, left: int, crop: int, feat_lr: torch.Tensor, hover_bnd_call: Optional[torch.Tensor]) -> torch.Tensor:
            ys = torch.arange(top, top + crop, device=device)
            xs = torch.arange(left, left + crop, device=device)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            hr_xy_crop = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(torch.float32)

            res_pts = _ppb(b, hr_xy_crop, feat_lr, hover_bnd_call=hover_bnd_call)
            return res_pts.transpose(1, 2).contiguous().view(1, 3, crop, crop)

        def _predict_sr_crop_from_residual(b: int, top: int, left: int, crop: int, feat_lr: torch.Tensor, hover_bnd_call: Optional[torch.Tensor]) -> torch.Tensor:
            lr_up_crop = lr_up[b:b + 1, :, top:top + crop, left:left + crop]
            res_map = _predict_residual_crop(b, top, left, crop, feat_lr, hover_bnd_call)

            sr_pred = (lr_up_crop + res_map).clamp(0, 1)

            if bool(getattr(self.cfg, "use_residual", True)) and (self.res_refiner is not None):
                res = self.res_refiner(lr_up_crop, res_map)
                sr_pred = (lr_up_crop + res).clamp(0, 1)

            if self.training:
                return sr_pred
            return _fuse_keep_color(lr_up_crop, sr_pred)

        # ----------------------------
        # Gradient loss (student↔HR)
        # ----------------------------
        lam_g = float(getattr(self.cfg, "lambda_grad", 0.2))
        crop_g = int(getattr(self.cfg, "grad_crop", 192))
        loss_grad = torch.tensor(0.0, device=device)

        if lam_g > 0 and crop_g > 0:
            crop_g = min(max(64, crop_g), H_hr, W_hr)
            gacc = 0.0
            for b in range(B):
                top = torch.randint(0, H_hr - crop_g + 1, (1,), device=device).item()
                left = torch.randint(0, W_hr - crop_g + 1, (1,), device=device).item()
                sr_crop = _predict_sr_crop_from_residual(b, top, left, crop_g, feat_student, hover_bnd_call=None)
                hr_crop = hr[b:b + 1, :, top:top + crop_g, left:left + crop_g]
                w_crop = 1.0 + W_edge[b:b + 1, :, top:top + crop_g, left:left + crop_g]
                gacc = gacc + _image_grad_l1(sr_crop, hr_crop, weight=w_crop)
            loss_grad = gacc / B

        # ----------------------------
        # HoVer boundary gradient constraint (student)
        # ----------------------------
        lam_hg = float(getattr(self.cfg, "lambda_hover_grad", 0.15))
        loss_hgrad = torch.tensor(0.0, device=device)

        if use_hover and lam_hg > 0 and (hover_bnd is not None):
            crop_h = int(getattr(self.cfg, "hover_grad_crop", 192))
            crop_h = min(max(64, crop_h), H_hr, W_hr)
            hacc = 0.0
            for b in range(B):
                wb = W_bnd_soft[b, 0].reshape(-1)
                if float(wb.sum().detach().cpu()) > 1e-6:
                    prob = (wb + 1e-6) / (wb.sum() + 1e-6)
                    idx0 = torch.multinomial(prob, num_samples=1, replacement=True)[0].item()
                    cy = idx0 // W_hr
                    cx = idx0 - cy * W_hr
                    top = int(max(0, min(H_hr - crop_h, cy - crop_h // 2)))
                    left = int(max(0, min(W_hr - crop_h, cx - crop_h // 2)))
                else:
                    top = torch.randint(0, H_hr - crop_h + 1, (1,), device=device).item()
                    left = torch.randint(0, W_hr - crop_h + 1, (1,), device=device).item()

                sr_crop = _predict_sr_crop_from_residual(b, top, left, crop_h, feat_student, hover_bnd_call=None)
                hr_crop = hr[b:b + 1, :, top:top + crop_h, left:left + crop_h]
                w_crop = 1.0 + W_bnd_soft[b:b + 1, :, top:top + crop_h, left:left + crop_h]
                hacc = hacc + _image_grad_l1(sr_crop, hr_crop, weight=w_crop)
            loss_hgrad = hacc / B

        # ----------------------------
        # HoVer interior gray consistency (student)
        # ----------------------------
        lam_in = float(getattr(self.cfg, "lambda_in_gray", 0.05))
        loss_in = torch.tensor(0.0, device=device)

        if use_hover and lam_in > 0 and (hover_mask is not None):
            crop_i = int(getattr(self.cfg, "in_gray_crop", 192))
            crop_i = min(max(64, crop_i), H_hr, W_hr)
            iacc = 0.0
            for b in range(B):
                top = torch.randint(0, H_hr - crop_i + 1, (1,), device=device).item()
                left = torch.randint(0, W_hr - crop_i + 1, (1,), device=device).item()
                sr_crop = _predict_sr_crop_from_residual(b, top, left, crop_i, feat_student, hover_bnd_call=None)
                hr_crop = hr[b:b + 1, :, top:top + crop_i, left:left + crop_i]
                w_crop = W_in_soft[b:b + 1, :, top:top + crop_i, left:left + crop_i]
                g_sr = _rgb_to_gray(sr_crop)
                g_hr = _rgb_to_gray(hr_crop)
                iacc = iacc + (w_crop * (g_sr - g_hr).abs()).mean()
            loss_in = iacc / B

        # ----------------------------
        # Distillation: EMA teacher + residual high-pass
        # ----------------------------
        lam_max = float(getattr(self.cfg, "lambda_cond_consistency", 0.0))
        warm_epochs = int(getattr(self.cfg, "distill_warmup_epochs", 10))
        ramp_epochs = int(getattr(self.cfg, "distill_ramp_epochs", 10))

        if not hasattr(self, "_epoch"):
            self._epoch = 0

        if self._epoch < warm_epochs:
            lam_c = 0.0
        else:
            if ramp_epochs <= 0:
                lam_c = lam_max
            else:
                t = (self._epoch - warm_epochs + 1) / float(ramp_epochs)
                lam_c = lam_max * float(max(0.0, min(1.0, t)))

        loss_cons = torch.tensor(0.0, device=device)
        hp_s_dbg = 0.0
        hp_t_dbg = 0.0

        if (
            lam_c > 0
            and getattr(self, "_use_ema_teacher", False)
            and use_hover
            and (len(getattr(self, "_ema_shadow", {})) > 0)
            and (hover_bnd is not None or hover_mask is not None)
        ):
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
                    idx0 = torch.multinomial(prob, num_samples=1, replacement=True)[0].item()
                    cy = idx0 // W_hr
                    cx = idx0 - cy * W_hr
                    top = int(max(0, min(H_hr - crop_c, cy - crop_c // 2)))
                    left = int(max(0, min(W_hr - crop_c, cx - crop_c // 2)))
                else:
                    top = torch.randint(0, H_hr - crop_c + 1, (1,), device=device).item()
                    left = torch.randint(0, W_hr - crop_c + 1, (1,), device=device).item()

                res_s = _predict_residual_crop(b, top, left, crop_c, feat_student, hover_bnd_call=None)

                with torch.no_grad():
                    res_t = _predict_residual_crop(b, top, left, crop_c, feat_teacher, hover_bnd_call=W_bnd_soft[b:b + 1])

                hp_s = self._gray_highpass(res_s, sigma=sigma_hp)
                hp_t = self._gray_highpass(res_t, sigma=sigma_hp)

                hp_s_dbg += float(hp_s.abs().mean().detach().cpu())
                hp_t_dbg += float(hp_t.abs().mean().detach().cpu())

                w_crop = 1.0 + W_bnd_soft[b:b + 1, :, top:top + crop_c, left:left + crop_c]
                cacc = cacc + (w_crop * (hp_s - hp_t).abs()).mean()

            loss_cons = cacc / B
            hp_s_dbg = hp_s_dbg / max(B, 1)
            hp_t_dbg = hp_t_dbg / max(B, 1)

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
            idx = torch.randint(0, H_hr * W_hr, (diag_n,), device=device)
            y = torch.div(idx, W_hr, rounding_mode="floor")
            x = idx - y * W_hr
            hr_xy_int = torch.stack([x, y], dim=1).to(dtype=torch.float32, device=device)
            hr_xy_norm = _hr_int_to_hr_norm(hr_xy_int, H_hr, W_hr)
            lr_xy_cont, lr_frac, lr_xy_norm_center = self._lr_continuous_from_hr_int(
                hr_xy_int=hr_xy_int, lr_hw=(H_lr, W_lr), hr_hw=(H_hr, W_hr)
            )
            w = self._predict_kernel_weights(
                feat_lr=feat_student[0:1],
                hr_xy_norm=hr_xy_norm,
                lr_xy_norm_center=lr_xy_norm_center,
                lr_frac=lr_frac,
            )  # [1,N,K]
            w0 = w[0]
            wsum = w0.sum(dim=-1).mean()
            wmax = w0.max(dim=-1).values.mean()
            wmin = w0.min(dim=-1).values.mean()
            wpos = w0.clamp_min(1e-12)
            ent = (-wpos * torch.log(wpos)).sum(dim=-1).mean()
            center_idx = (w0.shape[-1] // 2)
            cw = w0[:, center_idx].mean()
            dbg_kernel_entropy = float(ent.detach().cpu())
            dbg_kernel_center_w = float(cw.detach().cpu())
            dbg_kernel_wsum = float(wsum.detach().cpu())
            dbg_kernel_wmax = float(wmax.detach().cpu())
            dbg_kernel_wmin = float(wmin.detach().cpu())
        except Exception:
            pass

        # ----------------------------
        # Total loss
        # ----------------------------
        loss = loss_pix + lam_g * loss_grad + lam_hg * loss_hgrad + lam_in * loss_in + lam_c * loss_cons

        # ----------------------------
        # Gate diagnostics (for logging only)
        # ----------------------------
        dbg_gate_entropy = None
        dbg_gate_max = None
        try:
            if getattr(self, "_last_gate_used", False) and (getattr(self, "_last_gate_prob", None) is not None):
                g = self._last_gate_prob  # [B,N,H]
                gpos = g.clamp_min(1e-12)
                # entropy per point: -sum_h p log p
                ent = (-(gpos * gpos.log()).sum(dim=-1)).mean()  # scalar
                gmax = g.max(dim=-1).values.mean()               # scalar
                dbg_gate_entropy = float(ent.detach().cpu())
                dbg_gate_max = float(gmax.detach().cpu())
        except Exception:
            dbg_gate_entropy = None
            dbg_gate_max = None

        if return_debug:
            dbg = {
                "loss_pix": float(loss_pix.detach().cpu()),
                "loss_grad": float(loss_grad.detach().cpu()),
                "loss_hgrad": float(loss_hgrad.detach().cpu()),
                "loss_in": float(loss_in.detach().cpu()),
                "loss_cons": float(loss_cons.detach().cpu()),
                "mean_edge": float(mean_edge),
                "tau": float(getattr(self, "_tau", 1.0)),
                "gate_tau": float(getattr(self, "_gate_tau", 1.0)),
                "global_step": int(getattr(self, "_global_step", 0)),
                "lambda_grad": float(lam_g),
                "lambda_hover_grad": float(lam_hg),
                "lambda_in_gray": float(lam_in),
                "lambda_cond_consistency": float(lam_c),
                "distill_lambda_max": float(lam_max),
                "distill_warmup_epochs": int(warm_epochs),
                "distill_ramp_epochs": int(ramp_epochs),
                "res_pred_abs": float(dbg_res_pred),
                "res_tgt_abs": float(dbg_res_tgt),
                "Wmix_mean": float(W_mix.mean().detach().cpu()),
                "ker_entropy": float(dbg_kernel_entropy),
                "ker_center_w": float(dbg_kernel_center_w),
                "ker_wsum": float(dbg_kernel_wsum),
                "ker_wmax": float(dbg_kernel_wmax),
                "ker_wmin": float(dbg_kernel_wmin),
                "hp_s": float(hp_s_dbg),
                "hp_t": float(hp_t_dbg),

                # ---- NEW: gate stats for train.py logging ----
                "gate_entropy": dbg_gate_entropy,
                "gate_max": dbg_gate_max,
            }
            return loss, dbg

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