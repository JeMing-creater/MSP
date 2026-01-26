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
# FFT high-pass loss
# =============================================================================
def _fft_highpass_mask(h: int, w: int, cutoff_ratio: float, device: torch.device) -> torch.Tensor:
    """
    Create a radial high-pass mask in frequency domain (fftshifted layout).
    cutoff_ratio: in (0,0.5). Frequencies with radius >= cutoff are kept.
    Returns mask [1,1,H,W] float.
    """
    cutoff_ratio = float(cutoff_ratio)
    cutoff_ratio = max(0.0, min(0.5, cutoff_ratio))

    yy = torch.arange(h, device=device).float()
    xx = torch.arange(w, device=device).float()
    Y, X = torch.meshgrid(yy, xx, indexing="ij")

    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    r = torch.sqrt((Y - cy) ** 2 + (X - cx) ** 2)

    # FIX: r_max should be float via math.sqrt (not torch.sqrt on floats)
    r_max = math.sqrt(cy * cy + cx * cx) + 1e-6
    r_norm = r / r_max

    mask = (r_norm >= cutoff_ratio).float()
    return mask.view(1, 1, h, w)


def _fftshift2(x: torch.Tensor) -> torch.Tensor:
    """2D fftshift for last two dims."""
    h, w = x.shape[-2], x.shape[-1]
    return torch.roll(torch.roll(x, shifts=h // 2, dims=-2), shifts=w // 2, dims=-1)


def fft_highpass_l1(
    sr: torch.Tensor,
    hr: torch.Tensor,
    cutoff_ratio: float = 0.15,
    use_logmag: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    SR/HR: [B,3,H,W] in [0,1]
    Compute L1 distance between high-frequency magnitude spectra (grayscale).
    """
    sr_g = _rgb_to_gray(sr).clamp(0, 1)
    hr_g = _rgb_to_gray(hr).clamp(0, 1)

    # remove DC bias: improves stability
    sr_g = sr_g - sr_g.mean(dim=(2, 3), keepdim=True)
    hr_g = hr_g - hr_g.mean(dim=(2, 3), keepdim=True)

    # fft2 -> complex
    F_sr = torch.fft.fft2(sr_g, dim=(-2, -1))
    F_hr = torch.fft.fft2(hr_g, dim=(-2, -1))

    # shift to center
    F_sr = _fftshift2(F_sr)
    F_hr = _fftshift2(F_hr)

    mag_sr = torch.abs(F_sr)
    mag_hr = torch.abs(F_hr)

    if use_logmag:
        mag_sr = torch.log1p(mag_sr + eps)
        mag_hr = torch.log1p(mag_hr + eps)

    B, _, H, W = mag_sr.shape
    mask = _fft_highpass_mask(H, W, cutoff_ratio=cutoff_ratio, device=mag_sr.device)  # [1,1,H,W]

    diff = (mag_sr - mag_hr).abs() * mask
    denom = mask.mean().clamp_min(1e-6)
    return diff.mean() / denom


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
    input = [local_feat(center), PE(hr_xy_norm), lr_frac(dx,dy)] -> kernel_logits (K*K)
    """
    def __init__(
        self,
        feat_dim: int,
        pe_dim: int,
        kernel_size: int = 7,
        hidden: int = 256,
        depth: int = 5,
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)
        out_dim = self.kernel_size * self.kernel_size
        in_dim = int(feat_dim + pe_dim + 2)

        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
    scale: int = 4

    # encoder & kernel predictor
    feat_ch: int = 64
    pe_bands: int = 10
    mlp_hidden: int = 256
    mlp_depth: int = 5

    # kernel prediction
    kernel_size: int = 7  # odd

    # training sampling
    num_points: int = 4096
    saliency_ratio: float = 0.7
    loss_alpha: float = 3.0  # pixel-weight strength using W_edge (and optionally HoVer band)

    # gradient consistency
    lambda_grad: float = 0.1
    grad_crop: int = 128

    # FFT high-pass consistency (Scheme 2) - default OFF
    lambda_fft: float = 0.0
    fft_crop: int = 256              # compute FFT on a random crop for speed
    fft_cutoff_ratio: float = 0.15   # higher => only very high frequencies
    fft_use_logmag: bool = True

    # inference chunking
    infer_chunk: int = 8192

    # kernel weight mode
    kernel_allow_negative: bool = True

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
    hover_mask_sigma: float = 1.5
    hover_bnd_weight: float = 2.0
    hover_in_weight: float = 0.5
    lambda_hover_grad: float = 0.2
    lambda_in_gray: float = 0.05

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
    hover_cond_dropout: float = 0.0

    # Consistency distillation: student (no hover) learns from teacher (with hover)
    lambda_cond_consistency: float = 0.1
    cond_consistency_crop: int = 192
    cond_consistency_hp_sigma: float = 1.5

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

        self.kernel_mlp = KernelMLP(
            feat_dim=self.cfg.feat_ch,
            pe_dim=pe_dim,
            kernel_size=self.cfg.kernel_size,
            hidden=self.cfg.mlp_hidden,
            depth=self.cfg.mlp_depth,
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

    # ----------------------------
    # temperature schedule
    # ----------------------------
    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        self._tau = float(self._compute_tau(self._epoch))

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
        tau = float(self._tau)
        logits = logits / max(tau, 1e-6)

        if not self.cfg.kernel_allow_negative:
            return torch.softmax(logits, dim=-1)

        w = torch.tanh(logits)  # no inplace
        K = w.shape[-1]
        center = K // 2
        oh = F.one_hot(torch.tensor(center, device=w.device), num_classes=K).to(w.dtype)
        w = w + oh.view(1, 1, K)  # add identity bias
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)
        return w

    def _predict_kernel_weights(
        self,
        feat_lr: torch.Tensor,
        hr_xy_norm: torch.Tensor,
        lr_xy_norm_center: torch.Tensor,
        lr_frac: torch.Tensor,
    ) -> torch.Tensor:
        B = feat_lr.shape[0]
        N = hr_xy_norm.shape[0]
        K = self.cfg.kernel_size * self.cfg.kernel_size

        grid = lr_xy_norm_center.view(1, 1, -1, 2).to(feat_lr.device).repeat(B, 1, 1, 1)
        cen = F.grid_sample(feat_lr, grid, mode="bilinear", align_corners=False)  # [B,C,1,N]
        cen = cen.squeeze(2).transpose(1, 2).contiguous()  # [B,N,C]

        pe = self.pe(hr_xy_norm.to(device=feat_lr.device, dtype=torch.float32))  # [N,pe_dim]
        pe = pe.unsqueeze(0).expand(B, -1, -1)
        frac = lr_frac.to(device=feat_lr.device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)

        inp = torch.cat([cen, pe, frac], dim=-1)  # [B,N,D]
        logits = self.kernel_mlp(inp.view(B * N, -1)).view(B, N, K)
        return self._kernel_weights_from_logits(logits)

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

        self._tau = float(self._compute_tau(self._epoch))

        # ----------------------------
        # Prepare maps (HR space)
        # ----------------------------
        lr_up = F.interpolate(lr, size=(H_hr, W_hr), mode="bicubic", align_corners=False).clamp(0, 1)

        # edge saliency from LR-up (always available)
        W_edge = self.compute_edge_saliency(lr, out_hw=(H_hr, W_hr))  # [B,1,H,W] in [0,1]

        def _blur01(m: torch.Tensor, sigma: float) -> torch.Tensor:
            # m: [B,1,H,W] float in [0,1]
            if sigma <= 0:
                return m
            k = int(round(sigma * 4 + 1))
            if k % 2 == 0:
                k += 1
            pad = k // 2
            mp = F.pad(m, (pad, pad, pad, pad), mode="replicate")
            return F.avg_pool2d(mp, kernel_size=k, stride=1)

        use_hover = bool(getattr(self.cfg, "use_hover", False))
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
            W_bnd_soft = _blur01(hb, float(self.cfg.hover_bnd_sigma))
            W_bnd_soft = _minmax_norm(W_bnd_soft)
        else:
            W_bnd_soft = torch.zeros((B, 1, H_hr, W_hr), device=lr.device, dtype=torch.float32)

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
            M_in_soft = _blur01(hm, float(self.cfg.hover_mask_sigma))
            M_in_soft = _minmax_norm(M_in_soft)
        else:
            M_in_soft = torch.zeros((B, 1, H_hr, W_hr), device=lr.device, dtype=torch.float32)

        # mix sampling weight: edge + hover boundary band
        r = float(getattr(self.cfg, "hover_sample_ratio", 0.0)) if use_hover else 0.0
        W_mix = (1.0 - r) * W_edge + r * W_bnd_soft
        W_mix = _minmax_norm(W_mix)

        # ----------------------------
        # Features: student vs teacher
        # ----------------------------
        feat_student = self.encoder(lr)  # validation path
        feat_teacher = None
        if bool(getattr(self.cfg, "use_hover_cond", False)) and ((hover_bnd is not None) or (hover_mask is not None)):
            feat_teacher = self._encode_with_condition(
                lr, hover_bnd=hover_bnd, hover_mask=hover_mask, apply_dropout=True
            )

        # ----------------------------
        # Pixel loss with saliency-guided sampling (student)
        # ----------------------------
        N = int(self.cfg.num_points)
        N_sal = int(round(N * float(self.cfg.saliency_ratio)))
        N_uni = max(0, N - N_sal)

        loss_pix = 0.0
        mean_edge = 0.0

        for b in range(B):
            w_flat = W_mix[b, 0].reshape(-1) + 1e-6
            prob = w_flat / w_flat.sum()

            idx_sal = torch.multinomial(prob, num_samples=N_sal, replacement=True) if N_sal > 0 else None
            idx_uni = torch.randint(0, H_hr * W_hr, (N_uni,), device=lr.device) if N_uni > 0 else None
            if idx_sal is None:
                idx = idx_uni
            elif idx_uni is None:
                idx = idx_sal
            else:
                idx = torch.cat([idx_sal, idx_uni], dim=0)

            y = torch.div(idx, W_hr, rounding_mode="floor")
            x = idx - y * W_hr
            hr_xy_int = torch.stack([x, y], dim=1).to(dtype=torch.float32, device=lr.device)

            # predict residual at points (student)
            res_pred_pts = self.predict_points_base(
                lr=lr[b:b + 1],
                hr_xy_int=hr_xy_int,
                hr_hw=(H_hr, W_hr),
                feat_lr=feat_student[b:b + 1],
            )[0].clamp(-1, 1)  # [N,3]

            lr_up_pts = _get_points_rgb(lr_up[b], idx)      # [N,3]
            hr_pts = _get_points_rgb(hr[b], idx)            # [N,3]
            res_tgt_pts = (hr_pts - lr_up_pts)              # [N,3]

            # weighting
            ws = _get_points_map(W_mix[b], idx)            # [N]
            mean_edge += float(ws.mean().detach().cpu())
            weight = 1.0 + float(self.cfg.loss_alpha) * ws

            l1 = (res_pred_pts - res_tgt_pts).abs().mean(dim=1)  # [N]
            loss_pix = loss_pix + (weight * l1).mean()

        loss_pix = loss_pix / B
        mean_edge = mean_edge / max(B, 1)

        # ----------------------------
        # Helper: predict SR crop given feat
        # ----------------------------
        
        def _predict_sr_crop(b: int, top: int, left: int, crop: int, feat_lr: torch.Tensor) -> torch.Tensor:
            """
            Predict SR crop using residual definition:
            residual = predict_points_base(..., lr_up=lr_up)
            sr = lr_up + residual
            """
            lr_up_crop = lr_up[b:b + 1, :, top:top + crop, left:left + crop]

            ys = torch.arange(top, top + crop, device=lr.device)
            xs = torch.arange(left, left + crop, device=lr.device)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            hr_xy_crop = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(torch.float32)

            # IMPORTANT: this now returns RESIDUAL points
            res_pts = self.predict_points_base(
                lr=lr[b:b + 1],
                hr_xy_int=hr_xy_crop,
                hr_hw=(H_hr, W_hr),
                feat_lr=feat_lr[b:b + 1],
                lr_up=lr_up[b:b + 1],   # consistent residual baseline
            )  # [1, crop*crop, 3]

            res_map = res_pts.transpose(1, 2).contiguous().view(1, 3, crop, crop)

            # base SR from residual
            sr_pred = (lr_up_crop + res_map).clamp(0, 1)

            # optional residual refiner (still outputs residual)
            if self.cfg.use_residual and (self.res_refiner is not None):
                res0 = res_map
                res = self.res_refiner(lr_up_crop, res0)
                sr_pred = (lr_up_crop + res).clamp(0, 1)

            # hard color-preserving fusion
            sr_crop = _fuse_keep_color(lr_up_crop, sr_pred)
            return sr_crop

        
        # ----------------------------
        # Gradient consistency (student)
        # ----------------------------
        lam_g = float(self.cfg.lambda_grad)
        crop_g = int(self.cfg.grad_crop)
        loss_grad = torch.tensor(0.0, device=lr.device)

        if lam_g > 0 and crop_g > 0:
            crop_g = min(crop_g, H_hr, W_hr)
            grad_acc = 0.0
            for b in range(B):
                top = torch.randint(0, H_hr - crop_g + 1, (1,), device=lr.device).item()
                left = torch.randint(0, W_hr - crop_g + 1, (1,), device=lr.device).item()

                sr_crop = _predict_sr_crop(b, top, left, crop_g, feat_student)
                hr_crop = hr[b:b + 1, :, top:top + crop_g, left:left + crop_g]
                w_crop = W_mix[b:b + 1, :, top:top + crop_g, left:left + crop_g]
                grad_acc = grad_acc + _image_grad_l1(sr_crop, hr_crop, weight=(1.0 + w_crop))
            loss_grad = grad_acc / B

        # ----------------------------
        # HoVer boundary grad consistency (student)
        # ----------------------------
        lam_hg = float(getattr(self.cfg, "lambda_hover_grad", 0.0)) if use_hover else 0.0
        loss_hgrad = torch.tensor(0.0, device=lr.device)

        if lam_hg > 0:
            crop_h = min(int(self.cfg.grad_crop), H_hr, W_hr)
            hacc = 0.0
            for b in range(B):
                # choose crop center biased to boundary band
                wb = W_bnd_soft[b, 0].reshape(-1)
                if float(wb.sum().detach().cpu()) > 1e-6:
                    prob = (wb + 1e-6) / (wb.sum() + 1e-6)
                    idx0 = int(torch.multinomial(prob, 1, replacement=True).item())
                    cy = idx0 // W_hr
                    cx = idx0 - cy * W_hr
                    top = int(max(0, min(H_hr - crop_h, cy - crop_h // 2)))
                    left = int(max(0, min(W_hr - crop_h, cx - crop_h // 2)))
                else:
                    top = torch.randint(0, H_hr - crop_h + 1, (1,), device=lr.device).item()
                    left = torch.randint(0, W_hr - crop_h + 1, (1,), device=lr.device).item()

                sr_crop = _predict_sr_crop(b, top, left, crop_h, feat_student)
                hr_crop = hr[b:b + 1, :, top:top + crop_h, left:left + crop_h]
                wb_crop = W_bnd_soft[b:b + 1, :, top:top + crop_h, left:left + crop_h]
                w = 1.0 + float(self.cfg.hover_bnd_weight) * wb_crop
                hacc = hacc + _image_grad_l1(sr_crop, hr_crop, weight=w)
            loss_hgrad = hacc / B

        # ----------------------------
        # HoVer interior gray consistency (student) to avoid hollow edges
        # ----------------------------
        lam_in = float(getattr(self.cfg, "lambda_in_gray", 0.0)) if use_hover else 0.0
        loss_in = torch.tensor(0.0, device=lr.device)

        if lam_in > 0:
            crop_i = min(int(self.cfg.grad_crop), H_hr, W_hr)
            iacc = 0.0
            for b in range(B):
                top = torch.randint(0, H_hr - crop_i + 1, (1,), device=lr.device).item()
                left = torch.randint(0, W_hr - crop_i + 1, (1,), device=lr.device).item()

                sr_crop = _predict_sr_crop(b, top, left, crop_i, feat_student)
                hr_crop = hr[b:b + 1, :, top:top + crop_i, left:left + crop_i]
                mi = M_in_soft[b:b + 1, :, top:top + crop_i, left:left + crop_i]
                sr_g = _rgb_to_gray(sr_crop)
                hr_g = _rgb_to_gray(hr_crop)
                w = 1.0 + float(self.cfg.hover_in_weight) * mi
                iacc = iacc + (w * (sr_g - hr_g).abs()).mean()
            loss_in = iacc / B

        # ----------------------------
        # FFT high-pass (optional, student)
        # ----------------------------
        lam_f = float(self.cfg.lambda_fft)
        crop_f = int(self.cfg.fft_crop)
        loss_fft = torch.tensor(0.0, device=lr.device)

        if lam_f > 0:
            crop_f = min(max(32, crop_f), H_hr, W_hr)
            fft_acc = 0.0
            for b in range(B):
                top = torch.randint(0, H_hr - crop_f + 1, (1,), device=lr.device).item()
                left = torch.randint(0, W_hr - crop_f + 1, (1,), device=lr.device).item()
                sr_crop = _predict_sr_crop(b, top, left, crop_f, feat_student)
                hr_crop = hr[b:b + 1, :, top:top + crop_f, left:left + crop_f]
                fft_acc = fft_acc + fft_highpass_l1(
                    sr=sr_crop,
                    hr=hr_crop,
                    cutoff_ratio=float(self.cfg.fft_cutoff_ratio),
                    use_logmag=bool(self.cfg.fft_use_logmag),
                )
            loss_fft = fft_acc / B

        # ----------------------------
        # Consistency distillation (student <- teacher), crop-biased to boundary
        # ----------------------------
        lam_c = float(getattr(self.cfg, "lambda_cond_consistency", 0.0))
        loss_cons = torch.tensor(0.0, device=lr.device)

        if lam_c > 0 and (feat_teacher is not None):
            crop_c = int(getattr(self.cfg, "cond_consistency_crop", 192))
            crop_c = min(max(64, crop_c), H_hr, W_hr)
            sigma_hp = float(getattr(self.cfg, "cond_consistency_hp_sigma", 1.5))
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

                sr_s = _predict_sr_crop(b, top, left, crop_c, feat_student)
                sr_t = _predict_sr_crop(b, top, left, crop_c, feat_teacher).detach()

                hp_s = self._gray_highpass(sr_s, sigma=sigma_hp)
                hp_t = self._gray_highpass(sr_t, sigma=sigma_hp)

                wb_crop = W_bnd_soft[b:b + 1, :, top:top + crop_c, left:left + crop_c]
                w = 1.0 + float(self.cfg.hover_bnd_weight) * wb_crop
                cacc = cacc + (w * (hp_s - hp_t).abs()).mean()
            loss_cons = cacc / B

        # total
        loss = loss_pix + lam_g * loss_grad + lam_hg * loss_hgrad + lam_in * loss_in + lam_f * loss_fft + lam_c * loss_cons

        if return_debug:
            return loss, {
                "loss_pix": float(loss_pix.detach().cpu()),
                "loss_grad": float(loss_grad.detach().cpu()),
                "loss_hgrad": float(loss_hgrad.detach().cpu()),
                "loss_in": float(loss_in.detach().cpu()),
                "loss_fft": float(loss_fft.detach().cpu()),
                "loss_cons": float(loss_cons.detach().cpu()),
                "mean_edge": float(mean_edge),
                "tau": float(self._tau),
                "lambda_grad": lam_g,
                "lambda_hover_grad": lam_hg,
                "lambda_in_gray": lam_in,
                "lambda_fft": lam_f,
                "lambda_cond_consistency": lam_c,
                "hover_sample_ratio": float(r),
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