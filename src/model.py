# model.py
# -*- coding: utf-8 -*-
"""
Pathology Super-Resolution (SR) core model.

New training scheme (Scheme 1):
- Remove teacher/distillation entirely.
- Replace sparse point residual supervision with **dense HR-crop reconstruction**.
- Introduce HoVer-Net boundary maps ONLY as TRAIN-TIME loss weights (no inference dependency):
    - boundary-weighted high-pass consistency between SR and HR (cell-level detail guidance)

External interface:
  - model(lr) -> sr
  - model.compute_loss(lr, hr, hover_bnd=None, return_debug=False) -> loss (and debug dict)
  - model.set_epoch(epoch)

Inference/validation:
  - No HoVer inputs are required or used. Just call model(lr).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Small utilities
# =============================================================================
def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """x: [B,3,H,W] -> [B,1,H,W]"""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def _minmax_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample min-max normalize to [0,1]. x: [B,1,H,W]."""
    B = x.shape[0]
    v = x.view(B, -1)
    mn = v.min(dim=1)[0].view(B, 1, 1, 1)
    mx = v.max(dim=1)[0].view(B, 1, 1, 1)
    return (x - mn) / (mx - mn + eps)


def _fuse_keep_color(lr_up: torch.Tensor, sr_pred: torch.Tensor) -> torch.Tensor:
    """
    Color-preserving fusion for inference:
      sr_final = lr_up + (gray(sr_pred) - gray(lr_up)) broadcast to RGB.
    """
    lr_g = _rgb_to_gray(lr_up)
    sr_g = _rgb_to_gray(sr_pred)
    delta = sr_g - lr_g
    return (lr_up + delta.repeat(1, 3, 1, 1)).clamp(0, 1)


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


# =============================================================================
# Positional Encoding
# =============================================================================
class FourierPositionalEncoding(nn.Module):
    def __init__(self, num_bands: int = 10):
        super().__init__()
        self.num_bands = int(num_bands)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        xy: [N,2] normalized [-1,1]
        returns: [N, 4*num_bands]
        """
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

    input  = [local_feat(center), PE(hr_xy_norm), lr_frac(dx,dy)]
    output:
      - kernel_logits: [*, heads, K*K]
      - gate_logits:   [*, heads]
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
        for _ in range(max(0, depth - 1)):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        self.backbone = nn.Sequential(*layers)

        self.head_kernel = nn.Linear(hidden, self.num_heads * K2)
        self.head_gate = nn.Linear(hidden, self.num_heads)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        klogits = self.head_kernel(h)
        glogits = self.head_gate(h)
        return klogits, glogits


# =============================================================================
# Optional residual refiner (color-safe groups=3)
# =============================================================================
class ResidualRefiner(nn.Module):
    """
    Color-safe refiner (no cross-channel mixing): groups=3.
    input: [lr_up_rgb, res0_rgb] -> output residual RGB
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
        return self.net(torch.cat([lr_up, res0], dim=1))


# =============================================================================
# Config (minimal)
# =============================================================================
@dataclass
class SRModelConfig:
    # scale factor (LR->HR)
    scale: int = 4

    # encoder & predictor
    feat_ch: int = 64
    pe_bands: int = 10
    mlp_hidden: int = 256
    mlp_depth: int = 5
    kernel_heads: int = 4

    # kernel
    kernel_size: int = 7  # odd
    kernel_allow_negative: bool = False
    kernel_logit_norm: bool = True

    # residual refine
    use_residual: bool = True
    use_res_refiner: bool = True

    # inference
    infer_chunk: int = 8192

    # kernel temperature schedule (epoch)
    tau_start: float = 1.0
    tau_end: float = 0.5
    tau_warm_epochs: int = 2
    tau_anneal_epochs: int = 8

    # gate temperature schedule (step)
    kernel_gate_tau_start: float = 1.0
    kernel_gate_tau_end: float = 1.0
    kernel_gate_tau_warm_steps: int = 0
    kernel_gate_tau_anneal_steps: int = 0

    # TRAIN scheme 1: dense crop supervision
    train_crop: int = 192
    train_crops_per_img: int = 1
    train_crop_chunk: int = 8192

    lambda_crop_l1: float = 1.0
    lambda_grad: float = 0.0

    # HoVer-weighted high-pass (TRAIN ONLY; hover not used in inference)
    use_hover: bool = True
    hover_bnd_sigma: float = 2.0
    lambda_hover_hp: float = 0.2
    hover_hp_sigma: float = 1.5


# =============================================================================
# Main SR Model
# =============================================================================
class SRModel(nn.Module):
    def __init__(self, cfg: Optional[SRModelConfig] = None):
        super().__init__()
        self.cfg = cfg or SRModelConfig()

        if int(self.cfg.kernel_size) % 2 != 1:
            raise ValueError("kernel_size must be odd.")

        self.encoder = LocalEncoder(in_ch=3, feat_ch=self.cfg.feat_ch)
        self.pe = FourierPositionalEncoding(num_bands=self.cfg.pe_bands)
        pe_dim = 4 * int(self.cfg.pe_bands)

        self.kernel_heads = max(1, int(self.cfg.kernel_heads))
        self.kernel_mlp = KernelMLP(
            feat_dim=int(self.cfg.feat_ch),
            pe_dim=pe_dim,
            kernel_size=int(self.cfg.kernel_size),
            hidden=int(self.cfg.mlp_hidden),
            depth=int(self.cfg.mlp_depth),
            num_heads=self.kernel_heads,
        )

        self.res_refiner = ResidualRefiner() if bool(self.cfg.use_res_refiner) else None

        # offsets (K*K)
        r = int(self.cfg.kernel_size) // 2
        offsets = [(ox, oy) for oy in range(-r, r + 1) for ox in range(-r, r + 1)]
        self.register_buffer("_kernel_offsets", torch.tensor(offsets, dtype=torch.float32), persistent=False)

        # training state
        self._epoch = 0
        self._tau = float(self.cfg.tau_start)
        self._global_step = 0
        self._gate_tau = float(self.cfg.kernel_gate_tau_start)

        # debug cache
        self._last_gate_prob = None
        self._last_gate_used = False

    # ----------------------------
    # schedules
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

    def _update_gate_tau(self):
        tau_s = float(getattr(self.cfg, "kernel_gate_tau_start", 1.0))
        tau_e = float(getattr(self.cfg, "kernel_gate_tau_end", tau_s))
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

    # ----------------------------
    # core ops
    # ----------------------------
    @staticmethod
    def _gray_highpass(x_rgb: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
        """High-pass on grayscale using (g - blur(g))."""
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

    @staticmethod
    def _blur01(m: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0:
            return m
        k = int(round(sigma * 4 + 1))
        if k % 2 == 0:
            k += 1
        pad = k // 2
        mp = F.pad(m, (pad, pad, pad, pad), mode="replicate")
        return F.avg_pool2d(mp, kernel_size=k, stride=1)

    def _lr_continuous_from_hr_int(
        self,
        hr_xy_int: torch.Tensor,
        lr_hw: Tuple[int, int],
        hr_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        HR int -> LR continuous coordinate (center-aligned) and frac + LR norm.
        Returns:
          lr_xy_cont: [N,2] float in LR coords
          lr_frac: [N,2] in [0,1]
          lr_xy_norm: [N,2] normalized [-1,1] for grid_sample
        """
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

    def _kernel_weights_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: [..., K2]
        return: [..., K2] (sum=1, non-negative if kernel_allow_negative=False)
        """
        if bool(getattr(self.cfg, "kernel_logit_norm", True)):
            m = logits.mean(dim=-1, keepdim=True)
            s = logits.std(dim=-1, keepdim=True).clamp_min(1e-6)
            logits = (logits - m) / s

        tau = float(self._tau)
        logits = logits / max(tau, 1e-6)

        if not bool(self.cfg.kernel_allow_negative):
            return torch.softmax(logits, dim=-1)

        # optional negative-weight branch (kept for compatibility)
        w = torch.tanh(logits)
        K2 = w.shape[-1]
        center = K2 // 2
        oh = F.one_hot(torch.tensor(center, device=w.device), num_classes=K2).to(w.dtype)
        w = w + oh.view(*([1] * (w.ndim - 1)), K2)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)
        return w

    def _predict_kernel_weights(
        self,
        feat_lr: torch.Tensor,
        hr_xy_norm: torch.Tensor,
        lr_xy_norm_center: torch.Tensor,
        lr_frac: torch.Tensor,
        return_gate: bool = False,
    ):
        """
        Predict dynamic kernel weights (mixture of heads) for each HR point.
        This function is unconditional (no HoVer inputs), to keep inference clean.

        Returns:
          w: [B,N,K2]
          (optional) gate: [B,N,H]
        """
        B = feat_lr.shape[0]
        N = hr_xy_norm.shape[0]
        K2 = int(self.cfg.kernel_size) ** 2
        H = self.kernel_heads

        # sample LR feature at mapped centers
        grid = lr_xy_norm_center.view(1, 1, -1, 2).repeat(B, 1, 1, 1)
        cen = F.grid_sample(feat_lr, grid, mode="bilinear", align_corners=False)  # [B,C,1,N]
        cen = cen.squeeze(2).transpose(1, 2).contiguous()  # [B,N,C]

        pe = self.pe(hr_xy_norm).unsqueeze(0).expand(B, -1, -1)
        frac = lr_frac.unsqueeze(0).expand(B, -1, -1)

        x = torch.cat([cen, pe, frac], dim=-1).contiguous().view(B * N, -1)

        klogits, glogits = self.kernel_mlp(x)
        klogits = klogits.view(B, N, H, K2)
        glogits = glogits.view(B, N, H)

        gate = torch.softmax(glogits / max(float(self._gate_tau), 1e-6), dim=-1)  # [B,N,H]
        self._last_gate_prob = gate.detach()
        self._last_gate_used = True

        w_head = self._kernel_weights_from_logits(klogits.view(B, N * H, K2)).view(B, N, H, K2)
        w = (gate.unsqueeze(-1) * w_head).sum(dim=2)  # [B,N,K2]

        if return_gate:
            return w, gate
        return w

    def _gather_lr_patch(self, lr: torch.Tensor, lr_xy_cont: torch.Tensor) -> torch.Tensor:
        """
        Gather LR KxK neighborhood for each point.
        Returns: [B,N,K2,3]
        """
        B, C, H, W = lr.shape
        if C != 3:
            raise ValueError("This SRModel assumes RGB input (C=3).")

        K = int(self.cfg.kernel_size)
        r = K // 2
        device = lr.device

        x0 = torch.floor(lr_xy_cont[:, 0]).long()
        y0 = torch.floor(lr_xy_cont[:, 1]).long()

        offs = torch.arange(-r, r + 1, device=device, dtype=torch.long)
        oy, ox = torch.meshgrid(offs, offs, indexing="ij")
        ox = ox.reshape(-1)
        oy = oy.reshape(-1)

        xx = (x0[:, None] + ox[None, :]).clamp(0, W - 1)
        yy = (y0[:, None] + oy[None, :]).clamp(0, H - 1)

        idx = (yy * W + xx).view(-1)  # [N*K2]
        lr_flat = lr.view(B, C, H * W)
        gathered = lr_flat[:, :, idx]  # [B,3,N*K2]
        gathered = gathered.view(B, C, -1, ox.numel()).permute(0, 2, 3, 1).contiguous()  # [B,N,K2,3]
        return gathered

    def predict_points_base(
        self,
        lr: torch.Tensor,
        hr_xy_int: torch.Tensor,
        hr_hw: Tuple[int, int],
        feat_lr: torch.Tensor,
        lr_up: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict RGB residual at queried HR integer points.
        Returns: res [B,N,3]
        """
        B, _, H_lr, W_lr = lr.shape
        H_hr, W_hr = hr_hw

        hr_xy_norm = _hr_int_to_hr_norm(hr_xy_int, H_hr, W_hr)

        lr_xy_cont, lr_frac, lr_xy_norm_center = self._lr_continuous_from_hr_int(
            hr_xy_int=hr_xy_int,
            lr_hw=(H_lr, W_lr),
            hr_hw=(H_hr, W_hr),
        )

        xh = hr_xy_int[:, 0].long().clamp(0, W_hr - 1)
        yh = hr_xy_int[:, 1].long().clamp(0, H_hr - 1)
        lr_up_pts = lr_up[:, :, yh, xh].permute(0, 2, 1).contiguous()  # [B,N,3]

        w = self._predict_kernel_weights(
            feat_lr=feat_lr,
            hr_xy_norm=hr_xy_norm,
            lr_xy_norm_center=lr_xy_norm_center,
            lr_frac=lr_frac,
            return_gate=False,
        )  # [B,N,K2]

        patch = self._gather_lr_patch(lr, lr_xy_cont)  # [B,N,K2,3]
        sr_like = (patch * w.unsqueeze(-1)).sum(dim=2)  # [B,N,3]
        res = sr_like - lr_up_pts
        return res

    def _predict_sr_crop_unfused(
        self,
        lr: torch.Tensor,
        feat_lr: torch.Tensor,
        lr_up: torch.Tensor,
        top: int,
        left: int,
        crop: int,
    ) -> torch.Tensor:
        """
        Dense SR crop prediction (unfused, for training supervision).
        Returns: [B,3,crop,crop] in [0,1]
        """
        device = lr.device
        B = lr.shape[0]
        H_hr, W_hr = lr_up.shape[-2], lr_up.shape[-1]

        ys = torch.arange(top, top + crop, device=device)
        xs = torch.arange(left, left + crop, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        hr_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(torch.float32)  # [N,2]
        N = hr_xy.shape[0]

        chunk = int(getattr(self.cfg, "train_crop_chunk", 8192))
        outs = []
        for st in range(0, N, chunk):
            ed = min(N, st + chunk)
            res_pts = self.predict_points_base(
                lr=lr,
                hr_xy_int=hr_xy[st:ed],
                hr_hw=(H_hr, W_hr),
                feat_lr=feat_lr,
                lr_up=lr_up,
            )  # [B,n,3]
            outs.append(res_pts)

        res_all = torch.cat(outs, dim=1)  # [B,N,3]
        res_map = res_all.view(B, crop, crop, 3).permute(0, 3, 1, 2).contiguous()

        lr_up_crop = lr_up[:, :, top:top + crop, left:left + crop]
        sr_pred = (lr_up_crop + res_map).clamp(0, 1)

        if bool(getattr(self.cfg, "use_residual", True)) and (self.res_refiner is not None):
            res = self.res_refiner(lr_up_crop, res_map)
            sr_pred = (lr_up_crop + res).clamp(0, 1)

        return sr_pred

    # ----------------------------
    # inference
    # ----------------------------
    @torch.no_grad()
    def super_resolve(self, lr: torch.Tensor, out_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Inference: SR = lr_up + residual (predicted by dynamic kernel).
        HoVer is NOT used.
        Output uses keep-color fusion.
        """
        B, _, H_lr, W_lr = lr.shape
        if out_hw is None:
            out_hw = (H_lr * int(self.cfg.scale), W_lr * int(self.cfg.scale))
        H_hr, W_hr = out_hw

        lr_up = F.interpolate(lr, size=(H_hr, W_hr), mode="bicubic", align_corners=False).clamp(0, 1)
        feat_lr = self.encoder(lr)

        hr_xy_int_full = _make_hr_int_grid(H_hr, W_hr, device=lr.device, dtype=torch.float32)  # [N,2]
        N = hr_xy_int_full.shape[0]
        chunk = int(self.cfg.infer_chunk)

        outs = []
        for st in range(0, N, chunk):
            ed = min(N, st + chunk)
            res_pts = self.predict_points_base(
                lr=lr,
                hr_xy_int=hr_xy_int_full[st:ed],
                hr_hw=(H_hr, W_hr),
                feat_lr=feat_lr,
                lr_up=lr_up,
            )
            outs.append(res_pts)

        res_all = torch.cat(outs, dim=1)
        res_map = res_all.view(B, H_hr, W_hr, 3).permute(0, 3, 1, 2).contiguous()

        if bool(getattr(self.cfg, "use_residual", True)) and (self.res_refiner is not None):
            res = self.res_refiner(lr_up, res_map)
            sr_pred = (lr_up + res).clamp(0, 1)
        else:
            sr_pred = (lr_up + res_map).clamp(0, 1)

        return _fuse_keep_color(lr_up, sr_pred)

    def forward(self, lr: torch.Tensor, hover_bnd: Optional[torch.Tensor] = None, hover_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # hover_* are accepted for compatibility but ignored in inference
        return self.super_resolve(lr, out_hw=None)

    # ----------------------------
    # training loss (Scheme 1)
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
        Dense crop reconstruction + HoVer-weighted high-pass (train only).
        """
        B, _, H_lr, W_lr = lr.shape
        Bh, _, H_hr, W_hr = hr.shape
        if B != Bh:
            raise ValueError("lr/hr batch size mismatch")

        device = lr.device

        if self.training:
            self._global_step += 1
            self._update_gate_tau()

        self._tau = float(self._compute_tau(self._epoch))

        lr_up = F.interpolate(lr, size=(H_hr, W_hr), mode="bicubic", align_corners=False).clamp(0, 1)
        feat_student = self.encoder(lr)

        # build soft hover boundary weight (HR space) for TRAIN ONLY
        if bool(getattr(self.cfg, "use_hover", True)) and (hover_bnd is not None):
            hb = hover_bnd
            if hb.ndim == 3:
                hb = hb.unsqueeze(1)
            hb = hb[:, :1].to(device=device, dtype=torch.float32)
            if hb.max() > 1.5:
                hb = hb / 255.0
            hb = hb.clamp(0.0, 1.0)
            if hb.shape[-2:] != (H_hr, W_hr):
                hb = F.interpolate(hb, size=(H_hr, W_hr), mode="nearest")
            sigma_b = float(getattr(self.cfg, "hover_bnd_sigma", 2.0))
            W_bnd_soft = self._blur01(hb, sigma=sigma_b).clamp(0.0, 1.0)
        else:
            W_bnd_soft = torch.zeros((B, 1, H_hr, W_hr), device=device, dtype=torch.float32)

        crop = int(getattr(self.cfg, "train_crop", 192))
        crop = max(64, min(crop, H_hr, W_hr))
        crops_per_img = int(getattr(self.cfg, "train_crops_per_img", 1))

        lam_l1 = float(getattr(self.cfg, "lambda_crop_l1", 1.0))
        lam_hp = float(getattr(self.cfg, "lambda_hover_hp", 0.0))
        lam_g = float(getattr(self.cfg, "lambda_grad", 0.0))
        hp_sigma = float(getattr(self.cfg, "hover_hp_sigma", 1.5))

        loss_l1 = torch.tensor(0.0, device=device)
        loss_hp = torch.tensor(0.0, device=device)
        loss_grad = torch.tensor(0.0, device=device)

        # choose crops, bias to boundary if available
        for b in range(B):
            for _ in range(crops_per_img):
                if float(W_bnd_soft[b, 0].sum().detach().cpu()) > 1e-6:
                    w = (W_bnd_soft[b, 0].reshape(-1) + 1e-6)
                    prob = w / w.sum()
                    idx0 = torch.multinomial(prob, num_samples=1, replacement=True)[0].item()
                    cy = idx0 // W_hr
                    cx = idx0 - cy * W_hr
                    top = int(max(0, min(H_hr - crop, cy - crop // 2)))
                    left = int(max(0, min(W_hr - crop, cx - crop // 2)))
                else:
                    top = torch.randint(0, H_hr - crop + 1, (1,), device=device).item()
                    left = torch.randint(0, W_hr - crop + 1, (1,), device=device).item()

                sr_crop = self._predict_sr_crop_unfused(
                    lr=lr[b:b + 1],
                    feat_lr=feat_student[b:b + 1],
                    lr_up=lr_up[b:b + 1],
                    top=top,
                    left=left,
                    crop=crop,
                )
                hr_crop = hr[b:b + 1, :, top:top + crop, left:left + crop]

                loss_l1 = loss_l1 + (sr_crop - hr_crop).abs().mean()

                if lam_hp > 0:
                    w_crop = W_bnd_soft[b:b + 1, :, top:top + crop, left:left + crop]
                    # normalize by weight mass to stabilize magnitude
                    denom = w_crop.mean().clamp_min(1e-6)
                    hp_sr = self._gray_highpass(sr_crop, sigma=hp_sigma)
                    hp_hr = self._gray_highpass(hr_crop, sigma=hp_sigma)
                    loss_hp = loss_hp + (w_crop * (hp_sr - hp_hr).abs()).mean() / denom

                if lam_g > 0:
                    # edge weight from LR-up (no hover)
                    g_sr = _rgb_to_gray(sr_crop)
                    g_hr = _rgb_to_gray(hr_crop)
                    # simple edge weight (optional): emphasize strong gradients in HR target
                    w_edge = _minmax_norm((g_hr[:, :, :, 1:] - g_hr[:, :, :, :-1]).abs().mean(dim=3, keepdim=True).repeat(1,1,1,crop))
                    loss_grad = loss_grad + _image_grad_l1(sr_crop, hr_crop, weight=(1.0 + w_edge))

        norm = float(B * crops_per_img)
        loss_l1 = loss_l1 / norm
        loss_hp = loss_hp / norm
        loss_grad = loss_grad / norm

        total = lam_l1 * loss_l1 + lam_hp * loss_hp + lam_g * loss_grad

        # gate diagnostics (optional)
        gateH = None
        gateMax = None
        try:
            if self._last_gate_used and (self._last_gate_prob is not None):
                g = self._last_gate_prob  # [B,N,H]
                gpos = g.clamp_min(1e-12)
                ent = (-(gpos * gpos.log()).sum(dim=-1)).mean()
                gmax = g.max(dim=-1).values.mean()
                gateH = float(ent.detach().cpu())
                gateMax = float(gmax.detach().cpu())
        except Exception:
            gateH, gateMax = None, None

        if return_debug:
            dbg = {
                "loss_crop_l1": float(loss_l1.detach().cpu()),
                "loss_hover_hp": float(loss_hp.detach().cpu()),
                "loss_grad": float(loss_grad.detach().cpu()),
                "tau": float(getattr(self, "_tau", 1.0)),
                "gate_tau": float(getattr(self, "_gate_tau", 1.0)),
                "global_step": int(getattr(self, "_global_step", 0)),
                "gate_entropy": gateH,
                "gate_max": gateMax,
            }
            return total, dbg

        return total


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SRModelConfig()
    model = SRModel(cfg=cfg).to(device)

    lr = torch.rand(2, 3, 128, 128, device=device)
    hr = torch.rand(2, 3, 512, 512, device=device)
    hb = torch.rand(2, 1, 512, 512, device=device)

    sr = model(lr)
    print("sr:", sr.shape)

    loss, dbg = model.compute_loss(lr, hr, hover_bnd=hb, return_debug=True)
    print("loss:", float(loss), dbg)
