# model.py
# -*- coding: utf-8 -*-
"""
SR model (non-diffusion, non-GAN):
- Dynamic kernel prediction per HR pixel (local implicit SR)
- Optional residual SR: SR = upsample(LR) + residual
- Structure-aware sampling + loss reweighting using Cellpose nuclei masks (frozen guidance)
- Optional gradient-consistency loss on HR crops
- Kernel temperature annealing (tau schedule)

Public interface expected by train.py:
  - model = SRModel(cfg)
  - sr = model(lr)                      # inference (no HR required)
  - loss = model.compute_loss(lr, hr)   # training (HR required)
  - model.set_epoch(epoch)              # optional

Dependencies (for guidance):
  pip install cellpose
Cellpose is used ONLY to produce a weight map W (boundary emphasis) from HR,
and DOES NOT affect SR color generation directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Utilities
# =============================================================================
def _minmax_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample min-max normalize to [0,1]. x: [B,1,H,W]."""
    B = x.shape[0]
    v = x.view(B, -1)
    mn = v.min(dim=1)[0].view(B, 1, 1, 1)
    mx = v.max(dim=1)[0].view(B, 1, 1, 1)
    return (x - mn) / (mx - mn + eps)


def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """x: [B,3,H,W] in [0,1] -> [B,1,H,W]."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def _grad_mag_gray(x_gray: torch.Tensor) -> torch.Tensor:
    gx = x_gray[:, :, :, 1:] - x_gray[:, :, :, :-1]
    gy = x_gray[:, :, 1:, :] - x_gray[:, :, :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def _image_grad_l1(sr: torch.Tensor, hr: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Gradient L1 between sr and hr (grayscale).
    sr/hr: [B,3,H,W] in [0,1]
    weight: [B,1,H,W] optional, multiplies per-pixel gradient loss.
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
        w_dx = weight[:, :, :, :-1]
        w_dy = weight[:, :, :-1, :]
        dx_l1 = dx_l1 * w_dx
        dy_l1 = dy_l1 * w_dy

    return dx_l1.mean() + dy_l1.mean()


def _make_hr_int_grid(H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return [N,2] integer HR pixel coords (x,y) for all pixels."""
    ys = torch.arange(H, device=device)
    xs = torch.arange(W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(dtype=dtype)


def _hr_int_to_hr_norm(hr_xy_int: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Convert HR integer coords (x,y) to normalized coords [-1,1] at pixel centers."""
    x = hr_xy_int[:, 0].to(torch.float32)
    y = hr_xy_int[:, 1].to(torch.float32)
    x_norm = (x + 0.5) / float(W) * 2.0 - 1.0
    y_norm = (y + 0.5) / float(H) * 2.0 - 1.0
    return torch.stack([x_norm, y_norm], dim=1).to(dtype=torch.float32, device=hr_xy_int.device)


def _get_points_rgb(img: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """img: [3,H,W] -> gather [N,3] by flattened idx"""
    return img.permute(1, 2, 0).reshape(-1, 3)[idx]


def _get_points_map(map1: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """map1: [1,H,W] -> gather [N]"""
    return map1.reshape(-1)[idx]


# =============================================================================
# Positional Encoding
# =============================================================================
class FourierPositionalEncoding(nn.Module):
    """
    Fourier features for 2D coords (x,y) in [-1,1]:
      [sin(2^k*pi*x), cos(2^k*pi*x), sin(2^k*pi*y), cos(2^k*pi*y)]_{k=0..K-1}
    """

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
    Predict KxK kernel logits for each queried HR point.
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
# Optional residual head (tiny CNN)
# =============================================================================
class ResidualRefiner(nn.Module):
    """
    Color-safe refiner (no cross-channel mixing):
    - groups=3 so each channel (R/G/B) is processed independently.
    - input channels are 6: [lr_up_rgb, res0_rgb]
      3 groups, each group has 2 channels: (lr_up_c, res0_c)
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
# Cellpose guidance (stable)
# =============================================================================
class CellposeGuidance(nn.Module):
    """
    Use Cellpose (frozen) to produce a structural weight map W_cp in [0,1].
    - Run on HR only (during training/validation when HR is available).
    - DOES NOT backprop through Cellpose.
    - Output is boundary-emphasis derived from instance masks.

    Typical call:
      W = guidance(hr)  # hr: [B,3,H,W] in [0,1]
    """

    def __init__(
        self,
        model_type: str = "nuclei",
        use_gpu_if_available: bool = True,
        channels: Tuple[int, int] = (0, 0),
        diameter: float = 0.0,   # 0.0 = auto in cellpose
        use_gray: bool = True,   # recommended for H&E guidance stability
        max_side: int = 1024,    # downscale large HR for speed; then upscale masks back
    ):
        super().__init__()
        self.model_type = str(model_type)
        self.use_gpu_if_available = bool(use_gpu_if_available)
        self.channels = tuple(channels)
        self.diameter = float(diameter)
        self.use_gray = bool(use_gray)
        self.max_side = int(max_side)

        self._cp = None
        self._ready = False

    def _lazy_build(self, device: torch.device):
        if self._ready:
            return
        try:
            from cellpose import models  # type: ignore
        except Exception as e:
            raise ImportError(
                "Cellpose is required for CellposeGuidance.\n"
                "Install:\n  pip install cellpose\n"
                f"Original error: {e}"
            )

        gpu_flag = bool(self.use_gpu_if_available and device.type == "cuda")
        self._cp = models.CellposeModel(gpu=gpu_flag, model_type=self.model_type)
        self._ready = True

    @staticmethod
    def _morph_boundary_from_mask(mask01: torch.Tensor) -> torch.Tensor:
        """
        mask01: [1,1,H,W] float 0/1
        return: boundary [1,1,H,W] float [0,1]
        """
        dil = F.max_pool2d(mask01, 3, 1, 1)
        ero = -F.max_pool2d(-mask01, 3, 1, 1)
        boundary = (dil - ero).clamp(0, 1)
        # normalize
        boundary = boundary / (boundary.max() + 1e-6)
        return boundary

    @torch.no_grad()
    def forward(self, hr_01: torch.Tensor) -> torch.Tensor:
        """
        hr_01: [B,3,H,W] float in [0,1]
        return: W_cp [B,1,H,W] float in [0,1]
        """
        if hr_01.ndim != 4 or hr_01.shape[1] != 3:
            raise ValueError(f"hr must be [B,3,H,W], got {tuple(hr_01.shape)}")

        device = hr_01.device
        self._lazy_build(device)
        assert self._cp is not None

        B, _, H, W = hr_01.shape
        out_ws: List[torch.Tensor] = []

        # Process per-image (Cellpose API expects numpy)
        for b in range(B):
            x = hr_01[b].detach().clamp(0, 1)

            # optional grayscale guidance for stability (does NOT affect SR colors)
            if self.use_gray:
                xg = _rgb_to_gray(x.unsqueeze(0)).squeeze(0)  # [1,H,W]
                # cellpose accepts HxW or HxWx3; we pass HxW float
                im_np = xg.squeeze(0).float().cpu().numpy()
            else:
                # HWC RGB float
                im_np = x.permute(1, 2, 0).float().cpu().numpy()

            # optional downscale for speed; upscale boundary back
            orig_hw = (H, W)
            work_hw = (H, W)
            if self.max_side > 0:
                m = max(H, W)
                if m > self.max_side:
                    scale = self.max_side / float(m)
                    Hs = max(32, int(round(H * scale)))
                    Ws = max(32, int(round(W * scale)))
                    work_hw = (Hs, Ws)
                    # resize via torch for consistency
                    if self.use_gray:
                        tmp = torch.from_numpy(im_np)[None, None].to(device=device, dtype=torch.float32)
                        tmp = F.interpolate(tmp, size=work_hw, mode="bilinear", align_corners=False)
                        im_np = tmp[0, 0].cpu().numpy()
                    else:
                        tmp = torch.from_numpy(im_np).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32)
                        tmp = F.interpolate(tmp, size=work_hw, mode="bilinear", align_corners=False)
                        im_np = tmp[0].permute(1, 2, 0).cpu().numpy()

            # run cellpose
            # returns masks as HxW with instance ids (0=background)
            masks, flows, styles, diams = self._cp.eval(
                im_np,
                channels=list(self.channels),
                diameter=(None if self.diameter <= 0 else self.diameter),
                normalize=True,
            )

            # masks -> foreground binary
            import numpy as np  # local import
            fg = (masks.astype(np.int32) > 0).astype(np.float32)  # HxW

            mask_t = torch.from_numpy(fg)[None, None].to(device=device, dtype=torch.float32)  # [1,1,Hs,Ws]
            boundary = self._morph_boundary_from_mask(mask_t)  # [1,1,Hs,Ws]

            # resize back to original HR
            if work_hw != orig_hw:
                boundary = F.interpolate(boundary, size=orig_hw, mode="bilinear", align_corners=False)

            out_ws.append(boundary)

        W_cp = torch.cat(out_ws, dim=0).clamp(0, 1)  # [B,1,H,W]
        return W_cp


# =============================================================================
# Config
# =============================================================================
@dataclass
class SRModelConfig:
    # scaling
    scale: int = 4

    # encoder & kernel predictor
    feat_ch: int = 64
    pe_bands: int = 10
    mlp_hidden: int = 256
    mlp_depth: int = 5

    # kernel prediction
    kernel_size: int = 7  # odd
    kernel_allow_negative: bool = True

    # training sampling
    num_points: int = 4096
    saliency_ratio: float = 0.7
    loss_alpha: float = 3.0

    # gradient consistency
    lambda_grad: float = 0.2
    grad_crop: int = 128

    # inference chunking
    infer_chunk: int = 8192

    # residual SR
    use_residual: bool = True
    use_res_refiner: bool = False

    # temperature annealing (tau)
    tau_start: float = 1.0
    tau_end: float = 0.5
    tau_warm_epochs: int = 2
    tau_anneal_epochs: int = 8

    # Cellpose guidance
    use_cellpose_guidance: bool = True
    cellpose_model_type: str = "nuclei"
    cellpose_use_gray: bool = True
    cellpose_channels0: int = 0   # channels=[c0,c1] for cellpose; for grayscale use [0,0]
    cellpose_channels1: int = 0
    cellpose_diameter: float = 0.0
    cellpose_max_side: int = 1024

    # mix guidance with edge saliency
    cellpose_mix_beta: float = 0.6          # W_mix = beta*W_cp + (1-beta)*W_edge
    cellpose_weight_alpha: float = 1.0      # additional scaling on W_cp


# =============================================================================
# Main SR model
# =============================================================================
class SRModel(nn.Module):
    """
    Exposed:
      - forward(lr)->sr
      - compute_loss(lr,hr)->loss
      - set_epoch(epoch) (optional)
    """

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

        # guidance (lazy inside)
        self._cp = CellposeGuidance(
            model_type=self.cfg.cellpose_model_type,
            use_gpu_if_available=True,
            channels=(int(self.cfg.cellpose_channels0), int(self.cfg.cellpose_channels1)),
            diameter=float(self.cfg.cellpose_diameter),
            use_gray=bool(self.cfg.cellpose_use_gray),
            max_side=int(self.cfg.cellpose_max_side),
        )

        # precompute K*K offsets in LR pixel units
        r = self.cfg.kernel_size // 2
        offsets = []
        for oy in range(-r, r + 1):
            for ox in range(-r, r + 1):
                offsets.append((ox, oy))
        self.register_buffer("_kernel_offsets", torch.tensor(offsets, dtype=torch.float32), persistent=False)

        # temperature state
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
    # saliency / guidance
    # ----------------------------
    @torch.no_grad()
    def compute_mixed_saliency(
        self,
        lr: torch.Tensor,
        hr: torch.Tensor,
        out_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          W_mix: [B,1,H,W] mixed weight
          W_cp : [B,1,H,W] Cellpose boundary weight (zeros if disabled)
          W_edge:[B,1,H,W] edge weight from LR-upsample
        """
        H_hr, W_hr = out_hw
        lr_up = F.interpolate(lr, size=(H_hr, W_hr), mode="bicubic", align_corners=False).clamp(0, 1)

        # edge saliency from lr_up
        g = _rgb_to_gray(lr_up)
        W_edge = _minmax_norm(_grad_mag_gray(g))

        W_cp = torch.zeros_like(W_edge)
        if bool(self.cfg.use_cellpose_guidance):
            # try:
            W_cp = self._cp(hr.detach().clamp(0, 1))
            # except Exception:
            #     # fallback silently: keep zeros to not break training
            #     W_cp = torch.zeros_like(W_edge)

        alpha = float(getattr(self.cfg, "cellpose_weight_alpha", 1.0))
        if alpha != 1.0:
            W_cp = (alpha * W_cp).clamp(0, 1)

        beta = float(getattr(self.cfg, "cellpose_mix_beta", 0.6))
        W_mix = (beta * W_cp + (1.0 - beta) * W_edge).clamp(0, 1)
        return W_mix, W_cp, W_edge

    # ----------------------------
    # coord mapping
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
    # kernel weights + temperature
    # ----------------------------
    def _kernel_weights_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: [B,N,K]
        Apply temperature: logits / tau
        Then produce weights (allow negative if configured).
        """
        tau = float(self._tau)
        logits = logits / max(tau, 1e-6)

        if not bool(self.cfg.kernel_allow_negative):
            return torch.softmax(logits, dim=-1)

        # allow negative but keep normalization; avoid inplace ops
        w = torch.tanh(logits)  # [B,N,K]
        K = w.shape[-1]
        center = K // 2
        oh = F.one_hot(torch.tensor(center, device=w.device), num_classes=K).to(w.dtype)  # [K]
        w = w + oh.view(1, 1, K)
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
        pe = pe.unsqueeze(0).expand(B, -1, -1)  # [B,N,pe_dim]
        frac = lr_frac.to(device=feat_lr.device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)  # [B,N,2]

        inp = torch.cat([cen, pe, frac], dim=-1)  # [B,N,D]
        logits = self.kernel_mlp(inp.view(B * N, -1)).view(B, N, K)
        return self._kernel_weights_from_logits(logits)

    # ----------------------------
    # point prediction (base SR from kernel)
    # ----------------------------
    def predict_points_base(
        self,
        lr: torch.Tensor,
        hr_xy_int: torch.Tensor,
        hr_hw: Tuple[int, int],
        feat_lr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict base SR at points using dynamic kernel.
        Returns sr_base [B,N,3] in [0,1]
        """
        B, _, H_lr, W_lr = lr.shape
        H_hr, W_hr = hr_hw

        if feat_lr is None:
            feat_lr = self.encoder(lr)

        hr_xy_int = hr_xy_int.to(device=lr.device, dtype=torch.float32)
        hr_xy_norm = _hr_int_to_hr_norm(hr_xy_int, H_hr, W_hr)  # [N,2]
        lr_xy_cont, lr_frac, lr_xy_norm_center = self._lr_continuous_from_hr_int(
            hr_xy_int=hr_xy_int, lr_hw=(H_lr, W_lr), hr_hw=(H_hr, W_hr)
        )

        w = self._predict_kernel_weights(feat_lr, hr_xy_norm, lr_xy_norm_center, lr_frac)  # [B,N,K]
        offsets = self._kernel_offsets.to(device=lr.device, dtype=torch.float32)
        rgb = self._sample_lr_rgb_neighbors(lr, lr_xy_cont, offsets)  # [B,N,K,3]

        sr_base = (w.unsqueeze(-1) * rgb).sum(dim=2).clamp(0, 1)  # [B,N,3]
        return sr_base

    # ----------------------------
    # full image SR (inference)
    # ----------------------------
    @torch.no_grad()
    def super_resolve(self, lr: torch.Tensor, out_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        B, _, H_lr, W_lr = lr.shape
        if out_hw is None:
            out_hw = (H_lr * self.cfg.scale, W_lr * self.cfg.scale)
        H_hr, W_hr = out_hw

        lr_up = F.interpolate(lr, size=(H_hr, W_hr), mode="bicubic", align_corners=False).clamp(0, 1)
        feat_lr = self.encoder(lr)

        hr_xy_int_full = _make_hr_int_grid(H_hr, W_hr, device=lr.device, dtype=torch.float32)
        N = hr_xy_int_full.shape[0]
        chunk = int(self.cfg.infer_chunk)

        outs = []
        for st in range(0, N, chunk):
            ed = min(N, st + chunk)
            sr_pts = self.predict_points_base(
                lr=lr,
                hr_xy_int=hr_xy_int_full[st:ed],
                hr_hw=(H_hr, W_hr),
                feat_lr=feat_lr,
            )  # [B,chunk,3]
            outs.append(sr_pts)

        sr_base = torch.cat(outs, dim=1).transpose(1, 2).contiguous().view(B, 3, H_hr, W_hr).clamp(0, 1)

        if not bool(self.cfg.use_residual):
            return sr_base

        res0 = (sr_base - lr_up)
        if self.res_refiner is not None:
            res = self.res_refiner(lr_up, res0)
        else:
            res = res0

        sr = (lr_up + res).clamp(0, 1)
        return sr

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        # inference does not require HR
        return self.super_resolve(lr, out_hw=None)

    # ----------------------------
    # training loss
    # ----------------------------
    def compute_loss(self, lr: torch.Tensor, hr: torch.Tensor, return_debug: bool = False):
        """
        Mixed saliency:
          W_mix = beta*W_cellpose + (1-beta)*W_edge

        Pointwise loss on sampled HR pixels using SR_points and HR_points, with weights (1 + loss_alpha*W_mix).

        Optional gradient consistency loss on random HR crop:
          L_grad = L1(∇SR, ∇HR) weighted by (1 + W_crop)
        """
        B, _, H_lr, W_lr = lr.shape
        Bh, _, H_hr, W_hr = hr.shape
        if B != Bh:
            raise ValueError("lr/hr batch size mismatch")

        # keep tau updated even if caller never uses set_epoch()
        self._tau = float(self._compute_tau(self._epoch))

        # saliency maps (requires HR for cellpose)
        W_mix, W_cp, W_edge = self.compute_mixed_saliency(lr, hr, out_hw=(H_hr, W_hr))

        # upsample LR
        lr_up = F.interpolate(lr, size=(H_hr, W_hr), mode="bicubic", align_corners=False).clamp(0, 1)
        feat_lr = self.encoder(lr)

        # (1) point sampling guided by W_mix
        N = int(self.cfg.num_points)
        N_sal = int(round(N * float(self.cfg.saliency_ratio)))
        N_uni = N - N_sal

        loss_pix = 0.0
        mean_mix = 0.0
        mean_cp = 0.0
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
            hr_xy_int = torch.stack([x, y], dim=1).to(dtype=torch.float32, device=lr.device)  # [N,2]

            # base SR points
            sr_base_pts = self.predict_points_base(
                lr=lr[b:b + 1],
                hr_xy_int=hr_xy_int,
                hr_hw=(H_hr, W_hr),
                feat_lr=feat_lr[b:b + 1],
            )[0]  # [N,3]

            # residual SR points (cheap: pointwise res0)
            lr_up_pts = _get_points_rgb(lr_up[b], idx)  # [N,3]
            if bool(self.cfg.use_residual):
                res0_pts = sr_base_pts - lr_up_pts
                sr_pts = (lr_up_pts + res0_pts).clamp(0, 1)
            else:
                sr_pts = sr_base_pts

            gt_pts = _get_points_rgb(hr[b], idx)

            ws = _get_points_map(W_mix[b], idx)
            mean_mix += float(ws.mean().detach().cpu())
            mean_cp += float(_get_points_map(W_cp[b], idx).mean().detach().cpu())
            mean_edge += float(_get_points_map(W_edge[b], idx).mean().detach().cpu())

            weight = 1.0 + float(self.cfg.loss_alpha) * ws
            l1 = (sr_pts - gt_pts).abs().mean(dim=1)  # [N]
            loss_b = (weight * l1).mean()
            loss_pix = loss_pix + loss_b

        loss_pix = loss_pix / B
        mean_mix = mean_mix / max(B, 1)
        mean_cp = mean_cp / max(B, 1)
        mean_edge = mean_edge / max(B, 1)

        # (2) gradient consistency on random HR crop
        lam = float(self.cfg.lambda_grad)
        crop = int(self.cfg.grad_crop)

        loss_grad = torch.tensor(0.0, device=lr.device)
        if lam > 0 and crop > 0:
            crop = min(crop, H_hr, W_hr)
            grad_acc = 0.0
            for b in range(B):
                top = torch.randint(0, H_hr - crop + 1, (1,), device=lr.device).item()
                left = torch.randint(0, W_hr - crop + 1, (1,), device=lr.device).item()

                hr_crop = hr[b:b + 1, :, top:top + crop, left:left + crop]
                w_crop = W_mix[b:b + 1, :, top:top + crop, left:left + crop]

                ys = torch.arange(top, top + crop, device=lr.device)
                xs = torch.arange(left, left + crop, device=lr.device)
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                hr_xy_crop = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(torch.float32)

                sr_base_pts = self.predict_points_base(
                    lr=lr[b:b + 1],
                    hr_xy_int=hr_xy_crop,
                    hr_hw=(H_hr, W_hr),
                    feat_lr=feat_lr[b:b + 1],
                )
                sr_base_crop = sr_base_pts.transpose(1, 2).contiguous().view(1, 3, crop, crop).clamp(0, 1)

                lr_up_crop = lr_up[b:b + 1, :, top:top + crop, left:left + crop]
                if bool(self.cfg.use_residual):
                    res0 = sr_base_crop - lr_up_crop
                    if self.res_refiner is not None:
                        res = self.res_refiner(lr_up_crop, res0)
                    else:
                        res = res0
                    sr_crop = (lr_up_crop + res).clamp(0, 1)
                else:
                    sr_crop = sr_base_crop

                grad_acc = grad_acc + _image_grad_l1(sr_crop, hr_crop, weight=(1.0 + w_crop))

            loss_grad = grad_acc / B

        loss = loss_pix + lam * loss_grad

        if return_debug:
            return loss, {
                "loss_pix": float(loss_pix.detach().cpu()),
                "loss_grad": float(loss_grad.detach().cpu()),
                "mean_mix": mean_mix,
                "mean_cellpose": mean_cp,
                "mean_edge": mean_edge,
                "tau": float(self._tau),
            }
        return loss


if __name__ == "__main__":
    # quick self-test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SRModelConfig()
    model = SRModel(cfg).to(device)
    
    lr = torch.rand(2, 3, 128, 128).to(device)
    hr = torch.rand(2, 3, 512, 512).to(device)
    
    sr = model(lr)
    print("SR shape:", sr.shape)
    
    model.compute_mixed_saliency(lr, hr, out_hw=(512, 512))
