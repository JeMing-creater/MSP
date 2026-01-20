# src/vfm_control.py
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

try:
    from transformers import Dinov2Model, Dinov2Config
except Exception:
    Dinov2Model = None
    Dinov2Config = None


@dataclass
class VFMConfig:
    variant: str = "dinov2_vitb14"   # 仅用于记录
    local_dir: str = "./src/models/dinov2_vitb14"   # 本地目录（离线）
    image_size: int = 518           # dinov2常见输入
    patch_size: int = 14            # vitb14=14
    control_mode: str = "energy_edge_gray"
    normalize: bool = True


def _make_edge_from_gray(gray_1ch: torch.Tensor) -> torch.Tensor:
    # gray_1ch: [B,1,H,W] in [0,1]
    dx = torch.abs(gray_1ch[:, :, :, 1:] - gray_1ch[:, :, :, :-1])
    dy = torch.abs(gray_1ch[:, :, 1:, :] - gray_1ch[:, :, :-1, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    edge = torch.clamp(dx + dy, 0.0, 1.0)
    return edge


class EnergyHead(torch.nn.Module):
    """
    轻量结构能量头：feat_map [B,C,h,w] -> energy [B,1,h,w] in [0,1]
    只训练该模块，避免破坏 DINOv2 backbone 的语义先验。
    """
    def __init__(self, in_ch: int, mid_ch: int = 64):
        super().__init__()
        self.proj1 = torch.nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=True)
        self.proj2 = torch.nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.proj3 = torch.nn.Conv2d(mid_ch, 1, kernel_size=1, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.act(self.proj1(feat))
        x = self.act(self.proj2(x))
        x = self.proj3(x)
        x = torch.sigmoid(x)
        return x



class VFMControlGenerator(torch.nn.Module):
    """
    输入 lr_up [B,3,512,512] -> 输出 control [B,3,512,512]
    默认：推理/验证时冻结且 no_grad
    训练时：可选择只解冻 DINOv2 最后若干个 block 参与反传
    """
    def __init__(self, cfg: VFMConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        if Dinov2Model is None:
            raise RuntimeError(
                "transformers.Dinov2Model not available. "
                "Please upgrade transformers or install a version that includes Dinov2Model."
            )

        if not cfg.local_dir or (not os.path.isdir(cfg.local_dir)):
            raise FileNotFoundError(f"DINOv2 local_dir not found: {cfg.local_dir}")

        # ✅ 离线加载 DINOv2 backbone（永远冻结）
        self.vfm = Dinov2Model.from_pretrained(
            cfg.local_dir,
            local_files_only=True,
        ).to(device)

        # ✅ 建立 EnergyHead（可训练）
        hidden = getattr(getattr(self.vfm, "config", None), "hidden_size", None)
        if hidden is None:
            hidden = 768  # vitb14 常见 hidden size
        self.energy_head = EnergyHead(in_ch=int(hidden), mid_ch=64).to(device)

        # 冻结 backbone，训练 head
        self.freeze_all()

    def freeze_all(self):
        """
        冻结 DINOv2 backbone；EnergyHead 默认可训练。
        """
        self.vfm.eval()
        for p in self.vfm.parameters():
            p.requires_grad_(False)

        self.energy_head.train()
        for p in self.energy_head.parameters():
            p.requires_grad_(True)

    def save_ckpt(self, path: str):
        """
        Save ONLY trainable parts (EnergyHead) + minimal meta.
        DINOv2 backbone is frozen & loaded from cfg.local_dir, so we do not save it.
        """
        import os
        import torch

        os.makedirs(os.path.dirname(path), exist_ok=True)

        payload = {
            "type": "VFMControlGenerator",
            "version": 1,
            "cfg": {
                # keep minimal fields for sanity/debug; not strictly required for loading
                "variant": getattr(self.cfg, "variant", None),
                "local_dir": getattr(self.cfg, "local_dir", None),
                "image_size": int(getattr(self.cfg, "image_size", 0) or 0),
                "patch_size": int(getattr(self.cfg, "patch_size", 0) or 0),
                "control_mode": getattr(self.cfg, "control_mode", None),
                "normalize": bool(getattr(self.cfg, "normalize", False)),
            },
            "energy_head": self.energy_head.state_dict(),
        }
        torch.save(payload, path)


    def load_ckpt(self, path: str, strict: bool = True):
        """
        Load ONLY EnergyHead weights.
        Returns (missing_keys, unexpected_keys) like load_state_dict.
        """
        import torch

        obj = torch.load(path, map_location="cpu")

        # Accept both:
        # 1) payload dict with key "energy_head"
        # 2) raw state_dict (backward compatible)
        if isinstance(obj, dict) and "energy_head" in obj and isinstance(obj["energy_head"], dict):
            sd = obj["energy_head"]
        else:
            sd = obj

        missing, unexpected = self.energy_head.load_state_dict(sd, strict=strict)
        return missing, unexpected


    def forward_train_with_energy(self, lr_up: torch.Tensor):
        """
        训练用：返回 (control, energy_up)
        - lr_up: [B,3,512,512] in [0,1]
        - control: [B,3,512,512]
        - energy_up: [B,1,512,512]
        """
        feat = self._extract_feature_map_impl(lr_up)          # backbone frozen
        energy = self._feat_to_energy_impl(feat)              # head trainable
        ctrl = self._compose_control_impl(lr_up, energy)

        energy_up = F.interpolate(
            energy, size=(lr_up.shape[2], lr_up.shape[3]),
            mode="bilinear", align_corners=False
        ).clamp(0, 1)
        return ctrl, energy_up

    
    
    def set_trainable(self, unfreeze_last_n_blocks: int = 0, train_ln: bool = True):
        """
        为保持与你原来 trainer 调用兼容而保留该接口。
        现在策略：永远不解冻 DINOv2，仅训练 EnergyHead。
        """
        self.freeze_all()


    def _extract_feature_map_impl(self, lr_up: torch.Tensor) -> torch.Tensor:
        """
        lr_up: [B,3,512,512] in [0,1]
        return feat_map: [B,C,h,w]
        """
        B, C, H, W = lr_up.shape

        x = lr_up
        x = F.interpolate(x, size=(self.cfg.image_size, self.cfg.image_size), mode="bicubic", align_corners=False)

        # ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        out = self.vfm(pixel_values=x, output_hidden_states=False, return_dict=True)
        tokens = out.last_hidden_state  # [B, 1+N, C]
        tokens = tokens[:, 1:, :]       # drop CLS -> [B, N, C]

        gh = self.cfg.image_size // self.cfg.patch_size
        gw = self.cfg.image_size // self.cfg.patch_size
        N = tokens.shape[1]
        if gh * gw != N:
            g = int((N) ** 0.5)
            gh, gw = g, g
            tokens = tokens[:, : gh * gw, :]

        feat = tokens.transpose(1, 2).contiguous().view(B, -1, gh, gw)  # [B,C,gh,gw]
        return feat

    def _feat_to_energy_impl(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        feat_map: [B,C,h,w]
        return energy: [B,1,h,w] in [0,1]
        """
        energy = self.energy_head(feat_map)

        if self.cfg.normalize:
            # per-image min-max 归一化（可选；更利于可视化/数值稳定）
            B = energy.shape[0]
            e = energy.view(B, -1)
            e_min = e.min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
            e_max = e.max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
            energy = (energy - e_min) / torch.clamp(e_max - e_min, min=1e-6)
            energy = energy.clamp(0, 1)

        return energy


    def _compose_control_impl(self, lr_up: torch.Tensor, energy_1ch: torch.Tensor) -> torch.Tensor:
        """
        lr_up: [B,3,512,512] in [0,1]
        energy_1ch: [B,1,h,w] in [0,1]
        return: control [B,3,512,512] in [0,1]
        """
        gray = lr_up.mean(dim=1, keepdim=True)
        edge = _make_edge_from_gray(gray)

        energy = F.interpolate(energy_1ch, size=(lr_up.shape[2], lr_up.shape[3]), mode="bilinear", align_corners=False)
        energy = energy.clamp(0, 1)

        mode = self.cfg.control_mode.lower()
        if mode == "energy_edge_gray":
            ctrl = torch.cat([energy, edge, gray], dim=1)
        elif mode == "energy_only":
            ctrl = energy.repeat(1, 3, 1, 1)
        else:
            ctrl = torch.cat([gray, edge, gray], dim=1)

        return ctrl.clamp(0, 1)

    def forward_train(self, lr_up: torch.Tensor) -> torch.Tensor:
        """
        训练用：允许梯度流入 DINOv2（取决于 set_trainable 解冻了哪些层）
        """
        feat = self._extract_feature_map_impl(lr_up)
        energy = self._feat_to_energy_impl(feat)
        ctrl = self._compose_control_impl(lr_up, energy)
        return ctrl

    @torch.no_grad()
    def forward(self, lr_up: torch.Tensor) -> torch.Tensor:
        """
        推理/验证用：强制 no_grad
        """
        feat = self._extract_feature_map_impl(lr_up)
        energy = self._feat_to_energy_impl(feat)
        ctrl = self._compose_control_impl(lr_up, energy)
        return ctrl

