from calendar import c
from math import e
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
from datetime import datetime
from typing import Dict

import monai
import torch
import yaml
from tqdm import tqdm
import torch.nn as nn
from dataclasses import dataclass, field
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory


import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# WSI 切片流程：40x HR png + 10x LR png
# from CLAM import *
CLAM_ROOT = "CLAM"  # CLAM 根目录
sys.path.append(CLAM_ROOT)
import src.preprocess_tcga_luad_with_clam as slic_tool


# 读取 clinical.tsv
from src.tcga_clinical import TCGAClinicalTable, slide_id_from_path, slide_to_case_id
from src.loader import build_case_split_dataloaders
from src.utils import  Logger, same_seeds
from src.sr_diffusion_trainer import SRTrainConfig, DiffusionSRControlNetTrainer


if __name__ == "__main__":
    # 加载配置文件
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    
    same_seeds(config.data_loader.seed)
    
    logging_dir = (
        os.getcwd()
        + "/logs/"
        + config.checkpoint
        + str(datetime.now())
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
        .replace(".", "_")
    )
    
    accelerator = Accelerator(
        cpu=False, log_with=["tensorboard"], project_dir=logging_dir
    )
    
    Logger(logging_dir if accelerator.is_local_main_process else None)
    
    image_dict = {
        "TCGA-LUAD": [config.TCGA_LUAD.root, config.TCGA_LUAD.choose_WSI],
        "TCGA-KIRC": [config.TCGA_KIRC.root, config.TCGA_KIRC.choose_WSI],
        "TCGA-LIHC": [config.TCGA_LIHC.root, config.TCGA_LIHC.choose_WSI],
    }
    
    # 检查切片全面性
    slic_tool.run_clam_and_export(
        data_cfg = image_dict,
        out_img_dir=config.data_loader.out_img_dir,
        patch_size=config.data_loader.patch_size,
        step_size=config.data_loader.step_size,
        patch_level=config.data_loader.patch_level,
        down_scale=config.data_loader.down_scale,
        min_tissue_ratio = config.data_loader.min_tissue_ratio,
        seed=config.data_loader.seed,
        limit_samples = config.data_loader.patch_num
    )
    
    
    
    # 构建dataloaders
    train_loader, val_loader, test_loader = build_case_split_dataloaders(
        out_img_dir=config.data_loader.out_img_dir,
        batch_size=config.trainer.batch_size,
        patch_num=getattr(config.data_loader, "patch_num", 200),
        train_ratio=config.data_loader.train_ratio,
        val_ratio=config.data_loader.val_ratio,
        test_ratio=config.data_loader.test_ratio,
        split_seed=getattr(config.data_loader, "seed", 2025),
        num_workers=config.data_loader.num_workers,
        pin_memory=config.data_loader.pin_memory,
    )  
    
    
