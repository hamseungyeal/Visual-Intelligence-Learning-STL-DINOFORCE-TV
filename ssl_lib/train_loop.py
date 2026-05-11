"""
학습 루프 — 노트북과 CLI 양쪽에서 공유.

설계:
- train_one_epoch: 한 epoch 학습. 노트북에서도 직접 호출 가능.
- pretrain: 전체 학습 entry point. config 받아서 모든 것 빌드.
"""
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models.mocov2 import MoCoV2
from .models.byol import BYOL
from .data.stl10_unlabeled import build_stl10_loader
from .utils.seed import set_seed
from .utils.schedulers import build_optimizer, CosineLRScheduler
from .utils.checkpoint import save_checkpoint, cleanup_old_checkpoints
from .utils.logging import setup_logger


def build_model(cfg: dict) -> nn.Module:
    """Config에 따라 MoCo v2 또는 BYOL 빌드."""
    method = cfg["method"].lower()
    if method == "mocov2":
        return MoCoV2(cfg)
    elif method == "byol":
        return BYOL(cfg)
    else:
        raise ValueError(f"Unknown method: {method}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: CosineLRScheduler,
    epoch: int,
    total_epochs: int,
    device: torch.device,
    scaler=None,  # torch.amp.GradScaler 또는 None
    log_every: int = 50,
    logger=None,
    is_byol: bool = False,
) -> dict:
    """
    한 epoch 학습.
    
    Returns:
        epoch 통계 dict (avg_loss, avg_feature_std 등)
    """
    model.train()
    use_amp = scaler is not None
    
    n_steps = len(loader)
    loss_sum, std_sum = 0.0, 0.0
    t_epoch = time.time()
    
    for step, (v1, v2) in enumerate(loader):
        v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)
        
        # BYOL은 epoch-based EMA schedule
        if is_byol:
            global_progress = epoch + step / n_steps
            model.set_ema_tau(global_progress, total_epochs)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision forward
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            loss, log_dict = model(v1, v2)
        
        # Backward
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # BYOL: optimizer.step() 후 target encoder EMA 업데이트
        if is_byol:
            model._update_target()
        
        # LR schedule step
        current_lr = lr_scheduler.step()
        
        # Statistics
        loss_sum += log_dict["loss"]
        std_sum += log_dict["feature_std"]
        
        if logger is not None and (step + 1) % log_every == 0:
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            logger.info(
                f"Ep {epoch:3d} | step {step+1:4d}/{n_steps} | "
                f"loss {log_dict['loss']:.4f} | "
                f"feat_std {log_dict['feature_std']:.4f} | "
                f"lr {current_lr:.5f} | "
                f"gpu_mem {gpu_mem:.2f}GB"
            )
    
    elapsed = time.time() - t_epoch
    stats = {
        "epoch": epoch,
        "avg_loss": loss_sum / n_steps,
        "avg_feature_std": std_sum / n_steps,
        "elapsed_sec": elapsed,
        "final_lr": current_lr,
    }
    if logger is not None:
        logger.info(
            f"[Ep {epoch} done] avg_loss={stats['avg_loss']:.4f} | "
            f"avg_feat_std={stats['avg_feature_std']:.4f} | "
            f"time={elapsed:.1f}s"
        )
    return stats


def pretrain(cfg: dict, resume_from: Optional[str] = None) -> None:
    """
    Full pretraining entry point.
    
    노트북에서 호출 예:
        import yaml
        with open("configs/mocov2_r50.yaml") as f:
            cfg = yaml.safe_load(f)
        from ssl_lib.train_loop import pretrain
        pretrain(cfg)
    """
    # 시드 고정
    set_seed(cfg["training"]["seed"])
    
    # 출력 디렉토리
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = cfg["output"].get("log_file", None)
    
    # Logger
    logger = setup_logger(name=cfg["method"], log_file=log_file)
    logger.info(f"=== {cfg['method']} pretraining start ===")
    logger.info(f"Output dir: {output_dir}")
    
    # Device — CUDA_VISIBLE_DEVICES로 격리된 GPU만 보임
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Data
    loader = build_stl10_loader(cfg)
    logger.info(f"Dataset: {len(loader.dataset)} images, batch={cfg['training']['batch_size']}, "
                f"steps/epoch={len(loader)}")
    
    # Model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {cfg['method']}, trainable params = {n_params/1e6:.1f}M")
    
    # Optimizer + LR scheduler (step-based)
    optimizer = build_optimizer(model, cfg)
    total_steps = len(loader) * cfg["schedule"]["epochs"]
    warmup_steps = len(loader) * cfg["schedule"]["warmup_epochs"]
    lr_scheduler = CosineLRScheduler(
        optimizer=optimizer,
        base_lr=cfg["optimizer"]["lr"],
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    
    # AMP scaler — torch 2.x 새 API
    scaler = torch.amp.GradScaler("cuda") if cfg["training"]["amp"] else None
    
    # Resume
    start_epoch = 0
    if resume_from is not None:
        from .utils.checkpoint import load_checkpoint
        start_epoch = load_checkpoint(model, resume_from, optimizer)
        logger.info(f"Resumed from {resume_from}, starting at epoch {start_epoch}")
    
    is_byol = cfg["method"].lower() == "byol"
    
    # Training loop
    for epoch in range(start_epoch, cfg["schedule"]["epochs"]):
        stats = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            total_epochs=cfg["schedule"]["epochs"],
            device=device,
            scaler=scaler,
            log_every=cfg["training"]["log_every"],
            logger=logger,
            is_byol=is_byol,
        )
        
        # Checkpoint
        if (epoch + 1) % cfg["training"]["save_every"] == 0 \
                or epoch + 1 == cfg["schedule"]["epochs"]:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                output_dir=str(output_dir),
                extra={"config": cfg, "stats": stats},
            )
            cleanup_old_checkpoints(
                str(output_dir),
                keep_last_n=cfg["training"]["keep_last_n_ckpts"],
            )
            logger.info(f"Saved checkpoint at epoch {epoch+1}")
    
    logger.info(f"=== {cfg['method']} pretraining done ===")
