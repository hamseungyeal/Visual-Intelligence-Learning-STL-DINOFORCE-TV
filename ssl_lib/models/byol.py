"""
BYOL — Bootstrap Your Own Latent.

Grill et al., "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning",
NeurIPS 2020. arXiv:2006.07733.

핵심 메커니즘:
1. Online network: backbone + projector + predictor (gradient 학습)
2. Target network: backbone + projector (EMA로 업데이트)
3. Loss: predictor(online) vs projector(target)의 MSE — negative pair 없음
4. Stop-gradient: target 방향으로 gradient 안 흐름
5. Symmetric loss: 두 view를 모두 online과 target에 통과

설계 결정:
- EMA τ는 cosine schedule로 base_tau → 1.0 (BYOL 논문 표준).
- Projector에 마지막 BN 포함 (BYOL 권장 — MoCo와 다른 점).
- Predictor LR 분리는 build_optimizer 단계에서 처리.
"""
import copy
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .heads import ProjectionHead, Predictor


class BYOL(nn.Module):
    """
    BYOL SSL 모델.
    
    Forward 동작:
        Inputs: (view1, view2)
        Returns: (loss, log_dict)
    
    학습 후 LP 평가에는 self.online_encoder의 backbone만 사용.
    """
    
    def __init__(self, cfg: dict):
        """
        Args:
            cfg: YAML config dict. 사용되는 키:
                cfg["backbone"]
                cfg["projection"]["hidden_dim"]   : 4096 권장 (BYOL 논문)
                cfg["projection"]["output_dim"]   : 256 권장
                cfg["projection"]["num_layers"]   : 2
                cfg["predictor"]["hidden_dim"]    : 4096
                cfg["predictor"]["output_dim"]    : 256 (= projection output_dim)
                cfg["byol"]["ema_tau_base"]       : 초기 EMA 계수 (0.996 권장)
                cfg["byol"]["ema_schedule"]       : "cosine" | "constant"
                cfg["byol"]["symmetric_loss"]     : True 권장
        """
        super().__init__()
        
        byol_cfg = cfg["byol"]
        proj_cfg = cfg["projection"]
        pred_cfg = cfg["predictor"]
        
        self.ema_tau_base = byol_cfg["ema_tau_base"]
        self.ema_schedule = byol_cfg.get("ema_schedule", "cosine")
        self.symmetric = byol_cfg.get("symmetric_loss", True)
        
        # Online: backbone + projector + predictor
        backbone = build_backbone(cfg)
        projector = ProjectionHead(
            input_dim=backbone.feature_dim,
            hidden_dim=proj_cfg["hidden_dim"],
            output_dim=proj_cfg["output_dim"],
            num_layers=proj_cfg["num_layers"],
            last_bn=True,  # BYOL 표준 (MoCo와 다른 점)
        )
        self.online_encoder = nn.Sequential(backbone, projector)
        
        self.predictor = Predictor(
            input_dim=pred_cfg["input_dim"],
            hidden_dim=pred_cfg["hidden_dim"],
            output_dim=pred_cfg["output_dim"],
        )
        
        # Target = online의 deep copy (predictor 제외)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        
        # 현재 EMA τ (학습 진행에 따라 cosine으로 업데이트됨)
        self._current_tau = self.ema_tau_base
    
    @property
    def backbone(self) -> nn.Module:
        """LP 평가용 backbone 접근자."""
        return self.online_encoder[0]
    
    def set_ema_tau(self, current_step: int, total_steps: int) -> None:
        """
        EMA τ schedule. BYOL 논문 표준: cosine schedule로 base → 1.0.
        
        Args:
            current_step: 현재까지 진행된 step (또는 epoch)
            total_steps: 전체 학습 step (또는 epoch)
        """
        if self.ema_schedule == "constant":
            self._current_tau = self.ema_tau_base
        elif self.ema_schedule == "cosine":
            # τ = 1 - (1 - τ_base) * (cos(π * k / K) + 1) / 2
            progress = current_step / max(total_steps, 1)
            self._current_tau = 1 - (1 - self.ema_tau_base) * (
                math.cos(math.pi * progress) + 1
            ) / 2
        else:
            raise ValueError(f"Unknown ema_schedule: {self.ema_schedule}")
    
    @torch.no_grad()
    def _update_target(self) -> None:
        """Target encoder를 online의 EMA로 업데이트."""
        for po, pt in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            pt.data.mul_(self._current_tau).add_(po.data, alpha=1 - self._current_tau)
    
    @staticmethod
    def _byol_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        BYOL loss = 2 - 2 * cos_sim(p, z).
        L2 정규화된 두 벡터의 MSE와 동등 (BYOL 논문 식 (2)).
        
        Args:
            p: (B, D) predictor 출력 (online)
            z: (B, D) projector 출력 (target, stop-gradient 적용 가정)
        """
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return 2 - 2 * (p * z).sum(dim=1).mean()
    
    def forward(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        한 step의 학습 forward.
        
        Args:
            view1, view2: (B, 3, H, W)
        
        Returns:
            (loss, log_dict)
        """
        # Online forward (gradient O) → predictor 출력
        p1 = self.predictor(self.online_encoder(view1))
        
        # Target forward (gradient X)
        with torch.no_grad():
            z2 = self.target_encoder(view2)
        
        if self.symmetric:
            # 양방향
            p2 = self.predictor(self.online_encoder(view2))
            with torch.no_grad():
                z1 = self.target_encoder(view1)
            loss = (self._byol_loss(p1, z2) + self._byol_loss(p2, z1)) / 2
            # 모니터링용 — predictor 출력의 차원별 std
            feature_std = F.normalize(p1, dim=1).std(dim=0).mean().item()
        else:
            loss = self._byol_loss(p1, z2)
            feature_std = F.normalize(p1, dim=1).std(dim=0).mean().item()
        
        # Target update — loss 계산 후, optimizer.step() 후에 호출하는 게 정석이지만
        # 실용적으로는 여기서 해도 차이 미미. 학습 루프에서 명시적 호출 권장.
        # 여기서는 호출하지 않고 학습 루프에서 _update_target()을 호출하도록 설계.
        
        log_dict = {
            "loss": loss.item(),
            "feature_std": feature_std,
            "ema_tau": self._current_tau,
        }
        return loss, log_dict
