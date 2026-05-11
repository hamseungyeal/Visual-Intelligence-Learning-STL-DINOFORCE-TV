"""
MoCo v2 — Momentum Contrast (v2).

He et al., "Momentum Contrast for Unsupervised Visual Representation Learning", CVPR 2020.
Chen et al., "Improved Baselines with Momentum Contrastive Learning", arXiv:2003.04297.

핵심 메커니즘:
1. Query encoder (gradient 학습) + Key encoder (momentum update)
2. Queue (FIFO buffer) — 과거 key들을 negative pool로 사용
3. InfoNCE loss — query를 positive key 1개와 queue 안의 negative들과 대조

설계 결정:
- Symmetric loss 지원: 두 view를 모두 query로 써서 loss 2번 계산 (무료 ~0.5%).
- Queue size 16384 권장 (STL10 100k에선 65536 과함).
- BatchNorm: 기본 nn.BatchNorm1d 사용. multi-GPU shuffling BN은 single GPU라 불필요.
"""
import copy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .heads import ProjectionHead


class MoCoV2(nn.Module):
    """
    MoCo v2 SSL 모델.
    
    Forward 동작:
        Inputs: (view1, view2) — 같은 이미지의 두 augmented view
        Returns: (loss, log_dict) 형태로 학습용 loss + 로깅용 통계
    
    학습 후 LP 평가에는 self.encoder_q.backbone (즉 query encoder의 backbone)만 사용.
    """
    
    def __init__(self, cfg: dict):
        """
        Args:
            cfg: YAML config dict. 사용되는 키:
                cfg["backbone"]                    : backbone config
                cfg["projection"]["hidden_dim"]    : projection hidden dim
                cfg["projection"]["output_dim"]    : projection output dim (= queue feature dim)
                cfg["projection"]["num_layers"]    : projection MLP depth
                cfg["mocov2"]["queue_size"]        : queue 크기 (16384 권장)
                cfg["mocov2"]["momentum"]          : key encoder momentum (0.999 표준)
                cfg["mocov2"]["temperature"]       : InfoNCE temperature (0.2 권장)
                cfg["mocov2"]["symmetric_loss"]    : 양방향 loss 사용 (bool)
        """
        super().__init__()
        
        moco_cfg = cfg["mocov2"]
        proj_cfg = cfg["projection"]
        
        self.queue_size = moco_cfg["queue_size"]
        self.momentum = moco_cfg["momentum"]
        self.temperature = moco_cfg["temperature"]
        self.symmetric = moco_cfg.get("symmetric_loss", True)
        
        self.feature_dim = proj_cfg["output_dim"]
        
        # Query encoder = backbone + projector
        backbone_q = build_backbone(cfg)
        projector_q = ProjectionHead(
            input_dim=backbone_q.feature_dim,
            hidden_dim=proj_cfg["hidden_dim"],
            output_dim=proj_cfg["output_dim"],
            num_layers=proj_cfg["num_layers"],
            last_bn=False,  # MoCo v2 표준
        )
        self.encoder_q = nn.Sequential(backbone_q, projector_q)
        
        # Key encoder = query encoder의 deep copy (초기 가중치 동일)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        
        # Key encoder는 gradient 없이 momentum으로만 업데이트
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        
        # Queue 초기화 (정규화된 random vector로)
        # Buffer로 등록해서 state_dict에 자동 저장됨
        self.register_buffer(
            "queue",
            F.normalize(torch.randn(self.feature_dim, self.queue_size), dim=0),
        )
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @property
    def backbone(self) -> nn.Module:
        """LP 평가용 backbone 접근자. evaluate.py는 이걸 받아서 사용."""
        return self.encoder_q[0]
    
    @torch.no_grad()
    def _momentum_update(self) -> None:
        """Key encoder를 query encoder의 EMA로 업데이트."""
        for pq, pk in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            pk.data.mul_(self.momentum).add_(pq.data, alpha=1 - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """
        새로운 keys를 queue에 추가하고 가장 오래된 것을 빼냄 (FIFO).
        
        Args:
            keys: (batch, feature_dim) 정규화된 key.
        """
        batch = keys.shape[0]
        ptr = int(self.queue_ptr.item())
        
        # queue_size가 batch_size의 배수여야 깔끔하게 들어감
        assert self.queue_size % batch == 0, (
            f"queue_size ({self.queue_size}) must be divisible by "
            f"batch_size ({batch})"
        )
        
        # keys를 queue에 (transpose: queue는 (feat, queue_size))
        self.queue[:, ptr:ptr + batch] = keys.T
        ptr = (ptr + batch) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def _info_nce(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        InfoNCE loss 계산.
        
        Args:
            q: (B, D) query (정규화됨)
            k: (B, D) positive key (정규화됨)
        
        Returns:
            (loss, logits) — logits는 모니터링용
        
        Note:
            queue는 detach + clone — buffer가 forward 도중 in-place 수정되어도
            backward graph가 영향받지 않도록 안전하게 스냅샷.
        """
        # Positive logit: (B, 1) — q와 같은 이미지의 k
        l_pos = torch.einsum("nd,nd->n", q, k).unsqueeze(-1)
        # Negative logits: (B, K) — q와 queue 안의 모든 key
        # detach + clone: queue buffer의 안전한 스냅샷 (in-place 수정에 영향 없음)
        queue_snapshot = self.queue.detach().clone()
        l_neg = torch.einsum("nd,dk->nk", q, queue_snapshot)
        
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature  # (B, 1+K)
        # 0번 index가 항상 positive → cross-entropy with all-zero labels
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss, logits
    
    def _forward_one_direction(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        한 방향 forward (x_q를 query, x_k를 key로).
        
        Returns:
            (loss, q, k) — q는 모니터링용 std 계산에 사용,
            k는 backward 후 queue에 enqueue하는 데 사용 (detached).
        """
        # Query encoder forward (gradient O)
        q = F.normalize(self.encoder_q(x_q), dim=1)
        
        # Key encoder forward (gradient X)
        with torch.no_grad():
            k = F.normalize(self.encoder_k(x_k), dim=1)
        
        loss, _ = self._info_nce(q, k)
        return loss, q, k
    
    def forward(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        한 step의 학습 forward.
        
        Args:
            view1: (B, 3, H, W)
            view2: (B, 3, H, W)
        
        Returns:
            (loss, log_dict)
        
        Note:
            Queue enqueue는 loss 계산 모두 끝난 후에 한 번에 진행.
            (autograd graph가 queue buffer를 referencing 중이면 in-place modify가 문제됨)
        """
        # Momentum update — 매 step 시작 시
        self._momentum_update()
        
        if self.symmetric:
            # 양방향 loss — 두 방향 모두에서 queue는 같은 상태(이전 step 끝났을 때)를 봄
            loss_12, q1, k2 = self._forward_one_direction(view1, view2)
            loss_21, q2, k1 = self._forward_one_direction(view2, view1)
            loss = (loss_12 + loss_21) / 2
            
            # 모니터링용
            feature_std = q1.std(dim=0).mean().item()
            
            # Queue에 양방향 key 모두 추가 (loss 계산 끝난 뒤)
            # detach: gradient graph에서 분리
            self._dequeue_and_enqueue(k2.detach())
            self._dequeue_and_enqueue(k1.detach())
        else:
            loss, q1, k2 = self._forward_one_direction(view1, view2)
            feature_std = q1.std(dim=0).mean().item()
            self._dequeue_and_enqueue(k2.detach())
        
        log_dict = {
            "loss": loss.item(),
            "feature_std": feature_std,
        }
        return loss, log_dict
