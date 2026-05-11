"""
Projection Head & Predictor MLPs.

SSL의 backbone(인코더) 뒤에 붙는 작은 MLP들.
- ProjectionHead: 모든 SSL 방법론이 공유. 학습 종료 후 버려짐 (LP에서 사용 안 함).
- Predictor:      BYOL/SimSiam에서 사용. Online 네트워크에만 붙는 비대칭 구조.

설계 선택:
- 마지막 layer에 BN을 둘지 말지가 LP 성능에 영향 (특히 BYOL).
- 일반적으로 SimCLR/MoCo: 마지막 BN 없음, BYOL: 마지막 BN 있음.
- 여기서는 `last_bn` 인자로 조정 가능.
"""
from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    범용 MLP. Linear -> BN -> ReLU 블록을 반복하고 마지막 Linear.
    
    예: hidden_dims=[2048, 2048], output_dim=128
        → Linear(in, 2048) -> BN -> ReLU -> Linear(2048, 2048) -> BN -> ReLU -> Linear(2048, 128)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        use_bn: bool = True,
        last_bn: bool = False,
    ):
        """
        Args:
            input_dim: backbone feature dim (예: 2048 for ResNet-50)
            hidden_dims: 중간 hidden layer 차원 리스트. [] 면 단일 Linear.
            output_dim: 최종 projection 차원.
            use_bn: 중간 layer에 BatchNorm 사용.
            last_bn: 마지막 Linear 뒤에도 BN. BYOL 권장 설정.
        """
        super().__init__()
        
        layers = []
        prev = input_dim
        
        # Hidden layers
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        
        # Final layer
        layers.append(nn.Linear(prev, output_dim, bias=not last_bn))
        if last_bn:
            layers.append(nn.BatchNorm1d(output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectionHead(MLP):
    """
    Backbone feature를 projection space로 매핑하는 MLP.
    SimCLR/MoCo v2 표준: 2-layer (hidden -> output), 마지막 BN 없음.
    BYOL 표준: 2-layer, 마지막 BN 있음.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        num_layers: int = 2,
        last_bn: bool = False,
    ):
        """
        Args:
            input_dim: backbone feature dim.
            hidden_dim: hidden layer 차원.
            output_dim: projection 차원.
            num_layers: 총 Linear 개수 (2면 [Linear, Linear]).
            last_bn: 마지막 BN 사용 (BYOL 권장 True).
        """
        # num_layers - 1 개의 hidden layer
        hidden_dims = [hidden_dim] * (num_layers - 1)
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            use_bn=True,
            last_bn=last_bn,
        )


class Predictor(MLP):
    """
    BYOL/SimSiam의 predictor MLP.
    Online network에만 붙는다. 비대칭 구조로 collapse 방지에 기여.
    표준: 2-layer, 마지막 BN 없음.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 4096,
        output_dim: int = 256,
    ):
        """
        Args:
            input_dim: projection 차원 (projector output과 동일).
            hidden_dim: hidden 차원. BYOL 논문은 4096 사용.
            output_dim: 출력 차원 (= input_dim과 같아야 target과 비교 가능).
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dims=[hidden_dim],
            output_dim=output_dim,
            use_bn=True,
            last_bn=False,
        )
