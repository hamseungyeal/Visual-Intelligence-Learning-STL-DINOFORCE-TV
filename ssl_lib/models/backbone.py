"""
Backbone (encoder) 모듈.

torchvision의 ResNet을 wrap해서 SSL용으로 만든다.
- 마지막 fc layer 제거 (avgpool 출력만 반환)
- small_image=True일 때 첫 conv를 3x3/stride1, maxpool 제거
  (CIFAR/STL처럼 작은 이미지용. evaluate.py가 원본 ResNet을 가정하면 False로.)

MoCo v2와 BYOL이 모두 같은 backbone 클래스를 import해서 공정 비교.
"""
from typing import Literal

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50


BackboneName = Literal["resnet18", "resnet34", "resnet50"]


class ResNetBackbone(nn.Module):
    """
    ResNet backbone wrapper.
    
    Attributes:
        feature_dim: avgpool 출력 차원 (resnet18/34 → 512, resnet50 → 2048).
    """
    
    def __init__(
        self,
        name: BackboneName = "resnet50",
        small_image: bool = True,
        zero_init_residual: bool = True,
    ):
        """
        Args:
            name: "resnet18" | "resnet34" | "resnet50"
            small_image: True면 첫 conv를 3x3/stride1, maxpool 제거.
                STL10(96) / CIFAR(32)처럼 작은 이미지일 때 권장.
                ⚠️ evaluate.py가 원본 ResNet 구조를 가정한다면 False로 둬야 함.
            zero_init_residual: ResNet의 마지막 BN을 0으로 초기화 (학습 안정성).
                He et al. (2018) "Bag of Tricks"에서 권장.
        """
        super().__init__()
        
        builders = {
            "resnet18": resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50,
        }
        if name not in builders:
            raise ValueError(f"Unknown backbone: {name}")
        
        # weights=None 명시 — pretrained weight 절대 금지 (챌린지 규칙)
        net = builders[name](
            weights=None,
            zero_init_residual=zero_init_residual,
        )
        
        # 작은 이미지 대응: 첫 conv 수정 + maxpool 제거
        if small_image:
            net.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            net.maxpool = nn.Identity()
        
        # 마지막 fc 제거 — feature dim 보존
        self.feature_dim = net.fc.in_features
        net.fc = nn.Identity()
        
        self.net = net
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) 입력.
        Returns:
            (B, feature_dim) feature. 이게 LP 평가에 쓰이는 그것.
        """
        return self.net(x)


def build_backbone(cfg: dict) -> ResNetBackbone:
    """
    Config dict로부터 backbone 빌드.
    
    필수 cfg keys:
        cfg["backbone"]["name"]        : "resnet18"|"resnet34"|"resnet50"
        cfg["backbone"]["small_image"] : bool
    """
    bb_cfg = cfg["backbone"]
    return ResNetBackbone(
        name=bb_cfg["name"],
        small_image=bb_cfg.get("small_image", True),
        zero_init_residual=bb_cfg.get("zero_init_residual", True),
    )
