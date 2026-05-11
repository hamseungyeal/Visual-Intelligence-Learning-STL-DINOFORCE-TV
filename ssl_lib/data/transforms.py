"""
Two-view augmentation pipeline (SimCLR/MoCo v2 style).

같은 이미지에 독립적으로 augmentation을 두 번 적용해 두 view를 생성.
이 두 view가 모델의 두 입력 (positive pair)이 된다.

Augmentation 구성요소:
- RandomResizedCrop  : 줌/위치 invariance
- RandomHorizontalFlip
- ColorJitter        : 색상 invariance (확률적용)
- RandomGrayscale    : 색상 의존성 줄이기
- GaussianBlur       : 디테일 의존성 줄이기
- Normalize          : ImageNet 통계 (관행)

LP 점수에 가장 큰 영향을 주는 게 augmentation 강도. 추후 튜닝의 핵심 knob.
"""
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms


# ImageNet 통계 (관행). STL10 자체 통계로 바꿔도 무방.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GaussianBlur:
    """
    torchvision 0.10+ 의 GaussianBlur를 SSL용으로 wrap.
    Sigma를 [0.1, 2.0] 범위에서 랜덤 샘플링.
    """
    
    def __init__(self, kernel_size: int = 9, sigma=(0.1, 2.0)):
        self.transform = transforms.GaussianBlur(kernel_size, sigma=sigma)
    
    def __call__(self, x):
        return self.transform(x)


class TwoViewTransform:
    """
    하나의 transform을 같은 이미지에 두 번 독립 적용.
    
    Returns:
        (view1, view2) 튜플 — 각 view는 (C, H, W) tensor.
    """
    
    def __init__(self, transform: transforms.Compose):
        self.transform = transform
    
    def __call__(self, x: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transform(x), self.transform(x)


def build_train_transform(cfg: dict) -> TwoViewTransform:
    """
    Config dict로부터 two-view augmentation 빌드.
    
    필수 cfg keys:
        cfg["data"]["image_size"]              : 출력 해상도 (예: 96)
        cfg["augmentation"]["crop_scale"]      : [min, max] (예: [0.2, 1.0])
        cfg["augmentation"]["color_jitter"]    : [b, c, s, h]
        cfg["augmentation"]["color_jitter_p"]  : ColorJitter 적용 확률
        cfg["augmentation"]["grayscale_p"]     : Grayscale 적용 확률
        cfg["augmentation"]["gaussian_blur_p"] : GaussianBlur 적용 확률
    
    Returns:
        TwoViewTransform 인스턴스.
    """
    aug = cfg["augmentation"]
    image_size = cfg["data"]["image_size"]
    
    b, c, s, h = aug["color_jitter"]
    
    single = transforms.Compose([
        transforms.RandomResizedCrop(
            size=image_size,
            scale=tuple(aug["crop_scale"]),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(b, c, s, h)],
            p=aug["color_jitter_p"],
        ),
        transforms.RandomGrayscale(p=aug["grayscale_p"]),
        transforms.RandomApply(
            [GaussianBlur(kernel_size=9)],
            p=aug["gaussian_blur_p"],
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    return TwoViewTransform(single)


def build_eval_transform(cfg: dict) -> transforms.Compose:
    """
    평가/디버깅용 transform (augmentation 없음).
    실제 LP 평가는 evaluate.py에서 별도 처리되므로 이건 내부 sanity check용.
    """
    image_size = cfg["data"]["image_size"]
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
