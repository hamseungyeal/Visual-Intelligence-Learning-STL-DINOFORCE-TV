"""
ResNet backbone forward pass 검증.

프로젝트 루트에서 실행:
    python scripts/01_backbone_test.py

검증 항목:
    - ResNet-18 feature_dim = 512
    - ResNet-50 feature_dim = 2048
    - 96x96 / 32x32 / 224x224 모든 해상도에서 forward OK
    - small_image=True vs False 구조 비교
    - forward 속도 측정 (batch=64)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import torch

from ssl_lib.models.backbone import ResNetBackbone


def measure_speed(model, device, name, batch_size=64, image_size=96, runs=10):
    model.eval()
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    for _ in range(3):  # warmup
        with torch.no_grad():
            _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(runs):
        with torch.no_grad():
            _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    ms_per_batch = (time.time() - t) / runs * 1000
    print(f'{name}: {ms_per_batch:.2f} ms/batch (batch={batch_size})')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # ── ResNet-18 검증 ────────────────────────────────────────────────────────
    bb18 = ResNetBackbone(name='resnet18', small_image=True).to(device)
    print(f'\nResNet-18 small_image=True: feature_dim={bb18.feature_dim}')
    n_params = sum(p.numel() for p in bb18.parameters())
    print(f'params: {n_params / 1e6:.2f}M')

    feat96 = bb18(torch.randn(2, 3, 96, 96, device=device))
    print(f'96x96  input → feature: {feat96.shape}')

    feat32 = bb18(torch.randn(2, 3, 32, 32, device=device))
    print(f'32x32  input → feature: {feat32.shape}')

    feat224 = bb18(torch.randn(2, 3, 224, 224, device=device))
    print(f'224x224 input → feature: {feat224.shape}')

    # ── ResNet-50 검증 ────────────────────────────────────────────────────────
    bb50 = ResNetBackbone(name='resnet50', small_image=True).to(device)
    print(f'\nResNet-50 small_image=True: feature_dim={bb50.feature_dim}')
    n_params = sum(p.numel() for p in bb50.parameters())
    print(f'params: {n_params / 1e6:.2f}M')

    feat = bb50(torch.randn(2, 3, 96, 96, device=device))
    print(f'96x96  input → feature: {feat.shape}')

    # ── small_image=False vs True 구조 비교 ──────────────────────────────────
    bb_full = ResNetBackbone(name='resnet50', small_image=False).to(device)
    bb_small = ResNetBackbone(name='resnet50', small_image=True).to(device)

    print('\nsmall_image=False (original ResNet):')
    print(f'  conv1:   {bb_full.net.conv1}')
    print(f'  maxpool: {bb_full.net.maxpool}')
    print('\nsmall_image=True:')
    print(f'  conv1:   {bb_small.net.conv1}')
    print(f'  maxpool: {bb_small.net.maxpool}')

    # ── Forward 속도 측정 ─────────────────────────────────────────────────────
    print('\n--- Forward 시간 측정 (batch=64, 96x96) ---')
    measure_speed(bb18, device, 'R-18')
    measure_speed(bb50, device, 'R-50')

    # ── 체크리스트 ────────────────────────────────────────────────────────────
    assert bb18.feature_dim == 512, f'R-18 feature_dim 오류: {bb18.feature_dim}'
    assert bb50.feature_dim == 2048, f'R-50 feature_dim 오류: {bb50.feature_dim}'
    assert feat96.shape == (2, 512), f'R-18 출력 shape 오류: {feat96.shape}'
    assert feat.shape == (2, 2048), f'R-50 출력 shape 오류: {feat.shape}'
    print('\n[OK] ResNet-18 feature_dim = 512')
    print('[OK] ResNet-50 feature_dim = 2048')
    print('[OK] 96x96 / 32x32 / 224x224 모든 해상도 forward 통과')
    print('[OK] 모든 체크 통과')


if __name__ == '__main__':
    main()
