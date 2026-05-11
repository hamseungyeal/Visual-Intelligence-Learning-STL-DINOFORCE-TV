"""
STL10 unlabeled 데이터 로드 및 two-view augmentation 검증.

프로젝트 루트에서 실행:
    python scripts/00_data_check.py

출력:
    - 콘솔: dataset 크기, view shape, batch shape
    - logs/data_check.png: 4쌍의 view 시각화 (터미널 환경이므로 파일로 저장)
"""
import sys
from pathlib import Path

# ssl_lib를 pip install -e . 없이도 찾을 수 있도록 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
import matplotlib
matplotlib.use('Agg')  # 디스플레이 없는 터미널 환경용
import matplotlib.pyplot as plt
from pathlib import Path

from ssl_lib.data import build_stl10_loader, build_train_transform
from ssl_lib.data.stl10_unlabeled import STL10UnlabeledDataset
from ssl_lib.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def denorm(x):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def main():
    # ── Config 로드 ───────────────────────────────────────────────────────────
    with open('configs/mocov2_r50.yaml') as f:
        cfg = yaml.safe_load(f)

    # 데이터 검증용: 작은 batch / worker 수
    cfg['training']['batch_size'] = 4
    cfg['data']['num_workers'] = 0
    cfg['data']['persistent_workers'] = False

    print(yaml.dump(cfg['data'], allow_unicode=True))

    # ── 데이터셋 빌드 (처음 실행 시 STL10 ~2.6GB 다운로드) ─────────────────
    transform = build_train_transform(cfg)
    dataset = STL10UnlabeledDataset(
        root=cfg['data']['root'],
        transform=transform,
        download=True,
    )
    print(f'Dataset size: {len(dataset)}')

    v1, v2 = dataset[0]
    print(f'view1 shape: {v1.shape}, dtype: {v1.dtype}, '
          f'range: [{v1.min():.3f}, {v1.max():.3f}]')
    print(f'view2 shape: {v2.shape}, dtype: {v2.dtype}')

    # ── DataLoader 빌드 ───────────────────────────────────────────────────────
    loader = build_stl10_loader(cfg)
    v1_batch, v2_batch = next(iter(loader))
    print(f'view1 batch: {v1_batch.shape}')
    print(f'view2 batch: {v2_batch.shape}')

    # ── 시각화 (파일 저장) ────────────────────────────────────────────────────
    Path('logs').mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(4):
        axes[0, i].imshow(denorm(v1_batch[i]).permute(1, 2, 0))
        axes[0, i].set_title(f'view1 #{i}')
        axes[0, i].axis('off')
        axes[1, i].imshow(denorm(v2_batch[i]).permute(1, 2, 0))
        axes[1, i].set_title(f'view2 #{i}')
        axes[1, i].axis('off')
    plt.tight_layout()
    save_path = 'logs/data_check.png'
    plt.savefig(save_path, dpi=100)
    print(f'시각화 저장: {save_path}')

    # ── 체크리스트 ────────────────────────────────────────────────────────────
    assert len(dataset) == 100_000, f'Dataset size 오류: {len(dataset)}'
    assert v1.shape == (3, 96, 96), f'Shape 오류: {v1.shape}'
    assert not torch.allclose(v1, v2), 'view1 == view2: augmentation 미작동!'
    print('\n[OK] dataset size = 100,000')
    print('[OK] view shape = (3, 96, 96)')
    print('[OK] 두 view가 서로 다름 (augmentation 정상)')
    print('[OK] 모든 체크 통과')


if __name__ == '__main__':
    main()
