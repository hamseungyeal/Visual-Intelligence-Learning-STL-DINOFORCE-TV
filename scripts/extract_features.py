"""
SSL pretrain 완료 후 STL10 / CIFAR10 feature 추출 스크립트.

사용법:
    python scripts/extract_features.py \
        --backbone outputs/mocov2_r50_seed42/backbone_ep200.pth \
        --config   configs/mocov2_r50.yaml \
        --output-dir features/mocov2_ep200

결과물:
    features/mocov2_ep200/
    ├── stl10_train_features.npy   (5000,  2048)
    ├── stl10_train_labels.npy     (5000,)
    ├── stl10_test_features.npy    (8000,  2048)
    ├── stl10_test_labels.npy      (8000,)
    ├── cifar10_train_features.npy (50000, 2048)
    ├── cifar10_train_labels.npy   (50000,)
    ├── cifar10_test_features.npy  (10000, 2048)
    └── cifar10_test_labels.npy    (10000,)

추출 완료 후 evaluate.py 실행 명령이 자동으로 출력됩니다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ssl_lib.models.backbone import ResNetBackbone

# pretraining과 동일한 normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGE_SIZE = 96  # pretraining 해상도에 맞춰 통일 (CIFAR10도 resize)


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@torch.no_grad()
def extract_features(backbone, loader, device):
    backbone.eval()
    feats_list, labels_list = [], []
    total = len(loader.dataset)
    done = 0
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        feats = backbone(imgs).cpu()
        feats_list.append(feats)
        labels_list.append(lbls)
        done += lbls.size(0)
        print(f'  {done}/{total}', end='\r', flush=True)
    print()
    return torch.cat(feats_list).numpy(), torch.cat(labels_list).numpy()


def main():
    parser = argparse.ArgumentParser(description='SSL backbone feature 추출')
    parser.add_argument('--backbone', required=True,
                        help='backbone_ep*.pth 경로')
    parser.add_argument('--config', default=None,
                        help='학습 config YAML (backbone 구조 확인용). '
                             '없으면 resnet50 + small_image=True 기본값 사용.')
    parser.add_argument('--output-dir', default='features',
                        help='feature 저장 디렉토리 (default: features/)')
    parser.add_argument('--data-dir', default='./data',
                        help='STL10 / CIFAR10 데이터 루트 (default: ./data)')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # backbone 구조 결정
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        bb_cfg = cfg['backbone']
        backbone = ResNetBackbone(
            name=bb_cfg['name'],
            small_image=bb_cfg.get('small_image', True),
            zero_init_residual=bb_cfg.get('zero_init_residual', True),
            gradient_checkpoint=False,  # 추론 시 불필요
        )
    else:
        backbone = ResNetBackbone(name='resnet50', small_image=True, gradient_checkpoint=False)

    ckpt = torch.load(args.backbone, map_location='cpu')
    backbone.load_state_dict(ckpt['backbone_state_dict'])
    backbone = backbone.to(device)
    print(f'backbone loaded  (epoch={ckpt.get("epoch", "?")}, feature_dim={backbone.feature_dim})')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tf = build_transform()

    dataset_builders = [
        ('stl10_train',   lambda: datasets.STL10(args.data_dir,  split='train',  transform=tf, download=True)),
        ('stl10_test',    lambda: datasets.STL10(args.data_dir,  split='test',   transform=tf, download=True)),
        ('cifar10_train', lambda: datasets.CIFAR10(args.data_dir, train=True,     transform=tf, download=True)),
        ('cifar10_test',  lambda: datasets.CIFAR10(args.data_dir, train=False,    transform=tf, download=True)),
    ]

    for name, build_ds in dataset_builders:
        ds = build_ds()
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            drop_last=False,
        )
        print(f'\n[{name}] {len(ds):,} samples...')
        feats, labels = extract_features(backbone, loader, device)
        np.save(output_dir / f'{name}_features.npy', feats)
        np.save(output_dir / f'{name}_labels.npy', labels)
        print(f'  saved: {feats.shape}  →  {output_dir}/{name}_*.npy')

    d = output_dir
    print('\n=== 추출 완료. evaluate.py 실행 명령: ===')
    print(f'python evaluate.py \\')
    print(f'  --stl10-train-features   {d}/stl10_train_features.npy \\')
    print(f'  --stl10-train-labels     {d}/stl10_train_labels.npy \\')
    print(f'  --stl10-test-features    {d}/stl10_test_features.npy \\')
    print(f'  --stl10-test-labels      {d}/stl10_test_labels.npy \\')
    print(f'  --cifar10-train-features {d}/cifar10_train_features.npy \\')
    print(f'  --cifar10-train-labels   {d}/cifar10_train_labels.npy \\')
    print(f'  --cifar10-test-features  {d}/cifar10_test_features.npy \\')
    print(f'  --cifar10-test-labels    {d}/cifar10_test_labels.npy')


if __name__ == '__main__':
    main()
