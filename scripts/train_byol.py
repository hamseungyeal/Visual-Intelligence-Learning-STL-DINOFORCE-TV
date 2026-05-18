"""
BYOL 학습 스크립트 (GPU 1 전용).

프로젝트 루트에서 실행:
    CUDA_VISIBLE_DEVICES=1 python scripts/train_byol.py
    CUDA_VISIBLE_DEVICES=1 python scripts/train_byol.py --epochs 5    # sanity check
    CUDA_VISIBLE_DEVICES=1 python scripts/train_byol.py --resume outputs/byol_r50_seed42/ckpt_ep200.pth

백그라운드 실행 (nohup):
    bash scripts/run_byol.sh

예상 소요 시간: 400 epoch ≈ 20~24시간
로그 파일: logs/byol_seed42.log
체크포인트: outputs/byol_r50_seed42/

모니터링: logs에서 feature_std 확인.
    0.05 미만으로 떨어지면 representation collapse 신호.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import torch
import yaml

from ssl_lib.train_loop import pretrain


def main():
    parser = argparse.ArgumentParser(description='BYOL pretraining on STL10')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume할 checkpoint 경로')
    parser.add_argument('--epochs', type=int, default=None,
                        help='학습 epoch 수 override (미지정 시 config 값 사용)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size override (OOM 시 256으로 줄임)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='DataLoader worker 수 override (Colab 등에서 2 권장)')
    args = parser.parse_args()

    assert torch.cuda.is_available(), 'GPU 사용 불가! CUDA 환경을 확인하세요.'
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    with open('configs/byol_r50.yaml') as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg['schedule']['epochs'] = args.epochs
    if args.batch_size is not None:
        cfg['training']['batch_size'] = args.batch_size
    if args.num_workers is not None:
        cfg['data']['num_workers'] = args.num_workers
        cfg['data']['persistent_workers'] = args.num_workers > 0

    pretrain(cfg, resume_from=args.resume)


if __name__ == '__main__':
    main()
