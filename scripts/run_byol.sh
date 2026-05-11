#!/bin/bash
# ==========================================================
# BYOL — GPU 1에서 백그라운드 학습
# 사용법:  bash scripts/run_byol.sh
# ==========================================================

set -e

mkdir -p logs outputs

CUDA_VISIBLE_DEVICES=1 nohup python3 scripts/pretrain.py \
    --config configs/byol_r50.yaml \
    --seed 42 \
    > logs/byol_seed42.stdout 2>&1 &

echo "BYOL started on GPU 1 (PID: $!)"
echo "Tail log: tail -f logs/byol_seed42.log"
echo "Stop:    kill $!"
