#!/bin/bash
# ==========================================================
# MoCo v2 — GPU 0에서 백그라운드 학습
# 사용법:  bash scripts/run_mocov2.sh
# ==========================================================

set -e

mkdir -p logs outputs

CUDA_VISIBLE_DEVICES=0 nohup python3 scripts/pretrain.py \
    --config configs/mocov2_r50.yaml \
    --seed 42 \
    > logs/mocov2_seed42.stdout 2>&1 &

echo "MoCo v2 started on GPU 0 (PID: $!)"
echo "Tail log: tail -f logs/mocov2_seed42.log"
echo "Stop:    kill $!"
