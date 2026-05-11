# SSL Project — STL10 unlabeled SSL pretraining

MoCo v2와 BYOL을 듀얼 베이스라인으로 학습하여 STL10/CIFAR10 Linear Probing 성능 비교.

## 디렉토리 구조

```
ssl_project/
├── ssl_lib/             # 공유 라이브러리 (editable install)
│   ├── data/            # STL10 unlabeled, two-view augmentation
│   ├── models/          # backbone, heads, MoCoV2, BYOL
│   ├── utils/           # seed, schedulers, checkpoint, logging
│   └── train_loop.py    # 학습 루프
├── configs/             # YAML configs
├── notebooks/           # 디버깅 + 학습 노트북
├── scripts/             # CLI 진입점 + bash 실행 스크립트
├── outputs/             # 학습 산출물 (체크포인트)
└── logs/                # 학습 로그
```

## 1. 셋업 (한 번만)

```bash
cd ssl_project
pip install -r requirements.txt
pip install -e .          # ssl_lib을 editable mode로 설치
```

## 2. 디버깅 (CPU/GPU 어디서든)

순서대로 실행해서 환경 검증:

1. `notebooks/00_data_check.ipynb` — STL10 다운로드 + augmentation 확인
2. `notebooks/01_backbone_test.ipynb` — backbone forward + feature dim 검증

## 3. 학습 — 두 가지 방법

### 방법 A: 노트북에서 (개발/디버깅 시)

서로 다른 GPU에 노트북 서버 두 개 띄우기:

```bash
# 터미널 1
CUDA_VISIBLE_DEVICES=0 jupyter lab --port 8888

# 터미널 2
CUDA_VISIBLE_DEVICES=1 jupyter lab --port 8889
```

- 8888에서 `notebooks/train_mocov2.ipynb` 실행
- 8889에서 `notebooks/train_byol.ipynb` 실행

### 방법 B: 백그라운드 (장시간 안정 학습)

```bash
# MoCo v2 → GPU 0
bash scripts/run_mocov2.sh

# BYOL → GPU 1
bash scripts/run_byol.sh
```

진행 상황 확인:
```bash
tail -f logs/mocov2_seed42.log
tail -f logs/byol_seed42.log
nvidia-smi
```

학습 중단:
```bash
# PID는 run_*.sh 실행 시 출력됨
kill <PID>
```

## 4. Resume

학습이 중단되면:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/pretrain.py \
    --config configs/mocov2_r50.yaml \
    --resume outputs/mocov2_r50_seed42/ckpt_ep200.pth
```

## 5. 산출물

```
outputs/mocov2_r50_seed42/
├── ckpt_ep{10,20,...,400}.pth        # 전체 학습 state (resume용)
├── backbone_ep{10,20,...,400}.pth    # backbone 가중치만 (LP 평가용)
└── (최근 3개만 유지, 오래된 건 자동 정리)
```

LP 평가에는 `backbone_ep*.pth`만 있으면 됨. evaluate.py가 이 파일의 `backbone_state_dict`를 로드해서 사용.

## 주요 설계 결정

- **공유 라이브러리 + 별도 노트북**: backbone/data/aug 코드가 단일 진실 소스 → 공정 비교 보장.
- **CUDA_VISIBLE_DEVICES로 GPU 격리**: 코드 안에서 device 인자 안 받음. 실수 방지.
- **AMP 필수**: 24GB GPU + 두 인코더 동시 메모리 → mixed precision 없으면 batch size 반토막.
- **MoCo queue 16384**: STL10 100k에 65536은 과함. 16384가 적절.
- **두 모델 동일 seed**: 같은 batch order, 같은 augmentation 시퀀스로 공정 비교.
- **checkpoint 매 epoch 저장 + 최근 N개만 유지**: 디스크 절약 + 중단 시 복구 가능.

## 다음 단계

evaluate.py를 받으면:
1. `forward()` 인터페이스 확인 → backbone wrapper 조정 필요시 수정
2. 입력 해상도 확인 → augmentation crop size 조정
3. LP 점수 확인 → 어느 모델/설정이 좋았는지 결정
