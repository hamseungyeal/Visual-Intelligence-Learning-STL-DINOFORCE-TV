"""
Random seed 고정.

재현성 평가가 있는 챌린지라 모든 seed 소스를 통제.
"""
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    모든 random source의 seed 고정.
    
    Args:
        seed: 시드 값.
        deterministic: True면 cudnn deterministic 모드. 성능 약간 손해 가능.
            재현성 평가 통과가 우선이므로 True 권장.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # 비결정적 모드 — 학습 속도 우선
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
