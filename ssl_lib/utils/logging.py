"""
파일 + 콘솔 통합 logger.

노트북에서 학습할 때 print만 쓰면 SSH 끊겼을 때 진행상황을 못 봄.
파일에도 항상 찍어둬야 안전.
"""
import logging
from pathlib import Path


def setup_logger(
    name: str = "ssl",
    log_file: str = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Console + file 로깅 logger 생성.
    
    Args:
        name: logger 이름.
        log_file: 파일 경로. None이면 콘솔만.
        level: 로깅 레벨.
    
    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 중복 핸들러 방지 (노트북에서 셀 재실행 시 logger가 누적되는 문제)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # 콘솔 핸들러
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # 파일 핸들러
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
