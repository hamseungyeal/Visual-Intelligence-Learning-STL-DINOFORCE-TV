"""
SSL Project — editable install용 setup.py

서버에서 프로젝트 루트에 진입한 뒤 한 번만 실행:
    pip install -e .

이렇게 하면 ssl_lib가 site-packages에 등록되어
어디서든 `from ssl_lib...` 형태로 import 가능.
파일 수정사항은 즉시 반영됨 (editable mode).
"""
from setuptools import setup, find_packages

setup(
    name="ssl_lib",
    version="0.1.0",
    description="Self-Supervised Learning library for STL10 challenge",
    packages=find_packages(exclude=["notebooks", "scripts", "configs", "outputs", "logs"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy",
        "pyyaml",
        "tqdm",
    ],
)
