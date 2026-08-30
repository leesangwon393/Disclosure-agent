"""연산 장치 선택을 한 곳에서 정한다.

왜 한 곳인가: 개발·실행 환경이 **Apple Silicon 맥(M5 Pro)** 이다. CUDA 도 NVIDIA GPU 도
없다. 그런데 라이브러리마다 자동 감지 방식이 달라서(특히 FlagEmbedding 은 버전에 따라
cuda 를 기본값으로 잡는다) 각자에게 맡기면 어딘가에서 조용히 CPU 로 떨어지거나
`AssertionError: Torch not compiled with CUDA` 로 죽는다.

우선순위: **cuda(있으면) > mps(애플 GPU) > cpu**
서버 배포 시 NVIDIA 장비를 쓰게 되면 코드 수정 없이 cuda 로 올라간다.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_cached: str | None = None


def pick_device(explicit: str | None = None) -> str:
    """`DISCLOSURE_DEVICE` 환경변수 > 인자 > 자동감지 순."""
    global _cached
    env = os.environ.get("DISCLOSURE_DEVICE")
    if explicit:
        return explicit
    if env:
        return env
    if _cached:
        return _cached

    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
    except Exception as e:  # noqa: BLE001
        logger.warning("[DEVICE] torch 확인 실패(%s) — cpu 로 진행", type(e).__name__)

    _cached = device
    logger.info("[DEVICE] %s", device)
    return device


def use_fp16_for(device: str) -> bool:
    """fp16 은 GPU 에서만 의미가 있다. CPU 에서 켜면 오히려 느리거나 지원되지 않는다."""
    return device in ("cuda", "mps")
