"""ColBERT late-interaction reranker (BGE-M3 multi-vector 출력 사용).

왜 이게 우리 도메인에 맞나
--------------------------
Dense 는 조각 전체를 **벡터 하나로 뭉갠다**. 그래서
  - "계약금액 3,000억"과 "계약금액 5,000억"의 벡터가 거의 같고
  - 원본 공시와 정정공시(텍스트 몇 글자 차이)를 사실상 구분하지 못하며
  - 표를 한 덩어리로 평균내 버린다
EACL 2026 벤치마크는 금융문서 검색 실패의 **73%가 표 구조 불일치**라고 보고한다.

late interaction 은 **토큰마다 벡터를 유지**하고, 질의 토큰 하나하나를 문서 토큰
전체와 대조해 최대값을 취한다(MaxSim). 즉 "3,000억"이라는 질의 토큰이 문서의
"3,000억" 토큰과 직접 만난다. 뭉개지지 않는다.

비용
----
토큰마다 벡터라 저장이 크다. 그래서 **전체 인덱스가 아니라 상위 후보 재정렬용**
으로만 쓴다(BM25/dense/sparse 로 50개 뽑고 그 50개만 late interaction).
cross-encoder 리랭커와 달리 문서 쪽 인코딩이 이미 끝나 있어 훨씬 싸다.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from disclosure_rag.chunking.chunk_schema import ChunkSchema

logger = logging.getLogger(__name__)


def maxsim(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    """ColBERT MaxSim: 질의 토큰마다 문서 토큰 중 최대 유사도를 찾아 합산."""
    if query_vecs.size == 0 or doc_vecs.size == 0:
        return 0.0
    sim = query_vecs @ doc_vecs.T          # (q_len, d_len)
    return float(sim.max(axis=1).sum() / query_vecs.shape[0])


class ColbertReranker:
    """vector_lookup(chunk_id) -> (n_tokens, dim) 배열 또는 None.

    저장 전략을 주입 가능하게 두었다(전체 RAM 적재 / npz mmap / 후보만 lazy 로드).
    벡터가 없는 후보는 fusion 점수를 유지하고 뒤로 보낸다 — 조용히 버리지 않는다.
    """

    name = "colbert"

    def __init__(self, query_encoder: Callable[[str], np.ndarray],
                 vector_lookup: Callable[[str], "np.ndarray | None"]):
        self._encode = query_encoder
        self._lookup = vector_lookup

    def rerank(
        self, query: str, candidates: list[tuple[ChunkSchema, float]], *, top_k: int = 10,
    ) -> list[tuple[ChunkSchema, float]]:
        if not candidates:
            return []
        qv = self._encode(query)
        scored: list[tuple[ChunkSchema, float]] = []
        missing = 0
        for chunk, fused_score in candidates:
            dv = self._lookup(chunk.chunk_id)
            if dv is None:
                missing += 1
                # 점수 체계가 다르므로 -1 오프셋으로 뒤에 배치하되 제거하지는 않는다
                scored.append((chunk, fused_score - 1.0))
                continue
            scored.append((chunk, maxsim(qv, dv)))
        if missing:
            logger.warning("[COLBERT] 후보 %d/%d 개의 벡터가 없어 fusion 점수로 대체",
                           missing, len(candidates))
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:top_k]
