"""Fusion (§53): BM25 / Dense / Sparse 의 결과를 합친다.

=== KIM 브랜치 변경점 (2026-08-23) ===
Stage 4 의 승자는 `normalized_weighted` 였는데 **저장소에 구현이 없었고**
프로덕션(`hybrid_retriever.py`)은 RRF 를 쓰고 있었다.

    RRF (프로덕션)                R@10 = 0.860
    normalized_weighted (승자)    R@10 = 0.900

RRF 는 **점수를 버리고 순위만** 쓴다. BM25 가 "확실한 1등"이라 한 것과 "간신히
1등"이라 한 것을 똑같이 취급하므로, Dense 가 애매하게 올린 후보가 BM25 의 확신
있는 답을 top-5 에서 밀어낸다. 실측으로 하이브리드가 BM25 단독에 지는 구간이
정확히 k=5 였다(R@5 0.661 vs 0.706).

공시 도메인은 Dense 에 구조적으로 불리하다(표 조각의 숫자 비중 20%, 정정공시 43%가
원본과 거의 동일 텍스트, 법정 통제어휘, leaf 의 절반 이상이 표). 그래서 기본
가중치를 **BM25 쪽으로 기울여** 둔다. 최종 값은 실제 규모에서 sweep 으로 정할 것.
"""

from __future__ import annotations

from disclosure_rag.chunking.chunk_schema import ChunkSchema

RankedList = list[tuple[ChunkSchema, float]]

# 공시 도메인 기본값. BM25 우위는 문헌(EACL 2026: 금융문서에서 BM25 0.644 > dense 0.587)
# 과 우리 실측 양쪽에서 지지된다. sweep 으로 재조정할 것.
DEFAULT_WEIGHTS = {"bm25": 0.6, "dense": 0.25, "sparse": 0.15}


def reciprocal_rank_fusion(
    ranked_lists: list[RankedList], *, k: int = 60, top_k: int | None = None
) -> RankedList:
    """score(d) = sum_list 1 / (k + rank(d)). 순위만 쓰고 점수 크기는 버린다.

    비교/폴백용으로 남겨둔다. 기본 경로는 normalized_weighted_fusion 이다.
    """
    scores: dict[str, float] = {}
    lookup: dict[str, ChunkSchema] = {}
    for ranked in ranked_lists:
        for rank, (chunk, _score) in enumerate(ranked, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            lookup.setdefault(chunk.chunk_id, chunk)
    fused = sorted(((lookup[c], s) for c, s in scores.items()), key=lambda p: p[1], reverse=True)
    return fused[:top_k] if top_k is not None else fused


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        # 전부 같은 점수 -> 순위 정보가 없다. 1.0 으로 두면 이 리스트가 통째로
        # 최고점이 되어버리므로 0.5 로 중립화한다.
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def normalized_weighted_fusion(
    named_lists: dict[str, RankedList],
    *,
    weights: dict[str, float] | None = None,
    top_k: int | None = None,
) -> RankedList:
    """각 retriever 의 점수를 min-max 로 [0,1] 정규화한 뒤 가중합한다 (Stage 4 승자).

    RRF 와 달리 **점수 크기를 보존**하므로, 한 retriever 가 확신하는 후보가
    다른 retriever 의 애매한 후보에 밀리지 않는다.

    named_lists: {"bm25": [...], "dense": [...], "sparse": [...]}
                 값이 비어 있는 항목은 무시하고 가중치를 재정규화한다
                 (Dense 인덱스가 아직 없어도 그대로 동작 -> BM25 단독으로 수렴).
    """
    w = dict(weights or DEFAULT_WEIGHTS)
    active = {n: lst for n, lst in named_lists.items() if lst}
    if not active:
        return []
    wsum = sum(w.get(n, 0.0) for n in active) or 1.0

    scores: dict[str, float] = {}
    lookup: dict[str, ChunkSchema] = {}
    for name, ranked in active.items():
        weight = w.get(name, 0.0) / wsum
        if weight <= 0:
            continue
        norm = _minmax([s for _c, s in ranked])
        for (chunk, _raw), n in zip(ranked, norm):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight * n
            lookup.setdefault(chunk.chunk_id, chunk)

    fused = sorted(((lookup[c], s) for c, s in scores.items()), key=lambda p: p[1], reverse=True)
    return fused[:top_k] if top_k is not None else fused
