"""좁은 필터에서 리트리버가 빈 결과를 내지 않는지 — 2단계 검색의 전제 조건.

## 실측 배경 (2026-08-30, 60문항 full 실행)

    같은 54문항, 정답이 든 문서를 회수한 비율
      필터 없이 그냥 검색   94.4%
      에이전트가 검색       66.7%

원인 중 하나가 이거다. 리트리버가 "전체에서 상위 N개를 먼저 뽑고 그 다음에
필터로 거른다" 순서라, 문서 1건으로 좁히면 통과 대상이 상위 N개 안에 하나도
없어 결과가 0건이 된다. 2단계 검색(문서 확정 -> 그 안에서 재검색)은 이게
고쳐져야 성립한다.
"""
from __future__ import annotations

import numpy as np
import pytest

from disclosure_rag.chunking.chunk_schema import ChunkSchema
from disclosure_rag.retrieval.metadata_filter import RetrievalFilter
from disclosure_rag.retrieval.numpy_dense_retriever import NumpyDenseRetriever


# --- is_selective -----------------------------------------------------------

def test_report_ids_and_companies_are_selective():
    assert RetrievalFilter(report_ids=["periodic_1"]).is_selective
    assert RetrievalFilter(companies=["삼성전자"]).is_selective


def test_broad_filters_are_not_selective():
    """doc_group/period 는 코퍼스의 큰 덩어리라 pool 상한을 풀 이유가 없다."""
    assert not RetrievalFilter(doc_groups=["periodic"]).is_selective
    assert not RetrievalFilter(periods=["2024"]).is_selective
    assert not RetrievalFilter().is_selective


# --- numpy dense: 좁은 필터에서 pool 상한 때문에 놓치지 않는가 --------------

class _StubProvider:
    """질의 벡터를 고정해 첫 번째 차원이 큰 chunk 가 상위에 오게 한다."""

    name = "stub"

    def embed_query(self, q):
        return np.array([1.0, 0.0], dtype=np.float32)


def _build_retriever(n: int, target_index: int) -> NumpyDenseRetriever:
    """n개 chunk 중 target 만 다른 문서에 속하고, 점수는 **가장 낮게** 만든다.

    옛 구현은 전역 상위 pool(최대 1,000개)만 보고 걸렀으므로 이런 배치에서
    target 을 영영 못 찾는다.
    """
    r = NumpyDenseRetriever.__new__(NumpyDenseRetriever)
    matrix = np.zeros((n, 2), dtype=np.float32)
    matrix[:, 0] = np.linspace(1.0, 0.0, n)      # 앞쪽일수록 고득점
    chunks = []
    for i in range(n):
        report = "periodic_TARGET" if i == target_index else "periodic_OTHER"
        chunks.append(ChunkSchema(
            chunk_id=f"C{i}", report_id=report, company="삼성전자",
            text="본문", raw_text="본문",
        ))
    r.matrix = matrix
    r.chunks = chunks
    r.embedding_provider = _StubProvider()
    return r


def test_selective_filter_finds_a_low_ranked_chunk():
    """정답 chunk 가 전역 5,000등이어도 문서로 좁히면 찾아야 한다."""
    n = 5_000
    r = _build_retriever(n, target_index=n - 1)   # 꼴등
    flt = RetrievalFilter(report_ids=["periodic_TARGET"])
    got = r.search("질의", k=5, flt=flt)
    assert len(got) == 1, "좁은 필터인데 결과가 비었다 — pool 상한이 다시 생겼다"
    assert got[0][0].chunk_id == f"C{n - 1}"


def test_selective_filter_returns_all_matching_up_to_k():
    n = 3_000
    r = _build_retriever(n, target_index=0)
    # 여러 chunk 를 같은 문서로 만든다 (뒤쪽 = 저득점)
    for i in (2_500, 2_700, 2_900):
        r.chunks[i].report_id = "periodic_TARGET"
    got = r.search("질의", k=10, flt=RetrievalFilter(report_ids=["periodic_TARGET"]))
    assert len(got) == 4
    scores = [s for _, s in got]
    assert scores == sorted(scores, reverse=True), "점수 내림차순이 깨졌다"


def test_broad_filter_still_uses_the_fast_path():
    """넓은 필터는 예전처럼 pool 을 잘라도 된다 — 느려지면 안 된다."""
    n = 3_000
    r = _build_retriever(n, target_index=0)
    got = r.search("질의", k=5, flt=RetrievalFilter(doc_groups=[None]))
    # doc_groups=[None] 은 report_type 이 None 인 chunk 와 일치한다
    assert len(got) == 5


def test_no_filter_path_unchanged():
    r = _build_retriever(100, target_index=0)
    got = r.search("질의", k=3, flt=None)
    assert [c.chunk_id for c, _ in got] == ["C0", "C1", "C2"]
