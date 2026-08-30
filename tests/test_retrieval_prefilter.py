"""회사 사전선별이 결과를 바꾸지 않는지.

## 왜 넣었나

예전 구조는 626,497행 전부에 점수를 매기고 정렬한 뒤 파이썬 루프로 걸렀다.
대상 회사의 청크가 드문 공시(거래소·주요사항)일수록 상위권에 안 걸려 목록을
깊이 훑어야 했다. 실측(2026-08-31, suite_v2 296문항):

    검색이 전체 소요시간의 83%
    lookup_form      검색 1회 67초
    lookup_periodic  검색 1회  3.4초   ← 같은 코드, 회사당 청크 밀도만 다름

최대 응답 196초로 대회 제한 300초의 65%까지 갔다. 회사가 지정되면 그 회사
행만 계산하도록 바꿨는데, **결과가 달라지면 안 된다** — 어차피 필터가
떨어뜨릴 행을 계산에서 빼는 것뿐이기 때문이다. 그 등가성을 여기서 고정한다.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from disclosure_rag.chunking.chunk_schema import ChunkSchema
from disclosure_rag.retrieval.metadata_filter import RetrievalFilter
from disclosure_rag.retrieval.numpy_dense_retriever import NumpyDenseRetriever

_COMPANIES = ("삼성전자", "한미반도체", "현대건설")


def _chunks(n: int = 300) -> list[ChunkSchema]:
    return [
        ChunkSchema(
            chunk_id=f"c{i}", report_id=f"r{i % 5}", text="", raw_text=f"본문{i}",
            company=_COMPANIES[i % 3], report_type="exchange",
            report_name="단일판매ㆍ공급계약체결", filing_date="20240101",
        )
        for i in range(n)
    ]


def _retriever(chunks, vecs, qv):
    r = NumpyDenseRetriever.__new__(NumpyDenseRetriever)
    r.embedding_provider = types.SimpleNamespace(embed_query=lambda _q: qv)
    r.matrix = vecs
    r.chunks = chunks
    buckets: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        buckets.setdefault(c.company, []).append(i)
    r._rows_by_company = {k: np.asarray(v, dtype=np.int64) for k, v in buckets.items()}
    return r


@pytest.fixture
def pair():
    chunks = _chunks()
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(len(chunks), 8)).astype(np.float32)
    qv = rng.normal(size=8).astype(np.float32)
    fast = _retriever(chunks, vecs, qv)
    slow = _retriever(chunks, vecs, qv)
    slow._rows_for = lambda flt: None          # 예전 방식(전체 계산 후 필터)
    return fast, slow


@pytest.mark.parametrize("names", [["삼성전자"], ["삼성전자", "한미반도체"],
                                   list(_COMPANIES)])
def test_prefilter_returns_identical_results(pair, names):
    fast, slow = pair
    flt = RetrievalFilter(companies=names)
    a = [(c.chunk_id, round(s, 6)) for c, s in fast.search("q", k=10, flt=flt)]
    b = [(c.chunk_id, round(s, 6)) for c, s in slow.search("q", k=10, flt=flt)]
    assert a == b and a, (names, a[:3], b[:3])


def test_no_filter_is_unaffected(pair):
    fast, slow = pair
    a = [c.chunk_id for c, _ in fast.search("q", k=10, flt=None)]
    b = [c.chunk_id for c, _ in slow.search("q", k=10, flt=None)]
    assert a == b and len(a) == 10


def test_unknown_company_returns_nothing(pair):
    """필터를 무시하고 전체로 되돌리면 안 된다 — 0건이 맞는 답이다."""
    fast, _ = pair
    assert fast.search("q", k=10, flt=RetrievalFilter(companies=["없는회사"])) == []


def test_rows_are_actually_narrowed(pair):
    """실제로 계산량이 줄었는지 — 안 줄면 고친 의미가 없다."""
    fast, _ = pair
    rows = fast._rows_for(RetrievalFilter(companies=["삼성전자"]))
    assert rows is not None and len(rows) == 100      # 300개 중 1/3
    assert fast._rows_for(None) is None
