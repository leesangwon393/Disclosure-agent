"""Phase 10+11 회귀 테스트: RRF Fusion + Reranker."""

from __future__ import annotations

from pathlib import Path

import pytest

from disclosure_rag.chunking.chunk_schema import ChunkSchema, filter_leaf_chunks
from disclosure_rag.common.manifest_loader import load_manifest
from disclosure_rag.pipeline import build_all_chunks
from disclosure_rag.retrieval.bm25_retriever import BM25Retriever
from disclosure_rag.retrieval.fusion import reciprocal_rank_fusion
from disclosure_rag.retrieval.hybrid_retriever import HybridRetriever
from disclosure_rag.retrieval.reranker import NoOpReranker
from disclosure_rag.retrieval.tokenizers import build_tokenizer

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus"
pytestmark = pytest.mark.skipif(not CORPUS_ROOT.is_dir(), reason="corpus/ 없음")

SAMPLE_DOC_IDS = {
    "periodic_20240312000736",
    "major_20241118000171",
    "exchange_20250728800035",
    "holding_20241025000530",
}


def _c(chunk_id, **kw):
    base = dict(report_id="r", text="t", raw_text="t")
    base.update(kw)
    return ChunkSchema(chunk_id=chunk_id, **base)


def test_rrf_agrees_when_both_lists_rank_same_top():
    a = _c("a")
    b = _c("b")
    list1 = [(a, 9.0), (b, 1.0)]
    list2 = [(a, 0.9), (b, 0.1)]
    fused = reciprocal_rank_fusion([list1, list2])
    assert fused[0][0].chunk_id == "a"


def test_rrf_boosts_doc_appearing_in_both_lists_over_single_list_winner():
    """BM25 1등이지만 Dense 에는 전혀 안 잡히는 문서보다, 두 리스트 모두에서
    중간 순위인 문서가 RRF 상에서 더 높아질 수 있어야 한다 (핵심 동작 검증)."""
    a, b, c = _c("a"), _c("b"), _c("c")
    bm25_list = [(a, 9.0), (b, 5.0), (c, 4.0)]
    dense_list = [(c, 0.9), (b, 0.8)]  # a 는 dense 에 없음
    fused = reciprocal_rank_fusion([bm25_list, dense_list])
    fused_ids = [c.chunk_id for c, _ in fused]
    assert fused_ids.index("b") < fused_ids.index("a")


def test_rrf_dedups_by_chunk_id():
    a = _c("a")
    fused = reciprocal_rank_fusion([[(a, 1.0)], [(a, 2.0)]])
    assert len(fused) == 1


def test_rrf_top_k_truncates():
    chunks = [_c(str(i)) for i in range(5)]
    ranked = [(c, 5 - i) for i, c in enumerate(chunks)]
    fused = reciprocal_rank_fusion([ranked], top_k=2)
    assert len(fused) == 2


def test_noop_reranker_passthrough():
    a, b = _c("a"), _c("b")
    reranker = NoOpReranker()
    result = reranker.rerank("query", [(a, 1.0), (b, 0.5)], top_k=1)
    assert result == [(a, 1.0)]


def test_hybrid_retriever_bm25_only_matches_bm25_search():
    """dense=None 이면 HybridRetriever 가 순수 BM25 only 로 동작해야 한다 (§74 비교軸)."""
    manifest = load_manifest(CORPUS_ROOT)
    rows = [r for r in manifest if r.doc_id in SAMPLE_DOC_IDS]
    chunks = filter_leaf_chunks(build_all_chunks(str(CORPUS_ROOT), rows=rows, validate=False))
    tok = build_tokenizer("whitespace")
    bm25 = BM25Retriever(chunks, tok)
    hybrid = HybridRetriever(bm25, dense=None, reranker=None)

    q = "계약금액"
    bm25_top = [c.chunk_id for c, _ in bm25.search(q, k=5)]
    hybrid_top = [c.chunk_id for c, _ in hybrid.search(q, k=5)]
    assert bm25_top == hybrid_top
