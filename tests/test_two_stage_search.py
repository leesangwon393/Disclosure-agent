"""2단계 검색(문서 확정 -> 그 문서 안에서 재검색) 계약 테스트.

실측 근거(2026-08-30, 314문항):
    정기공시는 문서 1건이 leaf chunk 559.7개를 만든다.
    정답 문서는 94.1% 로 찾는데 상한은 71.5% — 문서 안에서 조각을 못 골랐다.

여기서는 리트리버를 스텁으로 두고 **호출 방식**을 검증한다. 실제 순위 개선은
채점기(scripts/score_answers.py)로 잰다.
"""
from __future__ import annotations

from disclosure_rag.agent.tools import make_search_disclosures_tool
from disclosure_rag.chunking.chunk_schema import ChunkSchema


def _chunk(cid, report_id, company="삼성전자", text="본문"):
    return ChunkSchema(chunk_id=cid, report_id=report_id, company=company,
                       text=text, raw_text=text)


class _StubRetriever:
    """호출 이력을 기록하고, report_ids 필터가 오면 그 문서 조각만 돌려준다."""

    def __init__(self, corpus):
        self.corpus = corpus          # list[(chunk, score)] — 전역 점수 순
        self.calls = []

    def search(self, query, *, k=10, flt=None):
        self.calls.append({"k": k, "report_ids": getattr(flt, "report_ids", None),
                           "companies": getattr(flt, "companies", None)})
        pool = self.corpus
        if flt is not None and flt.report_ids:
            pool = [(c, s) for c, s in pool if c.report_id in flt.report_ids]
        return pool[:k]


# 전역 순위: A문서 조각 1개가 1등, 그 뒤로 다른 문서 조각들, A문서의 정답 조각은 한참 뒤
_CORPUS = (
    [(_chunk("A-1", "periodic_A"), 0.90)]
    + [(_chunk(f"B-{i}", "periodic_B"), 0.80 - i * 0.001) for i in range(40)]
    + [(_chunk("A-answer", "periodic_A"), 0.40)]          # 정답 조각 — 전역 42등
    + [(_chunk(f"C-{i}", "periodic_C"), 0.30) for i in range(200)]
)


def _run(**kwargs):
    # 2026-08-30: 실측 결과가 나빠 프로덕션 기본값은 two_stage=False 다
    # (tools.make_search_disclosures_tool docstring 참고). 이 테스트들은
    # 기능 자체의 계약을 검증하므로 명시적으로 켜서 돌린다.
    kwargs.setdefault("two_stage", True)
    r = _StubRetriever(list(_CORPUS))
    tool = make_search_disclosures_tool(r, **kwargs)
    return r, tool.handler(query="질의", company="삼성전자", top_k=5)


def test_two_stage_promotes_a_chunk_from_the_target_document():
    """전역 42등이던 A문서 정답 조각이, 문서로 좁히면 상위로 올라온다."""
    r, out = _run()
    ids = [x["chunk_id"] for x in out["results"]]
    assert "A-answer" in ids, f"2단계가 정답 조각을 못 올렸다: {ids}"


def test_single_stage_misses_it():
    """대조군 — 예전 방식(1단계)에서는 top_k=5 안에 못 들어온다."""
    r, out = _run(two_stage=False)
    ids = [x["chunk_id"] for x in out["results"]]
    assert "A-answer" not in ids


def test_stage1_is_wide_and_stage2_is_narrow():
    r, out = _run()
    assert len(r.calls) == 2, "검색이 2번 일어나야 한다"
    wide, narrow = r.calls
    assert wide["k"] >= 50, "1단계는 후보 문서를 뽑을 만큼 넓게 봐야 한다"
    assert wide["report_ids"] is None
    assert narrow["report_ids"], "2단계는 문서로 좁혀야 한다"
    # 2단계는 문서별 몫을 확보하려고 넉넉히 받아온 뒤 라운드로빈으로 top_k 로 줄인다.
    # HybridRetriever 가 k 와 무관하게 후보 50개를 뽑으므로 추가 비용이 없다.
    assert narrow["k"] >= 50


def test_candidate_docs_are_capped():
    r, out = _run()
    _wide, narrow = r.calls
    assert len(narrow["report_ids"]) <= 3


def test_result_count_does_not_exceed_top_k():
    r, out = _run()
    assert len(out["results"]) <= 5


def test_stage1_hits_are_not_lost():
    """2단계가 1단계 1등을 밀어내면 안 된다 — 나빠질 수 없어야 한다."""
    r, out = _run()
    ids = [x["chunk_id"] for x in out["results"]]
    assert "A-1" in ids, "1단계 최상위 조각이 사라졌다"


def test_no_results_falls_through_without_crashing():
    r = _StubRetriever([])
    tool = make_search_disclosures_tool(r, two_stage=True)
    out = tool.handler(query="질의", company="없는회사")
    assert out["results"] == []
    assert "찾지 못함" in (out["note"] or "")


def test_report_id_path_is_unchanged():
    """report_id 를 직접 주면 예전처럼 그 문서만 본다(2단계 불필요)."""
    r = _StubRetriever(list(_CORPUS))
    tool = make_search_disclosures_tool(r, two_stage=True)
    out = tool.handler(query="질의", report_id="periodic_A")
    assert r.calls[0]["report_ids"] == ["periodic_A"]
    assert all(x["report_id"] == "periodic_A" for x in out["results"])
