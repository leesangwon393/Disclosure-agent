"""Parent Expansion 계약 테스트."""
from __future__ import annotations

from disclosure_rag.chunking.chunk_schema import ChunkSchema
from disclosure_rag.retrieval.parent_expander import ParentExpander


def _c(cid, parent, text):
    return ChunkSchema(chunk_id=cid, report_id="r1", parent_chunk_id=parent,
                       text=text, raw_text=text, company="삼성전자")


def test_child_is_expanded_to_parent_section():
    parent_text = "앞부분 " * 50 + "핵심내용" + " 뒷부분" * 50
    exp = ParentExpander({"P1": parent_text})
    out = exp.expand([(_c("P1::C1", "P1", "핵심내용"), 1.0)])
    assert len(out) == 1 and out[0]["expanded"] is True
    assert len(out[0]["text"]) > len("핵심내용"), "parent 로 넓혀지지 않았다"


def test_missing_parent_falls_back_to_chunk():
    """parent 가 없으면 조각 그대로. 조용히 빈 근거를 만들지 않는다."""
    exp = ParentExpander({})
    out = exp.expand([(_c("X::C1", "X", "본문"), 1.0)])
    assert out[0]["text"] == "본문" and out[0]["expanded"] is False


def test_same_parent_is_not_duplicated():
    exp = ParentExpander({"P1": "섹션 전체 본문"})
    out = exp.expand([(_c("P1::C1", "P1", "가"), 1.0), (_c("P1::C2", "P1", "나"), 0.9)])
    assert len(out) == 1, "같은 parent 가 중복 근거로 들어갔다"


def test_one_evidence_cannot_eat_the_whole_budget():
    """회귀: 3,000자 예산에 parent 하나가 3,000자를 차지해 나머지 근거가 전부 탈락했다.
    섹션 전체가 parent 이므로 흔한 일이다."""
    big = "x" * 20000
    exp = ParentExpander({"P1": big, "P2": big, "P3": big})
    out = exp.expand(
        [(_c("P1::C1", "P1", "a"), 1.0), (_c("P2::C1", "P2", "b"), .9), (_c("P3::C1", "P3", "c"), .8)],
        budget_chars=3000, min_evidences=3,
    )
    assert len(out) == 3, f"근거가 {len(out)}개만 남았다 — 1인분 상한이 동작하지 않는다"
    assert sum(len(o["text"]) for o in out) <= 3000


# ── 형제 조각 확장 (2026-08-25) ────────────────────────────────────────────
def _tchunk(cid, raw, tids, parent=None):
    from disclosure_rag.chunking.chunk_schema import ChunkSchema
    return ChunkSchema(
        chunk_id=cid, parent_chunk_id=parent, raw_text=raw, text=raw,
        report_id="R", company="A", corp_code="1", report_type="periodic",
        report_name="사업보고서", filing_date="20250101", section_path=["S"],
        content_type="table", source_path="p.xml", table_ids=tids,
    )


def test_build_table_map_groups_fragments():
    from disclosure_rag.retrieval.parent_expander import ParentExpander
    cs = [_tchunk("c1", "합계 1.매출액 100", ["R::T0"]),
          _tchunk("c2", "2.영업이익 47", ["R::T0"]),
          _tchunk("c3", "딴 표", ["R::T1"])]
    m = ParentExpander.build_table_map(cs)
    assert m["R::T0"] == ["c1", "c2"]
    assert m["R::T1"] == ["c3"]


def test_sibling_fragment_is_attached():
    """같은 표의 다른 조각이 근거에 붙어야 한다 — 정답 숫자와 상위 라벨이
    다른 조각에 흩어져도 함께 보이게 하는 안전망."""
    from disclosure_rag.retrieval.parent_expander import ParentExpander
    c1 = _tchunk("c1", "합계 1.매출액 계 192,972,588", ["R::T0"])
    c2 = _tchunk("c2", "2.영업이익 계 47,206,319", ["R::T0"])
    exp = ParentExpander({}, max_chars=4000,
                         table_map=ParentExpander.build_table_map([c1, c2]),
                         chunk_by_id={"c1": c1, "c2": c2})
    ev = exp.expand([(c2, 1.0)], budget_chars=4000)
    assert len(ev) == 1
    assert "47,206,319" in ev[0]["text"]
    assert "192,972,588" in ev[0]["text"], "형제 조각이 안 붙었다"
    assert ev[0]["table_sibling_ids"] == ["c1"]


def test_sibling_off_by_default_keeps_old_behavior():
    from disclosure_rag.retrieval.parent_expander import ParentExpander
    c = _tchunk("c1", "표 조각", ["R::T0"])
    exp = ParentExpander({})              # table_map 없음 = 기능 꺼짐
    ev = exp.expand([(c, 1.0)], budget_chars=4000)
    assert ev[0]["text"] == "표 조각"
    assert ev[0]["table_sibling_ids"] == []


def test_sibling_respects_budget():
    from disclosure_rag.retrieval.parent_expander import ParentExpander
    c1 = _tchunk("c1", "가" * 3000, ["R::T0"])
    c2 = _tchunk("c2", "나" * 100, ["R::T0"])
    exp = ParentExpander({}, table_map=ParentExpander.build_table_map([c1, c2]),
                         chunk_by_id={"c1": c1, "c2": c2})
    ev = exp.expand([(c2, 1.0)], budget_chars=200)
    assert len(ev[0]["text"]) <= 200, "형제 조각이 예산을 넘겼다"
