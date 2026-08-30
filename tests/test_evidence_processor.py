"""Stage 9 근거 구조화 + Stage 10 충분성 판정.

이 두 단계가 하는 일은 '근거 건수 세기'가 아니다. 건수만 보면
  - 10건 있는데 필요한 항목이 하나도 없는 경우
  - 정정 diff 인데 최종본만 10건인 경우
  - 두 회사 비교인데 한 회사 근거만 10건인 경우
가 전부 통과한다. 그 세 가지를 각각 잡는지 본다.
"""
from __future__ import annotations

from types import SimpleNamespace

from disclosure_rag.agent.evidence_processor import process_evidence
from disclosure_rag.agent.query_plan import QueryPlan
from disclosure_rag.agent.sufficiency import DEFAULT_MAX_NUDGES, check_sufficiency


def ref(key, text, unit=None):
    return SimpleNamespace(key=key, text=text, unit=unit, code=None)


def chunk(cid, rid="r1", refs=(), raw="", order=0, latest=True):
    return SimpleNamespace(chunk_id=cid, report_id=rid, field_codes=list(refs),
                           raw_text=raw, text=raw,
                           correction_order=order, is_latest=latest)


# --------------------------------------------------------------------------- 값 추출

def test_extracts_value_from_structured_field():
    plan = QueryPlan(expected_fields=["계약금액"])
    ev = [chunk("c1", refs=[ref("계약금액", "1,200,000,000")])]
    out = process_evidence(plan, ev)
    hit = out.by_field["계약금액"][0]
    assert hit.value_text == "1,200,000,000"
    assert hit.value_num == 1200000000.0
    assert hit.source == "field_code"
    assert hit.chunk_id == "c1"          # 출처를 끝까지 들고 간다


def test_field_name_matching_ignores_spacing():
    plan = QueryPlan(expected_fields=["계약 금액"])
    out = process_evidence(plan, [chunk("c1", refs=[ref("계약금액", "100")])])
    assert out.found_fields == ["계약 금액"]


def test_negative_dart_notation():
    """DART 는 음수를 △ 로 쓴다."""
    plan = QueryPlan(expected_fields=["당기순이익"])
    out = process_evidence(plan, [chunk("c1", refs=[ref("당기순이익", "△22,638")])])
    assert out.by_field["당기순이익"][0].value_num == -22638.0


def test_text_fallback_only_when_structure_misses():
    """구조에서 찾았으면 본문은 안 본다."""
    plan = QueryPlan(expected_fields=["계약금액", "투자목적"])
    ev = [chunk("c1", refs=[ref("계약금액", "100")], raw="투자목적 500 이며 계약금액 999")]
    out = process_evidence(plan, ev)
    assert out.by_field["계약금액"][0].value_text == "100"     # 구조 우선
    assert out.by_field["투자목적"][0].source == "text"        # 본문 폴백


def test_no_expected_fields_means_no_extraction():
    """fail open — 요구 항목이 없으면 구조화도 안 한다."""
    out = process_evidence(QueryPlan(), [chunk("c1", refs=[ref("계약금액", "100")])])
    assert out.found_fields == []
    assert out.documents == {"r1"}


def test_accepts_scored_pairs():
    plan = QueryPlan(expected_fields=["계약금액"])
    out = process_evidence(plan, [(chunk("c1", refs=[ref("계약금액", "100")]), 0.9)])
    assert out.found_fields == ["계약금액"]


# --------------------------------------------------------------------------- 정정 짝

def test_pairs_first_and_final():
    plan = QueryPlan(task="correction_diff", expected_fields=["계약금액"])
    ev = [chunk("c1", "orig", [ref("계약금액", "100")], order=0, latest=False),
          chunk("c2", "fix", [ref("계약금액", "150")], order=2, latest=True)]
    out = process_evidence(plan, ev)
    p = out.pairs[0]
    assert p.complete and p.changed is True
    assert p.first.value_text == "100" and p.final.value_text == "150"


def test_pair_detects_unchanged_value():
    plan = QueryPlan(task="correction_diff", expected_fields=["계약금액"])
    ev = [chunk("c1", "orig", [ref("계약금액", "100")], order=0),
          chunk("c2", "fix", [ref("계약금액", "100")], order=1)]
    assert process_evidence(plan, ev).pairs[0].changed is False


def test_only_one_version_leaves_the_pair_incomplete():
    """최종본만 10건 있어도 정정 비교는 안 된다 — 건수로는 안 잡힌다."""
    plan = QueryPlan(task="correction_diff", expected_fields=["계약금액"])
    ev = [chunk(f"c{i}", "fix", [ref("계약금액", "150")], order=2) for i in range(10)]
    out = process_evidence(plan, ev)
    assert len(out.hits) == 10                  # 근거는 많다
    assert out.incomplete_pairs                 # 그런데 짝이 안 맞는다


def test_pairs_only_built_for_correction_diff():
    plan = QueryPlan(task="lookup", expected_fields=["계약금액"])
    out = process_evidence(plan, [chunk("c1", refs=[ref("계약금액", "100")])])
    assert out.pairs == []


# --------------------------------------------------------------------------- 피연산자

def test_operands_collected_for_calculation():
    plan = QueryPlan(task="calculate", expected_fields=["투자금액", "자기자본"])
    ev = [chunk("c1", refs=[ref("투자금액", "1,000"), ref("자기자본", "5,000")])]
    out = process_evidence(plan, ev)
    assert sorted(o.value_num for o in out.operands) == [1000.0, 5000.0]


def test_non_numeric_values_are_not_operands():
    plan = QueryPlan(task="calculate", expected_fields=["투자목적"])
    out = process_evidence(plan, [chunk("c1", refs=[ref("투자목적", "생산능력 확대")])])
    assert out.operands == []


# --------------------------------------------------------------------------- Stage 10

def test_sufficient_when_all_fields_found():
    plan = QueryPlan(expected_fields=["계약금액", "계약상대"])
    ev = [chunk("c1", refs=[ref("계약금액", "100"), ref("계약상대", "A사")])]
    rep = check_sufficiency(plan, process_evidence(plan, ev))
    assert rep.ok and rep.reasons == []


def test_missing_field_is_named_not_just_counted():
    """'더 찾아봐'가 아니라 빠진 항목 이름을 준다."""
    plan = QueryPlan(expected_fields=["계약금액", "계약상대", "계약기간"])
    ev = [chunk(f"c{i}", refs=[ref("계약금액", "100")]) for i in range(10)]
    rep = check_sufficiency(plan, process_evidence(plan, ev))
    assert not rep.ok
    assert rep.missing_fields == ["계약상대", "계약기간"]
    assert "계약상대" in rep.retry_message()


def test_empty_sub_query_makes_it_insufficient():
    """두 회사 비교인데 한 회사 근거만 있는 경우 — 건수로는 통과한다."""
    plan = QueryPlan(expected_fields=["계약금액"])
    ev = [chunk(f"c{i}", refs=[ref("계약금액", "100")]) for i in range(10)]
    dec = SimpleNamespace(empty_labels=["company:한미반도체"])
    rep = check_sufficiency(plan, process_evidence(plan, ev), decompose_result=dec)
    assert not rep.ok
    assert rep.empty_sub_queries == ["company:한미반도체"]


def test_incomplete_pair_makes_it_insufficient():
    plan = QueryPlan(task="correction_diff", expected_fields=["계약금액"])
    ev = [chunk("c1", "fix", [ref("계약금액", "150")], order=2)]
    rep = check_sufficiency(plan, process_evidence(plan, ev))
    assert not rep.ok and rep.incomplete_pairs == ["계약금액"]


def test_no_expected_fields_never_blocks():
    """Field Schema 가 모르는 유형이면 제약을 안 건다 — 잘못된 기준으로
    답할 수 있는 질문을 거부하는 쪽이 훨씬 비싸다."""
    plan = QueryPlan(expected_fields=[])
    rep = check_sufficiency(plan, process_evidence(plan, [chunk("c1")]))
    assert rep.ok


def test_retry_budget_is_bounded():
    plan = QueryPlan(expected_fields=["없는항목"])
    proc = process_evidence(plan, [chunk("c1")])
    assert check_sufficiency(plan, proc, nudges_used=0).should_retry
    assert not check_sufficiency(plan, proc, nudges_used=DEFAULT_MAX_NUDGES).should_retry


def test_sufficient_never_retries():
    plan = QueryPlan(expected_fields=["계약금액"])
    ev = [chunk("c1", refs=[ref("계약금액", "100")])]
    assert not check_sufficiency(plan, process_evidence(plan, ev)).should_retry
