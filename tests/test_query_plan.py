"""QueryPlan — 파이프라인 전체를 제어하는 계획서.

여기가 틀리면 검색 개수·분해 여부·충족 조건·거부 판정이 전부 어긋난다.
그래서 (1) 분류 규칙 (2) 계획 조립 (3) 검증기 (4) 실제 정답셋 회귀
네 갈래로 나눠 본다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from disclosure_rag.agent.query_plan import (  # noqa: E402
    PlanValidator,
    QueryPlan,
    RulePlanBuilder,
    classify_answer_mode,
    classify_task,
    decide_latest_policy,
    detect_report_types,
    fill_missing_with_hcx,
)

SUITE = ROOT / "eval" / "suite_v1.jsonl"
SCHEMA = ROOT / "config" / "field_schema.json"
REGISTRY = ROOT / "artifacts_v2" / "registry" / "entities.json"
CORPUS = ROOT / "corpus"


# --------------------------------------------------------------------------- answer_mode

def test_closed_ending_gives_closed():
    assert classify_answer_mode("한미반도체의 자산총계는 얼마인가?") == "closed"


def test_open_ending_gives_open():
    assert classify_answer_mode("한화오션의 자금조달 내역을 유형별로 정리해줘.") == "open"


def test_both_endings_give_mixed():
    """S023~S026 형태 — 앞은 예/아니오, 뒤는 설명."""
    q = "현대건설의 공시가 정정된 내역이 있는가? 있다면 무엇이 달라졌는지 설명해줘."
    assert classify_answer_mode(q) == "mixed"


def test_no_ending_gives_unknown():
    """명사로 끝나면 규칙이 '모른다'고 말해야 한다 — 억지로 찍으면 안 된다."""
    assert classify_answer_mode("삼성전자 자기주식 취득 관련 사항.") == "unknown"


def test_content_words_do_not_decide_the_mode():
    """같은 지표라도 끝말이 답의 모양을 정한다."""
    assert classify_answer_mode("한미반도체의 자산총계는 얼마인가?") == "closed"
    assert classify_answer_mode("한미반도체의 자산총계 추이를 설명해줘.") == "open"


def test_mixed_is_treated_as_open_superset():
    p = QueryPlan(answer_mode="mixed")
    assert p.is_open_ended
    assert QueryPlan(answer_mode="open").is_open_ended
    assert not QueryPlan(answer_mode="closed").is_open_ended


# --------------------------------------------------------------------------- task

def test_correction_diff_needs_both_signals():
    assert classify_task("정정된 내역이 있는가? 무엇이 달라졌는지") == "correction_diff"
    # '정정'만 있고 차이를 묻지 않으면 correction_diff 가 아니다
    assert classify_task("정정공시가 있는가?", mode="closed") != "correction_diff"


def test_two_companies_make_it_compare():
    assert classify_task("A와 B 중 최대 계약금액은?", n_companies=2) == "compare"


def test_compare_beats_calculate_for_two_companies():
    """S007~S014 — 최대값 계산이 들어가지만 본질은 비교다. compare 로 분류해야
    회사별로 질문이 쪼개진다."""
    q = "삼성전자와 한미반도체 중 최대 계약금액은 얼마이며 더 큰 쪽은 어느 기업인가?"
    assert classify_task(q, n_companies=2, mode="closed") == "compare"


def test_count_question():
    assert classify_task("계약이 총 몇 건인가?", mode="closed") == "count"


def test_open_defaults_to_summarize():
    assert classify_task("주요 내용을 정리해줘", mode="open") == "summarize"


def test_closed_defaults_to_lookup():
    assert classify_task("순자산액은 얼마인가?", mode="closed") == "lookup"


# --------------------------------------------------------------------------- latest_policy

def test_no_correction_word_means_latest_only():
    assert decide_latest_policy("삼성전자의 계약금액은 얼마인가?") == "latest_only"


def test_first_and_final_for_diff_questions():
    assert decide_latest_policy("최초 공시와 최종 정정본 사이에 무엇이 달라졌는지") == "first_and_final"


def test_all_versions_for_history_questions():
    assert decide_latest_policy("정정 이력을 전부 보여줘") == "all_versions"


# --------------------------------------------------------------------------- report_types

def test_detect_report_types():
    assert detect_report_types("삼성전자 사업보고서") == ["periodic"]
    assert detect_report_types("주요사항보고서(자기주식취득결정)") == ["major"]
    assert "exchange" in detect_report_types("단일판매ㆍ공급계약 체결")


# --------------------------------------------------------------------------- 분해 조건

def test_decomposition_is_decided_by_complexity_not_answer_mode():
    """S007~S014 는 answer_mode 가 closed 인데 두 회사를 각각 조회해야 한다.
    open 일 때만 쪼개면 이 8문항(21%)이 반쪽 답변이 된다."""
    p = QueryPlan(answer_mode="closed", companies=["삼성전자", "한미반도체"],
                  needs_multiple_documents=True, operations=["조회:A", "조회:B", "비교"])
    assert p.needs_decomposition


def test_single_lookup_is_not_decomposed():
    p = QueryPlan(answer_mode="closed", companies=["삼성전자"], operations=["조회:삼성전자"])
    assert not p.needs_decomposition


def test_top_k_scales_with_targets():
    assert QueryPlan(answer_mode="closed", companies=["A"]).top_k() == 8
    assert QueryPlan(answer_mode="open").top_k() == 24
    assert QueryPlan(task="correction_diff").top_k() == 24
    two = QueryPlan(companies=["A", "B"], needs_multiple_documents=True)
    assert two.top_k() == 20
    kinds = QueryPlan(companies=["A"], report_kinds=["체결", "해지"],
                      needs_multiple_documents=True)
    assert kinds.top_k() == 20


# --------------------------------------------------------------------------- is_complete

def test_is_complete_requires_mode_task_and_company():
    assert not QueryPlan().is_complete
    assert not QueryPlan(answer_mode="closed", task="lookup").is_complete       # 회사 없음
    assert not QueryPlan(answer_mode="closed", companies=["A"]).is_complete     # task 없음
    assert QueryPlan(answer_mode="closed", task="lookup", companies=["A"]).is_complete


# --------------------------------------------------------------------------- 검증기

def test_validator_repairs_invalid_enum():
    p = QueryPlan(task="존재하지_않는_작업")  # type: ignore[arg-type]
    res = PlanValidator().validate(p)
    assert p.task == "unknown"
    assert res.ok                       # 고쳤으므로 통과
    assert res.issues[0].repaired_to == "unknown"


def test_validator_strips_natural_language_periods():
    """'2024년' 이 그대로 남으면 기간 필터가 영구 0건이 된다."""
    p = QueryPlan(periods=["2024", "2024년", "최근 3년", "2024-12"])
    PlanValidator().validate(p)
    assert p.periods == ["2024", "2024-12"]


def test_validator_rejects_compare_with_one_target():
    p = QueryPlan(task="compare", companies=["삼성전자"])
    res = PlanValidator().validate(p)
    assert not res.ok
    assert "비교 대상이 하나" in res.errors[0].message


def test_validator_fixes_correction_diff_policy():
    """정정 전후 비교인데 최종본만 남기면 최초본이 사라진다."""
    p = QueryPlan(task="correction_diff", latest_policy="latest_only")
    PlanValidator().validate(p)
    assert p.latest_policy == "first_and_final"


def test_validator_requires_operands_for_calculation():
    p = QueryPlan(task="calculate", companies=["A"], expected_fields=[])
    assert not PlanValidator().validate(p).ok


def test_validator_flags_company_absent_from_corpus():
    class _Reg:
        def types_for(self, name):
            return {"universe"} if name == "삼성전자" else set()

    p = QueryPlan(companies=["삼성전자", "존재하지않는회사"])
    res = PlanValidator(registry=_Reg()).validate(p)
    assert not res.ok
    assert "존재하지않는회사" in res.errors[0].message


def test_validator_drops_fields_absent_from_the_schema():
    """없는 항목을 요구하면 충분성 검사가 영원히 만족되지 않는다."""
    class _Schema:
        def classify(self, kind, key):
            return "required" if key == "계약금액" else "unknown"

    p = QueryPlan(report_kinds=["단일판매공급계약체결"],
                  expected_fields=["계약금액", "없는항목"])
    PlanValidator(schema=_Schema()).validate(p)
    assert p.expected_fields == ["계약금액"]


# --------------------------------------------------------------------------- HCX 폴백

class _FakeClient:
    def __init__(self, reply): self.reply, self.calls = reply, 0
    def chat(self, messages): self.calls += 1; return self.reply


def test_hcx_is_not_called_when_rules_suffice():
    c = _FakeClient('{"answer_mode":"open","task":"summarize"}')
    p = QueryPlan(answer_mode="closed", task="lookup", companies=["A"])
    fill_missing_with_hcx(p, "질문", c)
    assert c.calls == 0
    assert p.answer_mode == "closed"


def test_hcx_fills_only_the_blanks():
    c = _FakeClient('{"answer_mode":"open","task":"summarize"}')
    p = QueryPlan(companies=["A"], task="lookup")     # answer_mode 만 비어 있다
    fill_missing_with_hcx(p, "질문", c)
    assert p.answer_mode == "open" and p.source["answer_mode"] == "hcx"
    assert p.task == "lookup" and p.source.get("task") != "hcx"


def test_hcx_garbage_is_ignored():
    c = _FakeClient('{"answer_mode":"완전히 이상한 값","task":"없는작업"}')
    p = QueryPlan(companies=["A"])
    fill_missing_with_hcx(p, "질문", c)
    assert p.answer_mode == "open"          # 안전한 기본값
    assert p.source["answer_mode"] == "default"
    assert p.task == "unknown"


def test_hcx_failure_keeps_the_rule_plan():
    class _Boom:
        def chat(self, messages): raise RuntimeError("429")

    p = QueryPlan(companies=["A"], task="lookup")
    fill_missing_with_hcx(p, "질문", _Boom())
    assert p.task == "lookup"
    assert any("실패" in n for n in p.notes)


def test_unresolved_mode_defaults_to_open_side():
    """closed 를 open 으로 오판하면 근거가 늘 뿐이지만, 반대는 항목이 빠진다."""
    class _Empty:
        def chat(self, messages): return "{}"

    p = QueryPlan(companies=["A"], task="lookup")
    fill_missing_with_hcx(p, "질문", _Empty())
    assert p.answer_mode == "open"


# --------------------------------------------------------------------------- 정답셋 회귀

@pytest.mark.skipif(not (SUITE.exists() and SCHEMA.exists() and CORPUS.exists()),
                    reason="정답셋/명세/코퍼스 미구비")
class TestOnGoldSuite:
    @pytest.fixture(scope="class")
    @classmethod
    def built(cls):
        from disclosure_rag.agent.field_schema import FieldSchema
        from disclosure_rag.entity.entity_extractor import EntityExtractor
        schema = FieldSchema.load(SCHEMA)
        builder = RulePlanBuilder(
            schema=schema,
            extractor=EntityExtractor(corpus_root=str(CORPUS),
                                      metric_terms_path=str(ROOT / "config" / "metric_terms.txt")),
        )
        rows = [json.loads(l) for l in SUITE.read_text(encoding="utf-8").splitlines() if l.strip()]
        return schema, [(r, builder.build(r["query"])) for r in rows]

    def test_answer_mode_matches_every_label(self, built):
        """mixed 는 open 의 상위집합으로 본다(라벨에 mixed 칸이 없다)."""
        _schema, pairs = built
        wrong = [r["id"] for r, p in pairs
                 if ("open" if p.answer_mode == "mixed" else p.answer_mode) != r["mode"]]
        assert wrong == []

    def test_every_question_is_planned_without_hcx(self, built):
        _schema, pairs = built
        assert [r["id"] for _r, p in pairs if not p.is_complete for r in [_r]] == []
        assert all(v != "hcx" for _r, p in pairs for v in p.source.values())

    def test_two_company_questions_are_decomposed(self, built):
        """S007~S014 — closed 지만 회사별로 쪼개야 한다."""
        _schema, pairs = built
        for r, p in pairs:
            if r["id"] in {f"S{i:03d}" for i in range(7, 15)}:
                assert p.needs_decomposition, r["id"]
                assert len(p.companies) == 2, r["id"]

    def test_contract_termination_questions_are_decomposed(self, built):
        """S015~S022 — 회사 1곳·기간 없음이지만 체결 공시와 해지 공시가 둘 다
        필요하다. 공시유형 개수로 잡아야 걸린다."""
        _schema, pairs = built
        for r, p in pairs:
            if r["id"] in {f"S{i:03d}" for i in range(15, 23)}:
                assert len(p.report_kinds) == 2, r["id"]
                assert p.needs_decomposition, r["id"]
                assert p.top_k() >= 20, r["id"]

    def test_correction_questions_keep_the_first_version(self, built):
        _schema, pairs = built
        for r, p in pairs:
            if r["id"] in {"S023", "S024", "S025", "S026"}:
                assert p.latest_policy == "first_and_final", r["id"]

    def test_no_validation_errors_on_the_whole_suite(self, built):
        from disclosure_rag.entity.entity_registry import EntityRegistry
        schema, pairs = built
        reg = EntityRegistry.load(REGISTRY) if REGISTRY.exists() else None
        v = PlanValidator(registry=reg, schema=schema)
        bad = {r["id"]: [i.message for i in v.validate(p).errors]
               for r, p in pairs if not v.validate(p).ok}
        assert bad == {}

    def test_periods_are_in_filter_format(self, built):
        import re
        _schema, pairs = built
        for _r, p in pairs:
            for period in p.periods:
                assert re.match(r"^\d{4}(-\d{2})?$", period), period


# ------------------------------------------------------- 개념어 -> 공시유형 전개
#
# 실측 실패(S030~S032, 2026-08-30): "2025년에 실시한 자금조달 내역" 은
# 공시유형명을 하나도 안 써서 report_kinds 가 비었고, 그래서 doc_group 도
# 안 정해져 검색이 전체 문서를 훑었다. 정답 문서는 상위에 못 올라왔다.

from disclosure_rag.agent.query_plan import expand_concept_kinds  # noqa: E402

_KNOWN = [
    "주요사항보고서(유상증자결정)", "주요사항보고서(전환사채권발행결정)",
    "주요사항보고서(교환사채권발행결정)", "주요사항보고서(무상증자결정)",
    "주요사항보고서(자본으로인정되는채무증권발행결정)", "유상증자결정",
    "상각형조건부자본증권발행결정",
    "단일판매공급계약체결", "신규시설투자등",
]


def test_funding_concept_expands_to_every_issuing_kind():
    got = expand_concept_kinds("한화오션이 2023년에 실시한 자금조달 내역을 정리해줘", _KNOWN)
    assert "주요사항보고서(유상증자결정)" in got
    assert "주요사항보고서(전환사채권발행결정)" in got
    assert "주요사항보고서(자본으로인정되는채무증권발행결정)" in got


def test_expansion_must_not_be_narrower_than_the_question():
    """한 종류만 잡히면 나머지 문서가 필터에서 잘려 오히려 나빠진다.

    디앤디파마텍(S030)의 정답 문서는 전환사채권발행결정과
    자본으로인정되는채무증권발행결정 둘이다. 유상증자만 잡으면 둘 다 사라진다.
    """
    got = expand_concept_kinds("디앤디파마텍이 2025년에 실시한 자금조달 내역", _KNOWN)
    assert len(got) >= 5


def test_expansion_only_returns_kinds_that_exist_in_the_schema():
    got = expand_concept_kinds("자금조달 내역", ["주요사항보고서(유상증자결정)"])
    assert got == ["주요사항보고서(유상증자결정)"]


def test_unrelated_question_is_not_expanded():
    assert expand_concept_kinds("삼성전자의 순자산액은 얼마인가?", _KNOWN) == []
    assert expand_concept_kinds("신규시설투자등 공시를 정리해줘", _KNOWN) == []


# --------------------------------------------------- 정정본을 지목한 질문
from disclosure_rag.agent.query_plan import detect_corrections_only  # noqa: E402


def test_correction_marker_in_question_restricts_to_corrections():
    assert detect_corrections_only("[기재정정]주요사항보고서에 기재된 순자산액은?")
    assert detect_corrections_only("삼성전자의 정정공시 내용을 알려줘")
    assert detect_corrections_only("기재정정 보고서의 계약금액은?")


def test_plain_question_does_not_restrict():
    assert not detect_corrections_only("주요사항보고서에 기재된 순자산액은?")
    assert not detect_corrections_only("2024년 신규시설투자등 공시를 정리해줘")


def test_correction_diff_needs_the_original_too():
    """정정 전후 비교는 최초본이 있어야 성립한다 — 여기서 정정본만 남기면 안 된다."""
    q = "정정된 내역이 있는가? 있다면 최초 공시와 최종 정정본 사이 무엇이 달라졌나"
    assert not detect_corrections_only(q, task="correction_diff")
    assert detect_corrections_only(q, task="lookup")     # 유형이 다르면 적용된다
