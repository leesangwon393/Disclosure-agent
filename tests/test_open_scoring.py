"""서술형(open) 답변 채점 — 정답 문장 없이.

suite_v1 38문항 중 16문항이 정답 문장이 비어 있고 **전부 open 형**이다.
그래서 지금까지 서술형 답변을 한 번도 채점하지 못했다(전체의 42%).

여기서 잡아야 하는 건 특히 이것 하나다:
    요구 항목을 **조용히 빼는 것**. 답변은 그럴듯한데 읽는 사람은
    다 확인된 것으로 오해한다.
"""
from __future__ import annotations

from disclosure_rag.agent.open_scoring import cited_report_ids, score_open_answer

FIELDS = ["투자금액", "투자목적", "투자기간", "자기자본"]


# --------------------------------------------------------------------------- 항목 커버리지

def test_all_fields_with_values():
    a = ("- 투자금액: 1,200,000,000원\n- 투자목적: 생산능력 확대\n"
         "- 투자기간: 2024-01-01 ~ 2025-12-31\n- 자기자본: 5,000,000,000원")
    s = score_open_answer(a, required_fields=FIELDS)
    assert s.field_coverage == 1.0
    assert s.silent == []


def test_silently_dropped_field_is_the_worst_case():
    """항목을 빼면 읽는 사람이 다 확인된 것으로 오해한다."""
    a = "- 투자금액: 1,200,000,000원\n- 투자목적: 생산능력 확대"
    s = score_open_answer(a, required_fields=FIELDS)
    assert set(s.silent) == {"투자기간", "자기자본"}
    assert s.silent_omission_rate == 0.5
    assert s.field_coverage == 0.5


def test_acknowledged_gap_is_not_counted_as_silent():
    """'(확인되지 않음)'이라고 밝힌 건 조용히 뺀 것과 다르다 — 프롬프트가
    그렇게 하라고 지시한다."""
    a = ("- 투자금액: 1,200,000,000원\n- 투자목적: 생산능력 확대\n"
         "- 투자기간: (확인되지 않음)\n- 자기자본: (확인되지 않음)")
    s = score_open_answer(a, required_fields=FIELDS)
    assert s.silent == []
    assert set(s.acknowledged) == {"투자기간", "자기자본"}
    assert s.field_coverage == 0.5          # 값이 없으니 커버는 아니다
    assert s.silent_omission_rate == 0.0    # 정직성은 만점


def test_field_name_without_a_value_is_not_covered():
    a = "투자금액"
    assert score_open_answer(a, required_fields=["투자금액"]).covered == []


def test_prose_field_counts_without_a_number():
    """투자목적 같은 서술형 항목은 숫자가 없어도 제시된 것이다."""
    s = score_open_answer("투자목적은 생산능력 확대를 위한 것입니다",
                          required_fields=["투자목적"])
    assert s.covered == ["투자목적"]


def test_spacing_in_the_answer_does_not_matter():
    s = score_open_answer("투자 금액 : 1,000원", required_fields=["투자금액"])
    assert s.covered == ["투자금액"]


def test_no_required_fields_means_no_score():
    """Field Schema 가 모르는 유형이면 커버리지를 재지 않는다 — 0%로 찍으면
    모른다는 사실이 '나쁘다'로 오독된다."""
    s = score_open_answer("아무 답변", required_fields=[])
    assert s.field_coverage is None and s.silent_omission_rate is None


# --------------------------------------------------------------------------- 근거 정확성

def test_cited_ids_are_extracted_in_order():
    a = "근거: report_id(major_20241118000328), report_id(exchange_20240607800056)"
    assert cited_report_ids(a) == ["major_20241118000328", "exchange_20240607800056"]


def test_duplicate_citations_count_once():
    a = "major_20241118000328 ... major_20241118000328"
    assert cited_report_ids(a) == ["major_20241118000328"]


def test_citation_recall_and_precision():
    a = "근거: major_1, major_2, major_999"
    s = score_open_answer(a, required_fields=[], gold_doc_ids=["major_1", "major_2", "major_3"])
    assert s.citation_recall == 2 / 3          # gold 3건 중 2건 인용
    assert s.citation_precision == 2 / 3       # 인용 3건 중 2건이 gold


def test_no_citation_gives_zero_recall_and_no_precision():
    s = score_open_answer("근거 없음", required_fields=[], gold_doc_ids=["major_1"])
    assert s.citation_recall == 0.0
    assert s.citation_precision is None        # 인용이 없으면 정밀도는 정의 불가


def test_no_gold_docs_means_no_citation_score():
    s = score_open_answer("major_1", required_fields=[], gold_doc_ids=[])
    assert s.citation_recall is None


# --------------------------------------------------------------------------- 실제 형태

def test_real_answer_shape_from_v2_run():
    """v2 실행에서 실제로 나온 형태."""
    a = ("네, LG에너지솔루션이 체결한 단일판매·공급계약 중 이후 해지된 계약이 존재합니다.\n"
         "1. Ford Motor Company와의 전기차 배터리 공급계약\n"
         "   - 해지금액: 9,603,075,000,000원\n"
         "   - 해지주요사유: 발주처 사정\n"
         "근거: report_id(exchange_20251217800111)")
    s = score_open_answer(a, required_fields=["해지금액", "해지주요사유", "해지계약명"],
                          gold_doc_ids=["exchange_20251217800111"])
    assert set(s.covered) == {"해지금액", "해지주요사유"}
    assert s.silent == ["해지계약명"]
    assert s.citation_recall == 1.0
