"""정답셋 청소 규칙 테스트.

규칙이 과하면 어려운 문항만 지워져 점수가 공짜로 오른다. 그래서
"지워야 하는 것"만큼 **"지우면 안 되는 것"** 을 같은 비중으로 본다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from clean_gold import judge  # noqa: E402


def _row(query, key, company="셀트리온", answer="1,234,567"):
    return {"query": query, "key": key, "company": company,
            "answers": [answer], "answer": answer, "gold_report_ids": ["periodic_1"]}


# --- CROSS_ENTITY: 다른 법인 표에서 뽑힌 정답 ------------------------------

def test_other_company_table_is_dropped():
    """실측 사례: 삼성중공업 질문의 정답이 삼성전자 수치였다."""
    doc = "구 분 법인 또는 단체의 명칭 삼성전자주식회사 자산총계 448,424,507 부채총계 93,674,903"
    row = _row("삼성중공업 반기보고서 (2023.06) 자산총계 얼마야?", "자산총계",
               company="삼성중공업", answer="448,424,507")
    verdict = judge(row, doc, threshold=3)
    assert verdict is not None and verdict[0] == "CROSS_ENTITY"
    assert "삼성전자" in verdict[1]


def test_own_company_table_is_kept():
    doc = "구 분 법인 또는 단체의 명칭 삼성중공업 자산총계 448,424,507"
    row = _row("삼성중공업 반기보고서 (2023.06) 자산총계 얼마야?", "자산총계",
               company="삼성중공업", answer="448,424,507")
    assert judge(row, doc, threshold=3) is None


# --- AMBIGUOUS: 무엇을 묻는지 정할 수 없는 질문 ----------------------------

def test_generic_label_without_qualifier_is_dropped():
    doc = "합계 100 합계 200 합계 300 합계 400"
    row = _row("셀트리온 합계이(가) 어떻게 돼?", "합계")
    verdict = judge(row, doc, threshold=3)
    assert verdict is not None and verdict[0] == "AMBIGUOUS"


def test_same_question_with_a_named_disclosure_is_kept():
    """공시를 지정하면 답할 수 있다 — 지우면 안 된다."""
    doc = "합계 100 합계 200 합계 300 합계 400"
    row = _row("셀트리온 사업보고서 (2025.12) 합계 얼마야?", "합계")
    assert judge(row, doc, threshold=3) is None


def test_year_alone_counts_as_a_qualifier():
    doc = "합계 100 합계 200 합계 300 합계 400"
    row = _row("셀트리온 2025년 합계이(가) 어떻게 돼?", "합계")
    assert judge(row, doc, threshold=3) is None


def test_unique_label_without_qualifier_is_kept():
    """'순자산액' 처럼 문서에 하나뿐인 항목은 공시를 안 적어도 답할 수 있다."""
    doc = "가. 순자산액: 224,787,773,988,054 원"
    row = _row("셀트리온 순자산액이(가) 어떻게 돼?", "순자산액")
    assert judge(row, doc, threshold=3) is None


def test_threshold_is_respected():
    doc = "소계 1 소계 2"          # 2회 — threshold 3 미만
    row = _row("셀트리온 소계이(가) 어떻게 돼?", "소계")
    assert judge(row, doc, threshold=3) is None
    assert judge(row, doc, threshold=2) is not None


def test_answer_absent_from_document_is_not_dropped_as_cross_entity():
    """원문에서 정답을 못 찾으면 판정 근거가 없다 — 함부로 지우지 않는다."""
    doc = "법인 또는 단체의 명칭 삼성전자주식회사 자산총계 999"
    row = _row("삼성중공업 반기보고서 자산총계 얼마야?", "자산총계",
               company="삼성중공업", answer="448,424,507")
    assert judge(row, doc, threshold=3) is None
