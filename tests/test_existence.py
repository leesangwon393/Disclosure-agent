"""존재 전수 확인 테스트 — '없다'와 '모른다'를 가르는 장치.

실측 배경(results/v2_off4, 2026-08-30): 이 유형 2문항이 전부 "확인할 수
없습니다"로 나갔고 정답은 둘 다 "아니오"였다. 검색은 상위 k건만 보므로
거기 없다고 부재를 단정할 수 없다. manifest 전수 조회가 그걸 가능하게 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from disclosure_rag.agent.existence import (  # noqa: E402
    check_existence, detect_event, is_existence_question, norm_key,
)


def _row(corp, nm, dt="20240101", doc_id="", is_correction=False):
    return SimpleNamespace(corp_name=corp, report_nm=nm, rcept_dt=dt,
                           doc_id=doc_id or f"exchange_{dt}000001",
                           is_correction=is_correction)


# 한미반도체 실제 구성을 축약한 것. 해지 공시가 7건 있지만 전부
# 자기주식취득신탁계약 해지고, 단일판매·공급계약 해지는 0건이다.
HANMI = [
    _row("한미반도체", "단일판매ㆍ공급계약체결", "20230901"),
    _row("한미반도체", "단일판매ㆍ공급계약체결(자율공시)", "20230612"),
    _row("한미반도체", "주요사항보고서(자기주식취득신탁계약해지결정)", "20240417"),
    _row("한미반도체", "사업보고서 (2024.12)", "20250310"),
]


def test_existence_question_is_detected_by_ending():
    assert is_existence_question("해지된 계약이 존재하는가?")
    assert is_existence_question("정정된 내역이 있는가?")
    assert is_existence_question("유상증자를 한 적이 있나요?")
    assert not is_existence_question("순자산액은 얼마인가?")
    assert not is_existence_question("주요 내용을 정리해줘.")


def test_event_word_is_picked_from_query():
    assert detect_event("이후 해지된 계약이 존재하는가") == "해지"
    assert detect_event("정정된 내역이 있는가") == "정정"
    assert detect_event("계약을 체결했는가") == ""


def test_middle_dot_variants_normalize_to_the_same_key():
    """코퍼스는 `ㆍ`(U+318D), 질문은 `·`(U+00B7)를 쓴다. 안 지우면 절대 안 맞는다."""
    assert norm_key("단일판매ㆍ공급계약체결") == norm_key("단일판매·공급계약 체결")
    assert norm_key("단일판매ㆍ공급계약체결(자율공시)") != norm_key("단일판매")


def test_absence_is_confirmed_not_unknown():
    """핵심 케이스. 계약 공시는 있는데 해지 공시가 0건이면 '아니오'다."""
    r = check_existence("한미반도체가 체결한 단일판매·공급계약 중 이후 해지된 계약이 존재하는가?",
                        HANMI, companies=["한미반도체"],
                        report_kinds=["단일판매ㆍ공급계약체결"])
    assert r.applicable
    assert r.verdict == "아니오"
    assert len(r.matches) == 0
    assert len(r.related) == 2          # 계약 체결 공시 2건은 근거로 남는다
    assert r.scanned == 4


def test_absence_block_tells_the_model_to_say_no():
    r = check_existence("한미반도체가 체결한 단일판매·공급계약 중 이후 해지된 계약이 존재하는가?",
                        HANMI, companies=["한미반도체"],
                        report_kinds=["단일판매ㆍ공급계약체결"])
    block = r.prompt_block()
    assert "전수 확인" in block and "0건" in block
    assert "없습니다" in block and "확인할 수 없습니다" in block  # 하지 말라는 지시로


def test_unrelated_event_disclosures_do_not_count():
    """`해지` 라는 낱말만 보고 세면 자기주식 신탁 해지 7건이 잡혀 '예'가 된다."""
    r = check_existence("한미반도체가 체결한 단일판매·공급계약 중 이후 해지된 계약이 존재하는가?",
                        HANMI, companies=["한미반도체"],
                        report_kinds=["단일판매ㆍ공급계약체결"])
    assert r.verdict == "아니오"


def test_presence_is_reported_with_documents():
    r = check_existence("한미반도체가 자기주식취득신탁계약을 해지한 적이 있는가?",
                        HANMI, companies=["한미반도체"])
    assert r.verdict == "예"
    assert len(r.matches) == 1
    assert "예" in r.prompt_block()


def test_correction_flag_counts_even_if_name_lacks_the_word():
    rows = [_row("A사", "주요사항보고서(유상증자결정)", is_correction=True)]
    r = check_existence("A사의 공시가 정정된 내역이 있는가?", rows, companies=["A사"])
    assert r.verdict == "예"


def test_not_applicable_for_two_companies():
    """비교 질문은 여기서 다루지 않는다 — 한쪽만 보고 단정하면 위험하다."""
    r = check_existence("A사와 B사 중 해지된 계약이 있는가?", HANMI,
                        companies=["A사", "B사"])
    assert not r.applicable


def test_not_applicable_when_company_absent_from_manifest():
    """코퍼스 밖 회사는 범위 게이트 소관이다. 여기서 '없다'고 하면 안 된다."""
    r = check_existence("없는회사의 계약이 해지된 적이 있는가?", HANMI,
                        companies=["없는회사"])
    assert not r.applicable


def test_not_applicable_for_non_existence_question():
    r = check_existence("한미반도체의 계약금액은 얼마인가?", HANMI, companies=["한미반도체"])
    assert not r.applicable
    assert r.prompt_block() == ""


def test_not_applicable_when_target_cannot_be_identified():
    """무엇의 존재를 묻는지 특정 못 하면 '없다'고 단정하지 않는다."""
    r = check_existence("한미반도체에 그런 게 있는가?", HANMI, companies=["한미반도체"])
    assert not r.applicable
    assert r.prompt_block() == ""
