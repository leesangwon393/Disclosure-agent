"""평가셋 생성기의 품질 관문.

정답셋이 오염되면 그 위의 모든 판단이 틀린다. 과거에 사후 정리로 결함 28건을
걷어냈는데(모호 22 + 교차주체 6), 사후 정리는 이미 오염된 걸 줄일 뿐이다.
그래서 관문을 생성기 안에 뒀고, 여기서 그 관문들을 하나씩 고정한다.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_eval_suite_v2 import (  # noqa: E402
    CAPS,
    STRUCTURAL_KEYS,
    Suite,
    clean_key,
    gen_correction,
    gen_lookup,
    gen_termination,
    is_bad_key,
    josa,
    norm_report,
    subtype_of,
    unique_value,
    value_is_distinctive,
)


# --------------------------------------------------------------------------- 항목명 관문

def test_structural_labels_are_rejected():
    """'합계'가 한 문서에 66개 있는 사례가 실제로 있다 — 무엇을 묻는지 정해지지 않는다."""
    for key in ("합계", "소계", "구분", "당기", "전기", "기타"):
        assert is_bad_key(key) == "구조라벨", key


def test_value_shaped_key_is_rejected():
    """facts 추출기가 값을 항목명으로 잡은 흔적."""
    assert is_bad_key("△298,342") == "값이항목명"
    assert is_bad_key("2023년03월31일") == "값이항목명"


def test_short_or_non_hangul_key_is_rejected():
    assert is_bad_key("EBITDA") == "한글2자미만"
    assert is_bad_key("") == "빈항목명"


def test_real_account_names_pass():
    for key in ("계약금액", "당기순이익", "자기주식취득금액한도", "투자목적"):
        assert is_bad_key(key) is None, key


def test_numbering_prefix_is_stripped():
    assert clean_key("가. 계약금액") == "계약금액"
    assert clean_key("1) 투자금액") == "투자금액"
    assert clean_key("(2) 자기자본") == "자기자본"


# --------------------------------------------------------------------------- 값 관문

def _db(tmp_path, rows):
    """rows = [(doc_id, company, key_norm, value_text)]"""
    p = tmp_path / "f.sqlite"
    db = sqlite3.connect(p)
    db.execute("CREATE TABLE facts (doc_id TEXT, company TEXT, doc_group TEXT, "
               "key_norm TEXT, value_text TEXT, value_num REAL, filing_date TEXT, "
               "is_latest INT, is_correction INT, correction_group_id TEXT, report_name TEXT)")
    db.executemany(
        "INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [(d, c, "exchange", k, v, float(v.replace(",", "")), "20240101", 1, 0, d, "공시")
         for d, c, k, v in rows])
    db.commit()
    return db


def test_ambiguous_value_is_rejected(tmp_path):
    """같은 (회사,문서,항목)에 값이 둘이면 질문이 답을 특정하지 못한다.

    periodic 은 이 비율이 61.4% 다(연결/별도 재무제표). 반드시 막아야 한다.
    """
    db = _db(tmp_path, [("d1", "A사", "자산총계", "100"),
                        ("d1", "A사", "자산총계", "200")])
    assert unique_value(db, "A사", "d1", "자산총계") is None


def test_single_value_passes(tmp_path):
    db = _db(tmp_path, [("d1", "A사", "자산총계", "100")])
    assert unique_value(db, "A사", "d1", "자산총계") == "100"


def test_value_shared_with_another_key_is_rejected(tmp_path):
    """다른 항목이 같은 값을 가지면, 모델이 엉뚱한 항목을 보고 답해도
    채점기(숫자 일치)가 정답 처리한다 — 성능이 부풀려진다."""
    db = _db(tmp_path, [("d1", "A사", "주당액면가액", "5,000"),
                        ("d1", "A사", "액면가", "5,000")])
    assert not value_is_distinctive(db, "d1", "주당액면가액", "5,000")


def test_distinctive_value_passes(tmp_path):
    db = _db(tmp_path, [("d1", "A사", "계약금액", "1,234,567"),
                        ("d1", "A사", "최근매출액", "999")])
    assert value_is_distinctive(db, "d1", "계약금액", "1,234,567")


# --------------------------------------------------------------------------- 교차주체

def test_cross_entity_is_rejected(tmp_path):
    """삼성중공업이 제출한 문서의 값을 삼성전자에게 물으면 안 된다.
    과거 정답셋 결함 6건이 전부 이 형태였다."""
    db = _db(tmp_path, [("d1", "삼성전자", "계약금액", "1,000,000")])
    doc_of = {"d1": {"doc_id": "d1", "corp_name": "삼성중공업", "report_nm": "공시",
                     "rcept_dt": "20240101", "doc_group": "exchange"}}
    suite = Suite()
    import random
    gen_lookup(suite, db, doc_of, tag="lookup_form",
               doc_groups=("exchange",), target=5, rng=random.Random(0))
    assert suite.rows == []
    assert suite.rejected["lookup_form:교차주체"] == 1


# --------------------------------------------------------------------------- 중복

def test_duplicate_question_is_rejected():
    suite = Suite()
    doc = [{"doc_id": "d1", "corp_name": "A사", "report_nm": "공시", "rcept_dt": "20240101",
            "doc_group": "exchange"}]
    kw = dict(task_type="검색·정보추출", mode="closed", gold_docs=doc, answer="1",
              answer_source="auto_facts", check_points=["x"], tag="t")
    assert suite.add(query="같은 질문", **kw) is True
    assert suite.add(query="같은 질문", **kw) is False
    assert len(suite.rows) == 1
    assert suite.rejected["t:질문중복"] == 1


# --------------------------------------------------------------------------- 쏠림 상한

def test_caps_are_per_generator():
    """항목명이 상수인 생성기(해지·정정·자금조달)에 항목 상한을 씌우면
    그 상수 하나로 유형 전체가 12문항에서 잘린다 — 실제로 그랬다."""
    for tag in ("funding", "termination", "correction"):
        assert CAPS[tag][1] is None, tag
    assert CAPS["lookup_form"][1] is not None


def test_company_cap_blocks_overuse():
    suite = Suite()
    tag = "lookup_form"
    limit = CAPS[tag][0]
    for _ in range(limit):
        assert suite.room_for("A사", "계약금액", tag=tag)
        suite.per_company[(tag, "A사")] += 1
    assert not suite.room_for("A사", "계약금액", tag=tag)
    assert suite.rejected[f"{tag}:회사쏠림"] == 1


# --------------------------------------------------------------------------- 음성 사례

def test_termination_includes_negative_cases():
    """양성만 넣으면 '있습니다'라고 항상 답해도 만점이라 할루시네이션을 못 잡는다."""
    import random
    docs = ([{"doc_id": f"p{i}", "corp_name": f"양성{i}", "report_nm": "단일판매ㆍ공급계약체결",
              "rcept_dt": "20240101", "doc_group": "exchange"} for i in range(3)]
            + [{"doc_id": f"pe{i}", "corp_name": f"양성{i}", "report_nm": "단일판매ㆍ공급계약해지",
                "rcept_dt": "20240201", "doc_group": "exchange"} for i in range(3)]
            + [{"doc_id": f"n{i}", "corp_name": f"음성{i}", "report_nm": "단일판매ㆍ공급계약체결",
                "rcept_dt": "20240101", "doc_group": "exchange"} for i in range(3)])
    suite = Suite()
    gen_termination(suite, docs, target=6, rng=random.Random(0))
    pol = [r["polarity"] for r in suite.rows]
    assert pol.count("positive") >= 1 and pol.count("negative") >= 1
    for r in suite.rows:
        expect = "예" if r["polarity"] == "positive" else "아니오"
        assert r["answer"].startswith(expect)


# --------------------------------------------------------------------------- 조사

def test_josa_is_correct():
    assert josa("한화오션", "가", "이") == "이"        # 받침 있음
    assert josa("셀트리온", "가", "이") == "이"
    assert josa("한국항공우주", "가", "이") == "가"    # 받침 없음
    assert josa("HD현대일렉트릭", "와", "과") == "과"


# --------------------------------------------------------------------------- 공시명 정규화

def test_report_name_normalization():
    assert norm_report("[기재정정]단일판매ㆍ공급계약체결") == "단일판매ㆍ공급계약체결"
    assert subtype_of("주요사항보고서(자기주식 취득 결정)") == "자기주식취득결정"
