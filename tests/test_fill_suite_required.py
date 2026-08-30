"""서술형 정답(required_all) 자동 생성 규칙.

수치사전에는 `참석 = 5`(이사회 참석 인원) 같은 값도 들어 있다. 이런 게
필수 항목에 들어가면 **정답을 오답으로 만든다.** 그래서 키 화이트리스트로
거르고, 걸리는 게 없으면 문항을 채점 불가로 그대로 둔다.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fill_suite_v2_required import _iso, _usable, build, collect  # noqa: E402


@pytest.fixture
def cur():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE facts (doc_id TEXT, key_norm TEXT, value_text TEXT, "
                "value_num REAL, value_date TEXT)")
    con.executemany("INSERT INTO facts VALUES (?,?,?,?,?)", [
        ("d1", "계약금액", "1,011,120,000,000", 1011120000000.0, None),
        ("d1", "매출액대비", "13.51", 13.51, None),
        ("d1", "참석", "5", 5.0, None),                     # 이사회 참석 인원 — 잡음
        ("d1", "1주당액면가액", "5,000", 5000.0, None),        # 액면가 — 잡음
        ("d1", "계약(수주)일자", "2023-06-24", None, "20230624"),
        ("d1", "주요내용", "아주 긴 서술 " * 20, None, None),   # 긴 텍스트 — 토큰 부적합
        ("d2", "시설자금", "570,000,000,000", 570000000000.0, None),
    ])
    return con.cursor()


def test_blocked_keys_never_become_required(cur):
    got = collect(cur, ["d1"], {"계약금액", "매출액대비", "참석", "1주당액면가액"}, set())
    assert "5" not in got and "5,000" not in got
    assert "1,011,120,000,000" in got


def test_only_whitelisted_or_hinted_keys_are_used(cur):
    assert collect(cur, ["d1"], set(), set()) == []
    assert collect(cur, ["d1"], set(), {"계약금액"}) == ["1,011,120,000,000"]


def test_dates_are_normalized_to_iso(cur):
    got = collect(cur, ["d1"], set(), {"계약(수주)일자"})
    assert got == ["2023-06-24"]
    assert _iso("20230624") == "2023-06-24"
    assert _iso("2023") == ""


def test_long_text_is_not_a_usable_token(cur):
    assert collect(cur, ["d1"], set(), {"주요내용"}) == []


def test_short_numbers_are_rejected():
    """두 자리 수는 답변 어디에나 우연히 나온다."""
    assert _usable("5", 5.0, None) == ""
    assert _usable("100", 100.0, None) == "100"
    assert _usable("13.51", 13.51, None) == "13.51"   # 소수는 자릿수 예외


def test_correction_uses_the_final_values_and_changed_fields(cur):
    row = {"generator": "correction", "gold_doc_ids": ["d1"],
           "changed_fields": ["계약금액", "매출액대비"],
           "before": {"계약금액": "1,011,120,000,000", "매출액대비": "13.51"},
           "after": {"계약금액": "1,189,540,000,000", "매출액대비": "15.89"}}
    got = build(row, cur)
    assert "1,189,540,000,000" in got and "15.89" in got   # 최종 정정본 값
    assert "1,011,120,000,000" not in got                  # 원본 값은 필수가 아니다
    assert "계약금액" in got                                 # 바뀐 항목명


def test_funding_uses_purpose_amounts(cur):
    row = {"generator": "funding", "gold_doc_ids": ["d2"], "expected_fields_hint": []}
    assert build(row, cur) == ["570,000,000,000"]


def test_row_stays_ungradeable_when_nothing_usable(cur):
    """억지로 만들지 않는다 — 잘못된 정답보다 '채점 불가'가 낫다."""
    row = {"generator": "funding", "gold_doc_ids": ["d1"], "expected_fields_hint": []}
    assert build(row, cur) == []


def test_token_count_is_capped(cur):
    row = {"generator": "correction", "changed_fields": [f"f{i}" for i in range(10)],
           "after": {f"k{i}": f"{i}00,000,000" for i in range(10)}, "gold_doc_ids": []}
    assert len(build(row, cur)) <= 5


def test_real_suite_is_still_well_formed():
    """실제 파일이 깨지지 않았는지 — 문항 수와 기존 정답 보존."""
    import json
    p = Path(__file__).resolve().parents[1] / "eval" / "suite_v2.jsonl"
    if not p.exists():
        pytest.skip("suite_v2 없음")
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 296
    gradeable = [r for r in rows if (r.get("answer") or "").strip() or r.get("required_all")]
    assert len(gradeable) >= 276
    # 자동 정답이 있는 문항에 문자열 정답이 같이 들어가면 안 된다(중복 채점)
    for r in rows:
        if r.get("required_all"):
            assert not (r.get("answer") or "").strip(), r["id"]
