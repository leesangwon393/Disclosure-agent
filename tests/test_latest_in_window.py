"""질문이 특정 공시를 지목했을 때의 버전 선택 (2026-09-01).

실측: 정답 문서가 코퍼스 전체의 최신본이 **아닌** 22문항의 정답률이 27%,
최신본인 214문항은 81%였다. 현대건설 계약 하나가 판본 15개인데 최신 1건만
남기고 14개를 버려서, "2024년 05월 그 공시" 를 물으면 답할 게 없었다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from disclosure_rag.agent.query_plan import decide_latest_policy, pinpoints_a_filing  # noqa: E402
from disclosure_rag.agent.version_dedup import deduplicate_versions  # noqa: E402
from disclosure_rag.retrieval.metadata_filter import normalize_period_tokens  # noqa: E402


class _Doc:
    """현대건설 정정 체인의 실제 판본 형태."""

    def __init__(self, report_id, filing_date, is_latest=False):
        self.report_id = report_id
        self.filing_date = filing_date
        self.correction_group_id = "exchange_20230626800002"
        self.correction_order = None      # Facts 행에는 이 컬럼이 없다
        self.is_latest = is_latest
        self.period = None


# 실제 체인 (일부). 최신은 2026-01-20 하나뿐이다.
CHAIN = [
    _Doc("exchange_20230626800002", "20230626"),
    _Doc("exchange_20240215801246", "20240215"),
    _Doc("exchange_20240522800200", "20240522"),
    _Doc("exchange_20240523800361", "20240523"),
    _Doc("exchange_20240524800345", "20240524"),
    _Doc("exchange_20240614800515", "20240614"),
    _Doc("exchange_20241128800562", "20241128"),
    _Doc("exchange_20260120800597", "20260120", is_latest=True),
]


# --------------------------------------------------------------------------- 정책 판정

def test_a_question_that_names_year_and_month_pinpoints_a_filing():
    assert pinpoints_a_filing("현대건설의 2024년 05월 [기재정정]단일판매ㆍ공급계약체결에 기재된 최근매출액은?")
    assert pinpoints_a_filing("LG에너지솔루션의 2025년 12월 단일판매ㆍ공급계약체결에 기재된 계약금액은?")
    assert not pinpoints_a_filing("현대건설의 계약금액은 얼마인가?")
    assert not pinpoints_a_filing("삼성전자의 2024년 매출액은?")


def test_policy_switches_only_when_a_filing_is_pinpointed():
    assert decide_latest_policy("현대건설의 2024년 05월 [기재정정]단일판매ㆍ공급계약체결") == "latest_in_window"
    assert decide_latest_policy("현대건설의 계약금액은?") == "latest_only"
    # 최초/최종 비교 질문은 그대로 유지된다
    assert decide_latest_policy("공시가 정정된 내역이 있는가? 최초 공시와 최종 정정본을 비교") == "first_and_final"


def test_the_month_survives_normalisation():
    """이게 안 되면 5월 안에서 고르는 것 자체가 불가능하다."""
    assert normalize_period_tokens("2024년 05월") == ["2024-05"]
    assert normalize_period_tokens("2024년 5월") == ["2024-05"]
    # 분기 표현을 월로 잘못 읽으면 안 된다
    assert normalize_period_tokens("2024년 1분기") == ["2024-03"]


# --------------------------------------------------------------------------- 버전 선택

def test_picks_the_latest_within_the_month_the_question_named():
    """5월에 판본이 셋이면 5월 안에서 최신(05-24)을 고른다."""
    kept, _report = deduplicate_versions(CHAIN, "latest_in_window", periods=["2024-05"])
    assert [d.report_id for d in kept] == ["exchange_20240524800345"]


def test_without_the_window_it_still_picks_the_corpus_wide_latest():
    """기간을 안 주면 기존과 같다 — 체인 전체의 최신."""
    kept, _report = deduplicate_versions(CHAIN, "latest_only")
    assert [d.report_id for d in kept] == ["exchange_20260120800597"]


def test_falls_back_when_nothing_is_in_the_window():
    """지목한 기간에 판본이 없으면 근거를 0건으로 만들지 않고 최신으로 물러선다."""
    kept, _report = deduplicate_versions(CHAIN, "latest_in_window", periods=["2019-03"])
    assert [d.report_id for d in kept] == ["exchange_20260120800597"]


def test_year_window_picks_the_latest_of_that_year():
    kept, _report = deduplicate_versions(CHAIN, "latest_in_window", periods=["2024"])
    assert [d.report_id for d in kept] == ["exchange_20241128800562"]


def test_filing_date_is_used_when_correction_order_is_missing():
    """Facts 행에는 correction_order 컬럼이 아예 없다.

    예전에는 전부 0 이 되어 "가장 최신을 남긴다" 는 대비책이 아무거나 남겼다.
    """
    from disclosure_rag.agent.version_dedup import _order
    assert _order(CHAIN[0]) == (0, 20230626)
    assert _order(CHAIN[-1]) == (0, 20260120)


def test_a_correction_order_is_never_compared_against_a_filing_date():
    """`correction_order=2` 와 `filing_date=20240508` 을 같은 축에서 견주면
    접수일 쪽이 무조건 이긴다. 척도를 앞자리에 둬 같은 척도끼리만 견준다."""
    from disclosure_rag.agent.version_dedup import _order
    ordered = _order({"correction_order": 2, "filing_date": "20200101"})
    dated = _order({"filing_date": "20240508"})
    assert ordered > dated


def test_an_unnormalized_period_token_still_finds_its_window():
    """`2024년 05월` 이 그대로 들어오면 예전에는 전부 창 밖으로 판정됐다."""
    from disclosure_rag.agent.version_dedup import _in_window
    chunk = {"filing_date": "20240508", "period": ""}
    assert _in_window(chunk, ["2024년 05월"])
    assert _in_window(chunk, ["2024-05"])
    assert _in_window(chunk, ["2024년"])
    assert not _in_window(chunk, ["2024년 06월"])
