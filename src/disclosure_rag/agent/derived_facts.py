"""근거에서 파생되는 수치를 **파이썬이 계산**해 프롬프트에 못 박는다.

## 왜

실측(2026-09-01, suite_v2 296문항):

    최대/최소  24건  96%   <- 계산이 붙는다
    대소비교   60건  68%
    합계/총액   8건  50%
    비율        10건  40%
    평균         2건   0%

계산이 붙은 유형만 96% 다. 모델은 자릿수 큰 수의 비교·합계·평균을 자주
틀린다. 계산은 파이썬이 하고 모델은 결과를 옮겨 적기만 한다.

## 원칙

1. **공시에 적힌 값이 있으면 계산하지 않는다.** 계산은 공시에 없는 값을
   물었을 때만. (실측: 공시에 141,238.2 로 적혀 있는데 (A+B+C)/3 을 다시
   나눠 141,238.33 으로 답해 오답)
2. 근거가 부족하면 **계산하지 않는다.** 반쪽짜리 합계는 틀린 답보다 나쁘다.
"""

from __future__ import annotations

import re
from datetime import date

from disclosure_rag.common.korean_number import describe_amount

# 질문이 무엇을 계산해 달라는가. 하나만 고른다 — 여러 개면 앞의 것이 이긴다.
_SUM_WORDS = ("합계", "총액", "모두 합", "총합", "다 더하면", "합치면")
_AVG_WORDS = ("평균", "산술평균")
_DIFF_WORDS = ("차이", "차액", "얼마나 많", "얼마나 적", "격차")
_COUNT_WORDS = ("몇 건", "몇건", "건수", "개수", "몇 개", "몇개", "몇 차례", "몇 번", "총 몇")
_RANK_WORDS = ("순서대로", "순위", "큰 순", "작은 순", "나열")


def wanted_operations(query: str) -> list[str]:
    """질문이 요구하는 파생 계산 목록."""
    ops: list[str] = []
    if any(w in query for w in _SUM_WORDS):
        ops.append("sum")
    if any(w in query for w in _AVG_WORDS):
        ops.append("avg")
    if any(w in query for w in _DIFF_WORDS):
        ops.append("diff")
    if any(w in query for w in _COUNT_WORDS):
        ops.append("count")
    if any(w in query for w in _RANK_WORDS):
        ops.append("rank")
    return ops


def _numeric(rows) -> list[dict]:
    return [r for r in rows if isinstance(r.get("value_num"), (int, float))]


def _label(row: dict) -> str:
    return str(row.get("item") or row.get("key_norm") or "값")


def _amount(value: float, rows: list[dict]) -> str:
    """읽기 쉬운 금액 표기. 환산해도 달라지지 않는 작은 수에는 안 붙인다."""
    unit = next((r.get("value_unit") for r in rows if r.get("value_unit")), None)
    if unit in (None, "원") and abs(value) < 10_000:
        return ""
    text = describe_amount(value, unit or "원")
    return f" = {text}" if text else ""


def derive(query: str, fact_rows: list[dict]) -> list[str]:
    """요청된 계산만 수행해 `▶▶` 줄로 돌려준다. 못 하면 빈 목록."""
    ops = wanted_operations(query)
    if not ops:
        return []
    rows = _numeric(fact_rows)
    if not rows:
        return []

    # 항목별로 묶는다. 서로 다른 항목을 더하면 안 된다.
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(_label(row), []).append(row)

    out: list[str] = []
    for item, items in groups.items():
        values = [r["value_num"] for r in items]
        if "count" in ops:
            out.append(f"▶▶ {item} 건수: {len(items)}건")
        if "sum" in ops and len(values) >= 2:
            total = sum(values)
            out.append(f"▶▶ {item} 합계: {total:,.0f}{_amount(total, items)} ({len(values)}건)")
        if "avg" in ops and len(values) >= 2:
            mean = sum(values) / len(values)
            out.append(f"▶▶ {item} 평균: {mean:,.2f}{_amount(mean, items)} ({len(values)}건)")
        if "diff" in ops and len(values) >= 2:
            hi, lo = max(values), min(values)
            out.append(f"▶▶ {item} 최대-최소 차이: {hi - lo:,.0f}{_amount(hi - lo, items)}")
        if "rank" in ops and len(items) >= 2:
            ranked = sorted(items, key=lambda r: r["value_num"], reverse=True)
            order = " > ".join(
                f"{r.get('company') or '?'} {r['value_num']:,.0f}" for r in ranked[:8])
            out.append(f"▶▶ {item} 큰 순서: {order}")
    return out


# --- 증감률·비율·CAGR --------------------------------------------------------
# `calculation.py` 에 완성돼 있는데 지금 파이프라인(ask_v2)이 한 번도 안 불렀다.
# 옛 agent_loop 경로에만 연결돼 있었다. 파일 첫 줄에 "계산은 LLM 에 맡기지
# 않는다" 고 적혀 있는데 정작 안 쓰이고 있었다(2026-09-01 발견).

_GROWTH_WORDS = ("증감률", "증가율", "감소율", "성장률", "전년 대비", "전기 대비",
                 "얼마나 증가", "얼마나 감소", "몇 % 늘", "몇 % 줄")
_CAGR_WORDS = ("연평균성장률", "CAGR", "연평균 성장률")
_RATIO_WORDS = ("비율", "비중", "퍼센트", "몇 %", "차지하는")


def derive_calculations(query: str, fact_rows: list[dict]) -> list[str]:
    """증감률·CAGR·비율을 파이썬으로 계산해 붙인다.

    같은 항목의 값이 **시점 순으로 둘 이상** 있을 때만 한다. 근거가 모자라면
    계산하지 않는다 — 반쪽짜리 계산은 틀린 답보다 나쁘다.
    """
    from disclosure_rag.agent.calculation import calculate_cagr, calculate_growth_rate

    want_cagr = any(w in query for w in _CAGR_WORDS)
    # 연평균성장률을 물었으면 그것만 답한다. "성장률" 이 두 목록에 다 걸려
    # 증감률까지 같이 나가면 모델이 둘 중 아무거나 골라 적는다.
    want_growth = (not want_cagr) and any(w in query for w in _GROWTH_WORDS)
    want_ratio = any(w in query for w in _RATIO_WORDS)

    out: list[str] = []
    if want_growth or want_cagr:
        groups: dict[str, list[dict]] = {}
        for row in _numeric(fact_rows):
            groups.setdefault(_label(row), []).append(row)
        for item, rows in groups.items():
            # 접수일 오름차순 = 시점 순. 날짜가 없으면 계산하지 않는다.
            dated = [r for r in rows if str(r.get("filing_date") or "").isdigit()]
            dated.sort(key=lambda r: str(r["filing_date"]))
            if len(dated) < 2:
                continue
            before, after = dated[0], dated[-1]
            if want_growth:
                g = calculate_growth_rate(before["value_num"], after["value_num"])
                if g.get("growth_rate_pct") is not None:
                    out.append(
                        f"▶▶ {item} 증감률: {g['growth_rate_pct']:+.2f}% "
                        f"({before['value_num']:,.0f} -> {after['value_num']:,.0f}, "
                        f"증감 {g['abs_change']:+,.0f})")
            if want_cagr:
                years = (int(str(after["filing_date"])[:4])
                         - int(str(before["filing_date"])[:4]))
                c = calculate_cagr(before["value_num"], after["value_num"], years)
                if c.get("cagr_pct") is not None:
                    out.append(
                        f"▶▶ {item} 연평균성장률(CAGR, {years}년): {c['cagr_pct']:+.2f}%")
    if want_ratio:
        out.extend(_derive_ratio(query, fact_rows))
    return out


# 비율은 **서로 다른 두 항목** 사이의 몫이다. 같은 항목의 올해/작년을 나눠
# "비율" 이라 붙이면 그건 증감률이지 비중이 아니다(2026-09-01 수정).
_RATIO_MARKERS = ("중", "에서", "대비", "차지", "가운데")


def _latest(rows: list[dict]) -> dict:
    dated = [r for r in rows if str(r.get("filing_date") or "").isdigit()]
    if not dated:
        return rows[0]
    return max(dated, key=lambda r: str(r["filing_date"]))


def _derive_ratio(query: str, fact_rows: list[dict]) -> list[str]:
    """`A 중 B가 차지하는 비율` 처럼 두 항목이 모두 질문에 적혀 있을 때만."""
    groups: dict[str, list[dict]] = {}
    for row in _numeric(fact_rows):
        groups.setdefault(_label(row), []).append(row)

    named = sorted((query.find(item), item) for item in groups if query.find(item) >= 0)
    if len(named) != 2:
        return []
    (p1, first), (p2, second) = named
    tail = query[p1 + len(first):]
    between = query[p1 + len(first):p2]
    if tail.startswith("의"):
        numerator, denominator = first, second   # "B의 A 대비 비율"
    elif any(m in between for m in _RATIO_MARKERS):
        numerator, denominator = second, first   # "A 중 B가 차지하는 비율"
    else:
        return []                                 # 방향이 불분명하면 계산하지 않는다

    num_row, den_row = _latest(groups[numerator]), _latest(groups[denominator])
    # 다른 회사, 다른 시점의 값을 나누면 안 된다.
    if num_row.get("company") and den_row.get("company"):
        if num_row["company"] != den_row["company"]:
            return []
    if num_row.get("period") and den_row.get("period"):
        if num_row["period"] != den_row["period"]:
            return []
    if num_row.get("value_unit") and den_row.get("value_unit"):
        if num_row["value_unit"] != den_row["value_unit"]:
            return []

    from disclosure_rag.agent.calculation import calculate_ratio
    r = calculate_ratio(num_row["value_num"], den_row["value_num"], label=numerator)
    if r.get("ratio_pct") is None:
        return []
    return [f"▶▶ {denominator} 대비 {numerator} 비율: {r['ratio_pct']:.2f}% "
            f"({numerator} {num_row['value_num']:,.0f} / "
            f"{denominator} {den_row['value_num']:,.0f})"]


# --- 날짜 --------------------------------------------------------------------
# "계약기간이 며칠인가", "공시일로부터 몇 개월" 은 모델이 자주 틀린다.

_DATE_RE = re.compile(r"(20\d{2})[.\-/년]\s*(\d{1,2})[.\-/월]\s*(\d{1,2})")


def parse_dates(text: str) -> list[date]:
    """본문에서 날짜를 모두 뽑는다. 해석 실패는 버린다."""
    out: list[date] = []
    for m in _DATE_RE.finditer(text or ""):
        try:
            out.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            continue
    return out


def days_between(start: date, end: date) -> int:
    return (end - start).days


def months_between(start: date, end: date) -> int:
    """달 수. 일자가 모자라면 내림한다 (1/31 -> 2/28 은 0개월)."""
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return months


def describe_span(start: date, end: date) -> str:
    days = days_between(start, end)
    months = months_between(start, end)
    return (f"▶▶ 기간: {start.isoformat()} ~ {end.isoformat()} "
            f"= {days:,}일 ({months}개월)")


__all__ = ["wanted_operations", "derive", "derive_calculations", "parse_dates",
           "days_between", "months_between", "describe_span"]
