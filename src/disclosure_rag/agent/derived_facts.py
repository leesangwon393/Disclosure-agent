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

## 원칙 — 이 파일에서 제일 중요한 부분

`▶▶` 로 나간 줄에는 "계산이 끝난 값이니 그대로 베껴라" 가 붙는다. 그래서
**틀린 `▶▶` 는 없는 것보다 훨씬 나쁘다.** 모델의 애매한 실수가 확정된
오답으로 바뀐다. 그래서 아래를 지킨다.

1. **공시에 적힌 값이 있으면 계산하지 않는다.** 계산은 공시에 없는 값을
   물었을 때만. (실측: 공시에 141,238.2 로 적혀 있는데 (A+B+C)/3 을 다시
   나눠 141,238.33 으로 답해 오답)
2. **단위가 다른 값을 섞지 않는다.** 같은 `3,112,850` 도 백만원 표와 천원
   표에서 1,000배 다르다. 주(株)와 원(₩)은 아예 더할 수 없다.
3. **회사·시점·연결/별도를 섞지 않는다.** 같은 "매출액" 이라도 다른 회사,
   다른 분기, 연결/별도는 다른 값이다.
4. **애매하면 계산하지 않는다.** 같은 회사·같은 시점에 값이 둘 이상인데
   어느 것인지 못 고르면 그 항목은 통째로 건너뛴다. 반쪽짜리 계산은 틀린
   답보다 나쁘다.

2·3·4 는 2026-09-01 교차 검수에서 실제로 뚫려 있던 구멍이다. 두 회사의
매출을 더한 합계와, 383조를 "3억 8,383만원" 이라고 적은 환산이 실제로
`▶▶` 로 나가고 있었다.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date

from disclosure_rag.common.korean_number import (
    describe_amount, normalize_unit_text, unit_multiplier,
)

# 질문이 무엇을 계산해 달라는가.
_SUM_WORDS = ("합계", "총액", "모두 합", "총합", "다 더하면", "합치면")
_AVG_WORDS = ("평균", "산술평균")
_DIFF_WORDS = ("차이", "차액", "얼마나 많", "얼마나 적", "격차")
_COUNT_WORDS = ("몇 건", "몇건", "건수", "개수", "몇 개", "몇개", "몇 차례", "몇 번", "총 몇")
_RANK_WORDS = ("순서대로", "순위", "큰 순", "작은 순", "나열")


# 다른 낱말 안에 갇힌 글자가 계산을 부르면 안 된다. "연평균성장률" 의
# "평균" 이 평균 계산을 켜서, CAGR 질문에 엉뚱한 평균이 붙었다(2026-09-01).
_MASKED_PHRASES = ("연평균성장률", "연평균 성장률", "평균성장률",
                   "가중평균", "이동평균", "평균잔존만기")


def wanted_operations(query: str) -> list[str]:
    """질문이 요구하는 파생 계산 목록."""
    for phrase in _MASKED_PHRASES:
        query = query.replace(phrase, " ")
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


# --- 행 읽기 -----------------------------------------------------------------
#
# 같은 값이 파이프라인 안에서 두 가지 키 이름으로 돌아다닌다. Facts 저장소는
# `value_unit`, 생성기로 넘어가는 dict 는 `unit` 이다(`dual_channel._fact_dict`).
# 2026-09-01 이전 코드는 `value_unit` 만 봐서 **실제 파이프라인에서는 단위를
# 한 번도 못 읽었다** — 백만원 표의 383조를 "3억 8,383만원" 으로 환산했다.

_UNIT_KEYS = ("unit", "value_unit")
_STATEMENT_SPACES = re.compile(r"[\s​　]+")


_UNIT_PREFIX = re.compile(r"^[\s(（\[]*단위\s*[:：]?\s*")
_UNIT_TAIL = re.compile(r"[\s)）\]]+$")


def unit_of(row: dict) -> str:
    """행의 단위. 표 머리 표기(`(단위: 백만원)`)는 단위 이름만 남긴다."""
    for key in _UNIT_KEYS:
        raw = row.get(key)
        if raw:
            known = normalize_unit_text(raw)
            if known:
                return known
            # 금액이 아닌 단위(주·건·명…)는 사전에 없다. 표기만 다듬어 쓴다.
            return _UNIT_TAIL.sub("", _UNIT_PREFIX.sub("", str(raw))).strip()
    return ""


def statement_kind(row: dict) -> str:
    """연결재무제표 / 별도재무제표 / 그 외. 섞으면 안 되는 축이다."""
    joined = unicodedata.normalize(
        "NFC", " > ".join(str(p) for p in (row.get("section_path") or [])))
    flat = _STATEMENT_SPACES.sub("", joined)
    if "연결재무제표" in flat:
        return "연결"
    if "재무제표" in flat:
        return "별도"
    return ""


def _label(row: dict) -> str:
    return str(row.get("item") or row.get("key_norm") or "값")


def _numeric(rows) -> list[dict]:
    return [r for r in rows if isinstance(r.get("value_num"), (int, float))]


def _amount(value: float, unit: str) -> str:
    """읽기 쉬운 금액 표기. 금액 단위가 아니면 안 붙인다."""
    if unit_multiplier(unit) is None:
        return ""
    if unit in ("", "원") and abs(value) < 10_000:
        return ""
    text = describe_amount(value, unit)
    return f" = {text}" if text else ""


def _with_unit(value: float, unit: str, *, decimals: int = 0) -> str:
    """숫자에 **항상 단위를 붙여** 적는다. 단위를 모르면 붙이지 않는다."""
    return f"{value:,.{decimals}f}{unit}"


# --- 묶기 --------------------------------------------------------------------


def group_rows(rows: list[dict]) -> dict[tuple, list[dict]]:
    """더해도 되는 것끼리만 묶는다: (항목, 단위, 연결/별도)."""
    groups: dict[tuple, list[dict]] = {}
    for row in _numeric(rows):
        groups.setdefault((_label(row), unit_of(row), statement_kind(row)), []).append(row)
    return groups


def pick_one_per_subject(rows: list[dict]) -> list[dict] | None:
    """(회사, 시점) 마다 값 하나만 남긴다. 못 고르면 `None`.

    같은 회사·같은 시점의 **가장 최근 접수분**을 쓴다. 그 최신 접수분 안에서
    값이 둘 이상 갈리면 어느 것인지 알 수 없다 — 그때는 이 항목을 통째로
    포기한다(원칙 4).
    """
    by_subject: dict[tuple, list[dict]] = {}
    for row in rows:
        by_subject.setdefault((row.get("company"), row.get("period") or ""), []).append(row)

    picked: list[dict] = []
    for group in by_subject.values():
        newest = max(str(r.get("filing_date") or "") for r in group)
        same = [r for r in group if str(r.get("filing_date") or "") == newest]
        if len({r["value_num"] for r in same}) > 1:
            return None
        picked.append(same[0])
    return picked


def _count_documents(rows: list[dict]) -> int:
    """건수는 행 수가 아니라 **문서 수**다. 같은 값이 여러 청크에서 나온다."""
    ids = {r.get("report_id") or r.get("doc_id") for r in rows}
    ids.discard(None)
    return len(ids) or len(rows)


# --- 합계·평균·차이·건수·순위 ------------------------------------------------


def derive(query: str, fact_rows: list[dict]) -> list[str]:
    """요청된 계산만 수행해 `▶▶` 줄로 돌려준다. 못 하면 빈 목록."""
    ops = wanted_operations(query)
    if not ops:
        return []

    out: list[str] = []
    for (item, unit, kind), rows in group_rows(fact_rows).items():
        name = f"{item}({kind})" if kind else item
        if "count" in ops:
            out.append(f"▶▶ {name} 건수: {_count_documents(rows)}건")

        picked = pick_one_per_subject(rows)
        if picked is None or len(picked) < 2:
            continue
        values = [r["value_num"] for r in picked]

        if "sum" in ops:
            total = sum(values)
            out.append(f"▶▶ {name} 합계: {_with_unit(total, unit)}"
                       f"{_amount(total, unit)} ({len(values)}건)")
        if "avg" in ops:
            mean = sum(values) / len(values)
            out.append(f"▶▶ {name} 평균: {_with_unit(mean, unit, decimals=2)}"
                       f"{_amount(mean, unit)} ({len(values)}건)")
        if "diff" in ops:
            gap = max(values) - min(values)
            out.append(f"▶▶ {name} 최대-최소 차이: {_with_unit(gap, unit)}{_amount(gap, unit)}")
        if "rank" in ops:
            ranked = sorted(picked, key=lambda r: r["value_num"], reverse=True)
            order = " > ".join(
                f"{r.get('company') or '?'} {_with_unit(r['value_num'], unit)}"
                for r in ranked[:8])
            out.append(f"▶▶ {name} 큰 순서: {order}")
    return out


# --- 증감률·비율·CAGR --------------------------------------------------------
# `calculation.py` 에 완성돼 있는데 지금 파이프라인(ask_v2)이 한 번도 안 불렀다.
# 옛 agent_loop 경로에만 연결돼 있었다. 파일 첫 줄에 "계산은 LLM 에 맡기지
# 않는다" 고 적혀 있는데 정작 안 쓰이고 있었다(2026-09-01 발견).

_GROWTH_WORDS = ("증감률", "증가율", "감소율", "성장률", "전년 대비", "전기 대비",
                 "얼마나 증가", "얼마나 감소", "몇 % 늘", "몇 % 줄")
_CAGR_WORDS = ("연평균성장률", "CAGR", "연평균 성장률")
_RATIO_WORDS = ("비율", "비중", "퍼센트", "몇 %", "차지하는")


def _ends_of_series(rows: list[dict]) -> tuple[dict, dict] | None:
    """시점 순 첫 값과 마지막 값. 양 끝이 애매하면 `None`.

    접수일이 같은 행이 양 끝에 여러 개인데 값이 갈리면 어느 쪽이 '전' 이고
    어느 쪽이 '후' 인지 정할 수 없다 — 그때는 계산하지 않는다. 예전 코드는
    입력 순서에 따라 증감률 부호가 뒤집혔다(2026-09-01 발견).
    """
    dated = [r for r in rows if str(r.get("filing_date") or "").isdigit()]
    if len(dated) < 2:
        return None
    first_day = min(str(r["filing_date"]) for r in dated)
    last_day = max(str(r["filing_date"]) for r in dated)
    if first_day == last_day:
        return None
    head = [r for r in dated if str(r["filing_date"]) == first_day]
    tail = [r for r in dated if str(r["filing_date"]) == last_day]
    if len({r["value_num"] for r in head}) > 1 or len({r["value_num"] for r in tail}) > 1:
        return None
    return head[0], tail[0]


def derive_calculations(query: str, fact_rows: list[dict]) -> list[str]:
    """증감률·CAGR·비율을 파이썬으로 계산해 붙인다.

    **같은 회사·같은 항목·같은 단위·같은 재무제표 구분** 안에서, 시점이
    다른 값이 둘 이상 있을 때만 한다.
    """
    from disclosure_rag.agent.calculation import calculate_cagr, calculate_growth_rate

    want_cagr = any(w in query for w in _CAGR_WORDS)
    # 연평균성장률을 물었으면 그것만 답한다. "성장률" 이 두 목록에 다 걸려
    # 증감률까지 같이 나가면 모델이 둘 중 아무거나 골라 적는다.
    want_growth = (not want_cagr) and any(w in query for w in _GROWTH_WORDS)
    want_ratio = any(w in query for w in _RATIO_WORDS)

    out: list[str] = []
    if want_growth or want_cagr:
        series: dict[tuple, list[dict]] = {}
        for row in _numeric(fact_rows):
            key = (row.get("company"), _label(row), unit_of(row), statement_kind(row))
            series.setdefault(key, []).append(row)
        for (company, item, unit, kind), rows in series.items():
            ends = _ends_of_series(rows)
            if ends is None:
                continue
            before, after = ends
            name = " ".join(x for x in (company, item, f"({kind})" if kind else "") if x)
            if want_growth:
                g = calculate_growth_rate(before["value_num"], after["value_num"])
                if g.get("growth_rate_pct") is not None:
                    out.append(
                        f"▶▶ {name} 증감률: {g['growth_rate_pct']:+.2f}% "
                        f"({_with_unit(before['value_num'], unit)} -> "
                        f"{_with_unit(after['value_num'], unit)}, "
                        f"증감 {g['abs_change']:+,.0f}{unit})")
            if want_cagr:
                years = (int(str(after["filing_date"])[:4])
                         - int(str(before["filing_date"])[:4]))
                c = calculate_cagr(before["value_num"], after["value_num"], years)
                if c.get("cagr_pct") is not None:
                    out.append(f"▶▶ {name} 연평균성장률(CAGR, {years}년): {c['cagr_pct']:+.2f}%")
    if want_ratio:
        out.extend(_derive_ratio(query, fact_rows))
    return out


# --- 비율 --------------------------------------------------------------------
#
# 비율은 **서로 다른 두 항목** 사이의 몫이다. 같은 항목의 올해/작년을 나눠
# "비율" 이라 붙이면 그건 증감률이지 비중이 아니다.
#
# 분자·분모 방향은 자리표시 낱말로 정한다. 두 항목 **사이**에 있으면
# "A 중 B" 꼴이라 뒤가 분자, 두 항목 **뒤**에 있으면 "B의 A 대비" 꼴이라
# 앞이 분자다. 둘 다 아니면 방향을 알 수 없으니 계산하지 않는다.
#
#   매출액 중 영업이익이 차지하는 비율   -> 영업이익 / 매출액
#   매출액 대비 영업이익 비율            -> 영업이익 / 매출액
#   영업이익의 매출액 대비 비율          -> 영업이익 / 매출액
#   매출액의 몇 퍼센트가 영업이익인가    -> 자리표시 낱말이 없다. 계산 안 함.

_RATIO_MARKERS = ("대비", "중", "에서", "가운데", "차지")
_HANGUL = re.compile(r"[가-힣A-Za-z0-9]")
# 항목 이름 뒤에 붙을 수 있는 조사. 긴 것부터 맞춰야 "에서" 를 "에" 로
# 잘못 끊지 않는다.
_PARTICLES = ("으로서", "이라는", "에서는", "에서의", "으로", "에서", "에게", "까지",
              "부터", "이라", "라는", "이나", "만큼", "처럼", "보다", "이란",
              "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도",
              "만", "로", "란", "나")


def _find_standalone(query: str, item: str) -> int:
    """항목 이름이 **낱말 통째로** 나오는 자리. 없으면 -1.

    부분 문자열을 그냥 받으면 질문의 "총자산" 안에서 Facts 의 별개 항목
    "자산" 이 걸린다(2026-09-01 발견). 반대로 한국어는 이름 뒤에 조사가
    붙으므로("영업이익이") 뒤 글자가 한글이라고 무조건 버려도 안 된다.
    """
    start = 0
    while True:
        pos = query.find(item, start)
        if pos < 0:
            return -1
        before_ok = pos == 0 or not _HANGUL.match(query[pos - 1])
        if before_ok and _word_ends_here(query, pos + len(item)):
            return pos
        start = pos + 1


def _word_ends_here(query: str, end: int) -> bool:
    """`end` 자리에서 낱말이 끝나는가 — 조사 하나까지는 붙어도 된다."""
    if end >= len(query) or not _HANGUL.match(query[end]):
        return True
    rest = query[end:]
    for particle in _PARTICLES:
        if rest.startswith(particle):
            after = rest[len(particle):]
            if not after or not _HANGUL.match(after[0]):
                return True
    return False


def _latest(rows: list[dict]) -> dict | None:
    dated = [r for r in rows if str(r.get("filing_date") or "").isdigit()]
    pool = dated or rows
    newest = max(str(r.get("filing_date") or "") for r in pool)
    same = [r for r in pool if str(r.get("filing_date") or "") == newest]
    if len({r["value_num"] for r in same}) > 1:
        return None
    return same[0]


def _derive_ratio(query: str, fact_rows: list[dict]) -> list[str]:
    """두 항목이 모두 질문에 적혀 있고 방향이 분명할 때만."""
    groups: dict[str, list[dict]] = {}
    for row in _numeric(fact_rows):
        groups.setdefault(_label(row), []).append(row)

    named = sorted((pos, item) for item in groups
                   if (pos := _find_standalone(query, item)) >= 0)
    if len(named) != 2:
        return []
    (p1, first), (p2, second) = named
    between = query[p1 + len(first):p2]
    after = query[p2 + len(second):]
    if any(m in between for m in _RATIO_MARKERS):
        numerator, denominator = second, first
    elif any(m in after for m in _RATIO_MARKERS):
        numerator, denominator = first, second
    else:
        return []

    num_row, den_row = _latest(groups[numerator]), _latest(groups[denominator])
    if num_row is None or den_row is None:
        return []
    # 다른 회사, 다른 시점, 다른 단위, 다른 재무제표의 값을 나누면 안 된다.
    if (num_row.get("company") or "") != (den_row.get("company") or ""):
        return []
    if (num_row.get("period") or "") != (den_row.get("period") or ""):
        return []
    if unit_of(num_row) != unit_of(den_row):
        return []
    if statement_kind(num_row) != statement_kind(den_row):
        return []

    from disclosure_rag.agent.calculation import calculate_ratio
    r = calculate_ratio(num_row["value_num"], den_row["value_num"], label=numerator)
    if r.get("ratio_pct") is None:
        return []
    unit = unit_of(num_row)
    return [f"▶▶ {denominator} 대비 {numerator} 비율: {r['ratio_pct']:.2f}% "
            f"({numerator} {_with_unit(num_row['value_num'], unit)} / "
            f"{denominator} {_with_unit(den_row['value_num'], unit)})"]


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
           "days_between", "months_between", "describe_span",
           "unit_of", "statement_kind", "group_rows", "pick_one_per_subject"]
