"""한국 공시 숫자 표기 — 읽기와 환산.

## 왜 필요한가

공시 표에는 숫자만 적히고 단위는 표 머리의 별도 줄에 있다.

    (단위: 백만원)
    영업비용   3,112,850

`3,112,850` 만 떼어 근거로 주면 모델이 단위를 모른다. 실측(2026-09-01):
숫자를 묻는 문항 189건 중 33건(17%)이 단위 없이 답했고, 크래프톤 영업비용은
정답 `3,112,850`(백만원 표) 대신 다른 표의 `255,698,325천원` 을 답했다.

같은 `3,112,850` 이라도
    (단위: 백만원) 표면  ->  3조 1,128억 5천만원
    (단위: 천원)   표면  ->  31억 1,285만원
**1,000배 차이다.**

## 왜 파이썬이 하나

자릿수 큰 수의 환산은 모델이 자주 틀린다. 계산은 파이썬이 하고 모델은 그
결과를 옮겨 적기만 한다.
"""

from __future__ import annotations

import re
import unicodedata

# 표 머리의 "(단위 : 백만원, %)" 에서 금액 단위를 읽는다.
# 큰 단위부터 본다 — "백만원" 이 "만원" 보다 먼저 걸려야 한다.
_UNIT_MULTIPLIER: tuple[tuple[str, int], ...] = (
    ("조원", 1_000_000_000_000),
    ("십억원", 1_000_000_000),
    ("억원", 100_000_000),
    ("백만원", 1_000_000),
    ("십만원", 100_000),
    ("만원", 10_000),
    ("천원", 1_000),
    ("원", 1),
)

_SCALES: tuple[tuple[int, str], ...] = (
    (1_000_000_000_000, "조"),
    (100_000_000, "억"),
    (10_000, "만"),
)


def normalize_unit_text(text: str | None) -> str:
    """`(단위 : 백만원, %)` -> `백만원`. 못 읽으면 빈 문자열."""
    if not text:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    s = re.sub(r"[\s ​　]", "", s)
    for unit, _mul in _UNIT_MULTIPLIER:
        if unit in s:
            return unit
    return ""


def unit_multiplier(unit: str | None) -> int | None:
    """단위 이름 -> 원 단위 배수. 모르면 None."""
    key = normalize_unit_text(unit)
    for name, mul in _UNIT_MULTIPLIER:
        if name == key:
            return mul
    return None


def to_won(value: float, unit: str | None) -> float | None:
    """표에 적힌 값을 원 단위로 바꾼다."""
    mul = unit_multiplier(unit)
    return None if mul is None else value * mul


def format_korean_amount(won: float, *, max_terms: int = 2) -> str:
    """원 단위 금액을 조/억/만 으로 읽어 준다.

        3_112_850_000_000  ->  "3조 1,128억"
        22_764_764_160_000 ->  "22조 7,647억"
                 55_000    ->  "5만 5,000원"

    `max_terms` 는 몇 자리까지 읽을지. 기본 2 — 사람이 읽는 방식에 가깝다.
    """
    sign = "-" if won < 0 else ""
    remain = int(abs(round(won)))
    if remain == 0:
        return "0원"
    parts: list[str] = []
    for scale, name in _SCALES:
        if remain >= scale and len(parts) < max_terms:
            parts.append(f"{remain // scale:,}{name}")
            remain %= scale
    if remain and len(parts) < max_terms:
        parts.append(f"{remain:,}")
    text = " ".join(parts)
    return f"{sign}{text}원" if not text.endswith("원") else f"{sign}{text}"


def describe_amount(value: float, unit: str | None) -> str:
    """표의 값과 단위를 사람이 읽는 금액으로. 단위를 모르면 빈 문자열.

        (3_112_850, "백만원")  ->  "3조 1,128억원(3,112,850백만원)"
    """
    won = to_won(value, unit)
    if won is None:
        return ""
    unit_name = normalize_unit_text(unit)
    return f"{format_korean_amount(won)}({value:,.0f}{unit_name})"


# --- 비율 -------------------------------------------------------------------
# 공시는 비율을 두 가지로 적는다. `0.0430` 과 `4.30%` 는 같은 값이다.
# 채점기가 이걸 모르면 **더 정확한 답이 오답**이 된다(G0146 실측).

def ratio_variants(value: float) -> list[float]:
    """비율로 볼 수 있는 값들. 0.043 -> [0.043, 4.3], 4.3 -> [4.3, 0.043]."""
    out = [value]
    if value != 0:
        for candidate in (value * 100, value / 100):
            if candidate not in out:
                out.append(candidate)
    return out


def same_ratio(a: float, b: float, *, rel_tol: float = 1e-6) -> bool:
    """두 값이 같은 비율인가 (100배 차이를 같다고 본다)."""
    for candidate in ratio_variants(a):
        if abs(candidate - b) <= max(abs(b), abs(candidate)) * rel_tol + 1e-12:
            return True
    return False


__all__ = [
    "normalize_unit_text", "unit_multiplier", "to_won",
    "format_korean_amount", "describe_amount",
    "ratio_variants", "same_ratio",
]
