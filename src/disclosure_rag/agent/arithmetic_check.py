"""답변 숫자의 유도 검산 (validator 보조).

## 왜 따로 만들었나

답변에 나온 숫자가 근거 원문에 **글자 그대로** 있는지만 보면, 계산해서 나온
값(증감률·비율·차액·합계·CAGR)은 전부 "근거 없는 숫자"로 오탐된다. 반대로
검산을 아무렇게나 붙이면 무작위 숫자까지 통과시켜 지표가 무의미해진다.

상원 레포(`Dart-Agent`)의 `validator.py` 가 후자에 해당한다. 허용오차를
`tol = max(abs(target) * 0.005, 0.5)` 로 **분모 값 기준**으로 잡아놓고, 그
값을 비율식 `v1 * 100 / target` 의 **결과 공간**에서 그대로 창(window)으로
썼다. target 이 10억이면 창이 500만인데 찾는 값은 100 언저리라, 근거에 작은
숫자가 하나라도 있으면 무엇이든 통과한다. 실제로 그 레포 결과 파일에
`'004413': '38841749 / 876043 * 100(%)'`(참값 4433.77) 처럼 **계산 결과가
아닌 코드값이 "검산 통과"로 기록**돼 있다.

## 이 구현이 그걸 피하는 방법

1. **허용오차를 표기 정밀도에서 얻는다.** `"12.4"` 는 반올림 전 참값이
   `[12.35, 12.45)` 이므로 `d = 0.05`. `"4413"` 이면 `d = 0.5`.
   상대오차 상수를 쓰지 않는다.
2. **그 오차를 식을 따라 전파해서 상대 쪽 공간의 창을 만든다.**
   비율 `c = a/b*100` 에서 `a = b*c/100`, 따라서 `da = b*d/100`.
   창을 다른 공간에 그대로 옮겨 쓰지 않는다 — 위 버그의 정확한 원인이다.
3. **후보 탐색은 정렬 + 이분탐색**이라 O(n log n). 근거 숫자가 수천 개여도
   된다(O(n^2) cap 을 두고 조용히 검산을 포기하는 일이 없다).
4. **문맥으로 식을 제한한다.** 답변에서 그 숫자 뒤에 `%`/`포인트` 가 붙어
   있을 때만 비율·증감률 식을 시도한다. 아무 숫자에나 5개 식을 다 던지면
   오탐이 급증한다.
5. **자명한 해를 배제한다.** 분모 0, `a == b`, 연도로 보이는 값(1900~2100),
   문서 ID 자릿수 값은 후보에서 뺀다.

6. **후보 풀을 답변이 실제로 제시한 숫자로 제한한다.** 이게 결정적이다.
   근거 숫자 2,000개를 전부 피연산자 후보로 놓으면 400만 쌍이 생겨서,
   허용오차를 아무리 정확히 잡아도 무작위 숫자의 **51.2%** 가 우연히
   설명된다(초기 구현을 이 파일의 적대적 테스트로 실측한 값이다).
   그래서 풀을 **답변에도 등장하고 근거에도 등장하는 숫자**로 좁힌다.
   계산을 제시하는 답변은 입력 수치를 같이 쓴다("2025년 1,200억 →
   2024년 1,000억 대비 20% 증가"). 입력을 안 밝힌 답변은 검산이 되는
   척하지 않고 "유도 근거 불명" 으로 남긴다 — 그게 정직한 결과다.

오탐률은 추정하지 않고 측정한다 — `tests/test_arithmetic_check.py` 의
적대적 테스트가 실제 근거 숫자 분포 위에서 무작위 숫자의 통과율을 재고,
임계치를 넘으면 실패한다.
"""

from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass

# 연도로 보이는 값 — 검산 후보(피연산자)에서 제외한다.
_YEAR_MIN, _YEAR_MAX = 1900, 2100
# 분모가 너무 작으면 창이 폭발한다.
_MIN_DENOMINATOR = 1e-6


@dataclass(frozen=True)
class Derivation:
    """답변 숫자 하나가 근거 숫자들로 설명된 결과."""

    claimed: float
    formula: str
    exact: float          # 식의 참값 (반올림 전)

    def __str__(self) -> str:  # 로그/경고용
        return f"{self.formula} = {self.exact:.6g} (답변 표기 {self.claimed:g})"


def display_tolerance(token: str) -> float:
    """표기된 문자열의 정밀도에서 반올림 허용오차를 얻는다.

    `"12.4"` -> 0.05, `"12"` -> 0.5, `"224787773988054"` -> 0.5.
    """
    token = token.replace(",", "").strip()
    if "." in token:
        decimals = len(token.split(".", 1)[1])
    else:
        decimals = 0
    return 0.5 * (10.0 ** -decimals)


def _is_year_like(value: float) -> bool:
    return float(value).is_integer() and _YEAR_MIN <= value <= _YEAR_MAX


class _SortedPool:
    """근거 숫자 풀. 값 -> 원본 표기 를 유지한 채 이분탐색으로 창 조회."""

    def __init__(self, values: dict[float, str]):
        self._values = sorted(values)
        self._labels = values

    def __len__(self) -> int:
        return len(self._values)

    def find(self, target: float, window: float) -> float | None:
        """`target` 에서 ±window 안에 있는 근거 숫자 중 가장 가까운 것."""
        if window < 0 or not self._values:
            return None
        lo = bisect.bisect_left(self._values, target - window)
        hi = bisect.bisect_right(self._values, target + window)
        if lo >= hi:
            return None
        best, best_d = None, math.inf
        for v in self._values[lo:hi]:
            d = abs(v - target)
            if d < best_d:
                best, best_d = v, d
        return best

    def label(self, value: float) -> str:
        return self._labels.get(value, f"{value:g}")


# 풀이 이보다 크면 우연한 설명이 급증한다. 검산을 포기하는 편이 정직하다.
_MAX_POOL = 200


def build_pool(numbers: dict[float, str]) -> _SortedPool:
    """숫자 dict 로 검산 후보 풀을 만든다. 연도는 피연산자에서 뺀다.

    직접 쓰기보다 `build_derivation_pool()` 을 쓴다.
    """
    return _SortedPool({v: s for v, s in numbers.items() if not _is_year_like(v)})


def build_derivation_pool(
    answer_numbers: dict[float, str], evidence_numbers: dict[float, str]
) -> _SortedPool:
    """검산에 쓸 피연산자 풀 = **답변 ∩ 근거**.

    근거 전체를 후보로 놓으면 무작위 숫자의 절반이 우연히 설명된다(모듈
    docstring 6번). 답변이 스스로 제시한 입력 수치만 후보로 삼는다.
    """
    common = {v: s for v, s in evidence_numbers.items() if v in answer_numbers}
    return build_pool(common)


def verify_derived(
    claimed: float,
    claimed_token: str,
    pool: _SortedPool,
    *,
    percent_context: bool,
) -> Derivation | None:
    """`claimed` 가 근거 숫자 둘로 유도되는지 찾는다. 못 찾으면 None.

    `percent_context` 가 참일 때만 비율·증감률 식을 시도한다(문맥 제한).
    """
    if claimed == 0 or not math.isfinite(claimed):
        return None
    if len(pool) > _MAX_POOL:
        # 후보가 너무 많으면 "설명됐다" 가 우연일 확률이 높다 — 검산 포기.
        return None
    d = display_tolerance(claimed_token)

    for b in list(pool._values):
        if abs(b) < _MIN_DENOMINATOR:
            continue

        if percent_context:
            # 1) 비율:  c = a / b * 100      ->  a = b * c / 100,  da = |b| * d / 100
            target = b * claimed / 100.0
            window = abs(b) * d / 100.0
            a = pool.find(target, window)
            if a is not None and a != b:
                return Derivation(claimed, f"{pool.label(a)} / {pool.label(b)} * 100", a / b * 100.0)

            # 2) 증감률: c = (a - b) / b * 100  ->  a = b * (1 + c/100)
            target = b * (1.0 + claimed / 100.0)
            a = pool.find(target, window)
            if a is not None and a != b:
                return Derivation(
                    claimed, f"({pool.label(a)} - {pool.label(b)}) / {pool.label(b)} * 100",
                    (a - b) / b * 100.0,
                )
        else:
            # 3) 차액:  c = a - b   ->  a = b + c,  da = d
            a = pool.find(b + claimed, d)
            if a is not None and a != b:
                return Derivation(claimed, f"{pool.label(a)} - {pool.label(b)}", a - b)

            # 4) 합계:  c = a + b   ->  a = c - b
            a = pool.find(claimed - b, d)
            if a is not None and a != b:
                return Derivation(claimed, f"{pool.label(a)} + {pool.label(b)}", a + b)

    return None


_SCALE_EXPONENTS = tuple(e for e in range(-12, 13) if e != 0)


def detect_unit_rescale(claimed: float, pool: _SortedPool) -> str | None:
    """근거 숫자를 10의 거듭제곱으로 다시 환산한 값인지 본다.

    `ANSWER_SYSTEM_PROMPT` 는 단위 재환산을 금지한다(원 -> 억원 등으로 바꿔
    쓰다 자릿수를 틀리는 실패가 반복 관측됐다). 통과시키면 안 되고, 왜
    근거에 없는지 설명해 주는 용도다.
    """
    if claimed == 0 or not math.isfinite(claimed):
        return None
    for e in _SCALE_EXPONENTS:
        target = claimed * (10.0 ** -e)
        window = abs(target) * 1e-6
        a = pool.find(target, window)
        if a is not None:
            return f"{pool.label(a)} x 10^{e}"
    return None


_PERCENT_AFTER = re.compile(r"^\s*(?:%|퍼센트|포인트|%p|퍼센트포인트)")


def has_percent_context(answer: str, token: str) -> bool:
    """답변에서 그 숫자 바로 뒤에 % 계열 단위가 붙는지."""
    for m in re.finditer(re.escape(token), answer):
        if _PERCENT_AFTER.match(answer[m.end():m.end() + 8]):
            return True
    return False
