"""유도 검산 테스트 — 미탐과 **오탐(아무 숫자나 통과)** 을 둘 다 본다.

상원 레포(`Dart-Agent`)의 같은 기능은 오탐 테스트가 없었고, 유일한 회귀
테스트가 근거 숫자 1개짜리 인위적 조건에서만 통과하고 있었다. 그래서
허용오차 차원 오류로 무작위 숫자의 대부분을 통과시키면서도 초록불이었다.
여기서는 **실제 근거에 가까운 숫자 분포 위에서 통과율을 측정**한다.
"""
from __future__ import annotations

import random

from disclosure_rag.agent.arithmetic_check import (
    build_derivation_pool,
    build_pool,
    detect_unit_rescale,
    display_tolerance,
    has_percent_context,
    verify_derived,
)


def _fmt(v: float) -> str:
    return f"{v:.0f}" if float(v).is_integer() else f"{v}"


def _pool(*values: float):
    return build_pool({float(v): _fmt(v) for v in values})


def test_display_tolerance_from_decimals():
    assert display_tolerance("12.4") == 0.05
    assert display_tolerance("12.45") == 0.005
    assert display_tolerance("12") == 0.5
    assert display_tolerance("1,234") == 0.5


def test_ratio_is_verified():
    got = verify_derived(4433.8, "4433.8", _pool(38841749, 876043), percent_context=True)
    assert got is not None and "38841749" in got.formula


def test_growth_rate_is_verified():
    got = verify_derived(20.0, "20.0", _pool(1000, 1200), percent_context=True)
    assert got is not None


def test_difference_is_verified():
    got = verify_derived(200.0, "200", _pool(1000, 1200), percent_context=False)
    assert got is not None


def test_sum_is_verified():
    got = verify_derived(687.0, "687", _pool(381, 306), percent_context=False)
    assert got is not None


def test_rounded_display_is_accepted():
    got = verify_derived(23.4, "23.4", _pool(1000, 1234), percent_context=True)
    assert got is not None


def test_sangwon_false_pass_case_is_rejected():
    """`'004413': '38841749 / 876043 * 100(%)'` — 참값 4433.77. 통과하면 안 된다."""
    assert verify_derived(4413.0, "4413", _pool(38841749, 876043), percent_context=True) is None


def test_sangwon_sum_false_pass_case_is_rejected():
    """`'684': '381 + 306'` — 참값 687."""
    assert verify_derived(684.0, "684", _pool(381, 306), percent_context=False) is None


def test_percent_formulas_need_percent_context():
    assert verify_derived(20.0, "20.0", _pool(1000, 1200), percent_context=False) is None


def test_has_percent_context_reads_the_answer():
    assert has_percent_context("영업이익률은 12.4%입니다", "12.4")
    assert not has_percent_context("계약금액은 12.4조원입니다", "12.4")


def test_years_are_not_operands():
    assert verify_derived(1.0, "1", _pool(2024, 2025), percent_context=False) is None


def test_unit_rescale_is_detected_not_passed():
    pool = _pool(7661584000000)
    assert verify_derived(7661584.0, "7661584", pool, percent_context=False) is None
    assert detect_unit_rescale(7661584.0, pool) is not None


def _realistic_evidence(rng: random.Random, n: int) -> dict[float, str]:
    """실제 공시 근거에 가까운 분포: 금액(큰 수), 비율(작은 수), 건수, 연도."""
    out: dict[float, str] = {}
    while len(out) < n:
        r = rng.random()
        if r < 0.55:
            v = round(rng.uniform(1, 1000) * 10 ** rng.randint(3, 13))
        elif r < 0.75:
            v = round(rng.uniform(0.1, 99.9), 1)
        elif r < 0.9:
            v = rng.randint(100, 999_999)
        else:
            v = rng.randint(2020, 2026)
        out[float(v)] = f"{v:g}"
    return out


def test_random_numbers_are_rarely_verified():
    """프로덕션과 같은 조건에서 무작위 숫자의 통과율을 측정한다.

    근거 2,000개 중 답변이 실제로 인용한 8개만 피연산자 후보가 된다
    (`build_derivation_pool` = 답변 ∩ 근거). 상원 구현은 근거 전체를
    후보로 놓아 96~100% 가 통과했고, 이 파일의 초기 구현도 같은 방식일 때
    51.2% 가 통과했다. 3% 를 상한으로 둔다.
    """
    rng = random.Random(20260830)
    evidence = _realistic_evidence(rng, 2000)

    trials, passed = 400, 0
    for _ in range(trials):
        quoted = dict(rng.sample(sorted(evidence.items()), 8))
        if rng.random() < 0.5:
            token, percent = f"{rng.uniform(0.1, 99.9):.1f}", True
        else:
            token, percent = f"{round(rng.uniform(1, 1000) * 10 ** rng.randint(3, 13)):d}", False
        pool = build_derivation_pool({**quoted, float(token): token}, evidence)
        if verify_derived(float(token), token, pool, percent_context=percent) is not None:
            passed += 1

    rate = passed / trials
    assert rate <= 0.03, f"무작위 숫자 통과율 {rate:.1%} — 검산기가 오탐을 걸러내지 못한다"


def test_whole_evidence_pool_is_refused():
    """근거 전체(수천 개)를 그대로 후보로 넘기면 검산을 포기해야 한다."""
    rng = random.Random(1)
    pool = build_pool(_realistic_evidence(rng, 2000))
    assert verify_derived(23.4, "23.4", pool, percent_context=True) is None


def test_true_derivations_still_pass_in_a_large_pool():
    """오탐을 막느라 참인 유도까지 놓치면 안 된다(미탐 회귀)."""
    rng = random.Random(7)
    evidence = _realistic_evidence(rng, 2000)
    evidence[1_000_000.0] = "1000000"
    evidence[1_234_000.0] = "1234000"
    # 답변이 입력 수치를 같이 제시한 상황 — 이때는 반드시 검산돼야 한다.
    answer_numbers = {1_000_000.0: "1000000", 1_234_000.0: "1234000", 23.4: "23.4"}
    pool = build_derivation_pool(answer_numbers, evidence)
    assert verify_derived(23.4, "23.4", pool, percent_context=True) is not None
