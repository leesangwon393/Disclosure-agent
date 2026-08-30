"""Calculation Tool (§62): 계산은 LLM 에 맡기지 않고 deterministic Python 으로 한다.
계산 결과를 source of truth 로 두고, HCX 는 이 결과를 설명만 한다."""

from __future__ import annotations


def calculate_growth_rate(before: float, after: float) -> dict:
    if before == 0:
        return {"before": before, "after": after, "growth_rate_pct": None, "abs_change": after - before, "error": "before=0, 증가율 계산 불가"}
    return {
        "before": before, "after": after,
        "abs_change": after - before,
        "growth_rate_pct": round((after - before) / abs(before) * 100, 4),
    }


def calculate_ratio(numerator: float, denominator: float, *, label: str | None = None) -> dict:
    if denominator == 0:
        return {"numerator": numerator, "denominator": denominator, "ratio_pct": None, "label": label, "error": "denominator=0"}
    return {
        "numerator": numerator, "denominator": denominator, "label": label,
        "ratio": numerator / denominator,
        "ratio_pct": round(numerator / denominator * 100, 4),
    }


def calculate_cagr(begin_value: float, end_value: float, n_years: float) -> dict:
    if begin_value <= 0 or n_years <= 0:
        return {"begin_value": begin_value, "end_value": end_value, "n_years": n_years, "cagr_pct": None, "error": "begin_value<=0 또는 n_years<=0"}
    cagr = ((end_value / begin_value) ** (1 / n_years) - 1) * 100
    return {"begin_value": begin_value, "end_value": end_value, "n_years": n_years, "cagr_pct": round(cagr, 4)}


CALCULATION_TOOLS = {
    "calculate_growth_rate": calculate_growth_rate,
    "calculate_ratio": calculate_ratio,
    "calculate_cagr": calculate_cagr,
}
