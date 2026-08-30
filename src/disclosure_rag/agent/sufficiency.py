"""근거 충분성 게이트 — 에이전트가 한 번 검색하고 멈추는 걸 막는다.

## 왜 필요한가

문서(§57~61)는 "근거가 부족하면 Agent 가 추가 검색한다"고 하지만, 실측에서는
그런 일이 거의 일어나지 않는다.

- 상원 레포 100문항 배치(`results/generalization_check/100q_batch_v2/results.json`):
  tool 호출 **1회가 54건**, `iterations=2`(= tool 한 턴 후 종료)가 **74건**.
- 우리 스모크(2026-08-30): 2회 반복 / 도구 1개(`lookup_fact`) 로 종료.

즉 루프 코드는 있는데 **모델이 그 루프를 쓰지 않는다.** 프롬프트로 시키는 데도
한계가 있다 — `AGENT_SYSTEM_PROMPT` 는 300자를 넘기면 HCX 가 2턴째부터 400 을
반환해서(실측 3회 재현) 지침을 더 넣을 공간이 없다.

## 그래서 규칙으로 판정한다

모델이 tool 호출을 멈췄을 때, **질문이 요구하는 커버리지를 실제로 확보했는지**
결정론적으로 검사한다. 못 채웠으면 무엇이 빠졌는지 짧게 적어 한 번 더 돌린다.

    두 회사 비교인데 한 회사만 검색됨      -> 나머지 회사를 지목해 재검색
    두 기간 비교인데 한 기간만 검색됨      -> 나머지 기간을 지목해 재검색
    정정 질문인데 원본/정정본 한쪽만 있음  -> get_correction_history 를 지목

프롬프트로 "더 찾아봐" 라고 막연히 미는 게 아니라 **빠진 항목의 이름을 준다.**
LLM 판단에 기대지 않으므로 API 호출 없이 테스트할 수 있다.
"""

from __future__ import annotations

import re

from disclosure_rag.entity.entity_extractor import ExtractedEntities

# 근거 없이 무한히 재촉하지 않는다. 한 질문당 이 횟수만 추가 요청한다.
DEFAULT_MAX_NUDGES = 2

_YEAR_RE = re.compile(r"(20\d{2})")


def _covered(tool_calls) -> tuple[set[str], set[str], bool, bool]:
    """지금까지 회수한 근거가 덮은 회사/연도, 그리고 정정본·원본 존재 여부."""
    companies: set[str] = set()
    years: set[str] = set()
    has_correction = has_original = False
    for tc in tool_calls:
        result = getattr(tc, "result", None) or {}
        for item in result.get("results", []) or []:
            if item.get("company"):
                companies.add(item["company"])
            for src in (item.get("period"), item.get("filing_date")):
                m = _YEAR_RE.search(str(src or ""))
                if m:
                    years.add(m.group(1))
            if item.get("is_correction"):
                has_correction = True
            else:
                has_original = True
    return companies, years, has_correction, has_original


def _wanted_years(entities: ExtractedEntities) -> set[str]:
    return {m.group(1) for p in entities.period for m in _YEAR_RE.finditer(str(p))}


def missing_coverage(entities: ExtractedEntities, tool_calls) -> list[str]:
    """아직 못 채운 항목을 사람이 읽을 수 있는 문장으로 돌려준다. 없으면 빈 리스트."""
    covered_companies, covered_years, has_correction, has_original = _covered(tool_calls)
    missing: list[str] = []

    wanted_companies = [c for c in entities.companies if c]
    if len(wanted_companies) >= 2:
        absent = [c for c in wanted_companies if c not in covered_companies]
        # 전부 없으면 검색 자체가 실패한 것 — 그건 재촉이 아니라 다른 문제다.
        if absent and len(absent) < len(wanted_companies):
            missing.append(f"아직 근거가 없는 회사: {', '.join(absent)}")

    wanted_years = _wanted_years(entities)
    if len(wanted_years) >= 2:
        absent_years = sorted(y for y in wanted_years if y not in covered_years)
        if absent_years and len(absent_years) < len(wanted_years):
            missing.append(f"아직 근거가 없는 기간: {', '.join(absent_years)}")

    if entities.explicit_correction and (has_correction or has_original):
        if not has_correction:
            missing.append("정정본 근거가 없습니다 (get_correction_history 로 정정 이력을 확인하세요)")
        elif not has_original:
            missing.append("원본 공시 근거가 없습니다 (get_correction_history 로 최초 공시를 확인하세요)")

    return missing


def nudge_message(missing: list[str]) -> str:
    """추가 검색을 요청하는 짧은 user 메시지.

    system 프롬프트가 아니라 user 메시지다 — system 은 300자를 넘기면 HCX 가
    400 을 반환하므로 건드리지 않는다.
    """
    lines = "\n".join(f"- {m}" for m in missing)
    return (
        "[근거 점검] 아직 질문에 답하기에 근거가 모자랍니다.\n"
        f"{lines}\n"
        "위 항목을 tool 로 더 찾은 뒤 답하세요. 정말로 자료가 없으면 없다고 하세요."
    )


# ============================================================================
# Stage 10: QueryPlan 기반 충분성 판정 (2026-08-30 추가)
# ============================================================================
#
# 위쪽 `missing_coverage` 는 옛 에이전트 루프용이다 — entities 와 tool_calls 를
# 본다. 아래는 새 파이프라인용으로, **QueryPlan 의 expected_fields 와
# Stage 9 가 구조화한 결과**를 본다. 둘은 공존한다.
#
# 판정 기준이 세 갈래다. 근거 '건수'는 보지 않는다 — 10건 있어도 필요한
# 항목이 하나도 없으면 답을 못 만들고, 1건이어도 그게 답이면 충분하다.
#
#   ① 필요한 항목이 다 있나          expected_fields vs 실제로 찾은 항목
#   ② 하위 질의가 다 채워졌나        두 회사 비교인데 한쪽만 나온 경우
#   ③ 정정 짝이 다 맞춰졌나          최초본만 있고 최종본이 없는 경우
#
# ②는 Stage 4(분해)가, ③은 Stage 9(구조화)가 알려준다. 이 셋을 각각 재검색
# 지시문으로 바꾼다.

from dataclasses import dataclass, field as _field  # noqa: E402


@dataclass
class SufficiencyReport:
    ok: bool
    missing_fields: list[str] = _field(default_factory=list)
    empty_sub_queries: list[str] = _field(default_factory=list)
    incomplete_pairs: list[str] = _field(default_factory=list)
    nudges_used: int = 0

    @property
    def reasons(self) -> list[str]:
        out: list[str] = []
        if self.missing_fields:
            out.append(f"근거에서 확인되지 않은 항목: {', '.join(self.missing_fields)}")
        if self.empty_sub_queries:
            out.append(f"근거를 하나도 못 찾은 대상: {', '.join(self.empty_sub_queries)}")
        if self.incomplete_pairs:
            out.append(f"정정 전후 중 한쪽만 확인된 항목: {', '.join(self.incomplete_pairs)}")
        return out

    @property
    def should_retry(self) -> bool:
        return not self.ok and self.nudges_used < DEFAULT_MAX_NUDGES

    def retry_message(self) -> str:
        """무엇을 더 찾아야 하는지 **이름을 지목해서** 준다.

        "더 찾아봐" 같은 막연한 지시는 실측상 작동하지 않았다(도구 1회 호출
        후 종료가 100문항 중 54건).
        """
        lines = "\n".join(f"- {r}" for r in self.reasons)
        return ("[근거 점검] 아직 답하기에 근거가 모자랍니다.\n"
                f"{lines}\n"
                "위 항목을 더 찾은 뒤 답하세요. 정말로 자료가 없으면 없다고 하세요.")


def check_sufficiency(
    plan, processed, *, decompose_result=None, nudges_used: int = 0,
) -> SufficiencyReport:
    """Stage 9 결과와 계획을 대조한다. LLM 을 쓰지 않는다.

    `plan.expected_fields` 가 비어 있으면 **제약을 걸지 않는다**(fail open).
    Field Schema 가 모르는 공시유형이거나 질문이 유형을 특정하지 못한 경우인데,
    거기에 억지 기준을 걸면 답할 수 있는 질문이 영원히 '부족'이 된다.
    """
    missing = list(processed.missing(plan.expected_fields)) if plan.expected_fields else []
    empty = list(getattr(decompose_result, "empty_labels", []) or [])
    pairs = [p.field for p in getattr(processed, "incomplete_pairs", [])] \
        if plan.task == "correction_diff" else []

    return SufficiencyReport(
        ok=not (missing or empty or pairs),
        missing_fields=missing, empty_sub_queries=empty, incomplete_pairs=pairs,
        nudges_used=nudges_used,
    )
