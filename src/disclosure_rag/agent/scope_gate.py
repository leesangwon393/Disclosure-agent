"""Stage 3: DART 코퍼스로 원천적으로 답할 수 없는 질문만 조기 차단한다.

이 게이트는 검색 전에 실행되므로 보수적이어야 한다. 회사가 유니버스
70개사 밖이라는 이유로 거부하지 않는다. Entity Registry의 submitter,
counterparty, shareholder, subsidiary도 모두 코퍼스에 실재하는 주체다.

즉시 거부는 오직 실시간 주가·뉴스·모델의 미래 예측·투자 추천이다. 잘 모르는
주체나 모호한 질문은 `possibly_scope`로 남겨 검색/역질문 단계에 기회를 준다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

from disclosure_rag.agent.query_plan import QueryPlan, Scope

ScopeAction = Literal["proceed", "refuse"]


class RegistryLike(Protocol):
    def types_for(self, name: str) -> set[str]: ...


@dataclass(frozen=True)
class ScopeDecision:
    scope: Scope
    action: ScopeAction
    reason: str
    matched_rule: str | None = None
    entity_types: dict[str, list[str]] = field(default_factory=dict)
    needs_clarification: bool = False
    clarification_reason: str | None = None

    @property
    def should_refuse(self) -> bool:
        return self.action == "refuse"

    @property
    def message(self) -> str | None:
        if not self.should_refuse:
            return None
        return (
            "이 질문은 보유한 DART 공시 원문만으로 확인할 수 있는 범위를 벗어납니다. "
            f"범위 밖 항목: {self.reason}"
        )

    @property
    def clarification_message(self) -> str | None:
        if not self.needs_clarification:
            return None
        return f"질문의 대상을 더 특정해 주세요. {self.clarification_reason or ''}".strip()


# 넓은 단어 하나가 아니라 '외부 현재 정보/판단을 요구하는 표현'을 잡는다.
# 예: "공시에 주가 영향이 기재됐는가"는 DART 본문 조회이므로 차단하면 안 된다.
_HARD_OUT_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("market_price", re.compile(
        r"(?:현재|오늘|지금|실시간|금일).{0,8}(?:주가|시세|종가|시가총액)"
        r"|(?:주가|시세|종가|시가총액).{0,8}(?:얼마|조회|알려|보여)"
    ), "실시간 주가·시세"),
    ("news", re.compile(
        r"(?:최신|오늘|지금|최근).{0,8}(?:뉴스|기사|언론보도)"
        r"|(?:뉴스|기사).{0,8}(?:알려|찾아|요약)"
    ), "최신 뉴스·기사"),
    ("prediction", re.compile(
        r"(?:전망|예측|예상).{0,5}(?:해줘|해줘|해봐|해봐|좀|가능)"
        r"|(?:오를까|내릴까|상승할까|하락할까|목표주가|파산확률|부도확률)"
        r"|(?:다음|향후|미래).{0,10}(?:주가|실적|환율).{0,8}(?:예측|전망|예상)"
    ), "미래 예측"),
    ("recommendation", re.compile(
        r"(?:매수|매도|투자).{0,8}(?:추천|해도|할까|해줘|해줘|의견)"
        r"|(?:사야|팔아야|사도돼|팔아도돼)"
    ), "투자·매매 추천"),
)


def _hard_out_match(question: str) -> tuple[str, str] | None:
    text = question or ""
    for code, pattern, description in _HARD_OUT_RULES:
        if pattern.search(text):
            return code, description
    return None


def _clarification_reason(plan: QueryPlan) -> str | None:
    """거부가 아니라 역질문이 필요한 명백한 빈칸만 찾는다."""
    if not plan.companies:
        return "회사명이 필요합니다."
    if plan.answer_mode == "unknown" or plan.task == "unknown":
        return "알고 싶은 항목이나 작업을 특정해 주세요."
    if (plan.task == "summarize" and not plan.report_types and not plan.report_kinds
            and not plan.periods and not plan.expected_fields):
        return "요약할 공시 종류나 기간이 필요합니다."
    if plan.task == "compare" and len(plan.companies) < 2 and len(plan.periods) < 2:
        return "비교할 회사나 기간을 두 개 이상 지정해 주세요."
    return None


def evaluate_scope(plan: QueryPlan, question: str, registry: RegistryLike) -> ScopeDecision:
    """QueryPlan과 Entity Registry를 읽어 범위를 판정한다.

    이 함수는 plan을 바꾸지 않는다. 변경이 필요한 온라인 파이프라인은
    `apply_scope_gate` 를 쓴다.
    """
    hard_out = _hard_out_match(question)
    entity_types = {company: sorted(registry.types_for(company)) for company in plan.companies}
    if hard_out:
        code, description = hard_out
        return ScopeDecision(
            scope="hard_out_scope", action="refuse", reason=description,
            matched_rule=code, entity_types=entity_types,
        )

    if not plan.companies:
        clarification = _clarification_reason(plan)
        return ScopeDecision(
            scope="possibly_scope", action="proceed",
            reason="회사를 특정하지 못했으므로 검색 또는 역질문이 필요함",
            entity_types=entity_types,
            needs_clarification=bool(clarification), clarification_reason=clarification,
        )

    unknown = [company for company, types in entity_types.items() if not types]
    if unknown:
        clarification = _clarification_reason(plan)
        return ScopeDecision(
            scope="possibly_scope", action="proceed",
            reason=("레지스트리에 없는 주체가 있지만 검색 전에는 거부하지 않음: "
                    + ", ".join(unknown)),
            entity_types=entity_types,
            needs_clarification=bool(clarification), clarification_reason=clarification,
        )

    clarification = _clarification_reason(plan)
    return ScopeDecision(
        scope="in_scope", action="proceed",
        reason="Entity Registry에 코퍼스 등장 주체로 확인됨",
        entity_types=entity_types,
        needs_clarification=bool(clarification), clarification_reason=clarification,
    )


def apply_scope_gate(plan: QueryPlan, question: str, registry: RegistryLike) -> ScopeDecision:
    """판정 결과를 QueryPlan의 scope 필드에 기록하고 반환한다."""
    decision = evaluate_scope(plan, question, registry)
    plan.scope = decision.scope
    plan.scope_reason = decision.reason
    plan.source["scope"] = "rule"
    return decision


__all__ = ["ScopeDecision", "ScopeAction", "evaluate_scope", "apply_scope_gate"]
