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
        return ("질문의 대상을 특정하기 어렵습니다. "
                f"{self.clarification_reason or ''} "
                "회사명·공시 종류·기간 중 필요한 것을 알려주시면 찾아드리겠습니다.").strip()

    @property
    def out_of_corpus(self) -> tuple[str, ...]:
        """코퍼스에 없는 회사. 검색해도 나올 수 없다."""
        return tuple(name for name, types in (self.entity_types or {}).items()
                     if not types)

    @property
    def out_of_corpus_message(self) -> str | None:
        names = self.out_of_corpus
        if not names:
            return None
        return (f"{', '.join(names)} 은(는) 제공된 DART 공시 코퍼스에 없는 회사입니다. "
                "해당 회사의 공시 정보는 확인할 수 없습니다.")


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


# 질문 첫머리의 "<이름>의 …" / "<이름>가 …" 를 잡는다.
#
# 왜 필요한가 (2026-08-31, gold_abstention 160문항)
# ------------------------------------------------
# "쿠팡의 최근 사업보고서 매출액은?" 처럼 **코퍼스 밖 회사**를 물으면
# EntityExtractor 가 아무것도 못 뽑아 `plan.companies` 가 빈다. 그러면
# "회사명이 필요합니다" 라는 역질문이 나간다 — 회사명은 이미 말했는데.
# 옳은 답은 "그 회사는 우리 코퍼스에 없습니다" 다.
#
# 실측: 이 패턴이 wrong_entity 40건을 40건 다 잡고, ambiguous 40건 중
# 오탐은 `그것의`(대명사) 하나뿐이었다. 그래서 대명사만 제외한다.
# 조사를 `은/는/도/에서` 까지 넓혀 보았지만 되돌렸다(2026-09-01).
# "KB금융지주는 얼마인가" 를 잡으려던 건데, 같은 모양의 "매출액은 얼마야?"
# 에서 **항목명이 회사명으로** 잡힌다. 여기에는 항목 사전이 없어 둘을 가를
# 방법이 없다. `의/가` 는 이름 뒤에 거의 항상 붙으므로 그대로 둔다.
_SUBJECT_PAT = re.compile(r"^\s*(?P<name>[가-힣A-Za-z0-9&·\-\.]{2,20})(?:의|가)\s")

# 사람·사물을 가리키는 말. 회사 이름이 아니다.
_PRONOUNS = frozenset({
    "그것", "이것", "저것", "그거", "이거", "그곳", "이곳", "여기", "거기",
    "그때", "이때", "그분", "이분", "우리", "저희", "당사", "해당",
})


def unknown_subject(question: str, registry: RegistryLike) -> str | None:
    """질문이 이름을 댔는데 코퍼스에 없는 회사면 그 이름을, 아니면 None."""
    m = _SUBJECT_PAT.match(question or "")
    if not m:
        return None
    name = m.group("name")
    if name in _PRONOUNS:
        return None
    try:
        if registry.types_for(name):
            return None
    except Exception:  # noqa: BLE001
        return None
    return name


# 되묻기 게이트 (2026-09-01)
#
# "KB금융지주는 얼마인가" — 회사 이름만 있고 **무엇을** 묻는지가 없다. 지금은
# 이런 질문에도 넘겨짚어 아무 숫자나 답한다. 실서비스에서는 틀린 답보다
# 되묻는 편이 낫다.
#
# 판정: 질문에서 회사 이름을 지우고 남은 말이 조사와 의문 표현뿐이면
# 물어본 항목이 없는 것이다. 항목 사전(`expected_fields`)에 걸린 게 하나라도
# 있으면 이 규칙은 발동하지 않는다.

_ASK_WORDS = frozenset((
    "는", "은", "이", "가", "의", "를", "을", "도", "에", "에서", "와", "과",
    "얼마", "얼마인가", "얼마인가요", "얼마야", "얼마인지", "얼마나", "얼마였나",
    "얼마입니까", "얼마죠", "얼마예요", "얼마인가여", "얼마나요", "얼마되나요",
    "뭐야", "뭔가요", "무엇인가", "무엇인가요", "무엇", "뭐", "뭐죠", "뭐예요",
    "어떻게", "어때", "어때요", "어떤가", "어떤가요", "어떠한가", "어떠한가요",
    "되나요", "되죠", "될까요", "얼마나되나요",
    "알려줘", "알려주세요", "알려줄래", "알려주라", "말해줘", "말해주세요",
    "궁금해", "궁금합니다", "보여줘", "보여주세요", "찾아줘", "찾아주세요",
    "인가", "인가요", "인지", "입니까", "이야", "야", "해줘", "해주세요",
    "주세요", "요", "몇이야", "몇인가", "몇인가요", "몇", "좀",
))
_WORD_ONLY = re.compile(r"[^가-힣A-Za-z]+")


def _asked_item_is_missing(plan: QueryPlan, question: str) -> bool:
    """질문이 **무엇을** 묻는지가 비어 있는가."""
    if plan.expected_fields or not plan.companies:
        return False
    if plan.task not in ("lookup", "calculate", "compare", "count", "unknown"):
        return False
    rest = question or ""
    # 질문에 실제로 쓰인 표기까지 지운다. 정식명만 지우면 "삼성SDI" 가 남는다.
    names: list[str] = list(plan.companies)
    for canonical, mentions in (plan.company_mentions or {}).items():
        names.append(canonical)
        names.extend(mentions or [])
    for name in sorted(set(names), key=len, reverse=True):
        if name:
            rest = rest.replace(name, " ")
    words = [w for w in _WORD_ONLY.sub(" ", rest).split() if w]
    return all(w in _ASK_WORDS for w in words)


def _clarification_reason(plan: QueryPlan, question: str = "", *,
                          can_fill_blanks: bool = False) -> str | None:
    """거부가 아니라 역질문이 필요한 명백한 빈칸만 찾는다.

    `can_fill_blanks` 는 뒤에 계획 보완(2b, HCX)이 남아 있다는 뜻이다.
    그때는 answer_mode/task 빈칸으로 되묻지 않는다 — 그건 사용자에게 물을
    것이 아니라 **우리가 채울 칸**이다.

    실측(2026-08-31): "삼성전자의 2024년 매출액은?" 은 규칙만으로는
    mode/task 가 unknown 이라 역질문으로 끝났다. 사람이 실제로 이렇게 묻는데
    되묻는 건 데모에서 치명적이다. ("...얼마인가?" 로 물으면 정상 동작했다 —
    문장 끝 표현 하나로 갈렸다.)
    """
    if not plan.companies:
        return "회사명이 필요합니다."
    if _asked_item_is_missing(plan, question):
        return "회사 이름만 있고 무엇을 묻는지가 없습니다. 알고 싶은 항목을 적어 주세요."
    if not can_fill_blanks and (plan.answer_mode == "unknown" or plan.task == "unknown"):
        return "알고 싶은 항목이나 작업을 특정해 주세요."
    if (plan.task == "summarize" and not plan.report_types and not plan.report_kinds
            and not plan.periods and not plan.expected_fields):
        return "요약할 공시 종류나 기간이 필요합니다."
    if plan.task == "compare" and len(plan.companies) < 2 and len(plan.periods) < 2:
        return "비교할 회사나 기간을 두 개 이상 지정해 주세요."
    return None


def evaluate_scope(plan: QueryPlan, question: str, registry: RegistryLike,
                   *, can_fill_blanks: bool = False) -> ScopeDecision:
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
        # 이름은 댔는데 코퍼스에 없는 회사인 경우와, 아예 이름을 안 댄 경우는
        # 답이 다르다. 앞은 "그 회사는 없습니다", 뒤는 역질문이다.
        unknown_name = unknown_subject(question, registry)
        if unknown_name:
            return ScopeDecision(
                scope="possibly_scope", action="proceed",
                reason=f"코퍼스에 없는 주체: {unknown_name}",
                entity_types={unknown_name: []},
            )
        clarification = _clarification_reason(plan, question, can_fill_blanks=can_fill_blanks)
        return ScopeDecision(
            scope="possibly_scope", action="proceed",
            reason="회사를 특정하지 못했으므로 검색 또는 역질문이 필요함",
            entity_types=entity_types,
            needs_clarification=bool(clarification), clarification_reason=clarification,
        )

    unknown = [company for company, types in entity_types.items() if not types]
    if unknown:
        clarification = _clarification_reason(plan, question, can_fill_blanks=can_fill_blanks)
        return ScopeDecision(
            scope="possibly_scope", action="proceed",
            reason=("레지스트리에 없는 주체가 있지만 검색 전에는 거부하지 않음: "
                    + ", ".join(unknown)),
            entity_types=entity_types,
            needs_clarification=bool(clarification), clarification_reason=clarification,
        )

    clarification = _clarification_reason(plan, question, can_fill_blanks=can_fill_blanks)
    return ScopeDecision(
        scope="in_scope", action="proceed",
        reason="Entity Registry에 코퍼스 등장 주체로 확인됨",
        entity_types=entity_types,
        needs_clarification=bool(clarification), clarification_reason=clarification,
    )


def apply_scope_gate(plan: QueryPlan, question: str, registry: RegistryLike,
                     *, can_fill_blanks: bool = False) -> ScopeDecision:
    """판정 결과를 QueryPlan의 scope 필드에 기록하고 반환한다."""
    decision = evaluate_scope(plan, question, registry, can_fill_blanks=can_fill_blanks)
    plan.scope = decision.scope
    plan.scope_reason = decision.reason
    plan.source["scope"] = "rule"
    return decision


__all__ = ["ScopeDecision", "ScopeAction", "evaluate_scope", "apply_scope_gate"]
