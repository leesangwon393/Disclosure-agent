"""Stage 4: 질문 분해 (Query Decomposition).

## 왜 필요한가

한 번 검색해서는 답이 안 나오는 질문이 있다. suite_v1 38문항 중 **23문항**이다.

    S007~S014 (8문항)  "삼성전자와 한미반도체 중 최대 계약금액은?"
                       -> 회사별로 따로 찾아야 한다. 한 번 검색하면 한쪽만 나온다.

    S015~S022 (8문항)  "체결한 계약 중 이후 해지된 것이 있는가?"
                       -> 체결 공시와 해지 공시를 **둘 다** 찾아야 한다.

    S023~S026 (4문항)  "최초 공시와 최종 정정본 사이에 무엇이 달라졌는가?"
                       -> 최초본과 최종본을 따로 확보해야 한다.

## 분해 조건은 answer_mode 가 아니다

S007~S014 는 `answer_mode` 가 **closed** 다(답은 숫자 하나). open 일 때만
쪼개면 이 8문항, 전체의 21% 가 반쪽 답변이 된다. 그래서 작업 복잡도
(`plan.needs_decomposition`)로 판정한다.

## 검색기를 주입받는다

이 모듈은 "무엇을 몇 번 검색할지"만 정한다. 검색이 어떻게 이뤄지는지
(하이브리드 구성, 이중 채널, 융합 방식)는 알지 않는다. 그래서 Stage 5~6 이
바뀌어도 이 파일은 안 바뀐다.

## 병합에 대한 주의

여기서 하는 병합은 **하위 질의들의 결과를 합치는 것**이다. Stage 5c 의
'정형 채널 + 비정형 채널 병합'과는 다른 층위다. 헷갈리지 말 것.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from disclosure_rag.agent.query_plan import QueryPlan

logger = logging.getLogger(__name__)

# search_fn(query_text, plan_for_this_subquery, k) -> [(chunk, score), ...]
SearchFn = Callable[[str, QueryPlan, int], Sequence[tuple[Any, float]]]


@dataclass
class SubQuery:
    """하위 질의 하나. `plan` 은 원본 계획을 좁힌 사본이다."""
    text: str
    plan: QueryPlan
    kind: str                    # company | report_kind | period | base
    label: str = ""
    top_k: int = 8

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.kind}:{self.text[:24]}"


@dataclass
class DecomposeResult:
    sub_queries: list[SubQuery] = field(default_factory=list)
    merged: list[tuple[Any, float]] = field(default_factory=list)
    per_query: dict[str, int] = field(default_factory=dict)   # label -> 회수 건수
    empty_labels: list[str] = field(default_factory=list)     # 0건인 하위 질의

    @property
    def decomposed(self) -> bool:
        return len(self.sub_queries) > 1

    @property
    def partial(self) -> bool:
        """하위 질의 중 하나라도 빈손이면 부분 실패다.

        Stage 11 이 이걸 봐야 한다 — "두 회사 비교인데 한 회사 근거만 있음"이
        정확히 이 경우다. 근거 건수만 보면 통과해버린다.
        """
        return bool(self.empty_labels)


def _narrow(plan: QueryPlan, **overrides) -> QueryPlan:
    """원본 계획을 복사해 일부만 좁힌다. 원본은 건드리지 않는다.

    `dataclasses.replace` 는 **얕은 복사**라 리스트·dict 를 원본과 공유한다.
    지금은 뒤 단계가 읽기만 해서 사고가 없지만, 어디선가
    `plan.expected_fields.append(...)` 하나만 생기면 하위 질의 전체와 원본이
    동시에 오염된다. 주석이 단언한 "원본은 건드리지 않는다"를 실제로 지킨다.
    """
    from dataclasses import replace
    clone = replace(plan)
    for name in ("companies", "periods", "report_types", "report_kinds",
                 "expected_fields", "operations", "notes"):
        value = getattr(clone, name, None)
        if isinstance(value, list):
            setattr(clone, name, list(value))
    if isinstance(getattr(clone, "source", None), dict):
        clone.source = dict(clone.source)
    for key, value in overrides.items():
        setattr(clone, key, value)
    # 하위 질의는 이미 쪼개진 단위라 더 쪼개지 않는다.
    clone.needs_multiple_documents = False
    clone.operations = []
    return clone


# 한 질문을 몇 조각까지 쪼갤 것인가.
#
# 실측(suite_v1 S025·S026): "두산로보틱스의 **주요사항보고서** 공시가 정정된
# 내역이 있는가?" 는 괄호 안 유형을 말하지 않아 주요사항보고서 19종 전부에
# 걸린다. 그대로 쪼개면 한 질문에 검색을 19번 하고, 각 하위 질의는 8칸씩만
# 받아 정작 정답이 밀린다.
#
# 축이 이만큼 벌어졌다는 건 **질문이 대상을 특정하지 못했다**는 뜻이다.
# 그럴 때는 쪼개지 말고 한 번에 넓게 찾는 편이 낫다. Field Schema 의
# `expected_fields` 도 같은 이유로 유형 3종 이상이면 빈 목록을 돌려준다.
MAX_SUB_QUERIES = 3


def build_sub_queries(plan: QueryPlan, question: str) -> list[SubQuery]:
    """계획을 보고 무엇을 몇 번 검색할지 정한다.

    쪼개는 축은 하나만 고른다. 회사 2곳 × 공시유형 2종을 곱하면 4번 검색이
    되는데, 실측상 그런 질문이 없고 검색 횟수만 늘어난다. 우선순위는
    **회사 > 공시유형 > 기간** 이다 — 회사가 다르면 문서가 확실히 다르지만,
    공시유형은 한 문서에 섞여 있을 수 있다.

    축 후보가 `MAX_SUB_QUERIES` 를 넘으면 쪼개지 않는다(위 상수 참조).
    """
    if not plan.needs_decomposition:
        return [SubQuery(text=question, plan=plan, kind="base",
                         label="base", top_k=plan.top_k())]

    per_target = max(8, plan.top_k() // max(1, _n_targets(plan)))

    def _too_many(values) -> bool:
        return len(values) > MAX_SUB_QUERIES

    if len(plan.companies) >= 2 and not _too_many(plan.companies):
        return [
            SubQuery(text=_focus(question, company), kind="company", label=f"company:{company}",
                     plan=_narrow(plan, companies=[company]), top_k=per_target)
            for company in plan.companies
        ]

    if len(plan.report_kinds) >= 2 and not _too_many(plan.report_kinds):
        return [
            SubQuery(text=_focus(question, kind), kind="report_kind", label=f"kind:{kind}",
                     plan=_narrow(plan, report_kinds=[kind]), top_k=per_target)
            for kind in plan.report_kinds
        ]

    if len(plan.periods) >= 2 and not _too_many(plan.periods):
        return [
            SubQuery(text=_focus(question, period), kind="period", label=f"period:{period}",
                     plan=_narrow(plan, periods=[period]), top_k=per_target)
            for period in plan.periods
        ]

    # 여기 오는 경우는 둘이다.
    #   ① correction_diff 처럼 축은 하나인데 여러 문서가 필요한 경우
    #      -> 버전 정리(Stage 7)가 최초·최종을 남기므로 검색량만 키운다
    #   ② 축이 MAX_SUB_QUERIES 를 넘어 쪼개기를 포기한 경우
    #      -> 대상을 특정하지 못했으므로 한 번에 넓게 찾는다
    widened = max(plan.top_k(), 24) if _n_targets(plan) > MAX_SUB_QUERIES else plan.top_k()
    return [SubQuery(text=question, plan=plan, kind="base", label="base",
                     top_k=widened)]


def _n_targets(plan: QueryPlan) -> int:
    return max(len(plan.companies), len(plan.report_kinds), len(plan.periods), 1)


def _focus(question: str, term: str) -> str:
    """하위 질의 문장. 원 질문을 유지한 채 초점을 앞에 붙인다.

    원 질문을 통째로 버리고 `"삼성전자 계약금액"` 같은 짧은 문장을 만들면
    무엇을 묻는지가 사라진다("최대"인지 "최초"인지 등). 그래서 덧붙인다.
    """
    return f"{term} 기준으로: {question}"


def merge_results(
    per_query: Iterable[tuple[SubQuery, Sequence[tuple[Any, float]]]],
) -> tuple[list[tuple[Any, float]], dict[str, int], list[str]]:
    """하위 질의 결과를 합친다.

    **점수로 다시 줄 세우지 않는다.** 하위 질의끼리 점수 분포가 달라 비교
    의미가 없고, 무엇보다 한쪽 회사의 근거가 통째로 밀려나면 비교 질문이
    답을 못 한다. 대신 하위 질의를 **번갈아 가며**(round-robin) 뽑아
    모든 하위 질의가 상위권에 자리를 갖게 한다.
    """
    buckets: list[tuple[SubQuery, list[tuple[Any, float]]]] = [
        (sq, list(hits)) for sq, hits in per_query
    ]
    counts = {sq.label: len(hits) for sq, hits in buckets}
    empty = [sq.label for sq, hits in buckets if not hits]

    merged: list[tuple[Any, float]] = []
    seen: set[str] = set()
    for i in range(max((len(h) for _s, h in buckets), default=0)):
        for _sq, hits in buckets:
            if i >= len(hits):
                continue
            chunk, score = hits[i]
            cid = str(getattr(chunk, "chunk_id", None) or id(chunk))
            if cid in seen:
                continue
            seen.add(cid)
            merged.append((chunk, score))
    return merged, counts, empty


def decompose_and_search(
    plan: QueryPlan, question: str, search_fn: SearchFn,
) -> DecomposeResult:
    """분해 -> 하위 질의별 검색 -> 병합. 이 모듈의 진입점이다."""
    subs = build_sub_queries(plan, question)
    results = []
    for sq in subs:
        try:
            hits = list(search_fn(sq.text, sq.plan, sq.top_k))
        except Exception as exc:  # noqa: BLE001
            # 하위 질의 하나가 실패해도 나머지는 살린다. 빈손으로 기록되어
            # Stage 11 이 '부분 근거'로 판단할 수 있다.
            # **로그는 반드시 남긴다** — 무음으로 삼키면 인덱스 장애나 필터
            # 버그가 "근거 0건 -> 확인되지 않습니다" 로 나가고 원인이 어디에도
            # 안 남는다(2026-08-31 점검에서 발견).
            logger.warning("[DECOMPOSE] 하위 질의 검색 실패(%s) label=%s: %s",
                           type(exc).__name__, sq.label, exc)
            hits = []
        results.append((sq, hits))

    merged, counts, empty = merge_results(results)
    return DecomposeResult(sub_queries=subs, merged=merged,
                           per_query=counts, empty_labels=empty)


__all__ = ["MAX_SUB_QUERIES", "SubQuery", "DecomposeResult", "build_sub_queries",
           "merge_results", "decompose_and_search"]
