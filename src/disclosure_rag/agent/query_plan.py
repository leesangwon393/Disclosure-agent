"""QueryPlan — 파이프라인 전체를 제어하는 계획서.

## 이 파일의 상태 (2026-08-30)

**데이터 계약(dataclass·타입·기본값)만 먼저 확정한다.** rule_builder /
plan_validator 본체는 이어서 채운다. 계약을 먼저 박는 이유는 Stage 3(범위
게이트)와 Stage 11(거부 게이트)이 이 구조만 알면 병렬로 만들 수 있기 때문이다.
그쪽 코드는 `QueryPlan` 을 **읽기만** 하고 만들지 않는다.

## 필드를 누가 채우나

    규칙 (LLM 안 씀)   scope, companies, periods, report_types, report_kinds,
                       latest_policy, expected_fields
    규칙 -> HCX 폴백    answer_mode, task
                       어미 목록 20개로 판정한다. suite_v1 38문항 실측에서
                       34/38 일치했고, 틀린 4건은 라벨이 open 인데 규칙이
                       mixed 라 한 것으로 문장을 보면 규칙이 맞다. mixed 를
                       open 의 상위집합으로 처리하면 오분류 0건.

`source` 에 필드별 출처를 남긴다 — 나중에 "이 계획을 왜 이렇게 세웠나"를
역추적할 수 있어야 하고, HCX 가 채운 값만 골라 검증할 수 있어야 한다.

## scope 가 3단계인 이유

회사 이름이 유니버스 70개사에 없다는 이유로 거부하면 안 된다. 실측:

    제출인 기준 155개 중 86개가 유니버스 밖 (문서 1,029건)
    삼성전자가 counterparty 로 등장한 근거 5건 -> 전부 삼성중공업 제출 문서
    LG전자는 universe 가 아니지만 LG이노텍 대량보유보고서 5건의 제출인이다

즉 유니버스 밖 회사도 계약상대·주주·자회사·제출인으로 코퍼스에 실재한다.
따라서 조기 차단은 `hard_out_scope`(주가·뉴스·예측 등 코퍼스에 있을 수 없는
것)만 하고, 나머지 거부 판단은 **검색을 다 해본 뒤** Stage 11 에서 내린다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------- 타입

Scope = Literal["in_scope", "possibly_scope", "hard_out_scope"]
SCOPES: tuple[Scope, ...] = ("in_scope", "possibly_scope", "hard_out_scope")

AnswerMode = Literal["closed", "open", "mixed", "unknown"]
ANSWER_MODES: tuple[AnswerMode, ...] = ("closed", "open", "mixed", "unknown")

Task = Literal[
    "lookup",           # 단일 사실 조회
    "compare",          # 둘 이상 비교
    "calculate",        # 연산 (증감률·비율·CAGR)
    "timeline",         # 시간순 정리
    "summarize",        # 요약
    "correction_diff",  # 정정 전후 비교
    "count",            # 건수
    "unknown",
]
TASKS: tuple[Task, ...] = (
    "lookup", "compare", "calculate", "timeline",
    "summarize", "correction_diff", "count", "unknown",
)

# 질문이 값 자체가 아니라 **값들 중 하나**를 묻는 경우.
#   "최대 계약금액은 얼마인가"  -> 계약금액 여러 건 중 최댓값
# 실측 실패(v2 38문항, S007~S014): Facts 가 최신순 10건만 주는 바람에
# 삼성바이오로직스 54건 중 최댓값 2,074,694,843,000 대신 상위 10건의 최댓값
# 1,110,278,292,000 이 답으로 나갔다. 모델은 받은 것 중 정확히 최대를 골랐다.
Aggregation = Literal["max", "min", "count", "none"]
AGGREGATIONS: tuple[Aggregation, ...] = ("max", "min", "count", "none")

LatestPolicy = Literal["latest_only", "first_and_final", "all_versions"]
LATEST_POLICIES: tuple[LatestPolicy, ...] = ("latest_only", "first_and_final", "all_versions")

# 필드를 누가 채웠는가. plan_validator 가 HCX 가 채운 값만 골라 검증할 때 쓴다.
FieldSource = Literal["rule", "hcx", "default"]


# ---------------------------------------------------------------------------- 계획서

@dataclass
class QueryPlan:
    """질문 하나에 대한 실행 계획. 이 객체는 Stage 2 에서 한 번 만들어지고
    이후 단계는 읽기만 한다."""

    # --- 범위 ---------------------------------------------------------------
    scope: Scope = "possibly_scope"
    scope_reason: str = ""

    # --- 답의 모양 ----------------------------------------------------------
    answer_mode: AnswerMode = "unknown"
    task: Task = "unknown"

    # --- 검색 대상 ----------------------------------------------------------
    companies: list[str] = field(default_factory=list)
    periods: list[str] = field(default_factory=list)
    report_types: list[str] = field(default_factory=list)   # doc_group
    report_kinds: list[str] = field(default_factory=list)   # field_schema 의 유형명

    # --- 집계 --------------------------------------------------------------
    aggregation: Aggregation = "none"

    # --- 버전 처리 ----------------------------------------------------------
    latest_policy: LatestPolicy = "latest_only"

    # --- 충족 조건 ----------------------------------------------------------
    expected_fields: list[str] = field(default_factory=list)

    # --- 작업 복잡도 --------------------------------------------------------
    needs_multiple_documents: bool = False
    operations: list[str] = field(default_factory=list)

    # --- 추적 ---------------------------------------------------------------
    source: dict[str, FieldSource] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ 판정

    @property
    def is_complete(self) -> bool:
        """규칙만으로 계획이 다 세워졌는가 — False 일 때만 HCX 를 부른다.

        소수점 신뢰도(0.85 같은 것)를 쓰지 않는다. 규칙 기반 빌더가 뱉는
        소수점은 근거가 없는 숫자다. '빈칸이 있나 없나'만 본다.
        """
        return (
            self.answer_mode != "unknown"
            and self.task != "unknown"
            and bool(self.companies)
        )

    @property
    def needs_decomposition(self) -> bool:
        """Stage 4 질문 분해 조건.

        answer_mode 로 정하면 안 된다 — suite_v1 의 S007~S014 여덟 문항은
        answer_mode 가 closed 인데 두 회사를 각각 조회해 비교해야 한다.
        전체의 21% 다. 작업 복잡도로 정한다.
        """
        return self.needs_multiple_documents or len(self.operations) >= 2

    @property
    def is_open_ended(self) -> bool:
        """mixed 는 open 의 상위집합으로 다룬다 — open 만큼 검색하고 closed
        만큼 검증한다."""
        return self.answer_mode in ("open", "mixed")

    def top_k(self) -> int:
        """검색에서 남길 청크 수. 전부 가설이므로 측정 후 조정한다.

        비대칭이 있다: closed 를 open 으로 오판하면 근거가 늘 뿐이지만,
        open 을 closed 로 오판하면 항목이 빠진다. 그래서 애매하면 크게 잡는다.
        """
        if self.task == "correction_diff":
            return 24
        if self.is_open_ended:
            return 24
        if self.needs_multiple_documents:
            # 대상 하나당 10칸. 회사 2곳이든 공시유형 2종이든 같은 이유로 늘린다.
            return 10 * max(1, len(self.companies), len(self.report_kinds))
        return 8

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


__all__ = [
    "QueryPlan",
    "Scope", "SCOPES",
    "AnswerMode", "ANSWER_MODES",
    "Task", "TASKS",
    "LatestPolicy", "LATEST_POLICIES",
    "Aggregation", "AGGREGATIONS",
    "FieldSource",
]


# ============================================================================
# 규칙 기반 빌더
# ============================================================================
#
# 이 아래는 전부 규칙이다. LLM 을 부르지 않는다.
#
# `scope` 는 여기서 채우지 않는다 — Stage 3(scope_gate)이 Entity Registry 를
# 보고 채우고 동시에 차단 여부까지 결정한다. 두 곳에서 같은 판정을 하면
# 어긋날 때 어느 쪽이 맞는지 알 수 없으므로 소유자를 하나로 둔다.

import re  # noqa: E402
import unicodedata  # noqa: E402

# ---------------------------------------------------------------- answer_mode
#
# 질문의 **끝말(서술어)** 만 본다. 내용 단어(자산총계·계약금액)는 답의 모양을
# 결정하지 못하기 때문이다:
#
#     "한미반도체의 자산총계는 얼마인가?"      -> 숫자 하나  (closed)
#     "한미반도체의 자산총계 추이를 설명해줘." -> 문단      (open)
#
# suite_v1 38문항 실측: 34/38 일치. 틀린 4건(S023~S026)은 라벨이 open 인데
# 규칙이 mixed 라 한 것이고, 문장을 보면 규칙이 맞다("있는가?" + "설명해줘").
# mixed 를 open 의 상위집합으로 처리하면 오분류 0건.

_CLOSED_ENDINGS = (
    "얼마인가", "얼마야", "얼마인지", "얼마입니까", "얼마죠",
    "몇 ", "몇건", "몇 건", "몇개", "몇 개",
    "어디인가", "어디야", "누구인가", "누구야", "언제인가", "언제야",
    "무엇인가", "뭐야", "존재하는가", "있는가", "있나요", "있습니까",
    "더 큰", "더 많은", "가장 큰", "가장 많은", "맞는가",
)

_OPEN_ENDINGS = (
    "정리해", "설명해", "요약해", "서술해", "알려줘", "말해줘",
    "어떻게 변화", "어떻게 달라", "무엇이 달라", "어떤 내용", "어떤 차이",
    "비교했을 때", "기준으로 주요",
)


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def classify_answer_mode(query: str) -> AnswerMode:
    q = _nfc(query)
    closed = any(w in q for w in _CLOSED_ENDINGS)
    open_ = any(w in q for w in _OPEN_ENDINGS)
    if closed and open_:
        return "mixed"
    if closed:
        return "closed"
    if open_:
        return "open"
    return "unknown"


# ---------------------------------------------------------------- task
#
# 우선순위가 있다. 위에서 걸리면 아래는 보지 않는다.
#   correction_diff > count > compare > calculate > timeline > summarize > lookup
#
# compare 를 calculate 보다 위에 두는 이유: "A와 B 중 최대 계약금액은 얼마이며
# 더 큰 쪽은?"(S007~S014)은 최대값 계산이 들어가지만 본질은 비교다. 비교로
# 분류해야 회사별로 질문이 쪼개진다(needs_multiple_documents).

_CORRECTION_WORDS = ("정정", "기재정정")
_DIFF_WORDS = ("달라", "차이", "변경", "바뀐", "전후", "비교")
_COUNT_WORDS = ("몇 건", "몇건", "건수", "개수", "몇 개", "몇개", "총 몇")
_COMPARE_WORDS = ("비교", "더 큰", "더 많은", "더 높은", "중 어느", "어느 기업", "대비 어느")
_CALC_WORDS = ("증감률", "증가율", "감소율", "성장률", "비율", "CAGR", "합계는",
               "평균", "대비 몇", "얼마나 증가", "얼마나 감소")
_TIMELINE_WORDS = ("추이", "연도별", "분기별", "시간순", "변화 과정", "이력")
_SUMMARY_WORDS = ("정리해", "요약해", "주요 내용", "핵심", "어떤 내용")


def classify_task(query: str, *, n_companies: int = 0, mode: AnswerMode = "unknown") -> Task:
    q = _nfc(query)
    has = lambda words: any(w in q for w in words)  # noqa: E731

    if has(_CORRECTION_WORDS) and has(_DIFF_WORDS):
        return "correction_diff"
    if has(_COUNT_WORDS):
        return "count"
    if n_companies >= 2 or has(_COMPARE_WORDS):
        return "compare"
    if has(_CALC_WORDS):
        return "calculate"
    if has(_TIMELINE_WORDS):
        return "timeline"
    if mode in ("open", "mixed") or has(_SUMMARY_WORDS):
        return "summarize"
    if mode == "closed":
        return "lookup"
    return "unknown"


# ---------------------------------------------------------------- latest_policy
#
# 정정본을 어떻게 다룰지. 실측상 정정공시의 43%가 원본과 텍스트가 거의 같아서,
# 정리하지 않으면 같은 내용이 top-k 를 채운다.

_ALL_VERSION_WORDS = ("이력", "전부", "모든 정정", "몇 번 정정", "정정 내역 전부")
_FIRST_AND_FINAL_WORDS = ("최초", "최종", "정정 전", "정정전", "전후", "달라", "차이")


_MAX_WORDS = ("최대", "최고", "가장 큰", "가장 많은", "가장 높은", "최다", "최고액")
_MIN_WORDS = ("최소", "최저", "가장 작은", "가장 적은", "가장 낮은", "최소액")


def detect_aggregation(query: str) -> Aggregation:
    """질문이 값들 중 하나를 고르라고 하는가.

    이게 없으면 Facts 는 '최근 N건'을 준다. 최댓값이 오래된 공시에 있으면
    그대로 놓친다(실측: 삼성바이오로직스 54건 중 최댓값이 상위 10건 밖).
    """
    q = _nfc(query)
    if any(w in q for w in _COUNT_WORDS):
        return "count"
    if any(w in q for w in _MAX_WORDS):
        return "max"
    if any(w in q for w in _MIN_WORDS):
        return "min"
    return "none"


def decide_latest_policy(query: str) -> LatestPolicy:
    q = _nfc(query)
    if not any(w in q for w in _CORRECTION_WORDS):
        return "latest_only"
    if any(w in q for w in _ALL_VERSION_WORDS):
        return "all_versions"
    if any(w in q for w in _FIRST_AND_FINAL_WORDS):
        return "first_and_final"
    return "latest_only"


# ---------------------------------------------------------------- report_types

_DOC_GROUP_WORDS: dict[str, tuple[str, ...]] = {
    "periodic": ("사업보고서", "반기보고서", "분기보고서", "재무제표", "감사보고서"),
    "major": ("주요사항보고서",),
    "holding": ("대량보유", "주식등의 대량보유", "지분 공시", "5% 공시"),
    "exchange": ("단일판매", "공급계약", "신규시설투자", "투자판단", "자율공시"),
}


def detect_report_types(query: str) -> list[str]:
    q = _nfc(query)
    out = [g for g, words in _DOC_GROUP_WORDS.items() if any(w in q for w in words)]
    return out


# ---------------------------------------------------------------- 빌더


class RulePlanBuilder:
    """질문 -> QueryPlan. HCX 를 부르지 않는다.

    의존 두 개는 모두 선택이다. 없으면 그 필드를 비워둘 뿐 죽지 않는다:

        schema     ⓑ 공시유형별 표준항목 명세 (report_kinds / expected_fields)
        extractor  회사·기간·지표 추출기 (companies / periods)

    `scope` 는 채우지 않는다 — Stage 3 이 소유한다.
    """

    def __init__(self, *, schema=None, extractor=None):
        self.schema = schema
        self.extractor = extractor

    # ------------------------------------------------------------------

    def build(self, query: str, *, entities=None) -> QueryPlan:
        q = _nfc(query)
        src: dict[str, FieldSource] = {}

        if entities is None and self.extractor is not None:
            entities = self.extractor.extract(query)

        companies = list(getattr(entities, "companies", []) or [])
        raw_periods = list(getattr(entities, "period", []) or [])
        metrics = list(getattr(entities, "metrics", []) or [])
        if companies:
            src["companies"] = "rule"

        periods = self._normalize_periods(raw_periods)
        if periods:
            src["periods"] = "rule"

        report_kinds = self.schema.match_kinds(q) if self.schema is not None else []
        if report_kinds:
            src["report_kinds"] = "rule"

        report_types = detect_report_types(q)
        # 유형이 확정됐으면 그쪽 doc_group 을 우선한다 — 키워드 매칭보다 정확하다.
        if self.schema is not None:
            from_kinds = [self.schema.get(k).doc_group for k in report_kinds
                          if self.schema.get(k) is not None]
            for g in from_kinds:
                if g and g not in report_types:
                    report_types.append(g)
        if report_types:
            src["report_types"] = "rule"

        mode = classify_answer_mode(q)
        if mode != "unknown":
            src["answer_mode"] = "rule"

        task = classify_task(q, n_companies=len(companies), mode=mode)
        if task != "unknown":
            src["task"] = "rule"

        latest_policy = decide_latest_policy(q)
        src["latest_policy"] = "rule"

        aggregation = detect_aggregation(q)
        if aggregation != "none":
            src["aggregation"] = "rule"

        expected_fields = self._expected_fields(q, mode=mode, metrics=metrics,
                                                report_kinds=report_kinds)
        if expected_fields:
            src["expected_fields"] = "rule"

        operations = self._operations(companies=companies, periods=periods,
                                      report_kinds=report_kinds, task=task)
        # 공시유형이 2개 이상이면 문서도 2건 이상 필요하다. suite_v1 의 S015~S022
        # 여덟 문항이 이 경우다 — "체결한 계약 중 이후 해지된 것이 있는가?"는
        # 회사 1개·기간 0개·task=lookup 이라 나머지 조건에 전부 걸리지 않지만,
        # 체결 공시와 해지 공시를 **둘 다** 찾아야 답이 나온다.
        needs_multi = (
            len(companies) >= 2
            or len(periods) >= 2
            or len(report_kinds) >= 2
            or task in ("compare", "correction_diff", "timeline")
        )

        return QueryPlan(
            answer_mode=mode,
            task=task,
            companies=companies,
            periods=periods,
            report_types=report_types,
            report_kinds=report_kinds,
            latest_policy=latest_policy,
            aggregation=aggregation,
            expected_fields=expected_fields,
            needs_multiple_documents=needs_multi,
            operations=operations,
            source=src,
        )

    # ------------------------------------------------------------------ 내부

    @staticmethod
    def _normalize_periods(raw: list[str]) -> list[str]:
        """자연어 기간을 chunk 의 period 포맷으로 바꾼다.

        이 변환을 빠뜨리면 필터가 영구 0건이 된다 — "2024년" 을 그대로 넣으면
        chunk.period("2024-12")와 절대 일치하지 않는다. 실측으로 확인된 버그다.
        """
        if not raw:
            return []
        try:
            from disclosure_rag.retrieval.metadata_filter import normalize_period_tokens
        except Exception:  # noqa: BLE001
            return []
        return list(normalize_period_tokens(raw) or [])

    def _expected_fields(self, query: str, *, mode: AnswerMode,
                         metrics: list[str], report_kinds: list[str]) -> list[str]:
        """closed 는 질문이 지목한 항목, open 은 공시유형의 required.

        두 용법이 다르다. "투자금액은 얼마인가"의 충족 조건은 투자금액 하나지,
        신규시설투자등의 required 11개 전부가 아니다. 반대로 "주요 내용을
        정리해줘"에는 질문이 지목한 항목이 없으므로 명세에서 가져와야 한다.

        **closed 에서 항목을 못 찾으면 빈 목록을 돌려준다(fail open).**
        required 전부로 대체하면 답할 수 있는 질문이 충분성 검사에서 막히고
        거부까지 간다 — 통합 테스트에서 실제로 발생했다.
        """
        named = sorted({_nfc(m).strip() for m in metrics if m})

        if mode == "closed":
            if named:
                return named
            if self.schema is not None:
                # 손으로 쓴 지표 목록(35줄)에 없어도, 명세의 항목명으로 찾는다.
                return self.schema.fields_mentioned(query, report_kinds)
            return []

        if self.schema is None:
            return named
        return self.schema.expected_fields(query) or named

    @staticmethod
    def _operations(*, companies: list[str], periods: list[str],
                    report_kinds: list[str], task: Task) -> list[str]:
        """질문을 원자 작업으로 쪼갠 목록. Stage 4 가 이 길이로 분해를 결정한다."""
        ops: list[str] = []
        for c in companies:
            ops.append(f"조회:{c}")
        if not ops and periods:
            ops += [f"조회:{p}" for p in periods]
        # 회사는 하나인데 공시유형이 여럿이면 유형별로 따로 찾아야 한다.
        if len(companies) <= 1 and len(report_kinds) >= 2:
            ops = [f"조회:{k}" for k in report_kinds]
        if task == "compare":
            ops.append("비교")
        elif task == "calculate":
            ops.append("연산")
        elif task == "correction_diff":
            ops += ["최초본조회", "최종본조회", "차이산출"]
        elif task == "count":
            ops.append("집계")
        elif task == "timeline":
            ops.append("시간정렬")
        return ops


# ============================================================================
# 계획 검증기
# ============================================================================
#
# HCX 가 채운 계획을 그대로 실행하면 안 된다. 존재하지 않는 회사·필드·작업을
# 계획에 넣어도 막을 방법이 없기 때문이다. 규칙이 마지막에 한 번 거른다.
#
# 규칙이 만든 계획도 같이 검증한다 — 검증기가 규칙의 버그도 잡아준다.


@dataclass
class PlanIssue:
    field: str
    severity: Literal["error", "warning"]
    message: str
    repaired_to: object | None = None


@dataclass
class PlanValidation:
    issues: list[PlanIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> list[PlanIssue]:
        return [i for i in self.issues if i.severity == "error"]

    def __bool__(self) -> bool:
        return self.ok


class PlanValidator:
    """계획의 자기모순과 존재하지 않는 값을 잡는다.

    `repair=True` 면 고칠 수 있는 것은 고치고 무엇을 고쳤는지 남긴다. 고칠 수
    없는 것(존재하지 않는 회사 등)은 error 로 남긴다.
    """

    def __init__(self, *, registry=None, schema=None):
        self.registry = registry
        self.schema = schema

    def validate(self, plan: QueryPlan, *, repair: bool = True) -> PlanValidation:
        out = PlanValidation()

        self._check_enums(plan, out, repair)
        self._check_companies(plan, out)
        self._check_periods(plan, out, repair)
        self._check_task_consistency(plan, out, repair)
        self._check_fields(plan, out, repair)

        return out

    # ------------------------------------------------------------------

    @staticmethod
    def _check_enums(plan: QueryPlan, out: PlanValidation, repair: bool) -> None:
        for name, allowed, fallback in (
            ("scope", SCOPES, "possibly_scope"),
            ("answer_mode", ANSWER_MODES, "unknown"),
            ("task", TASKS, "unknown"),
            ("latest_policy", LATEST_POLICIES, "latest_only"),
            ("aggregation", AGGREGATIONS, "none"),
        ):
            value = getattr(plan, name)
            if value in allowed:
                continue
            msg = f"{name}={value!r} 은 허용되지 않는 값이다"
            if repair:
                setattr(plan, name, fallback)
                out.issues.append(PlanIssue(name, "warning", msg, repaired_to=fallback))
            else:
                out.issues.append(PlanIssue(name, "error", msg))

    def _check_companies(self, plan: QueryPlan, out: PlanValidation) -> None:
        """회사가 코퍼스에 실재하는가. 없는 회사를 계획에 넣으면 검색이 헛돈다.

        여기서 '없다'는 유니버스 70개사에 없다는 뜻이 아니다 — 계약상대·주주·
        자회사·제출인 어느 역할로든 등장하지 않는다는 뜻이다. 거부 여부는
        Stage 3/11 이 정하고, 여기서는 사실만 기록한다.
        """
        if self.registry is None:
            return
        for name in plan.companies:
            if not self.registry.types_for(name):
                out.issues.append(PlanIssue(
                    "companies", "error",
                    f"'{name}' 은 코퍼스 어디에도 등장하지 않는다"))

    @staticmethod
    def _check_periods(plan: QueryPlan, out: PlanValidation, repair: bool) -> None:
        """`YYYY` 또는 `YYYY-MM` 이어야 한다. 자연어가 남아 있으면 필터가 0건이 된다."""
        pat = re.compile(r"^\d{4}(-\d{2})?$")
        bad = [p for p in plan.periods if not pat.match(p)]
        if not bad:
            return
        msg = f"기간 형식이 아니다: {bad} (YYYY 또는 YYYY-MM 이어야 한다)"
        if repair:
            plan.periods = [p for p in plan.periods if pat.match(p)]
            out.issues.append(PlanIssue("periods", "warning", msg, repaired_to=plan.periods))
        else:
            out.issues.append(PlanIssue("periods", "error", msg))

    @staticmethod
    def _check_task_consistency(plan: QueryPlan, out: PlanValidation, repair: bool) -> None:
        """작업과 나머지 필드가 앞뒤가 맞는가."""
        if plan.task == "compare" and len(plan.companies) < 2 and len(plan.periods) < 2:
            out.issues.append(PlanIssue(
                "task", "error",
                "task=compare 인데 비교 대상이 하나뿐이다 (회사·기간 모두 1개 이하)"))

        if plan.task == "correction_diff" and plan.latest_policy == "latest_only":
            msg = "task=correction_diff 인데 latest_policy=latest_only 다 — 최초본이 버려진다"
            if repair:
                plan.latest_policy = "first_and_final"
                out.issues.append(PlanIssue("latest_policy", "warning", msg,
                                            repaired_to="first_and_final"))
            else:
                out.issues.append(PlanIssue("latest_policy", "error", msg))

        if plan.task == "calculate" and not plan.expected_fields:
            out.issues.append(PlanIssue(
                "expected_fields", "error",
                "task=calculate 인데 피연산자 필드가 지정되지 않았다"))

        if plan.needs_multiple_documents and not plan.operations:
            out.issues.append(PlanIssue(
                "operations", "warning",
                "여러 문서가 필요한데 원자 작업 목록이 비어 있다 — 분해가 안 된다"))

    def _check_fields(self, plan: QueryPlan, out: PlanValidation, repair: bool) -> None:
        """요구 항목이 그 공시유형에 실제로 존재하는 항목인가.

        존재하지 않는 항목을 요구하면 충분성 검사가 영원히 만족되지 않아,
        답할 수 있는 질문이 '근거 부족'으로 거부된다.
        """
        if self.schema is None or not plan.report_kinds or not plan.expected_fields:
            return
        unknown = []
        for f in plan.expected_fields:
            if all(self.schema.classify(kind, f) == "unknown" for kind in plan.report_kinds):
                unknown.append(f)
        if not unknown:
            return
        msg = f"공시유형 {plan.report_kinds} 에 없는 항목을 요구한다: {unknown}"
        if repair:
            plan.expected_fields = [f for f in plan.expected_fields if f not in unknown]
            out.issues.append(PlanIssue("expected_fields", "warning", msg,
                                        repaired_to=plan.expected_fields))
        else:
            out.issues.append(PlanIssue("expected_fields", "error", msg))


__all__ += [
    "RulePlanBuilder", "PlanValidator", "PlanValidation", "PlanIssue",
    "classify_answer_mode", "classify_task", "decide_latest_policy", "detect_report_types",
    "detect_aggregation",
]


# ============================================================================
# HCX 폴백 — 규칙이 못 채운 칸만 채운다
# ============================================================================
#
# suite_v1 38문항에서는 `is_complete` 가 38/38 이라 **한 번도 호출되지 않는다**.
# 보험이지 주력이 아니다. 그래서 다음 두 가지를 지킨다:
#
#   1. 계획 전체를 자유롭게 만들라고 하지 않는다. 빈 칸만 채우게 한다.
#      LLM 이 존재하지 않는 회사·항목·작업을 계획에 넣는 것을 막기 위해서다.
#   2. 결과는 반드시 PlanValidator 를 거친다.
#
# 호출을 최소로 유지하는 이유: 60문항 실행에서 6개를 HCX 429 로 날린 전적이 있다.

_FILL_PROMPT = """다음 질문을 분류해라. 아래 두 가지만 판단한다.

answer_mode: 답이 (A) 숫자나 이름 하나로 끝나는가 = closed
                    (B) 여러 항목을 나열해야 하는가 = open
                    (C) 둘 다인가 = mixed
task: lookup | compare | calculate | timeline | summarize | correction_diff | count

질문: {query}

JSON 만 출력해라: {{"answer_mode": "...", "task": "..."}}"""


def fill_missing_with_hcx(plan: QueryPlan, query: str, client) -> QueryPlan:
    """규칙이 비워둔 answer_mode / task 만 HCX 로 채운다.

    `client` 는 `.chat(messages) -> str` 만 있으면 된다(테스트에서 가짜를 넣는다).
    실패하면 조용히 원래 계획을 돌려준다 — 분류 하나 때문에 질문 전체를 버리지
    않는다. 채운 값은 `source` 에 'hcx' 로 표시되어 검증기가 골라낼 수 있다.
    """
    if plan.is_complete:
        return plan
    try:
        raw = client.chat([{"role": "user", "content": _FILL_PROMPT.format(query=query)}])
        m = re.search(r"\{.*\}", raw or "", re.S)
        data = __import__("json").loads(m.group(0)) if m else {}
    except Exception:  # noqa: BLE001
        plan.notes.append("HCX 계획 보완 실패 — 규칙 결과를 그대로 쓴다")
        return plan

    if plan.answer_mode == "unknown" and data.get("answer_mode") in ANSWER_MODES:
        plan.answer_mode = data["answer_mode"]
        plan.source["answer_mode"] = "hcx"
    if plan.task == "unknown" and data.get("task") in TASKS:
        plan.task = data["task"]
        plan.source["task"] = "hcx"

    # 규칙이 answer_mode 를 못 정했을 때의 기본값은 open 쪽이다. closed 를
    # open 으로 오판하면 근거가 늘 뿐이지만, 반대는 항목이 빠진다.
    if plan.answer_mode == "unknown":
        plan.answer_mode = "open"
        plan.source["answer_mode"] = "default"
        plan.notes.append("answer_mode 미확정 — 안전한 쪽(open)으로 기본값 적용")
    return plan


__all__ += ["fill_missing_with_hcx"]
