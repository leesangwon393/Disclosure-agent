"""신 파이프라인 진입점 — 결정론적 14단계.

## 구 경로(`ask.py`)와 무엇이 다른가

구 경로는 **HCX 가 도구를 호출할지 스스로 정한다.** 실측상 그게 잘 안 됐다:
100문항 중 54건이 도구 1회 호출 후 종료했고, 정정 이력 도구는 부르라고 해도
안 불렀다. 신 경로는 무엇을 몇 번 검색할지 **규칙이 정한다.**

HCX 호출 지점이 셋뿐이고, 앞의 둘은 거의 안 탄다:

    2b. QueryPlan 보완    규칙이 못 정했을 때만 (suite_v1 38문항 중 0건)
    14. 답변 생성         유일한 필수 호출
    15. 재생성            Validator 실패 시

그리고 **범위 게이트와 거부 게이트를 통과하지 못하면 HCX 를 아예 부르지
않는다.** 지어낼 기회 자체가 없다.

## 단계

     1  질의 이해          정규화 + 회사·기간·지표 추출
     2  QueryPlan          규칙 -> (필요시) HCX -> 규칙 검증
     3  범위 게이트         hard_out_scope 만 즉시 거부
     4  질문 분해          작업 복잡도로 판정 (answer_mode 아님)
    5~7 이중채널 검색       정형(Facts) ∥ 비정형(BM25/Dense/Sparse) + 버전 정리
     8  부모 확장          leaf -> 상위 문단
     9  근거 구조화         항목별 값 추출, 정정 짝 맞추기
    10  충분성 검사         부족하면 4로 복귀 (최대 2회)
    11  거부 게이트         여기서 막히면 HCX 안 부름
    13  Evidence Pack
    14  답변 생성          answer_mode 별 프롬프트
    15  Validator
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from disclosure_rag.agent.abstention import AbstentionDecision, decide_from_processed
from disclosure_rag.agent.answer_generator import generate_answer
from disclosure_rag.agent.decompose import DecomposeResult, decompose_and_search
from disclosure_rag.agent.evidence import EvidencePack, build_evidence_pack_from_retrieval
from disclosure_rag.agent.evidence_processor import ProcessedEvidence, process_evidence
from disclosure_rag.agent.query_plan import (
    PlanValidation,
    QueryPlan,
    RulePlanBuilder,
    fill_missing_with_hcx,
)
from disclosure_rag.agent.scope_gate import ScopeDecision, apply_scope_gate
from disclosure_rag.agent.sufficiency import (
    DEFAULT_MAX_NUDGES,
    SufficiencyReport,
    check_sufficiency,
)

logger = logging.getLogger(__name__)

# 프롬프트 길이 상한. 실측(diag_answer.py): 근거 13,542자에서 HCX 가 정답을
# 못 찾은 사례가 있다. open 질문은 청크 24개가 들어와 쉽게 넘어간다.
DEFAULT_MAX_EVIDENCE_CHARS = 12_000


@dataclass
class AskV2Result:
    question: str
    answer: str
    plan: QueryPlan | None = None
    validation: PlanValidation | None = None
    scope: ScopeDecision | None = None
    decomposed: DecomposeResult | None = None
    processed: ProcessedEvidence | None = None
    sufficiency: SufficiencyReport | None = None
    abstention: AbstentionDecision | None = None
    evidence: list = field(default_factory=list)   # 부모 확장까지 끝난 최종 근거
    evidence_pack: EvidencePack | None = None
    validation_result: Any = None
    hcx_calls: int = 0
    retries: int = 0
    thinking: dict | None = None      # 이 질문에 실제로 쓴 설정 (A/B 추적용)
    stopped_at: str = "answered"      # 어느 단계에서 끝났는지 — 진단용
    notes: list[str] = field(default_factory=list)

    @property
    def citations(self) -> list:
        return list(self.evidence_pack.citations) if self.evidence_pack else []


class AskV2:
    """부품을 한 번 조립해 두고 질문마다 `run()` 을 부른다.

    인덱스 로딩이 90초 넘게 걸리므로 질문마다 다시 만들지 않는다.
    """

    def __init__(
        self, *, client, dual_retriever, plan_builder: RulePlanBuilder,
        plan_validator=None, registry=None, parent_expander=None,
        answer_validator=None, max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
        max_nudges: int = DEFAULT_MAX_NUDGES, thinking_policy: str = "off",
    ):
        self.client = client
        self.dual = dual_retriever
        self.plan_builder = plan_builder
        self.plan_validator = plan_validator
        self.registry = registry
        self.parent_expander = parent_expander
        self.answer_validator = answer_validator
        self.max_evidence_chars = max_evidence_chars
        self.max_nudges = max_nudges
        # off | auto | on. 기본 off — 지금까지의 운영 상태와 동일하다.
        # answer_generator.resolve_thinking 참조.
        self.thinking_policy = thinking_policy

    # ------------------------------------------------------------------

    def run(self, question: str) -> AskV2Result:
        out = AskV2Result(question=question, answer="")
        # 질문마다 초기화한다. 조기 종료(범위·거부 게이트)하면 `_facts_of` 가
        # 안 불려서, 이걸 안 비우면 **앞 질문의 Facts 가 다음 질문에 섞인다.**
        self._last_facts = []

        # --- 1~2. 질의 이해 + 계획 ---------------------------------------
        plan = self.plan_builder.build(question)
        if not plan.is_complete and self.client is not None:
            plan = fill_missing_with_hcx(plan, question, self.client)
            if plan.source.get("answer_mode") == "hcx" or plan.source.get("task") == "hcx":
                out.hcx_calls += 1
        out.plan = plan
        if self.plan_validator is not None:
            out.validation = self.plan_validator.validate(plan)
            if not out.validation.ok:
                # 계획이 자기모순이면 검색해봐야 헛돈다. 다만 계획 오류를
                # 사용자에게 그대로 노출하지 않고 일반 거부 문구를 쓴다.
                out.notes += [i.message for i in out.validation.errors]

        # --- 3. 범위 게이트 ----------------------------------------------
        if self.registry is not None:
            out.scope = apply_scope_gate(plan, question, self.registry)
            if out.scope.should_refuse:
                out.stopped_at = "scope_gate"
                out.answer = out.scope.message or "확인되지 않습니다."
                return out          # HCX 호출 0회

        # --- 4~10. 검색 -> 구조화 -> 충분성 (부족하면 재검색) --------------
        hint = ""
        for attempt in range(self.max_nudges + 1):
            out.retries = attempt
            decomposed = decompose_and_search(
                plan, question if not hint else f"{question}\n{hint}", self._search)
            merged = self._expand_parents(decomposed.merged)
            processed = process_evidence(plan, merged)
            report = check_sufficiency(plan, processed, decompose_result=decomposed,
                                       nudges_used=attempt)
            out.decomposed, out.processed, out.sufficiency = decomposed, processed, report
            out.evidence = merged
            if report.ok or not report.should_retry:
                break
            hint = report.retry_message()
            logger.info("[ASKv2] 재검색 %d회차: %s", attempt + 1, report.reasons)

        # --- 11. 거부 게이트 ---------------------------------------------
        out.abstention = decide_from_processed(
            plan, out.processed, decompose_result=out.decomposed)
        if out.abstention.should_abstain:
            out.stopped_at = "abstention_gate"
            out.answer = out.abstention.message or "확인되지 않습니다."
            return out              # HCX 호출 0회

        # --- 13. Evidence Pack -------------------------------------------
        # 확장 전 목록(decomposed.merged)이 아니라 **부모 확장까지 끝난** 것을 쓴다.
        out.evidence_pack = build_evidence_pack_from_retrieval(
            question, out.evidence,
            facts=self._facts_of(out.decomposed),
            aggregation=getattr(plan, "aggregation", "none"),
            max_chars=self.max_evidence_chars,
        )

        # --- 14. 답변 생성 ------------------------------------------------
        answer = generate_answer(self.client, out.evidence_pack, plan=plan,
                                 thinking_policy=self.thinking_policy)
        out.hcx_calls += 1
        from disclosure_rag.agent.answer_generator import resolve_thinking
        out.thinking = resolve_thinking(plan, policy=self.thinking_policy)

        # --- 15. Validator -----------------------------------------------
        # 기본값으로 항상 검증한다. 지어낸 숫자를 LLM 없이 잡는 유일한 장치라
        # 껐다 켰다 할 성격이 아니다(과거 v2 경로에서 통째로 빠져 있었다).
        validator = self.answer_validator or _default_validator
        try:
            out.validation_result = validator(answer, out.evidence_pack, plan)
        except Exception:  # noqa: BLE001
            logger.warning("[ASKv2] 답변 검증 실패 — 검증 없이 진행")

        # --- 16. partial 이면 한계를 명시한다 ------------------------------
        out.answer = out.abstention.finalize(answer)
        out.stopped_at = "answered"
        return out

    # ------------------------------------------------------------------ 내부

    def _search(self, text: str, sub_plan: QueryPlan, k: int):
        result = self.dual.search(text, sub_plan, k=k)
        # 이 하위 질의의 Facts 를 결과에 붙여 둔다(병합 뒤에 모아 쓴다).
        self._last_facts = getattr(self, "_last_facts", [])
        self._last_facts.extend(result.facts)
        return result.unstructured_results

    def _facts_of(self, decomposed) -> list[dict]:
        facts = getattr(self, "_last_facts", [])
        self._last_facts = []
        seen, out = set(), []
        for row in facts:
            sig = (row.get("report_id") or row.get("doc_id"), row.get("chunk_id"),
                   row.get("item") or row.get("key_norm"), row.get("value") or row.get("value_text"))
            if sig in seen:
                continue
            seen.add(sig)
            out.append(row)
        return out

    def _expand_parents(self, scored):
        """leaf 청크의 본문을 상위 문단으로 넓힌다.

        `ParentExpander.expand()` 를 쓰지 않는 이유: 그쪽은 `list[dict]` 를
        돌려주는데, 그러면 청크의 `field_codes` 가 사라져 Stage 9(근거 구조화)가
        항목 값을 못 뽑는다. 그래서 `expand_one()` 으로 텍스트만 받아
        **청크 형태를 유지한 채** 본문을 교체한다.

        같은 parent 를 가진 청크가 여러 개면 **첫 번째만** 확장한다. 전부
        확장하면 같은 문단이 프롬프트에 여러 번 들어가 예산을 잡아먹는다.
        나머지는 원문을 유지하므로 구조 정보는 그대로 남는다.
        """
        if self.parent_expander is None:
            return list(scored)
        out, used_parents = [], set()
        for chunk, score in scored:
            pid = getattr(chunk, "parent_chunk_id", None)
            if pid and pid in used_parents:
                out.append((chunk, score))
                continue
            try:
                text = self.parent_expander.expand_one(chunk)
            except Exception:  # noqa: BLE001
                logger.warning("[ASKv2] 부모 확장 실패(%s) — 원본 청크 사용",
                               getattr(chunk, "chunk_id", "?"))
                out.append((chunk, score))
                continue
            if pid:
                used_parents.add(pid)
            out.append((_with_text(chunk, text), score))
        return out


def _default_validator(answer: str, evidence_pack, plan):
    """기존 `validate_answer` 를 신 파이프라인 형태에 맞춰 부른다.

    그쪽은 `ExtractedEntities` 를 받는데(구 파이프라인 산물), 실제로 쓰는 건
    `companies` 와 `explicit_correction` 둘뿐이다. QueryPlan 에서 그 둘을
    옮겨 담아 넘긴다 — 검증기를 고치지 않고 재사용한다.
    """
    from disclosure_rag.agent.validator import validate_answer
    from disclosure_rag.entity.entity_extractor import ExtractedEntities

    entities = ExtractedEntities(
        raw_query="",
        companies=list(getattr(plan, "companies", []) or []),
        company_count=len(getattr(plan, "companies", []) or []),
        explicit_correction=(getattr(plan, "task", "") == "correction_diff"
                             or getattr(plan, "latest_policy", "") != "latest_only"),
    )
    return validate_answer(answer, evidence_pack, entities)


def _with_text(chunk, text: str):
    """본문만 바꾼 사본. 나머지 메타데이터와 field_codes 는 그대로 둔다."""
    if hasattr(chunk, "model_copy"):          # pydantic (ChunkSchema)
        return chunk.model_copy(update={"raw_text": text})
    import copy
    clone = copy.copy(chunk)
    try:
        clone.raw_text = text
    except Exception:  # noqa: BLE001
        return chunk
    return clone


__all__ = ["AskV2", "AskV2Result", "DEFAULT_MAX_EVIDENCE_CHARS"]
