"""전체 온라인 파이프라인 진입점 (§34 다이어그램의 우측 절반):
질문 -> Entity/Router -> HCX Agent Loop(Tool Calling) -> Evidence Pack -> HCX Answer -> Validation.

## 2026-08-30 — 검증 경고에 실제로 조치한다

그 전까지 이 함수는 `validation.warnings` 를 로그로 찍기만 하고 그대로
반환했다. 실측(상원 레포 100문항 배치)에서 `정정 분석 질문인데 원본/정정본
근거가 모두 확보되지 않음` 이 **해당 질문의 100%(16/16)** 에서 떴는데도
재검색이 한 번도 일어나지 않았다.

재생성을 붙일 때 조심할 게 하나 있다. 상원 레포는 재생성 결과로 **무조건
덮어썼는데**, 실측에서 재생성된 답변이 `"제공된 근거로는 ... 확인할 수
없습니다"` 가 되면서 숫자가 사라져 지표만 통과하고 답변 품질은 떨어졌다.
그래서 여기서는 원본과 재생성본을 **같은 기준으로 채점해서 더 나은 쪽을
고른다**. 동점이면 원본을 유지한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from disclosure_rag.agent.agent_loop import AgentTrace, run_agent_loop
from disclosure_rag.agent.answer_generator import generate_answer
from disclosure_rag.agent.evidence import EvidencePack, build_evidence_pack
from disclosure_rag.agent.hcx_client import HCXClient
from disclosure_rag.agent.tools import ToolDef
from disclosure_rag.agent.validator import ValidationResult, validate_answer
from disclosure_rag.entity.entity_extractor import EntityExtractor
from disclosure_rag.router.semantic_router_wrapper import Router

logger = logging.getLogger(__name__)

# 답변이 사실상 거부인지 — 재생성이 "모르겠다"로 도망가는 걸 잡는다.
_REFUSAL_MARKERS = ("확인할 수 없습니다", "확인되지 않습니다", "찾을 수 없습니다")


@dataclass
class AskResult:
    question: str
    trace: AgentTrace
    evidence_pack: EvidencePack
    answer: str
    validation: ValidationResult
    # 2026-08-30 추가 — 무엇을 왜 다시 했는지 추적 가능하게 남긴다.
    remediation: list[str] = field(default_factory=list)


def _is_refusal(answer: str) -> bool:
    return any(m in answer for m in _REFUSAL_MARKERS)


def _score(answer: str, validation: ValidationResult) -> tuple[int, int, int, int]:
    """클수록 좋은 답변. 재생성본을 무조건 채택하지 않기 위한 기준."""
    return (
        int(validation.has_any_evidence),
        int(validation.numbers_grounded),
        int(validation.has_citation),
        int(not _is_refusal(answer)),
    )


def _correction_note(validation: ValidationResult) -> str | None:
    """검증 결과에서 **원인별** 재작성 지시를 만든다. 원인을 단정하지 않는다."""
    notes: list[str] = []
    if validation.rescaled_numbers:
        notes.append(
            "근거의 숫자를 다른 단위로 바꿔 쓰지 마세요. 근거에 적힌 표기를 "
            f"글자 그대로 옮기세요. 문제가 된 표기: {sorted(validation.rescaled_numbers)}"
        )
    plain_ungrounded = sorted(set(validation.ungrounded_numbers) - set(validation.rescaled_numbers))
    if plain_ungrounded:
        notes.append(
            "다음 숫자는 근거에서 확인되지 않습니다. 근거에 있는 값만 쓰거나, "
            "계산한 값이라면 계산에 쓴 입력 수치를 답변에 함께 적으세요: "
            f"{plain_ungrounded}"
        )
    if not validation.has_citation and validation.has_any_evidence:
        notes.append("각 수치 뒤에 근거의 report_id 를 그대로 적으세요.")
    return "\n".join(notes) if notes else None


def ask(
    client: HCXClient,
    tools: list[ToolDef],
    question: str,
    *,
    entity_extractor: EntityExtractor,
    router: Router | None = None,
    max_iterations: int = 6,
    max_answer_retries: int = 1,
) -> AskResult:
    trace = run_agent_loop(
        client, tools, question,
        entity_extractor=entity_extractor, router=router, max_iterations=max_iterations,
    )
    evidence_pack = build_evidence_pack(trace)
    answer = generate_answer(client, evidence_pack)
    validation = validate_answer(answer, evidence_pack, trace.entities)
    remediation: list[str] = []

    for attempt in range(max_answer_retries):
        if validation.passed:
            break
        note = _correction_note(validation)
        if note is None:
            # 재작성으로 고칠 수 있는 종류가 아니다(예: 근거 0건, 정정 체인 부족).
            # 지어내는 방향으로 유도하지 않는다 — 여기서 멈추는 게 맞다.
            break

        logger.info("[REMEDIATION] question=%r 재생성 %d회차 사유=%s", question, attempt + 1, note)
        retry_answer = generate_answer(client, evidence_pack, correction_note=note)
        retry_validation = validate_answer(retry_answer, evidence_pack, trace.entities)

        if _score(retry_answer, retry_validation) > _score(answer, validation):
            remediation.append(f"재생성 {attempt + 1}회차 채택: {note.splitlines()[0]}")
            answer, validation = retry_answer, retry_validation
        else:
            # 상원 레포가 여기서 무조건 덮어써서, 숫자가 사라진 거부 답변이
            # "지표 통과" 로 기록됐다. 나빠지면 원본을 유지한다.
            remediation.append(f"재생성 {attempt + 1}회차 기각(개선 없음): {note.splitlines()[0]}")
            break

    if validation.warnings:
        for w in validation.warnings:
            logger.warning("[VALIDATION] question=%r %s", question, w)

    return AskResult(
        question=question, trace=trace, evidence_pack=evidence_pack,
        answer=answer, validation=validation, remediation=remediation,
    )
