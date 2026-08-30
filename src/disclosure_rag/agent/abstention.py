"""Stage 11: 검색을 다 한 뒤 답변/거부/부분답변을 결정하는 최종 게이트.

규칙은 세 개뿐이다.

1. 공시 근거 0건 -> answer mode와 무관하게 거부.
2. 근거는 있지만 expected_fields가 부족 -> closed/unknown은 거부.
3. 같은 상황의 open/mixed -> 확인된 부분은 답하고 누락 필드를 명시.

LLM 판단이 아니라 QueryPlan과 근거 커버리지만 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from disclosure_rag.agent.field_schema import normalize_field_key
from disclosure_rag.agent.query_plan import QueryPlan

AbstentionAction = Literal["answer", "abstain", "partial"]
AbstentionReason = Literal[
    "evidence_absent", "required_fields_missing",
    # 2026-08-30 추가. 아래 둘은 '근거 건수'로는 절대 안 잡힌다:
    "target_evidence_missing",   # 두 회사 비교인데 한쪽 근거가 0건
    "version_pair_incomplete",   # 정정 비교인데 최초본/최종본 한쪽만 있음
    "sufficient",
]


def _normalized(values: Iterable[str]) -> set[str]:
    return {value for item in values if (value := normalize_field_key(item))}


@dataclass(frozen=True)
class AbstentionDecision:
    action: AbstentionAction
    reason: AbstentionReason
    evidence_count: int
    required_fields: tuple[str, ...] = ()
    found_fields: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    empty_targets: tuple[str, ...] = ()
    incomplete_pairs: tuple[str, ...] = ()

    @property
    def should_abstain(self) -> bool:
        return self.action == "abstain"

    @property
    def can_generate(self) -> bool:
        return self.action in ("answer", "partial")

    @property
    def message(self) -> str | None:
        if self.action == "answer":
            return None
        if self.reason == "evidence_absent":
            return "제공된 DART 공시 근거에서는 확인되지 않습니다."
        if self.reason == "target_evidence_missing":
            targets = ", ".join(self.empty_targets)
            if self.action == "partial":
                return f"근거를 확인하지 못한 대상: {targets}"
            return (f"비교 대상 중 {targets} 의 공시 근거를 확인할 수 없어 "
                    "결론을 내지 않습니다.")
        if self.reason == "version_pair_incomplete":
            items = ", ".join(self.incomplete_pairs)
            if self.action == "partial":
                return f"정정 전후 중 한쪽만 확인된 항목: {items}"
            return (f"{items} 항목의 정정 전후 중 한쪽만 확인되어 "
                    "변경 내역을 단정하지 않습니다.")
        fields = ", ".join(self.missing_fields)
        if self.action == "partial":
            return f"확인되지 않은 필수 항목: {fields}"
        return f"필수 항목({fields})을 공시 근거에서 확인할 수 없어 답변하지 않습니다."

    def finalize(self, verified_answer: str = "") -> str:
        """생성기 호출 전/후에 최종 형태를 강제한다."""
        if self.action == "abstain":
            return self.message or "제공된 DART 공시 근거에서는 확인되지 않습니다."
        if self.action == "partial":
            prefix = verified_answer.strip()
            return f"{prefix}\n\n한계: {self.message}" if prefix else f"확인된 내용이 없습니다.\n\n한계: {self.message}"
        return verified_answer


def decide_abstention(
    plan: QueryPlan, *, evidence_count: int,
    found_fields: Iterable[str] = (), required_fields: Iterable[str] | None = None,
    empty_targets: Iterable[str] = (), incomplete_pairs: Iterable[str] = (),
) -> AbstentionDecision:
    """검색/처리 결과를 받아 최종 조치를 결정한다.

    `required_fields`를 주지 않으면 `plan.expected_fields`를 쓴다.

    `empty_targets` / `incomplete_pairs` (2026-08-30 추가)
    -----------------------------------------------------
    근거 '건수'만 보면 통과하는데 실제로는 답을 못 만드는 두 경우다.

        "삼성전자와 한미반도체 중 최대 계약금액은?"
          -> 삼성전자 근거 10건, 한미반도체 0건
          -> 건수 10건이라 통과 -> 한쪽 값만으로 "더 크다"고 답한다

        "최초 공시와 최종 정정본의 차이는?"
          -> 최종본 근거 10건, 최초본 0건
          -> 건수 10건이라 통과 -> 비교 대상이 없는데 비교했다고 답한다

    Stage 4(분해)의 `empty_labels` 와 Stage 9(구조화)의 `incomplete_pairs` 가
    이 둘을 알려준다. 프롬프트에 "한쪽 근거가 없으면 결론 내지 마라"를 넣어
    뒀지만 그건 LLM 판단이라 게이트가 아니다. 여기서 규칙으로 막는다.
    """
    if evidence_count < 0:
        raise ValueError("evidence_count must be >= 0")

    required_display = tuple(dict.fromkeys(
        str(item).strip() for item in (plan.expected_fields if required_fields is None else required_fields)
        if str(item).strip()
    ))
    found_display = tuple(dict.fromkeys(str(item).strip() for item in found_fields if str(item).strip()))
    required_norm = _normalized(required_display)
    found_norm = _normalized(found_display)
    missing_norm = required_norm - found_norm
    missing_display = tuple(
        item for item in required_display if normalize_field_key(item) in missing_norm
    )

    if evidence_count == 0:
        return AbstentionDecision(
            action="abstain", reason="evidence_absent", evidence_count=0,
            required_fields=required_display, found_fields=found_display,
            missing_fields=missing_display or required_display,
        )
    empty_display = tuple(dict.fromkeys(str(t).strip() for t in empty_targets if str(t).strip()))
    pair_display = tuple(dict.fromkeys(str(t).strip() for t in incomplete_pairs if str(t).strip()))

    # 대상 자체가 비면 필드 부족보다 앞선다 — 한쪽 근거만으로 비교 결론을
    # 내는 게 항목 하나 빠지는 것보다 위험하다.
    if empty_display:
        return AbstentionDecision(
            action="partial" if plan.is_open_ended else "abstain",
            reason="target_evidence_missing", evidence_count=evidence_count,
            required_fields=required_display, found_fields=found_display,
            missing_fields=missing_display, empty_targets=empty_display,
            incomplete_pairs=pair_display,
        )
    if pair_display and plan.task == "correction_diff":
        return AbstentionDecision(
            action="partial" if plan.is_open_ended else "abstain",
            reason="version_pair_incomplete", evidence_count=evidence_count,
            required_fields=required_display, found_fields=found_display,
            missing_fields=missing_display, incomplete_pairs=pair_display,
        )
    if missing_display:
        action: AbstentionAction = "partial" if plan.is_open_ended else "abstain"
        return AbstentionDecision(
            action=action, reason="required_fields_missing", evidence_count=evidence_count,
            required_fields=required_display, found_fields=found_display,
            missing_fields=missing_display,
        )
    return AbstentionDecision(
        action="answer", reason="sufficient", evidence_count=evidence_count,
        required_fields=required_display, found_fields=found_display,
    )


def fields_found_in_text(expected_fields: Iterable[str], evidence_text: str) -> list[str]:
    """Evidence Processor가 없는 현재 파이프라인을 위한 보수적 fallback.

    텍스트에 필드명이 직접 등장한 것만 인정한다. 유사도나 LLM으로
    추측하지 않는다.
    """
    haystack = normalize_field_key(evidence_text)
    return [field for field in expected_fields
            if normalize_field_key(field) and normalize_field_key(field) in haystack]


def decide_from_evidence_pack(
    plan: QueryPlan, evidence_pack, *, found_fields: Iterable[str] | None = None,
) -> AbstentionDecision:
    """현재 EvidencePack을 Stage 11 입력으로 변환하는 adapter.

    근거 건수는 citation과 `doc_id/chunk_id`를 보존한 정형 fact 행만
    세어서 계산기 결과를 공시 근거로 오인하지 않는다.
    """
    count = len(getattr(evidence_pack, "citations", ()) or ())
    seen_refs: set[tuple[str, str]] = set()

    def visit(value):
        nonlocal count
        if isinstance(value, dict):
            doc_id = str(value.get("doc_id") or value.get("report_id") or "")
            chunk_id = str(value.get("chunk_id") or "")
            # get_latest_report 등이 주는 doc_id만으로는 내용 근거가 아니다.
            # 원문으로 되짚을 수 있는 chunk_id가 있을 때만 세다.
            if doc_id and chunk_id and (doc_id, chunk_id) not in seen_refs:
                seen_refs.add((doc_id, chunk_id))
                count += 1
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for item in getattr(evidence_pack, "tool_results_summary", ()) or ():
        if item.get("tool") not in {"calculate_growth_rate", "calculate_ratio", "calculate_cagr"}:
            visit(item.get("result"))

    if found_fields is None:
        prompt = getattr(evidence_pack, "prompt_text", "") or ""
        # prompt_text 첫부분의 [USER QUESTION]에도 요구 필드명이 있다.
        # 그 부분까지 검색하면 근거에 없는 항목이 항상 '발견'된다.
        starts = [i for marker in ("[EVIDENCE ", "[TOOL RESULT]")
                  if (i := prompt.find(marker)) >= 0]
        evidence_text = prompt[min(starts):] if starts else ""
        found_fields = fields_found_in_text(plan.expected_fields, evidence_text)
    return decide_abstention(plan, evidence_count=count, found_fields=found_fields)


__all__ = [
    "AbstentionDecision", "AbstentionAction", "AbstentionReason",
    "decide_abstention", "decide_from_evidence_pack", "fields_found_in_text",
]


def decide_from_processed(
    plan: QueryPlan, processed, *, decompose_result=None,
    evidence_count: int | None = None,
) -> AbstentionDecision:
    """**신 파이프라인의 표준 진입점.** Stage 9 결과를 그대로 받는다.

    왜 `decide_from_evidence_pack` 과 따로 두나
    -----------------------------------------
    그쪽은 프롬프트 텍스트에 항목명이 등장하는지로 '찾았다'를 판정한다
    (`fields_found_in_text`). 표에서 파싱된 구조를 안 보므로 Stage 9 와 결론이
    어긋날 수 있다 — 충분성 검사는 통과인데 거부 게이트는 실패하는(또는 반대인)
    상태가 된다. 판정을 한 곳으로 모은다.

    Stage 9(`ProcessedEvidence`)가 있으면 **항상 이 함수를 쓴다.**
    `decide_from_evidence_pack` 은 Stage 9 가 없는 구 경로용으로만 남긴다.
    """
    found = list(getattr(processed, "found_fields", []) or [])
    empty = list(getattr(decompose_result, "empty_labels", []) or [])
    pairs = [p.field for p in getattr(processed, "incomplete_pairs", [])] \
        if plan.task == "correction_diff" else []

    if evidence_count is None:
        merged = getattr(decompose_result, "merged", None)
        evidence_count = len(merged) if merged is not None else len(
            getattr(processed, "documents", ()) or ())

    return decide_abstention(
        plan, evidence_count=evidence_count, found_fields=found,
        empty_targets=empty, incomplete_pairs=pairs,
    )


__all__.append("decide_from_processed")
