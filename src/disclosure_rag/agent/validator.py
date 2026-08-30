"""Answer Validation (§67).

체크 항목:

1. 답변의 숫자가 근거에 있는가 (글자 그대로 / 유도 검산 / 단위 재환산 구분)
2. 근거 인용이 있는가
3. 정정 분석 질문인데 원본·정정본이 둘 다 확보됐는가

## 2026-08-30 개정 내역

- **유도 검산 추가.** 그 전까지는 근거 원문과 글자 그대로 일치하는 숫자만
  인정해서, 증감률·비율처럼 계산해서 나온 값이 전부 "근거 없는 숫자"로
  오탐됐다. `arithmetic_check` 로 근거 숫자에서 유도되는지 확인한다.
  다만 **근거 전체를 피연산자 후보로 쓰지 않는다** — 그러면 무작위 숫자의
  50%가 우연히 설명된다(실측). 답변이 스스로 제시한 수치만 후보로 쓴다.
- **연도 분리.** `"2025년"` 의 `2025` 를 근거 없는 숫자로 잡던 오탐을
  없앴다. 연도는 사실 주장이라기보다 기간 지정이라 `period_claims` 로
  따로 모은다(정보를 버리지는 않는다).
- **`"근거" in answer` 폴백 제거.** `ANSWER_SYSTEM_PROMPT` 가 답변에
  `"근거: ..."` 를 쓰라고 지시하므로 이 조건은 사실상 항상 참이었고,
  `has_citation` 이 아무것도 검증하지 못했다. 이제 report_id/chunk_id 가
  실제로 답변에 있어야 한다.
- **근거 0건 침묵 제거.** 근거가 아예 없으면 경고조차 안 나가고 숫자도
  없으니 `numbers_grounded=True` 로 만점 통과하던 경로를 막았다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from disclosure_rag.agent.arithmetic_check import (
    Derivation,
    build_derivation_pool,
    build_pool,
    detect_unit_rescale,
    has_percent_context,
)
from disclosure_rag.agent.evidence import EvidencePack
from disclosure_rag.entity.entity_extractor import ExtractedEntities

_NUMBER_PAT = re.compile(r"\d[\d,]*\.?\d*")
# 괄호 안 환산 표기는 통째로 검증에서 뺀다.
# 예전엔 `(약 …)` 만 뺐는데, 프롬프트 규칙 2번은 "원문 숫자를 먼저 쓰고
# 괄호로 덧붙이세요" 라고만 하지 '약' 을 쓰라고 하지 않는다. 그래서
# **규칙을 정확히 지킨 답변이 검증에서 탈락**했다(2026-08-31 재현:
# "224,787,773,988,054원 (224조 7,877억원)" -> ungrounded {'224','7877'}).
# suite_v1 S001 의 채점 항목에 "단위 표기"가 있어 오히려 권장되는 형태다.
_APPROX_PAREN_PAT = re.compile(r"\((?:약\s*)?[^)]*?(?:조|억|만|천|백만|십억|%|퍼센트)[^)]*\)")
_DOC_ID_PAT = re.compile(r"\b(?:periodic|major|exchange|holding)_\d{10,}\b")
# "2025년" "2025 년" "2025년도" — 기간 지정이지 새로운 수치 주장이 아니다.
_YEAR_CONTEXT_PAT = re.compile(r"\b(19\d{2}|20\d{2})\s*(?=년|년도|연도|분기|회계)")


def _extract_numbers(text: str, *, min_digits: int = 3) -> set[str]:
    """콤마/소수점을 제거한 뒤, 우연한 오검출(1~2자리 숫자 등)을 줄이기 위해
    min_digits 자리 이상만 취급한다.

    회귀 발견(2026-08-16, 회사 일반화 스모크테스트): "7,661,584백만원 (약 7조
    6,615억원)"처럼 답변이 같은 숫자를 조/억 단위로 다시 풀어 쓰면, 괄호 안의
    "6615"가 evidence 원문 문자열과 글자 그대로 일치하지 않아 "근거 없는 숫자"로
    오탐됐다(실제로는 같은 숫자의 재표기일 뿐 새로운 주장이 아님). "(약 ...)"
    괄호는 근사 재표기라는 걸 답변 스스로 명시한 것이므로 grounding 검사에서
    제외한다."""
    text = _APPROX_PAREN_PAT.sub(" ", text)
    out = set()
    for m in _NUMBER_PAT.finditer(text):
        norm = m.group(0).replace(",", "")
        digits_only = norm.replace(".", "")
        if len(digits_only) >= min_digits:
            out.add(norm)
    return out


def _numbers_with_tokens(text: str, *, min_digits: int = 3) -> dict[float, str]:
    """값 -> 원본 표기. 유도 검산에서 사람이 읽을 수 있는 식을 만들기 위해 표기를 남긴다."""
    text = _APPROX_PAREN_PAT.sub(" ", text)
    out: dict[float, str] = {}
    for m in _NUMBER_PAT.finditer(text):
        raw = m.group(0)
        norm = raw.replace(",", "")
        if len(norm.replace(".", "")) < min_digits:
            continue
        try:
            value = float(norm)
        except ValueError:
            continue
        out.setdefault(value, raw)
    return out


def _period_year_tokens(answer: str) -> set[str]:
    return {m.group(1) for m in _YEAR_CONTEXT_PAT.finditer(answer)}


@dataclass
class ValidationResult:
    numbers_grounded: bool
    ungrounded_numbers: set[str]
    has_citation: bool
    correction_evidence_complete: bool | None  # None = 해당 없음(정정 질문 아님)
    warnings: list[str] = field(default_factory=list)
    # 2026-08-30 추가
    derived_numbers: dict[str, str] = field(default_factory=dict)   # 표기 -> 검산식
    rescaled_numbers: dict[str, str] = field(default_factory=dict)  # 표기 -> "근거 x 10^k"
    period_claims: set[str] = field(default_factory=set)            # 답변이 지정한 연도
    has_any_evidence: bool = True

    @property
    def passed(self) -> bool:
        return (
            self.has_any_evidence
            and self.numbers_grounded
            and self.has_citation
            and (self.correction_evidence_complete is not False)
        )


def validate_answer(answer: str, evidence_pack: EvidencePack, entities: ExtractedEntities) -> ValidationResult:
    warnings: list[str] = []

    evidence_text = evidence_pack.prompt_text + "\n" + "\n".join(str(tr) for tr in evidence_pack.tool_results_summary)
    evidence_numbers = _numbers_with_tokens(evidence_text)
    evidence_tokens = _extract_numbers(evidence_text)

    answer_numbers = _numbers_with_tokens(answer)
    answer_tokens = _extract_numbers(answer)

    # 연도는 기간 지정이지 수치 주장이 아니다 — 따로 뺀다.
    period_claims = _period_year_tokens(answer)
    candidates = answer_tokens - evidence_tokens - period_claims

    # 유도 검산 후보 풀 = 답변 ∩ 근거 (근거 전체를 쓰면 오탐 50%)
    pool = build_derivation_pool(answer_numbers, evidence_numbers)
    rescale_pool = build_pool(evidence_numbers)

    derived: dict[str, str] = {}
    rescaled: dict[str, str] = {}
    ungrounded: set[str] = set()

    for token in sorted(candidates):
        try:
            value = float(token)
        except ValueError:
            ungrounded.add(token)
            continue

        derivation: Derivation | None = None
        if len(pool):
            derivation = _verify(value, token, pool, answer)
        if derivation is not None:
            derived[token] = str(derivation)
            continue

        scale = detect_unit_rescale(value, rescale_pool)
        if scale is not None:
            # 통과시키지 않는다. ANSWER_SYSTEM_PROMPT 가 단위 재환산을 금지한다.
            rescaled[token] = scale
            ungrounded.add(token)
            continue

        ungrounded.add(token)

    numbers_grounded = not ungrounded
    if ungrounded:
        warnings.append(
            f"[근거 없는 숫자 의심] 답변에 있지만 근거에도 없고 검산으로도 설명되지 않는 숫자: {sorted(ungrounded)}"
        )
    if rescaled:
        warnings.append(
            f"[단위 재환산 의심] 근거 수치를 10의 거듭제곱으로 바꿔 쓴 것으로 보임: {rescaled}"
        )
    if derived:
        warnings.append(f"[참고] 근거 수치로 검산된 계산값: {derived}")

    # 회귀 발견(2026-08-16): get_correction_history/get_latest_report 만 호출된
    # 답변(search_disclosures 를 안 써서 evidence_pack.citations 가 비어있는
    # 경우)은, 답변이 근거를 정확히 인용했어도 evidence_pack.citations 가
    # 비어있다는 이유만으로 무조건 has_citation=False 로 잡혔다. tool_results_
    # summary(예: get_correction_history 결과)에 등장한 report_id 를 답변이
    # 그대로 인용했는지도 함께 확인한다.
    tool_result_doc_ids = {
        m.group(0) for tr in evidence_pack.tool_results_summary for m in _DOC_ID_PAT.finditer(str(tr))
    }
    has_any_evidence = bool(evidence_pack.citations) or bool(tool_result_doc_ids)
    has_citation = has_any_evidence and (
        any(c.report_id in answer or c.chunk_id in answer for c in evidence_pack.citations)
        or any(doc_id in answer for doc_id in tool_result_doc_ids)
    )
    if not has_any_evidence:
        # 여기서 침묵하면 "근거 0건 + 숫자 0개" 답변이 만점으로 통과한다.
        warnings.append("근거가 하나도 없다 — 검색이 전부 실패했거나 tool 결과가 비었다")
    elif not has_citation:
        warnings.append("답변에 근거(report_id/chunk_id) 인용이 없음")

    correction_evidence_complete: bool | None = None
    if entities.explicit_correction:
        has_correction = any(c.is_correction for c in evidence_pack.citations)
        has_original = any(not c.is_correction for c in evidence_pack.citations)
        correction_evidence_complete = has_correction and has_original
        if not correction_evidence_complete:
            warnings.append("정정 분석 질문인데 원본/정정본 근거가 모두 확보되지 않음 (정정 체인 재검색 필요할 수 있음)")

    return ValidationResult(
        numbers_grounded=numbers_grounded, ungrounded_numbers=ungrounded,
        has_citation=has_citation, correction_evidence_complete=correction_evidence_complete,
        warnings=warnings, derived_numbers=derived, rescaled_numbers=rescaled,
        period_claims=period_claims, has_any_evidence=has_any_evidence,
    )


def _verify(value: float, token: str, pool, answer: str) -> Derivation | None:
    from disclosure_rag.agent.arithmetic_check import verify_derived

    percent = has_percent_context(answer, token)
    got = verify_derived(value, token, pool, percent_context=percent)
    if got is None and percent:
        # "%p" 없이 단위를 생략한 표기도 있어 반대 문맥도 한 번 더 본다.
        got = verify_derived(value, token, pool, percent_context=False)
    return got
