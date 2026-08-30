"""서술형(open) 답변 채점 — 정답 문장 없이.

## 왜 필요한가

`suite_v1` 38문항 중 **16문항이 정답 문장이 비어 있다.** 전부 open 형이다
("주요 내용을 정리해줘", "무엇이 달라졌는지 설명해줘"). 서술형 정답을 사람이
써 넣는 데 비용이 커서 미뤄졌고, 그 결과 지금까지 **open 답변을 한 번도
채점하지 못했다.** 전체의 42% 가 보이지 않는 상태다.

## 정답 문장 없이 무엇을 잴 수 있나

세 가지다. 셋 다 LLM 을 쓰지 않는다.

    1. 항목 커버리지   그 공시유형의 required 항목이 답변에 값과 함께 있는가
                      (Field Schema 가 데이터에서 뽑아둔 목록)
    2. 근거 정확성     답변이 인용한 공시가 gold 문서인가 / gold 를 다 인용했나
    3. 누락 정직성     확인 못 한 항목을 '(확인되지 않음)'이라고 밝혔는가

3번이 중요하다. 프롬프트가 "확인되지 않은 항목은 '(확인되지 않음)'이라고
적으세요"를 시키는데, **항목을 조용히 빼면** 읽는 사람이 다 확인된 것으로
오해한다. 그래서 '언급 안 함'과 '없다고 밝힘'을 구분해서 센다.

## 합산 점수를 만들지 않는다

세 값을 하나로 뭉치면 무엇이 나빠졌는지 알 수 없다. 각각 따로 보고한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from disclosure_rag.agent.field_schema import normalize_field_key

# 항목명 뒤에 값이 붙었는지 볼 때 훑는 범위. 표 형식 답변("- 투자금액: 1,000원")
# 과 문장 형식("투자금액은 1,000원이며") 을 모두 담을 만큼만.
_VALUE_WINDOW = 60

_NUMBER = re.compile(r"\d")
_UNKNOWN_MARKS = ("확인되지않음", "확인되지않았", "확인불가", "해당사항없음", "기재되지않")
_REPORT_ID = re.compile(r"\b(?:periodic|major|exchange|holding)_\d+")


@dataclass
class OpenScore:
    required: list[str] = field(default_factory=list)
    covered: list[str] = field(default_factory=list)      # 값과 함께 제시됨
    acknowledged: list[str] = field(default_factory=list)  # 없다고 밝힘
    silent: list[str] = field(default_factory=list)        # 조용히 뺌 ★
    cited: list[str] = field(default_factory=list)
    gold_docs: list[str] = field(default_factory=list)

    @property
    def field_coverage(self) -> float | None:
        """required 항목 중 값과 함께 제시된 비율."""
        if not self.required:
            return None
        return len(self.covered) / len(self.required)

    @property
    def silent_omission_rate(self) -> float | None:
        """확인 못 한 걸 밝히지도 않고 빼버린 비율. 낮을수록 좋다."""
        if not self.required:
            return None
        return len(self.silent) / len(self.required)

    @property
    def citation_recall(self) -> float | None:
        """gold 문서 중 답변이 인용한 비율."""
        if not self.gold_docs:
            return None
        return len(set(self.cited) & set(self.gold_docs)) / len(set(self.gold_docs))

    @property
    def citation_precision(self) -> float | None:
        """인용한 것 중 gold 인 비율. 엉뚱한 공시를 근거로 달면 떨어진다."""
        if not self.cited:
            return None
        return len(set(self.cited) & set(self.gold_docs)) / len(set(self.cited))

    def to_dict(self) -> dict:
        return {
            "n_required": len(self.required),
            "n_covered": len(self.covered),
            "n_acknowledged": len(self.acknowledged),
            "n_silent": len(self.silent),
            "field_coverage": _round(self.field_coverage),
            "silent_omission_rate": _round(self.silent_omission_rate),
            "citation_recall": _round(self.citation_recall),
            "citation_precision": _round(self.citation_precision),
            "silent_fields": list(self.silent),
        }


def _round(v: float | None) -> float | None:
    return None if v is None else round(v, 4)


def cited_report_ids(answer: str) -> list[str]:
    """답변이 근거로 든 공시 id. 등장 순서대로, 중복 제거."""
    out: list[str] = []
    for m in _REPORT_ID.finditer(answer or ""):
        if m.group(0) not in out:
            out.append(m.group(0))
    return out


def _normalize_keep_lines(text: str) -> str:
    """줄바꿈만 남기고 공백·가운뎃점을 지운다.

    `normalize_field_key` 를 통째로 쓰면 줄바꿈까지 사라져서 항목의 경계가
    없어진다. 그러면 이런 답변에서

        - 투자금액: 1,200,000,000원
        - 투자기간: (확인되지 않음)

    '투자금액' 뒤 60자 안에 다음 줄의 '(확인되지 않음)'이 들어와, 값이
    멀쩡히 있는 항목을 '확인 못 함'으로 잘못 읽는다(실제로 발생).
    """
    return "\n".join(normalize_field_key(line) for line in (text or "").splitlines())


def _field_status(answer_norm: str, field_name: str, others: list[str]) -> str:
    """covered | acknowledged | silent.

    항목 뒤를 훑되 **다음 항목이 시작되기 전까지만** 본다. 줄바꿈과 다른
    항목명이 경계다.
    """
    key = normalize_field_key(field_name)
    if not key:
        return "silent"
    idx = answer_norm.find(key)
    if idx < 0:
        return "silent"

    start = idx + len(key)
    window = answer_norm[start: start + _VALUE_WINDOW]
    # 경계 ① 줄바꿈
    cut = window.find("\n")
    if cut >= 0:
        window = window[:cut]
    # 경계 ② 다른 요구 항목의 시작
    for other in others:
        o = normalize_field_key(other)
        if not o or o == key:
            continue
        pos = window.find(o)
        if pos >= 0:
            window = window[:pos]

    if any(mark in window for mark in _UNKNOWN_MARKS):
        return "acknowledged"
    if _NUMBER.search(window):
        return "covered"
    # 항목명은 있는데 숫자가 안 붙었다. 서술형 항목(투자목적 등)일 수 있으므로
    # 뒤에 글자가 이어지면 제시된 것으로 본다.
    return "covered" if len(window.strip(" :=-")) >= 4 else "silent"


def score_open_answer(answer: str, *, required_fields, gold_doc_ids=()) -> OpenScore:
    """정답 문장 없이 서술형 답변을 채점한다."""
    answer = answer or ""
    answer_norm = _normalize_keep_lines(answer)
    score = OpenScore(required=[f for f in (required_fields or []) if f],
                      gold_docs=[d for d in (gold_doc_ids or []) if d],
                      cited=cited_report_ids(answer))
    for name in score.required:
        status = _field_status(answer_norm, name, score.required)
        getattr(score, status).append(name)
    return score


__all__ = ["OpenScore", "score_open_answer", "cited_report_ids"]
