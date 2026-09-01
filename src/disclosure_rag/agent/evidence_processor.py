"""Stage 9: 근거 구조화 (Evidence Processing).

## 왜 충분성 검사보다 앞에 오나

"근거가 충분한가"는 **근거를 정리해봐야 알 수 있다.**

    정정 diff 질문에서 최초본·최종본을 둘 다 찾았다  -> 근거 2건, 충분해 보인다
    그런데 두 문서에서 비교할 필드가 서로 안 맞는다  -> 실제로는 답을 못 만든다

    계산 질문에서 근거를 10건 찾았다                -> 충분해 보인다
    그런데 피연산자 둘 중 하나가 숫자로 안 잡힌다     -> 계산이 안 된다

건수만 세면 둘 다 통과한다. 그래서 먼저 **필드 단위로 쪼개서 값을 뽑고**,
그 결과를 Stage 10 이 판정한다.

## 어디서 값을 뽑나 — 두 경로

    ① field_codes  청크에 붙은 (key, text, unit) 구조. 표에서 파싱된 것이라
                   신뢰도가 높다. 이걸 우선한다.
    ② raw_text     ①에서 못 찾았을 때만. 항목명이 본문에 등장하는지만 보고
                   값은 그 뒤의 숫자를 취한다. `source="text"` 로 표시해
                   나중에 구분할 수 있게 한다.

②를 아예 빼지 않는 이유: 서술형 문단에 답이 있는 경우가 있다. 다만 ①보다
약하므로 출처를 남긴다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from disclosure_rag.agent.field_schema import normalize_field_key
from disclosure_rag.agent.query_plan import QueryPlan

# 숫자 하나. 콤마·소수점·음수(△ 포함, DART 표기) 를 받는다.
_NUMBER = re.compile(r"[△▲-]?\d[\d,]*(?:\.\d+)?")


def _attr(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _to_number(text: str) -> float | None:
    m = _NUMBER.search(text or "")
    if not m:
        return None
    raw = m.group(0).replace(",", "")
    sign = -1.0 if raw[0] in "△▲-" else 1.0
    try:
        return sign * float(raw.lstrip("△▲-"))
    except ValueError:
        return None


@dataclass(frozen=True)
class FieldHit:
    """근거에서 찾아낸 '항목 = 값' 하나. 출처를 끝까지 들고 다닌다."""
    field: str
    value_text: str
    value_num: float | None
    unit: str | None
    report_id: str
    chunk_id: str
    source: str                       # "field_code" | "text"
    correction_order: int = 0
    is_latest: bool | None = None


@dataclass
class VersionPair:
    """정정 전후 한 항목의 짝. 둘 중 하나가 없으면 `complete` 가 False."""
    field: str
    first: FieldHit | None = None
    final: FieldHit | None = None

    @property
    def complete(self) -> bool:
        return self.first is not None and self.final is not None

    @property
    def changed(self) -> bool | None:
        if not self.complete:
            return None
        return normalize_field_key(self.first.value_text) != normalize_field_key(self.final.value_text)


@dataclass
class ProcessedEvidence:
    hits: list[FieldHit] = field(default_factory=list)
    by_field: dict[str, list[FieldHit]] = field(default_factory=dict)
    pairs: list[VersionPair] = field(default_factory=list)
    operands: list[FieldHit] = field(default_factory=list)
    documents: set[str] = field(default_factory=set)

    @property
    def found_fields(self) -> list[str]:
        return sorted(self.by_field)

    def missing(self, expected: Iterable[str], also_have: frozenset[str] = frozenset()) -> list[str]:
        """`also_have`: 정형(Facts) 채널이 이미 찾아준 항목(정규화된 key).

        `by_field`는 비정형 검색으로 물어온 청크의 field_codes/텍스트에서만
        채워진다(아래 `process_evidence` 참조) — Facts SQL 조회 결과는 안
        본다. 그래서 Facts 가 정답을 정확히 찾아도 비정형 검색이 우연히 같은
        내용을 안 물어오면 "확인 안 됨"으로 판정되어 재검색 루프에 빠졌다
        (2026-09-01 실측: SK텔레콤 리스부채 — facts_rows 에 정답이 있는데도
        missing 처리됨). 호출부(sufficiency.check_sufficiency)가 Facts 결과를
        `also_have` 로 넘겨 이 단절을 메운다.
        """
        have = {normalize_field_key(f) for f in self.by_field} | also_have
        return [f for f in expected if normalize_field_key(f) and normalize_field_key(f) not in have]

    @property
    def incomplete_pairs(self) -> list[VersionPair]:
        """한쪽 버전만 있는 항목. 정정 diff 질문에서 이게 있으면 답이 반쪽이다."""
        return [p for p in self.pairs if not p.complete]


# ---------------------------------------------------------------- 추출

def _hits_from_field_codes(chunk: Any, wanted: dict[str, str]) -> list[FieldHit]:
    out: list[FieldHit] = []
    for ref in (_attr(chunk, "field_codes") or []):
        key = normalize_field_key(_attr(ref, "key"))
        if key not in wanted:
            continue
        text = (_attr(ref, "text") or "").strip()
        if not text:
            continue
        out.append(FieldHit(
            field=wanted[key], value_text=text, value_num=_to_number(text),
            unit=_attr(ref, "unit"),
            report_id=str(_attr(chunk, "report_id") or ""),
            chunk_id=str(_attr(chunk, "chunk_id") or ""),
            source="field_code",
            correction_order=int(_attr(chunk, "correction_order") or 0),
            is_latest=_attr(chunk, "is_latest"),
        ))
    return out


def _hits_from_text(chunk: Any, wanted: dict[str, str], already: set[str]) -> list[FieldHit]:
    """구조에서 못 찾은 항목만 본문에서 찾는다."""
    raw = _attr(chunk, "raw_text") or _attr(chunk, "text") or ""
    flat = normalize_field_key(raw)
    out: list[FieldHit] = []
    for key, display in wanted.items():
        if display in already or key not in flat:
            continue
        idx = flat.find(key) + len(key)
        value = _to_number(flat[idx: idx + 40])
        if value is None:
            continue
        out.append(FieldHit(
            field=display, value_text=flat[idx: idx + 40].strip()[:40], value_num=value,
            unit=None,
            report_id=str(_attr(chunk, "report_id") or ""),
            chunk_id=str(_attr(chunk, "chunk_id") or ""),
            source="text",
            correction_order=int(_attr(chunk, "correction_order") or 0),
            is_latest=_attr(chunk, "is_latest"),
        ))
    return out


def process_evidence(
    plan: QueryPlan, evidence: Sequence[Any] | Sequence[tuple[Any, float]],
) -> ProcessedEvidence:
    """근거 청크를 항목 단위로 쪼갠다. `(chunk, score)` 목록도 받는다."""
    chunks = [e[0] if isinstance(e, tuple) else e for e in evidence]
    wanted = {normalize_field_key(f): f for f in plan.expected_fields if normalize_field_key(f)}

    out = ProcessedEvidence()
    if not wanted:
        out.documents = {str(_attr(c, "report_id") or "") for c in chunks} - {""}
        return out

    structured: list[FieldHit] = []
    for c in chunks:
        structured += _hits_from_field_codes(c, wanted)
    found_display = {h.field for h in structured}

    textual: list[FieldHit] = []
    for c in chunks:
        textual += _hits_from_text(c, wanted, found_display)

    out.hits = structured + textual
    out.documents = {h.report_id for h in out.hits if h.report_id}
    out.documents |= {str(_attr(c, "report_id") or "") for c in chunks} - {""}

    for h in out.hits:
        out.by_field.setdefault(h.field, []).append(h)

    if plan.task == "correction_diff":
        out.pairs = _build_pairs(out.by_field)
    if plan.task in ("calculate", "compare"):
        out.operands = [h for h in out.hits if h.value_num is not None]
    return out


def _build_pairs(by_field: dict[str, list[FieldHit]]) -> list[VersionPair]:
    """항목마다 최초본(가장 작은 correction_order)과 최종본(가장 큰 것)을 짝짓는다.

    한 버전만 있으면 짝이 미완성으로 남는다 — Stage 10 이 그걸 보고 재검색한다.
    """
    pairs: list[VersionPair] = []
    for name, hits in by_field.items():
        orders = sorted({h.correction_order for h in hits})
        if len(orders) < 2:
            only = hits[0]
            pairs.append(VersionPair(field=name,
                                     first=only if only.correction_order == min(orders) else None,
                                     final=only if only.correction_order == max(orders) else None))
            # 버전이 하나뿐이면 first/final 이 같은 객체를 가리키므로 한쪽을 비운다
            if pairs[-1].first is not None and pairs[-1].final is not None:
                pairs[-1] = VersionPair(field=name, first=only, final=None)
            continue
        lo, hi = orders[0], orders[-1]
        pairs.append(VersionPair(
            field=name,
            first=next(h for h in hits if h.correction_order == lo),
            final=next(h for h in hits if h.correction_order == hi),
        ))
    return pairs


__all__ = ["FieldHit", "VersionPair", "ProcessedEvidence", "process_evidence"]
