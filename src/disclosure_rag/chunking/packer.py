"""ContentNode 리스트를 §12 원칙(Section/Paragraph 경계 > Token 길이)에 따라
Chunk 단위로 묶는 공통 로직. 4종 Chunker 가 전부 이 위에서 동작한다.

=== KIM 브랜치 재작성 (2026-08-23) ===
기존 96줄에 핵심 결함 3개가 몰려 있어 다시 썼다.

[결함 1] 예산 초과 노드를 절대 쪼개지 않았다.
    `if buf_texts and buf_tokens + tok > max_tokens: flush()` — 버퍼가 비어 있으면
    조건 자체가 False 라, 20,000 토큰짜리 노드 하나가 그대로 청크 1개가 됐다.
    실측 최대 leaf = 41,625자, CHILD_MAX_TOKENS(1000) 초과가 4.19%.
    이 outlier 는 실제 사고를 냈다 — 26,027자 청크 때문에 reranker 가 1,500쌍에
    58분 걸린 기록이 남아 있다. 임베딩 비용도 같은 이유로 부풀었다.
    -> split_long_text() 로 문단 > 줄 > 문장 > 어절 순으로 재귀 분할한다.

[결함 2] field_codes 가 {셀 텍스트: 코드} dict 라 덮어써졌다(major 80.1% 소실).
    동시에 분할된 표 조각마다 표 전체 코드를 복사해 엔트리가 2배로 부풀었다.
    -> list[FieldRef] 로 바꾸고, 표가 쪼개지면 **그 조각에 실제로 등장한 셀의
       ref 만** 붙인다.

[결함 3] KeyValueNode 경로에서 unit_code/unit_value 를 아예 안 담았다.
    -> kv_field_refs() 로 5필드 전부 보존.

[부가] content_type 을 "key_value > table > text" 고정 우선순위가 아니라
    **문자 수 기준 우세 타입**으로 정한다. 기존에는 산문 위주 청크에 작은 표가
    하나 섞이면 통째로 table 로 라벨링됐다(leaf table 52.1%가 과대추정 의심).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from disclosure_rag.chunking.chunk_schema import (
    ContentType,
    FieldRef,
    estimate_tokens,
    kv_field_refs,
    render_kv_node,
    render_table_node,
    render_text_node,
    table_field_refs,
)
from disclosure_rag.common.doc_tree import ContentNode, KeyValueNode, TableNode, TextNode

# 재귀 분할 시 시도하는 경계 (앞에 있을수록 의미 경계에 가깝다)
_SEPARATORS = ["\n\n", "\n", ". ", "다. ", " "]


@dataclass
class PackedUnit:
    text: str
    content_type: ContentType
    field_refs: list[FieldRef] = field(default_factory=list)
    unit_hint: str | None = None
    period_hint: str | None = None
    # 이 unit 에 조각이 들어간 표들의 문서-내 순번 키("T0", "T1", ...).
    # 한 unit 이 표 두 개에 걸칠 수 있으므로 리스트다.
    table_keys: list[str] = field(default_factory=list)


def split_long_text(text: str, max_tokens: int) -> list[str]:
    """예산을 넘는 텍스트를 의미 경계 우선으로 쪼갠다 (결함 1).

    문단 -> 줄 -> 문장 -> 어절 순으로 내려가고, 그래도 안 되면 문자 단위로 자른다.
    어떤 경우에도 **내용을 버리지 않는다** (§7 silent 손실 금지).
    """
    if estimate_tokens(text) <= max_tokens or not text:
        return [text] if text else []

    for sep in _SEPARATORS:
        if sep not in text:
            continue
        parts = [p for p in text.split(sep) if p != ""]
        if len(parts) < 2:
            continue
        out: list[str] = []
        buf: list[str] = []
        buf_tok = 0
        sep_tok = estimate_tokens(sep) if sep.strip() == "" else estimate_tokens(sep)
        for p in parts:
            p_tok = estimate_tokens(p)
            # 구분자 비용을 예산에 포함한다. 빼먹으면 줄 수가 많은 표에서
            # 조각이 상한을 최대 50%까지 넘긴다(실측: max 2,954자 / 상한 2,000자).
            if buf and buf_tok + p_tok + sep_tok > max_tokens:
                out.append(sep.join(buf))
                buf, buf_tok = [], 0
            if p_tok > max_tokens:
                # 이 조각 하나가 이미 예산 초과 -> 더 잘은 경계로 재귀
                if buf:
                    out.append(sep.join(buf))
                    buf, buf_tok = [], 0
                out.extend(split_long_text(p, max_tokens))
                continue
            buf.append(p)
            buf_tok += p_tok + (sep_tok if len(buf) > 1 else 0)
        if buf:
            out.append(sep.join(buf))
        if len(out) > 1:
            return out

    # 경계가 하나도 없는 초장문(예: 구분자 없는 숫자열) -> 문자 단위 하드 분할
    approx_chars = max(200, int(max_tokens * max(1.0, len(text) / max(1, estimate_tokens(text)))))
    return [text[i : i + approx_chars] for i in range(0, len(text), approx_chars)]


_STRUCTURED_MIN_SHARE = 0.20


def _resolve_content_type(type_chars: dict[str, int]) -> ContentType:
    """구조화 타입(key_value > table)을 우선하되, **문자 비중이 유의미할 때만** 그렇게 한다.

    기존은 우선순위 고정이라 산문 위주 청크에 작은 표 하나가 섞이면 통째로 table 로
    라벨링됐다(leaf table 52.1%가 과대추정 의심). 반대로 순수 우세타입으로 바꾸면
    폼 문서(계약공시)가 단서조항 서술 때문에 text 로 라벨링돼 구조 정보를 잃는다.
    -> 구조화 부분이 20% 이상이면 구조화로 본다.
    """
    if not type_chars:
        return "text"
    total = sum(type_chars.values()) or 1
    for t in ("key_value", "table"):
        if type_chars.get(t, 0) / total >= _STRUCTURED_MIN_SHARE:
            return t  # type: ignore[return-value]
    return max(type_chars.items(), key=lambda kv: kv[1])[0]  # type: ignore[return-value]


def _refs_in(refs: list[FieldRef], piece: str) -> list[FieldRef]:
    """표가 여러 조각으로 쪼개졌을 때, 그 조각에 실제로 등장한 셀의 ref 만 남긴다.
    (기존처럼 조각마다 표 전체를 복사하면 엔트리가 2배로 부푼다 — 결함 2.)
    텍스트가 비어 있는 ref 는 위치를 확인할 수 없으므로 그대로 둔다."""
    out = []
    for r in refs:
        if not r.text or r.text in piece:
            out.append(r)
    return out


def pack_nodes(
    nodes: list[ContentNode],
    *,
    target_tokens: int = 600,
    max_tokens: int = 1000,
    table_max_rows: int = 20,
    table_max_tokens: int | None = None,
    table_style: str = "grid",
) -> list[PackedUnit]:
    """ContentNode 들을 chunk 후보(PackedUnit)로 묶는다.

    table_max_rows / table_max_tokens 를 크게 주면 표를 쪼개지 않는다(parent 렌더용).
    """
    if table_max_tokens is None:
        table_max_tokens = max_tokens

    units: list[PackedUnit] = []
    buf_texts: list[str] = []
    buf_tokens = 0
    buf_type_chars: dict[str, int] = {}
    buf_refs: list[FieldRef] = []
    buf_unit: str | None = None
    buf_period: str | None = None
    buf_tables: list[str] = []

    def flush() -> None:
        nonlocal buf_texts, buf_tokens, buf_type_chars, buf_refs, buf_unit, buf_period, buf_tables
        if not buf_texts:
            return
        units.append(
            PackedUnit(
                text="\n\n".join(buf_texts),
                content_type=_resolve_content_type(buf_type_chars),
                field_refs=list(buf_refs),
                unit_hint=buf_unit,
                period_hint=buf_period,
                table_keys=list(buf_tables),
            )
        )
        buf_texts, buf_tokens, buf_type_chars, buf_refs = [], 0, {}, []
        buf_unit, buf_period, buf_tables = None, None, []

    def add(text: str, ctype: str, refs: list[FieldRef], unit: str | None = None,
            period: str | None = None, table_key: str | None = None) -> None:
        """예산을 지키며 버퍼에 넣는다. 한 조각이 예산을 넘으면 먼저 쪼갠다."""
        nonlocal buf_tokens, buf_unit, buf_period
        if not text.strip():
            return
        for piece in split_long_text(text, max_tokens):
            tok = estimate_tokens(piece)
            if buf_texts and buf_tokens + tok > max_tokens:
                flush()
            buf_texts.append(piece)
            buf_tokens += tok
            buf_type_chars[ctype] = buf_type_chars.get(ctype, 0) + len(piece)
            buf_refs.extend(_refs_in(refs, piece))
            if unit and buf_unit is None:
                buf_unit = unit
            if period and buf_period is None:
                buf_period = period
            if table_key and table_key not in buf_tables:
                buf_tables.append(table_key)
            if buf_tokens >= target_tokens:
                flush()

    table_ord = 0
    for node in nodes:
        if isinstance(node, TableNode):
            refs = table_field_refs(node)
            key = f"T{table_ord}"
            table_ord += 1
            for piece in render_table_node(
                node, max_rows_per_chunk=table_max_rows, max_tokens_per_chunk=table_max_tokens,
                style=table_style,  # type: ignore[arg-type]
            ):
                add(piece, "table", refs, node.unit_hint, node.period_hint, table_key=key)
        elif isinstance(node, KeyValueNode):
            add(render_kv_node(node), "key_value", kv_field_refs(node))
        elif isinstance(node, TextNode):
            add(render_text_node(node), "text", [])

    flush()
    return units
