"""Stage 1 실험 전용: Fixed-500 / Section-aware(no parent-child) 청킹 변형.

프로덕션 청커(chunking/chunkers.py, "Section-aware+Parent-Child")와 비교하기 위한
ablation 용도로만 쓴다. 실제 파이프라인에는 연결하지 않는다.
"""

from __future__ import annotations

from disclosure_rag.chunking.chunk_schema import ChunkSchema, estimate_tokens, render_kv_node, render_table_node, render_text_node
from disclosure_rag.chunking.chunkers import _base_fields, _content_bearing_sections, _flatten_with_section
from disclosure_rag.common.doc_tree import KeyValueNode, ParsedDocument, TableNode, TextNode
from disclosure_rag.common.manifest_loader import ManifestRow
from disclosure_rag.correction.correction_graph_builder import CorrectionRecord


def _render_node_flat(node) -> str:
    if isinstance(node, TextNode):
        return render_text_node(node)
    if isinstance(node, KeyValueNode):
        return render_kv_node(node)
    if isinstance(node, TableNode):
        return "\n\n".join(render_table_node(node, max_rows_per_chunk=10**9, max_tokens_per_chunk=10**9))
    return ""


def chunk_fixed_window(
    parsed: ParsedDocument, row: ManifestRow, correction: CorrectionRecord, *, window_tokens: int = 500,
) -> list[ChunkSchema]:
    """구조를 완전히 무시하고 문서 전체를 이어붙인 뒤 고정 크기(문자 기준)로 자른다.
    §74 baseline("Fixed-size")과 동일한 취지: overlap 없음, section 경계 무시."""
    base = _base_fields(row, correction, parsed.document_name)
    pieces = [_render_node_flat(n) for _s, n in _flatten_with_section(parsed.sections)]
    full_text = "\n\n".join(p for p in pieces if p.strip())
    if not full_text.strip():
        return []

    window_chars = window_tokens * 2  # chunk_schema.TOKEN_CHARS_PER_TOKEN 과 동일한 rough heuristic
    chunks: list[ChunkSchema] = []
    for i, start in enumerate(range(0, len(full_text), window_chars)):
        piece = full_text[start:start + window_chars].strip()
        if not piece:
            continue
        chunks.append(ChunkSchema(
            chunk_id=f"{row.doc_id}::{parsed.report_subtype}::FIX{i}",
            parent_chunk_id=None,
            raw_text=piece,
            text=piece,  # 구조 무시 baseline 이므로 [회사]/[Section] 컨텍스트 헤더도 붙이지 않음
            section_path=[],
            content_type="text",
            source_path=parsed.source_path,
            **base,
        ))
    return chunks


def chunk_section_only(parsed: ParsedDocument, row: ManifestRow, correction: CorrectionRecord) -> list[ChunkSchema]:
    """Section 경계로는 나누되, Parent-Child 이원화 없이 "section 하나 = chunk 하나"
    로만 처리한다 (큰 section 도 쪼개지 않음 — Parent-Child 대비 "표현력은 있지만
    세밀한 검색 단위가 없는" baseline)."""
    from disclosure_rag.chunking.chunk_schema import render_search_text

    base = _base_fields(row, correction, parsed.document_name)
    chunks: list[ChunkSchema] = []

    for section, direct_content in _content_bearing_sections(parsed.sections):
        pieces = [_render_node_flat(n) for n in direct_content]
        body = "\n\n".join(p for p in pieces if p.strip())
        if not body.strip():
            continue
        chunk_id = f"{row.doc_id}::{parsed.report_subtype}::SEC{len(chunks)}"
        chunks.append(ChunkSchema(
            chunk_id=chunk_id,
            parent_chunk_id=None,
            raw_text=body,
            text=render_search_text(
                company=base["company"], report_name=base["report_name"],
                period=base["period"], section_path=section.path, body_text=body,
            ),
            section_path=section.path,
            content_type="text",
            source_path=parsed.source_path,
            **base,
        ))
    return chunks
