"""Phase 5+6 회귀 테스트: Chunker + 공통 Chunk Schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from disclosure_rag.chunking.chunk_schema import ChunkSchema
from disclosure_rag.common.manifest_loader import load_manifest
from disclosure_rag.pipeline import build_all_chunks

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus"
pytestmark = pytest.mark.skipif(not CORPUS_ROOT.is_dir(), reason="corpus/ 없음")


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(CORPUS_ROOT)


def _chunks_for(manifest, doc_ids):
    rows = [r for r in manifest if r.doc_id in doc_ids]
    return build_all_chunks(str(CORPUS_ROOT), rows=rows, validate=False)


def test_periodic_parent_child_structure(manifest):
    """§13: periodic 은 Parent(section) + Child(작은 조각) 구조여야 하고,
    Parent 자체는 다른 chunk 의 parent_chunk_id 로만 참조되어야 한다."""
    chunks = _chunks_for(manifest, {"periodic_20240312000736"})
    main_chunks = [c for c in chunks if c.report_id == "periodic_20240312000736"]
    assert len(main_chunks) > 10

    parent_ids = {c.chunk_id for c in main_chunks if c.parent_chunk_id is None}
    referenced_as_parent = {c.parent_chunk_id for c in main_chunks if c.parent_chunk_id}
    assert referenced_as_parent, "Parent-Child 구조가 전혀 없음"
    assert referenced_as_parent <= parent_ids

    child = next(c for c in main_chunks if c.parent_chunk_id is not None)
    parent = next(c for c in main_chunks if c.chunk_id == child.parent_chunk_id)
    assert child.section_path == parent.section_path
    assert child.report_type == "periodic"
    assert child.company == "삼성전자"
    assert child.period == "2023-12"


def test_major_flat_no_parent(manifest):
    """§15: major 는 Parent-Child 를 쓰지 않는다 (전부 top-level)."""
    chunks = _chunks_for(manifest, {"major_20241118000171"})
    assert chunks
    assert all(c.parent_chunk_id is None for c in chunks)
    assert all(c.report_type == "major" for c in chunks)


def test_exchange_whole_doc_single_chunk(manifest):
    """§18: 짧은 exchange 문서는 공시 1건 = chunk 1개."""
    chunks = _chunks_for(manifest, {"exchange_20250728800035"})
    assert len(chunks) == 1
    c = chunks[0]
    assert c.content_type == "key_value"
    assert "22,764,764,160,000" in c.raw_text


def test_holding_deep_section_path_preserved(manifest):
    chunks = _chunks_for(manifest, {"holding_20241025000530"})
    assert chunks
    assert all(c.report_type == "holding" for c in chunks)
    assert any(len(c.section_path) >= 2 for c in chunks)


def test_field_codes_not_dropped_in_table_or_kv_chunks(manifest):
    """§14 금지: TE[ACODE]/TU[AUNIT] 값이 KeyValueNode 든 TableNode(큰 표) 든
    field_codes 로 보존돼야 한다 (회귀: 이전에는 TableNode 경로에서 누락됐었음)."""
    chunks = _chunks_for(manifest, {"major_20241118000171"})
    refs = [r for c in chunks for r in c.field_codes]
    assert refs, "field_codes 가 하나도 보존되지 않음 (회귀)"
    assert any(r.code for r in refs), "ACODE 가 하나도 없음"
    # KIM 브랜치: 같은 셀 텍스트끼리 덮어쓰던 dict 를 위치 보존 리스트로 바꿨다.
    # 위치가 있어야 나중에 facts 층에서 열을 되짚을 수 있다.
    assert any(r.row is not None for r in refs), "위치 정보가 없음"


def test_no_duplicated_text_from_colspan_expansion(manifest):
    """회귀: colspan 이 3인 값 셀이 RLE 로 축약되지 않아 같은 값이 3번 반복 출력되던 버그."""
    chunks = _chunks_for(manifest, {"major_20241118000171"})
    joined = "\n".join(c.raw_text for c in chunks)
    assert "50,144,628 | 50,144,628 | 50,144,628" not in joined


def test_search_text_has_context_header(manifest):
    """§26: 검색용 text 는 [회사]/[공시]/[Section]/[내용] 컨텍스트를 포함해야 한다."""
    chunks = _chunks_for(manifest, {"periodic_20240312000736"})
    c = chunks[0]
    assert "[회사]" in c.text
    assert "[공시]" in c.text
    assert "[Section]" in c.text
    assert "[내용]" in c.text
    assert c.raw_text not in c.text or "[내용]" in c.text  # text 는 raw_text 를 감싼 형태


def test_toc_table_excluded_from_chunks(manifest):
    """회귀(2026-08-15, 대회 참고 질의 스모크테스트에서 발견): periodic 문서의
    목차(TOC) 페이지가 표로 파싱되어 그대로 인덱싱되면 [제목 | 대시 필러 |
    페이지번호] 텍스트가 실제 section 제목과 키워드가 겹쳐 검색을 오염시킨다
    (진짜 본문 대신 목차 chunk 가 top-k 를 차지해 "핵심 사업" 질의가 실패로
    이어짐 — 실제 재현됨). 목차 표는 chunk 후보에서 제외돼야 하고, 원문의
    모든 섹션 제목은 실제 section 으로도 존재하므로 정보 손실은 없다."""
    chunks = _chunks_for(manifest, {"periodic_20240312000736"})
    toc_like = [c for c in chunks if c.section_path == ["사업보고서"]]
    assert not toc_like, f"목차 표가 여전히 chunk 로 남아있음: {[c.chunk_id for c in toc_like]}"
    # 진짜 본문(예: 사업의 내용)은 그대로 살아있어야 한다 — 과도하게 걸러내지 않았는지 확인.
    assert any("사업의 내용" in "".join(c.section_path) for c in chunks)


def test_toc_detector_does_not_flag_real_data_table(manifest):
    """회귀: 초기 구현은 셀 텍스트가 "-" 하나만 있어도(재무표의 결측/0 표기 관행)
    목차로 오탐했다(실측: 내부통제담당인력 현황표, periodic_20250319000665).
    대시 필러는 목차 특유의 긴 리더 점선(예: 35자)이어야만 걸러야 한다."""
    chunks = _chunks_for(manifest, {"periodic_20250319000665"})
    control_chunks = [c for c in chunks if "내부통제에 관한 사항" in "".join(c.section_path)]
    assert control_chunks, "내부통제 관련 chunk 가 통째로 사라짐(오탐으로 필터링됐을 가능성)"


def test_all_chunks_are_valid_schema(manifest):
    """무작위 표본에 대해 스키마 필수 필드가 채워지는지 확인."""
    import random

    sample_rows = random.Random(7).sample(manifest, 20)
    chunks = build_all_chunks(str(CORPUS_ROOT), rows=sample_rows, validate=False)
    assert chunks
    for c in chunks:
        assert isinstance(c, ChunkSchema)
        assert c.chunk_id
        assert c.report_id
        assert c.text
        assert c.content_type in ("text", "table", "key_value")
