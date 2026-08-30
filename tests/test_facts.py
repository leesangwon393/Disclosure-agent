"""facts 층 계약 테스트 — 숫자 조회가 검색보다 정확해야 하는 이유를 코드로 고정한다."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pytest

from disclosure_rag.chunking.chunkers import chunk_document
from disclosure_rag.common.manifest_loader import load_manifest
from disclosure_rag.common.unicode_utils import PathResolver
from disclosure_rag.correction.correction_graph_builder import build_correction_index
from disclosure_rag.facts.extractor import (extract_facts, is_meaningful_key,
                                            link_facts_to_chunks, normalize_key, parse_value)
from disclosure_rag.parsing.document_detector import parse_documents_for_row

CORPUS = os.environ.get("CORPUS_ROOT", "corpus")
# 삼성전자 반도체 위탁생산 공급계약 — 계약금액 22,764,764,160,000원
SAMPLE_DOC = "exchange_20250728800035"


@lru_cache(maxsize=1)
def _facts_and_chunks():
    if not Path(CORPUS).exists():
        pytest.skip(f"코퍼스 없음: {CORPUS}")
    manifest = load_manifest(CORPUS)
    resolver = PathResolver(CORPUS)
    corrections = build_correction_index(manifest, resolver)
    row = next(r for r in manifest if r.doc_id == SAMPLE_DOC)
    facts, chunks = [], []
    for parsed in parse_documents_for_row(row, resolver):
        if parsed.report_subtype == "unsupported_pdf_html":
            continue
        facts.extend(extract_facts(parsed, row, corrections[row.doc_id]))
        chunks.extend(chunk_document(parsed, row, corrections[row.doc_id]))
    return facts, chunks


# ------------------------------------------------------------------ 정규화
def test_key_normalization_merges_spacing_variants():
    """한글 서식은 자간을 띄우는 경우가 많다("최근 매출액" vs "최근매출액").
    공백을 제거하지 않으면 같은 항목이 둘로 갈린다(실측으로 확인된 현상)."""
    assert normalize_key("2. 계약금액(원)") == ("계약금액", "원")
    assert normalize_key("매출액대비(%)") == ("매출액대비", "%")
    assert normalize_key("최근 매출액")[0] == normalize_key("최근매출액")[0]


def test_value_parsing_keeps_original_and_adds_number():
    """value_text(근거 표시용)와 value_num(범위 질의용)을 둘 다 가져야 한다."""
    num, unit, date = parse_value("22,764,764,160,000", key_unit="원")
    assert num == 22764764160000.0 and unit == "원" and date is None
    num, _u, _d = parse_value("7.6", key_unit="%")
    assert num == 7.6
    _n, _u, date = parse_value("2025-07-24")
    assert date == "20250724"
    # 해석 불가는 지어내지 않는다
    assert parse_value("경영상 비밀유지")[0] is None


def test_boilerplate_keys_are_filtered():
    """서식 머리말·서명란은 사실이 아니다. 안 거르면 상위 항목을 이것들이 차지한다."""
    assert not is_meaningful_key("금융위원회 / 한국거래소 귀중", "금융위원회/한국거래소귀중")
    assert not is_meaningful_key("회 사 명 :", "회사명")
    assert not is_meaningful_key("104-81-26688", "104-81-26688")   # 값이 항목명 자리에 온 것
    assert is_meaningful_key("2. 계약금액(원)", "계약금액")


# ------------------------------------------------------------------ 실데이터
def test_contract_amount_is_extracted_with_number():
    facts, _ = _facts_and_chunks()
    hit = [f for f in facts if f.key_norm == "계약금액"]
    assert hit, "계약금액이 추출되지 않았다"
    f = hit[0]
    assert f.value_text == "22,764,764,160,000"
    assert f.value_num == 22764764160000.0, "숫자 파싱 실패 — 범위 질의가 불가능해진다"


def test_every_fact_links_to_a_source_chunk():
    """근거 표시(대회 평가 항목)의 전제. 출처를 못 찾으면 그 fact 는 쓸 수 없다."""
    facts, chunks = _facts_and_chunks()
    linked = link_facts_to_chunks(facts, chunks)
    assert facts, "fact 가 하나도 없다"
    assert linked / len(facts) >= 0.9, f"출처 연결률 {linked}/{len(facts)} — 근거를 못 붙인다"
    for f in facts:
        if f.chunk_id:
            src = next(c for c in chunks if c.chunk_id == f.chunk_id)
            assert f.value_text in src.raw_text, "연결된 조각에 값이 실제로 없다"
