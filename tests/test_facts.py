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


# --------------------------------------------------------------------------- 한국 회계 표기 (2026-09-01)

def test_triangle_marks_are_negative_in_form_disclosures_too():
    """△ 는 정기공시에만 처리되고 서식공시(계약·주요사항)에는 없었다.

    실측: 청크 원문에 "매출액: △59,917 / 비중: △9.4%" 형태로 690건+,
    표에 "[△는 부(-)의 값임]" 범례가 함께 온다.
    """
    assert parse_value("△59,917")[0] == -59917
    assert parse_value("▲1,234")[0] == -1234
    assert parse_value("△9.4%")[0] == -9.4
    assert parse_value("△9.4%")[1] == "%"


def test_percent_point_is_a_different_unit_from_percent():
    """5%에서 7%로 오르면 "2%p 상승" 이지 "2% 상승" 이 아니다.

    실측 164건이 전부 해석 실패였다.
    """
    assert parse_value("1%p") == (1.0, "%p", None)
    assert parse_value("1.0%P") == (1.0, "%p", None)
    assert parse_value("2.5퍼센트포인트") == (2.5, "%p", None)
    # 보통 퍼센트와 섞이면 안 된다
    assert parse_value("1%")[1] == "%"


def test_compound_korean_amounts():
    """"63조 7,454억원" 같은 조·억 혼용. 실측 12,441건."""
    assert parse_value("63조 7,454억원")[0] == 63_745_400_000_000
    assert parse_value("3조 1,659억원")[0] == 3_165_900_000_000
    # 단일 단위는 기존 경로가 그대로 처리한다
    assert parse_value("155억원")[0] == 15_500_000_000


def test_footnote_markers_after_a_number_are_stripped():
    """"69,406주 (주1)", "1,234*" 실측 938건."""
    assert parse_value("69,406주 (주1)")[0] == 69406
    assert parse_value("1,234*")[0] == 1234
    assert parse_value("1,234**")[0] == 1234
    assert parse_value("5,678 주2)")[0] == 5678


def test_fullwidth_digits_are_read():
    assert parse_value("０")[0] == 0
    assert parse_value("１２３")[0] == 123


def test_company_names_with_the_corporation_symbol_are_not_mangled():
    """NFKC 를 통째로 쓰면 `㈜LS` 가 `(주)LS` 로 바뀐다.

    최대주주 이름이 그 값으로 저장되므로 회사 이름이 훼손되면 치명적이다.
    parse_value 는 숫자가 아닌 값을 그대로 두어야 한다.
    """
    assert parse_value("㈜LS") == (None, None, None)


def test_parenthesised_negatives_still_work_in_periodic():
    """이미 되던 것이 깨지지 않았는지. 90,084건이 이 경로로 음수 저장된다."""
    from disclosure_rag.facts.extractor import parse_periodic_value
    assert parse_periodic_value("(4,935,379)")[0] == -4_935_379
    assert parse_periodic_value("(54,702)")[0] == -54_702


# ------------------------------------------------- 회계 괄호 음수 (2026-09-01)
#
# `(4,935,379)` 는 -4,935,379 다. 예전에는 정기공시 경로에서만 처리해
# 서식공시(주요사항·거래소·대량보유)에서는 부호를 통째로 잃었다.

def test_a_parenthesised_number_is_negative_in_every_document_type():
    from disclosure_rag.facts.extractor import parse_value
    assert parse_value("(4,935,379)")[0] == -4_935_379
    assert parse_value("(12.5)")[0] == -12.5


def test_parentheses_that_are_not_a_number_are_left_alone():
    from disclosure_rag.facts.extractor import parse_value
    assert parse_value("(주1)")[0] is None
    assert parse_value("(단위: 백만원)")[0] is None


# ------------------------------------------- 연·월 조회는 그 달로 좁힌다
#
# `2024-05` 를 물었는데 `filing_date LIKE '2024%'` 로 넓히면 1년치가 전부
# 근거로 들어온다(실측: 현대건설 4문서 -> 29문서).

def test_a_year_month_period_does_not_widen_to_the_whole_year(tmp_path):
    from disclosure_rag.facts.extractor import Fact
    from disclosure_rag.facts.store import FactStore

    store = FactStore(str(tmp_path / "facts.sqlite"))
    store.insert_many([
        Fact(doc_id=f"d{i}", chunk_id=f"c{i}", company="가", doc_group="exchange",
             key="계약금액", key_norm="계약금액", value_text="100", value_num=100.0,
             filing_date=filing, period=None, is_latest=True)
        for i, filing in enumerate(("20240515", "20240612", "20241120"))
    ])

    got = {r["filing_date"] for r in store.lookup(company="가", period="2024-05", limit=100)}
    assert got == {"20240515"}
    year = {r["filing_date"] for r in store.lookup(company="가", period="2024", limit=100)}
    assert len(year) == 3
