"""Phase 4 회귀 테스트: 4종 공시 Parser.

실제 코퍼스의 특정 문서(Phase 0 에서 수동 분석한 샘플들)를 대상으로,
section 계층 / TE·TU field_code 보존 / key-value 관계 보존이 깨지지 않는지 고정한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from disclosure_rag.common.doc_tree import TableNode, KeyValueNode, SectionNode, TableNode
from disclosure_rag.common.manifest_loader import load_manifest
from disclosure_rag.common.unicode_utils import PathResolver
from disclosure_rag.parsing.document_detector import parse_documents_for_row

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus"
pytestmark = pytest.mark.skipif(not CORPUS_ROOT.is_dir(), reason="corpus/ 없음")


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(CORPUS_ROOT)


@pytest.fixture(scope="module")
def resolver():
    return PathResolver(CORPUS_ROOT)


def _row(manifest, doc_id):
    return next(r for r in manifest if r.doc_id == doc_id)


def _all_children(section: SectionNode):
    for c in section.children:
        yield c
        if isinstance(c, SectionNode):
            yield from _all_children(c)


def test_periodic_annual_produces_main_plus_two_audit_attachments(manifest, resolver):
    row = _row(manifest, "periodic_20240312000736")  # 삼성전자 2023 사업보고서 (§3, §11 샘플)
    docs = parse_documents_for_row(row, resolver)
    subtypes = {d.report_subtype for d in docs}
    assert subtypes == {"main", "separate_audit_report", "consolidated_audit_report"}
    main = next(d for d in docs if d.report_subtype == "main")
    assert main.document_name == "사업보고서"
    top_titles = [s.title for s in main.sections]
    # §3 실측 목차와 대조
    assert any(t.startswith("I. 회사의 개요") for t in top_titles)
    assert any(t.startswith("II. 사업의 내용") for t in top_titles)
    assert any(t.startswith("III. 재무에 관한 사항") for t in top_titles)


def test_periodic_section3_depth_preserved(manifest, resolver):
    """§3: 'III. 재무 > 7. 자금조달 > 7-1/7-2' 처럼 SECTION-3 까지 존재해야 한다."""
    row = _row(manifest, "periodic_20240312000736")
    docs = parse_documents_for_row(row, resolver)
    main = next(d for d in docs if d.report_subtype == "main")
    fin = next(s for s in main.sections if s.title.startswith("III. 재무"))
    sub = next(s for s in fin.children if isinstance(s, SectionNode) and "자금조달" in s.title and s.level == 2)
    level3_titles = [c.title for c in sub.children if isinstance(c, SectionNode)]
    assert any("7-1" in t for t in level3_titles)


def test_major_extraction_table_preserves_acode(manifest, resolver):
    """§4, §14: TABLE-GROUP[ACLASS] 안의 TE[ACODE]/TU[AUNIT] 값이 버려지지 않아야 한다."""
    row = _row(manifest, "major_20241118000171")  # 삼성전자 자기주식취득 정정
    docs = parse_documents_for_row(row, resolver)
    main = docs[0]
    all_kv = [n for s in main.sections for n in _all_children(s) if isinstance(n, KeyValueNode)]
    all_tables = [n for s in main.sections for n in _all_children(s) if isinstance(n, TableNode)]
    cells = [c for t in all_tables for row in t.rows for c in row]
    # 이 문서는 AUNIT/AUNITVALUE 가 KeyValueNode 가 아니라 큰 표(TableNode) 쪽에 있다.
    # 두 경로 어디에서도 버려지면 안 되므로 합쳐서 본다.
    field_codes = ({p.field_code for kv in all_kv for p in kv.pairs if p.field_code}
                   | {c.field_code for c in cells if c.field_code})
    unit_codes = ({p.unit_code for kv in all_kv for p in kv.pairs if p.unit_code}
                  | {c.unit_code for c in cells if c.unit_code})
    unit_values = ({p.unit_value for kv in all_kv for p in kv.pairs if p.unit_value}
                   | {c.unit_value for c in cells if c.unit_value})
    assert field_codes, "ACODE 가 하나도 보존되지 않음"
    assert unit_codes, "AUNIT 이 하나도 보존되지 않음"
    # KIM 브랜치 추가: AUNITVALUE(DART 가 정규화해준 값)는 기존에 100% 폐기됐다.
    assert unit_values, "AUNITVALUE 가 하나도 보존되지 않음 (회귀)"
    assert any(kv.acode_group for kv in all_kv), "TABLE-GROUP[ACLASS] 가 보존되지 않음"


def test_exchange_key_value_structure(manifest, resolver):
    """§17: exchange 는 key-value 관계를 보존해야 한다 (Phase0 §5 실측 샘플과 대조)."""
    row = _row(manifest, "exchange_20250728800035")  # 삼성전자 반도체 위탁생산 공급계약
    docs = parse_documents_for_row(row, resolver)
    main = docs[0]
    all_kv = [n for s in main.sections for n in _all_children(s) if isinstance(n, KeyValueNode)]
    pairs = {p.key: p.value for kv in all_kv for p in kv.pairs}
    assert pairs.get("계약금액(원)") == "22,764,764,160,000"
    assert pairs.get("매출액대비(%)") == "7.6"


def test_holding_deep_section_and_library_wrapper_not_dropped(manifest, resolver):
    """회귀 테스트: LIBRARY 로 감싸진 SECTION 이 silent 하게 유실되던 버그 고정.

    이 문서는 정정공시라 SECTION-1 앞에 "정정대상 공시서류" 안내문이 SECTION 밖
    loose content 로 존재한다 — 이것도 synthetic section 으로 보존돼야 한다
    (경고 로그는 남기되 내용은 버리지 않는다, §7)."""
    row = _row(manifest, "holding_20241025000530")
    docs = parse_documents_for_row(row, resolver)
    main = docs[0]
    assert len(main.sections) > 0, "LIBRARY wrapper 안의 SECTION-1 이 유실됨 (회귀)"
    top_titles = [s.title for s in main.sections]
    assert any("제1부" in t for t in top_titles)
    # SECTION 밖 loose content(정정 안내문)가 synthetic section 으로 보존됐는지 확인
    assert any(w.startswith("SECTION 밖의") for w in main.parse_warnings)
    all_text = " ".join(
        n.text for s in main.sections for n in s.children if hasattr(n, "text")
    )
    assert "정정대상" in all_text


def test_no_silent_empty_content_across_random_sample(manifest, resolver):
    """모든 doc_group 무작위 표본에서 파싱 결과가 완전히 비어있지 않아야 한다."""
    import random

    xml_rows = [r for r in manifest if r.file_format == "xml"]
    sample = random.Random(42).sample(xml_rows, 80)
    empty = []
    for row in sample:
        docs = parse_documents_for_row(row, resolver)
        for d in docs:
            has_content = any(True for s in d.sections for _ in _all_children(s))
            if not has_content:
                empty.append((row.doc_id, d.report_subtype))
    assert empty == [], f"내용이 완전히 비어있는 파싱 결과: {empty}"
