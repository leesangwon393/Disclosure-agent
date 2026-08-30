"""periodic facts 잡음 필터 — 단어가 아니라 (항목명, 값) 쌍으로 판정한다."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from disclosure_rag.common.doc_tree import KVPair, KeyValueNode, ParsedDocument, SectionNode
from disclosure_rag.common.manifest_loader import ManifestRow
from disclosure_rag.correction.correction_graph_builder import CorrectionRecord
from disclosure_rag.facts.extractor import (
    extract_facts,
    normalize_key,
    parse_periodic_value,
    periodic_noise_reason,
)

ROOT = Path(__file__).resolve().parents[1]


def _reason(key: str, value: str):
    key_norm = normalize_key(key)[0]
    num, _unit, date = parse_periodic_value(value)
    return periodic_noise_reason(key_norm, value, value_num=num, value_date=date)


def test_header_pairs_are_removed_only_when_value_is_untyped():
    assert _reason("당기", "(단위 : 백만원)") == "header_pair"
    assert _reason("구분", "당분기말") == "header_pair"
    assert _reason("전기", "전기말") == "header_pair"


def test_gold_structure_labels_with_numeric_values_are_kept():
    assert _reason("합계", "1,234,567") is None
    assert _reason("기초", "(1,234)") is None
    assert _reason("기타", "△298,342") is None


def test_accounting_number_in_key_position_is_removed():
    assert _reason("△298,342", "51,749") == "numeric_key"


def test_periodic_accounting_numbers_are_parsed_without_changing_form_parser():
    assert parse_periodic_value("(1,234)")[0] == -1234
    assert parse_periodic_value("△298,342")[0] == -298342
    assert parse_periodic_value("(1,234천원)")[0] == -1_234_000


def _row(group: str) -> ManifestRow:
    return ManifestRow(
        doc_id=f"{group}_test", corp_code="00000000", corp_name="테스트",
        listed_name="테스트", stock_code="000000", industry="", sector="",
        doc_group=group, doc_subtype=None, report_nm="사업보고서",
        is_correction=False, rcept_no="1", rcept_dt="20250101", flr_nm="테스트",
        base_year=2024, base_month=12, file_path="x.xml", file_format="xml", n_files=1,
    )


def _parsed(group: str) -> ParsedDocument:
    kv = KeyValueNode(pairs=[
        KVPair("당기", "(단위 : 백만원)"),
        KVPair("구분", "당분기말"),
        KVPair("합계", "1,234,567"),
        KVPair("기타", "(1,234)"),
        KVPair("△298,342", "51,749"),
    ])
    return ParsedDocument(
        doc_id=f"{group}_test", doc_group=group, doc_subtype=None,
        report_subtype="main", source_path="x.xml", document_name="사업보고서",
        sections=[SectionNode("재무제표", 1, ["재무제표"], [kv])],
    )


def _correction(group: str) -> CorrectionRecord:
    return CorrectionRecord(
        doc_id=f"{group}_test", correction_group_id=f"{group}_test",
        correction_order=0, is_correction=False, is_latest=True,
        resolution_source="original",
    )


def test_extract_filter_is_strictly_periodic_only():
    periodic = extract_facts(_parsed("periodic"), _row("periodic"), _correction("periodic"))
    form = extract_facts(_parsed("exchange"), _row("exchange"), _correction("exchange"))

    assert [(f.key_norm, f.value_text) for f in periodic] == [
        ("합계", "1,234,567"), ("기타", "(1,234)"),
    ]
    assert {f.key_norm for f in form} == {"당기", "구분", "합계", "기타", "△298,342"}


def test_clean_periodic_gold_186_pairs_survive_filter():
    gold_path = ROOT / "eval" / "gold_passages_clean.jsonl"
    db_path = ROOT / "artifacts_v2" / "facts_periodic" / "facts.sqlite"
    if not gold_path.exists() or not db_path.exists():
        pytest.skip("periodic baseline artifact 없음")

    gold = [
        json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("doc_group") == "periodic"
    ]
    assert len(gold) == 186
    db = sqlite3.connect(db_path)
    misses = []
    for item in gold:
        ids = item["gold_report_ids"]
        marks = ",".join("?" for _ in ids)
        key_norm = normalize_key(item["key"])[0]
        rows = db.execute(
            f"SELECT value_text, value_num, value_date FROM facts "
            f"WHERE doc_id IN ({marks}) AND key_norm = ?", [*ids, key_norm],
        )
        answers = {str(x).strip() for x in item["answers"]}
        survived = False
        for value, _old_num, old_date in rows:
            if str(value).strip() not in answers:
                continue
            num, _unit, date = parse_periodic_value(value)
            if periodic_noise_reason(
                key_norm, value, value_num=num, value_date=date or old_date,
            ) is None:
                survived = True
                break
        if not survived:
            misses.append((item["id"], item["key"]))
    db.close()
    assert misses == []
