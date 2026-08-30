import gzip
import json
import sqlite3

from disclosure_rag.entity.entity_registry import (
    EntityEvidence,
    EntityRegistry,
    EntityRegistryBuilder,
    iter_role_mentions_from_chunks,
    iter_subsidiary_mentions_from_chunks,
    lookup_key,
    normalize_name,
)


def test_name_normalization_keeps_display_and_stabilizes_legal_mark():
    assert normalize_name("  삼성전자㈜  ") == "삼성전자(주)"
    assert lookup_key("삼성전자 주식회사") == lookup_key("삼성전자(주)")


def test_registry_merges_alias_roles_and_round_trips(tmp_path):
    builder = EntityRegistryBuilder()
    builder.add("삼성전자", "universe", source="u", aliases=["삼성전자(주)"])
    builder.add(
        "삼성전자㈜", "counterparty", source="facts",
        evidence=EntityEvidence(
            source="facts", role="counterparty", report_id="r1",
            document_company="삼성중공업", field="계약상대",
        ),
    )
    registry = builder.build()
    assert registry.contains("삼성전자", "universe")
    assert registry.contains("삼성전자(주)", "counterparty")
    assert registry.types_for("삼성전자㈜") == {"universe", "counterparty"}

    path = tmp_path / "entities.json"
    registry.save(path)
    loaded = EntityRegistry.load(path)
    assert loaded.contains("삼성전자 주식회사", "counterparty")


def test_evidence_limit_is_per_role_not_per_entity():
    builder = EntityRegistryBuilder(max_evidence_per_role=2)
    for index in range(4):
        builder.add(
            "A", "submitter", source="manifest",
            evidence=EntityEvidence(source="manifest", role="submitter", report_id=f"s{index}"),
        )
    builder.add(
        "A", "counterparty", source="facts",
        evidence=EntityEvidence(source="facts", role="counterparty", report_id="c1"),
    )
    evidence = builder.build().resolve("A").evidence
    assert [item.role for item in evidence].count("submitter") == 2
    assert [item.role for item in evidence].count("counterparty") == 1


def test_chunk_fallback_only_accepts_exact_role_field(tmp_path):
    path = tmp_path / "chunks.jsonl.gz"
    rows = [
        {"chunk_id": "c1", "report_id": "r1", "company": "삼성중공업",
         "raw_text": "[계약]\n- 계약상대: 삼성전자(주)\n- 계약금액: 100"},
        {"chunk_id": "c2", "report_id": "r2", "company": "A",
         "raw_text": "본문에서 계약상대를 설명하지만 KV 필드가 아님"},
    ]
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    mentions = list(iter_role_mentions_from_chunks(path))
    assert len(mentions) == 1
    assert mentions[0][0] == "삼성전자(주)"
    assert mentions[0][1].document_company == "삼성중공업"


def test_subsidiary_parser_requires_table_context_and_columns(tmp_path):
    path = tmp_path / "chunks.jsonl.gz"
    rows = [
        {"chunk_id": "c1", "report_id": "r1", "company": "삼성전자", "is_leaf": True,
         "raw_text": "나. 종속기업 현황\n지역: 미주 / 기업명: Samsung Electronics America, Inc. (SEA) / 업종: 판매 / 지분율: 100"},
        {"chunk_id": "c2", "report_id": "r2", "company": "A", "is_leaf": True,
         "raw_text": "B는 종속기업이다. 기업명: B / 업종: 판매"},
        {"chunk_id": "p1", "report_id": "r1", "company": "삼성전자", "is_leaf": False,
         "raw_text": "나. 종속기업 현황\n기업명: Parent Duplicate / 업종: 판매"},
    ]
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    mentions = list(iter_subsidiary_mentions_from_chunks(path))
    assert [item[0] for item in mentions] == ["Samsung Electronics America, Inc. (SEA)"]


def test_registry_loads_legacy_payload_without_optional_metadata(tmp_path):
    path = tmp_path / "minimal.json"
    path.write_text(json.dumps({
        "entities": [{"entity_id": "e1", "canonical_name": "A", "types": ["submitter"]}],
    }), encoding="utf-8")
    registry = EntityRegistry.load(path)
    assert registry.contains("A", "submitter")
