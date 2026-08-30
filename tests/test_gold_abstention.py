import json
import sqlite3
from collections import Counter
from pathlib import Path

from disclosure_rag.agent.field_schema import FieldSchema, normalize_field_key
from disclosure_rag.agent.query_plan import QueryPlan
from disclosure_rag.agent.scope_gate import evaluate_scope
from disclosure_rag.entity.entity_registry import EntityRegistry

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "eval" / "gold_abstention.jsonl"


def _rows():
    return [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_gold_has_at_least_100_balanced_unique_questions():
    rows = _rows()
    assert len(rows) >= 100
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({row["query"] for row in rows}) == len(rows)
    assert Counter(row["reason"] for row in rows) == {
        "hard_out": 40, "ambiguous": 40, "no_answer_in_doc": 40, "wrong_entity": 40,
    }


def test_wrong_entities_really_do_not_exist_in_registry():
    registry = EntityRegistry.load(ROOT / "artifacts_v2" / "registry" / "entities.json")
    wrong = [row for row in _rows() if row["reason"] == "wrong_entity"]
    assert wrong
    assert all(not registry.types_for(row["company"]) for row in wrong)


def test_lg_electronics_is_not_rejected_just_for_being_outside_universe():
    registry = EntityRegistry.load(ROOT / "artifacts_v2" / "registry" / "entities.json")
    types = registry.types_for("LG전자")
    assert "universe" not in types and "submitter" in types
    decision = evaluate_scope(
        QueryPlan(companies=["LG전자"]), "LG전자가 제출한 대량보유보고서를 찾아줘.", registry,
    )
    assert decision.action == "proceed" and decision.scope == "in_scope"


def test_all_hard_out_examples_are_immediate_refusals():
    registry = EntityRegistry.load(ROOT / "artifacts_v2" / "registry" / "entities.json")
    for row in (x for x in _rows() if x["reason"] == "hard_out"):
        decision = evaluate_scope(QueryPlan(companies=[row["company"]]), row["query"], registry)
        assert decision.should_refuse, row["id"]
        assert decision.scope == "hard_out_scope"


def test_wrong_entities_are_not_early_refused():
    registry = EntityRegistry.load(ROOT / "artifacts_v2" / "registry" / "entities.json")
    for row in (x for x in _rows() if x["reason"] == "wrong_entity"):
        decision = evaluate_scope(QueryPlan(companies=[row["company"]]), row["query"], registry)
        assert decision.action == "proceed", row["id"]
        assert decision.scope == "possibly_scope"


def test_no_answer_documents_exist_but_expected_field_does_not_in_doc_or_schema():
    schema = FieldSchema.load(ROOT / "config" / "field_schema.json")
    connection = sqlite3.connect(ROOT / "artifacts_v2" / "facts" / "facts.sqlite")
    try:
        for row in (x for x in _rows() if x["reason"] == "no_answer_in_doc"):
            for doc_id in row["gold_doc_ids"]:
                assert connection.execute("SELECT 1 FROM facts WHERE doc_id=? LIMIT 1", (doc_id,)).fetchone(), row["id"]
                facts = connection.execute(
                    "SELECT report_name, key_norm FROM facts WHERE doc_id=?", (doc_id,)
                ).fetchall()
                keys = {normalize_field_key(value) for _, value in facts}
                assert not keys.intersection(map(normalize_field_key, row["expected_fields"])), row["id"]
                kinds = schema.match_kinds(facts[0][0])
                assert kinds, row["id"]
                assert all(
                    schema.classify(kind, field) == "unknown"
                    for kind in kinds for field in row["expected_fields"]
                ), row["id"]
    finally:
        connection.close()
