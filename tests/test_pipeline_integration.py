"""파이프라인 전 구간 배선 검사.

모듈 9개를 각자 따로 테스트했지만 **서로 연결되는지는 아무도 안 봤다.**
타입이 안 맞거나 필드 이름이 어긋나면 단위 테스트는 다 통과하는데 실제로는
아무것도 안 돈다. 여기서 질문 하나가 처음부터 끝까지 통과하는지 본다.

    질의이해 -> QueryPlan -> 범위게이트 -> 분해 -> 이중채널(+버전정리)
    -> 근거구조화 -> 충분성 -> 거부게이트 -> 답변프롬프트

검색기와 Facts 는 가짜를 쓴다. 여기서 보는 건 검색 품질이 아니라 **배선**이다.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "eval" / "suite_v1.jsonl"
SCHEMA = ROOT / "config" / "field_schema.json"
REGISTRY = ROOT / "artifacts_v2" / "registry" / "entities.json"
CORPUS = ROOT / "corpus"

pytestmark = pytest.mark.skipif(
    not (SUITE.exists() and SCHEMA.exists() and REGISTRY.exists() and CORPUS.exists()),
    reason="정답셋/명세/레지스트리/코퍼스 미구비",
)

from disclosure_rag.agent.abstention import decide_abstention  # noqa: E402
from disclosure_rag.agent.answer_generator import (  # noqa: E402
    answer_token_budget,
    build_answer_prompt,
)
from disclosure_rag.agent.decompose import decompose_and_search  # noqa: E402
from disclosure_rag.agent.dual_channel import (  # noqa: E402
    DualChannelRetriever,
    retrieval_filter_from_plan,
)
from disclosure_rag.agent.evidence_processor import process_evidence  # noqa: E402
from disclosure_rag.agent.scope_gate import apply_scope_gate  # noqa: E402
from disclosure_rag.agent.sufficiency import check_sufficiency  # noqa: E402
from disclosure_rag.chunking.chunk_schema import ChunkSchema  # noqa: E402


# ------------------------------------------------------------------ 가짜 부품

def make_chunk(cid, rid, company, fields: dict, order=0, latest=True, group=None):
    return ChunkSchema(
        chunk_id=cid, report_id=rid, text="", raw_text=" ".join(fields),
        company=company, report_type="exchange", correction_group_id=group or rid,
        correction_order=order, is_latest=latest,
        field_codes=[{"key": k, "text": v} for k, v in fields.items()],
    )


class FakeUnstructured:
    """HybridRetriever 자리. 필터에 걸린 회사만 돌려준다."""
    def __init__(self, chunks): self.chunks, self.calls = chunks, []

    def search(self, query, *, k=10, flt=None, **kw):
        self.calls.append((query, k))
        hits = [c for c in self.chunks if flt is None or flt.matches(c)]
        return [(c, 1.0 - i * 0.01) for i, c in enumerate(hits[:k])]


class FakeFactStore:
    def __init__(self, rows): self.rows = rows
    def distinct_keys(self, limit=10_000):
        return [(r["key_norm"], 1) for r in self.rows]
    def lookup(self, *, company=None, key=None, doc_group=None, period=None,
               latest_only=True, limit=20, **kw):
        return [r for r in self.rows
                if (company is None or r["company"] == company)
                and (key is None or r["key_norm"] == key)][:limit]


@pytest.fixture(scope="module")
def parts():
    from disclosure_rag.agent.field_schema import FieldSchema
    from disclosure_rag.agent.query_plan import PlanValidator, RulePlanBuilder
    from disclosure_rag.entity.entity_extractor import EntityExtractor
    from disclosure_rag.entity.entity_registry import EntityRegistry

    schema = FieldSchema.load(SCHEMA)
    return SimpleNamespace(
        schema=schema,
        registry=EntityRegistry.load(REGISTRY),
        builder=RulePlanBuilder(
            schema=schema,
            extractor=EntityExtractor(corpus_root=str(CORPUS),
                                      metric_terms_path=str(ROOT / "config" / "metric_terms.txt")),
        ),
        validator=PlanValidator(registry=EntityRegistry.load(REGISTRY), schema=schema),
    )


def run_pipeline(parts, question, chunks, fact_rows=()):
    """전 구간을 한 번 통과시킨다. 각 단계의 산출물을 그대로 돌려준다."""
    plan = parts.builder.build(question)
    validation = parts.validator.validate(plan)
    scope = apply_scope_gate(plan, question, parts.registry)

    dual = DualChannelRetriever(FakeUnstructured(chunks), FakeFactStore(list(fact_rows)))

    def search_fn(text, sub_plan, k):
        return dual.search(text, sub_plan, k=k).unstructured_results

    decomposed = decompose_and_search(plan, question, search_fn)
    processed = process_evidence(plan, decomposed.merged)
    sufficiency = check_sufficiency(plan, processed, decompose_result=decomposed)
    abstention = decide_abstention(
        plan, evidence_count=len(decomposed.merged),
        found_fields=processed.found_fields,
    )
    prompt = build_answer_prompt(plan)
    return SimpleNamespace(plan=plan, validation=validation, scope=scope,
                           decomposed=decomposed, processed=processed,
                           sufficiency=sufficiency, abstention=abstention, prompt=prompt)


# ------------------------------------------------------------------ 전 구간

def test_single_lookup_flows_end_to_end(parts):
    """S005 형태 — 한 회사, 한 항목."""
    chunks = [make_chunk("c1", "exchange_1", "SK하이닉스",
                         {"투자금액": "1,200,000,000", "투자목적": "생산 확대"})]
    out = run_pipeline(parts, "SK하이닉스의 신규시설투자등에 기재된 투자금액은 얼마인가?", chunks)

    assert out.validation.ok
    assert not out.scope.should_refuse
    assert not out.decomposed.decomposed          # 쪼개지 않는다
    assert out.plan.answer_mode == "closed"
    assert out.abstention.action == "answer"
    assert "단일 값" in out.prompt


def test_two_company_comparison_flows_end_to_end(parts):
    """S007 형태 — 회사별로 쪼개고, 양쪽 근거가 다 모여야 한다."""
    chunks = [
        make_chunk("c1", "ex_a", "삼성전자", {"계약금액": "1,000"}),
        make_chunk("c2", "ex_b", "한미반도체", {"계약금액": "2,000"}),
    ]
    q = "삼성전자와 한미반도체 중, 각각 공시한 단일판매·공급계약 가운데 최대 계약금액은 얼마이며 더 큰 쪽은 어느 기업인가?"
    out = run_pipeline(parts, q, chunks)

    assert out.decomposed.decomposed and len(out.decomposed.sub_queries) == 2
    assert not out.decomposed.partial
    companies = {c.company for c, _s in out.decomposed.merged}
    assert companies == {"삼성전자", "한미반도체"}
    assert out.plan.task == "compare"
    assert "한쪽 대상의 근거가 없으면" in out.prompt


def test_missing_one_company_is_caught_not_answered(parts):
    """근거 건수만 보면 통과한다 — 분해의 빈손 신호가 잡아야 한다."""
    chunks = [make_chunk(f"c{i}", "ex_a", "삼성전자", {"계약금액": "1,000"}) for i in range(10)]
    q = "삼성전자와 한미반도체 중, 각각 공시한 단일판매·공급계약 가운데 최대 계약금액은 얼마이며 더 큰 쪽은 어느 기업인가?"
    out = run_pipeline(parts, q, chunks)

    assert len(out.decomposed.merged) == 10        # 근거는 많다
    assert out.decomposed.partial                  # 그런데 한쪽이 비었다
    assert not out.sufficiency.ok
    assert any("한미반도체" in label for label in out.sufficiency.empty_sub_queries)


def test_correction_diff_keeps_both_versions(parts):
    """S023 형태 — 버전 정리가 최초·최종을 남기고, 구조화가 짝을 맞춘다."""
    chunks = [
        # 같은 정정 체인이므로 correction_group_id 를 공유한다(실제 데이터 형태)
        make_chunk("c1", "ex_orig", "현대건설", {"계약금액": "1,000"}, 0, False, "ex_orig"),
        make_chunk("c2", "ex_mid", "현대건설", {"계약금액": "1,100"}, 1, False, "ex_orig"),
        make_chunk("c3", "ex_fin", "현대건설", {"계약금액": "1,500"}, 2, True, "ex_orig"),
    ]
    q = "현대건설의 단일판매ㆍ공급계약체결 공시가 정정된 내역이 있는가? 있다면 최초 공시와 최종 정정본 사이에 무엇이 달라졌는지 설명해줘."
    out = run_pipeline(parts, q, chunks)

    assert out.plan.task == "correction_diff"
    assert out.plan.latest_policy == "first_and_final"
    kept = {c.report_id for c, _s in out.decomposed.merged}
    assert kept == {"ex_orig", "ex_fin"}           # 가운데 버전만 빠진다
    pair = out.processed.pairs[0]
    assert pair.complete and pair.changed is True
    assert "첫 줄에서 예/아니오를 단정" in out.prompt


def test_evidence_absent_abstains_before_calling_the_model(parts):
    """근거 0건이면 HCX 를 부르지 않는다 — hallucination 이 물리적으로 불가능."""
    out = run_pipeline(parts, "SK하이닉스의 신규시설투자등에 기재된 투자금액은 얼마인가?", [])
    assert out.abstention.should_abstain
    assert out.abstention.reason == "evidence_absent"
    assert "확인되지 않습니다" in out.abstention.message


def test_out_of_scope_question_refuses_at_the_gate(parts):
    out = run_pipeline(parts, "삼성전자 현재 주가가 얼마야?", [])
    assert out.scope.should_refuse
    assert out.scope.scope == "hard_out_scope"


def test_facts_channel_runs_when_fields_are_structured(parts):
    """정형 항목이 있으면 Facts 를 검색과 **병렬로** 조회한다."""
    dual = DualChannelRetriever(
        FakeUnstructured([]),
        FakeFactStore([{"key_norm": "계약금액", "company": "삼성전자", "doc_id": "ex_a",
                        "chunk_id": "ex_a::c1", "value_text": "1,000", "value_num": 1000.0,
                        "is_latest": True, "correction_order": 0,
                        "correction_group_id": "ex_a"}]),
    )
    plan = parts.builder.build("삼성전자의 단일판매ㆍ공급계약체결 공시 내용을 정리해줘")
    res = dual.search("질문", plan)
    assert res.facts_executed
    assert "계약금액" in res.structured_fields
    assert res.facts and res.facts[0]["report_id"] == "ex_a"


def test_facts_are_not_scored_or_fused(parts):
    """Facts 는 sqlite 조회라 점수가 없다 — 융합에 섞으면 안 된다."""
    dual = DualChannelRetriever(
        FakeUnstructured([]),
        FakeFactStore([{"key_norm": "계약금액", "company": "삼성전자", "doc_id": "ex_a",
                        "chunk_id": "ex_a::c1", "value_text": "1,000", "is_latest": True,
                        "correction_order": 0, "correction_group_id": "ex_a"}]),
    )
    plan = parts.builder.build("삼성전자의 단일판매ㆍ공급계약체결 공시 내용을 정리해줘")
    payload = dual.search("질문", plan).to_dict()
    assert all("score" not in row for row in payload["facts"])


def test_token_budget_follows_the_mode(parts):
    closed = run_pipeline(parts, "SK하이닉스의 신규시설투자등에 기재된 투자금액은 얼마인가?", [])
    open_ = run_pipeline(parts, "SK하이닉스의 2024년 신규시설투자등 공시를 기준으로 주요 내용을 정리해줘.", [])
    assert answer_token_budget(closed.plan) < answer_token_budget(open_.plan)


# ------------------------------------------------------------------ 정답셋 전체

def test_whole_suite_passes_through_without_crashing(parts):
    """38문항 전부가 배선을 통과하는지. 품질이 아니라 **연결**을 본다."""
    rows = [json.loads(l) for l in SUITE.read_text(encoding="utf-8").splitlines() if l.strip()]
    refused, errors = [], []
    for r in rows:
        chunks = [make_chunk("c1", "ex_1", (r.get("company") or "삼성전자"),
                             {"계약금액": "1,000"})]
        try:
            out = run_pipeline(parts, r["query"], chunks)
        except Exception as exc:  # noqa: BLE001
            errors.append((r["id"], f"{type(exc).__name__}: {exc}"))
            continue
        if out.scope.should_refuse:
            refused.append(r["id"])
    assert errors == []
    assert refused == []          # 정상 질문을 조기 차단하면 안 된다
