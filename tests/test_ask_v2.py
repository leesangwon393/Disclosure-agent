"""신 파이프라인 진입점 — 14단계가 실제로 순서대로 도는가.

가장 중요한 검사는 **HCX 호출 횟수**다. 범위 게이트나 거부 게이트에서 막힌
질문에 모델을 부르면, 지어낼 기회를 준 것이다. 그건 규칙으로 막아야 한다.
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

from disclosure_rag.agent.ask_v2 import AskV2  # noqa: E402
from disclosure_rag.chunking.chunk_schema import ChunkSchema  # noqa: E402


class SpyClient:
    """HCX 자리. 호출 횟수를 센다."""
    def __init__(self, reply="계약금액은 1,000원입니다.\n근거: ex_a(c1)"):
        self.reply, self.calls = reply, 0

    def chat(self, messages, **kw):
        self.calls += 1
        return {"content": self.reply}


class FakeDual:
    def __init__(self, chunks, facts=(), manifest=()):
        self.chunks, self.facts, self.queries = chunks, list(facts), []
        # 존재 전수 확인(Stage 3.5)이 읽는다. 비어 있으면 그 단계를 건너뛴다.
        self.manifest = list(manifest)

    def search(self, query, plan, *, k=10, **kw):
        self.queries.append((query, k))
        hits = [c for c in self.chunks
                if not plan.companies or c.company in plan.companies]
        return SimpleNamespace(
            unstructured_results=[(c, 1.0 - i * 0.01) for i, c in enumerate(hits[:k])],
            facts=list(self.facts), facts_executed=bool(self.facts),
        )


def make_chunk(cid, rid, company, fields, order=0, latest=True, group=None, text=""):
    return ChunkSchema(
        # raw_text 에 **값까지** 넣는다. 실제 청크는 표 내용을 그대로 담으므로
        # 항목명만 넣으면 Validator 가 정상 답변을 '지어낸 숫자'로 잡는다.
        chunk_id=cid, report_id=rid, text="",
        raw_text=text or " ".join(f"{k}: {v}" for k, v in fields.items()),
        company=company, report_type="exchange", report_name="단일판매ㆍ공급계약체결",
        filing_date="20240101", correction_group_id=group or rid,
        correction_order=order, is_latest=latest,
        field_codes=[{"key": k, "text": v} for k, v in fields.items()],
    )


@pytest.fixture(scope="module")
def builder():
    from disclosure_rag.agent.field_schema import FieldSchema
    from disclosure_rag.agent.query_plan import RulePlanBuilder
    from disclosure_rag.entity.entity_extractor import EntityExtractor
    return RulePlanBuilder(
        schema=FieldSchema.load(SCHEMA),
        extractor=EntityExtractor(corpus_root=str(CORPUS),
                                  metric_terms_path=str(ROOT / "config" / "metric_terms.txt")),
    )


@pytest.fixture(scope="module")
def registry():
    from disclosure_rag.entity.entity_registry import EntityRegistry
    return EntityRegistry.load(REGISTRY)


def make_ask(builder, registry, chunks, facts=(), client=None, manifest=(), **kw):
    return AskV2(client=client or SpyClient(),
                 dual_retriever=FakeDual(chunks, facts, manifest),
                 plan_builder=builder, registry=registry, **kw)


# --------------------------------------------------------------------------- 정상 경로

def test_happy_path_calls_the_model_once(builder, registry):
    chunks = [make_chunk("c1", "ex_a", "삼성전자", {"계약금액": "1,000"})]
    ask = make_ask(builder, registry, chunks)
    out = ask.run("삼성전자의 단일판매ㆍ공급계약체결 공시에 기재된 계약금액은 얼마인가?")

    assert out.stopped_at == "answered"
    assert ask.client.calls == 1                 # 답변 생성 1회뿐
    assert out.hcx_calls == 1
    assert out.evidence_pack is not None and out.evidence_pack.citations
    assert "1,000" in out.answer


def test_evidence_pack_keeps_the_existing_block_format(builder, registry):
    """답변 프롬프트가 이 형식에 맞춰 튜닝돼 있다 — 바뀌면 튜닝이 무효가 된다."""
    chunks = [make_chunk("c1", "ex_a", "삼성전자", {"계약금액": "1,000"})]
    out = make_ask(builder, registry, chunks).run(
        "삼성전자의 단일판매ㆍ공급계약체결 공시에 기재된 계약금액은 얼마인가?")
    text = out.evidence_pack.prompt_text
    assert "[USER QUESTION]" in text and "[EVIDENCE 1]" in text
    assert "report_id: ex_a" in text and "chunk_id: c1" in text
    assert "정정 상태:" in text


# --------------------------------------------------------------------------- 모델을 안 부르는 경로

def test_out_of_scope_never_calls_the_model(builder, registry):
    ask = make_ask(builder, registry, [])
    out = ask.run("삼성전자 현재 주가가 얼마야?")
    assert out.stopped_at == "scope_gate"
    assert ask.client.calls == 0                 # ★ 지어낼 기회 자체가 없다
    assert out.evidence_pack is None


def test_zero_evidence_never_calls_the_model(builder, registry):
    ask = make_ask(builder, registry, [])
    out = ask.run("삼성전자의 단일판매ㆍ공급계약체결 공시에 기재된 계약금액은 얼마인가?")
    assert out.stopped_at == "abstention_gate"
    assert ask.client.calls == 0
    assert "확인되지 않습니다" in out.answer


def test_missing_comparison_target_never_calls_the_model(builder, registry):
    """근거 10건이지만 한쪽 회사가 통째로 비었다 — 건수로는 통과한다."""
    chunks = [make_chunk(f"c{i}", "ex_a", "삼성전자", {"계약금액": "1,000"})
              for i in range(10)]
    ask = make_ask(builder, registry, chunks)
    out = ask.run("삼성전자와 한미반도체 중, 각각 공시한 단일판매·공급계약 가운데 "
                  "최대 계약금액은 얼마이며 더 큰 쪽은 어느 기업인가?")
    assert out.stopped_at == "abstention_gate"
    assert ask.client.calls == 0
    assert out.abstention.reason == "target_evidence_missing"
    assert len(out.decomposed.merged) == 10      # 근거는 많았다


# --------------------------------------------------------------------------- 재검색

def test_insufficient_evidence_triggers_retry(builder, registry):
    """항목이 모자라면 빠진 항목을 지목해 다시 검색한다."""
    chunks = [make_chunk("c1", "ex_a", "SK하이닉스", {"투자금액": "1,000"})]
    ask = make_ask(builder, registry, chunks)
    out = ask.run("SK하이닉스의 2024년 신규시설투자등 공시를 기준으로 주요 내용을 정리해줘.")
    assert out.retries > 0
    assert not out.sufficiency.ok
    assert ask.dual.queries and len(ask.dual.queries) > 1     # 실제로 다시 검색했다


def test_retry_budget_is_bounded(builder, registry):
    chunks = [make_chunk("c1", "ex_a", "SK하이닉스", {"투자금액": "1,000"})]
    ask = make_ask(builder, registry, chunks, max_nudges=2)
    out = ask.run("SK하이닉스의 2024년 신규시설투자등 공시를 기준으로 주요 내용을 정리해줘.")
    assert out.retries <= 2


def test_open_question_with_gaps_answers_partially(builder, registry):
    """open 은 확인된 내용을 답하고 누락을 명시한다 — 통째로 거부하지 않는다."""
    chunks = [make_chunk("c1", "ex_a", "SK하이닉스", {"투자금액": "1,000"})]
    ask = make_ask(builder, registry, chunks)
    out = ask.run("SK하이닉스의 2024년 신규시설투자등 공시를 기준으로 주요 내용을 정리해줘.")
    assert out.abstention.action == "partial"
    assert ask.client.calls == 1
    assert "한계:" in out.answer


# --------------------------------------------------------------------------- 부모 확장

def test_parent_expansion_keeps_field_codes(builder, registry):
    """expand() 는 dict 를 돌려줘 field_codes 가 사라진다 — Stage 9 가 죽는다."""
    class Expander:
        def expand_one(self, chunk):
            return "상위 문단 전체 내용 " + (chunk.raw_text or "")

    chunks = [make_chunk("c1", "ex_a", "삼성전자", {"계약금액": "1,000"})]
    out = make_ask(builder, registry, chunks, parent_expander=Expander()).run(
        "삼성전자의 단일판매ㆍ공급계약체결 공시에 기재된 계약금액은 얼마인가?")
    assert "상위 문단 전체 내용" in out.evidence_pack.prompt_text
    assert out.processed.found_fields == ["계약금액"]      # 구조가 살아 있다


def test_parent_expansion_failure_does_not_kill_the_question(builder, registry):
    class Broken:
        def expand_one(self, chunk):
            raise RuntimeError("확장 실패")

    chunks = [make_chunk("c1", "ex_a", "삼성전자", {"계약금액": "1,000"})]
    out = make_ask(builder, registry, chunks, parent_expander=Broken()).run(
        "삼성전자의 단일판매ㆍ공급계약체결 공시에 기재된 계약금액은 얼마인가?")
    assert out.stopped_at == "answered"


# --------------------------------------------------------------------------- Facts

def test_facts_go_into_a_separate_block_without_scores(builder, registry):
    facts = [{"report_id": "ex_a", "item": "계약금액", "value": "1,000",
              "company": "삼성전자", "report_name": "단일판매ㆍ공급계약체결",
              "filing_date": "20240101"}]
    chunks = [make_chunk("c1", "ex_a", "삼성전자", {"계약금액": "1,000"})]
    out = make_ask(builder, registry, chunks, facts).run(
        "삼성전자의 단일판매ㆍ공급계약체결 공시에 기재된 계약금액은 얼마인가?")
    text = out.evidence_pack.prompt_text
    assert "[FACT]" in text and "계약금액: 1,000" in text


# --------------------------------------------------------------------------- 길이 상한

def test_long_evidence_is_truncated_with_a_notice(builder, registry):
    """실측: 근거 13,542자에서 HCX 가 정답을 못 찾은 사례가 있다."""
    big = "가" * 3000
    chunks = [make_chunk(f"c{i}", f"ex_{i}", "삼성전자", {"계약금액": "1,000"}, text=big)
              for i in range(10)]
    out = make_ask(builder, registry, chunks, max_evidence_chars=6000).run(
        "삼성전자의 단일판매ㆍ공급계약체결 공시에 기재된 계약금액은 얼마인가?")
    text = out.evidence_pack.prompt_text
    assert "길이 제한으로 생략" in text
    assert len(out.evidence_pack.citations) < 10
    assert len(text) < 12000


# --------------------------------------------------------------------------- 정답셋 전체

def test_whole_suite_runs_without_crashing(builder, registry):
    rows = [json.loads(l) for l in SUITE.read_text(encoding="utf-8").splitlines() if l.strip()]
    errors, stopped = [], {}
    for r in rows:
        chunks = [make_chunk("c1", "ex_1", (r.get("company") or "삼성전자"),
                             {"계약금액": "1,000"})]
        ask = make_ask(builder, registry, chunks)
        try:
            out = ask.run(r["query"])
        except Exception as exc:  # noqa: BLE001
            errors.append((r["id"], f"{type(exc).__name__}: {exc}"))
            continue
        stopped[r["id"]] = out.stopped_at
        # 조기 종료했다면 모델을 부르지 않았어야 한다
        if out.stopped_at != "answered":
            assert ask.client.calls == 0, r["id"]
    assert errors == []
    assert len(stopped) == len(rows)


# --------------------------------------------------------------------------- 답변 검증

def test_validator_runs_by_default(builder, registry):
    """지어낸 숫자를 LLM 없이 잡는 유일한 장치다 — v2 경로에서 통째로 빠져
    있었다(2026-08-30 발견). 껐다 켰다 할 성격이 아니라 기본으로 돈다."""
    chunks = [make_chunk("c1", "ex_a", "삼성전자", {"계약금액": "1,000"})]
    ask = make_ask(builder, registry, chunks,
                   client=SpyClient("계약금액은 1,000원입니다.\n근거: ex_a(c1)"))
    out = ask.run("삼성전자의 단일판매ㆍ공급계약체결 공시에 기재된 계약금액은 얼마인가?")
    assert out.validation_result is not None
    assert out.validation_result.numbers_grounded


def test_validator_flags_a_fabricated_number(builder, registry):
    """근거에 없는 숫자를 답하면 잡혀야 한다."""
    chunks = [make_chunk("c1", "ex_a", "삼성전자", {"계약금액": "1,000"})]
    ask = make_ask(builder, registry, chunks,
                   client=SpyClient("계약금액은 987,654,321원입니다.\n근거: ex_a(c1)"))
    out = ask.run("삼성전자의 단일판매ㆍ공급계약체결 공시에 기재된 계약금액은 얼마인가?")
    assert not out.validation_result.numbers_grounded
    assert out.validation_result.ungrounded_numbers


def test_validator_failure_does_not_kill_the_answer(builder, registry):
    def broken(answer, pack, plan):
        raise RuntimeError("검증기 오류")

    chunks = [make_chunk("c1", "ex_a", "삼성전자", {"계약금액": "1,000"})]
    out = make_ask(builder, registry, chunks, answer_validator=broken).run(
        "삼성전자의 단일판매ㆍ공급계약체결 공시에 기재된 계약금액은 얼마인가?")
    assert out.stopped_at == "answered" and out.answer


# ------------------------------------------------------- 존재 전수 확인 (Stage 3.5)
#
# 검색은 상위 k건만 본다. 거기 없다고 '없다'고 말할 수 없어서 모델이
# "확인할 수 없습니다"로 물러섰다(v2_off4 실측 2문항, 정답은 둘 다 '아니오').
# manifest 는 전체 목록이라 0건을 근거 있게 말할 수 있다.

def _mrow(corp, nm, dt="20240101", doc_id=None, is_correction=False):
    return SimpleNamespace(corp_name=corp, report_nm=nm, rcept_dt=dt,
                           doc_id=doc_id or f"exchange_{dt}800001",
                           is_correction=is_correction)


_HANMI_MANIFEST = [
    _mrow("한미반도체", "단일판매ㆍ공급계약체결", "20230901"),
    _mrow("한미반도체", "단일판매ㆍ공급계약체결(자율공시)", "20230612"),
    _mrow("한미반도체", "주요사항보고서(자기주식취득신탁계약해지결정)", "20240417"),
]

_EXISTENCE_Q = "한미반도체가 체결한 단일판매·공급계약 중 이후 해지된 계약이 존재하는가?"


def test_confirmed_absence_reaches_the_model_with_a_verdict(builder, registry):
    """근거 0건이어도 전수 확인이 '아니오'면 거부하지 않고 답하게 한다."""
    ask = make_ask(builder, registry, [], manifest=_HANMI_MANIFEST)
    out = ask.run(_EXISTENCE_Q)

    assert out.existence.applicable and out.existence.verdict == "아니오"
    assert out.stopped_at == "answered"          # 거부 게이트를 넘겼다
    assert ask.client.calls == 1
    block = out.evidence_pack.prompt_text
    assert "[전수 확인]" in block and "0건" in block


def test_absence_block_sits_before_the_evidence(builder, registry):
    """긴 근거 뒤에 붙이면 모델이 못 보고 '확인할 수 없습니다'로 돌아간다."""
    chunks = [make_chunk("c1", "ex_a", "한미반도체", {"계약금액": "1,000"})]
    out = make_ask(builder, registry, chunks, manifest=_HANMI_MANIFEST).run(_EXISTENCE_Q)
    text = out.evidence_pack.prompt_text
    assert text.index("[전수 확인]") < text.index("[EVIDENCE 1]")


def test_present_event_is_reported_as_yes(builder, registry):
    q = "한미반도체가 자기주식취득신탁계약을 해지한 적이 있는가?"
    out = make_ask(builder, registry, [], manifest=_HANMI_MANIFEST).run(q)
    assert out.existence.verdict == "예"
    assert "예" in out.evidence_pack.prompt_text


def test_without_manifest_behaviour_is_unchanged(builder, registry):
    """manifest 가 없으면 이 단계를 통째로 건너뛴다 — 기존 동작 그대로."""
    ask = make_ask(builder, registry, [])
    out = ask.run(_EXISTENCE_Q)
    assert not out.existence.applicable
    assert out.stopped_at == "abstention_gate"
    assert ask.client.calls == 0


def test_non_existence_question_is_untouched(builder, registry):
    """값을 묻는 질문에는 전수 확인이 끼어들지 않는다."""
    chunks = [make_chunk("c1", "ex_a", "삼성전자", {"계약금액": "1,000"})]
    out = make_ask(builder, registry, chunks, manifest=_HANMI_MANIFEST).run(
        "삼성전자의 단일판매ㆍ공급계약체결 공시에 기재된 계약금액은 얼마인가?")
    assert not out.existence.applicable
    assert "[전수 확인]" not in out.evidence_pack.prompt_text
