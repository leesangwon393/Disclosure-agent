"""검증 실패 후 재생성 루프 회귀 테스트.

핵심은 "재생성했다"가 아니라 **어느 쪽을 채택하는가** 이다.
상원 레포는 재생성본으로 무조건 덮어썼고, 실측에서 재생성 결과가
"확인할 수 없습니다" 가 되며 숫자가 사라져 지표만 통과했다.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from disclosure_rag.agent import ask as ask_mod
from disclosure_rag.agent.evidence import Citation, EvidencePack
from disclosure_rag.entity.entity_extractor import ExtractedEntities

RID = "periodic_20260310002820"


def _cit():
    return Citation(chunk_id="C1", report_id=RID, company="삼성전자",
                    report_name="사업보고서", filing_date="20260310",
                    section_path=[], is_correction=False, is_latest=True)


def _pack(prompt_text: str, *, citations=None):
    return EvidencePack(
        question="q", prompt_text=prompt_text,
        citations=list(citations if citations is not None else [_cit()]),
        tool_results_summary=[],
    )


PROFIT_PACK = _pack(f"[EVIDENCE 1]\n2025년 영업이익: 1,200\n2024년 영업이익: 1,000\nreport_id: {RID}\n")
RESCALE_PACK = _pack(f"[EVIDENCE 1]\n순자산액: 7,661,584,000,000\nreport_id: {RID}\n")
EMPTY_PACK = _pack("[USER QUESTION]\nq\n", citations=[])


@dataclass
class _Trace:
    entities: ExtractedEntities


def _wire(monkeypatch, pack):
    trace = _Trace(entities=ExtractedEntities(raw_query="q"))
    monkeypatch.setattr(ask_mod, "run_agent_loop", lambda *a, **k: trace)
    monkeypatch.setattr(ask_mod, "build_evidence_pack", lambda t: pack)


def _run(monkeypatch, answers: list[str]):
    calls: list[str | None] = []

    def fake_generate(client, evidence_pack, *, correction_note=None, **kwargs):
        calls.append(correction_note)
        return answers[min(len(calls) - 1, len(answers) - 1)]

    monkeypatch.setattr(ask_mod, "generate_answer", fake_generate)
    return ask_mod.ask(object(), [], "q", entity_extractor=object()), calls


@pytest.fixture
def profit(monkeypatch):
    _wire(monkeypatch, PROFIT_PACK)


def test_no_retry_when_validation_passes(profit, monkeypatch):
    good = f"1,000에서 1,200으로 20.0% 증가했습니다. 근거: {RID}"
    res, calls = _run(monkeypatch, [good])
    assert len(calls) == 1 and calls[0] is None, "통과한 답변을 다시 만들면 안 된다"
    assert res.validation.passed and res.remediation == []


def test_retry_is_accepted_when_it_improves(profit, monkeypatch):
    bad = "영업이익이 37.5% 증가했습니다."                       # 근거 없는 숫자 + 인용 없음
    good = f"1,000에서 1,200으로 20.0% 증가했습니다. 근거: {RID}"
    res, calls = _run(monkeypatch, [bad, good])
    assert len(calls) == 2 and calls[1] is not None, "재생성 지시가 전달돼야 한다"
    assert res.answer == good and res.validation.passed
    assert any("채택" in r for r in res.remediation)


def test_refusal_is_accepted_over_ungrounded_numbers(profit, monkeypatch):
    """근거로 뒷받침 안 되는 수치보다 "확인 불가"가 낫다 — 대회 채점 기준과 같다.

    상원 레포에서 이 교체가 문제였던 건 규칙이 틀려서가 아니라, validator 가
    문서 ID·연도를 '근거 없는 숫자'로 오탐해 **멀쩡한 답변**을 거부로 바꿨기
    때문이다. 오탐 쪽은 tests/test_validator.py 에서 막는다.
    """
    bad = f"영업이익이 37.5% 증가했습니다. 근거: {RID}"
    refusal = "제공된 근거로는 확인할 수 없습니다."
    res, calls = _run(monkeypatch, [bad, refusal])
    assert len(calls) == 2 and res.answer == refusal
    assert any("채택" in r for r in res.remediation)


def test_refusal_does_not_replace_an_equally_scored_answer(profit, monkeypatch):
    """검증 점수가 같으면 정보가 있는 원본을 유지한다 — 거부로 도망가지 않는다."""
    original = "1,000에서 1,200으로 20.0% 증가했습니다."   # 인용만 빠짐
    refusal = "제공된 근거로는 확인할 수 없습니다."
    res, calls = _run(monkeypatch, [original, refusal])
    assert len(calls) == 2 and res.answer == original
    assert any("기각" in r for r in res.remediation)


def test_correction_note_is_cause_specific(monkeypatch):
    """지시가 항상 '암산하지 마'로 고정되면 안 된다 — 원인이 단위 재환산이면 그걸 말한다."""
    _wire(monkeypatch, RESCALE_PACK)
    bad = f"순자산액은 7,661,584백만원입니다. 근거: {RID}"
    res, calls = _run(monkeypatch, [bad, bad])
    assert calls[1] is not None
    assert "단위" in calls[1] or "글자 그대로" in calls[1]


def test_no_evidence_does_not_trigger_a_hallucination_retry(monkeypatch):
    """근거 0건은 재작성으로 고칠 수 있는 게 아니다 — 지어내라고 시키면 안 된다."""
    _wire(monkeypatch, EMPTY_PACK)
    res, calls = _run(monkeypatch, ["제공된 근거로는 확인할 수 없습니다."])
    assert len(calls) == 1
    assert not res.validation.has_any_evidence and not res.validation.passed
