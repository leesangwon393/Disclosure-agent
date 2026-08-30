#!/usr/bin/env python3
"""HCX 가 왜 근거를 읽고도 답을 못 뽑는지 좁힌다. **인덱스를 안 올린다(수 초).**

    python3 scripts/diag_answer.py

검색·근거확장은 정상임이 확인됐다(diag_retrieval [2.5]). 정답이 컨텍스트
4,734자 지점에 명확히 있는데 "확인할 수 없습니다"가 나왔다.
그래서 조건을 하나씩 바꿔가며 어디서 갈리는지 본다.

  A. 정답 조각 1개만        -> 이것도 실패하면 프롬프트/모델 문제
  B. 정답 조각 + 방해 조각  -> A는 되는데 B가 실패하면 '방해 조각' 문제
  C. 시스템 프롬프트 완화   -> 지나치게 보수적인 지시가 원인인지
  D. thinking OFF          -> reasoning 모드 영향인지
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SNAPSHOT = "artifacts_v2/l1/chunks.jsonl.gz"
ANSWER = "224,787,773,988,054"
QUESTION = "삼성전자의 주요사항보고서(자기주식취득결정)에 기재된 순자산액은 얼마인가?"

# diag_retrieval 이 실제로 넘긴 근거 9건의 chunk_id (순서 그대로)
EVIDENCE_IDS = [
    "major_20260129000003::main::C2",
    "major_20260107000715::main::C2",
    "major_20260318001062::main::C2",
    "major_20241118000171::main::C2",   # ★ 정답
    "major_20241118000328::main::C2",   # ★ 정답 (최신 유효본)
    "major_20241115000375::main::C2",   # ★ 정답 (원본)
    "periodic_20240312000736::main::P16::C51",
    "major_20260107000715::main::C3",
    "major_20250708000011::main::C2",
]

LOOSE_PROMPT = """당신은 금융공시(DART) 근거 기반 답변 생성기입니다.

아래 [EVIDENCE] 안에서 질문에 해당하는 항목을 찾아 그 값을 그대로 답하세요.
- 값이 여러 공시에 나오면 각각 공시일과 함께 모두 제시하세요.
- 근거에 정말로 없을 때만 "제공된 근거로는 확인할 수 없습니다"라고 답하세요.
- 답변 마지막 줄에 "근거: report_id(chunk_id)" 형식으로 사용한 근거를 나열하세요."""


def load_chunks(ids: list[str]) -> dict[str, dict]:
    want = set(ids)
    out: dict[str, dict] = {}
    with gzip.open(SNAPSHOT, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.startswith('{"chunk_id": "'):
                continue
            e = line.index('"', 14)
            cid = line[14:e]
            if cid in want:
                out[cid] = json.loads(line)
                if len(out) == len(want):
                    break
    return out


def build_prompt(chunks: list[dict]) -> str:
    lines = [f"[USER QUESTION]\n{QUESTION}\n"]
    for i, c in enumerate(chunks, 1):
        status = "정정본" if c.get("is_correction") else "원본"
        if c.get("is_latest"):
            status += " (최신)"
        lines.append(
            f"[EVIDENCE {i}]\n"
            f"회사: {c.get('company')}\n공시명: {c.get('report_name')}\n"
            f"공시일: {c.get('filing_date')}\n기간: {c.get('period')}\n"
            f"Section: {' > '.join(c.get('section_path') or [])}\n"
            f"정정 상태: {status}\n내용:\n{c.get('raw_text')}\n"
            f"report_id: {c.get('report_id')}\nchunk_id: {c.get('chunk_id')}\n"
        )
    return "\n".join(lines)


def ask(client, system: str, prompt: str, *, thinking=None) -> tuple[str, float]:
    t = time.time()
    kw = {"max_tokens": 800, "temperature": 0.2}
    if thinking is not None:
        kw["thinking"] = thinking
    msg = client.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}], **kw)
    return msg.get("content", ""), time.time() - t


_MANGLED = ("224조", "224 조")


def verdict(out: str) -> str:
    if ANSWER in out:
        return "✅ 정답(원문 표기 보존)"
    if any(m in out for m in _MANGLED):
        return "⚠️ 값은 찾았으나 숫자를 한글 단위로 바꿔 씀(자릿수 오류 위험)"
    if "확인할 수 없" in out or "확인되지 않" in out:
        return "❌ 확인불가 응답"
    return "⚠️ 다른 답"


def main() -> int:
    from disclosure_rag.agent.answer_generator import ANSWER_SYSTEM_PROMPT
    from disclosure_rag.agent.hcx_client import HCXClient

    print("스냅샷에서 근거 조각 읽는 중…", flush=True)
    got = load_chunks(EVIDENCE_IDS)
    ordered = [got[i] for i in EVIDENCE_IDS if i in got]
    print(f"  {len(ordered)}/{len(EVIDENCE_IDS)}건 로드")
    ans_ids = [c["chunk_id"] for c in ordered if ANSWER in (c.get("raw_text") or "")]
    print(f"  정답 보유 조각: {ans_ids}")

    c = HCXClient(timeout=60.0)
    cases = [
        ("A. 정답조각 1개 + 원래 프롬프트",
         [x for x in ordered if ANSWER in (x.get("raw_text") or "")][:1],
         ANSWER_SYSTEM_PROMPT, None),
        ("B. 근거 9건 전부 + 원래 프롬프트 (실패 재현)",
         ordered, ANSWER_SYSTEM_PROMPT, None),
        ("C. 근거 9건 전부 + 완화 프롬프트",
         ordered, LOOSE_PROMPT, None),
        ("D. 근거 9건 전부 + 원래 프롬프트 + thinking OFF",
         ordered, ANSWER_SYSTEM_PROMPT, {"effort": "none"}),
        ("E. 근거 9건 전부 + 완화 프롬프트 + thinking OFF",
         ordered, LOOSE_PROMPT, {"effort": "none"}),
    ]

    # F: 실제 서빙 경로(generate_answer) 그대로 — 고친 프롬프트 + thinking OFF 가
    #    기본값으로 들어갔는지 확인한다.
    from disclosure_rag.agent.answer_generator import generate_answer
    from disclosure_rag.agent.evidence import Citation, EvidencePack
    print(f"\n{'='*70}\nF. 실제 서빙 경로 generate_answer() — 근거 9건")
    pack = EvidencePack(
        question=QUESTION, prompt_text=build_prompt(ordered),
        citations=[Citation(chunk_id=c["chunk_id"], report_id=c.get("report_id") or "",
                            company=c.get("company"), report_name=c.get("report_name"),
                            filing_date=c.get("filing_date"),
                            section_path=c.get("section_path") or [],
                            is_correction=bool(c.get("is_correction")),
                            is_latest=c.get("is_latest")) for c in ordered],
        tool_results_summary=[])
    try:
        t0 = time.time(); out = generate_answer(c, pack); el = time.time() - t0
        print(f"  {verdict(out)}  ({el:.1f}초)")
        print("  " + (out or "(빈 응답)").replace("\n", "\n  ")[:700])
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ {type(e).__name__}: {str(e)[:200]}")

    for name, chunks, sysp, think in cases:
        if not chunks:
            print(f"\n{'='*70}\n{name}\n  (근거 없음 — 건너뜀)")
            continue
        prompt = build_prompt(chunks)
        has = ANSWER in prompt
        print(f"\n{'='*70}\n{name}")
        print(f"  근거 {len(chunks)}건 / 프롬프트 {len(prompt):,}자 / 정답 포함 {has}")
        try:
            out, el = ask(c, sysp, prompt, thinking=think)
        except Exception as e:  # noqa: BLE001
            print(f"  ❌ {type(e).__name__}: {str(e)[:200]}")
            continue
        print(f"  {verdict(out)}  ({el:.1f}초)")
        print("  " + (out or "(빈 응답)").replace("\n", "\n  ")[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
