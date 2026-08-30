#!/usr/bin/env python3
"""주최측 6개 질의 유형을 본뜬 평가셋 초안 생성.

    python3 scripts/make_eval_suite.py [--out eval/suite_v1]

왜 이게 필요한가
----------------
`eval/gold_passages.jsonl` 314문항은 **검색·추출 Closed 한 유형**만 담고 있고,
표에서 뽑은 KV 를 기계 문형에 끼운 것이라 실제 심사 질의와 생김새가 다르다.
주최측 과제소개자료의 참고 질의 set 은 6유형이며 채점도 정확성 외에
근거 완전성·요구사항 충족·할루시네이션·추론 논리성·정보한계 대응을 본다.

이 스크립트가 하는 일 / 안 하는 일
----------------------------------
한다  : 코퍼스에 **실제로 존재하는** 재료만 골라 질문을 만들고, 정답이 들어있는
        공시 문서번호를 붙인다. 숫자 비교처럼 facts 층에서 기계적으로 확정
        가능한 답은 계산해서 채운다.
안 한다: Open 유형과 복합추론의 서술형 정답은 **만들지 않는다**. 공시를 읽어야
        하는 판단이라 지어내면 평가셋 자체가 오염된다. `answer=null` +
        `answer_source="human_todo"` 로 두고 검수 대상임을 명시한다.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

FUND_TYPES = {
    "유상증자결정", "전환사채권발행결정", "신주인수권부사채권발행결정",
    "교환사채권발행결정", "상각형조건부자본증권발행결정",
    "자본으로인정되는채무증권발행결정",
}


def norm_report(name: str) -> str:
    return re.sub(r"\s*\(.*?\)\s*", "", name).replace("[기재정정]", "").replace("[첨부추가]", "").strip()


def subtype(name: str) -> str | None:
    m = re.search(r"주요사항보고서\s*\((.+?)\)", name)
    return m.group(1) if m else None




_NUMBERING = re.compile(r"^\s*(?:[가-힣]\.|[0-9]+[.)]|\([0-9가-힣]+\)|[①-⑳])\s*")


def clean_key(key: str) -> str:
    """항목명 앞의 '가.' '1)' '(2)' 같은 번호를 떼고 읽기 좋게 만든다."""
    out = key.strip()
    for _ in range(3):
        new = _NUMBERING.sub("", out)
        if new == out:
            break
        out = new
    return re.sub(r"\s+", " ", out).strip()


def has_batchim(word: str) -> bool:
    """마지막 글자에 받침이 있는가. 한글이 아니면 있는 것으로 친다."""
    ch = word.strip()[-1] if word.strip() else ""
    if not ("가" <= ch <= "힣"):
        return True
    return (ord(ch) - 0xAC00) % 28 != 0


def josa(word: str, no_bat: str, bat: str) -> str:
    """조사를 고른다. 인자 순서는 항상 (받침없을때, 받침있을때).

    '와/과' 는 받침 없는 쪽이 먼저지만 '이/가' 는 반대라서, 한 문자열에 슬래시로
    적는 방식은 순서를 헷갈리기 쉽다(실제로 틀렸다). 그래서 두 인자로 분리한다.
    """
    return bat if has_batchim(word) else no_bat


def fmt(n: float) -> str:
    return f"{n:,.0f}" if abs(n - round(n)) < 1e-9 else f"{n:,.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--facts", default="artifacts_v2/facts/facts.sqlite")
    ap.add_argument("--out", default="eval/suite_v1")
    args = ap.parse_args()

    docs = [json.loads(l) for l in open(Path(args.corpus) / "manifest.jsonl", encoding="utf-8") if l.strip()]
    by_co: dict[str, dict[str, list]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in docs:
        by_co[d["corp_name"]][norm_report(d["report_nm"])].append(d)
    sector_of = {d["corp_name"]: (d.get("sector") or d.get("industry") or "") for d in docs}

    db = sqlite3.connect(args.facts)
    db.row_factory = sqlite3.Row
    out: list[dict] = []

    def add(t, mode, query, gold_docs, answer, source, checks, notes=""):
        out.append({
            "id": f"S{len(out)+1:03d}",
            "task_type": t, "mode": mode, "query": query,
            "gold_doc_ids": [d["doc_id"] for d in gold_docs],
            "gold_reports": [f'{d["corp_name"]} / {d["report_nm"]} / {d["rcept_dt"]}' for d in gold_docs],
            "answer": answer, "answer_source": source,
            "check_points": checks, "notes": notes,
        })

    doc_of = {d["doc_id"]: d for d in docs}

    # ── 유형1: 검색·추출 Closed — facts 층에서 정답 확정 ────────────────────
    rows = db.execute("""
        SELECT company, key, value_text, value_num, value_unit, doc_id, report_name, filing_date
        FROM facts WHERE value_num IS NOT NULL AND is_latest=1
          AND key_norm IN ('계약금액','순자산액','자기주식취득금액한도','투자금액','자본금의액')
        GROUP BY company, key_norm ORDER BY value_num DESC LIMIT 60
    """).fetchall()
    seen, used_keys = set(), collections.Counter()
    for r in rows:
        if r["company"] in seen or r["doc_id"] not in doc_of:
            continue
        # 항목이 한 종류로 쏠리면 '검색·추출' 유형을 대표하지 못한다
        if used_keys[r["key"]] >= 2:
            continue
        seen.add(r["company"]); used_keys[r["key"]] += 1
        _k = clean_key(r["key"])
        add("검색·정보추출", "closed",
            f'{r["company"]}의 {doc_of[r["doc_id"]]["report_nm"]}에 기재된 '
            f'{_k}{josa(_k, "는", "은")} 얼마인가?',
            [doc_of[r["doc_id"]]], r["value_text"], "auto_facts",
            ["수치 정확성", "근거 공시 표시(접수번호/보고서명)", "단위 표기"])
        if len(seen) >= 6:
            break

    # ── 유형3: 비교·연산 Closed — 같은 항목·같은 섹터 두 기업, 답을 계산 ───
    # 질문을 '최대 계약금액'으로 특정한다. 그냥 "공시된 계약금액"이라고 물으면
    # 한 기업이 같은 항목을 15건씩 공시한 경우 어느 건을 말하는지 정해지지 않아
    # 맞는 답도 오답으로 채점된다(2026-08-23 정답셋 오염과 같은 유형의 실수).
    KEY_CTX = {
        "계약금액": ("단일판매·공급계약", "최대 계약금액"),
        "투자금액": ("신규시설투자 등", "최대 투자금액"),
        "최근매출액": ("공시에 기재된 최근매출액", "최근매출액"),
    }
    pairs_made = 0
    for key in ("계약금액", "투자금액", "최근매출액"):
        ctx, label = KEY_CTX[key]
        rs = db.execute("""
            SELECT f.company, f.value_text, f.value_num, f.doc_id, f.report_name,
                   (SELECT COUNT(*) FROM facts x WHERE x.key_norm=f.key_norm
                      AND x.company=f.company AND x.value_num IS NOT NULL AND x.is_latest=1) AS n_all
            FROM facts f
            JOIN (SELECT company, MAX(value_num) mx FROM facts
                  WHERE key_norm=? AND value_num IS NOT NULL AND is_latest=1
                  GROUP BY company) m
              ON m.company=f.company AND m.mx=f.value_num
            WHERE f.key_norm=? AND f.is_latest=1
            GROUP BY f.company ORDER BY f.value_num DESC LIMIT 40
        """, (key, key)).fetchall()
        bucket: dict[str, list] = collections.defaultdict(list)
        for r in rs:
            if r["doc_id"] in doc_of:
                bucket[sector_of.get(r["company"], "")].append(r)
        for sec, rs2 in bucket.items():
            if len(rs2) < 2 or pairs_made >= 8:
                continue
            a, b = rs2[0], rs2[1]
            win = a["company"] if a["value_num"] >= b["value_num"] else b["company"]
            add("다중조회·비교·연산", "closed",
                f'{a["company"]}{josa(a["company"], "와", "과")} {b["company"]} 중, 각각 공시한 '
                f'{ctx} 가운데 {label}은 얼마이며 더 큰 쪽은 어느 기업인가?',
                [doc_of[a["doc_id"]], doc_of[b["doc_id"]]],
                f'{win} ({a["company"]} {a["value_text"]} vs {b["company"]} {b["value_text"]})',
                "auto_compare",
                ["비교 결론 정확성", "양쪽 수치 모두 제시", "근거 공시 2건 모두 표시",
                 "단위·기준시점 혼동 없음"],
                f'섹터: {sec} · 해당 항목 공시 건수 {a["company"]} {a["n_all"]}건 / '
                f'{b["company"]} {b["n_all"]}건 (최댓값 기준으로 물었음)')
            pairs_made += 1

    # ── 유형5: 복합추론 Closed — 체결 후 해지 존재 여부 (양성/음성 모두) ───
    pos = [(c, v) for c, v in by_co.items() if v.get("단일판매ㆍ공급계약해지")]
    for c, v in pos[:5]:
        add("복합문서추론", "closed",
            f"{c}{josa(c, chr(44032), chr(51060))} 체결한 단일판매·공급계약 중 이후 해지된 계약이 존재하는가? 존재한다면 어떤 계약인가?",
            v["단일판매ㆍ공급계약해지"], "예 (해지 공시 존재)", "auto_meta",
            ["존재 여부 정확성", "해지된 계약의 상대방·금액 특정", "해지 공시를 근거로 표시",
             "원 계약 체결 공시와 연결"],
            f'체결 {len(v.get("단일판매ㆍ공급계약체결", []))}건 / 해지 {len(v["단일판매ㆍ공급계약해지"])}건. '
            "해지 사유·대상 계약은 사람이 확인해 채울 것")
    neg = [(c, v) for c, v in by_co.items()
           if v.get("단일판매ㆍ공급계약체결") and not v.get("단일판매ㆍ공급계약해지")]
    for c, v in neg[:3]:
        add("복합문서추론", "closed",
            f"{c}{josa(c, chr(44032), chr(51060))} 체결한 단일판매·공급계약 중 이후 해지된 계약이 존재하는가?",
            v["단일판매ㆍ공급계약체결"][:2], "아니오 (해지 공시 없음)", "auto_meta",
            ["없다고 정확히 답하는가", "**없는 사실을 지어내지 않는가(할루시네이션)**",
             "확인 범위(보유 공시 기준)를 밝히는가"],
            "음성 사례 — 정보한계 대응·할루시네이션 채점용")

    # ── 유형6-a: 복합추론 Open — 정정 체인에서 무엇이 바뀌었나 ─────────────
    cor = [json.loads(l) for l in __import__("gzip").open(
        "artifacts_v2/l1/corrections.jsonl.gz", "rt", encoding="utf-8") if l.strip()]
    groups: dict[str, list] = collections.defaultdict(list)
    for c in cor:
        groups[c["correction_group_id"]].append(c)
    chains = [g for g in groups.values() if len(g) >= 3]
    chains.sort(key=lambda g: -len(g))
    for g in chains[:4]:
        g = sorted(g, key=lambda x: int(x["correction_order"]))
        ds = [doc_of[x["doc_id"]] for x in g if x["doc_id"] in doc_of]
        if len(ds) < 2:
            continue
        add("복합문서추론", "open",
            f'{ds[0]["corp_name"]}의 {norm_report(ds[0]["report_nm"])} 공시가 정정된 내역이 있는가? '
            "있다면 최초 공시와 최종 정정본 사이에 무엇이 달라졌는지 설명해줘.",
            ds, None, "human_todo",
            ["정정 사실을 인지하는가", "**최신 정정본 수치를 답하는가(원본 수치를 그대로 답하면 오답)**",
             "변경 항목을 특정하는가", "정정 전/후 공시를 모두 근거로 표시하는가"],
            f"정정 체인 {len(ds)}단계. 달라진 항목은 사람이 대조해 채울 것")

    # ── 유형6-b: 사업보고서 다년 비교 Open ────────────────────────────────
    multi = []
    for c, v in by_co.items():
        ys = sorted({d["rcept_dt"][:4] for d in v.get("사업보고서", [])})
        if len(ys) >= 3:
            multi.append((c, v, ys))
    for c, v, ys in multi[:3]:
        ds = sorted(v["사업보고서"], key=lambda d: d["rcept_dt"])
        add("복합문서추론", "open",
            f"{c}의 {ys[0]}년 사업보고서와 {ys[-1]}년 사업보고서를 비교했을 때 "
            "핵심 사업과 주요 재무지표가 어떻게 변화했는지 설명해줘.",
            [ds[0], ds[-1]], None, "human_todo",
            ["두 시점 모두 근거 제시", "변화 방향(증감) 정확성", "요구사항(핵심사업+재무지표) 누락 없음",
             "추론 논리성"],
            "서술형 — 사람이 요지 3~5개를 채점 기준으로 적을 것")

    # ── 유형4: 비교·연산 Open — 자금조달 유형별 정리 ──────────────────────
    fund_cases = []
    for c, v in by_co.items():
        per: dict[str, list] = collections.defaultdict(list)
        for name, ds in v.items():
            for d in ds:
                st = subtype(d["report_nm"])
                if st in FUND_TYPES:
                    per[d["rcept_dt"][:4]].append(d)
        for y, ds in per.items():
            if len({subtype(d["report_nm"]) for d in ds}) >= 2:
                fund_cases.append((c, y, ds))
    for c, y, ds in fund_cases[:3]:
        add("다중조회·비교·연산", "open",
            f"{c}{josa(c, chr(44032), chr(51060))} {y}년에 실시한 자금조달 내역을 유형별로 정리해줘.",
            ds, None, "human_todo",
            ["유형 누락 없음", "각 건의 금액·시점 정확성", "근거 공시 전건 표시", "합계 계산 시 연산 정확성"],
            f'유형: {sorted({subtype(d["report_nm"]) for d in ds})}')

    # ── 유형2: 검색·추출 Open — 투자계획/경영사항 정리 ────────────────────
    op = []
    for c, v in by_co.items():
        for nm in ("신규시설투자등", "투자판단관련주요경영사항"):
            for d in v.get(nm, []):
                op.append((c, nm, d))
    seen_co = set()
    for c, nm, d in op:
        if c in seen_co:
            continue
        seen_co.add(c)
        add("검색·정보추출", "open",
            f'{c}의 {d["rcept_dt"][:4]}년 {nm} 공시를 기준으로 주요 내용을 정리해줘.',
            [d], None, "human_todo",
            ["핵심 항목(금액·목적·기간) 누락 없음", "근거 공시 표시", "공시에 없는 내용 생성 금지"])
        if len(seen_co) >= 6:
            break

    # ── 저장 ──────────────────────────────────────────────────────────────
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_type = collections.Counter((r["task_type"], r["mode"]) for r in out)
    todo = sum(1 for r in out if r["answer_source"] == "human_todo")
    lines = ["# 6유형 평가셋 초안 (suite_v1)", "",
             f"총 **{len(out)}문항** · 자동 확정 정답 **{len(out)-todo}** · 사람 검수 필요 **{todo}**", "",
             "> 주최측 참고 질의 set(과제소개자료 07p)의 6유형을 본떴다. 실제 평가셋은 비공개이므로",
             "> 이건 **대리 지표**다. 절대점수보다 유형별 실패 패턴을 보는 데 쓴다.", "",
             "## 구성", "", "| 유형 | 모드 | 문항 |", "|---|---|---:|"]
    for (t, m), n in sorted(by_type.items()):
        lines.append(f"| {t} | {m} | {n} |")
    lines += ["", "## 검수 방법", "",
              "`answer` 가 `null` 인 문항은 아래 근거 공시를 열어 **채점 기준 3~5줄**을 적어주세요.",
              "정답 문장을 완성할 필요는 없고, '이 숫자·이 사실이 답변에 있어야 한다' 수준이면 됩니다.", ""]
    cur = None
    for r in out:
        head = f'{r["task_type"]} / {r["mode"]}'
        if head != cur:
            cur = head
            lines += [f"### {head}", ""]
        mark = "✅ 자동" if r["answer_source"] != "human_todo" else "⬜ **검수 필요**"
        lines += [f'**{r["id"]}** {mark}', "", f'> {r["query"]}', ""]
        if r["answer"]:
            lines.append(f'- 정답: `{r["answer"]}`')
        lines.append("- 근거 공시:")
        for g, i in zip(r["gold_reports"], r["gold_doc_ids"]):
            lines.append(f"    - {g}  `{i}`")
        lines.append("- 채점 포인트: " + " / ".join(r["check_points"]))
        if r["notes"]:
            lines.append(f'- 메모: {r["notes"]}')
        lines.append("")
    outp.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(out)}문항 -> {outp.with_suffix('.jsonl')} / {outp.with_suffix('.md')}")
    print(f"  자동 확정 {len(out)-todo} · 사람 검수 {todo}")
    for (t, m), n in sorted(by_type.items()):
        print(f"  {t:14s} {m:6s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
