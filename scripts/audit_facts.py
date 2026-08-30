#!/usr/bin/env python3
"""periodic facts 정제 전/후와 기존 폼 facts 회귀를 한 번에 측정한다."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from disclosure_rag.facts.extractor import normalize_key  # noqa: E402


def _db_path(path: str | Path) -> Path:
    path = Path(path)
    return path / "facts.sqlite" if path.is_dir() else path


def _connect(path: str | Path) -> sqlite3.Connection:
    resolved = _db_path(path)
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return sqlite3.connect(f"file:{resolved.resolve()}?mode=ro", uri=True)


def stats(path: str | Path) -> dict:
    db = _connect(path)
    row = db.execute(
        "SELECT COUNT(*), COUNT(DISTINCT key_norm), COUNT(DISTINCT doc_id), "
        "COUNT(DISTINCT company), COUNT(value_num), COUNT(value_date) FROM facts"
    ).fetchone()
    top = db.execute(
        "SELECT key_norm, COUNT(*) n FROM facts GROUP BY key_norm "
        "ORDER BY n DESC, key_norm ASC LIMIT 15"
    ).fetchall()
    db.close()
    total, keys, docs, companies, numeric, dated = row
    return {
        "facts": total, "distinct_keys": keys, "documents": docs,
        "companies": companies, "numeric": numeric, "dated": dated,
        "numeric_pct": round(numeric / max(total, 1) * 100, 1),
        "top_keys": top,
    }


def load_periodic_gold(path: str | Path) -> list[dict]:
    return [
        row for row in (
            json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if row.get("doc_group") == "periodic"
    ]


def reproduce_gold(path: str | Path, gold_path: str | Path, *, partial: bool = False) -> dict:
    db = _connect(path)
    gold = load_periodic_gold(gold_path)
    available_docs = {row[0] for row in db.execute("SELECT DISTINCT doc_id FROM facts")}
    if partial:
        gold = [g for g in gold if set(g.get("gold_report_ids") or []) & available_docs]

    hits, misses = 0, []
    for item in gold:
        report_ids = item.get("gold_report_ids") or []
        if not report_ids:
            misses.append(item)
            continue
        marks = ",".join("?" for _ in report_ids)
        key_norm = normalize_key(item.get("key") or "")[0]
        rows = db.execute(
            f"SELECT value_text FROM facts WHERE doc_id IN ({marks}) AND key_norm = ?",
            [*report_ids, key_norm],
        ).fetchall()
        answers = {str(value).strip() for value in (item.get("answers") or [item.get("answer")])}
        if any(str(row[0]).strip() in answers for row in rows):
            hits += 1
        else:
            misses.append(item)
    db.close()
    return {
        "hits": hits, "total": len(gold),
        "misses": [{"id": x.get("id"), "key": x.get("key"),
                    "company": x.get("company")} for x in misses],
    }


def _fmt_delta(before: int, after: int) -> str:
    delta = after - before
    return f"{before:,} → {after:,} ({delta:+,})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="artifacts_v2/facts_periodic")
    ap.add_argument("--candidate", default="artifacts_v2/facts_periodic_v2")
    ap.add_argument("--gold", default="eval/gold_passages_clean.jsonl")
    ap.add_argument("--forms-baseline", default="artifacts_v2/facts")
    ap.add_argument("--forms-candidate", default="artifacts_v2/facts_forms_regression")
    ap.add_argument("--partial", action="store_true", help="candidate에 들어 있는 gold 문서만 측정")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    before, after = stats(args.baseline), stats(args.candidate)
    gold = reproduce_gold(args.candidate, args.gold, partial=args.partial)
    forms_before = stats(args.forms_baseline)
    forms_after = stats(args.forms_candidate)

    result = {
        "gold": gold, "periodic_before": before, "periodic_after": after,
        "forms_before": forms_before, "forms_after": forms_after,
    }
    print(f"1. 정답셋 재현       {gold['hits']}/{gold['total']}")
    if gold["misses"]:
        print(f"   누락: {gold['misses']}")
    print(f"2. 총 fact 건수      {_fmt_delta(before['facts'], after['facts'])}")
    print("3. 상위 항목 15종")
    for key, count in after["top_keys"]:
        print(f"   {count:>8,}  {key}")
    print(f"4. 숫자값 비율       {before['numeric_pct']:.1f}% → {after['numeric_pct']:.1f}%")
    print(f"5. 항목 종류         {_fmt_delta(before['distinct_keys'], after['distinct_keys'])}")
    unchanged = forms_before["facts"] == forms_after["facts"]
    print(f"6. 기존 서식 facts   {_fmt_delta(forms_before['facts'], forms_after['facts'])} "
          f"({'그대로' if unchanged else '변경됨'})")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    return 0 if not gold["misses"] and unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
