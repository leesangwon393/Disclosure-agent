#!/usr/bin/env python3
"""Rule-based Entity Registry 빌더."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from disclosure_rag.entity.entity_registry import build_entity_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DART 등장주체 레지스트리를 빌드합니다.")
    parser.add_argument("--corpus-root", type=Path, default=Path("corpus"))
    parser.add_argument("--facts-db", type=Path, default=Path("artifacts_v2/facts/facts.sqlite"))
    parser.add_argument("--chunks", type=Path, default=Path("artifacts_v2/l1/chunks.jsonl.gz"))
    parser.add_argument("--role-source", choices=("auto", "facts", "chunks"), default="auto")
    parser.add_argument("--include-subsidiaries", action=argparse.BooleanOptionalAction, default=True,
                        help="정기공시 청크의 종속기업 현황 표를 보완 스캔합니다.")
    parser.add_argument("--output", type=Path, default=Path("artifacts_v2/registry/entities.json"))
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True,
                        help="완료 기준(70/155/86/삼성전자 counterparty)을 강제합니다.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry, stats = build_entity_registry(
        corpus_root=args.corpus_root, facts_db=args.facts_db, chunks_path=args.chunks,
        role_source=args.role_source, include_subsidiaries=args.include_subsidiaries,
    )
    checks = {
        "universe_is_70": stats["universe"] == 70,
        "submitter_is_155": stats["submitter_raw"] == 155,
        "outside_submitter_is_86": stats["submitter_outside_universe_raw"] == 86,
        "samsung_electronics_is_counterparty": registry.contains("삼성전자", "counterparty"),
        "samsung_heavy_evidence_exists": any(
            item.role == "counterparty" and item.document_company == "삼성중공업"
            for item in (registry.resolve("삼성전자").evidence if registry.resolve("삼성전자") else [])
        ),
    }
    payload = registry.to_dict()
    payload["completion_checks"] = checks
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"output": str(args.output), "summary": registry.summary,
                      "build_stats": stats, "completion_checks": checks}, ensure_ascii=False, indent=2))
    if args.strict and not all(checks.values()):
        failed = ", ".join(key for key, ok in checks.items() if not ok)
        raise SystemExit(f"완료 기준 미달: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
