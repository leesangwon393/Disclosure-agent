#!/usr/bin/env python3
"""회사별 "제N기 -> 연도" 오프셋을 코퍼스에서 계산한다.

## 왜 필요한가

공시 표는 연도 대신 기수로 적는다 — `제55기 1분기`, `제 54 기말`.
실측(2026-09-01) 청크 16,711개가 이 표기를 쓴다. 질문은 연도로 묻는데
근거는 기수로 적혀 있으면 모델이 연결을 못 짓는다.

## 왜 청크마다 추측하면 안 되나

한 청크에서 가장 큰 기수를 당기로 보고 환산하면 **일치율이 75.8%** 다
(21개사 전수, 전부 90% 미만). 당기가 안 나오고 전기만 나오는 표가 많기
때문이다. 4건 중 1건이 틀린 환산을 근거에 박으면 모델이 그걸 믿는다.

## 그래서 회사별로 한 번 정한다

`(그 청크의 최대 기수) - (보고 기준연도)` 의 **최빈값**을 회사별 오프셋으로
쓴다. 최빈값 자체는 정확하다 — 삼성전자는 793/924 가 -1968 이고 나머지는
전기·전전기만 나온 표다. 연도 = 기수 + |오프셋|.

    python3 scripts/build_fiscal_offsets.py --out config/fiscal_offsets.json
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
from pathlib import Path

_GI = re.compile(r"제\s?(\d{1,3})\s?기")
MIN_SAMPLES = 20            # 이보다 적으면 신뢰할 수 없어 넣지 않는다
MIN_AGREEMENT = 0.60        # 최빈값이 이 정도는 되어야 한다


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="artifacts_v2/l1/chunks.jsonl.gz")
    ap.add_argument("--out", default="config/fiscal_offsets.json")
    ap.add_argument("--limit", type=int, default=0, help="0 이면 전부")
    args = ap.parse_args()

    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    n = 0
    with gzip.open(args.chunks, "rt", encoding="utf-8") as f:
        for line in f:
            n += 1
            if args.limit and n > args.limit:
                break
            row = json.loads(line)
            company, period = row.get("company"), row.get("period") or ""
            if not company or len(period) < 4:
                continue
            text = row.get("raw_text") or ""
            found = [int(m.group(1)) for m in _GI.finditer(text)]
            found = [g for g in found if 1 <= g <= 200]
            if not found:
                continue
            votes[company][max(found) - int(period[:4])] += 1

    table: dict[str, dict] = {}
    skipped: list[str] = []
    for company, counter in votes.items():
        offset, count = counter.most_common(1)[0]
        total = sum(counter.values())
        if total < MIN_SAMPLES or count / total < MIN_AGREEMENT:
            skipped.append(company)
            continue
        table[company] = {"offset": offset, "samples": total,
                          "agreement": round(count / total, 4)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"note": "연도 = 기수 - offset  (offset 은 음수)", "companies": table,
         "skipped": sorted(skipped)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"청크 {n:,} 읽음 / 기수가 나온 회사 {len(votes)} / 저장 {len(table)} / 제외 {len(skipped)}")
    for company, info in sorted(table.items())[:8]:
        print(f"   {company:16s} 제N기 -> N{-info['offset']:+d}년  "
              f"({info['samples']}건, 일치 {info['agreement']:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
