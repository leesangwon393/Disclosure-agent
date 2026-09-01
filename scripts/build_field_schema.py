#!/usr/bin/env python3
"""ⓑ 공시유형별 표준항목 명세를 facts.sqlite 에서 자동 생성한다.

## 왜 자동 생성인가

"신규시설투자등이면 투자금액·투자목적이 있어야 한다" 같은 목록을 손으로 쓰면
(a) 빠뜨리고 (b) 실제 데이터에 없는 항목을 요구하게 된다. 그래서 **그 유형의
문서 몇 %에 그 항목이 실제로 등장하는가**로 정한다.

## 분류 규칙

    등장비율 >= --required-ratio (기본 0.80)  -> required
    0.20 <= 비율 <  0.80                      -> conditional
    비율 <  0.20                              -> optional

기본값 0.75 는 손으로 고른 값이 아니라 **실측 분포의 골짜기**다. 유형별로 비율을
정렬하면 항목이 두 덩어리로 갈리고 그 사이가 비어 있다:

    단일판매공급계약체결   99.9 … 81.6  |  50.9 …   (골 30.7%p)
    신규시설투자등        100.0 … 83.7  |  51.2 …   (골 32.5%p)
    투자판단관련주요경영사항 100.0 … 91.3 |  51.3 …   (골 40.0%p)
    주요사항보고서(자기주식취득결정)
                         100.0 …  92.3 |  69.2 …   (골 23.1%p)
    대량보유상황보고서      79.7 …  79.5 |  45.4 …   (골 34.1%p)

마지막 줄이 0.80 을 쓰면 안 되는 이유다 — 0.80 은 대량보유상황보고서(1,068건,
전체의 37%)의 상위 덩어리를 통째로 잘라 required 를 0개로 만든다. 0.75 는 위
다섯 유형 **전부에서 골짜기 안에** 떨어지며, 0.80 대비 바뀌는 것은 대량보유
2개 항목뿐이다(실측 확인).

반대로 0.95 로 올리면 단일판매의 `계약상대`(94.6%)·`계약금액`(91.9%) 이
required 에서 빠져 명세가 무의미해진다.

아래쪽 덩어리는 대부분 정정 관련 항목(정정사유·정정전·정정일자)이다 — 정정본에만
존재하므로 conditional 이 맞다.

각 유형의 실제 골짜기 위치(`max_gap`)를 결과에 함께 적어두므로, 임계값이
덩어리를 갈랐는지 골짜기에 떨어졌는지 사람이 나중에 검증할 수 있다.

## 표본이 적은 유형

문서 10건 미만인 유형은 required 를 **비운다**. 3건 중 3건에 있다고 해서
"항상 있다"고 말할 수 없기 때문이다(무결점 3회 관측의 신뢰구간은 사실상
0.37~1.0 이다). 잘못된 required 는 정답 가능한 질문을 거부하게 만들고, 그쪽이
제약을 안 거는 것보다 훨씬 비싸다 — 그래서 fail open 한다.

## 한계

facts 에 periodic(정기공시) 항목이 0건이다. 이 명세는 exchange/holding/major
2,913 문서만 덮는다. 사업보고서 질문에는 쓸 수 없다.

## 사용

    python3 scripts/build_field_schema.py
    python3 scripts/build_field_schema.py --facts artifacts_v2/facts/facts.sqlite \
                                          --out config/field_schema.json
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from disclosure_rag.agent.field_schema import (  # noqa: E402
    normalize_field_key,
    normalize_report_kind,
)

logger = logging.getLogger("build_field_schema")

# major 의 doc_subtype 은 전부 None 이라 report_name 을 쓴다. 그 이름은
# "주요사항보고서(자기주식취득결정)" 형태라 괄호 안이 실질 유형명이다 —
# 질문이 괄호 안만 언급하는 경우("자기주식취득결정 공시")도 잡아야 한다.
_PAREN = re.compile(r"\(([^()]{2,})\)")

# exchange 의 report_name 은 "회사/공시유형/(날짜)공시유형" 합성 문자열이다.
# doc_subtype 이 채워져 있으므로 그쪽을 우선 쓴다.


# --------------------------------------------------------------------------- 잡음 제거
#
# facts 추출기가 일부 공시에서 **값을 key 로** 잘못 잡는다. 회사합병결정 22건에서
# key 종수가 218개로 튀는데, 그 안에 `80,472원`, `경기도성남시…두산타워26층IR팀`,
# `두산로보틱스주식회사` 같은 것이 들어 있다. 이건 명세의 문제가 아니라 추출기의
# 결함이고, 여기서 고칠 수는 없으므로 **명세에 들어오지 않게만** 막는다.
# (추출기 자체는 별건으로 손봐야 한다 — 제외 건수를 로그로 남긴다.)

_DIGIT_RUN = re.compile(r"\d{3,}")          # "80,472원", "(기준일2023.12.31)", "…155분당…"
_CORP_SUFFIX = re.compile(r"(주식회사|\(주\)|㈜)")

# 2026-09-01: 정기공시 스키마를 처음 추가하면서 발견 — "SK텔레콤본사" 같은
# 회사명+사업장 라벨이 필드명 후보로 새어 들어와, 회사명으로 조회할 때 그
# 회사 자신이 "확인 안 된 필수항목"이 되어 (a) facts 조회가 주소·계열사
# 목록에 파묻히고 (b) 충분성 검사가 절대 못 채우는 항목 때문에 최대
# 재검색까지 다 돌아 300초 근처까지 느려지고 결국 거부로 끝난다.
# 전수조사(2026-09-01): 90개사·323종·3,557건. `_CORP_SUFFIX`는 "㈜"/"주식회사"가
# 붙은 것만 걸러서 "SK텔레콤"·"KT"·"NAVER"처럼 접미사 없는 이름은 통과했다.
_REPORT_TYPE_LABELS = {"사업보고서", "분기보고서", "반기보고서"}


def _load_company_names(registry_path: Path = Path("artifacts_v2/registry/entities.json")) -> tuple[str, ...]:
    """universe 회사명 + 별칭을 접두사 매칭용으로 로드한다. registry 가 없으면 빈 튜플(fail open)."""
    if not registry_path.exists():
        logger.warning("[SCHEMA] registry 없음(%s) — 회사명 오염 필터 없이 진행", registry_path)
        return ()
    reg = json.loads(registry_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for e in reg.get("entities", []):
        if "universe" not in e.get("types", []):
            continue
        for n in [e.get("canonical_name")] + e.get("aliases", []):
            n = normalize_field_key(n or "")
            if len(n) >= 2:
                names.add(n)
    # 긴 이름부터 검사해야 짧은 이름의 부분집합 오매칭을 피한다.
    return tuple(sorted(names, key=len, reverse=True))


def _is_parse_artifact(key: str, company_names: tuple[str, ...] = ()) -> bool:
    """항목명이 아니라 값(또는 회사명·문서유형 라벨)이 잘못 들어온 것인가."""
    if not key or len(key) > 60:
        return True
    if _DIGIT_RUN.search(key):
        return True
    if _CORP_SUFFIX.search(key):
        return True
    if key in _REPORT_TYPE_LABELS:
        return True
    for name in company_names:
        # 접두사 매칭(예: "SK텔레콤본사")뿐 아니라 완전 일치도 잡는다.
        # 2026-09-01: "SK텔레콤" 이 다른 회사 보고서의 특수관계자거래 표에서
        # 열 헤더로 쓰이며 **숫자 값을 가진 독립 키**로도 나타남을 발견했다
        # (예: SK텔레콤=61,974). 회사명 자체가 재무항목일 리는 없으므로
        # 길이가 같아도(완전 일치) 잡는다 — 원래는 접두사만 걸렀다.
        if key.startswith(name):
            return True
    return False


# 표의 구조 라벨. 실제 문서에 자주 나오지만 "답변에 있어야 하는 항목"은 아니다.
# 정답셋 정리 때도 `합계` 가 381회 등장해 오답을 만든 전력이 있다. 지우지 않고
# optional 로 낮춘다 — 조회 자체는 가능해야 하고, 추적도 남아야 한다.
_STRUCTURE_LABELS = {
    "항목", "소계", "합계", "계", "구분", "비고", "참석", "불참",
    "주1)", "주2)", "주3)", "기타",
}


def _kind_of(doc_group: str, doc_subtype: str | None, report_name: str | None) -> str:
    return normalize_report_kind(doc_subtype) if doc_subtype else normalize_report_kind(report_name)


def _search_terms(kind: str) -> list[str]:
    terms = [kind]
    for m in _PAREN.finditer(kind):
        inner = m.group(1)
        if len(inner) >= 3:
            terms.append(inner)
    return terms


def _assign_core_terms(kinds: dict) -> None:
    """자연어 질문에서 공시유형을 찾기 위한 '어간 + 동작어'를 데이터로 도출한다.

    질문은 공시 이름을 통째로 쓰지 않는다:

        공시 이름  : 단일판매공급계약해지
        실제 질문  : "체결한 단일판매·공급계약 중 이후 **해지**된 계약이 있는가?"

    이름이 통문자열로 등장하지 않으므로 부분 문자열 매칭이 전부 실패한다.
    정답셋 38문항 중 8문항(S015~S022)이 이 형태다.

    그래서 이름이 긴 공통 앞부분을 공유하는 유형들을 찾아, 그 앞부분을
    `core_terms`(어간), 나머지를 `action_terms`(동작어)로 쪼갠다:

        단일판매공급계약체결 ┐
        단일판매공급계약해지 ┘ -> core "단일판매공급계약" / action "체결","해지"

    손으로 쓴 목록이 아니라 실제 유형 이름들의 공통 접두사에서 나온다.
    """
    names = sorted(kinds)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            prefix = os.path.commonprefix([a, b]).rstrip("(")
            # 6자 미만이면 우연한 겹침이다. 괄호가 남아 있으면 이름이 중간에
            # 잘린 것이므로 버린다("주요사항보고서(자기주식" 같은 것).
            if len(prefix) < 6 or "(" in prefix:
                continue
            for kind in (a, b):
                action = kind[len(prefix):]
                if not action:
                    continue
                d = kinds[kind]
                d.setdefault("core_terms", [])
                d.setdefault("action_terms", [])
                if prefix not in d["core_terms"]:
                    d["core_terms"].append(prefix)
                if action not in d["action_terms"]:
                    d["action_terms"].append(action)


def _max_gap(ratios: list[float]) -> dict:
    """정렬된 비율 목록에서 가장 큰 낙차. 임계값 검증용 진단 정보."""
    best = {"gap": 0.0, "above": None, "below": None}
    for a, b in zip(ratios, ratios[1:]):
        if a - b > best["gap"]:
            best = {"gap": round(a - b, 4), "above": round(a, 4), "below": round(b, 4)}
    return best


def build(facts_path: Path, *, required_ratio: float = 0.75,
          conditional_ratio: float = 0.20, min_docs: int = 10,
          registry_path: Path = Path("artifacts_v2/registry/entities.json")) -> dict:
    company_names = _load_company_names(registry_path)
    logger.info("[SCHEMA] 회사명 오염 필터: %d개 회사명 로드", len(company_names))
    db = sqlite3.connect(facts_path)

    doc_kind: dict[str, tuple[str, str]] = {}
    for doc_id, group, subtype, name in db.execute(
        "SELECT DISTINCT doc_id, doc_group, doc_subtype, report_name FROM facts"
    ):
        doc_kind[doc_id] = (group or "?", _kind_of(group or "", subtype, name))

    # (유형, 항목) -> 그 항목이 등장한 문서 집합
    seen: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    n_docs: collections.Counter = collections.Counter()
    dropped: collections.Counter = collections.Counter()
    for doc_id, key_norm in db.execute("SELECT doc_id, key_norm FROM facts"):
        if doc_id not in doc_kind:
            continue
        _, kind = doc_kind[doc_id]
        key = normalize_field_key(key_norm)
        if _is_parse_artifact(key, company_names):
            dropped[kind] += 1
            continue
        seen[(kind, key)].add(doc_id)
    for group, kind in doc_kind.values():
        n_docs[kind] += 1
    db.close()

    by_kind: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for (kind, key), docs in seen.items():
        by_kind[kind][key] = len(docs) / n_docs[kind]

    group_of = {kind: group for group, kind in doc_kind.values()}

    kinds: dict[str, dict] = {}
    for kind, ratios in by_kind.items():
        ordered = sorted(ratios.items(), key=lambda kv: (-kv[1], kv[0]))
        enough = n_docs[kind] >= min_docs
        req, cond, opt = [], [], []
        for key, r in ordered:
            if key in _STRUCTURE_LABELS:
                opt.append(key)     # 표 구조 라벨 — required 로 올리지 않는다
                continue
            # 표본이 적은 유형은 비율이 신뢰할 수 없다(문서 3건이면 1건만 있어도
            # 33%, 3건 모두 있어도 "항상"의 근거가 못 된다). required 를 비워
            # 제약을 걸지 않고 전부 conditional 로 둔다 — fail open.
            if not enough:
                cond.append(key)
            elif r >= required_ratio:
                req.append(key)
            elif r >= conditional_ratio:
                cond.append(key)
            else:
                opt.append(key)
        kinds[kind] = {
            "doc_group": group_of.get(kind, "?"),
            "n_docs": n_docs[kind],
            "sufficient_data": enough,
            "dropped_artifact_facts": dropped.get(kind, 0),
            "search_terms": _search_terms(kind),
            "required": req,
            "conditional": cond,
            "optional": opt,
            "ratios": {k: round(v, 4) for k, v in ordered},
            "max_gap": _max_gap([r for _, r in ordered]),
        }

    _assign_core_terms(kinds)

    groups: collections.Counter = collections.Counter()
    for group, _kind in doc_kind.values():
        groups[group] += 1

    return {
        "built_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source": str(facts_path),
        "thresholds": {
            "required_ratio": required_ratio,
            "conditional_ratio": conditional_ratio,
            "min_docs": min_docs,
        },
        "coverage": {
            "n_docs_with_facts": len(doc_kind),
            "n_kinds": len(kinds),
            "doc_groups": dict(groups.most_common()),
            "note": ("facts 에 periodic(정기공시) 항목이 0건이다 — 이 명세는 "
                     "exchange/holding/major 만 덮는다. 사업보고서 질문의 "
                     "expected_fields 는 여기서 나오지 않는다."),
        },
        "kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1]["n_docs"])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="artifacts_v2/facts/facts.sqlite")
    ap.add_argument("--out", default="config/field_schema.json")
    ap.add_argument("--required-ratio", type=float, default=0.75)
    ap.add_argument("--conditional-ratio", type=float, default=0.20)
    ap.add_argument("--min-docs", type=int, default=10,
                    help="이 건수 미만인 유형은 required 를 비운다(표본 부족).")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    facts = Path(args.facts)
    if not facts.exists():
        logger.error("facts DB 없음: %s", facts)
        return 1

    payload = build(facts, required_ratio=args.required_ratio,
                    conditional_ratio=args.conditional_ratio, min_docs=args.min_docs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    cov = payload["coverage"]
    logger.info("문서 %d건 / 공시유형 %d종 -> %s", cov["n_docs_with_facts"], cov["n_kinds"], out)
    print()
    print(f"{'문서':>6} {'req':>4} {'cond':>5} {'opt':>4}  {'골짜기':>7}  공시유형")
    for kind, d in payload["kinds"].items():
        if d["n_docs"] < args.min_docs:
            continue
        gap = d["max_gap"]
        gap_s = f"{gap['above']:.2f}>{gap['below']:.2f}" if gap["above"] is not None else "-"
        print(f"{d['n_docs']:6d} {len(d['required']):4d} {len(d['conditional']):5d} "
              f"{len(d['optional']):4d}  {gap_s:>11}  {kind}")
    n_dropped = sum(d.get("dropped_artifact_facts", 0) for d in payload["kinds"].values())
    if n_dropped:
        worst = sorted(payload["kinds"].items(),
                       key=lambda kv: -kv[1].get("dropped_artifact_facts", 0))[:3]
        print(f"\n값이 key 로 잘못 추출된 fact {n_dropped}건 제외 (facts 추출기 결함): "
              + ", ".join(f"{k}({d['dropped_artifact_facts']})" for k, d in worst if d.get("dropped_artifact_facts")))
    skipped = [k for k, d in payload["kinds"].items() if d["n_docs"] < args.min_docs]
    if skipped:
        print(f"\n표본 부족({args.min_docs}건 미만)으로 required 를 비운 유형 {len(skipped)}종: "
              f"{', '.join(skipped[:6])}{' …' if len(skipped) > 6 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
