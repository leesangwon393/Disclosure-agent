#!/usr/bin/env python3
"""평가셋 양산 — 주최측 6유형을 본떠 70개사 전체에서 만든다.

    python3 scripts/make_eval_suite_v2.py --out eval/suite_v2

## 왜 새로 만드나

지금 있는 두 셋이 각각 다른 이유로 부족하다.

    eval/gold_passages_clean.jsonl  286문항
        facts 추출기가 뽑은 (항목,값) 을 기계 문형에 끼운 것이다. 그래서
        **facts 가 답할 수 있는 형태의 질문만** 들어 있다. 이걸로 최적화하면
        "facts 로 만든 문제를 facts 로 푸니 잘 풀린다"를 발견하게 된다.
        유형도 '검색·추출 closed' 한 갈래뿐이다(periodic 65%).

    eval/suite_v1.jsonl              38문항
        대회 참고 질의 6유형을 본떠 만들어 문형은 맞다. 그런데 38문항이라
        유형당 3~8건이고, 회사도 10여 곳뿐이다. 여기서 나온 차이는 우연과
        구분되지 않는다.

이 스크립트는 **suite_v1 의 문형으로 gold_passages 의 규모를** 만든다.

## 정답을 지어내지 않는다

문항마다 정답의 출처를 `answer_source` 에 남긴다.

    auto_facts       facts 층에서 값이 유일하게 확정됨
    auto_compare     두 회사 최댓값을 계산해 비교 결론까지 확정
    auto_meta        manifest/정정그래프에서 존재 여부가 확정됨 (예/아니오)
    auto_diff        정정 최초본↔최종본 값 차이를 계산해 확정
    rubric_only      서술형. 정답 문장 없이 **채점 기준**만 준다
                     (open_scoring 이 항목 커버리지로 채점한다)

`rubric_only` 는 정답이 비어 있지만 버리는 문항이 아니다. `expected_fields`
기준으로 채점된다.

## 품질 관문을 생성기 안에 둔다

과거에 정답셋 결함 28건(모호 22 + 교차주체 6)을 사후 정리 스크립트로
걸러냈다. 사후 정리는 이미 오염된 걸 줄일 뿐이라, 여기서는 **애초에 안
만든다.** 거른 이유는 전부 리포트에 남는다.

    모호       같은 (회사,문서,항목)에 값이 둘 이상 -> 질문이 답을 특정 못 함
               periodic 은 이 비율이 61.4% 다(연결/별도 + 여러 표). 반드시 막는다
    구조라벨   '합계'·'소계'·'구분' 처럼 표 구조를 가리키는 항목명
               (한 문서에 '합계'가 66개 있는 사례 실측)
    교차주체   삼성중공업이 제출한 문서의 값을 삼성전자에게 묻는 것
    쏠림       한 회사·한 항목이 과반을 차지하지 않게 상한을 둔다
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import random
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logger = logging.getLogger("make_eval_suite_v2")

# ---------------------------------------------------------------- 상수

# 표 구조 라벨. 항목명처럼 보이지만 무엇을 묻는지 정해지지 않는다.
STRUCTURAL_KEYS = {
    "합계", "소계", "계", "구분", "기타", "비고", "항목", "총계", "당기", "전기",
    "당분기", "전분기", "당반기", "전반기", "기초", "기말", "당기말", "전기말",
    "주1)", "주2)", "주3)", "참석", "불참", "미국", "해외", "국내",
}

FUND_TYPES = {
    "유상증자결정", "전환사채권발행결정", "신주인수권부사채권발행결정",
    "교환사채권발행결정", "상각형조건부자본증권발행결정",
    "자본으로인정되는채무증권발행결정", "무상증자결정",
}

# 한 회사/한 항목이 셋을 지배하지 않게. **생성기마다 상한이 다르다.**
#
# 처음에 전 생성기에 같은 상한을 씌웠더니, 항목명이 상수인 생성기
# (해지·정정·자금조달)가 그 상수 하나로 상한에 걸려 12문항에서 멈췄다.
# 쓸 수 있는 재료가 훨씬 많은데 상한이 유형 자체를 잘라낸 것이다.
#
#   (회사 상한, 항목 상한)  — 항목 상한 None = 제한 없음
CAPS: dict[str, tuple[int, int | None]] = {
    "lookup_form":     (3, 8),     # 항목이 680종이라 고르게 퍼진다
    "lookup_periodic": (3, 8),     # 항목이 4만종
    "summary":         (2, 24),    # 항목 = 공시유형(10여 종)이라 넉넉히
    "compare":         (4, 15),
    "funding":         (2, None),  # 항목이 '자금조달' 하나 -> 상한 두면 안 된다
    "termination":     (2, None),  # 항목이 '해지' 하나
    "correction":      (4, None),  # 항목이 '정정' 하나. 유형별로 다른 질문이 된다
}
DEFAULT_CAP = (6, 12)

_NUMBERING = re.compile(r"^\s*(?:[가-힣]\.|[0-9]+[.)]|\([0-9가-힣]+\)|[①-⑳])\s*")
_HANGUL2 = re.compile(r"[가-힣].*[가-힣]")


def norm_report(name: str) -> str:
    return (re.sub(r"\s*\(.*?\)\s*", "", name or "")
            .replace("[기재정정]", "").replace("[첨부추가]", "").strip())


def subtype_of(name: str) -> str:
    m = re.search(r"주요사항보고서\s*\((.+?)\)", name or "")
    return (m.group(1) if m else norm_report(name)).replace(" ", "")


def clean_key(key: str) -> str:
    out = (key or "").strip()
    for _ in range(3):
        new = _NUMBERING.sub("", out)
        if new == out:
            break
        out = new
    return out.strip()


def has_batchim(word: str) -> bool:
    if not word:
        return False
    ch = word[-1]
    return "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 != 0


def josa(word: str, no_bat: str, bat: str) -> str:
    return bat if has_batchim(word) else no_bat


def fmt(n: float) -> str:
    return f"{n:,.0f}" if abs(n - round(n)) < 1e-9 else f"{n:,.2f}"


def period_label(doc: dict) -> str:
    """'사업보고서 (2025.12)' 처럼 사람이 읽는 기간 표기."""
    base = doc.get("base_year") or ""
    month = doc.get("base_month") or ""
    if base and month:
        return f"({base}.{str(month).zfill(2)})"
    return f"({doc['rcept_dt'][:4]}.{doc['rcept_dt'][4:6]})"


# ---------------------------------------------------------------- 수집기


@dataclass
class Suite:
    """생성 결과와 **왜 버렸는지**를 함께 들고 다닌다."""
    rows: list[dict] = field(default_factory=list)
    rejected: collections.Counter = field(default_factory=collections.Counter)
    per_company: collections.Counter = field(default_factory=collections.Counter)
    per_key: collections.Counter = field(default_factory=collections.Counter)
    _queries: set = field(default_factory=set)

    def reject(self, reason: str) -> None:
        self.rejected[reason] += 1

    def room_for(self, company: str, key: str, *, tag: str) -> bool:
        """한 회사·한 항목이 셋을 지배하지 않게 막는다. 상한은 생성기별이다."""
        max_company, max_key = CAPS.get(tag, DEFAULT_CAP)
        if self.per_company[(tag, company)] >= max_company:
            self.reject(f"{tag}:회사쏠림")
            return False
        if key and max_key is not None and self.per_key[(tag, key)] >= max_key:
            self.reject(f"{tag}:항목쏠림")
            return False
        return True

    def add(self, *, task_type: str, mode: str, query: str, gold_docs: list[dict],
            answer, answer_source: str, check_points: list[str],
            company: str = "", key: str = "", tag: str = "", notes: str = "",
            extra: dict | None = None) -> bool:
        # 최종 안전망. 문형이 겹치면 같은 질문에 다른 정답이 붙어, 맞는 답도
        # 오답으로 채점된다. 생성기마다 막는 것보다 여기서 한 번 더 본다.
        if query in self._queries:
            self.reject(f"{tag}:질문중복")
            return False
        self._queries.add(query)
        self.per_company[(tag, company)] += 1
        if key:
            self.per_key[(tag, key)] += 1
        row = {
            "id": f"G{len(self.rows) + 1:04d}",
            "task_type": task_type, "mode": mode, "query": query,
            "company": company,
            "doc_group": (gold_docs[0]["doc_group"] if gold_docs else ""),
            "gold_doc_ids": [d["doc_id"] for d in gold_docs],
            "gold_reports": [f'{d["corp_name"]} / {d["report_nm"]} / {d["rcept_dt"]}'
                             for d in gold_docs],
            "answer": answer,
            "answers": [answer] if isinstance(answer, str) and answer else [],
            "answer_source": answer_source,
            "check_points": check_points,
            "generator": tag,
            "notes": notes,
        }
        row.update(extra or {})
        self.rows.append(row)
        return True


# ---------------------------------------------------------------- 품질 관문


def is_bad_key(key: str) -> str | None:
    """항목명으로 쓸 수 없으면 이유를 돌려준다."""
    k = clean_key(key)
    if not k:
        return "빈항목명"
    if k in STRUCTURAL_KEYS:
        return "구조라벨"
    # 값이 항목명 자리에 들어온 경우를 먼저 본다. 한글 검사보다 앞에 둬야
    # 리포트에 '한글2자미만'이 아니라 진짜 원인이 남는다(추출기 결함 추적용).
    if re.search(r"\d{3,}", k):
        return "값이항목명"
    if not _HANGUL2.search(k):
        return "한글2자미만"
    if len(k) > 40:
        return "항목명과장"
    return None


def value_is_distinctive(db: sqlite3.Connection, doc_id: str, key: str, value: str) -> bool:
    """같은 문서 안에서 **다른 항목이 같은 값**을 갖고 있으면 질문이 못 쓴다.

    예: "주당액면가액 5,000" 인데 그 문서에 5,000 인 항목이 여럿이면, 모델이
    엉뚱한 항목을 보고 답해도 채점기가 정답 처리한다(채점은 숫자 일치로 본다).
    답이 '우연히 맞는' 문항은 성능을 부풀린다.
    """
    n = db.execute(
        "SELECT COUNT(DISTINCT key_norm) FROM facts WHERE doc_id=? AND value_text=?",
        (doc_id, value)).fetchone()[0]
    return n <= 1


def unique_value(db: sqlite3.Connection, company: str, doc_id: str, key: str) -> str | None:
    """(회사, 문서, 항목) 에 값이 **정확히 하나**일 때만 그 값을 돌려준다.

    periodic 은 이 조건을 어기는 비율이 61.4% 다(연결/별도 재무제표 + 같은
    항목이 여러 표에 등장). 값이 둘 이상이면 질문이 답을 특정하지 못하므로
    맞는 답도 오답으로 채점된다 — 과거 정답셋 오염 22건의 원인이다.
    """
    rows = db.execute(
        "SELECT DISTINCT value_text FROM facts "
        "WHERE company=? AND doc_id=? AND key_norm=? AND value_num IS NOT NULL",
        (company, doc_id, key)).fetchall()
    return rows[0][0] if len(rows) == 1 else None


# ---------------------------------------------------------------- 유형 1
# 검색·정보추출 / closed — "X의 <공시>에 기재된 <항목>은 얼마인가?"


def gen_lookup(suite: Suite, db: sqlite3.Connection, doc_of: dict, *,
               tag: str, doc_groups: tuple[str, ...], target: int, rng: random.Random) -> None:
    """facts 에서 값이 유일하게 확정되는 (회사, 문서, 항목)만 질문으로 만든다."""
    same_name_docs: collections.Counter = collections.Counter()
    for d in doc_of.values():
        same_name_docs[(d["corp_name"], d["report_nm"])] += 1

    placeholders = ",".join("?" * len(doc_groups))
    rows = db.execute(f"""
        SELECT company, doc_id, key_norm, COUNT(DISTINCT value_text) n
        FROM facts
        WHERE value_num IS NOT NULL AND doc_group IN ({placeholders})
        GROUP BY company, doc_id, key_norm
        HAVING n = 1
    """, doc_groups).fetchall()
    rng.shuffle(rows)

    made = 0
    for company, doc_id, key, _n in rows:
        if made >= target:
            break
        doc = doc_of.get(doc_id)
        if doc is None:
            suite.reject(f"{tag}:문서없음")
            continue
        # 교차주체 방지: 값이 실린 문서의 제출사와 질문의 회사가 같아야 한다.
        # (삼성중공업 문서의 값을 삼성전자에게 묻는 결함이 6건 있었다)
        if doc["corp_name"] != company:
            suite.reject(f"{tag}:교차주체")
            continue
        bad = is_bad_key(key)
        if bad:
            suite.reject(f"{tag}:{bad}")
            continue
        if not suite.room_for(company, key, tag=tag):
            continue
        value = unique_value(db, company, doc_id, key)
        if value is None:
            suite.reject(f"{tag}:값모호")
            continue
        if not value_is_distinctive(db, doc_id, key, value):
            suite.reject(f"{tag}:값중복")
            continue

        k = clean_key(key)
        # 같은 회사가 같은 이름의 공시를 여러 건 냈으면, 공시명만으로는 어느
        # 문서를 묻는지 정해지지 않는다 -> 답이 갈려 맞는 답도 오답이 된다.
        # 실측: 이 관문 없이 만들었더니 질문 17개가 서로 중복이었다.
        siblings = same_name_docs.get((company, doc["report_nm"]), 1)
        if doc["doc_group"] == "periodic":
            label = f'{norm_report(doc["report_nm"])} {period_label(doc)}'
        elif siblings > 1:
            label = f'{doc["rcept_dt"][:4]}년 {doc["rcept_dt"][4:6]}월 {doc["report_nm"]}'
        else:
            label = doc["report_nm"]
        suite.add(
            task_type="검색·정보추출", mode="closed",
            query=f'{company}의 {label}에 기재된 {k}{josa(k, "는", "은")} 얼마인가?',
            gold_docs=[doc], answer=value, answer_source="auto_facts",
            check_points=["수치 정확성", "단위 표기", "근거 공시 표시(접수번호/보고서명)"],
            company=company, key=key, tag=tag,
            notes=f"항목 {k} / 문서 내 유일값 확인됨",
            extra={"key": k, "expected_fields_hint": [k]},
        )
        made += 1


# ---------------------------------------------------------------- 유형 2
# 검색·정보추출 / open — "X의 <연도> <공시유형> 공시 주요 내용을 정리해줘"


def gen_summary(suite: Suite, docs: list[dict], schema, *, target: int,
                rng: random.Random) -> None:
    """서술형. 정답 문장 대신 **다뤄야 할 항목 목록**을 채점 기준으로 준다.

    Field Schema 가 그 공시유형의 required 항목을 갖고 있어야 채점이 가능하다.
    없으면 만들지 않는다 — 기준 없는 문항은 채점할 수 없다.
    """
    tag = "summary"
    pool = [d for d in docs if d["doc_group"] in ("exchange", "major")]
    rng.shuffle(pool)
    made = 0
    for doc in pool:
        if made >= target:
            break
        kind = subtype_of(doc["report_nm"])
        fields = schema.required(kind) if schema else []
        if len(fields) < 3:
            suite.reject(f"{tag}:채점기준부족")
            continue
        if not suite.room_for(doc["corp_name"], kind, tag=tag):
            continue
        year = doc["rcept_dt"][:4]
        suite.add(
            task_type="검색·정보추출", mode="open",
            query=f'{doc["corp_name"]}의 {year}년 {norm_report(doc["report_nm"])} '
                  f'공시를 기준으로 주요 내용을 정리해줘.',
            gold_docs=[doc], answer=None, answer_source="rubric_only",
            check_points=["요구 항목을 모두 다루는가",
                          "확인되지 않은 항목을 밝히는가(조용히 빼면 안 됨)",
                          "근거 공시 표시", "근거에 없는 내용을 덧붙이지 않는가"],
            company=doc["corp_name"], key=kind, tag=tag,
            notes=f"공시유형 {kind} / 채점 항목 {len(fields)}개",
            extra={"expected_fields_hint": fields},
        )
        made += 1


# ---------------------------------------------------------------- 유형 3
# 다중조회·비교·연산 / closed — "A와 B 중 최대 <항목>은? 더 큰 쪽은?"

COMPARE_KEYS = {
    "계약금액": "단일판매·공급계약 가운데 최대 계약금액",
    "투자금액": "신규시설투자등 가운데 최대 투자금액",
    "순자산액": "주요사항보고서에 기재된 순자산액",
    "자기주식취득금액한도": "주요사항보고서에 기재된 자기주식 취득금액 한도",
    "최근매출액": "공시에 기재된 최근매출액",
}


def gen_compare(suite: Suite, db: sqlite3.Connection, doc_of: dict, *,
                target: int, rng: random.Random) -> None:
    """회사별 최댓값을 계산해 비교 결론까지 확정한다.

    '최대'로 특정하는 이유: 그냥 "공시된 계약금액"이라고 물으면 한 회사가 같은
    항목을 15건씩 공시한 경우 어느 건인지 정해지지 않아 맞는 답도 오답이 된다.
    """
    tag = "compare"
    made = 0
    for key, phrase in COMPARE_KEYS.items():
        rows = db.execute("""
            SELECT f.company, f.value_text, f.value_num, f.doc_id
            FROM facts f
            JOIN (SELECT company, MAX(value_num) mx FROM facts
                  WHERE key_norm=? AND value_num IS NOT NULL AND is_latest=1
                  GROUP BY company) m
              ON m.company=f.company AND m.mx=f.value_num
            WHERE f.key_norm=? AND f.value_num IS NOT NULL AND f.is_latest=1
            GROUP BY f.company
        """, (key, key)).fetchall()
        cands = [r for r in rows if r[3] in doc_of and doc_of[r[3]]["corp_name"] == r[0]]
        rng.shuffle(cands)
        for a, b in zip(cands[0::2], cands[1::2]):
            if made >= target:
                return
            if abs(a[2] - b[2]) < 1e-9:
                # 두 값이 같으면 "더 큰 쪽"이 없다 — 답할 수 없는 질문이다.
                suite.reject(f"{tag}:동점")
                continue
            if not (suite.room_for(a[0], key, tag=tag) and suite.room_for(b[0], key, tag=tag)):
                continue
            winner = a if a[2] > b[2] else b
            k = clean_key(key)
            suite.add(
                task_type="다중조회·비교·연산", mode="closed",
                query=f'{a[0]}{josa(a[0], "와", "과")} {b[0]} 중, 각각 공시한 {phrase}은 '
                      f'얼마이며 더 큰 쪽은 어느 기업인가?',
                gold_docs=[doc_of[a[3]], doc_of[b[3]]],
                answer=f'{winner[0]} ({a[0]} {fmt(a[2])} vs {b[0]} {fmt(b[2])})',
                answer_source="auto_compare",
                check_points=["비교 결론 정확성", "양쪽 수치 모두 제시",
                              "근거 공시 2건 모두 표시", "단위·기준시점 혼동 없음"],
                company=f"{a[0]}|{b[0]}", key=key, tag=tag,
                notes=f"{k} 최댓값 비교",
                extra={"compare_values": {a[0]: a[1], b[0]: b[1]}, "key": k,
                       "expected_fields_hint": [k]},
            )
            made += 1


# ---------------------------------------------------------------- 유형 4
# 다중조회·비교·연산 / open — "X가 <연도>에 실시한 자금조달 내역을 유형별로"


def gen_funding(suite: Suite, docs: list[dict], *, target: int, rng: random.Random) -> None:
    tag = "funding"
    by_year: dict[tuple, list] = collections.defaultdict(list)
    for d in docs:
        if subtype_of(d["report_nm"]) in FUND_TYPES:
            by_year[(d["corp_name"], d["rcept_dt"][:4])].append(d)
    # 자금조달 건이 2건 이상이면 '정리'가 성립한다. 다만 **유형이 1종뿐이면
    # "유형별로"라고 묻지 않는다** — 답이 한 덩어리인데 유형별 분류를 요구하면
    # 질문 자체가 성립하지 않는다. 실측: (회사,연도) 34조합 중 2종 이상은
    # 3개뿐이다. 질문을 데이터에 맞춘다.
    cands = [(k, v) for k, v in by_year.items() if len(v) >= 2]
    rng.shuffle(cands)
    for (company, year), group in cands[:target]:
        if not suite.room_for(company, "", tag=tag):
            continue
        kinds = sorted({subtype_of(d["report_nm"]) for d in group})
        multi = len(kinds) >= 2
        how = "유형별로 정리해줘" if multi else "정리해줘"
        checks = ["자금조달 건을 빠짐없이 다루는가", "각 건의 금액·시기를 제시하는가",
                  "근거 공시를 모두 표시하는가", "해당 연도 외의 건을 섞지 않는가"]
        if multi:
            checks[0] = f"자금조달 유형 {len(kinds)}종을 모두 다루는가"
        suite.add(
            task_type="다중조회·비교·연산", mode="open",
            query=f'{company}{josa(company, "가", "이")} {year}년에 실시한 자금조달 '
                  f'내역을 {how}.',
            gold_docs=group, answer=None, answer_source="rubric_only",
            check_points=checks,
            company=company, key="자금조달", tag=tag,
            notes=f"{year}년 {len(group)}건 / 유형 {kinds}"
                  + ("" if multi else " — 단일 유형이라 '유형별' 표현을 빼고 물었다"),
            extra={"expected_kinds": kinds, "expected_fields_hint": kinds},
        )


# ---------------------------------------------------------------- 유형 5
# 복합문서추론 / closed — "체결한 계약 중 이후 해지된 것이 있는가?"


def gen_termination(suite: Suite, docs: list[dict], *, target: int,
                    rng: random.Random) -> None:
    """존재 여부가 manifest 로 확정되는 예/아니오 문항.

    **음성 사례(해지 없음)를 반드시 섞는다.** 양성만 넣으면 "있습니다"라고
    항상 답해도 만점이 나와, 없는 사실을 지어내는 실패를 못 잡는다.
    """
    tag = "termination"
    by_co: dict[str, dict[str, list]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in docs:
        by_co[d["corp_name"]][norm_report(d["report_nm"])].append(d)

    positives, negatives = [], []
    for company, kinds in by_co.items():
        signed = [d for k, v in kinds.items() if "단일판매" in k and "해지" not in k for d in v]
        ended = [d for k, v in kinds.items() if "해지" in k for d in v]
        if not signed:
            continue
        (positives if ended else negatives).append((company, signed, ended))
    rng.shuffle(positives)
    rng.shuffle(negatives)

    # 양성/음성을 반반. 음성이 적으면 있는 만큼만 쓰되 비율을 기록한다.
    half = target // 2
    picked = positives[:half] + negatives[:target - half]
    for company, signed, ended in picked:
        if not suite.room_for(company, "", tag=tag):
            continue
        exists = bool(ended)
        gold = (ended + signed[:1]) if exists else signed[:2]
        suite.add(
            task_type="복합문서추론", mode="closed",
            query=f'{company}{josa(company, "가", "이")} 체결한 단일판매·공급계약 중 '
                  f'이후 해지된 계약이 존재하는가? 존재한다면 어떤 계약인가?',
            gold_docs=gold,
            answer="예 (해지 공시 존재)" if exists else "아니오 (해지 공시 없음)",
            answer_source="auto_meta",
            check_points=(["존재 여부 정확성", "해지된 계약의 상대방·금액 특정",
                           "해지 공시를 근거로 표시", "원 계약 체결 공시와 연결"]
                          if exists else
                          ["없다고 정확히 답하는가",
                           "없는 사실을 지어내지 않는가(할루시네이션)",
                           "확인 범위(보유 공시 기준)를 밝히는가"]),
            company=company, key="해지", tag=tag,
            notes=f"체결 {len(signed)}건 / 해지 {len(ended)}건"
                  + ("" if exists else " — 음성 사례(할루시네이션 채점용)"),
            extra={"polarity": "positive" if exists else "negative"},
        )


# ---------------------------------------------------------------- 유형 6
# 복합문서추론 / open — "정정 내역이 있는가? 무엇이 달라졌는가"


def gen_correction(suite: Suite, db: sqlite3.Connection, doc_of: dict, *,
                   target: int, rng: random.Random) -> None:
    """정정 최초본↔최종본의 **달라진 항목을 계산해서** 채점 기준으로 넣는다.

    suite_v1 의 같은 유형은 "달라진 항목은 사람이 대조해 채울 것"으로 비워
    뒀는데, facts 에 두 버전의 값이 다 있으므로 기계로 구할 수 있다.
    """
    tag = "correction"
    groups = db.execute("""
        SELECT correction_group_id, COUNT(DISTINCT doc_id) n
        FROM facts WHERE correction_group_id IS NOT NULL AND is_correction IS NOT NULL
        GROUP BY 1 HAVING n >= 2
    """).fetchall()
    rng.shuffle(groups)

    used_slots: set = set()

    made = 0
    for gid, _n in groups:
        if made >= target:
            break
        # facts 에는 correction_order 가 없다(청크 메타에만 있다).
        # 제출일로 최초/최종을 가른다 — 정정본은 원본보다 늦게 제출된다.
        rows = db.execute("""
            SELECT doc_id, filing_date, key_norm, value_text, company, is_latest
            FROM facts WHERE correction_group_id=? AND value_num IS NOT NULL
        """, (gid,)).fetchall()
        if not rows:
            continue
        dates = {r[0]: (r[1] or "") for r in rows}
        latest = {r[0] for r in rows if r[5]}
        first_doc = min(dates, key=lambda d: dates[d])
        final_doc = (max(latest, key=lambda d: dates[d]) if latest
                     else max(dates, key=lambda d: dates[d]))
        if first_doc == final_doc:
            suite.reject(f"{tag}:버전1개")
            continue
        company = rows[0][4]
        if first_doc not in doc_of or final_doc not in doc_of:
            suite.reject(f"{tag}:문서없음")
            continue
        if not suite.room_for(company, "", tag=tag):
            continue

        def values(doc):
            out = {}
            for d, _dt, key, val, _c, _l in rows:
                if d == doc and not is_bad_key(key):
                    out.setdefault(clean_key(key), set()).add(val)
            return {k: v.pop() for k, v in out.items() if len(v) == 1}

        before, after = values(first_doc), values(final_doc)
        changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
        if len(changed) < 2:
            # 값이 안 바뀐 정정(첨부 추가 등)은 "무엇이 달라졌나"의 답이
            # '없음'이라 채점 기준이 안 선다. 1개만 바뀐 것도 '설명해줘'라는
            # 서술형 질문에는 얇다 — closed 로 물었어야 할 것이다.
            suite.reject(f"{tag}:변경항목부족")
            continue

        kind = norm_report(doc_of[final_doc]["report_nm"])
        # 같은 회사가 같은 유형을 여러 번 정정했으면 질문이 어느 체인을 가리키는지
        # 정해지지 않는다. 질문에 날짜를 붙여 구분할 수도 있지만, 그러면 대회
        # 참고 질의의 문형에서 멀어지고 문제도 쉬워진다(검색 범위가 좁아진다).
        # 그래서 **(회사, 공시유형)당 체인 하나만** 쓴다 — 변경 항목이 가장 많은
        # 것을 고른다(서술형 채점 기준이 두꺼운 쪽).
        slot = (company, kind)
        if slot in used_slots:
            suite.reject(f"{tag}:동일유형체인다수")
            continue
        used_slots.add(slot)
        suite.add(
            task_type="복합문서추론", mode="open",
            query=f"{company}의 {kind} 공시가 정정된 내역이 있는가? 있다면 "
                  f"최초 공시와 최종 정정본 사이에 무엇이 달라졌는지 설명해줘.",
            gold_docs=[doc_of[first_doc], doc_of[final_doc]],
            answer=None, answer_source="rubric_only",
            check_points=["정정 사실을 인지하는가",
                          "최신 정정본 수치를 답하는가(원본 수치를 그대로 답하면 오답)",
                          f"변경 항목({', '.join(changed[:4])})을 특정하는가",
                          "정정 전/후 공시를 모두 근거로 표시하는가"],
            company=company, key="정정", tag=tag,
            notes=f"정정 체인 {len(dates)}단계 / 변경 항목 {len(changed)}개",
            extra={"changed_fields": changed,
                   "before": {k: before[k] for k in changed},
                   "after": {k: after[k] for k in changed},
                   "expected_fields_hint": changed},
        )
        made += 1


# ---------------------------------------------------------------- 리포트


def write_report(path: Path, suite: Suite, meta: dict) -> None:
    rows = suite.rows
    by_type = collections.Counter((r["task_type"], r["mode"]) for r in rows)
    by_source = collections.Counter(r["answer_source"] for r in rows)
    by_gen = collections.Counter(r["generator"] for r in rows)
    companies = {r["company"] for r in rows if r["company"] and "|" not in r["company"]}

    lines = [
        f"# 평가셋 v2 — {len(rows)}문항", "",
        f"- 생성 {meta['generated_at']} / seed {meta['seed']}",
        f"- 회사 {len(companies)}곳 (유니버스 {meta['n_universe']}곳 중)",
        "", "## 유형별", "", "| 유형 | 모드 | 문항 |", "|---|---|---:|",
    ]
    for (t, m), n in sorted(by_type.items()):
        lines.append(f"| {t} | {m} | {n} |")

    lines += ["", "## 정답 출처", "",
              "정답을 **어떻게 확정했는지**. `rubric_only` 는 정답 문장이 없는 대신",
              "채점 항목이 붙어 있다(open_scoring 이 커버리지로 채점).", "",
              "| 출처 | 문항 | 뜻 |", "|---|---:|---|"]
    meanings = {
        "auto_facts": "facts 층에서 값이 유일하게 확정",
        "auto_compare": "두 회사 최댓값을 계산해 비교 결론까지 확정",
        "auto_meta": "manifest 에서 존재 여부가 확정 (예/아니오)",
        "rubric_only": "서술형 — 정답 문장 없이 채점 항목만",
    }
    for src, n in by_source.most_common():
        lines.append(f"| {src} | {n} | {meanings.get(src, '')} |")

    lines += ["", "## 생성기별", "", "| 생성기 | 문항 |", "|---|---:|"]
    for g, n in by_gen.most_common():
        lines.append(f"| {g} | {n} |")

    if suite.rejected:
        lines += ["", "## 버린 후보와 이유", "",
                  "품질 관문을 생성기 안에 뒀다. 사후 정리는 이미 오염된 걸 줄일 뿐이라,",
                  "여기서는 애초에 만들지 않는다.", "",
                  "| 이유 | 건수 |", "|---|---:|"]
        for reason, n in suite.rejected.most_common(25):
            lines.append(f"| {reason} | {n} |")

    lines += ["", "## 회사 분포 (상위 15)", "", "| 회사 | 문항 |", "|---|---:|"]
    per_co = collections.Counter()
    for r in rows:
        for c in (r["company"] or "").split("|"):
            if c:
                per_co[c] += 1
    for c, n in per_co.most_common(15):
        lines.append(f"| {c} | {n} |")

    lines += ["", "## 쓰는 법", "", "```bash",
              "# 검색 상한 (HCX 없이, 빠름)",
              "./run.sh python3 scripts/score_answers.py --gold eval/suite_v2.jsonl \\",
              "    --mode retrieval --k 10 --out results/v2suite_retrieval", "",
              "# 최종 답변 (HCX 사용)",
              "./run.sh python3 scripts/score_answers.py --gold eval/suite_v2.jsonl \\",
              "    --mode full --pipeline v2 --yes --out results/v2suite_full",
              "```", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--facts", default="artifacts_v2/facts/facts.sqlite")
    ap.add_argument("--periodic-facts", default="artifacts_v2/facts_periodic_v2/facts.sqlite")
    ap.add_argument("--schema", default="config/field_schema.json")
    ap.add_argument("--out", default="eval/suite_v2")
    ap.add_argument("--seed", type=int, default=20260830)
    # 유형별 목표 문항 수. 대회 6유형이 고르게 들어가야 한다.
    ap.add_argument("--n-lookup-form", type=int, default=60)
    ap.add_argument("--n-lookup-periodic", type=int, default=40)
    ap.add_argument("--n-summary", type=int, default=60)
    ap.add_argument("--n-compare", type=int, default=60)
    ap.add_argument("--n-funding", type=int, default=30)
    ap.add_argument("--n-termination", type=int, default=40)
    ap.add_argument("--n-correction", type=int, default=40)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    rng = random.Random(args.seed)
    docs = [json.loads(l) for l in open(Path(args.corpus) / "manifest.jsonl", encoding="utf-8")
            if l.strip()]
    doc_of = {d["doc_id"]: d for d in docs}
    universe = {d["corp_name"] for d in docs}

    schema = None
    if Path(args.schema).exists():
        from disclosure_rag.agent.field_schema import FieldSchema
        schema = FieldSchema.load(args.schema)
    else:
        logger.warning("Field Schema 없음 — 서술형 채점 기준을 만들 수 없어 요약 문항을 건너뛴다")

    suite = Suite()
    form_db = sqlite3.connect(args.facts)
    gen_lookup(suite, form_db, doc_of, tag="lookup_form",
               doc_groups=("exchange", "major", "holding"),
               target=args.n_lookup_form, rng=rng)
    gen_compare(suite, form_db, doc_of, target=args.n_compare, rng=rng)
    gen_correction(suite, form_db, doc_of, target=args.n_correction, rng=rng)

    if Path(args.periodic_facts).exists():
        per_db = sqlite3.connect(args.periodic_facts)
        gen_lookup(suite, per_db, doc_of, tag="lookup_periodic",
                   doc_groups=("periodic",), target=args.n_lookup_periodic, rng=rng)
        per_db.close()
    else:
        logger.warning("periodic facts 없음(%s) — 사업보고서 문항을 건너뛴다",
                       args.periodic_facts)

    if schema is not None:
        gen_summary(suite, docs, schema, target=args.n_summary, rng=rng)
    gen_funding(suite, docs, target=args.n_funding, rng=rng)
    gen_termination(suite, docs, target=args.n_termination, rng=rng)
    form_db.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
        for row in suite.rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    from datetime import datetime, timezone
    meta = {"generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "seed": args.seed, "n_universe": len(universe)}
    write_report(out.with_suffix(".md"), suite, meta)

    by_type = collections.Counter((r["task_type"], r["mode"]) for r in suite.rows)
    logger.info("생성 %d문항 -> %s", len(suite.rows), out.with_suffix(".jsonl"))
    for (t, m), n in sorted(by_type.items()):
        logger.info("   %-16s %-7s %3d", t, m, n)
    logger.info("버린 후보 %d건 (사유는 리포트 참조)", sum(suite.rejected.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
