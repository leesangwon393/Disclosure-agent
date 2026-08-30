"""공시유형별 표준항목 명세 (ⓑ Field Schema).

## 무엇인가

"이 공시유형의 답변에는 어떤 항목이 있어야 하는가"를 정의한 표다.
`scripts/build_field_schema.py` 가 facts.sqlite 에서 **자동 생성**하고,
이 모듈은 그 결과를 읽어 조회만 한다. 손으로 쓴 목록이 아니다.

    신규시설투자등 -> required = [투자금액, 투자목적, 자기자본, 투자구분, ...]
                      conditional = [정정사유, 정정전, ...]   # 정정본일 때만 존재

## 어디에 쓰이나 — 두 가지 용법을 헷갈리지 말 것

1. `expected_fields(query)` — **요약형(open) 질문**의 충족 목표.
   "신규시설투자등 공시 내용을 정리해줘" -> 투자금액·투자목적·투자기간…
   Sufficiency Check 의 종료 조건과 Abstention Gate 의 '핵심 근거 부족' 판정이
   이 값을 쓴다.

2. `classify(kind, key)` — **조회형(closed) 질문**에서 질문이 말한 항목이
   그 공시유형에 실제로 존재하는 항목인지 검증. closed 질문의 expected_fields
   는 질문에서 뽑은 지표(예: "계약금액")지 이 표가 아니다.

## 정답셋 실측 (suite_v1 38문항, 2026-08-30)

    유형 매칭            32/38 = 84.2%
    expected_fields 확보 30/38 = 78.9%

못 잡는 6문항은 원인이 분명하다:

    S027~S029  "2024년 사업보고서와 2026년 사업보고서를 비교" -> periodic. facts 미포함
    S030~S032  "자금조달 내역을 유형별로 정리"              -> 질문이 공시유형을 말하지 않음

둘 다 이 표로는 풀 수 없는 문제이므로 빈 목록을 돌려준다(제약 없음).

## 왜 required / conditional / optional 로 나누나

모든 항목이 항상 존재한다고 가정하면 재검색이 끝나지 않는다. 예를 들어
`정정사유` 는 정정본에만 있으므로(단일판매공급계약체결 문서의 50.9%),
이걸 required 로 두면 원본 공시 질문에서 영원히 '부족' 판정이 난다.

## 한계 — 반드시 알고 쓸 것

facts.sqlite 에는 **periodic(정기공시) fact 가 0건**이다. 이 스키마는
exchange / holding / major 만 덮는다(2,913 문서). 사업보고서 질문의
expected_fields 는 여기서 나오지 않으므로, 알 수 없는 종류에 대해서는
빈 목록을 돌려준다 — **모르면 제약을 걸지 않는다**(fail open). 잘못된
required 를 걸어 정답 가능한 질문을 거부하는 쪽이 더 위험하기 때문이다.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SCHEMA_PATH = "config/field_schema.json"

# 공시명 표기 흔들림을 없앤다.
#   "단일판매ㆍ공급계약체결" / "단일판매·공급계약 체결" -> "단일판매공급계약체결"
#   "주요사항보고서(자기주식 취득 결정)"                 -> "주요사항보고서(자기주식취득결정)"
_STRIP_PAT = re.compile(r"[\s·ㆍ・:∙]")
_BRACKET_PREFIX = re.compile(r"^\[[^\]]*\]")   # "[기재정정]주요사항보고서(...)"


def normalize_report_kind(name: str | None) -> str:
    """공시유형 이름 정규화. 공백·가운뎃점·머리 대괄호를 없앤다."""
    s = unicodedata.normalize("NFC", name or "")
    s = _BRACKET_PREFIX.sub("", s.strip())
    return _STRIP_PAT.sub("", s)


def normalize_field_key(key: str | None) -> str:
    """항목명 정규화. facts.key_norm 과 같은 규칙을 쓴다."""
    s = unicodedata.normalize("NFC", key or "")
    return _STRIP_PAT.sub("", s.strip())


@dataclass(frozen=True)
class KindSchema:
    kind: str
    doc_group: str
    n_docs: int
    required: list[str]
    conditional: list[str]
    optional: list[str]
    ratios: dict[str, float]
    search_terms: list[str]
    core_terms: list[str]
    action_terms: list[str]
    sufficient_data: bool

    def classify(self, key: str) -> str:
        """항목이 required / conditional / optional / unknown 중 무엇인가."""
        k = normalize_field_key(key)
        if k in self.required:
            return "required"
        if k in self.conditional:
            return "conditional"
        if k in self.optional:
            return "optional"
        return "unknown"


class FieldSchema:
    def __init__(self, payload: dict):
        self.meta = {k: v for k, v in payload.items() if k != "kinds"}
        self._kinds: dict[str, KindSchema] = {}
        for kind, d in (payload.get("kinds") or {}).items():
            self._kinds[kind] = KindSchema(
                kind=kind,
                doc_group=d.get("doc_group", ""),
                n_docs=int(d.get("n_docs", 0)),
                required=list(d.get("required") or []),
                conditional=list(d.get("conditional") or []),
                optional=list(d.get("optional") or []),
                ratios={k: float(v) for k, v in (d.get("ratios") or {}).items()},
                search_terms=list(d.get("search_terms") or [kind]),
                core_terms=list(d.get("core_terms") or []),
                action_terms=list(d.get("action_terms") or []),
                sufficient_data=bool(d.get("sufficient_data", False)),
            )
        # 질문 매칭용: 긴 표현부터 본다("단일판매공급계약해지" 가
        # "단일판매공급계약체결" 보다 먼저 걸리는 일이 없도록 길이 우선).
        self._terms: list[tuple[str, str]] = sorted(
            ((normalize_report_kind(t), kind)
             for kind, ks in self._kinds.items() for t in ks.search_terms),
            key=lambda p: -len(p[0]),
        )
        # 어간 -> 그 어간을 공유하는 유형들. 2단계 매칭에 쓴다.
        cores: dict[str, list[str]] = {}
        for kind, ks in self._kinds.items():
            for core in ks.core_terms:
                cores.setdefault(normalize_report_kind(core), []).append(kind)
        self._cores: dict[str, list[str]] = {
            c: sorted(v) for c, v in sorted(cores.items(), key=lambda kv: -len(kv[0]))
        }

    # ------------------------------------------------------------------ 로딩

    @classmethod
    def load(cls, path: str | Path = DEFAULT_SCHEMA_PATH) -> "FieldSchema":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def empty(cls) -> "FieldSchema":
        """스키마 파일이 없을 때의 무해한 기본값 — 아무 제약도 걸지 않는다."""
        return cls({"kinds": {}})

    # ------------------------------------------------------------------ 조회

    def kinds(self) -> list[str]:
        return sorted(self._kinds)

    def get(self, kind: str | None) -> KindSchema | None:
        if not kind:
            return None
        return self._kinds.get(normalize_report_kind(kind))

    def required(self, kind: str | None) -> list[str]:
        """모르는 종류면 빈 목록 — 제약을 걸지 않는다(fail open)."""
        ks = self.get(kind)
        return list(ks.required) if ks else []

    def conditional(self, kind: str | None) -> list[str]:
        ks = self.get(kind)
        return list(ks.conditional) if ks else []

    def classify(self, kind: str | None, key: str) -> str:
        ks = self.get(kind)
        return ks.classify(key) if ks else "unknown"

    # ------------------------------------------------------------------ 질문 매칭

    def match_kinds(self, query: str) -> list[str]:
        """질문에 등장하는 공시유형을 찾는다. 2단계다.

        1단계 — 공시 이름이 통째로 등장하는 경우(길이 긴 표현 우선).
        2단계 — 이름이 흩어져 등장하는 경우. 질문은 공시 이름을 그대로 쓰지
                않는다:

                    이름 : 단일판매공급계약해지
                    질문 : "체결한 단일판매·공급계약 중 이후 해지된 계약이…"

                어간("단일판매공급계약")이 보이면 형제 유형들을 후보로 놓고,
                동작어("체결"/"해지") 중 질문에 나온 것만 남긴다. 동작어가
                하나도 없으면 형제를 **전부** 돌려준다 — 어느 쪽인지 모를 때
                좁히면 근거를 놓치기 때문이다(fail open).

        위 예시처럼 '체결'과 '해지'가 둘 다 나오면 둘 다 돌려준다. 실제로 그
        질문에 답하려면 체결 공시와 해지 공시가 모두 필요하므로 이게 맞다.
        """
        q = normalize_report_kind(query)
        found: list[str] = []
        spans: list[tuple[int, int]] = []

        for term, kind in self._terms:                       # 1단계
            if not term:
                continue
            i = q.find(term)
            if i < 0:
                continue
            if any(i >= s and i + len(term) <= e for s, e in spans):
                continue      # 더 긴 표현 안에 들어 있다
            spans.append((i, i + len(term)))
            if kind not in found:
                found.append(kind)

        for core, siblings in self._cores.items():           # 2단계
            if core not in q:
                continue
            hit = [k for k in siblings
                   if any(a and a in q for a in self._kinds[k].action_terms)]
            for kind in (hit or siblings):
                if kind not in found:
                    found.append(kind)
        return found

    def match_one(self, query: str) -> str | None:
        got = self.match_kinds(query)
        return got[0] if got else None

    def expected_fields(self, query: str, *, max_kinds: int = 2) -> list[str]:
        """질문에 대해 '답변에 있어야 하는 항목'을 돌려준다.

        Sufficiency Check 의 종료 조건이 되는 값이라, 넘치면 답할 수 있는 질문이
        영원히 '부족' 판정을 받는다. 그래서 보수적으로 간다:

            유형 1개  -> 그 유형의 required
            유형 2개  -> 두 유형의 **교집합** (둘 중 어느 쪽이든 반드시 필요한 것)
            유형 3개+ -> 빈 목록. 질문이 유형을 특정하지 못한 것이므로 제약을
                        걸지 않는다(fail open).
            유형 0개  -> 빈 목록

        마지막 두 경우가 실제로 발생한다. 예를 들어 "주요사항보고서 공시가
        정정된 내역이 있는가?"(S025)는 괄호 안 유형을 말하지 않아 주요사항보고서
        전체(15종)에 걸린다 — 그 15종의 required 를 합쳐 요구하면 어떤 답변도
        통과하지 못한다.
        """
        kinds = self.match_kinds(query)
        if not kinds or len(kinds) > max_kinds:
            return []
        sets = [set(self.required(k)) for k in kinds]
        sets = [x for x in sets if x]
        if not sets:
            return []
        out = sets[0]
        for other in sets[1:]:
            out &= other
        return sorted(out)

    def fields_mentioned(self, query: str, kinds: list[str] | None = None) -> list[str]:
        """질문 문장에 직접 등장한 항목명을 돌려준다(긴 것 우선).

        closed 질문의 `expected_fields` 는 질문이 **지목한 항목** 하나지, 그
        공시유형의 required 전부가 아니다. 실측 실패: "투자금액은 얼마인가?"에
        신규시설투자등의 required 11개를 다 요구해서, 답할 수 있는 질문이
        충분성 검사에서 막히고 거부까지 갔다.

        `config/metric_terms.txt`(손으로 쓴 35줄) 대신 이 표를 쓴다 — 여기엔
        데이터에서 뽑은 항목명이 유형별로 들어 있다.
        """
        q = normalize_field_key(query)
        pool = kinds if kinds else self.kinds()
        names: set[str] = set()
        for kind in pool:
            ks = self.get(kind)
            if ks is not None:
                names |= set(ks.ratios) | set(ks.required) | set(ks.conditional)
        # 공시유형 이름이 항목명 자리에 들어온 경우를 뺀다. facts 추출기가
        # 문서명을 key 로 잡은 흔적이라 `ratios` 안에 실제로 들어 있다.
        #
        # **core_terms 를 빠뜨리면 안 된다.** 실측 실패(v2 스모크 5문항 중 4문항):
        # "주요사항보고서(자기주식취득결정)에 기재된 순자산액은 얼마인가?" 에서
        # `주요사항보고서` 가 필수 항목으로 잡혀 거부 게이트가 답할 수 있는
        # 질문을 막았다. 그건 kind 이름도 search_term 도 아닌 **어간**이라
        # 앞의 두 집합만으로는 안 걸린다.
        type_names = {normalize_field_key(t) for t, _k in self._terms}
        type_names |= {normalize_field_key(k) for k in self._kinds}
        for ks in self._kinds.values():
            type_names |= {normalize_field_key(t) for t in ks.core_terms}
            type_names |= {normalize_field_key(t) for t in ks.action_terms}

        hits: list[str] = []
        for name in sorted(names, key=len, reverse=True):
            if len(name) < 2 or name not in q:
                continue
            if normalize_field_key(name) in type_names:
                continue
            if any(name in longer for longer in hits):
                continue      # 더 긴 항목명 안에 들어 있다
            hits.append(name)
        return hits

    # ------------------------------------------------------------------ 진단

    def coverage_note(self) -> str:
        cov = self.meta.get("coverage") or {}
        return cov.get("note", "")
