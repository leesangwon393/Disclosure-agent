"""Stage 3.5: '있는가 / 없는가' 를 **목록 전수 조회**로 답한다.

## 왜 검색으로는 못 답하나

유사도 검색은 상위 k건만 본다. 거기에 없다고 해서 코퍼스에 없는 게 아니다.
그래서 검색 결과만 보고는 "없다"를 말할 근거가 없고, 모델은 정직하게
"확인할 수 없습니다"로 물러선다.

실측(results/v2_off4, 2026-08-30) — 이 유형 2문항이 전부 그랬다:

    Q  한미반도체가 체결한 단일판매·공급계약 중 이후 해지된 계약이 존재하는가?
    정답  아니오 (해지 공시 없음)
    답변  제공된 근거로는 ... 확인할 수 없습니다

대회 평가항목이 「근거 기반 = 없는 내용을 지어내지 않는가」와 「정확성」 둘
다이므로, **확인된 부재를 '모름'으로 답하는 것도 감점이다.** 지어내는 것과
반대 방향의 실패다.

## 그래서 무엇이 달라지나

`manifest` 는 코퍼스 **전체 목록**이다. "회사 X 의 공시 중 보고서명이 Y 인 것"
은 여기서 빠짐없이 셀 수 있다. 0건이면 "없다"가 근거 있는 답이 된다.

주의: 세는 대상을 좁게 잡아야 한다. 한미반도체는 보고서명에 '해지' 가 들어간
공시가 7건 있지만 전부 `주요사항보고서(자기주식취득신탁계약해지결정)` 이고,
질문이 묻는 `단일판매ㆍ공급계약해지` 는 0건이다. 그래서 **유형 + 사건어**를
함께 보고, 유형만 맞는 문서는 근거로 따로 돌려준다.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# 존재를 묻는 말끝. 내용어가 아니라 **어미**로 잡는다 — 회사명·지표명이
# 무엇이든 문장이 이렇게 끝나면 존재 질문이다.
EXISTENCE_ENDINGS: tuple[str, ...] = (
    "존재하는가", "존재하나", "존재합니까", "존재하나요", "존재했는가",
    "있는가", "있나", "있나요", "있습니까", "있었는가", "있는지",
    "적이 있는가", "적이 있나", "된 적", "한 적",
)

# 사건어. 계약이 '체결' 됐는지가 아니라 그 뒤에 무슨 일이 있었는지를 묻는 말.
# 보고서명에 그대로 등장하는 낱말만 넣는다(추측으로 늘리지 않는다).
EVENT_WORDS: tuple[str, ...] = (
    "해지", "해제", "취소", "철회", "중단", "종료", "파기", "정정",
)

# 공시유형 낱말. plan.report_kinds 가 비었을 때 질문에서 직접 줍는다.
KIND_HINTS: tuple[str, ...] = (
    "단일판매", "공급계약", "자기주식", "유상증자", "무상증자", "전환사채",
    "신주인수권부사채", "교환사채", "회사합병", "회사분할", "영업양수",
    "영업양도", "주식교환", "신규시설투자", "타법인주식", "투자판단",
    "대량보유", "임원", "소송",
)

_STRIP = re.compile(r"[\sㆍ·・‧∙、,()\[\]{}<>「」『』\"'’‘“”·．.\-–—/]")


def norm_key(s: str) -> str:
    """비교용 키. 가운뎃점·괄호·공백을 지운다.

    코퍼스는 `단일판매ㆍ공급계약체결`(U+318D), 질문은 `단일판매·공급계약`
    (U+00B7)을 쓴다. 지우지 않으면 절대 안 맞는다.
    """
    return _STRIP.sub("", unicodedata.normalize("NFC", s or ""))


def is_existence_question(query: str) -> bool:
    q = unicodedata.normalize("NFC", query or "")
    return any(e in q for e in EXISTENCE_ENDINGS)


def detect_event(query: str) -> str:
    """질문이 묻는 사건어. 없으면 빈 문자열."""
    q = unicodedata.normalize("NFC", query or "")
    for w in EVENT_WORDS:
        if w in q:
            return w
    return ""


def detect_kind_keys(query: str, report_kinds=()) -> list[str]:
    """대상 공시유형의 비교 키 목록. plan 이 준 유형을 우선한다."""
    keys = [norm_key(k) for k in (report_kinds or []) if norm_key(k)]
    if keys:
        return list(dict.fromkeys(keys))
    q = norm_key(query)
    return [norm_key(h) for h in KIND_HINTS if norm_key(h) in q]


@dataclass(frozen=True)
class DocRef:
    doc_id: str
    report_nm: str
    rcept_dt: str

    def label(self) -> str:
        return f"{self.report_nm} ({self.rcept_dt}) [{self.doc_id}]"


@dataclass(frozen=True)
class ExistenceResult:
    """존재 확인 결과. `applicable=False` 면 파이프라인은 이걸 무시한다."""

    applicable: bool = False
    verdict: str = "불명"          # 예 | 아니오 | 불명
    company: str = ""
    event: str = ""
    kind_label: str = ""
    matches: tuple[DocRef, ...] = ()
    related: tuple[DocRef, ...] = ()   # 유형은 맞지만 사건어가 없는 문서
    scanned: int = 0                   # 그 회사 공시 몇 건을 전수 확인했나
    note: str = ""
    _unused: tuple = field(default=(), repr=False)

    def prompt_block(self) -> str:
        """Evidence Pack 에 넣을 텍스트. 없으면 빈 문자열."""
        if not self.applicable or self.verdict == "불명":
            return ""
        if self.event and self.kind_label:
            target = f"{self.kind_label} 중 '{self.event}' 에 해당하는 공시"
        elif self.event:
            target = f"보고서명에 '{self.event}' 가 들어간 공시"
        else:
            target = f"{self.kind_label} 에 해당하는 공시"
        head = (f"[전수 확인] {self.company} 의 공시 {self.scanned}건 전체를 목록에서 "
                f"확인한 결과입니다. 상위 검색 결과가 아니라 **빠짐없이 센 것**입니다.")
        if self.verdict == "예":
            body = [f"- 찾는 대상: {target}", f"- 결과: **{len(self.matches)}건 존재**"]
            body += [f"    · {d.label()}" for d in self.matches[:10]]
            if len(self.matches) > 10:
                body.append(f"    · (그 외 {len(self.matches) - 10}건)")
            body.append("→ 이 질문의 답은 **예**입니다. 위 공시를 근거로 제시하세요.")
        else:
            body = [f"- 찾는 대상: {target}", "- 결과: **0건**"]
            if self.related:
                body.append(f"- 참고: 같은 유형의 공시는 {len(self.related)}건 있습니다.")
                body += [f"    · {d.label()}" for d in self.related[:5]]
                if len(self.related) > 5:
                    body.append(f"    · (그 외 {len(self.related) - 5}건)")
            body.append("→ 이 질문의 답은 **아니오**입니다. "
                        "'확인할 수 없습니다'가 아니라 '없습니다'라고 답하세요. "
                        "전수 확인했으므로 부재가 확인된 것입니다.")
        return head + "\n" + "\n".join(body) + "\n"


NOT_APPLICABLE = ExistenceResult()


def check_existence(query: str, manifest, *, companies=(), report_kinds=()) -> ExistenceResult:
    """회사 공시 목록을 전수로 훑어 존재 여부를 판정한다.

    적용 조건 — 셋 다 만족해야 한다:
      1. 질문이 존재를 묻는 어미로 끝난다
      2. 회사가 정확히 하나로 특정됐다 (둘 이상이면 비교 질문이라 여기서 안 다룬다)
      3. 그 회사의 공시가 manifest 에 있다 (없으면 범위 게이트 소관이다)
    """
    if not is_existence_question(query):
        return NOT_APPLICABLE
    names = [c for c in (companies or []) if str(c).strip()]
    if len(names) != 1:
        return NOT_APPLICABLE
    company = str(names[0]).strip()

    rows = [r for r in (manifest or [])
            if norm_key(getattr(r, "corp_name", "")) == norm_key(company)]
    if not rows:
        return NOT_APPLICABLE

    event = detect_event(query)
    kind_keys = detect_kind_keys(query, report_kinds)
    if not event and not kind_keys:
        # 무엇의 존재를 묻는지 특정 못 했다. 억지로 '없다'고 하면 안 된다.
        return ExistenceResult(applicable=False, company=company, scanned=len(rows),
                               note="대상 유형·사건을 특정하지 못했다")

    def ref(r) -> DocRef:
        return DocRef(doc_id=getattr(r, "doc_id", ""),
                      report_nm=getattr(r, "report_nm", ""),
                      rcept_dt=getattr(r, "rcept_dt", ""))

    matches, related = [], []
    for r in rows:
        nm = norm_key(getattr(r, "report_nm", ""))
        kind_ok = (not kind_keys) or any(k in nm for k in kind_keys)
        event_ok = (not event) or (event in nm)
        # 정정본은 보고서명이 `[기재정정]...` 이라 이름으로도 잡히지만,
        # manifest 가 별도 플래그를 갖고 있으므로 그쪽을 더 믿는다.
        if event == "정정" and getattr(r, "is_correction", False):
            event_ok = True
        if kind_ok and event_ok:
            matches.append(ref(r))
        elif kind_ok:
            related.append(ref(r))

    kind_label = " / ".join(report_kinds) if report_kinds else (
        "해당 유형" if kind_keys else "")
    return ExistenceResult(
        applicable=True,
        verdict="예" if matches else "아니오",
        company=company, event=event, kind_label=kind_label,
        matches=tuple(matches), related=tuple(related), scanned=len(rows),
    )


__all__ = [
    "DocRef", "ExistenceResult", "NOT_APPLICABLE",
    "EXISTENCE_ENDINGS", "EVENT_WORDS", "KIND_HINTS",
    "check_existence", "detect_event", "detect_kind_keys",
    "is_existence_question", "norm_key",
]
