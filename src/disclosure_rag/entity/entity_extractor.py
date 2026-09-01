"""Entity Extraction (§35).

질문에서 검색에 필요한 entity(회사/기간/지표/공시명/정정 명시 여부)를 추출한다.
기업명은 universe.csv 의 corp_name + listed_name(통용명, 예: 현대차→현대자동차)
를 alias map 으로 써서 매칭하고, 전부 NFC 로 정규화한다 (§35 "기업명 metadata 는
NFC normalize").
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from disclosure_rag.common.manifest_loader import load_universe
from disclosure_rag.common.unicode_utils import normalize_nfc

_YEAR = r"20\d{2}\s*년?"
_RECENT_N_YEAR = r"최근\s*\d+\s*년"
_QUARTER = r"[1-4]\s*분기"
_HALF = r"상반기|하반기"
_YM = r"20\d{2}[.\-]\s*\d{1,2}\s*월?"
# "2024년 05월" 형태. _YEAR 보다 **먼저** 와야 "2024년" 으로 잘리지 않는다.
# 이게 없어서 연·월을 지목한 질문 94건 중 54건이 월 정보를 잃었다(2026-09-01).
_YM_KO = r"20\d{2}\s*년\s*\d{1,2}\s*월"
_PERIOD_PAT = re.compile("|".join([_RECENT_N_YEAR, _YM_KO, _YM, _YEAR, _QUARTER, _HALF]))

_CORRECTION_KEYWORDS = ("기재정정", "정정공시", "정정")

# 회사명 뒤에 붙어도 **같은 회사**인 꼬리들(조사·의존명사). 이 목록에 없는
# 한글/영숫자가 바로 뒤에 붙으면 그건 다른 회사다.
#
#   "카카오뱅크의 직원 수"  -> '카카오' 가 잡혀 **카카오 답을 카카오뱅크 답으로**
#   "카카오모빌리티", "카카오페이증권" 도 같은 일이 벌어진다
#   (2026-08-31 gold_abstention 160문항에서 3건 실측)
#
# 반대로 "삼성전자와", "현대차의" 는 조사가 붙은 같은 회사다. 그래서 뒤 글자를
# 무조건 막으면 안 되고 조사만 허용한다.
_COMPANY_TAILS = (
    "의", "와", "과", "은", "는", "이", "가", "를", "을", "에", "도", "만",
    "로", "으로", "에서", "에게", "께", "보다", "부터", "까지", "이나", "나",
    "랑", "이랑", "이라", "라", "및", "등", "측", "사", "요", "야", "인",
)
_WORD_CHAR = re.compile(r"[가-힣A-Za-z0-9]")

# 회사명 **뒤에 공백을 두고** 이 말이 오면 그건 해외 자회사·현지법인이지
# 같은 회사가 아니다.
#
#   "현대로템 주식회사"        -> 같은 회사 (법인격 표기)
#   "현대로템 USA"            -> 다른 회사 (미국 현지법인)
#   "LS ELECTRIC AMERICA"    -> 다른 회사
#
# 문자열 구조가 같아서 조사 규칙만으로는 못 가른다. 그래서 지명·해외 법인격을
# 목록으로 둔다. 목록에 없는 말(사업보고서, 매출액, 2024년 …)은 그대로
# 통과하므로 평상시 질문은 영향받지 않는다(평가 문항 836개 회귀 0건).
_FOREIGN_SUFFIX_EN = frozenset({
    "USA", "AMERICA", "EUROPE", "ASIA", "CHINA", "JAPAN", "VIETNAM", "INDIA",
    "MEXICO", "BRAZIL", "POLAND", "HUNGARY", "GERMANY", "FRANCE", "UK", "CANADA",
    "AUSTRALIA", "SINGAPORE", "THAILAND", "INDONESIA", "PHILIPPINES", "MALAYSIA",
    "TURKEY", "RUSSIA", "EGYPT", "TUNISIA", "SLOVAKIA", "CZECH",
    "GMBH", "BV", "NV", "SA", "SAS", "SRL", "SDN", "BHD", "PTE", "PVT", "LLC", "LLP",
})
_FOREIGN_SUFFIX_KO = (
    "유한공사", "유한책임회사", "현지법인", "해외법인",
    "아메리카", "유럽", "차이나", "재팬", "베트남", "인디아",
)
_NEXT_EN = re.compile(r"\s+([A-Za-z][A-Za-z.]*)")
_NEXT_KO = re.compile(r"\s+([가-힣]+)")


def _foreign_affiliate_follows(tail: str) -> bool:
    """이름 뒤에 오는 말이 해외 법인 표시인가."""
    m = _NEXT_EN.match(tail)
    if m and m.group(1).upper().replace(".", "") in _FOREIGN_SUFFIX_EN:
        return True
    m = _NEXT_KO.match(tail)
    if m and any(m.group(1).startswith(k) for k in _FOREIGN_SUFFIX_KO):
        return True
    return False

_REPORT_NAME_TERMS = [
    "사업보고서", "반기보고서", "분기보고서",
    "주요사항보고서", "주식등의대량보유상황보고서", "대량보유상황보고서",
    "감사보고서", "연결감사보고서",
]


class ExtractedEntities(BaseModel):
    raw_query: str
    companies: list[str] = Field(default_factory=list)
    company_count: int = 0
    period: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    report_name: str | None = None
    explicit_correction: bool = False
    # (start, end) 는 normalize_query 가 재사용할 수 있도록 company 매칭 위치도 보존
    company_spans: list[tuple[int, int, str]] = Field(default_factory=list, exclude=True)


def _load_metric_terms(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.is_file():
        return []
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


class EntityExtractor:
    def __init__(self, *, corpus_root: str | Path, metric_terms_path: str | Path | None = None):
        universe = load_universe(corpus_root)
        alias_map: dict[str, str] = {}
        for _, row in universe.iterrows():
            corp = normalize_nfc(row["corp_name"])
            alias_map[corp] = corp
            listed = normalize_nfc(row.get("listed_name"))
            if listed and listed != corp:
                alias_map[listed] = corp
        self._alias_map = alias_map
        self._sorted_aliases = sorted(alias_map.keys(), key=len, reverse=True)
        self._metric_terms = _load_metric_terms(metric_terms_path) if metric_terms_path else []

    def _is_standalone(self, text: str, start: int, end: int) -> bool:
        """이 위치의 매칭이 **더 긴 다른 회사명의 앞부분**은 아닌가.

        앞뒤로 한글/영숫자가 이어지면 다른 이름의 일부다. 단, 뒤에 오는 것이
        조사면 같은 회사다("삼성전자와", "현대차의").
        """
        before = text[start - 1] if start > 0 else ""
        if before and _WORD_CHAR.match(before):
            return False
        tail = text[end:]
        # "현대로템 USA" 처럼 공백 뒤에 지명·해외 법인격이 오면 다른 회사다.
        if _foreign_affiliate_follows(tail):
            return False
        if not tail or not _WORD_CHAR.match(tail):
            return True
        # 조사인지 확인할 때 **조사 뒤 글자까지** 본다. 조사 뒤에 글자가 이어지면
        # 그건 조사가 아니라 이름의 일부다.
        #   "삼성전자로 인해"   -> '로' 뒤가 공백  -> 조사  -> 같은 회사
        #   "삼성전자로지텍"     -> '로' 뒤가 '지'  -> 이름  -> 다른 회사
        # 긴 조사부터 맞춰야 "에서" 를 "에" 로 잘못 끊지 않는다.
        for particle in sorted(_COMPANY_TAILS, key=len, reverse=True):
            if tail.startswith(particle):
                rest = tail[len(particle):]
                if not rest or not _WORD_CHAR.match(rest):
                    return True
        return False

    def _extract_companies(self, query_nfc: str) -> list[tuple[int, int, str]]:
        spans: list[tuple[int, int, str]] = []
        for alias in self._sorted_aliases:
            start = 0
            while True:
                idx = query_nfc.find(alias, start)
                if idx == -1:
                    break
                end = idx + len(alias)
                if (not any(not (end <= s or idx >= e) for s, e, _ in spans)
                        and self._is_standalone(query_nfc, idx, end)):
                    spans.append((idx, end, self._alias_map[alias]))
                start = idx + 1
        return sorted(spans, key=lambda s: s[0])

    def extract(self, query: str) -> ExtractedEntities:
        query_nfc = normalize_nfc(query)

        company_spans = self._extract_companies(query_nfc)
        companies: list[str] = []
        for _s, _e, corp in company_spans:
            if corp not in companies:
                companies.append(corp)

        periods = [m.group(0).strip() for m in _PERIOD_PAT.finditer(query_nfc)]

        metrics = [term for term in self._metric_terms if term.lower() in query_nfc.lower()]

        report_name = next((t for t in _REPORT_NAME_TERMS if t in query_nfc), None)

        explicit_correction = any(kw in query_nfc for kw in _CORRECTION_KEYWORDS)

        return ExtractedEntities(
            raw_query=query,
            companies=companies,
            company_count=len(companies),
            period=periods,
            metrics=metrics,
            report_name=report_name,
            explicit_correction=explicit_correction,
            company_spans=company_spans,
        )
