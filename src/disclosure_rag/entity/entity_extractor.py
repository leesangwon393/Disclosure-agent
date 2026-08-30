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
_PERIOD_PAT = re.compile("|".join([_RECENT_N_YEAR, _YM, _YEAR, _QUARTER, _HALF]))

_CORRECTION_KEYWORDS = ("기재정정", "정정공시", "정정")

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

    def _extract_companies(self, query_nfc: str) -> list[tuple[int, int, str]]:
        spans: list[tuple[int, int, str]] = []
        for alias in self._sorted_aliases:
            start = 0
            while True:
                idx = query_nfc.find(alias, start)
                if idx == -1:
                    break
                end = idx + len(alias)
                if not any(not (end <= s or idx >= e) for s, e, _ in spans):
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
