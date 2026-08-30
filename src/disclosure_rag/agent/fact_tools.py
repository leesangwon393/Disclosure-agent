"""facts 층 Agent 도구 — 숫자 질의를 검색이 아니라 조회로 처리한다.

Agent 가 이 도구를 쓰면 좋은 질의:
    "계약금액 얼마야"                -> lookup_fact
    "계약금액이 매출 대비 몇 %야"     -> lookup_fact 두 번 + calculate_ratio
    "계약금액 1조 넘는 계약 찾아줘"    -> lookup_fact(min_value=1e12)  ← 검색으로는 불가능
    "A사와 B사 계약금액 비교"         -> compare_facts

여전히 search_disclosures 를 써야 하는 질의:
    "계약 해지 조건이 뭐야"           -> 서술문. facts 에 없다
    "왜 유보했어"                    -> 서술문

주의(HCX): system prompt 300자 초과 시 tool-calling 이 2턴째부터 실패한다.
도구 description 도 짧게 유지한다.
"""

from __future__ import annotations

from disclosure_rag.agent.tools import ToolDef
from disclosure_rag.facts.store import FactStore

_MAX_ROWS = 20


def _fmt(rows: list[dict]) -> list[dict]:
    """LLM 에 넘길 형태로 줄인다. 근거(chunk_id/doc_id)는 반드시 남긴다."""
    out = []
    for r in rows:
        out.append({
            "company": r["company"],
            "item": r["key_norm"],
            "value": r["value_text"],
            "value_num": r["value_num"],
            "unit": r["value_unit"],
            "date": r["value_date"],
            "filing_date": r["filing_date"],
            "report_name": r["report_name"],
            "doc_id": r["doc_id"],
            "chunk_id": r["chunk_id"],     # 근거 표시용 — 원문 조각으로 되짚을 수 있다
            "is_correction": r["is_correction"],
            "is_latest": r["is_latest"],
        })
    return out


def make_lookup_fact_tool(store: FactStore) -> ToolDef:
    def handler(company: str | None = None, item: str | None = None,
                doc_group: str | None = None, date_from: str | None = None,
                date_to: str | None = None, min_value: float | None = None,
                max_value: float | None = None, latest_only: bool = True,
                limit: int = 10) -> dict:
        rows = store.lookup(
            company=company, key=item, doc_group=doc_group,
            date_from=date_from, date_to=date_to,
            min_value=min_value, max_value=max_value,
            numeric_only=(min_value is not None or max_value is not None),
            latest_only=latest_only, limit=min(limit, _MAX_ROWS),
        )
        if not rows:
            # 조용히 빈 결과를 주지 않는다 — 어떤 항목이 있는지 알려준다(§7).
            avail = [k for k, _ in store.distinct_keys(company=company, limit=15)]
            return {"results": [], "note": "해당 조건의 정형 항목 없음. 서술형이면 "
                                           "search_disclosures 를 쓸 것.",
                    "available_items": avail}
        return {"results": _fmt(rows)}

    return ToolDef(
        name="lookup_fact",
        description="공시 서식의 정형 항목(계약금액·매출액대비·지분율 등)을 정확히 조회한다. "
                    "숫자·날짜 질의는 검색보다 이쪽이 정확하다. 금액 범위 조건도 가능.",
        parameters={
            "type": "object",
            "properties": {
                "company": {"type": "string", "description": "회사명(DART 법인명)"},
                "item": {"type": "string", "description": "항목명. 예: 계약금액, 최근매출액, 매출액대비"},
                "doc_group": {"type": "string", "enum": ["exchange", "major", "holding", "periodic"]},
                "date_from": {"type": "string", "description": "접수일 YYYYMMDD 이상"},
                "date_to": {"type": "string", "description": "접수일 YYYYMMDD 이하"},
                "min_value": {"type": "number", "description": "값 하한(숫자 항목만)"},
                "max_value": {"type": "number", "description": "값 상한(숫자 항목만)"},
                "latest_only": {"type": "boolean", "description": "정정 최신본만. 기본 true"},
                "limit": {"type": "integer"},
            },
            "required": [],
        },
        handler=handler,
    )


def make_list_fact_items_tool(store: FactStore) -> ToolDef:
    def handler(company: str | None = None, doc_group: str | None = None, limit: int = 30) -> dict:
        return {"items": [{"item": k, "count": c}
                          for k, c in store.distinct_keys(company=company, doc_group=doc_group,
                                                          limit=min(limit, 50))]}

    return ToolDef(
        name="list_fact_items",
        description="조회 가능한 정형 항목 이름 목록. 항목명을 추측하지 말고 먼저 이걸로 확인한다.",
        parameters={
            "type": "object",
            "properties": {
                "company": {"type": "string"},
                "doc_group": {"type": "string", "enum": ["exchange", "major", "holding", "periodic"]},
                "limit": {"type": "integer"},
            },
            "required": [],
        },
        handler=handler,
    )


def make_compare_facts_tool(store: FactStore) -> ToolDef:
    def handler(companies: list[str], item: str, latest_only: bool = True,
                per_company: int = 3) -> dict:
        out: dict[str, list[dict]] = {}
        for c in companies[:10]:
            out[c] = _fmt(store.lookup(company=c, key=item, latest_only=latest_only,
                                       limit=min(per_company, 5)))
        missing = [c for c, v in out.items() if not v]
        return {"by_company": out,
                "note": None if not missing else f"정형 항목 없음: {missing} (검색으로 보완 필요)"}

    return ToolDef(
        name="compare_facts",
        description="여러 회사의 같은 정형 항목을 한 번에 조회해 비교한다.",
        parameters={
            "type": "object",
            "properties": {
                "companies": {"type": "array", "items": {"type": "string"}},
                "item": {"type": "string"},
                "latest_only": {"type": "boolean"},
                "per_company": {"type": "integer"},
            },
            "required": ["companies", "item"],
        },
        handler=handler,
    )


def build_fact_tools(store: FactStore) -> list[ToolDef]:
    return [make_lookup_fact_tool(store), make_list_fact_items_tool(store),
            make_compare_facts_tool(store)]
