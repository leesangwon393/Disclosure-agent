"""Agent Tool 정의 (§58~62).

각 Tool 의 역할은 deterministic 하고 명확해야 한다. 이름을 스펙 그대로 맞출
필요는 없지만(§58), 여기서는 스펙 이름을 그대로 따른다 — 이해하기 쉬우라고.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from disclosure_rag.agent.calculation import calculate_cagr, calculate_growth_rate, calculate_ratio
from disclosure_rag.chunking.chunk_schema import ChunkSchema
from disclosure_rag.common.manifest_loader import ManifestRow
from disclosure_rag.common.unicode_utils import normalize_nfc
from disclosure_rag.correction.correction_graph_builder import CorrectionRecord
from disclosure_rag.retrieval.metadata_filter import RetrievalFilter, normalize_period_tokens

if TYPE_CHECKING:
    from disclosure_rag.agent.dual_channel import DualChannelRetriever
    from disclosure_rag.agent.query_plan import QueryPlan


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict  # JSON schema (HCX tools[].function.parameters)
    handler: Callable[..., dict]

    def to_hcx_schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


def _evidence_dict(chunk: ChunkSchema, score: float | None = None) -> dict:
    d = {
        "chunk_id": chunk.chunk_id,
        "report_id": chunk.report_id,
        "company": chunk.company,
        "report_type": chunk.report_type,
        "report_name": chunk.report_name,
        "period": chunk.period,
        "filing_date": chunk.filing_date,
        "section_path": chunk.section_path,
        "content_type": chunk.content_type,
        "is_correction": chunk.is_correction,
        "correction_group_id": chunk.correction_group_id,
        "is_latest": chunk.is_latest,
        "text": chunk.raw_text,
    }
    if score is not None:
        d["score"] = round(float(score), 4)
    return d


# === KIM 브랜치 (2026-08-23): route 별 top_k ===
# 기존 default_k=5 는 하이브리드가 BM25 단독에 **지는** 구간이었다:
#     R@5   BM25 0.706 vs Hybrid 0.661   (하이브리드 열세)
#     R@10  BM25 0.802 vs Hybrid 0.900   (하이브리드 우세)
# Dense 는 후보를 넓게 볼 때만 이득을 준다. 5개만 보면 임베딩을 넣는 의미가 없다.
# 그리고 올릴 여유는 충분하다 — 검색 자체가 43ms, 평가 타임아웃은 300초다.
# (HybridRetriever 는 이미 후보를 50개씩 가져와놓고 뒤에서 5개로 자르고 있었다.)
TOP_K_BY_ROUTE: dict[str | None, int] = {
    "single_lookup": 8,
    "correction_analysis": 12,
    "calculation": 15,
    "multi_compare": 30,
    None: 12,          # route 미판정/fallback
}
DEFAULT_TOP_K = 12


def top_k_for_route(route: str | None) -> int:
    return TOP_K_BY_ROUTE.get(route, DEFAULT_TOP_K)


# 2단계 검색 설정.
#
# 왜 2단계인가 — 실측(2026-08-30, 314문항 채점):
#   정기공시는 문서 1건이 leaf chunk 를 평균 559.7개 만든다(거래소공시는 1.0개).
#   그래서 "코퍼스 전체 626,497개에서 top-k" 를 한 번에 고르면, 정답 문서를
#   맞게 찾아놓고도(evidence_hit 94.1%) 그 문서 안의 정답 조각이 상위 k 밖으로
#   밀려난다(상한 71.5%). 그 22.6%p 격차가 전부 이 경우였고 47문항이다.
#
#   1단계에서 넓게 훑어 **후보 문서**를 정하고, 2단계에서 그 문서들 안에서만
#   다시 고르면 경쟁 대상이 626,497 -> 수백 개로 줄어든다.
#
# 안전장치: 2단계 결과가 1단계보다 나빠질 수 없도록, 1단계 상위 결과를 뒤에
# 덧붙여 채운다(2단계에서 이미 나온 chunk 는 제외).
DOC_POOL_K = 50      # 1단계에서 훑을 chunk 수 (후보 문서를 뽑는 용도)
MAX_CANDIDATE_DOCS = 3


def _candidate_report_ids(results, max_docs: int) -> list[str]:
    """1단계 결과에서 등장 순서대로 유니크 report_id 를 추린다(점수 순 = 등장 순)."""
    seen: list[str] = []
    for chunk, _score in results:
        rid = chunk.report_id
        if rid and rid not in seen:
            seen.append(rid)
            if len(seen) >= max_docs:
                break
    return seen


def _merge_keep_order(primary, backup, limit: int):
    """primary 를 앞에 두고 backup 으로 채운다. chunk_id 중복 제거."""
    out = []
    seen: set[str] = set()
    for chunk, score in list(primary) + list(backup):
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        out.append((chunk, score))
        if len(out) >= limit:
            break
    return out


def _round_robin_by_document(results, doc_ids: list[str], limit: int):
    """문서별로 번갈아 뽑아 각 후보 문서가 반드시 자리를 갖게 한다.

    그냥 "후보 문서들 안에서 상위 k" 로 하면 **한 문서가 상위권을 독식**해서
    2단계를 해도 달라지는 게 없다(테스트 test_two_stage_promotes_... 로 실제
    확인했다: 후보 3문서 중 B문서가 상위 40칸을 먹어 A문서 정답 조각이 여전히
    밀렸다). 라운드로빈이면 후보 1순위 문서의 2등·3등 조각이 다른 문서의 40등
    조각보다 먼저 들어온다 — 이게 2단계의 실질적인 이득이다.
    """
    buckets: dict[str, list] = {rid: [] for rid in doc_ids}
    for chunk, score in results:
        if chunk.report_id in buckets:
            buckets[chunk.report_id].append((chunk, score))

    merged = []
    while len(merged) < limit and any(buckets.values()):
        progressed = False
        for rid in doc_ids:
            if not buckets[rid]:
                continue
            merged.append(buckets[rid].pop(0))
            progressed = True
            if len(merged) >= limit:
                break
        if not progressed:
            break
    return merged


def make_search_disclosures_tool(
    retriever, *, default_k: int = DEFAULT_TOP_K,
    two_stage: bool = False, doc_pool_k: int = DOC_POOL_K,
    max_candidate_docs: int = MAX_CANDIDATE_DOCS,
) -> ToolDef:
    """2026-08-30: `two_stage` 기본값을 **False 로 내렸다 — 실측에서 더 나빴다.**

    80문항 실측(`results/diag_two_stage/`):

        방식                     정답문서 회수   상한도달
        A 필터 없음                  87.5%      81.2%
        B 회사필터 + 옛 pool 규칙      75.0%      60.0%
        C 회사필터 + 새 pool 규칙      88.8%      85.0%   <- 최고
        D 2단계 검색                 71.2%      76.2%   <- A 보다도 낮다

    가설은 "정기공시는 문서 1건이 chunk 560개라, 문서를 정하고 그 안에서 고르면
    낫다" 였다. 필터 순서 버그를 고치고 나니 **회사로 좁히는 것만으로 이미 그
    효과가 났고**(C), 거기서 문서 3개로 더 좁히는 건 과했다. D 의 정답문서
    회수가 71.2% 로 제일 낮은 게 근거다 — 후보 문서를 고르는 단계에서 정답
    문서를 아예 빼먹는다. 그리고 정답 문서가 1순위여도 라운드로빈 때문에
    12칸 중 4칸밖에 못 받는다.

    코드는 남겨둔다(후보 선정 방식을 바꿔 다시 시도할 수 있다). 다만 **기본으로
    켜두지 않는다** — 효과 없는 걸 붙여두는 게 제일 나쁘다.
    """
    def handler(query: str, company: str | None = None, report_type: str | None = None,
                period: str | None = None, report_id: str | None = None,
                top_k: int = default_k, latest_only: bool = True) -> dict:
        if report_id:
            # 특정 문서(예: get_correction_history 로 알아낸 원본/정정본의 report_id)
            # 의 실제 본문을 그대로 가져온다 — latest_only 무관하게 그 문서 자체를 본다.
            flt = RetrievalFilter(report_ids=[report_id])
            results = retriever.search(query, k=max(top_k, 10), flt=flt)
            return {"results": [_evidence_dict(c, s) for c, s in results]}

        # Coarse-to-Fine (§51): 정확한 filter 로 먼저 시도하되, 0건이면 필터를
        # 하나씩 완화하며 재시도한다.
        #
        # 2026-08-30 정정: 여기 원래 "HCX 가 period 포맷을 잘못 추측해서" 라고
        # 적혀 있었는데 오진이었다. chunk 의 period 는 항상 "YYYY-MM" 인데
        # EntityExtractor 도 HCX 도 "2024년"/"1분기" 같은 토큰을 보냈고, 옛
        # RetrievalFilter 가 정확일치라서 **어떤 기간 필터도 영구 0건**이었다.
        # 즉 완화 단계가 매번 발동해 period 를 버려 왔고 그래서 문제가 안 보였다.
        # 이제 normalize_period_tokens() 로 포맷을 맞춘 뒤 필터에 넣는다.
        attempts = [
            dict(companies=[company] if company else None, doc_groups=[report_type] if report_type else None,
                 periods=normalize_period_tokens(period), latest_only=latest_only),
        ]
        if period:
            attempts.append(dict(companies=attempts[0]["companies"], doc_groups=attempts[0]["doc_groups"],
                                  periods=None, latest_only=latest_only))
        if latest_only:
            attempts.append(dict(companies=attempts[0]["companies"], doc_groups=attempts[0]["doc_groups"],
                                  periods=None, latest_only=False))
        if report_type:
            attempts.append(dict(companies=attempts[0]["companies"], doc_groups=None,
                                  periods=None, latest_only=False))

        for kwargs in attempts:
            flt = RetrievalFilter(**kwargs)
            relaxed_note = None if kwargs == attempts[0] else "일부 필터를 완화해 재검색함(원 필터로는 0건)"

            if not two_stage:
                results = retriever.search(query, k=top_k, flt=flt)
                if results:
                    return {"results": [_evidence_dict(c, s) for c, s in results], "note": relaxed_note}
                continue

            # --- 1단계: 넓게 훑어 후보 문서를 정한다
            wide = retriever.search(query, k=max(doc_pool_k, top_k), flt=flt)
            if not wide:
                continue

            doc_ids = _candidate_report_ids(wide, max_candidate_docs)

            # --- 2단계: 후보 문서 안에서만 다시 고른다
            # (여기서 report_ids 필터가 selective 로 잡혀 리트리버가 pool 상한 없이
            #  정확히 순위를 매긴다 — metadata_filter.is_selective 참고)
            # 문서별 몫을 확보하려면 넉넉히 받아온 뒤 라운드로빈으로 줄인다.
            # HybridRetriever 는 k 와 무관하게 리트리버당 candidate_k=50 을 뽑고
            # 리랭커도 상위 50개만 보므로, 여기서 k 를 50 근처로 올려도 추가
            # 비용이 거의 없다(같은 후보 풀을 더 많이 돌려받을 뿐이다).
            narrow_k = max(doc_pool_k, top_k * max(len(doc_ids), 1))
            narrow = retriever.search(query, k=narrow_k,
                                      flt=RetrievalFilter(report_ids=doc_ids))
            balanced = _round_robin_by_document(narrow, doc_ids, top_k)

            # 2단계가 1단계보다 나빠지지 않도록 1단계 결과로 뒤를 채운다.
            merged = _merge_keep_order(balanced, wide, top_k)
            notes = [n for n in (relaxed_note,) if n]
            notes.append(f"2단계 검색: 후보 문서 {len(doc_ids)}건으로 좁혀 재검색")
            return {"results": [_evidence_dict(c, s) for c, s in merged],
                    "note": " / ".join(notes)}

        return {"results": [], "note": "필터를 단계적으로 완화했지만 관련 근거를 찾지 못함"}

    return ToolDef(
        name="search_disclosures",
        description=(
            "공시 본문에서 질문과 관련된 근거를 검색한다. 처음에는 report_type/period 를 "
            "비워두고 query(+company)만으로 넓게 검색하는 것을 권장한다 — 결과가 관련없거나 "
            "너무 많을 때만 report_type/period 로 좁혀서 재검색하라. 필터가 결과를 0건으로 "
            "만들면 이 tool 이 자동으로 필터를 완화해 재시도한다. get_correction_history 나 "
            "get_latest_report 로 특정 문서의 report_id(doc_id) 를 이미 알고 있다면, "
            "그 문서의 실제 본문 내용을 보기 위해 report_id 파라미터로 바로 조회하라 "
            "(예: 정정 전후 내용을 비교하려면 원본/정정본 각각의 report_id 로 이 tool 을 호출)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색어 (한국어)"},
                "company": {"type": "string", "description": "회사명 (DART 정식 법인명)"},
                "report_type": {"type": "string", "enum": ["periodic", "major", "exchange", "holding"]},
                "period": {"type": "string", "description": "반드시 'YYYY-MM' 또는 'YYYY' 형식만 사용 (예: '2024-12', '2024'). 날짜 범위나 다른 포맷은 지원하지 않으므로 확실하지 않으면 비워두라."},
                "report_id": {"type": "string", "description": "특정 문서의 report_id(doc_id) — 알고 있으면 이걸로 그 문서 본문을 직접 가져온다. 이 경우 company/report_type/period/latest_only 는 무시된다."},
                "top_k": {"type": "integer", "description": f"검색 결과 개수 (기본 {default_k}). 특별한 이유가 없으면 지정하지 말고 기본값을 쓰라 — 작게 줄이면 근거가 모자라 답을 못 한다."},
                "latest_only": {"type": "boolean", "description": "최신 유효본만 검색할지 여부 (기본 true, 정정 분석 질문이면 false)"},
            },
            "required": ["query"],
        },
        handler=handler,
    )


def run_dual_channel_search(
    retriever: "DualChannelRetriever", query: str, plan: "QueryPlan", *,
    top_k: int | None = None, flt: RetrievalFilter | None = None,
) -> dict:
    """QueryPlan으로 두 채널을 항상 함께 실행하는 비-LLM 진입점.

    HCX가 ``lookup_fact``를 선택해야만 Facts가 실행되는 기존 방식과 달리,
    ``plan.expected_fields``에 실제 정형 항목이 하나라도 있으면 이 함수가
    Facts 조회를 결정론적으로 실행한다. Facts는 반환값에서도 별도 채널이며
    BM25/Dense/Sparse fusion 점수를 받지 않는다.
    """
    return retriever.search(query, plan, k=top_k, flt=flt).to_dict()


def make_planned_search_disclosures_tool(
    retriever: "DualChannelRetriever", plan: "QueryPlan",
) -> ToolDef:
    """질문별 QueryPlan에 묶인 Dual Channel 검색 도구를 만든다.

    ``expected_fields``나 ``latest_policy``는 도구 인자로 노출하지 않는다.
    앞 단계에서 검증된 계획을 HCX가 임의로 바꾸지 못하게 하기 위해서다.
    """
    def handler(query: str, top_k: int | None = None) -> dict:
        return run_dual_channel_search(retriever, query, plan, top_k=top_k)

    return ToolDef(
        name="search_disclosures",
        description="검증된 질문 계획에 따라 공시 본문과 정형 항목을 함께 조회한다.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색어 (한국어)"},
                "top_k": {"type": "integer", "description": "본문 근거 최대 개수"},
            },
            "required": ["query"],
        },
        handler=handler,
    )


_TITLE_NOISE_CHARS = str.maketrans("", "", "ㆍ· /[]()")


def _title_match(needle: str, *haystacks: str | None) -> bool:
    """report_nm 원문에는 'ㆍ' 같은 구분자가 섞여 있어(예: '단일판매ㆍ공급계약체결')
    Agent 가 자연스럽게 쓰는 doc_subtype 스타일 문자열('단일판매공급계약체결')과
    글자 그대로는 substring 매칭이 안 된다 — 실측으로 확인된 실패 케이스. 구분자를
    제거하고 비교한다. doc_subtype 도 같이 후보로 넣어 이중으로 매칭한다."""
    needle_norm = needle.translate(_TITLE_NOISE_CHARS).replace("[기재정정]", "")
    for h in haystacks:
        if h and needle_norm in h.translate(_TITLE_NOISE_CHARS).replace("[기재정정]", ""):
            return True
    return False


def make_get_correction_history_tool(manifest: list[ManifestRow], correction_index: dict[str, CorrectionRecord]) -> ToolDef:
    def handler(company: str, report_name_contains: str | None = None) -> dict:
        company_nfc = normalize_nfc(company)
        candidates = [r for r in manifest if r.corp_name == company_nfc]
        if report_name_contains:
            candidates = [r for r in candidates if _title_match(report_name_contains, r.report_nm, r.doc_subtype)]

        groups: dict[str, list[ManifestRow]] = {}
        for r in candidates:
            rec = correction_index.get(r.doc_id)
            if rec is None:
                continue
            groups.setdefault(rec.correction_group_id, []).append(r)

        out = []
        for group_id, rows in groups.items():
            rows_sorted = sorted(rows, key=lambda r: correction_index[r.doc_id].correction_order)
            out.append({
                "correction_group_id": group_id,
                "chain": [
                    {
                        "doc_id": r.doc_id, "report_nm": r.report_nm, "rcept_dt": r.rcept_dt,
                        "is_correction": r.is_correction,
                        "correction_order": correction_index[r.doc_id].correction_order,
                        "is_latest": correction_index[r.doc_id].is_latest,
                        "resolution_source": correction_index[r.doc_id].resolution_source,
                    }
                    for r in rows_sorted
                ],
            })
        return {"correction_groups": out}

    return ToolDef(
        name="get_correction_history",
        description="특정 회사의 특정 공시가 정정된 이력(원본->정정1->정정2->... 순서, 최신본 여부)을 조회한다.",
        parameters={
            "type": "object",
            "properties": {
                "company": {"type": "string"},
                "report_name_contains": {"type": "string", "description": "공시명에 포함될 키워드 (예: '사업보고서')"},
            },
            "required": ["company"],
        },
        handler=handler,
    )


def make_get_latest_report_tool(manifest: list[ManifestRow], correction_index: dict[str, CorrectionRecord]) -> ToolDef:
    def handler(company: str, report_type: str, report_name_contains: str | None = None) -> dict:
        company_nfc = normalize_nfc(company)
        candidates = [r for r in manifest if r.corp_name == company_nfc and r.doc_group == report_type]
        if report_name_contains:
            candidates = [r for r in candidates if _title_match(report_name_contains, r.report_nm, r.doc_subtype)]
        latest_rows = [r for r in candidates if correction_index.get(r.doc_id) and correction_index[r.doc_id].is_latest]
        latest_rows.sort(key=lambda r: r.rcept_dt, reverse=True)
        if not latest_rows:
            return {"found": False}
        r = latest_rows[0]
        return {
            "found": True, "doc_id": r.doc_id, "report_nm": r.report_nm,
            "rcept_dt": r.rcept_dt, "is_correction": r.is_correction,
        }

    return ToolDef(
        name="get_latest_report",
        description="특정 회사/공시유형의 가장 최근 '최신 유효본' 공시를 찾는다 (정정 체인 반영).",
        parameters={
            "type": "object",
            "properties": {
                "company": {"type": "string"},
                "report_type": {"type": "string", "enum": ["periodic", "major", "exchange", "holding"]},
                "report_name_contains": {"type": "string"},
            },
            "required": ["company", "report_type"],
        },
        handler=handler,
    )


def _calc_tool(name: str, description: str, fn: Callable[..., dict], properties: dict, required: list[str]) -> ToolDef:
    return ToolDef(
        name=name, description=description,
        parameters={"type": "object", "properties": properties, "required": required},
        handler=fn,
    )


def build_calculation_tools() -> list[ToolDef]:
    return [
        _calc_tool(
            "calculate_growth_rate", "증가율/감소율/증감액을 계산한다 (LLM 암산 대신 정확한 계산).",
            calculate_growth_rate,
            {"before": {"type": "number"}, "after": {"type": "number"}}, ["before", "after"],
        ),
        _calc_tool(
            "calculate_ratio", "두 수치의 비율(%)을 계산한다 (예: 영업이익률, 부채비율, 매출액대비).",
            calculate_ratio,
            {"numerator": {"type": "number"}, "denominator": {"type": "number"}, "label": {"type": "string"}},
            ["numerator", "denominator"],
        ),
        _calc_tool(
            "calculate_cagr", "연평균성장률(CAGR)을 계산한다.",
            calculate_cagr,
            {"begin_value": {"type": "number"}, "end_value": {"type": "number"}, "n_years": {"type": "number"}},
            ["begin_value", "end_value", "n_years"],
        ),
    ]


def build_all_tools(retriever, manifest: list[ManifestRow], correction_index: dict[str, CorrectionRecord]) -> list[ToolDef]:
    return [
        make_search_disclosures_tool(retriever),
        make_get_correction_history_tool(manifest, correction_index),
        make_get_latest_report_tool(manifest, correction_index),
        *build_calculation_tools(),
    ]


def dispatch_tool_call(tools: list[ToolDef], name: str, arguments: dict) -> dict:
    tool = next((t for t in tools if t.name == name), None)
    if tool is None:
        return {"error": f"알 수 없는 tool: {name}"}
    try:
        return tool.handler(**arguments)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
