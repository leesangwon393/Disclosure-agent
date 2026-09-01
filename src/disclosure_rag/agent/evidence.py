"""Evidence Pack Builder (§63~64).

최종 HCX 에 검색 결과를 그대로 dump 하지 않고 구조화한다. Citation Provenance
(report_id/chunk_id/section_path/filing_date/correction 상태)를 끝까지 보존해
최종 답변에서 근거를 사용자에게 보여줄 수 있게 한다 (§64)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from disclosure_rag.agent.agent_loop import AgentTrace

_CALC_TOOL_NAMES = {"calculate_growth_rate", "calculate_ratio", "calculate_cagr"}


@dataclass
class Citation:
    chunk_id: str
    report_id: str
    company: str | None
    report_name: str | None
    filing_date: str | None
    section_path: list[str]
    is_correction: bool
    is_latest: bool | None


@dataclass
class EvidencePack:
    question: str
    prompt_text: str  # HCX 에 그대로 넘길 최종 텍스트
    citations: list[Citation] = field(default_factory=list)
    tool_results_summary: list[dict] = field(default_factory=list)  # 계산류 등, 검증용


def build_evidence_pack(trace: AgentTrace) -> EvidencePack:
    lines = [f"[USER QUESTION]\n{trace.question}\n"]
    citations: list[Citation] = []
    tool_results_summary: list[dict] = []
    evidence_idx = 1

    for tc in trace.tool_calls:
        if tc.name == "search_disclosures":
            for item in tc.result.get("results", []):
                status = "정정본" if item.get("is_correction") else "원본"
                if item.get("is_latest"):
                    status += " (최신)"
                lines.append(
                    f"[EVIDENCE {evidence_idx}]\n"
                    f"회사: {item.get('company')}\n"
                    f"공시명: {item.get('report_name')}\n"
                    f"공시일: {item.get('filing_date')}\n"
                    f"기간: {item.get('period')}\n"
                    f"Section: {' > '.join(item.get('section_path') or [])}\n"
                    f"정정 상태: {status}\n"
                    f"{third_party_note(item.get('section_path'))}"
                    f"내용:\n{item.get('text')}\n"
                    f"report_id: {item.get('report_id')}\n"
                    f"chunk_id: {item.get('chunk_id')}\n"
                )
                citations.append(Citation(
                    chunk_id=item.get("chunk_id"), report_id=item.get("report_id"),
                    company=item.get("company"), report_name=item.get("report_name"),
                    filing_date=item.get("filing_date"), section_path=item.get("section_path") or [],
                    is_correction=bool(item.get("is_correction")), is_latest=item.get("is_latest"),
                ))
                evidence_idx += 1
        elif tc.name in _CALC_TOOL_NAMES:
            lines.append(f"[TOOL RESULT]\n계산 종류: {tc.name}\n입력 값: {tc.arguments}\n결과: {tc.result}\n")
            tool_results_summary.append({"tool": tc.name, "arguments": tc.arguments, "result": tc.result})
        else:
            lines.append(f"[TOOL RESULT]\n{tc.name}: {json.dumps(tc.result, ensure_ascii=False)}\n")
            tool_results_summary.append({"tool": tc.name, "arguments": tc.arguments, "result": tc.result})

    if not citations and not tool_results_summary:
        lines.append("[EVIDENCE 없음 — 검색/도구 호출로 근거를 확보하지 못했습니다]")

    return EvidencePack(
        question=trace.question, prompt_text="\n".join(lines),
        citations=citations, tool_results_summary=tool_results_summary,
    )


# ===========================================================================
# 신 파이프라인용 조립 (2026-08-30)
# ===========================================================================
#
# 위 `build_evidence_pack(trace)` 는 **HCX 가 호출한 도구 결과**에서 만든다.
# 신 파이프라인은 도구 호출을 하지 않고 결정론적으로 검색하므로, (청크, 점수)
# 목록과 Facts 행에서 직접 만들어야 한다.
#
# `[EVIDENCE n]` 블록 형식은 **한 글자도 바꾸지 않는다.** 답변 프롬프트가 이
# 형식에 맞춰 튜닝돼 있고(자릿수 오류·중복 값 나열 등 실측 교훈), 형식이
# 바뀌면 그 튜닝이 무효가 된다.


# 「VII. 주주에 관한 사항」 같은 절에는 **그 회사가 아닌 법인**의 재무현황이
# 실린다(최대주주 및 특수관계인 현황). 사람은 표 제목을 보고 구분하지만,
# 청크만 떼어 놓으면 회사 이름과 숫자만 남아 그 회사 값처럼 보인다.
# 실제로 그렇게 읽혔다 — 삼성전자 매출을 삼성SDI 사업보고서의 최대주주
# 재무현황에서 가져온 사고가 여기서 나왔다.
_THIRD_PARTY_SECTIONS = ("주주에 관한 사항", "타법인출자", "타법인 출자", "계열회사")


def _fact_owner(row: dict) -> str:
    """이 수치의 주인 이름. 회사 자신이 아니면 그렇게 적는다."""
    owner = row.get("value_owner")
    company = row.get("company")
    if owner and not row.get("value_owner_is_company", True):
        return f"{owner} — {company} 공시에 실린 제3자 수치"
    return company or (owner or "?")


def third_party_note(section_path) -> str:
    """이 근거의 수치가 그 회사 것이 아닐 수 있으면 한 줄 경고를 만든다."""
    joined = " > ".join(str(part) for part in (section_path or []))
    if any(marker in joined for marker in _THIRD_PARTY_SECTIONS):
        return ("⚠ 주의: 이 절의 재무수치는 **이 회사가 아니라 최대주주·출자대상 등 "
                "다른 법인의 것**이다. 이 회사 값으로 쓰면 안 된다.\n")
    return ""


def _evidence_block(idx: int, *, company, report_name, filing_date, period,
                    section_path, is_correction, is_latest, text,
                    report_id, chunk_id) -> str:
    status = "정정본" if is_correction else "원본"
    if is_latest:
        status += " (최신)"
    return (
        f"[EVIDENCE {idx}]\n"
        f"회사: {company}\n"
        f"공시명: {report_name}\n"
        f"공시일: {filing_date}\n"
        f"기간: {period}\n"
        f"Section: {' > '.join(section_path or [])}\n"
        f"정정 상태: {status}\n"
        f"{third_party_note(section_path)}"
        f"내용:\n{text}\n"
        f"report_id: {report_id}\n"
        f"chunk_id: {chunk_id}\n"
    )


# 집계 질문에서 프롬프트에 나열할 fact 행 수. 계산은 전체로 하고 표시만 줄인다.
_FACTS_SHOWN = 8


def _mark_aggregate(fact_rows: list[dict], aggregation: str) -> list[str]:
    """최대/최소를 **우리가 계산해서** 명시한다.

    실측 실패(v2 38문항 S011): Facts 가 한전기술 계약금액 10건을 줬고 그 안에
    최댓값 1,250,850,298,678 이 있었는데, 모델은 5번째 값 373,449,426,066 을
    골랐다. 목록에서 최댓값을 고르는 건 모델이 자주 틀린다 — 파이썬이 하면
    틀릴 일이 없다.

    비교 질문(회사 2곳)이 있으므로 **회사·항목별로** 따로 계산한다.
    """
    if aggregation not in ("max", "min"):
        return []
    groups: dict[tuple, list[dict]] = {}
    for row in fact_rows:
        value = row.get("value_num")
        if value is None:
            continue
        key = (row.get("company"), row.get("item") or row.get("key_norm"))
        groups.setdefault(key, []).append(row)

    label = "최대" if aggregation == "max" else "최소"
    pick = max if aggregation == "max" else min
    lines = []
    for (company, item), rows in groups.items():
        best = pick(rows, key=lambda r: r["value_num"])
        lines.append(
            f"▶ {company} {item} {label}값: {best.get('value') or best.get('value_text')}"
            f" [report_id: {best.get('report_id') or best.get('doc_id')}]"
        )
    return lines


def build_evidence_pack_from_retrieval(
    question: str, chunks_with_scores, *, facts=(), aggregation: str = "none",
    max_chars: int | None = None, scope_note: str = "",
) -> EvidencePack:
    """검색 결과에서 Evidence Pack 을 만든다.

    facts
        Facts(sqlite) 조회 결과. **점수를 붙이지 않는다** — 순위 대상이 아니다.
        검색 근거와 섞이지 않게 `[FACT]` 블록으로 따로 넣는다.

    max_chars
        프롬프트 길이 상한. open 질문은 청크 24개가 들어와 수만 자가 될 수
        있는데, 실측상 긴 근거에서 HCX 가 답을 못 찾는 일이 있었다
        (`diag_answer.py`: 13,542자에서 thinking ON 이면 "확인할 수 없습니다").
        상한을 넘으면 **뒤쪽 근거부터** 자른다 — 앞쪽이 관련도가 높다.
        잘린 사실은 프롬프트에 명시해 모델이 '전부 봤다'고 착각하지 않게 한다.
    """
    lines = [f"[USER QUESTION]\n{question}\n"]
    # 전수 확인 결과는 **근거보다 먼저** 둔다. 뒤에 붙이면 긴 근거에 묻혀서
    # 모델이 못 보고 "확인할 수 없습니다"로 돌아간다.
    if scope_note:
        lines.append(scope_note)
    citations: list[Citation] = []
    used = sum(len(x) for x in lines)
    truncated = 0

    for idx, item in enumerate(chunks_with_scores, start=1):
        chunk = item[0] if isinstance(item, tuple) else item
        block = _evidence_block(
            idx,
            company=getattr(chunk, "company", None),
            report_name=getattr(chunk, "report_name", None),
            filing_date=getattr(chunk, "filing_date", None),
            period=getattr(chunk, "period", None),
            section_path=getattr(chunk, "section_path", None),
            is_correction=bool(getattr(chunk, "is_correction", False)),
            is_latest=getattr(chunk, "is_latest", None),
            text=getattr(chunk, "raw_text", None) or getattr(chunk, "text", "") or "",
            report_id=getattr(chunk, "report_id", None),
            chunk_id=getattr(chunk, "chunk_id", None),
        )
        if max_chars is not None and used + len(block) > max_chars and citations:
            truncated = len(chunks_with_scores) - idx + 1
            break
        lines.append(block)
        used += len(block)
        citations.append(Citation(
            chunk_id=getattr(chunk, "chunk_id", None),
            report_id=getattr(chunk, "report_id", None),
            company=getattr(chunk, "company", None),
            report_name=getattr(chunk, "report_name", None),
            filing_date=getattr(chunk, "filing_date", None),
            section_path=list(getattr(chunk, "section_path", None) or []),
            is_correction=bool(getattr(chunk, "is_correction", False)),
            is_latest=getattr(chunk, "is_latest", None),
        ))

    fact_rows = list(facts or [])
    tool_results_summary: list[dict] = []
    if fact_rows:
        # 집계 질문(최대/최소)은 회사·항목당 최대 50건을 조회한다. 그 50건을
        # 전부 프롬프트에 넣으면 비교 질문에서 100줄이 되어 프롬프트가 커지고,
        # HCX 응답이 느려진다(실측: 문항당 18초 -> 80~170초).
        #
        # **계산은 전부로 하고, 보여주는 건 몇 줄만 한다.** 최댓값은 이미 ▶ 로
        # 확정해 주므로 나머지 행은 참고용이다.
        shown = fact_rows
        hidden = 0
        if aggregation in ("max", "min") and len(fact_rows) > _FACTS_SHOWN:
            keyed = [r for r in fact_rows if r.get("value_num") is not None]
            keyed.sort(key=lambda r: r["value_num"], reverse=(aggregation == "max"))
            shown = keyed[:_FACTS_SHOWN]
            hidden = len(fact_rows) - len(shown)
        # Facts 는 표에서 확정된 값이라 검색 근거보다 신뢰도가 높다. 그 사실을
        # 프롬프트에 명시한다.
        # 값의 주인을 **회사가 아니라 실제 주인 이름으로** 적는다. 예전에는
        # 최대주주 재무현황의 값에도 보고서를 낸 회사 이름이 붙어서
        # "KB금융 자산총계 464,418" 로 보였다(실제로는 국민연금공단 값).
        body = "\n".join(
            f"- {row.get('item') or row.get('key_norm')}: {row.get('value')or row.get('value_text')}"
            f" ({_fact_owner(row)} / {row.get('report_name')} / {row.get('filing_date')})"
            f" [report_id: {row.get('report_id') or row.get('doc_id')}]"
            for row in shown
        )
        if hidden:
            # 예전 문구는 "위 ▶ 계산에 포함되었으나 생략" 이었는데, 생략된
            # 행 중 value_num 이 없는 것은 ▶ 계산에도 안 들어간다. 사실과
            # 다른 안내였다(2026-08-31 발견).
            body += f"\n- (그 외 {hidden}건 생략)"
        marks = _mark_aggregate(fact_rows, aggregation)
        header = "[FACT] 공시 표에서 직접 추출한 확정 값입니다."
        if marks:
            header += ("\n아래 ▶ 는 **계산이 끝난 값**입니다. 목록에서 직접 고르지 말고 "
                       "▶ 값을 그대로 쓰세요.\n" + "\n".join(marks))
        lines.append(header + "\n" + body + "\n")
        tool_results_summary.append({"tool": "fact_lookup", "arguments": {},
                                     "result": {"rows": fact_rows}})

    if truncated:
        lines.append(f"[안내] 근거 {truncated}건이 길이 제한으로 생략되었습니다. "
                     "위에 제시된 근거만으로 답하세요.\n")

    if not citations and not tool_results_summary:
        lines.append("[EVIDENCE 없음 — 검색으로 근거를 확보하지 못했습니다]")

    return EvidencePack(question=question, prompt_text="\n".join(lines),
                        citations=citations, tool_results_summary=tool_results_summary)
