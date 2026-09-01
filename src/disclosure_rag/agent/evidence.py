"""Evidence Pack Builder (§63~64).

최종 HCX 에 검색 결과를 그대로 dump 하지 않고 구조화한다. Citation Provenance
(report_id/chunk_id/section_path/filing_date/correction 상태)를 끝까지 보존해
최종 답변에서 근거를 사용자에게 보여줄 수 있게 한다 (§64)."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field

from disclosure_rag.agent.agent_loop import AgentTrace
from disclosure_rag.agent.derived_facts import derive, derive_calculations
from disclosure_rag.common.korean_number import describe_amount, normalize_unit_text

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


# 회사별 "제N기 -> 연도" 오프셋. scripts/build_fiscal_offsets.py 가 만든다.
# 청크마다 최대 기수를 당기로 추측하면 일치율이 75.8% 밖에 안 된다(21개사 전수).
# 회사별 최빈값으로 한 번 정해 두면 정확하다.
_FISCAL_OFFSETS: dict[str, int] | None = None


def _fiscal_offsets() -> dict[str, int]:
    global _FISCAL_OFFSETS
    if _FISCAL_OFFSETS is None:
        table: dict[str, int] = {}
        path = Path(__file__).resolve().parents[3] / "config" / "fiscal_offsets.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for company, info in (data.get("companies") or {}).items():
                table[company] = int(info["offset"])
        except Exception:  # noqa: BLE001  파일이 없어도 동작해야 한다
            table = {}
        _FISCAL_OFFSETS = table
    return _FISCAL_OFFSETS


_GI_RE = re.compile(r"제\s?(\d{1,3})\s?기")


def period_note(company: str | None, period: str | None,
                filing_date: str | None, text: str | None) -> str:
    """기간 표기를 실제 연도로 풀어 준다.

    공시 표는 연도 대신 `당기/전기` 나 `제55기` 로 적는다. 실측(2026-09-01):
    당기·전기 표기 35,300건, 기수 표기 16,711건. 질문은 연도로 묻는데 근거는
    저렇게 적혀 있으면 모델이 연결을 못 짓는다.

    기수 환산은 **회사별 오프셋을 아는 경우에만** 한다. 모르면 아무 말도
    하지 않는다 — 틀린 환산을 근거에 박으면 모델이 그걸 믿는다.
    """
    base = (period or "")[:4] or (filing_date or "")[:4]
    if not base.isdigit():
        return ""
    year = int(base)
    parts = [f"당기 = {year}년, 전기 = {year - 1}년"]

    offset = _fiscal_offsets().get(company or "")
    if offset is not None and text and _GI_RE.search(text):
        current = year + offset          # offset 은 음수: 기수 = 연도 + offset
        if 1 <= current <= 200:
            parts.append(
                f"제{current}기 = {year}년, 제{current - 1}기 = {year - 1}년")
    return "보고 기준: " + " / ".join(parts) + "\n"


# 재무제표는 두 벌이다. 같은 항목이 서로 다른 값으로 실린다.
#   연결재무제표  자회사까지 합친 것
#   별도재무제표  그 회사 하나만
# 실측(2026-09-01): 연결 352,161행 / 별도 321,650행 / 재무제표 절이 아닌 것 396,675행.
#
# "연결" 만 보고 판정하면 안 된다 —
#   `XII. 상세표 > 1. 연결대상 종속회사 현황(상세)`  종속회사 목록이지 재무제표가 아니다
#   `연결 내부회계관리제도 감사 또는 검토의견`         재무수치가 아니다
# 반대로 공백이 섞인 `(첨부)연 결 재 무 제 표` 는 반드시 잡아야 한다(22,162행).
_STATEMENT_SPACES = re.compile(r"[\s ​　]")


def statement_kind(section_path) -> str:
    """이 근거가 연결재무제표인가 별도재무제표인가. 재무제표 절이 아니면 빈 문자열.

    판정 순서
      1. 유니코드 정규화(NFC) — 한글이 두 방식으로 저장될 수 있다
      2. 공백류 전부 제거 — 일반 공백, 전각 공백, 무너비 공백까지
      3. "연결재무제표" 포함이면 연결
      4. 아니고 "재무제표" 포함이면 별도
      5. 둘 다 없으면 빈 문자열 -> **줄을 안 붙인다**

    5번이 중요하다. 「주주에 관한 사항」에 "별도재무제표" 를 붙이면 모델이
    최대주주 수치를 그 회사 재무제표 값으로 읽는다 — 오늘 고친 게 도로 망가진다.
    """
    joined = unicodedata.normalize("NFC", " > ".join(str(p) for p in (section_path or [])))
    flat = _STATEMENT_SPACES.sub("", joined)
    if "연결재무제표" in flat:
        return "연결재무제표"
    if "재무제표" in flat:
        return "별도재무제표"
    return ""


def statement_note(section_path) -> str:
    kind = statement_kind(section_path)
    return f"재무제표 구분: {kind}\n" if kind else ""


def unit_note(unit_hint: str | None) -> str:
    """이 표의 금액 단위를 한 줄로 알려준다.

    공시 표는 숫자만 적고 단위는 표 머리의 별도 줄에 둔다. 그 줄을 안 주면
    같은 3,112,850 이 백만원 표에서는 3조, 천원 표에서는 31억이 된다
    (1,000배). 실측(2026-09-01): 크래프톤 영업비용이 정답 3,112,850(백만원)
    대신 다른 표의 255,698,325천원 으로 나갔다.
    """
    unit = normalize_unit_text(unit_hint)
    if not unit:
        return ""
    return f"표 단위: {unit} (이 표의 숫자는 {unit} 단위입니다)\n"


def _fact_owner(row: dict) -> str:
    """이 수치의 주인 이름. 회사 자신이 아니면 그렇게 적는다."""
    owner = row.get("value_owner")
    company = row.get("company")
    if owner and not row.get("value_owner_is_company", True):
        return f"{owner} — {company} 공시에 실린 제3자 수치"
    return company or (owner or "?")


def third_party_note(section_path, owner: str | None = None, company: str | None = None) -> str:
    """이 근거의 수치가 그 회사 것이 아니면 한 줄로 알려준다.

    `owner` 가 있으면 **실제 주인 이름**을 적는다. 수치사전에 그 조각의 표
    주인이 기록돼 있으면 거기서 온다. 이름을 알면 두 가지가 동시에 된다 —
    그 회사 값으로 잘못 쓰는 것을 막고, "최대주주의 매출액은?" 에는 답한다.
    """
    if owner and owner != company:
        return (f"⚠ 이 표의 수치는 **{owner}의 것**이다({company} 공시에 실린 제3자 "
                f"수치). {company} 값으로 쓰면 안 된다.\n")
    joined = " > ".join(str(part) for part in (section_path or []))
    if any(marker in joined for marker in _THIRD_PARTY_SECTIONS):
        return ("⚠ 주의: 이 절의 재무수치는 **이 회사가 아니라 최대주주·출자대상 등 "
                "다른 법인의 것**일 수 있다. 이 회사 값으로 쓰면 안 된다.\n")
    return ""


def _evidence_block(idx: int, *, company, report_name, filing_date, period,
                    section_path, is_correction, is_latest, text,
                    report_id, chunk_id, owner: str | None = None,
                    unit_hint: str | None = None) -> str:
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
        f"{period_note(company, period, filing_date, text)}"
        f"{statement_note(section_path)}"
        f"{unit_note(unit_hint)}"
        f"{third_party_note(section_path, owner, company)}"
        f"내용:\n{text}\n"
        f"report_id: {report_id}\n"
        f"chunk_id: {chunk_id}\n"
    )


# 집계 질문에서 프롬프트에 나열할 fact 행 수. 계산은 전체로 하고 표시만 줄인다.
_FACTS_SHOWN = 8


def _fact_value(row: dict):
    return row.get("value") or row.get("value_text")


def _mark_aggregate(fact_rows: list[dict], aggregation: str, *,
                    compare_winner: bool = False) -> list[str]:
    """최대/최소와 **승자 판정**을 우리가 계산해서 명시한다.

    실측 실패(v2 38문항 S011): Facts 가 한전기술 계약금액 10건을 줬고 그 안에
    최댓값 1,250,850,298,678 이 있었는데, 모델은 5번째 값 373,449,426,066 을
    골랐다. 목록에서 최댓값을 고르는 건 모델이 자주 틀린다 — 파이썬이 하면
    틀릴 일이 없다.

    승자 판정을 여기 넣은 이유(2026-09-01 실측):
        G0066 "삼성중공업과 삼성전자 중 최대 계약금액이 더 큰 쪽은?"
          삼성중공업  4,571,600,000,000   <- 두 값 다 정확히 찾았다
          삼성전자   22,764,764,160,000
          답변: "삼성중공업의 계약금액이 더 큽니다"   <- 틀렸다
    회사별 최댓값까지는 파이썬이 계산해 주고 **거기서 멈춰서** 대소 비교를
    모델에 넘겼다. 0이 13개 붙은 수를 눈으로 견주다 틀린 것이다.
    비교 문항 60건 중 계산이 붙은 24건은 96%, 안 붙은 36건은 50%였다.

    회사별 대표값을 고르는 규칙
        aggregation=max/min  그 회사 값들의 극값
        aggregation=none     **가장 최신 공시의 값** (Facts 가 날짜 내림차순으로
                             주므로 첫 행). "지금 얼마인가" 를 묻는 것으로 본다.
    """
    if aggregation not in ("max", "min") and not compare_winner:
        return []

    groups: dict[tuple, list[dict]] = {}
    for row in fact_rows:
        if row.get("value_num") is None:
            continue
        key = (row.get("company"), row.get("item") or row.get("key_norm"))
        groups.setdefault(key, []).append(row)
    if not groups:
        return []

    lines: list[str] = []
    # (항목 -> [(회사, 대표행)]) 로 모아 두었다가 회사 간 비교에 쓴다
    by_item: dict[str, list[tuple[str, dict]]] = {}

    if aggregation in ("max", "min"):
        label = "최대" if aggregation == "max" else "최소"
        pick = max if aggregation == "max" else min
        for (company, item), rows in groups.items():
            best = pick(rows, key=lambda r: r["value_num"])
            lines.append(
                f"▶ {company} {item} {label}값: {_fact_value(best)}"
                f" [report_id: {best.get('report_id') or best.get('doc_id')}]"
            )
            by_item.setdefault(item, []).append((company, best))
    else:
        # 승자만 묻는 질문 — 회사마다 최신 값을 대표로 뽑는다.
        for (company, item), rows in groups.items():
            best = rows[0]
            lines.append(
                f"▶ {company} {item}: {_fact_value(best)}"
                f" [report_id: {best.get('report_id') or best.get('doc_id')}]"
            )
            by_item.setdefault(item, []).append((company, best))

    if compare_winner:
        lines.extend(_mark_winner(by_item, aggregation))
    return lines


def _mark_winner(by_item: dict[str, list[tuple[str, dict]]], aggregation: str) -> list[str]:
    """항목별로 회사 간 승자를 계산한다. 모델은 이 줄을 옮겨 적기만 한다."""
    smaller = aggregation == "min"
    out: list[str] = []
    for item, pairs in by_item.items():
        # 같은 회사가 두 번 들어오지 않게 (항목이 같으면 회사당 1행)
        uniq: dict[str, dict] = {}
        for company, row in pairs:
            if company and company not in uniq:
                uniq[company] = row
        if len(uniq) < 2:
            continue
        ranked = sorted(uniq.items(), key=lambda kv: kv[1]["value_num"], reverse=not smaller)
        top_company, top_row = ranked[0]
        second_company, second_row = ranked[1]
        if top_row["value_num"] == second_row["value_num"]:
            names = ", ".join(c for c, _r in ranked
                              if _r["value_num"] == top_row["value_num"])
            out.append(f"▶▶ {item} 비교 결과: 동일 ({names})")
            continue
        word = "작다" if smaller else "크다"
        gap = abs(top_row["value_num"] - second_row["value_num"])
        out.append(
            f"▶▶ {item} 비교 결과: {top_company}가 더 {word}"
            f" ({_fact_value(top_row)} vs {_fact_value(second_row)}, 차이 {gap:,.0f})"
        )
        # 자릿수가 크면 사람이 읽는 단위로도 적어 준다. 환산도 파이썬이 한다.
        readable = [
            f"{company} {describe_amount(row['value_num'], row.get('value_unit') or '원')}"
            for company, row in ranked[:3]
            if describe_amount(row["value_num"], row.get("value_unit") or "원")
        ]
        if readable and abs(top_row["value_num"]) >= 100_000_000:
            out.append("▶▶ 단위 환산: " + " / ".join(readable))
        if len(ranked) > 2:
            order = " > ".join(c for c, _r in ranked)
            out.append(f"▶▶ {item} 순위: {order}")
    return out


def build_evidence_pack_from_retrieval(
    question: str, chunks_with_scores, *, facts=(), aggregation: str = "none",
    max_chars: int | None = None, scope_note: str = "",
    chunk_owners: dict | None = None, compare_winner: bool = False,
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
            owner=(chunk_owners or {}).get(getattr(chunk, "chunk_id", None)),
            unit_hint=getattr(chunk, "unit_hint", None),
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
        marks = _mark_aggregate(fact_rows, aggregation, compare_winner=compare_winner)
        # 합계·평균·차이·건수·순위 — 질문이 요구한 것만 파이썬이 계산한다.
        marks = marks + derive(question, list(fact_rows))
        marks = marks + derive_calculations(question, list(fact_rows))
        header = "[FACT] 공시 표에서 직접 추출한 확정 값입니다."
        if marks:
            header += ("\n아래 ▶ 와 ▶▶ 는 **계산이 끝난 값**입니다. 목록에서 직접 고르거나 "
                       "직접 비교하지 말고 그대로 쓰세요.\n" + "\n".join(marks))
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
