"""평가용 API 서버 — 대회 제출 필수 항목.

규격 (디스코드 공지 확정)
------------------------
    GET {endpoint}/answer?question_id={id}&question={질의}

  - 경로 `/answer` 고정, **인증 헤더 없음**
  - 응답 JSON 5개 필드 **전부 string**: question_id, question, retrieved_context,
    think_trace, answer  (빈 문자열 허용, think_trace 형식 규정 없음)
  - **HTTP 80** 기본, HTTPS 선택(self-signed 무방). 도메인·정식 인증서 불필요
  - **sequential 호출** (동시 요청 없음)
  - 타임아웃 **300초**, 타임아웃/5xx 시 **최대 2회 재시도**

이 규격에서 나오는 설계 결정 세 가지
-----------------------------------
1. **멱등성.** 같은 question_id 가 최대 3번 올 수 있다. 재계산하면 답이 흔들리고
   크레딧도 3배 든다 -> 결과를 캐시하고 같은 답을 돌려준다.

2. **실패 종류에 따라 상태 코드를 다르게 낸다. 이게 점수를 가른다.**
   주최측 재시도는 *우리 서버가 응답을 못 할 때만* 작동한다. 그래서
     - CLOVA 429/5xx 같은 **일시적 실패** -> **5xx 를 반환해 재시도를 유도한다.**
       여기서 200 OK 에 빈 답을 담아 보내면 그 문항은 그대로 0점이다.
     - **근거를 못 찾은 것**(우리가 정상 동작한 결과) -> **200 OK** 로
       "공시에서 확인되지 않음" 을 반환한다. 평가 항목 「정보한계 대응」의 정답 경로다.
   이 둘을 섞으면 안 된다.

3. **하드 타임아웃 가드.** 300초를 넘기면 주최측이 끊고 재시도한다. 그 전에
   우리가 스스로 멈추고 **그때까지 찾은 근거로** 답하는 편이 낫다(부분 점수).

실행:
    uvicorn disclosure_rag.serving.api:app --host 0.0.0.0 --port 80
    (개발: --port 8000, 환경변수 ARTIFACTS 로 산출물 경로 변경 — 기본 artifacts_v2)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import threading
import time
import traceback
from typing import Any

logger = logging.getLogger(__name__)

# 300초 규격 대비 여유. 이 시간을 넘기면 그때까지 모은 근거로 답한다.
SOFT_BUDGET_SEC = float(os.environ.get("ANSWER_BUDGET_SEC", "240"))
# 현행 빌드는 artifacts_v2 다. 측정 스크립트(score_answers/score_abstention)도
# 전부 이 값을 기본으로 쓴다 — 여기 기본값이 다르면 "측정한 성능"과 "서비스되는
# 성능"이 갈린다. 구버전 artifacts 에는 registry/ 와 facts_periodic_v2/ 가 없어서
# 범위 게이트가 조용히 꺼지고 정기공시 수치 조회가 통째로 빠진다.
ARTIFACTS = os.environ.get("ARTIFACTS", "artifacts_v2")
NO_LLM = os.environ.get("NO_LLM", "").lower() in ("1", "true", "yes")
NOT_FOUND_ANSWER = "제공된 공시에서 확인되지 않습니다."

_state: dict[str, Any] = {"bundle": None, "ask": None, "askv2": None, "error": None}
_cache: dict[str, dict] = {}
_lock = threading.Lock()


class TransientUpstreamError(RuntimeError):
    """CLOVA 429/5xx 등 재시도로 해결될 수 있는 실패. 5xx 로 내보낸다."""


def _looks_transient(exc: BaseException) -> bool:
    s = f"{type(exc).__name__} {exc}".lower()
    return any(t in s for t in ("429", "too many requests", "rate limit", "timeout",
                                "timed out", "502", "503", "504", "connection"))


def _warm() -> None:
    """인덱스를 올린다. 45만 chunk 라 기동에 시간이 걸리므로 요청 전에 끝내둔다."""
    from disclosure_rag.retrieval.index_bundle import load_bundle

    t0 = time.time()
    bundle = load_bundle(ARTIFACTS)

    # 리랭커 연결 — 314문항 실측에서 전 지표 1위였다.
    #   hybrid_rrf  hit@5 0.583 / MRR 0.465
    #   +rerank     hit@5 0.618 / MRR 0.503   (문항당 +4.8초)
    # 주최측 타임아웃이 300초라 4.8초는 1.6% 다. 끄려면 USE_RERANKER=0.
    if os.environ.get("USE_RERANKER", "1") not in ("0", "false", "no"):
        try:
            tr = time.time()
            from disclosure_rag.retrieval.reranker import CrossEncoderReranker
            bundle.retriever.reranker = CrossEncoderReranker()
            logger.info("[SERVE] 리랭커 적재 %.0fs", time.time() - tr)
        except Exception as e:  # noqa: BLE001
            # 리랭커가 없어도 서비스는 되어야 한다 — 품질만 조금 떨어진다.
            logger.warning("[SERVE] 리랭커 적재 실패(%s) — 리랭커 없이 진행", type(e).__name__)

    _state["bundle"] = bundle

    if PIPELINE == "v2":
        try:
            tv = time.time()
            _prepare_v2(bundle)
            logger.info("[SERVE] v2 파이프라인 적재 %.0fs", time.time() - tv)
        except Exception as e:  # noqa: BLE001
            logger.error("[SERVE] v2 준비 실패(%s) — 아래 경로로 진행\n%s",
                         type(e).__name__, traceback.format_exc())
            _state["askv2"] = None

    if USE_AGENT:
        try:
            ta = time.time()
            _prepare_agent(bundle)
            logger.info("[SERVE] 에이전트 적재 %.0fs", time.time() - ta)
        except Exception as e:  # noqa: BLE001
            logger.error("[SERVE] 에이전트 준비 실패(%s) — 단순 경로로 진행\n%s",
                         type(e).__name__, traceback.format_exc())
            _state["tools"] = None

    logger.info("[SERVE] 인덱스 준비 완료 %.0fs — 검색 경로 %s / 융합 %s / 리랭커 %s",
                time.time() - t0, bundle.modes, bundle.retriever.fusion,
                "ON" if bundle.retriever.reranker else "OFF")


USE_AGENT = os.environ.get("USE_AGENT", "0") not in ("0", "false", "no")
CORPUS_ROOT = os.environ.get("CORPUS_ROOT", "corpus")

# 답변 경로. 기본값이 v2 다 — 측정을 전부 이걸로 했다.
#   v2      결정론적 신 파이프라인 (ask_v2). 규칙이 무엇을 몇 번 검색할지 정한다
#   agent   구 경로 (ask.py). HCX 가 도구를 부를지 스스로 정한다
#   simple  검색 1회 -> HCX 1회
# 앞의 것이 **예외로 실패하면** 뒤로 내려간다. 다만 v2 가 '거부'로 답한 것은
# 실패가 아니라 정상 동작이므로 폴백하지 않는다 — 폴백하면 거부 설계가 무의미해진다.
PIPELINE = os.environ.get("PIPELINE", "v2")


def _prepare_v2(bundle) -> None:
    """신 파이프라인을 조립한다.

    구 경로(`_prepare_agent`)와 무엇이 다른가: 저쪽은 HCX 에게 도구 목록을 주고
    무엇을 부를지 맡긴다. 실측상 그게 잘 안 됐다 — 100문항 중 54건이 도구를
    1회만 부르고 끝냈고, 정정 이력 도구는 시켜도 안 불렀다. 이쪽은 규칙이
    무엇을 몇 번 검색할지 정하고, HCX 는 답변 문장만 만든다.
    """
    from disclosure_rag.agent.ask_v2 import AskV2
    from disclosure_rag.agent.dual_channel import DualChannelRetriever
    from disclosure_rag.agent.field_schema import FieldSchema
    from disclosure_rag.agent.hcx_client import HCXClient
    from disclosure_rag.agent.query_plan import PlanValidator, RulePlanBuilder
    from disclosure_rag.common.manifest_loader import load_manifest
    from disclosure_rag.common.unicode_utils import PathResolver
    from disclosure_rag.correction.correction_graph_builder import build_correction_index
    from disclosure_rag.entity.entity_extractor import EntityExtractor
    from disclosure_rag.entity.entity_registry import EntityRegistry

    schema_path = Path("config/field_schema.json")
    schema = FieldSchema.load(schema_path) if schema_path.exists() else FieldSchema.empty()

    registry_path = Path(ARTIFACTS) / "registry" / "entities.json"
    registry = EntityRegistry.load(registry_path) if registry_path.exists() else None
    if registry is None:
        # 범위 게이트가 빠지면 코퍼스 밖 회사 질문을 조기 차단하지 못한다.
        # 서비스는 되지만 거부 정확도가 떨어지므로 반드시 로그로 남긴다.
        logger.warning("[SERVE] Entity Registry 없음(%s) — 범위 게이트 없이 진행",
                       registry_path)

    manifest = load_manifest(CORPUS_ROOT)
    corrections = build_correction_index(manifest, PathResolver(CORPUS_ROOT))

    _state["hcx"] = _state.get("hcx") or HCXClient()
    _state["askv2"] = AskV2(
        client=_state["hcx"],
        dual_retriever=DualChannelRetriever(
            bundle.retriever, bundle.fact_store,
            correction_index=corrections, manifest=manifest),
        plan_builder=RulePlanBuilder(
            schema=schema,
            extractor=EntityExtractor(corpus_root=CORPUS_ROOT,
                                      metric_terms_path="config/metric_terms.txt")),
        plan_validator=PlanValidator(registry=registry, schema=schema),
        registry=registry,
        parent_expander=bundle.parent_expander,
        thinking_policy=os.environ.get("THINKING", "off"),
    )
    logger.info("[SERVE] v2 준비 완료 — 범위게이트 %s / 공시유형 %d종",
                "ON" if registry else "OFF", len(schema.kinds()))


def _prepare_agent(bundle) -> None:
    """도구 호출 에이전트를 준비한다. 실패해도 단순 경로로 서비스는 계속된다.

    왜 필요한가: `_answer_once` 의 단순 경로는 "검색 1회 -> HCX 1회"라서
    비교·연산("A와 B 중 어디가 큰가")과 복합추론("정정 전후 무엇이 바뀌었나")을
    풀 수 없다. 주최측 6유형 중 4개가 여기 걸린다.
    실측(2026-08-29): HCX-007 이 tool calling 을 정확히 수행함을 확인했다.
    """
    from disclosure_rag.agent.tools import build_all_tools
    from disclosure_rag.agent.fact_tools import build_fact_tools
    from disclosure_rag.common.manifest_loader import load_manifest
    from disclosure_rag.common.unicode_utils import PathResolver
    from disclosure_rag.correction.correction_graph_builder import build_correction_index
    from disclosure_rag.entity.entity_extractor import EntityExtractor

    manifest = load_manifest(CORPUS_ROOT)
    resolver = PathResolver(CORPUS_ROOT)
    correction_index = build_correction_index(manifest, resolver)

    tools = build_all_tools(bundle.retriever, manifest, correction_index)
    if bundle.fact_store is not None:
        # facts 층(2.5만건)에 답이 있는 질문은 본문 검색보다 이쪽이 정확하다.
        tools += build_fact_tools(bundle.fact_store)

    _state["tools"] = tools
    _state["entity_extractor"] = EntityExtractor(
        corpus_root=CORPUS_ROOT, metric_terms_path="config/metric_terms.txt")
    logger.info("[SERVE] 에이전트 준비 완료 — 도구 %d개: %s",
                len(tools), ", ".join(t.name for t in tools))


def _build_context(evidences: list[dict], limit: int = 12000) -> str:
    """retrieved_context 는 string 이어야 한다. 중요한 근거를 앞에 둔다 —
    길이 초과분은 채점에 반영되지 않으므로 뒤로 밀리면 없는 것과 같다."""
    SEP = "\n\n---\n\n"
    parts, total = [], 0
    for i, e in enumerate(evidences, 1):
        src = f"[{i}] {e.get('company') or ''} {e.get('report_name') or ''}"
        sec = " > ".join(e.get("section_path") or [])
        head = f"{src} ({e.get('report_id')}){' | ' + sec if sec else ''}"
        body = e.get("text") or ""
        block = f"{head}\n{body}"
        cost = len(block) + (len(SEP) if parts else 0)   # 구분자 비용도 예산에 포함
        if total + cost > limit:
            room = max(0, limit - total - (len(SEP) if parts else 0))
            if room > 0:
                parts.append(block[:room])
            break
        parts.append(block)
        total += cost
    return SEP.join(parts)


def create_app():
    from fastapi import FastAPI, Query
    from fastapi.responses import JSONResponse

    app = FastAPI(title="공시 Agent 평가 API", docs_url=None, redoc_url=None)

    @app.on_event("startup")
    def _startup() -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        try:
            _warm()
        except Exception as e:  # noqa: BLE001
            # 기동 실패해도 프로세스는 살려 둔다 — /health 로 원인을 볼 수 있어야 한다.
            _state["error"] = f"{type(e).__name__}: {e}"
            logger.error("[SERVE] 인덱스 준비 실패: %s\n%s", e, traceback.format_exc())

    @app.get("/health")
    def health() -> dict:
        b = _state["bundle"]
        return {
            "ready": b is not None,
            "error": _state["error"],
            "modes": getattr(b, "modes", []),
            "n_chunks": len(getattr(b, "chunks", []) or []),
            "cached_answers": len(_cache),
            "no_llm": NO_LLM,
        }

    @app.get("/answer")
    def answer(question_id: str = Query(default=""), question: str = Query(default="")):
        t0 = time.time()

        # --- 멱등성: 같은 question_id 는 재계산하지 않는다 (최대 3회 수신 가능) ---
        with _lock:
            hit = _cache.get(question_id)
        if hit is not None and hit.get("question") == question:
            logger.info("[SERVE] qid=%s 캐시 응답", question_id)
            return JSONResponse(hit["payload"])

        if _state["bundle"] is None:
            # 아직 준비 안 됨 = 일시적 상태. 5xx 로 재시도를 유도한다.
            return JSONResponse(
                {"detail": f"index not ready: {_state['error']}"}, status_code=503)

        try:
            payload = _answer_once(question_id, question, deadline=t0 + SOFT_BUDGET_SEC)
        except TransientUpstreamError as e:
            logger.warning("[SERVE] qid=%s 일시적 실패 -> 503 (재시도 유도): %s", question_id, e)
            return JSONResponse({"detail": str(e)}, status_code=503)
        except Exception as e:  # noqa: BLE001
            if _looks_transient(e):
                logger.warning("[SERVE] qid=%s 일시적 추정 -> 503: %s", question_id, e)
                return JSONResponse({"detail": str(e)}, status_code=503)
            # 우리 버그는 재시도해도 같다. 빈 답 대신 '확인되지 않음' 을 200 으로 준다.
            logger.error("[SERVE] qid=%s 처리 실패: %s\n%s", question_id, e, traceback.format_exc())
            payload = {
                "question_id": str(question_id), "question": str(question),
                "retrieved_context": "", "think_trace": f"error: {type(e).__name__}",
                "answer": NOT_FOUND_ANSWER,
            }

        with _lock:
            _cache[question_id] = {"question": question, "payload": payload}
        logger.info("[SERVE] qid=%s %.1fs", question_id, time.time() - t0)
        return JSONResponse(payload)

    return app


def _answer_once(question_id: str, question: str, *, deadline: float) -> dict:
    bundle = _state["bundle"]
    trace: list[str] = []

    if not question.strip():
        return {"question_id": str(question_id), "question": str(question),
                "retrieved_context": "", "think_trace": "empty question",
                "answer": NOT_FOUND_ANSWER}

    # 0) v2 경로 — 규칙이 무엇을 몇 번 검색할지 정한다(기본값).
    #    **거부로 끝난 답변은 폴백하지 않는다.** 거부는 실패가 아니라 설계된
    #    동작이고, 여기서 아래 경로로 떨어뜨리면 그 경로가 없는 사실을 지어낸다.
    if _state.get("askv2"):
        try:
            return _answer_with_v2(question_id, question, deadline=deadline)
        except TransientUpstreamError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.error("[SERVE] v2 실패(%s) — 아래 경로로 재시도\n%s",
                         type(e).__name__, traceback.format_exc())
            trace.append(f"v2 failed: {type(e).__name__}")

    # 1) 구 에이전트 경로 — 도구를 써서 여러 번 조회/계산할 수 있다.
    #    USE_AGENT=1 일 때만. 실패하면 아래 단순 경로로 떨어진다.
    if _state.get("tools"):
        try:
            return _answer_with_agent(question_id, question, deadline=deadline)
        except TransientUpstreamError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.error("[SERVE] 에이전트 실패(%s) — 단순 경로로 재시도\n%s",
                         type(e).__name__, traceback.format_exc())
            trace.append(f"agent failed: {type(e).__name__}")

    # 1) 검색
    t = time.time()
    hits = bundle.retriever.search(question, k=12, candidate_k=50)
    trace.append(f"retrieve: {len(hits)}건 {time.time()-t:.2f}s (경로 {bundle.modes})")

    # 2) Parent Expansion — 작은 조각으로 찾고 섹션 전체로 넓힌다
    evidences = bundle.parent_expander.expand(hits, budget_chars=12000)
    trace.append(f"expand: {len(evidences)}개 근거")

    context = _build_context(evidences)
    if not evidences:
        return {"question_id": str(question_id), "question": str(question),
                "retrieved_context": "", "think_trace": " | ".join(trace),
                "answer": NOT_FOUND_ANSWER}

    if NO_LLM:
        # 검색 계층만 검증하는 모드 (크레딧 0). 답변 생성은 하지 않는다.
        return {"question_id": str(question_id), "question": str(question),
                "retrieved_context": context, "think_trace": " | ".join(trace) + " | NO_LLM",
                "answer": ""}

    if time.time() > deadline:
        trace.append("budget exceeded before LLM")
        return {"question_id": str(question_id), "question": str(question),
                "retrieved_context": context, "think_trace": " | ".join(trace),
                "answer": NOT_FOUND_ANSWER}

    # 3) HCX 답변 생성
    try:
        answer_text = _generate(question, evidences)
    except Exception as e:  # noqa: BLE001
        if _looks_transient(e):
            raise TransientUpstreamError(f"HCX 일시 실패: {type(e).__name__}: {e}") from e
        raise
    trace.append("answer generated")
    return {"question_id": str(question_id), "question": str(question),
            "retrieved_context": context, "think_trace": " | ".join(trace),
            "answer": answer_text or NOT_FOUND_ANSWER}


def _answer_with_v2(question_id: str, question: str, *, deadline: float) -> dict:
    """신 파이프라인으로 답한다. 5개 필드 규격은 다른 경로와 동일하게 맞춘다.

    `think_trace` 에 **어느 단계에서 끝났는지**를 남긴다. 거부한 답변이
    "못 찾아서"인지 "범위 밖이라서"인지 구분되어야 사후에 원인을 가릴 수 있다.
    """
    t = time.time()
    res = _state["askv2"].run(question)

    plan = res.plan
    trace = [f"stopped_at={res.stopped_at}", f"hcx={res.hcx_calls}회"]
    if plan is not None:
        trace.append(f"plan: {plan.answer_mode}/{plan.task}"
                     f" 회사={plan.companies} 공시={plan.report_kinds}"
                     f" 집계={plan.aggregation} 버전={plan.latest_policy}")
    if res.scope is not None:
        trace.append(f"scope={res.scope.scope}")
    if res.decomposed is not None:
        trace.append(f"하위질의 {len(res.decomposed.sub_queries)}개"
                     f" 근거 {len(res.decomposed.merged)}건")
        if res.decomposed.empty_labels:
            trace.append("근거 0건 대상: " + ", ".join(res.decomposed.empty_labels))
    if res.retries:
        trace.append(f"재검색 {res.retries}회")
    if res.sufficiency is not None and not res.sufficiency.ok:
        trace.append("충분성 미달: " + " / ".join(res.sufficiency.reasons))
    if res.abstention is not None:
        trace.append(f"게이트={res.abstention.action}({res.abstention.reason})")
    if res.validation_result is not None and res.validation_result.warnings:
        trace.append("검증경고: " + " / ".join(res.validation_result.warnings))
    trace.append(f"elapsed {time.time() - t:.1f}s")

    context = ((res.evidence_pack.prompt_text or "") if res.evidence_pack else "")[:12000]
    return {"question_id": str(question_id), "question": str(question),
            "retrieved_context": context, "think_trace": " | ".join(trace),
            "answer": res.answer or NOT_FOUND_ANSWER}


def _answer_with_agent(question_id: str, question: str, *, deadline: float) -> dict:
    """도구 호출 에이전트로 답한다. 5개 필드 규격은 단순 경로와 동일하게 맞춘다."""
    from disclosure_rag.agent.ask import ask
    from disclosure_rag.agent.hcx_client import HCXClient

    client = _state.get("hcx") or HCXClient()
    _state["hcx"] = client

    t = time.time()
    res = ask(client, _state["tools"], question,
              entity_extractor=_state["entity_extractor"], router=None)
    tr = res.trace

    trace = [
        f"agent: {tr.iterations}회 반복, 중단사유 {tr.stopped_reason}",
        f"tools: {', '.join(getattr(c, 'name', '?') for c in (tr.tool_calls or [])) or '없음'}",
        f"elapsed {time.time()-t:.1f}s",
    ]
    if getattr(tr, "nudges", None):
        trace.append(f"근거보강 {len(tr.nudges)}회: " + " / ".join(tr.nudges))
    if getattr(res, "remediation", None):
        trace.append("재생성: " + " / ".join(res.remediation))
    if res.validation and res.validation.warnings:
        trace.append("검증경고: " + " / ".join(res.validation.warnings))

    # retrieved_context 는 문자열이어야 한다 — Evidence Pack 원문을 그대로 쓴다.
    context = (res.evidence_pack.prompt_text or "")[:12000]
    return {"question_id": str(question_id), "question": str(question),
            "retrieved_context": context, "think_trace": " | ".join(trace),
            "answer": res.answer or NOT_FOUND_ANSWER}


def _generate(question: str, evidences: list[dict]) -> str:
    """HCX 로 답변을 만든다.

    지연 임포트 — 크레딧/키가 없는 환경에서도 서버는 뜨고 /health 가 응답해야 한다.
    EvidencePack 을 직접 조립하는 이유: 여기서는 Agent 루프를 돌지 않고 검색 결과를
    바로 답변 생성에 넘긴다(단순 경로). Agent 루프 경로는 ask.py 가 따로 쓴다.
    """
    from disclosure_rag.agent.answer_generator import generate_answer
    from disclosure_rag.agent.evidence import Citation, EvidencePack
    from disclosure_rag.agent.hcx_client import HCXClient

    client = _state.get("hcx") or HCXClient()
    _state["hcx"] = client

    lines = [f"[USER QUESTION]\n{question}\n"]
    citations: list[Citation] = []
    for i, e in enumerate(evidences, 1):
        status = "정정본" if e.get("is_correction") else "원본"
        if e.get("is_latest"):
            status += " (최신)"
        lines.append(
            f"[EVIDENCE {i}]\n"
            f"회사: {e.get('company')}\n"
            f"공시명: {e.get('report_name')}\n"
            f"공시일: {e.get('filing_date')}\n"
            f"기간: {e.get('period')}\n"
            f"Section: {' > '.join(e.get('section_path') or [])}\n"
            f"정정 상태: {status}\n"
            f"내용:\n{e.get('text')}\n"
            f"report_id: {e.get('report_id')}\n"
            f"chunk_id: {e.get('chunk_id')}\n"
        )
        citations.append(Citation(
            chunk_id=e.get("chunk_id") or "", report_id=e.get("report_id") or "",
            company=e.get("company"), report_name=e.get("report_name"),
            filing_date=e.get("filing_date"), section_path=e.get("section_path") or [],
            is_correction=bool(e.get("is_correction")), is_latest=e.get("is_latest"),
        ))

    pack = EvidencePack(question=question, prompt_text="\n".join(lines),
                        citations=citations, tool_results_summary=[])
    return generate_answer(client, pack)


app = create_app()
