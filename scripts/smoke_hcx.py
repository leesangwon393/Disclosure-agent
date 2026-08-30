#!/usr/bin/env python3
"""HCX 연결부터 답변 생성까지 단계별로 끊어서 확인한다.

    python3 scripts/smoke_hcx.py                    # 1~3단계 (인덱스 안 씀, 빠름)
    python3 scripts/smoke_hcx.py --full             # 4단계까지 (인덱스 적재 ~5분)
    python3 scripts/smoke_hcx.py --full --question "삼성전자 순자산액은?"

왜 단계별인가: agent/ 아래 1,259줄이 한 번도 실행된 적이 없다. 통째로 돌리면
어디서 깨졌는지 알기 어려우므로 연결 → 대화 → tool calling → 전체 순으로 끊는다.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

OK, NG = "✅", "❌"



class Heartbeat:
    """응답을 기다리는 동안 1초마다 경과시간을 찍는다.

    requests.post 는 블로킹이라 화면이 완전히 멈춘 것처럼 보인다. 그래서
    사용자가 Ctrl+C 를 누르게 된다(실제로 두 번 그랬다). 별도 스레드로
    "살아있다"는 신호를 준다.
    """

    def __init__(self, label: str):
        self.label = label
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        n = 0
        while not self._stop.wait(1.0):
            n += 1
            print(f"\r    {self.label} … {n}초 경과 (Ctrl+C 누르지 마세요)",
                  end="", flush=True)

    def __enter__(self):
        self._t.start(); return self

    def __exit__(self, *a):
        self._stop.set(); self._t.join(timeout=0.2)
        print("\r" + " " * 70 + "\r", end="", flush=True)


def check_venv() -> None:
    """엉뚱한 venv 로 돌리면 fastapi 등이 없어 엉뚱한 에러가 난다."""
    here = Path(__file__).resolve().parent.parent
    exe = Path(sys.executable).resolve()
    if here.name not in str(exe):
        print(f"⚠️  실행 중인 파이썬이 이 프로젝트 venv 가 아닙니다.")
        print(f"    지금: {exe}")
        print(f"    필요: {here}/.venv/bin/python3")
        print(f"    -> cd {here} && source .venv/bin/activate\n")



# ── SIGINT 진단 ──────────────────────────────────────────────────────────
# 2026-08-29: HCX 호출 중 KeyboardInterrupt 로 두 번 죽었다. 사용자는 키를 누른
# 적이 없다고 한다. 재시도로 덮으면 원인이 사라지는 게 아니라 안 보이게 될 뿐이고,
# 평가 서버에서 같은 일이 나면 점수를 잃고도 이유를 모른다. 그래서 '왜 왔는지'를
# 남기게 만든다.
#
# 핵심 분기:
#   (A) 우리 핸들러가 찍힌다  -> 진짜 SIGINT 가 외부에서 왔다 (터미널/상위 프로세스)
#   (B) 안 찍히는데 예외만 난다 -> 신호가 아니라 라이브러리가 던진 것 (전혀 다른 문제)
_SIGINT_LOG: list[dict] = []


def _install_sigint_probe() -> None:
    import signal

    def handler(signum, frame):
        import traceback as tb
        rec = {
            "at": time.strftime("%H:%M:%S"),
            "where": "".join(tb.format_stack(frame)[-2:]).strip()[:300],
        }
        _SIGINT_LOG.append(rec)
        print(f"\n  🔔 SIGINT 수신 ({rec['at']}) — 외부에서 중단 신호가 왔습니다.",
              flush=True)
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, handler)
    except Exception as e:  # noqa: BLE001
        print(f"  (SIGINT 프로브 설치 실패: {type(e).__name__})")


def _report_interrupt(label: str, elapsed: float) -> None:
    import os as _os
    print(f"\n{'='*60}\n중단 진단\n{'='*60}")
    print(f"  구간          : {label}")
    print(f"  경과          : {elapsed:.1f}초")
    print(f"  PID / PPID    : {_os.getpid()} / {_os.getppid()}")
    try:
        ppid = _os.getppid()
        import subprocess
        pname = subprocess.run(["ps", "-o", "comm=", "-p", str(ppid)],
                               capture_output=True, text=True).stdout.strip()
        print(f"  부모 프로세스 : {pname}")
    except Exception:  # noqa: BLE001
        pass
    if _SIGINT_LOG:
        print(f"  {NG} 진짜 SIGINT 가 왔습니다 ({len(_SIGINT_LOG)}회).")
        print("     -> 터미널(VS Code 등)이나 상위 프로세스가 신호를 보낸 것입니다.")
        print("     -> 다른 터미널 앱(기본 Terminal.app)에서 그대로 돌려보세요.")
        for r in _SIGINT_LOG:
            print(f"       {r['at']}  {r['where'][:160]}")
    else:
        print(f"  {NG} SIGINT 는 오지 않았는데 KeyboardInterrupt 가 났습니다.")
        print("     -> 신호 문제가 아니라 라이브러리/스레드 쪽 문제입니다.")
        print("     -> requests/urllib3 버전, 프록시 설정(HTTPS_PROXY)을 봐야 합니다.")
    print(f"\n  참고: 같은 요청을 파이썬 없이 curl 로 재현해보세요 —")
    print(f"        bash scripts/curl_hcx.sh")


def guarded(label: str, fn, *, retry: bool = False):
    """중단되면 **덮지 않고 진단을 찍는다.** --retry-on-interrupt 를 준 경우만 재시도."""
    t0 = time.time()
    try:
        with Heartbeat(label):
            return fn()
    except KeyboardInterrupt:
        _report_interrupt(label, time.time() - t0)
        if not retry:
            raise
        print("\n  (--retry-on-interrupt 지정됨 — 1회 재시도)")
        with Heartbeat(label + " 재시도"):
            return fn()


def step(n: int, title: str):
    print(f"\n{'='*60}\n[{n}] {title}\n{'='*60}", flush=True)


def main() -> int:
    # index_bundle / api 는 logger.info 로 진행상황을 알린다. basicConfig 가 없으면
    # 화면에 안 나와서 "멈춘 건지 도는 건지" 알 수 없다.
    import logging
    logging.basicConfig(level=logging.INFO, format="  [%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")
    check_venv()
    _install_sigint_probe()

    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="인덱스 적재 + 전체 파이프라인")
    ap.add_argument("--retry-on-interrupt", action="store_true",
                    help="중단돼도 1회 재시도(원인 진단은 그대로 출력)")
    ap.add_argument("--artifacts", default="artifacts_v2")
    ap.add_argument("--question", default="삼성전자의 주요사항보고서(자기주식취득결정)에 기재된 순자산액은 얼마인가?")
    args = ap.parse_args()

    # ── 1. 키 로드 ────────────────────────────────────────────────────────
    step(1, "HCX 키 로드")
    from disclosure_rag.agent.hcx_client import HCXClient, HCXError
    try:
        c = HCXClient(timeout=30.0)
        print(f"{OK} model={c.model}  key={c.api_key[:5]}…{c.api_key[-4:]}")
    except Exception as e:
        print(f"{NG} {type(e).__name__}: {e}")
        print("   -> pipeline-kim/.env 에 HCX_API_KEY, HCX_MODEL 이 있는지 확인")
        return 1

    # ── 2. 단순 대화 ──────────────────────────────────────────────────────
    # HCX-007 은 reasoning 모델이라 thinking 이 기본 ON 이다. 간단한 질문에도
    # 속으로 한참 추론해서 수십 초가 걸릴 수 있다 — 멈춘 게 아니다.
    # 300초 예산 설계에 직결되므로 thinking ON/OFF 를 나란히 잰다.
    step(2, "HCX 단순 호출 — thinking OFF / ON 시간 비교")
    Q = "한 문장으로만 답해라. 대한민국의 수도는?"

    print("  [2-a] thinking OFF (effort=none) … 응답 대기 중, 최대 90초", flush=True)
    try:
        t = time.time()
        m = guarded("HCX 응답 대기", retry=args.retry_on_interrupt, fn=lambda: c.chat(
            [{"role": "user", "content": Q}],
            max_tokens=50, thinking={"effort": "none"}))
        t_off = time.time() - t
        print(f"  {OK} {t_off:6.1f}초 | {str(m.get('content'))[:120]!r}")
    except Exception as e:
        t_off = None
        print(f"  {NG} {type(e).__name__}: {str(e)[:300]}")
        print("     -> 401/403=키·권한, 404=모델명(.env 의 HCX_MODEL), 그 외=네트워크")
        return 1

    print("  [2-b] thinking 기본값(ON) … 더 오래 걸릴 수 있음", flush=True)
    try:
        t = time.time()
        out = guarded("HCX 응답 대기", lambda: c.chat_simple(Q, max_tokens=50))
        t_on = time.time() - t
        print(f"  {OK} {t_on:6.1f}초 | {out[:120]!r}")
        if t_off and t_on > t_off * 2:
            print(f"  ⚠️  thinking 이 {t_on - t_off:.0f}초를 더 먹는다. "
                  "단순 조회 질문에는 effort=none 을 쓰는 게 맞다.")
    except Exception as e:
        print(f"  ⚠️  {type(e).__name__}: {str(e)[:200]} (2-a 가 됐으면 진행 가능)")

    # ── 3. tool calling ──────────────────────────────────────────────────
    step(3, "HCX tool calling (에이전트가 도구를 부를 수 있는지)")
    print("  … 응답 대기 중", flush=True)
    tools = [{
        "type": "function",
        "function": {
            "name": "lookup_fact",
            "description": "공시에서 특정 기업의 특정 항목 값을 조회한다",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "description": "기업명"},
                    "key": {"type": "string", "description": "항목명"},
                },
                "required": ["company", "key"],
            },
        },
    }]
    try:
        t = time.time()
        msg = guarded("HCX tool calling 대기", lambda: c.chat(
            [{"role": "user", "content": "삼성전자의 순자산액을 조회해줘."}],
            tools=tools, tool_choice="auto",
            thinking={"effort": "none"},   # HCX-007: tools 와 thinking 동시 사용 시 400
        ))
        calls = msg.get("toolCalls") or msg.get("tool_calls") or []
        if calls:
            f = calls[0].get("function", {})
            print(f"{OK} {time.time()-t:.1f}초 | 도구 호출: {f.get('name')} args={f.get('arguments')}")
        else:
            print(f"⚠️  {time.time()-t:.1f}초 | 도구를 안 불렀다. content={str(msg.get('content'))[:200]}")
            print("   -> 에이전트 라우팅이 tool calling 에 의존한다면 여기서 조정 필요")
    except Exception as e:
        print(f"{NG} {type(e).__name__}: {str(e)[:400]}")
        print("   -> tools 스키마 형식 문제일 수 있다. 2단계가 됐다면 키는 정상")

    if not args.full:
        print("\n여기까지 통과하면 HCX 연결은 정상. 전체 파이프라인은 --full 로.")
        return 0

    # ── 4. 검색 → 근거 → 답변 ────────────────────────────────────────────
    step(4, f"전체 파이프라인 (artifacts={args.artifacts})")

    # --- HCX 호출 계측 -------------------------------------------------
    # 300초 예산을 실제로 먹는 건 검색이 아니라 HCX 왕복 횟수다.
    # chat() 을 감싸서 호출마다 소요시간·용도(tools 유무)를 기록한다.
    from disclosure_rag.agent.hcx_client import HCXClient as _HC
    _calls: list[dict] = []
    _orig_chat = _HC.chat

    def _timed_chat(self, messages, **kw):
        t = time.time()
        try:
            return _orig_chat(self, messages, **kw)
        finally:
            _calls.append({
                "sec": time.time() - t,
                "kind": "tool판단" if kw.get("tools") else "문장생성",
                "in_chars": sum(len(str(m.get("content", ""))) for m in messages),
            })
    _HC.chat = _timed_chat
    # api._answer_once 는 모듈 전역 _state["bundle"] 을 읽는다. FastAPI startup 이
    # 안 도는 스크립트 실행에서는 _warm() 을 직접 불러 채워야 한다. 따로
    # load_bundle 을 또 부르면 인덱스가 두 벌 올라가 메모리가 두 배가 된다.
    import os
    os.environ["ARTIFACTS"] = args.artifacts
    try:
        t = time.time()
        print("  55만+ 조각 로드 → BM25 인덱스 → Qdrant 확인 → sparse 역색인 순으로 진행합니다.",
              flush=True)
        print("  'dense ... 이미 적재됨 — 재적재 건너뜀' 이 뜨면 1~2분, "
              "안 뜨면 15~20분 걸립니다.", flush=True)
        from disclosure_rag.serving import api as serving_api
        serving_api.ARTIFACTS = args.artifacts
        serving_api._warm()
        b = serving_api._state["bundle"]
        t_load = time.time() - t
        print(f"{OK} 인덱스 적재 {t_load:.0f}초 | 경로 {b.modes} | leaf {len(b.chunks):,}")
    except Exception as e:
        print(f"{NG} 인덱스 적재 실패: {type(e).__name__}: {str(e)[:300]}")
        traceback.print_exc()
        return 1

    try:
        t = time.time()
        hits = b.retriever.search(args.question, k=12)
        t_search = time.time() - t
        print(f"{OK} 검색 {t_search:.2f}초 | {len(hits)}건")
        for ch, sc in hits[:3]:
            print(f"    {sc:.3f}  {ch.chunk_id}  {ch.raw_text[:60]!r}")
    except Exception as e:
        print(f"{NG} 검색 실패: {type(e).__name__}: {str(e)[:300]}")
        traceback.print_exc(); return 1

    try:
        t = time.time()
        ev = b.parent_expander.expand(hits, budget_chars=12000)
        total = sum(len(e["text"]) for e in ev)
        t_expand = time.time() - t
        print(f"{OK} 근거 확장 {t_expand:.2f}초 | {len(ev)}건 / {total:,}자")
    except Exception as e:
        print(f"{NG} 근거 확장 실패: {type(e).__name__}: {str(e)[:300]}")
        traceback.print_exc(); return 1

    try:
        t = time.time()
        print("   (serving._answer_once 호출 — 실제 API 와 동일 경로)")
        payload = serving_api._answer_once("SMOKE-001", args.question,
                                           deadline=time.time() + 240)
        print(f"{OK} 답변 생성 {time.time()-t:.1f}초")
        for k in ("question_id", "question", "answer", "think_trace", "retrieved_context"):
            v = str(payload.get(k, ""))
            print(f"\n  --- {k} ({len(v)}자) ---\n  {v[:600]}")

        # --- 시간 분해 ---------------------------------------------------
        print(f"\n{'='*60}\n[시간 분해] 주최측 타임아웃 300초 기준\n{'='*60}")
        hcx_total = sum(c["sec"] for c in _calls)
        print(f"  검색            {t_search:7.2f}초")
        print(f"  근거 확장       {t_expand:7.2f}초")
        print(f"  HCX 합계        {hcx_total:7.2f}초   ({len(_calls)}회 호출)")
        for i, c in enumerate(_calls, 1):
            print(f"     {i}. {c['kind']:8s} {c['sec']:6.2f}초  입력 {c['in_chars']:,}자")
        e2e = time.time() - t
        print(f"  ─────────────────────────")
        print(f"  답변 1건 총     {e2e:7.2f}초   → 300초 예산의 {e2e/300*100:.1f}%")
        if e2e > 60:
            print("  ⚠️  60초 초과. 평가는 순차 호출이라 문항이 많으면 부담이 된다.")
        print(f"\n  (인덱스 적재 {t_load:.0f}초는 서버 기동 시 1회뿐이라 응답시간에 안 들어간다)")
    except Exception as e:
        print(f"{NG} 답변 생성 실패: {type(e).__name__}: {str(e)[:400]}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
