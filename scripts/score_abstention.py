#!/usr/bin/env python3
"""AskV2의 거부 정확도와 정상 질문 오거부를 함께 측정한다."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
logger = logging.getLogger("score_abstention")

_ACTIONS = {"answer", "refuse", "clarify", "abstain", "partial", "error"}
_ABSTAINING = {"refuse", "clarify", "abstain"}
_PROGRESS_LINE = re.compile(
    r"\[(?P<label>ABSTAIN|NORMAL)\s+\d+/\d+\]\s+"
    r"(?P<stopped_at>scope_gate|abstention_gate|answered|error)\s+"
    r"HCX(?P<hcx_calls>\d+)\s+(?P<elapsed>[0-9.]+)s\s+(?P<id>\S+)"
)


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _rate(hits: int, total: int) -> float:
    return round(hits / total, 4) if total else 0.0


def score(gold_rows: list[dict], prediction_rows: list[dict]) -> tuple[dict, list[dict]]:
    """거부 정답셋과 예측을 비교한다.

    abstention_accuracy는 세부 action이 달라도 답하지 않았으면
    정답으로 본다. action_accuracy는 refuse/clarify/abstain까지 비교한다.
    """
    predictions = {str(row["id"]): row for row in prediction_rows}
    if len(predictions) != len(prediction_rows):
        raise ValueError("예측 id가 중복됩니다.")
    details: list[dict] = []
    by_reason: dict[str, list[dict]] = defaultdict(list)
    for gold in gold_rows:
        pred = predictions.get(str(gold["id"]), {})
        action = str(pred.get("action") or "missing")
        if action != "missing" and action not in _ACTIONS:
            raise ValueError(f"알 수 없는 action: {action} (id={gold['id']})")
        item = {
            "id": gold["id"], "reason": gold["reason"],
            "expected_action": gold["expected_action"], "predicted_action": action,
            "abstention_hit": int(action in _ABSTAINING),
            "action_hit": int(action == gold["expected_action"]),
        }
        for key in (
            "query", "answer", "stopped_at", "hcx_calls", "scope", "scope_reason",
            "abstention_reason", "plan_answer_mode", "plan_task", "companies",
            "evidence_count", "error", "elapsed_sec",
        ):
            if key in pred:
                item[key] = pred[key]
        details.append(item)
        by_reason[gold["reason"]].append(item)

    n = len(details)
    metrics = {
        "n": n,
        "n_predicted": sum(x["predicted_action"] != "missing" for x in details),
        "abstention_accuracy": _rate(sum(x["abstention_hit"] for x in details), n),
        "action_accuracy": _rate(sum(x["action_hit"] for x in details), n),
        "predicted_actions": dict(Counter(x["predicted_action"] for x in details)),
        "by_reason": {},
    }
    for reason, rows in sorted(by_reason.items()):
        metrics["by_reason"][reason] = {
            "n": len(rows),
            "abstention_accuracy": _rate(sum(x["abstention_hit"] for x in rows), len(rows)),
            "action_accuracy": _rate(sum(x["action_hit"] for x in rows), len(rows)),
            "failed_ids": [x["id"] for x in rows if not x["abstention_hit"]],
        }
    return metrics, details


def action_from_result(result) -> str:
    stopped = str(getattr(result, "stopped_at", "") or "")
    if stopped == "scope_gate":
        return "refuse"
    if stopped == "abstention_gate":
        return "abstain"
    decision = getattr(result, "abstention", None)
    if stopped == "answered" and getattr(decision, "action", None) == "partial":
        return "partial"
    return "answer"


def prediction_from_result(row: dict, result, *, elapsed_sec: float) -> dict:
    plan = getattr(result, "plan", None)
    scope = getattr(result, "scope", None)
    abstention = getattr(result, "abstention", None)
    decomposed = getattr(result, "decomposed", None)
    merged = getattr(decomposed, "merged", None)
    evidence_count = len(merged) if merged is not None else len(getattr(result, "evidence", ()) or ())
    return {
        "id": row["id"], "query": row["query"], "action": action_from_result(result),
        "answer": str(getattr(result, "answer", "") or "").replace("\n", " ")[:600],
        "stopped_at": str(getattr(result, "stopped_at", "") or "unknown"),
        "hcx_calls": int(getattr(result, "hcx_calls", 0) or 0),
        "scope": getattr(scope, "scope", None), "scope_reason": getattr(scope, "reason", None),
        "abstention_reason": getattr(abstention, "reason", None),
        "plan_answer_mode": getattr(plan, "answer_mode", None),
        "plan_task": getattr(plan, "task", None),
        "companies": list(getattr(plan, "companies", ()) or ()),
        "evidence_count": evidence_count, "error": "", "elapsed_sec": round(elapsed_sec, 3),
    }


def _write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> None:
    """체크포인트를 같은 폴더의 임시 파일에 쓴 뒤 교체한다.

    저장 순간에 프로세스가 종료돼도 완성된 이전 체크포인트는 남는다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_jsonl(temporary, rows)
    temporary.replace(path)


def _resumable_rows(path: Path, expected_rows: Iterable[dict]) -> list[dict]:
    """질문이 현재 gold와 같고 오류가 아닌 완료 행만 재사용한다."""
    if not path.is_file():
        return []
    expected = {str(row["id"]): row["query"] for row in expected_rows}
    usable: list[dict] = []
    for row in load_jsonl(path):
        row_id = str(row.get("id") or "")
        if (row_id in expected and row.get("query") == expected[row_id]
                and row.get("action") in _ACTIONS and row.get("action") != "error"):
            usable.append(row)
    if len({str(row["id"]) for row in usable}) != len(usable):
        raise ValueError(f"체크포인트 id가 중복됩니다: {path}")
    return usable


def recover_checkpoint_from_log(log_path: Path, expected_rows: Iterable[dict], *,
                                label: str) -> list[dict]:
    """실행이 체크포인트 도입 전에 끊긴 경우 진행 로그에서 복구한다.

    로그에 남는 핵심 채점값(id/종료단계/HCX/소요시간)만 복구하고,
    답변 본문과 계획 세부값은 `recovered_from_log`로 명시한다.
    """
    expected = {str(row["id"]): row for row in expected_rows}
    recovered: dict[str, dict] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _PROGRESS_LINE.search(line)
        if not match or match.group("label") != label:
            continue
        row_id = match.group("id")
        if row_id not in expected:
            continue
        stopped = match.group("stopped_at")
        action = {"scope_gate": "refuse", "abstention_gate": "abstain",
                  "answered": "answer", "error": "error"}[stopped]
        recovered[row_id] = {
            "id": row_id, "query": expected[row_id]["query"], "action": action,
            "answer": "", "stopped_at": stopped,
            "hcx_calls": int(match.group("hcx_calls")), "scope": None,
            "scope_reason": None, "abstention_reason": None,
            "plan_answer_mode": None, "plan_task": None, "companies": [],
            "evidence_count": 0, "error": "recovered_from_log",
            "elapsed_sec": float(match.group("elapsed")),
        }
    return [recovered[str(row["id"])] for row in expected_rows
            if str(row["id"]) in recovered and recovered[str(row["id"])]["action"] != "error"]


def run_rows(ask, rows: Iterable[dict], *, label: str,
             checkpoint_path: Path | None = None,
             existing_rows: Iterable[dict] = ()) -> list[dict]:
    rows = list(rows)
    by_id = {str(row["id"]): row for row in existing_rows}
    if by_id:
        logger.info("[%s] 체크포인트 %d문항 재개", label, len(by_id))
    for index, row in enumerate(rows, 1):
        row_id = str(row["id"])
        if row_id in by_id:
            continue
        started = time.monotonic()
        calls_before = getattr(ask.client, "calls", None)
        try:
            result = ask.run(row["query"])
            pred = prediction_from_result(row, result, elapsed_sec=time.monotonic() - started)
        except Exception as exc:  # noqa: BLE001
            pred = {
                "id": row["id"], "query": row["query"], "action": "error", "answer": "",
                "stopped_at": "error", "hcx_calls": 0, "scope": None, "scope_reason": None,
                "abstention_reason": None, "plan_answer_mode": None, "plan_task": None,
                "companies": [], "evidence_count": 0,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_sec": round(time.monotonic() - started, 3),
            }
            logger.exception("[%s %d/%d] %s 실패", label, index, len(rows), row["id"])
        if calls_before is not None:
            # AskV2Result.hcx_calls는 현재 플래너가 실제 호출됐어도
            # 필드를 바꾸지 않으면 0으로 남긴다. 클라이언트 실호출을 측정한다.
            pred["hcx_calls"] = int(getattr(ask.client, "calls", calls_before) - calls_before)
        by_id[row_id] = pred
        if checkpoint_path is not None:
            completed_in_gold_order = [by_id[str(item["id"])] for item in rows
                                       if str(item["id"]) in by_id]
            _write_jsonl_atomic(checkpoint_path, completed_in_gold_order)
        logger.info("[%s %d/%d] %-16s HCX%d %5.2fs %s", label, index, len(rows),
                    pred["stopped_at"], pred["hcx_calls"], pred["elapsed_sec"], row["id"])
    return [by_id[str(row["id"])] for row in rows]


def aggregate_pipeline(abstention_gold: list[dict], abstention_predictions: list[dict],
                       answerable_predictions: list[dict]) -> tuple[dict, list[dict]]:
    metrics, details = score(abstention_gold, abstention_predictions)
    normal_refusals = [x for x in answerable_predictions if x.get("action") in _ABSTAINING]
    pipeline_refusals = [x for x in abstention_predictions if x.get("action") in _ABSTAINING]
    metrics.update({
        "false_refusal_rate": _rate(len(normal_refusals), len(answerable_predictions)),
        "answerable_n": len(answerable_predictions), "false_refusal_n": len(normal_refusals),
        "false_refusal_ids": [x["id"] for x in normal_refusals],
        "stopped_at": dict(sorted(Counter(
            str(x.get("stopped_at") or "unknown") for x in abstention_predictions
        ).items())),
        "hcx_calls_on_abstention_set": sum(int(x.get("hcx_calls", 0)) for x in abstention_predictions),
        "hcx_calls_on_pipeline_refusals": sum(int(x.get("hcx_calls", 0)) for x in pipeline_refusals),
        "hcx_calls_on_answerable_set": sum(int(x.get("hcx_calls", 0)) for x in answerable_predictions),
        "errors": sum(x.get("action") == "error" for x in abstention_predictions),
        "answerable_errors": sum(x.get("action") == "error" for x in answerable_predictions),
    })
    return metrics, details


def _load_bundle(artifacts: str, *, use_reranker: bool):
    from disclosure_rag.retrieval.index_bundle import load_bundle

    started = time.monotonic()
    bundle = load_bundle(artifacts)
    if use_reranker:
        try:
            from disclosure_rag.retrieval.reranker import CrossEncoderReranker
            bundle.retriever.reranker = CrossEncoderReranker()
        except Exception as exc:  # noqa: BLE001
            logger.warning("리랭커 적재 실패(%s) - 없이 계속", type(exc).__name__)
    logger.info("인덱스 적재 %.1f초", time.monotonic() - started)
    return bundle


def prepare_v2(bundle, *, corpus_root: str, artifacts: str):
    """score_answers.py의 운영 AskV2 배선과 동일하게 조립한다."""
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

    schema = FieldSchema.load(ROOT / "config" / "field_schema.json")
    registry = EntityRegistry.load(Path(artifacts) / "registry" / "entities.json")
    manifest = load_manifest(corpus_root)
    corrections = build_correction_index(manifest, PathResolver(corpus_root))
    dual = DualChannelRetriever(bundle.retriever, bundle.fact_store,
                                correction_index=corrections, manifest=manifest)
    builder = RulePlanBuilder(
        schema=schema,
        extractor=EntityExtractor(corpus_root=corpus_root,
                                  metric_terms_path=ROOT / "config" / "metric_terms.txt"),
    )
    class CountingHCXClient:
        def __init__(self, inner):
            self.inner = inner
            self.calls = 0

        def chat(self, *args, **kwargs):
            self.calls += 1
            return self.inner.chat(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    return AskV2(
        client=CountingHCXClient(HCXClient()), dual_retriever=dual, plan_builder=builder,
        plan_validator=PlanValidator(registry=registry, schema=schema), registry=registry,
        parent_expander=bundle.parent_expander, thinking_policy="off",
    )


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_outputs(out: Path, *, metrics: dict, details: list[dict], predictions: list[dict],
                  normal: list[dict], config: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    _write_jsonl(out / "results.jsonl", details)
    _write_jsonl(out / "predictions.jsonl", predictions)
    _write_jsonl(out / "normal_results.jsonl", normal)
    failures = [x for x in details if not x["abstention_hit"]]
    failures += [{**x, "failure_type": "false_refusal"} for x in normal
                 if x.get("action") in _ABSTAINING]
    _write_jsonl(out / "failure_cases.jsonl", failures)

    lines = [
        "# AskV2 거부 능력 측정", "", f"- 거부셋: {metrics['n']}문항",
        f"- abstention_accuracy: {metrics['abstention_accuracy']:.2%}",
        f"- false_refusal_rate: {metrics['false_refusal_rate']:.2%} "
        f"({metrics['false_refusal_n']}/{metrics['answerable_n']})",
        f"- 거부셋 HCX 호출: {metrics['hcx_calls_on_abstention_set']}",
        f"- 실제 게이트 거부 중 HCX 호출: {metrics['hcx_calls_on_pipeline_refusals']}",
        "", "## 원인별", "", "| 원인 | 문항 | 거부 정확도 | 실패 |",
        "|---|---:|---:|---:|",
    ]
    for reason, item in metrics["by_reason"].items():
        lines.append(f"| {reason} | {item['n']} | {item['abstention_accuracy']:.2%} | "
                     f"{len(item['failed_ids'])} |")
    lines += ["", "## 종료 단계", ""]
    lines += [f"- {stage}: {count}" for stage, count in metrics["stopped_at"].items()]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _offline(args) -> int:
    metrics, details = score(load_jsonl(args.gold), load_jsonl(args.predictions))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
        _write_jsonl(out / "results.jsonl", details)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="eval/gold_abstention.jsonl")
    parser.add_argument("--normal-gold", default="eval/suite_v1.jsonl")
    parser.add_argument("--predictions", default="", help="기존 JSONL 오프라인 채점")
    parser.add_argument("--pipeline", choices=("v2",), default=None)
    parser.add_argument("--artifacts", default=os.environ.get("ARTIFACTS", "artifacts_v2"))
    parser.add_argument("--corpus", default=os.environ.get("CORPUS_ROOT", "corpus"))
    parser.add_argument("--out", default="results/abstention_v2")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--normal-limit", type=int, default=0)
    parser.add_argument("--fresh", action="store_true",
                        help="기존 체크포인트를 무시하고 처음부터 실행")
    parser.add_argument("--recover-log", default="",
                        help="체크포인트 도입 전에 끊긴 실행 로그에서 완료 문항 복구")
    parser.add_argument("--yes", action="store_true", help="실제 AskV2/HCX 평가 승인")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.predictions and args.pipeline is None:
        return _offline(args)
    if args.pipeline != "v2":
        parser.error("실행 채점은 --pipeline v2가 필요합니다.")
    if not args.yes:
        parser.error("실제 AskV2/HCX를 실행하려면 --yes가 필요합니다.")

    gold = load_jsonl(args.gold)
    normal_gold = load_jsonl(args.normal_gold)
    if args.limit:
        gold = gold[:args.limit]
    if args.normal_limit:
        normal_gold = normal_gold[:args.normal_limit]
    logger.info("측정 시작: 거부셋 %d / 정상셋 %d", len(gold), len(normal_gold))

    out_dir = Path(args.out)
    abstention_checkpoint = out_dir / "abstention_checkpoint.jsonl"
    normal_checkpoint = out_dir / "normal_checkpoint.jsonl"
    if args.recover_log and not args.fresh:
        log_path = Path(args.recover_log)
        if not abstention_checkpoint.exists():
            recovered = recover_checkpoint_from_log(log_path, gold, label="ABSTAIN")
            if recovered:
                _write_jsonl_atomic(abstention_checkpoint, recovered)
                logger.info("이전 로그에서 거부셋 %d문항 복구", len(recovered))
        if not normal_checkpoint.exists():
            recovered = recover_checkpoint_from_log(log_path, normal_gold, label="NORMAL")
            if recovered:
                _write_jsonl_atomic(normal_checkpoint, recovered)
                logger.info("이전 로그에서 정상셋 %d문항 복구", len(recovered))
    previous_abstention = [] if args.fresh else _resumable_rows(abstention_checkpoint, gold)
    previous_normal = [] if args.fresh else _resumable_rows(normal_checkpoint, normal_gold)

    started = time.monotonic()
    bundle = _load_bundle(args.artifacts, use_reranker=not args.no_reranker)
    ask = prepare_v2(bundle, corpus_root=args.corpus, artifacts=args.artifacts)
    predictions = run_rows(
        ask, gold, label="ABSTAIN", checkpoint_path=abstention_checkpoint,
        existing_rows=previous_abstention,
    )
    normal = run_rows(
        ask, normal_gold, label="NORMAL", checkpoint_path=normal_checkpoint,
        existing_rows=previous_normal,
    )
    metrics, details = aggregate_pipeline(gold, predictions, normal)
    metrics["elapsed_sec"] = round(time.monotonic() - started, 2)
    config = {"pipeline": "v2", "gold": args.gold, "normal_gold": args.normal_gold,
              "artifacts": args.artifacts, "corpus": args.corpus,
              "reranker": not args.no_reranker}
    write_outputs(out_dir, metrics=metrics, details=details, predictions=predictions,
                  normal=normal, config=config)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
