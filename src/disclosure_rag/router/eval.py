"""§48, §76: Router 평가 — Accuracy / Macro F1 / Confusion Matrix / Fallback Rate,
threshold sweep 지원."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import confusion_matrix, f1_score

from disclosure_rag.router.eval_dataset import EvalExample
from disclosure_rag.router.semantic_router_wrapper import Router

FALLBACK_LABEL = "__FALLBACK__"


@dataclass
class RouterEvalReport:
    accuracy: float
    macro_f1: float
    fallback_rate: float
    labels: list[str]
    confusion_matrix: list[list[int]]
    n: int

    def render(self) -> str:
        lines = [
            f"n={self.n}  accuracy={self.accuracy:.3f}  macro_f1={self.macro_f1:.3f}  fallback_rate={self.fallback_rate:.3f}",
            "",
            "Confusion Matrix (rows=true, cols=pred):",
            "labels: " + ", ".join(self.labels),
        ]
        for label, row in zip(self.labels, self.confusion_matrix):
            lines.append(f"  {label:20s} {row}")
        return "\n".join(lines)


def evaluate_router(router: Router, examples: list[EvalExample]) -> RouterEvalReport:
    y_true, y_pred = [], []
    fallback_count = 0
    for ex in examples:
        result = router.route(ex.query)
        pred = result.route or FALLBACK_LABEL
        y_pred.append(pred)
        y_true.append(ex.expected_route or FALLBACK_LABEL)
        if result.route is None:
            fallback_count += 1

    labels = sorted(set(y_true) | set(y_pred))
    accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / len(examples)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return RouterEvalReport(
        accuracy=accuracy, macro_f1=macro_f1, fallback_rate=fallback_count / len(examples),
        labels=labels, confusion_matrix=cm, n=len(examples),
    )


def threshold_sweep(
    router: Router, examples: list[EvalExample], thresholds: list[float],
) -> dict[float, RouterEvalReport]:
    """router 를 재구축(재임베딩)하지 않고 threshold 만 바꿔가며 평가한다
    (SemanticRouterAdapter.set_threshold 사용)."""
    results = {}
    for t in thresholds:
        if hasattr(router, "set_threshold"):
            router.set_threshold(t)
        results[t] = evaluate_router(router, examples)
    return results
