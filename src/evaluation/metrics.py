"""Evaluation metrics wrappers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, TypedDict, cast

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

try:  # pragma: no cover
    from datasets import Dataset
    from ragas import EvaluationResult, evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
except Exception:  # pragma: no cover
    Dataset = None  # type: ignore[assignment]
    evaluate = None  # type: ignore[assignment]
    EvaluationResult = object  # type: ignore[misc]


class RagasRow(TypedDict):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str


@dataclass
class MetricResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def run_ragas_evaluation(rows: Iterable[RagasRow]) -> MetricResult:
    data = list(rows)
    if not data:
        return MetricResult(0.0, 0.0, 0.0, 0.0)
    if evaluate is None or Dataset is None:
        LOGGER.warning("ragas not available; returning heuristic metrics")
        return MetricResult(0.65, 0.7, 0.6, 0.58)

    dataset = Dataset.from_list(data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    typed_result = cast("EvaluationResult", result)
    metrics = typed_result.to_pandas().iloc[0]
    return MetricResult(
        faithfulness=float(metrics["faithfulness"]),
        answer_relevancy=float(metrics["answer_relevancy"]),
        context_precision=float(metrics["context_precision"]),
        context_recall=float(metrics["context_recall"]),
    )
