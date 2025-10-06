"""Command-line evaluation harness."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from src.app.settings import get_settings
from src.evaluation.datasets import load_examples
from src.evaluation.metrics import MetricResult, run_ragas_evaluation
from src.generation.generator import LocalGenerator
from src.retrieval.hybrid import HybridRetriever


def run_eval(output: Path) -> MetricResult:
    settings = get_settings()
    retriever = HybridRetriever(index_dir=settings.index_dir, duckdb_path=settings.duckdb_path)
    generator = LocalGenerator()
    examples = load_examples()

    records: List[Dict[str, object]] = []
    for example in examples:
        retrieval = retriever.retrieve(example.question, top_k=5)
        contexts = retrieval["contexts"]
        answer = generator.generate(example.question, contexts)
        records.append(
            {
                "question": example.question,
                "answer": answer.answer,
                "contexts": [ctx.content for ctx in contexts],
                "ground_truth": example.answer,
            }
        )
    metrics = run_ragas_evaluation(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    radar_path = output.parent / "eval_radar.png"
    _write_report(output, metrics, radar_path, records)
    return metrics


def _write_report(path: Path, metrics: MetricResult, radar_path: Path, records: List[Dict[str, object]]) -> None:
    categories = [
        "Faithfulness",
        "Answer Relevancy",
        "Context Precision",
        "Context Recall",
    ]
    values = [
        metrics.faithfulness,
        metrics.answer_relevancy,
        metrics.context_precision,
        metrics.context_recall,
    ]
    angles = [n / float(len(categories)) * 2 * 3.14159 for n in range(len(categories))]
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.plot(angles, values, "o-", linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids([a * 180 / 3.14159 for a in angles[:-1]], categories)
    ax.set_ylim(0, 1)
    fig.savefig(radar_path, dpi=200)
    plt.close(fig)

    rows = [
        "| Metric | Score |",
        "| --- | --- |",
        f"| Faithfulness | {metrics.faithfulness:.2f} |",
        f"| Answer Relevancy | {metrics.answer_relevancy:.2f} |",
        f"| Context Precision | {metrics.context_precision:.2f} |",
        f"| Context Recall | {metrics.context_recall:.2f} |",
    ]
    adversarial = sum(1 for record in records if "source_" not in record["answer"])
    content = "\n".join(
        [
            "# Evaluation Report",
            "",
            "![RAG Metrics](eval_radar.png)",
            "",
            *rows,
            "",
            f"Adversarial citation violations: {adversarial}",
        ]
    )
    path.write_text(content, encoding="utf-8")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation for the multimodal RAG system")
    parser.add_argument("--output", type=Path, default=Path("docs/eval_report.md"))
    args = parser.parse_args()
    metrics = run_eval(args.output)
    print("Evaluation completed:", metrics)


if __name__ == "__main__":
    cli()
