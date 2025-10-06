"""Evaluation datasets for the multimodal RAG system."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class EvaluationExample:
    question: str
    answer: str


def load_examples() -> List[EvaluationExample]:
    return [
        EvaluationExample(
            question="What sustainability improvements did Aurora report in 2024?",
            answer="Aurora reported an 18% carbon footprint reduction and renewable-powered warehouses.",
        ),
        EvaluationExample(
            question="Which product has the longest battery life?",
            answer="The Aurora Sense module offers up to 18 months of battery life.",
        ),
        EvaluationExample(
            question="Summarize Q3 revenue performance.",
            answer="Q3 revenue reached 16.8 million USD in North America.",
        ),
    ]
