from src.generation.generator import LocalGenerator
from src.retrieval.hybrid import Context


def test_local_generator_produces_citations():
    generator = LocalGenerator()
    contexts = [
        Context(
            source_id="source_1",
            content="Aurora reduced carbon footprint by 18%.",
            metadata={"modality": "text"},
            modality="text",
            score=0.9,
        ),
        Context(
            source_id="source_2",
            content="Aurora Sense runs for 18 months on battery.",
            metadata={"modality": "text"},
            modality="text",
            score=0.8,
        ),
    ]
    result = generator.generate("What are the highlights?", contexts)
    assert "source_1" in result.answer
    assert result.citations
