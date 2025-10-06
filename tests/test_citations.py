from src.generation.cite_guard import fill_missing_sources, validate


def test_citation_guard_detects_missing_sources():
    answer = "Aurora grew revenue by 20%."
    updated = fill_missing_sources(answer, ["source_1"])
    assert "Sources" in updated
    assert not validate(answer, ["source_1"])
    assert validate("Aurora grew revenue by 20%. [source_1]", ["source_1"])
