import pytest
from app.agents.retriever_agent import BibleRetrieverAgent


def test_format_ai_structured_single():
    agent = BibleRetrieverAgent()
    retrieval_result = {
        "query_type": "exact_reference",
        "query": "John 3:16",
        "results": [
            type("V", (), {"reference": "John 3:16", "text": "For God so loved the world"})()
        ],
        "found": True,
        "version": "kjv"
    }

    structured = agent.format_ai_structured(retrieval_result)
    assert structured["type"] == "single"
    assert structured["query"] == "John 3:16"
    assert isinstance(structured["verses"], list)
    assert structured["meta"]["found"] is True


def test_format_ai_structured_range():
    agent = BibleRetrieverAgent()
    retrieval_result = {
        "query_type": "exact_range",
        "query": "Psalm 23:1-2",
        "results": [
            type("V", (), {"reference": "Psalm 23:1", "text": "The Lord is my shepherd"})(),
            type("V", (), {"reference": "Psalm 23:2", "text": "He maketh me to lie down"})(),
        ],
        "found": True,
        "version": "kjv"
    }

    structured = agent.format_ai_structured(retrieval_result)
    assert structured["type"] == "range"
    assert structured["query"] == "Psalm 23:1-2"
    assert len(structured["verses"]) == 2
    assert structured["meta"]["count"] == 2
