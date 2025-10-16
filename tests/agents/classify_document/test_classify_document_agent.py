import pytest

from src.agents.classify_document.classify_document_agent import (
    ClassifyDocumentAgent,
)


@pytest.fixture
def agent():
    return ClassifyDocumentAgent()


def test_classify_document_basic(agent):
    """Test basic document classification."""
    input_data = {
        "document_text": """
        Our Q4 financial results show strong growth across all sectors. Revenue increased by 25%
        year-over-year, driven primarily by our cloud services division. Operating expenses were
        well-controlled, resulting in improved margins. We anticipate continued momentum in the
        coming quarters as we expand into new markets.
        """
    }

    response = agent.generate_response(input_data)

    assert response.output is not None
    assert "primary_category" in response.output
    assert "confidence_score" in response.output
    assert "sentiment" in response.output
    assert "document_type" in response.output
    assert 0.0 <= response.output["confidence_score"] <= 1.0
    assert response.output["sentiment"] in ["positive", "negative", "neutral", "mixed"]


def test_classify_document_with_custom_categories(agent):
    """Test classification with custom categories."""
    input_data = {
        "document_text": """
        The new machine learning model demonstrates state-of-the-art performance on the benchmark
        dataset. Our novel architecture combines attention mechanisms with convolutional layers.
        """,
        "custom_categories": ["Machine Learning", "Research", "Technology"],
    }

    response = agent.generate_response(input_data)

    assert response.output is not None
    assert len(response.output["primary_category"]) > 0
