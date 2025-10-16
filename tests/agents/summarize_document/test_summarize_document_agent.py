import pytest

from src.agents.summarize_document.summarize_document_agent import (
    SummarizeDocumentAgent,
)


@pytest.fixture
def agent():
    return SummarizeDocumentAgent()


def test_summarize_document_basic(agent):
    """Test basic document summarization."""
    input_data = {
        "document_text": """
        Artificial intelligence has transformed the technology landscape over the past decade.
        Machine learning algorithms now power everything from recommendation systems to autonomous vehicles.
        Natural language processing has enabled computers to understand and generate human language with
        remarkable accuracy. Deep learning techniques have achieved superhuman performance in tasks like
        image recognition and game playing. However, challenges remain in areas such as explainability,
        bias, and general intelligence.
        """,
        "max_length": 100,
        "language": "English",
    }

    response = agent.generate_response(input_data)

    assert response.output is not None
    assert "summary" in response.output
    assert "key_points" in response.output
    assert len(response.output["key_points"]) >= 3
    assert len(response.output["key_points"]) <= 5
    assert isinstance(response.output["word_count"], int)


def test_summarize_document_short_text(agent):
    """Test summarization with very short input."""
    input_data = {
        "document_text": "AI is changing the world.",
        "max_length": 50,
    }

    response = agent.generate_response(input_data)

    assert response.output is not None
    assert len(response.output["summary"]) > 0
