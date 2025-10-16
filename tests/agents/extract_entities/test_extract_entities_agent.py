import pytest

from src.agents.extract_entities.extract_entities_agent import ExtractEntitiesAgent


@pytest.fixture
def agent():
    return ExtractEntitiesAgent()


def test_extract_entities_basic(agent):
    """Test basic entity extraction."""
    input_data = {
        "document_text": """
        On January 15, 2024, CEO Sarah Johnson announced that TechCorp Inc. would be opening
        a new research facility in San Francisco. The company, based in New York, has been
        working on advanced machine learning systems. Dr. Michael Chen will lead the team,
        focusing on neural networks and deep learning architectures.
        """
    }

    response = agent.generate_response(input_data)

    assert response.output is not None
    assert "people" in response.output
    assert "organizations" in response.output
    assert "locations" in response.output
    assert "dates" in response.output
    assert isinstance(response.output["people"], list)
    assert isinstance(response.output["organizations"], list)


def test_extract_entities_empty_document(agent):
    """Test entity extraction with minimal content."""
    input_data = {"document_text": "This is a simple sentence with no named entities."}

    response = agent.generate_response(input_data)

    assert response.output is not None
    assert "people" in response.output
    assert "organizations" in response.output
