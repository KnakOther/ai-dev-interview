import json
from typing import Any

from src.agents.extract_entities.extract_entities_agent import ExtractEntitiesAgent
from src.shared.exceptions import LlmRaisedException

from . import (
    health_check_response,
    process_common_request,
    setup_handler,
    success_response,
)


def handle(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Lambda function to extract named entities from a document.

    **Expected Event Structure:**

    ```json
    {
      "document_text": "String containing the document text",
      "entity_types": ["people", "organizations", "locations", "dates", "key_terms"]
    }
    ```

    **Output:**

    ```json
    {
      "statusCode": 200,
      "headers": {"Content-Type": "application/json"},
      "body": {
            "people": ["Person 1", "Person 2"],
            "organizations": ["Org 1", "Org 2"],
            "locations": ["Location 1"],
            "dates": ["Date 1"],
            "key_terms": ["Term 1", "Term 2"],
            "execution_details": {
                "prompt_tokens": 123,
                "completion_tokens": 456,
                "total_tokens": 579
            }
        }
    }
    ```
    """
    agent = ExtractEntitiesAgent()
    setup_handler(agent.name, event)

    # Process request
    data, error = process_common_request(
        event, required_fields=["document_text"], optional_fields=["entity_types"]
    )

    if error:
        return error
    if data.get("health_check"):
        return health_check_response()

    # Set defaults
    if not data.get("entity_types"):
        data["entity_types"] = [
            "people",
            "organizations",
            "locations",
            "dates",
            "key_terms",
        ]

    # Run agent
    try:
        out = agent.run(
            input={
                "document_text": data["document_text"],
                "entity_types": data["entity_types"],
            }
        )
        return success_response(out)
    except LlmRaisedException as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"message": e.message}),
        }
