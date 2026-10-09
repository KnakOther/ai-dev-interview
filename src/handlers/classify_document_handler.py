import json
from typing import Any

from src.agents.classify_document.classify_document_agent import (
    ClassifyDocumentAgent,
)
from src.shared.exceptions import LlmRaisedException

from . import (
    health_check_response,
    process_common_request,
    setup_handler,
    success_response,
)


def handle(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Lambda function to classify a document by category, type, and sentiment.

    **Expected Event Structure:**

    ```json
    {
      "document_text": "String containing the document text",
      "custom_categories": ["Category 1", "Category 2"]
    }
    ```

    **Output:**

    ```json
    {
      "statusCode": 200,
      "headers": {"Content-Type": "application/json"},
      "body": {
            "primary_category": "Technology",
            "secondary_categories": ["Machine Learning", "Research"],
            "confidence_score": 0.95,
            "sentiment": "positive",
            "document_type": "article",
            "execution_details": {
                "prompt_tokens": 123,
                "completion_tokens": 456,
                "total_tokens": 579
            }
        }
    }
    ```
    """
    agent = ClassifyDocumentAgent()
    setup_handler(agent.name, event)

    # Process request
    data, error = process_common_request(
        event,
        required_fields=["document_text"],
        optional_fields=["custom_categories"],
    )

    if error:
        return error
    if data.get("health_check"):
        return health_check_response()

    # Set defaults
    if not data.get("custom_categories"):
        data["custom_categories"] = []

    # Run agent
    try:
        out = agent.run(
            input={
                "document_text": data["document_text"],
                "custom_categories": data["custom_categories"],
            }
        )
        return success_response(out)
    except LlmRaisedException as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"message": e.message}),
        }
