import json
from typing import Any, Dict

from src.agents.summarize_document.summarize_document_agent import (
    SummarizeDocumentAgent,
)
from src.shared.exceptions import LlmRaisedException

from . import (
    health_check_response,
    process_common_request,
    setup_handler,
    success_response,
)


def handle(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function to process a document and generate a summary.

    **Expected Event Structure:**

    ```json
    {
      "document_text": "String containing the document text to summarize",
      "max_length": 200,
      "language": "English"
    }
    ```

    **Output:**

    ```json
    {
      "statusCode": 200,
      "headers": {"Content-Type": "application/json"},
      "body": {
            "summary": "Concise document summary",
            "key_points": ["Point 1", "Point 2", "Point 3"],
            "word_count": 150,
            "execution_details": {
                "prompt_tokens": 123,
                "completion_tokens": 456,
                "total_tokens": 579
            }
        }
    }
    ```
    """
    agent = SummarizeDocumentAgent()
    setup_handler(agent.name, event)

    # Process request
    data, error = process_common_request(
        event,
        required_fields=["document_text"],
        optional_fields=["max_length", "language"],
    )

    if error:
        return error
    if data.get("health_check"):
        return health_check_response()

    # Set defaults
    data["max_length"] = data.get("max_length", 200)
    data["language"] = data.get("language", "English")

    # Run agent
    try:
        out = agent.run(
            input={
                "document_text": data["document_text"],
                "max_length": data["max_length"],
                "language": data["language"],
            }
        )
        return success_response(out)
    except LlmRaisedException as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"message": e.message}),
        }
