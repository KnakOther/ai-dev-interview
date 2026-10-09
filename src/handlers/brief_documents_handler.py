import json
from typing import Any

from src.agents.brief_documents.brief_documents_agent import BriefDocumentsAgent
from src.shared.exceptions import LlmRaisedException

from . import (
    health_check_response,
    process_common_request,
    setup_handler,
    success_response,
)


def handle(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Lambda function to write an executive briefing from a collection of documents.

    **Expected Event Structure:**

    ```json
    {
      "documents": [{"title": "Q3 report", "text": "..."}, {"title": "Board memo", "text": "..."}],
      "objective": "Write an executive briefing of the documents.",
      "language": "English",
      "max_summary_length": 250
    }
    ```

    **Output:**

    ```json
    {
      "statusCode": 200,
      "headers": {"Content-Type": "application/json"},
      "body": {
            "executive_summary": "...",
            "documents": [{"path": "/documents/q3-report.md", "summary": "...", "primary_category": "Finance", "key_entities": ["..."]}],
            "cross_document_themes": ["..."],
            "contradictions": ["..."],
            "open_questions": ["..."],
            "execution_details": {
                "input_tokens": 123,
                "output_tokens": 456,
                "total_tokens": 579
            }
        }
    }
    ```
    """
    agent = BriefDocumentsAgent()
    setup_handler(agent.name, event)

    # Process request
    data, error = process_common_request(
        event,
        required_fields=["documents"],
        optional_fields=["objective", "language", "max_summary_length"],
    )

    if error:
        return error
    if data.get("health_check"):
        return health_check_response()

    # Run agent. Deep agent runs are long and expensive: a single attempt, the agent recovers from its own errors.
    try:
        out = agent.run(input=data, retries=1)
        return success_response(out)
    except LlmRaisedException as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"message": e.message}),
        }
