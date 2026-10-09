import json
from typing import Any

from src.shared.logger import logger
from src.shared.utils import setup_gcp_credentials

JSON_HEADERS = {"Content-Type": "application/json"}


def setup_handler(agent_name: str, event: dict[str, Any]) -> None:
    """Prepares the Lambda environment before running an agent."""
    setup_gcp_credentials()
    logger.info(f"Received a request for agent {agent_name}.")


def process_common_request(
    event: dict[str, Any],
    required_fields: list[str],
    optional_fields: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Parses the request payload and checks the required fields.

    The payload is either the event itself (direct Lambda invocation) or a JSON string in the
    event `body` (API Gateway). Returns the parsed data and an error response, if any.
    """
    body = event.get("body", event)

    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            return {}, error_response(400, "The request body is not valid JSON.")

    if body.get("health_check"):
        return {"health_check": True}, None

    missing_fields = [field for field in required_fields if field not in body]

    if missing_fields:
        return {}, error_response(
            400, f"Missing required fields: {', '.join(missing_fields)}."
        )

    allowed_fields = required_fields + (optional_fields or [])
    data = {field: body[field] for field in allowed_fields if field in body}

    return data, None


def health_check_response() -> dict[str, Any]:
    return {
        "statusCode": 200,
        "headers": JSON_HEADERS,
        "body": json.dumps({"status": "healthy"}),
    }


def success_response(agent_output: dict[str, Any]) -> dict[str, Any]:
    body = {
        **agent_output["result"],
        "execution_details": agent_output["execution_details"],
    }
    return {"statusCode": 200, "headers": JSON_HEADERS, "body": json.dumps(body)}


def error_response(status_code: int, message: str) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": JSON_HEADERS,
        "body": json.dumps({"message": message}),
    }
