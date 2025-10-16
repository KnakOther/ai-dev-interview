import json
from typing import Any, Dict, Optional

import requests
from requests.exceptions import ConnectionError, RequestException, Timeout


class RestClient:
    """A client for making REST API calls via POST requests."""

    def __init__(
        self, base_url: str, timeout: int = 30, headers: Optional[Dict[str, str]] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_headers = headers or {}
        self.session = requests.Session()
        self.session.headers.update(self.default_headers)

    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make a POST request to the specified endpoint.

        Args:
            endpoint: API endpoint path
            data: Request payload
            headers: Additional headers for this request

        Returns:
            JSON response as dictionary

        Raises:
            RequestException: For HTTP errors
            ValueError: For invalid JSON responses
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_headers = {**self.default_headers}
        if headers:
            request_headers.update(headers)

        try:
            response = self.session.post(
                url, json=data, headers=request_headers, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()

        except (ConnectionError, Timeout) as e:
            raise RequestException(f"Network error calling {url}: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from {url}: {str(e)}")
