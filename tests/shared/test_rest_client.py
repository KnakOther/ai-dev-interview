import json
from unittest.mock import Mock, patch

import pytest
import requests
from requests.exceptions import ConnectionError, RequestException, Timeout

from src.shared.rest_client import RestClient


@pytest.fixture
def rest_client():
    """Factory fixture for RestClient instances."""

    def _create_client(base_url="https://api.example.com", timeout=30, headers=None):
        return RestClient(base_url, timeout, headers)

    return _create_client


def test__init__sets_correct_properties(rest_client):
    client = rest_client("https://api.test.com", 60, {"Auth": "Bearer token"})

    assert client.base_url == "https://api.test.com"
    assert client.timeout == 60
    assert client.default_headers == {"Auth": "Bearer token"}


def test__init__strips_trailing_slash(rest_client):
    client = rest_client("https://api.test.com/")
    assert client.base_url == "https://api.test.com"


def test__init__default_values(rest_client):
    client = rest_client("https://api.test.com")

    assert client.timeout == 30
    assert client.default_headers == {}


@patch("src.shared.rest_client.requests.Session")
def test__post__successful_request(mock_session_class, rest_client):
    mock_session = Mock()
    mock_response = Mock()
    mock_response.json.return_value = {"success": True}
    mock_session.post.return_value = mock_response
    mock_session_class.return_value = mock_session

    client = rest_client()
    result = client.post("/users", {"name": "John"})

    assert result == {"success": True}
    mock_session.post.assert_called_once_with(
        "https://api.example.com/users", json={"name": "John"}, headers={}, timeout=30
    )
    mock_response.raise_for_status.assert_called_once()


@patch("src.shared.rest_client.requests.Session")
def test__post__with_headers(mock_session_class, rest_client):
    mock_session = Mock()
    mock_response = Mock()
    mock_response.json.return_value = {"data": "test"}
    mock_session.post.return_value = mock_response
    mock_session_class.return_value = mock_session

    client = rest_client(headers={"Default": "header"})
    client.post("/test", headers={"Custom": "header"})

    mock_session.post.assert_called_once_with(
        "https://api.example.com/test",
        json=None,
        headers={"Default": "header", "Custom": "header"},
        timeout=30,
    )


@patch("src.shared.rest_client.requests.Session")
def test__post__connection_error(mock_session_class, rest_client):
    mock_session = Mock()
    mock_session.post.side_effect = ConnectionError("Connection failed")
    mock_session_class.return_value = mock_session

    client = rest_client()

    with pytest.raises(
        RequestException, match="Network error calling.*Connection failed"
    ):
        client.post("/test")


@patch("src.shared.rest_client.requests.Session")
def test__post__timeout_error(mock_session_class, rest_client):
    mock_session = Mock()
    mock_session.post.side_effect = Timeout("Request timed out")
    mock_session_class.return_value = mock_session

    client = rest_client()

    with pytest.raises(
        RequestException, match="Network error calling.*Request timed out"
    ):
        client.post("/test")


@patch("src.shared.rest_client.requests.Session")
def test__post__json_decode_error(mock_session_class, rest_client):
    mock_session = Mock()
    mock_response = Mock()
    mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
    mock_session.post.return_value = mock_response
    mock_session_class.return_value = mock_session

    client = rest_client()

    with pytest.raises(ValueError, match="Invalid JSON response from.*Invalid JSON"):
        client.post("/test")


@patch("src.shared.rest_client.requests.Session")
def test__post__http_error(mock_session_class, rest_client):
    mock_session = Mock()
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
    mock_session.post.return_value = mock_response
    mock_session_class.return_value = mock_session

    client = rest_client()

    with pytest.raises(requests.HTTPError):
        client.post("/test")


def test__post__endpoint_path_handling(rest_client):
    with patch("src.shared.rest_client.requests.Session") as mock_session_class:
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session

        client = rest_client()
        client.post("/users")  # Leading slash
        client.post("users")  # No leading slash

        calls = mock_session.post.call_args_list
        print(calls)
        assert calls[0][0][0] == "https://api.example.com/users"
        assert calls[1][0][0] == "https://api.example.com/users"
