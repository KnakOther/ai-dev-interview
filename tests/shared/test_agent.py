import logging
from typing import Any

import pytest
from langchain_core.messages.ai import UsageMetadata

from src.shared.agent import Agent, AgentResponse
from src.shared.exceptions import FriendlyException, LlmRaisedException


@pytest.fixture
def agent():
    """Concrete implementation of the abstract Agent class for testing purposes."""

    class TestAgent(Agent):
        name = "TestAgent"
        model = "TestModel"
        kwargs_received = None

        def generate_response(
            self,
            input: Any,
            *args: Any,
            **kwargs: Any,
        ) -> AgentResponse:
            self.kwargs_received = kwargs
            if input == "fail":
                raise ValueError("Forced failure")
            return AgentResponse(
                {"output": input},
                UsageMetadata(input_tokens=1, output_tokens=2, total_tokens=3),
            )

    return TestAgent


def test__run__successful_run(caplog, agent):
    caplog.set_level(logging.INFO)

    assert agent().run("success") == {
        "result": {"output": "success"},
        "execution_details": UsageMetadata(
            input_tokens=1, output_tokens=2, total_tokens=3
        ),
    }

    # Test logging
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "INFO"
    assert caplog.records[0].message == "Generated a response with agent TestAgent."
    assert caplog.records[0].retries == 0
    assert caplog.records[0].turn == 1
    assert caplog.records[0].input == "success"
    assert caplog.records[0].model == "TestModel"


def test__run__retries_on_failure(caplog, agent):
    caplog.set_level(logging.ERROR)

    with pytest.raises(Exception) as exc_info:
        agent().run("fail", retries=3)

    assert "Failed to generate a response with agent TestAgent" in str(exc_info.value)
    assert len(caplog.records) == 4  # 3 retries + 1 final failure


def test__run__conversation_history_passed(agent):
    conversation_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    result = agent().run("test", conversation_history=conversation_history)
    assert result["result"]["output"] == "test"


def test__run__thread_id_passed(agent):
    result = agent().run("test", thread_id=42)
    assert result["result"]["output"] == "test"


def test__run__friendly_exception_is_propagated(caplog, agent):
    class FriendlyExceptionAgent(Agent):
        name = "FriendlyExceptionAgent"
        model = "TestModel"

        def generate_response(
            self, input: Any, *args: Any, **kwargs: Any
        ) -> AgentResponse:
            raise FriendlyException("User-friendly error message")

    caplog.set_level(logging.WARNING)

    with pytest.raises(FriendlyException) as exc_info:
        FriendlyExceptionAgent().run("test")

    assert str(exc_info.value) == "User-friendly error message"
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"


def test__run__llm_raised_exception_is_propagated(agent):
    class LlmExceptionAgent(Agent):
        name = "LlmExceptionAgent"
        model = "TestModel"

        def generate_response(
            self, input: Any, *args: Any, **kwargs: Any
        ) -> AgentResponse:
            raise LlmRaisedException("LLM error message")

    with pytest.raises(LlmRaisedException) as exc_info:
        LlmExceptionAgent().run("test")

    assert str(exc_info.value) == "LLM error message"


def test__run__invalid_retries_raises_error(agent):
    with pytest.raises(ValueError) as exc_info:
        agent().run("test", retries=0)

    assert "'retries' must be greater than 0" in str(exc_info.value)


def test__run__kwargs_are_passed_to_generate_response(agent):
    test_agent = agent()
    test_agent.run("test", custom_param="custom_value", another_param=123)

    assert test_agent.kwargs_received["custom_param"] == "custom_value"
    assert test_agent.kwargs_received["another_param"] == 123
