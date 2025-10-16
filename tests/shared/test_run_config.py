import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.shared.run_config import RunConfig


@pytest.mark.parametrize(
    "message_history, expected_turn",
    [
        ([], 0),  # Empty message history
        (  # Only system and assistant messages
            [
                SystemMessage(content="Do things!"),
                AIMessage(content="Hello!"),
            ],
            0,
        ),
        (
            [
                SystemMessage(content="Do things!"),
                HumanMessage(content="Hello!"),
            ],
            1,
        ),
        (
            [
                SystemMessage(content="Do things!"),
                HumanMessage(content="Hello!"),
                AIMessage(content="Hi there! How can I help you today?"),
                HumanMessage(content="Can you tell me a joke?"),
            ],
            2,
        ),
    ],
)
def test__count_user_turns__counts_accurately(message_history, expected_turn):
    assert RunConfig.count_user_turns(message_history) == expected_turn


@pytest.mark.parametrize(
    "input_messages, expected_output",
    [
        ([{"role": "user", "content": "Hello"}], [HumanMessage("Hello")]),
        (
            [{"role": "assistant", "content": "How can I help you?"}],
            [AIMessage("How can I help you?")],
        ),
        (
            [{"role": "system", "content": "Initializing..."}],
            [SystemMessage("Initializing...")],
        ),
        (
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "system", "content": "Processing..."},
            ],
            [HumanMessage("Hi"), AIMessage("Hello"), SystemMessage("Processing...")],
        ),
        (
            [
                {
                    "role": "system",
                    "content": "Ensure jsons are surrounded with single quotes to avoid errors from Langchain data validation.",
                },
                {"role": "user", "content": "Please generate a subject line"},
                {
                    "role": "assistant",
                    "content": '{"titles": {"subject_line": "a subject line"}}',
                },
            ],
            [
                SystemMessage(
                    "Ensure jsons are surrounded with single quotes to avoid errors from Langchain data validation."
                ),
                HumanMessage("Please generate a subject line"),
                AIMessage('\'{"titles": {"subject_line": "a subject line"}}\''),
            ],
        ),
    ],
)
def test__langchain_messages_from_list_of_dict__valid_roles(
    input_messages,
    expected_output,
):
    result = RunConfig.langchain_messages_from_list_of_dict(input_messages)
    assert result == expected_output


@pytest.mark.parametrize(
    "input_messages",
    [
        ([{"role": "invalid_role", "content": "Ignored message"}]),
        ([{"content": "Missing role"}]),
        ([{"role": "", "content": "Empty role"}]),
    ],
)
def test_langchain__messages_from_list_of_dict__invalid_roles(caplog, input_messages):
    with caplog.at_level(logging.WARNING):
        result = RunConfig.langchain_messages_from_list_of_dict(input_messages)

    # The output should be an empty list because the invalid roles should be skipped
    assert result == []

    # Check if the logger warning was called for the invalid role
    assert "WARNING" in caplog.text
    assert (
        caplog.messages[0]
        == "Message skipped due to missing or invalid role in conversation history. The valid message roles are: 'user', 'assistant', and 'system'."
    )


def test__langchain_messages_from_list_of_dict__empty_list():
    result = RunConfig.langchain_messages_from_list_of_dict([])
    assert result == []


def test__langchain_messages_from_list_of_dict__mixed_roles(caplog):
    input_messages = [
        {"role": "user", "content": "Hi"},
        {"role": "invalid_role", "content": "Ignored"},
        {"role": "assistant", "content": "Hello"},
    ]

    expected_output = [HumanMessage("Hi"), AIMessage("Hello")]

    with caplog.at_level(logging.WARNING):
        result = RunConfig.langchain_messages_from_list_of_dict(input_messages)
    assert result == expected_output

    assert "WARNING" in caplog.text
    assert (
        caplog.messages[0]
        == "Message skipped due to missing or invalid role in conversation history. The valid message roles are: 'user', 'assistant', and 'system'."
    )
