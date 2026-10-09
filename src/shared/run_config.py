import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.shared.logger import logger
from src.shared.role import Role


class RunConfig:
    input: Any
    conversation_history: list[BaseMessage]
    thread_id: int | str | None

    def __init__(
        self,
        input: Any,
        conversation_history: list[dict] | None,
        thread_id: int | str | None = None,
    ):
        self.input = input

        if not conversation_history:
            conversation_history = []

        self.conversation_history = self.langchain_messages_from_list_of_dict(
            conversation_history
        )

        self.thread_id = thread_id

    @property
    def turn(self) -> int:
        return self.count_user_turns(self.conversation_history) + 1

    @staticmethod
    def count_user_turns(conversation_history: list[BaseMessage]) -> int:
        """Counts the number of conversation turns where the user's role is involved.

        Args:
            message_history (list[BaseMessage]): The list of messages in the chat history.

        Returns:
            int: The number of user turns.

        """
        return len(
            [msg for msg in conversation_history if isinstance(msg, HumanMessage)],
        )

    @staticmethod
    def langchain_messages_from_list_of_dict(messages: list[dict]) -> list[BaseMessage]:
        """Converts a list of message dictionaries into LangChain message objects.

        Each dictionary should have a "role" ("user", "assistant", or "system") and "content".
        Messages with invalid or missing roles are skipped.

        Args:
            messages (list[dict]): List of message dictionaries with "role" and "content".

        Returns:
            list[BaseMessage]: List of LangChain message objects.

        """
        out: list[BaseMessage] = []

        for message in messages:
            content = message["content"]

            try:
                # Load and re-encode JSON to fix surrogate character issues by preserving unicode.
                # Wrapping between additional quotes prevents eventual data validation errors from Langchain.
                content = json.loads(content)
                cleaned_content = f"'{json.dumps(content, ensure_ascii=False)}'"

            except json.JSONDecodeError:
                cleaned_content = content

            if message.get("role") == Role.USER.value:
                out.append(HumanMessage(cleaned_content))

            elif message.get("role") == Role.ASSISTANT.value:
                out.append(AIMessage(cleaned_content))

            elif message.get("role") == Role.SYSTEM.value:
                out.append(SystemMessage(cleaned_content))

            else:
                logger.warning(
                    "Message skipped due to missing or invalid role in conversation history. The valid message roles are: 'user', 'assistant', and 'system'.",
                )

        return out
