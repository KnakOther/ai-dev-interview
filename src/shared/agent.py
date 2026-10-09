import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.messages.ai import UsageMetadata, add_usage
from pydantic import BaseModel, ValidationError

from src.shared.exceptions import FriendlyException
from src.shared.logger import logger
from src.shared.run_config import RunConfig


@dataclass
class AgentResponse:
    result: dict | str | list | BaseModel
    execution_details: UsageMetadata | None = None


class Agent(ABC):
    """Abstract base class for all AI agents.

    An AI agent is responsible for processing input and generating a response
    based on interactions with a large language model (LLM). Agents can serve
    various purposes, such as engaging in conversation with a user, cleaning
    text, or triggering specific actions.

    Subclasses must implement the `generate_response` method, which should
    return an `AgentResponse` object. The `run` method serves as a wrapper
    around `generate_response`, providing exception handling, retries, and
    logging functionality.

    The `run` method is the main entry point for using an agent.
    """

    name: str
    model: str
    input_schema: type[BaseModel] | None = None

    def run(
        self,
        input: Any,
        conversation_history: list[dict] | None = None,
        thread_id: int | str | None = None,
        retries: int = 3,
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        if retries < 1:
            raise ValueError("'retries' must be greater than 0.")

        self.validate_input(input)

        run_config = RunConfig(input, conversation_history, thread_id)

        _retries = 0

        while _retries < retries:
            try:
                before = time.time()
                agent_response = self.generate_response(
                    run_config.input,
                    conversation_history=run_config.conversation_history,
                    *args,
                    **kwargs,
                )

                after = time.time()
                execution_duration = after - before

                self._log_succesful_generation(
                    agent_response, execution_duration, _retries, run_config
                )

                return asdict(agent_response)

            except FriendlyException as e:
                self._log_unsuccesful_generation(str(e), _retries, run_config, e)
                raise (e)

            except Exception as e:
                self._log_unsuccesful_generation(
                    f"{self.generic_failure_message} Retrying...",
                    _retries,
                    run_config,
                    e,
                    extra={"error_message": str(e)},
                )

            _retries += 1

        final_failure_message = f"{self.generic_failure_message}{' Retries exhausted.' if retries > 0 else ''}"

        self._log_unsuccesful_generation(final_failure_message, _retries, run_config)

        raise Exception(final_failure_message)

    @abstractmethod
    def generate_response(self, input: Any, *args: Any, **kwargs: Any) -> AgentResponse:
        pass

    @property
    def generic_failure_message(self) -> str:
        return f"Failed to generate a response with agent {self.name}."

    def validate_input(self, input: dict) -> None:
        if self.input_schema:
            if not isinstance(input, dict):
                raise ValueError(
                    "The input data must be a dictionary when a schema is provided via self.input_schema."
                )

            try:
                self.input_schema(**input)

            except ValidationError as e:
                raise ValueError(
                    f"The input data does not match the schema: {e.errors()}"
                )

    @staticmethod
    def unpack_agent_result(
        agent_result: dict,
    ) -> tuple[BaseModel, UsageMetadata | None]:
        """Unpack a LangChain `create_agent` result when using `response_format` and validate its typing.

        Returns the structured response and the token usage summed across every model call of the agent loop.
        """
        if not isinstance(
            agent_result, dict
        ):  # A compiled agent graph returns its final state as a dict.
            raise ValueError(
                f"The Langchain agent result is {type(agent_result)}. It should be a dict.",
            )

        parsed_output = agent_result.get("structured_response")

        if not isinstance(parsed_output, BaseModel):
            raise ValueError(
                f"The Langchain agent structured response has type {type(parsed_output)}. It should be a BaseModel.",
                0,
            )

        usage_metadata: UsageMetadata | None = None

        for message in agent_result.get("messages", []):
            if isinstance(message, AIMessage) and message.usage_metadata:
                usage_metadata = add_usage(usage_metadata, message.usage_metadata)

        return parsed_output, usage_metadata

    @staticmethod
    def cast_llm_output_to_ai_message(llm_output: AIMessage | BaseModel) -> AIMessage:
        if not isinstance(
            llm_output,
            AIMessage,
        ):  # To avoid receiving other types of BaseMessages.
            raise ValueError(
                f"The Langchain result of the output has type {type(llm_output)}. It should be an AIMessage.",
            )
        return llm_output

    def _log_succesful_generation(
        self,
        agent_response: AgentResponse,
        execution_duration: float,
        retries: int,
        run_config: RunConfig,
    ):
        logging_payload = {
            "retries": retries,
            "llm_duration_seconds": execution_duration,
            "turn": run_config.turn,
            "input": run_config.input,
            "conversation_history": run_config.conversation_history,
            "agent_response": asdict(agent_response),
            "model": self.model,
            "thread_id": run_config.thread_id,
        }

        logger.info(
            f"Generated a response with agent {self.name}.",
            extra=logging_payload,
        )

    def _log_unsuccesful_generation(
        self,
        message: str,
        retries: int,
        run_config: RunConfig,
        error: Exception | None = None,
        extra: dict | None = None,
    ):
        logging_payload = {
            "retries": retries,
            "turn": run_config.turn,
            "input": run_config.input,
            "conversation_history": run_config.conversation_history,
            "model": self.model,
            "thread_id": run_config.thread_id,
        }

        if error:
            logging_payload["error_type"] = type(error).__name__

        if extra:
            logging_payload.update(extra)

        if isinstance(error, FriendlyException):
            logger.warning(message, extra=logging_payload)

        else:
            logger.error(message, extra=logging_payload)
