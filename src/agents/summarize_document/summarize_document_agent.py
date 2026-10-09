from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import InputAgentState
from langchain.agents.structured_output import ProviderStrategy
from pydantic import BaseModel, Field

from src.agents.summarize_document.summarize_document_prompt import prompt_template
from src.shared.agent import Agent, AgentResponse
from src.shared.gemini import get_chat_model


class DocumentSummary(BaseModel):
    summary: str = Field(..., description="A concise summary of the document")
    key_points: list[str] = Field(
        ..., min_length=3, max_length=5, description="3-5 key points from the document"
    )
    word_count: int = Field(..., description="Approximate word count of the summary")


class SummarizeDocumentAgentInput(BaseModel):
    document_text: str
    max_length: int = 200
    language: str = "English"


class SummarizeDocumentAgent(Agent):
    def __init__(self) -> None:
        self.name = "Summarize document agent"
        self.model = "gemini-3.5-flash"
        self.input_schema = SummarizeDocumentAgentInput

    def generate_response(
        self,
        input: dict,
        *args: Any,
        **kwargs: Any,
    ) -> AgentResponse:
        agent = create_agent(
            model=get_chat_model(self.model),
            tools=[],
            response_format=ProviderStrategy(schema=DocumentSummary),
            name="summarize_document",
        )
        _input = input.copy()

        if not _input.get("language"):
            _input["language"] = "English"

        if not _input.get("max_length"):
            _input["max_length"] = 200

        agent_result = agent.invoke(
            cast(
                InputAgentState,
                {"messages": prompt_template.format_messages(**_input)},
            )
        )

        parsed_output, usage_metadata = self.unpack_agent_result(agent_result)

        if not isinstance(parsed_output, DocumentSummary):
            raise ValueError(
                f"Wrong structured output type. Expected {DocumentSummary}. Got {type(parsed_output)} instead."
            )

        if len(parsed_output.summary) == 0:
            raise ValueError("Generated summary is empty.")

        return AgentResponse(parsed_output.model_dump(), usage_metadata)
