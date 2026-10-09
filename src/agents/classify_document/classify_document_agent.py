from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import InputAgentState
from langchain.agents.structured_output import ProviderStrategy
from pydantic import BaseModel, Field

from src.agents.classify_document.classify_document_prompt import (
    custom_categories_instruction,
    prompt_template,
)
from src.shared.agent import Agent, AgentResponse
from src.shared.gemini import get_chat_model


class DocumentClassification(BaseModel):
    primary_category: str = Field(..., description="Primary category of the document")
    secondary_categories: list[str] = Field(
        default_factory=list, description="Additional relevant categories"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )
    sentiment: str = Field(
        ..., description="Overall sentiment: positive, negative, neutral, or mixed"
    )
    document_type: str = Field(
        ...,
        description="Type of document: article, report, email, memo, proposal, etc.",
    )


class ClassifyDocumentAgentInput(BaseModel):
    document_text: str
    custom_categories: list[str] = Field(default_factory=list)


class ClassifyDocumentAgent(Agent):
    def __init__(self) -> None:
        self.name = "Classify document agent"
        self.model = "gemini-3.5-flash"
        self.input_schema = ClassifyDocumentAgentInput

    def generate_response(
        self,
        input: dict,
        *args: Any,
        **kwargs: Any,
    ) -> AgentResponse:
        agent = create_agent(
            model=get_chat_model(self.model),
            tools=[],
            response_format=ProviderStrategy(schema=DocumentClassification),
            name="classify_document",
        )
        _input = input.copy()

        if not _input.get("custom_categories"):
            _input["custom_categories"] = []

        _input["custom_categories_instruction"] = custom_categories_instruction(
            _input["custom_categories"]
        )

        agent_result = agent.invoke(
            cast(
                InputAgentState,
                {"messages": prompt_template.format_messages(**_input)},
            )
        )

        parsed_output, usage_metadata = self.unpack_agent_result(agent_result)

        if not isinstance(parsed_output, DocumentClassification):
            raise ValueError(
                f"Wrong structured output type. Expected {DocumentClassification}. Got {type(parsed_output)} instead."
            )

        if len(parsed_output.primary_category) == 0:
            raise ValueError("Primary category cannot be empty.")

        return AgentResponse(parsed_output.model_dump(), usage_metadata)
