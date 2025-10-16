from typing import Any

from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field

from src.agents.classify_document.classify_document_prompt import prompt_template
from src.shared.agent import Agent, AgentResponse
from src.shared.gemini import GEMINI_CONFIG


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
        self.model = "gemini-2.5-flash"
        self.input_schema = ClassifyDocumentAgentInput

    def generate_response(
        self,
        input: dict,
        *args: Any,
        **kwargs: Any,
    ) -> AgentResponse:
        chat = ChatVertexAI(model=self.model, **GEMINI_CONFIG).with_structured_output(
            schema=DocumentClassification, include_raw=True
        )
        _input = input.copy()

        if not _input.get("custom_categories"):
            _input["custom_categories"] = []

        llm_output = (prompt_template | chat).invoke(_input)

        raw_output, parsed_output = self.unpack_llm_structured_output(llm_output)

        if not isinstance(parsed_output, DocumentClassification):
            raise ValueError(
                f"Wrong structured output type. Expected {DocumentClassification}. Got {type(parsed_output)} instead."
            )

        if len(parsed_output.primary_category) == 0:
            raise ValueError("Primary category cannot be empty.")

        return AgentResponse(parsed_output.model_dump(), raw_output.usage_metadata)
