from typing import Any

from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field

from src.agents.extract_entities.extract_entities_prompt import prompt_template
from src.shared.agent import Agent, AgentResponse
from src.shared.gemini import GEMINI_CONFIG


class ExtractedEntities(BaseModel):
    people: list[str] = Field(
        default_factory=list, description="Names of people mentioned"
    )
    organizations: list[str] = Field(
        default_factory=list, description="Organizations mentioned"
    )
    locations: list[str] = Field(
        default_factory=list, description="Locations mentioned"
    )
    dates: list[str] = Field(default_factory=list, description="Dates mentioned")
    key_terms: list[str] = Field(
        default_factory=list, description="Important technical or domain-specific terms"
    )


class ExtractEntitiesAgentInput(BaseModel):
    document_text: str
    entity_types: list[str] = Field(
        default_factory=lambda: ["people", "organizations", "locations", "dates", "key_terms"]
    )


class ExtractEntitiesAgent(Agent):
    def __init__(self) -> None:
        self.name = "Extract entities agent"
        self.model = "gemini-2.5-flash"
        self.input_schema = ExtractEntitiesAgentInput

    def generate_response(
        self,
        input: dict,
        *args: Any,
        **kwargs: Any,
    ) -> AgentResponse:
        chat = ChatVertexAI(model=self.model, **GEMINI_CONFIG).with_structured_output(
            schema=ExtractedEntities, include_raw=True
        )
        _input = input.copy()

        if not _input.get("entity_types"):
            _input["entity_types"] = [
                "people",
                "organizations",
                "locations",
                "dates",
                "key_terms",
            ]

        llm_output = (prompt_template | chat).invoke(_input)

        raw_output, parsed_output = self.unpack_llm_structured_output(llm_output)

        if not isinstance(parsed_output, ExtractedEntities):
            raise ValueError(
                f"Wrong structured output type. Expected {ExtractedEntities}. Got {type(parsed_output)} instead."
            )

        return AgentResponse(parsed_output.model_dump(), raw_output.usage_metadata)
