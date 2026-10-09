from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import InputAgentState
from langchain.agents.structured_output import ProviderStrategy
from pydantic import BaseModel, Field

from src.agents.extract_entities.extract_entities_prompt import prompt_template
from src.shared.agent import Agent, AgentResponse
from src.shared.gemini import get_chat_model


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
        default_factory=lambda: [
            "people",
            "organizations",
            "locations",
            "dates",
            "key_terms",
        ]
    )


class ExtractEntitiesAgent(Agent):
    def __init__(self) -> None:
        self.name = "Extract entities agent"
        self.model = "gemini-3.5-flash"
        self.input_schema = ExtractEntitiesAgentInput

    def generate_response(
        self,
        input: dict,
        *args: Any,
        **kwargs: Any,
    ) -> AgentResponse:
        agent = create_agent(
            model=get_chat_model(self.model),
            tools=[],
            response_format=ProviderStrategy(schema=ExtractedEntities),
            name="extract_entities",
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

        agent_result = agent.invoke(
            cast(
                InputAgentState,
                {"messages": prompt_template.format_messages(**_input)},
            )
        )

        parsed_output, usage_metadata = self.unpack_agent_result(agent_result)

        if not isinstance(parsed_output, ExtractedEntities):
            raise ValueError(
                f"Wrong structured output type. Expected {ExtractedEntities}. Got {type(parsed_output)} instead."
            )

        return AgentResponse(parsed_output.model_dump(), usage_metadata)
