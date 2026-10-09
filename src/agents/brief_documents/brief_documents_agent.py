from functools import reduce
from typing import Any, cast

from deepagents import SubAgent, create_deep_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    InputAgentState,
    ModelCallLimitMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
)
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import add_usage
from pydantic import BaseModel, Field

from src.agents.brief_documents.brief_documents_prompt import (
    FACT_CHECKER_PROMPT,
    HUMAN_PROMPT,
    SYSTEM_PROMPT,
)
from src.agents.brief_documents.brief_documents_tools import (
    DOCUMENTS_DIRECTORY,
    build_document_files,
    classify_document,
    count_words,
    document_not_found_error_handler,
    extract_entities,
    summarize_document,
)
from src.shared.agent import Agent, AgentResponse
from src.shared.constants import GEMINI_3_1_PRO, GEMINI_3_5_FLASH
from src.shared.gemini import get_chat_model


class DocumentFinding(BaseModel):
    path: str = Field(..., description="File path of the document")
    summary: str = Field(
        ..., description="Two to three sentence summary of the document"
    )
    primary_category: str = Field(..., description="Primary category of the document")
    key_entities: list[str] = Field(
        default_factory=list, description="Most important entities of the document"
    )


class DocumentBrief(BaseModel):
    executive_summary: str = Field(
        ..., description="Executive summary of the whole document collection"
    )
    documents: list[DocumentFinding] = Field(
        ..., min_length=1, description="One finding per source document"
    )
    cross_document_themes: list[str] = Field(
        default_factory=list, description="Themes shared by several documents"
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="Facts or figures that contradict each other across documents",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Questions the documents raise but do not answer",
    )


class Document(BaseModel):
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class BriefDocumentsAgentInput(BaseModel):
    documents: list[Document] = Field(..., min_length=1, max_length=20)
    objective: str = "Write an executive briefing of the documents."
    language: str = "English"
    max_summary_length: int = Field(default=250, gt=0)


class BriefDocumentsAgent(Agent):
    """Deep agent that writes an executive briefing from a collection of documents.

    Unlike the single-call agents, this agent plans its work (`write_todos`), works on a virtual filesystem
    (`ls`, `read_file`, `write_file`, ...), calls the other agents of this service as tools, and delegates the
    verification of its draft to a `fact-checker` subagent running in an isolated context (`task`).
    """

    def __init__(self) -> None:
        self.name = "Brief documents deep agent"
        self.model = GEMINI_3_1_PRO
        self.subagent_model = GEMINI_3_5_FLASH
        self.input_schema = BriefDocumentsAgentInput
        self.max_model_calls = 40
        self.max_tool_calls = 60

    def generate_response(
        self,
        input: dict,
        *args: Any,
        **kwargs: Any,
    ) -> AgentResponse:
        _input = BriefDocumentsAgentInput(**input)

        fact_checker: SubAgent = {
            "name": "fact-checker",
            "description": "Verifies every claim of a draft briefing against the source documents. "
            "Send it the full draft; it returns the unsupported or contradicted claims.",
            "system_prompt": FACT_CHECKER_PROMPT.format(
                documents_directory=DOCUMENTS_DIRECTORY
            ),
            "model": get_chat_model(self.subagent_model),
        }

        middleware: list[AgentMiddleware[Any, Any, Any]] = [
            TodoListMiddleware(),
            ToolErrorMiddleware(on_error=document_not_found_error_handler),
            ModelCallLimitMiddleware(
                run_limit=self.max_model_calls, exit_behavior="error"
            ),
            ToolCallLimitMiddleware(
                run_limit=self.max_tool_calls, exit_behavior="error"
            ),
        ]

        agent = create_deep_agent(
            model=get_chat_model(self.model),
            tools=[
                summarize_document,
                extract_entities,
                classify_document,
                count_words,
            ],
            system_prompt=SYSTEM_PROMPT.format(
                documents_directory=DOCUMENTS_DIRECTORY,
                objective=_input.objective,
                language=_input.language,
                max_summary_length=_input.max_summary_length,
            ),
            subagents=[fact_checker],
            response_format=ProviderStrategy(schema=DocumentBrief),
            middleware=middleware,
            name="brief_documents",
        )

        files = build_document_files(
            [document.model_dump() for document in _input.documents]
        )

        # The parent agent's messages do not include the tokens spent by subagents and by the agents called as
        # tools, so the usage is collected with a callback across the whole run instead.
        with get_usage_metadata_callback() as usage_callback:
            agent_result = agent.invoke(
                cast(
                    InputAgentState,  # The deep agent state also accepts the `files` of the virtual filesystem.
                    {
                        "messages": [
                            HumanMessage(
                                HUMAN_PROMPT.format(
                                    document_count=len(files),
                                    documents_directory=DOCUMENTS_DIRECTORY,
                                )
                            )
                        ],
                        "files": files,
                    },
                ),
                config={"recursion_limit": 200},
            )

        parsed_output, _ = self.unpack_agent_result(agent_result)

        if not isinstance(parsed_output, DocumentBrief):
            raise ValueError(
                f"Wrong structured output type. Expected {DocumentBrief}. Got {type(parsed_output)} instead."
            )

        missing_paths = set(files) - {
            finding.path for finding in parsed_output.documents
        }

        if missing_paths:
            raise ValueError(
                f"The briefing is missing documents: {', '.join(sorted(missing_paths))}."
            )

        usage_metadata = (
            reduce(add_usage, usage_callback.usage_metadata.values())
            if usage_callback.usage_metadata
            else None
        )

        return AgentResponse(parsed_output.model_dump(), usage_metadata)
