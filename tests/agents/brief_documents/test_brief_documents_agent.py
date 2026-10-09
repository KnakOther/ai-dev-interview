import json
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from src.agents.brief_documents import brief_documents_agent
from src.agents.brief_documents.brief_documents_agent import (
    BriefDocumentsAgent,
    BriefDocumentsAgentInput,
)
from src.agents.brief_documents.brief_documents_tools import (
    DocumentNotFoundError,
    build_document_files,
    read_document,
)
from src.shared.agent import AgentResponse

USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

DOCUMENTS = [
    {"title": "Q3 Report", "text": "Revenue grew by 12% in Q3."},
    {"title": "Q3 report", "text": "Revenue fell by 3% in Q3."},
]


class ScriptedChatModel(GenericFakeChatModel):
    """Fake chat model returning a scripted sequence of messages, and recording the messages it receives."""

    received: list[list] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(self, messages: list, *args: Any, **kwargs: Any) -> Any:
        self.received.append(messages)
        return super()._generate(messages, *args, **kwargs)


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
        usage_metadata=USAGE,
        response_metadata={"model_name": "scripted-model"},
    )


def final_brief(paths: list[str]) -> AIMessage:
    """Final answer in the native structured output format: the JSON document is the message content."""
    return AIMessage(
        content=json.dumps(
            {
                "executive_summary": "Q3 revenue figures contradict each other.",
                "documents": [
                    {
                        "path": path,
                        "summary": "Q3 revenue.",
                        "primary_category": "Finance",
                    }
                    for path in paths
                ],
                "contradictions": ["Revenue grew by 12% vs. fell by 3%."],
            }
        ),
        usage_metadata=USAGE,
        response_metadata={"model_name": "scripted-model"},
    )


@pytest.fixture
def run_with_script(mocker):
    """Runs the deep agent with a scripted main model. The inner summarize agent is mocked."""
    mocker.patch(
        "src.agents.summarize_document.summarize_document_agent.SummarizeDocumentAgent.generate_response",
        lambda self, input, *args, **kwargs: AgentResponse(
            {"summary": f"Summary of: {input['document_text'][:20]}"}, None
        ),
    )

    def _run(script: list[AIMessage]) -> tuple[AgentResponse, ScriptedChatModel]:
        main_model = ScriptedChatModel(messages=iter(script))
        main_model.received = []
        subagent_model = ScriptedChatModel(messages=iter([]))
        models = {
            brief_documents_agent.GEMINI_3_1_PRO: main_model,
            brief_documents_agent.GEMINI_3_5_FLASH: subagent_model,
        }
        mocker.patch.object(
            brief_documents_agent,
            "get_chat_model",
            lambda model, **kwargs: models[model],
        )
        return BriefDocumentsAgent().generate_response(
            {"documents": DOCUMENTS}
        ), main_model

    return _run


def test__build_document_files__deduplicates_paths():
    files = build_document_files(DOCUMENTS)

    assert list(files) == ["/documents/q3-report.md", "/documents/q3-report-2.md"]
    assert read_document(files, "/documents/q3-report.md") == (
        "# Q3 Report\n\nRevenue grew by 12% in Q3."
    )


def test__read_document__raises_on_unknown_path():
    with pytest.raises(DocumentNotFoundError, match="/documents/q3-report.md"):
        read_document(build_document_files(DOCUMENTS[:1]), "/documents/unknown.md")


@pytest.mark.parametrize(
    "input",
    [
        {"documents": []},
        {"documents": [{"title": "", "text": "Some text"}]},
        {"documents": DOCUMENTS, "max_summary_length": 0},
    ],
)
def test__input_schema__rejects_invalid_input(input):
    with pytest.raises(ValidationError):
        BriefDocumentsAgentInput(**input)


def test__generate_response__returns_brief_and_total_usage(run_with_script):
    response, main_model = run_with_script(
        [
            tool_call(
                "write_todos",
                {
                    "todos": [
                        {"content": "Summarize documents", "status": "in_progress"}
                    ]
                },
                "1",
            ),
            tool_call("summarize_document", {"path": "/documents/q3-report.md"}, "2"),
            final_brief(["/documents/q3-report.md", "/documents/q3-report-2.md"]),
        ]
    )

    assert response.result["contradictions"] == ["Revenue grew by 12% vs. fell by 3%."]
    assert response.execution_details == {
        "input_tokens": 30,
        "output_tokens": 15,
        "total_tokens": 45,
    }

    # The custom tool read the document from the virtual filesystem.
    assert "Summary of: # Q3 Report" in main_model.received[2][-1].content


def test__generate_response__unknown_path_is_sent_back_to_the_model(run_with_script):
    _, main_model = run_with_script(
        [
            tool_call("summarize_document", {"path": "/documents/unknown.md"}, "1"),
            final_brief(["/documents/q3-report.md", "/documents/q3-report-2.md"]),
        ]
    )

    assert main_model.received[1][-1].content.startswith("DocumentNotFoundError")


def test__generate_response__raises_when_a_document_is_missing_from_the_brief(
    run_with_script,
):
    with pytest.raises(ValueError, match="q3-report-2.md"):
        run_with_script([final_brief(["/documents/q3-report.md"])])
