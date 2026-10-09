import re
from typing import Any

from deepagents.backends.protocol import FileData
from deepagents.backends.utils import create_file_data
from langchain.agents.middleware import ToolCallRequest
from langchain.tools import ToolRuntime, tool

from src.agents.classify_document.classify_document_agent import ClassifyDocumentAgent
from src.agents.extract_entities.extract_entities_agent import ExtractEntitiesAgent
from src.agents.summarize_document.summarize_document_agent import (
    SummarizeDocumentAgent,
)

DOCUMENTS_DIRECTORY = "/documents/"


class DocumentNotFoundError(Exception):
    """Raised when a tool receives a path that is not in the virtual filesystem. Its message is sent back to the model."""


def build_document_files(documents: list[dict[str, str]]) -> dict[str, FileData]:
    """Converts a list of `{"title", "text"}` documents into deep agent virtual files.

    Each document is stored as `/documents/<slugified-title>.md`. Duplicated titles get a numeric suffix.
    """
    files: dict[str, FileData] = {}

    for document in documents:
        slug = slugify(document["title"]) or "document"
        path = f"{DOCUMENTS_DIRECTORY}{slug}.md"
        suffix = 2

        while path in files:
            path = f"{DOCUMENTS_DIRECTORY}{slug}-{suffix}.md"
            suffix += 1

        files[path] = create_file_data(f"# {document['title']}\n\n{document['text']}")

    return files


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def read_document(files: dict[str, Any], path: str) -> str:
    """Reads a document from the deep agent virtual filesystem."""
    if path not in files:
        available = ", ".join(sorted(files)) or "none"
        raise DocumentNotFoundError(
            f"The file '{path}' does not exist. Available files: {available}."
        )

    content = files[path]["content"]

    return "\n".join(content) if isinstance(content, list) else content


@tool
def summarize_document(path: str, runtime: ToolRuntime, max_length: int = 150) -> dict:
    """Summarize one document of the filesystem and extract its key points.

    Args:
        path: Absolute path of the document, e.g. /documents/q3-report.md.
        max_length: Maximum length of the summary, in words.
    """
    document_text = read_document(runtime.state.get("files", {}), path)
    return SummarizeDocumentAgent().run(
        {"document_text": document_text, "max_length": max_length}
    )["result"]


@tool
def extract_entities(path: str, runtime: ToolRuntime) -> dict:
    """Extract the people, organizations, locations, dates and key terms mentioned in one document.

    Args:
        path: Absolute path of the document, e.g. /documents/q3-report.md.
    """
    document_text = read_document(runtime.state.get("files", {}), path)
    return ExtractEntitiesAgent().run({"document_text": document_text})["result"]


@tool
def classify_document(path: str, runtime: ToolRuntime) -> dict:
    """Classify one document by category, document type and sentiment.

    Args:
        path: Absolute path of the document, e.g. /documents/q3-report.md.
    """
    document_text = read_document(runtime.state.get("files", {}), path)
    return ClassifyDocumentAgent().run({"document_text": document_text})["result"]


def document_not_found_error_handler(
    exception: Exception, request: ToolCallRequest
) -> str | None:
    """Sends DocumentNotFoundError back to the model so it can retry with a valid path. Other errors propagate."""
    if isinstance(exception, DocumentNotFoundError):
        return f"{type(exception).__name__}: {exception}"

    return None


@tool
def count_words(text: str) -> int:
    """Count the exact number of words of a text. Use it to check length constraints instead of estimating."""
    return len(text.split())
