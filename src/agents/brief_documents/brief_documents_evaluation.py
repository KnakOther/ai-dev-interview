from langsmith.schemas import Run
from pydantic import ValidationError

from src.agents.brief_documents.brief_documents_agent import (
    BriefDocumentsAgent,
    DocumentBrief,
)
from src.agents.brief_documents.brief_documents_tools import build_document_files
from src.datasets.document_collections_for_briefing import (
    DocumentCollectionsForBriefingDataset,
)
from src.shared.evaluation import Evaluation


class BriefDocumentsDefaultSettingsEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Brief documents with default settings evaluation"
        self.agents = [BriefDocumentsAgent()]
        self.datasets = [DocumentCollectionsForBriefingDataset()]
        self.evaluators = [
            did_not_raise_error_evaluator,
            has_valid_structure_evaluator,
            covers_all_documents_evaluator,
            total_tokens_evaluator,
        ]


def did_not_raise_error_evaluator(run: Run) -> dict:
    return {"key": "did not raise error", "score": 0 if run.error else 1}


def has_valid_structure_evaluator(outputs: dict) -> dict:
    try:
        DocumentBrief.model_validate(outputs.get("result"))
        score = 1
    except ValidationError:
        score = 0

    return {"key": "has valid structure", "score": score}


def covers_all_documents_evaluator(inputs: dict, outputs: dict) -> dict:
    """Ratio of the input documents that appear in the brief."""
    expected_paths = set(build_document_files(inputs["input"]["documents"]))
    result = outputs.get("result") or {}
    found_paths = {finding.get("path") for finding in result.get("documents", [])}

    return {
        "key": "covers all documents",
        "score": len(expected_paths & found_paths) / len(expected_paths),
    }


def total_tokens_evaluator(outputs: dict) -> dict:
    """Tracks the cost of the run. Deep agents can loop; a regression here is as important as a quality one."""
    execution_details = outputs.get("execution_details") or {}
    return {"key": "total tokens", "score": execution_details.get("total_tokens", 0)}


if __name__ == "__main__":
    BriefDocumentsDefaultSettingsEvaluation().run()
