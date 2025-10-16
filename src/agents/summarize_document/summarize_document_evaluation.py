from langsmith.schemas import Example, Run

from src.agents.summarize_document.summarize_document_agent import (
    SummarizeDocumentAgent,
)
from src.datasets.documents_for_summarization import DocumentsForSummarizationDataset
from src.shared.evaluation import Evaluation, EvaluationError


class SummarizeDocumentDefaultSettingsEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Summarize document with default settings evaluation"
        self.agents = [SummarizeDocumentAgent()]
        self.datasets = [DocumentsForSummarizationDataset()]
        self.evaluators = [
            did_not_raise_error_evaluator,
            has_valid_structure_evaluator,
            summary_length_evaluator,
        ]


class SummarizeDocumentShortSummaryEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Summarize document with short summary (100 words) evaluation"
        self.agents = [SummarizeDocumentAgent()]
        self.datasets = [DocumentsForSummarizationDataset()]
        self.additional_inputs = {"max_length": 100}
        self.evaluators = [
            did_not_raise_error_evaluator,
            has_valid_structure_evaluator,
        ]


class SummarizeDocumentFrenchLanguageEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Summarize document in French evaluation"
        self.agents = [SummarizeDocumentAgent()]
        self.datasets = [DocumentsForSummarizationDataset()]
        self.additional_inputs = {"language": "French"}
        self.evaluators = [did_not_raise_error_evaluator, has_valid_structure_evaluator]


def did_not_raise_error_evaluator(run: Run, example: Example) -> dict:
    if not run.outputs:
        raise EvaluationError("The evaluation run did not provide a model output.")

    score = 0 if run.error else 1
    return {"key": "did not raise error", "score": score}


def has_valid_structure_evaluator(run: Run, example: Example) -> dict:
    if not run.outputs:
        raise EvaluationError("The evaluation run did not provide a model output.")

    if run.error:
        return {"key": "has valid structure", "score": 0}

    output = run.outputs.get("output", {})
    has_summary = "summary" in output and len(output.get("summary", "")) > 0
    has_key_points = "key_points" in output and isinstance(output.get("key_points"), list)
    has_word_count = "word_count" in output and isinstance(output.get("word_count"), int)

    valid_key_points = (
        len(output.get("key_points", [])) >= 3 and len(output.get("key_points", [])) <= 5
    )

    score = 1 if (has_summary and has_key_points and has_word_count and valid_key_points) else 0
    return {"key": "has valid structure", "score": score}


def summary_length_evaluator(run: Run, example: Example) -> dict:
    if not run.outputs:
        raise EvaluationError("The evaluation run did not provide a model output.")

    if run.error:
        return {"key": "summary length appropriate", "score": 0}

    output = run.outputs.get("output", {})
    summary = output.get("summary", "")
    max_length = example.inputs.get("max_length", 200)

    # Check if summary is roughly within max_length (allow 20% tolerance)
    word_count = len(summary.split())
    score = 1 if word_count <= max_length * 1.2 else 0

    return {"key": "summary length appropriate", "score": score}


if __name__ == "__main__":
    SummarizeDocumentDefaultSettingsEvaluation().run()
    SummarizeDocumentShortSummaryEvaluation().run()
    SummarizeDocumentFrenchLanguageEvaluation().run()
