from langsmith.schemas import Run

from src.agents.classify_document.classify_document_agent import (
    ClassifyDocumentAgent,
)
from src.datasets.documents_for_classification import DocumentsForClassificationDataset
from src.shared.evaluation import Evaluation, EvaluationError


class ClassifyDocumentDefaultSettingsEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Classify document with default settings evaluation"
        self.agents = [ClassifyDocumentAgent()]
        self.datasets = [DocumentsForClassificationDataset()]
        self.evaluators = [
            did_not_raise_error_evaluator,
            has_valid_structure_evaluator,
            matches_expected_category_evaluator,
        ]


class ClassifyDocumentWithCustomCategoriesEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Classify document with custom categories evaluation"
        self.agents = [ClassifyDocumentAgent()]
        self.datasets = [DocumentsForClassificationDataset()]
        self.additional_inputs = {
            "custom_categories": ["Technology", "Business", "Science", "Finance"]
        }
        self.evaluators = [did_not_raise_error_evaluator, has_valid_structure_evaluator]


def did_not_raise_error_evaluator(run: Run, outputs: dict) -> dict:
    if not outputs:
        raise EvaluationError("The evaluation run did not provide a model output.")

    score = 0 if run.error else 1
    return {"key": "did not raise error", "score": score}


def has_valid_structure_evaluator(run: Run, outputs: dict) -> dict:
    if not outputs:
        raise EvaluationError("The evaluation run did not provide a model output.")

    if run.error:
        return {"key": "has valid structure", "score": 0}

    output = outputs.get("output", {})

    has_primary = (
        "primary_category" in output and len(output.get("primary_category", "")) > 0
    )
    has_secondary = "secondary_categories" in output and isinstance(
        output.get("secondary_categories"), list
    )
    has_confidence = (
        "confidence_score" in output
        and isinstance(output.get("confidence_score"), (int, float))
        and 0.0 <= output.get("confidence_score", -1) <= 1.0
    )
    has_sentiment = "sentiment" in output and output.get("sentiment") in [
        "positive",
        "negative",
        "neutral",
        "mixed",
    ]
    has_doc_type = (
        "document_type" in output and len(output.get("document_type", "")) > 0
    )

    score = (
        1
        if (
            has_primary
            and has_secondary
            and has_confidence
            and has_sentiment
            and has_doc_type
        )
        else 0
    )
    return {"key": "has valid structure", "score": score}


def matches_expected_category_evaluator(
    run: Run, outputs: dict, reference_outputs: dict
) -> dict:
    if not outputs:
        raise EvaluationError("The evaluation run did not provide a model output.")

    if run.error:
        return {"key": "matches expected category", "score": 0}

    # Check if the classification matches expected category (from metadata)
    expected_category = reference_outputs.get("expected_category", "")
    if not expected_category:
        # No expected category defined, skip this evaluator
        return {"key": "matches expected category", "score": 1}

    output = outputs.get("output", {})
    primary_category = output.get("primary_category", "")
    secondary_categories = output.get("secondary_categories", [])

    # Check if expected category is in primary or secondary
    expected_lower = expected_category.lower()
    primary_match = expected_lower in primary_category.lower()
    secondary_match = any(expected_lower in cat.lower() for cat in secondary_categories)

    score = 1 if (primary_match or secondary_match) else 0
    return {"key": "matches expected category", "score": score}


if __name__ == "__main__":
    ClassifyDocumentDefaultSettingsEvaluation().run()
    ClassifyDocumentWithCustomCategoriesEvaluation().run()
