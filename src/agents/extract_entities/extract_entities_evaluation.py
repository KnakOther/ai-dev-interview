from langsmith.schemas import Run

from src.agents.extract_entities.extract_entities_agent import ExtractEntitiesAgent
from src.datasets.documents_with_entities import DocumentsWithEntitiesDataset
from src.shared.evaluation import Evaluation, EvaluationError


class ExtractEntitiesDefaultSettingsEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Extract entities with default settings evaluation"
        self.agents = [ExtractEntitiesAgent()]
        self.datasets = [DocumentsWithEntitiesDataset()]
        self.evaluators = [
            did_not_raise_error_evaluator,
            has_valid_structure_evaluator,
            found_expected_entities_evaluator,
        ]


class ExtractEntitiesLimitedTypesEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Extract entities with limited types evaluation"
        self.agents = [ExtractEntitiesAgent()]
        self.datasets = [DocumentsWithEntitiesDataset()]
        self.additional_inputs = {"entity_types": ["people", "organizations"]}
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
    required_fields = ["people", "organizations", "locations", "dates", "key_terms"]

    all_fields_present = all(field in output for field in required_fields)
    all_fields_are_lists = all(
        isinstance(output.get(field), list) for field in required_fields
    )

    score = 1 if (all_fields_present and all_fields_are_lists) else 0
    return {"key": "has valid structure", "score": score}


def found_expected_entities_evaluator(
    run: Run, outputs: dict, reference_outputs: dict
) -> dict:
    if not outputs:
        raise EvaluationError("The evaluation run did not provide a model output.")

    if run.error:
        return {"key": "found expected entities", "score": 0}

    # Check if expected entities (from metadata) were found
    expected = reference_outputs.get("expected_entities", {})
    if not expected:
        # No expected entities defined, skip this evaluator
        return {"key": "found expected entities", "score": 1}

    output = outputs.get("output", {})
    found_count = 0
    total_count = 0

    for entity_type, expected_list in expected.items():
        actual_list = output.get(entity_type, [])
        for expected_entity in expected_list:
            total_count += 1
            # Case-insensitive partial match
            if any(expected_entity.lower() in actual.lower() for actual in actual_list):
                found_count += 1

    score = found_count / total_count if total_count > 0 else 1
    return {"key": "found expected entities", "score": score}


if __name__ == "__main__":
    ExtractEntitiesDefaultSettingsEvaluation().run()
    ExtractEntitiesLimitedTypesEvaluation().run()
