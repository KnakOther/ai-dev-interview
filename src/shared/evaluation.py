from abc import ABC
from collections.abc import Callable

from langsmith import Client as LangsmithClient
from langsmith import evaluate as langsmith_evaluate

from src.shared.agent import Agent
from src.shared.dataset import Dataset
from src.shared.utils import get_langsmith_api_key, runs_in_debug_mode


class Evaluation(ABC):
    """Abstract base class for LagSmith evaluation runs.

    This class defines the structure for creating and running evaluations with LangSmith.
    Each evaluation has a name, a list of agents, a list of datasets, and a list of evaluators.
    The evaluation run will generate one Langsmith evaluation by agent and dataset combination.

    Developing a new evaluation requires creating a subclass of this class and
    defining it's attributes. The `evaluators` attribute is a list of functions that receive any of the
    `inputs`, `outputs`, `reference_outputs`, `run` and `example` arguments (matched by name) and return a
    dictionary with at least a `key` and a `score` key with a numerical value.

    Here is an example evaluator:
    ```
    def length_evaluator(outputs: dict) -> dict:
        score = 0 if len(outputs["llm_output"]) < 50 else 1.
        return {"key": "length", "score": score}
    ```

    Refer to LangSmith's documentation for more information on how to create evaluators.


    The `run` method is the main entry point for running the evaluation.
    """

    name: str
    agents: list[Agent]
    datasets: list[Dataset]
    langsmith_api_key: str | None = None
    evaluators: list[Callable] | None = None
    additional_arguments: dict = {}
    additional_inputs: dict = {}
    metadata: dict = {"version": "1.0.0", "revision_id": "beta"}
    INPUT_KEY = "input"
    num_repetitions: int = 1

    def run(self):
        if not self.langsmith_api_key:
            self.langsmith_api_key = get_langsmith_api_key()

        reps = int(self.num_repetitions) if self.num_repetitions else 1
        for dataset in self.datasets:
            for agent in self.agents:
                langsmith_evaluate(
                    self._make_evaluation_function(agent),
                    evaluators=self.evaluators,
                    data=dataset.name,
                    experiment_prefix=f"{self.name} - {agent.name}",  # The name of the experiment
                    metadata=self.metadata,
                    client=LangsmithClient(api_key=self.langsmith_api_key),
                    max_concurrency=1 if runs_in_debug_mode() else None,
                    num_repetitions=reps,
                )

    def _make_evaluation_function(self, agent: Agent) -> Callable:
        """Creates a function that LangSmith will use to evaluate an agent's performance.

        This method generates a callable that LangSmith uses to run evaluations. The function
        accepts input data from a LangSmith dataset and executes the agent with that data.
        """

        def _evaluation_function(input: dict) -> dict:
            _additional_arguments = self.additional_arguments.copy()
            _additional_inputs = self.additional_inputs.copy()
            _input = input.copy()
            _input.update(_additional_inputs)

            return agent.run(
                _input[self.INPUT_KEY],
                **_additional_arguments,
            )

        return _evaluation_function


class EvaluationError(Exception):
    """Exception raised for errors that occur during the evaluation process."""
