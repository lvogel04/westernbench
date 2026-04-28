"""Inspect AI task for the WesternBench dataset."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.solver import generate

from westernbench.dataset import DEFAULT_DATASET_PATH, load_western_bias_dataset
from westernbench.scoring import western_defaulting_scorer


@task
def western_bias(
    dataset_path: str = str(DEFAULT_DATASET_PATH),
    limit: int | None = None,
    judge_model: str | None = None,
) -> Task:
    """Evaluate Western-defaulting behavior on open-ended prompts.

    Args:
        dataset_path: Path to the WesternBench JSON dataset.
        limit: Optional number of samples to run, useful for smoke tests.
        judge_model: Optional Inspect model name used for grading. If omitted,
            Inspect uses the active evaluation model for grading.
    """
    return Task(
        dataset=load_western_bias_dataset(dataset_path, limit=limit),
        solver=[generate()],
        scorer=western_defaulting_scorer(model=judge_model),
    )
