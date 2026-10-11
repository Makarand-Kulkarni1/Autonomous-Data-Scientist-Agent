import re
from pathlib import Path

import mlflow

from src.orchestrator import RunResult

# Matches lines like "Test accuracy: 0.9737" or "ROC-AUC: 0.99" and captures name + number
METRIC_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9 _\-]{2,40}?):\s*([0-9]*\.?[0-9]+)\s*$", re.MULTILINE)


def extract_metrics(stdout: str) -> dict:
    """Pull simple 'Name: number' lines out of a step's printed output.

    This is a heuristic, not a guarantee — it only catches metrics the coder
    happened to print in a recognizable format. Good enough for tracking trends
    across runs without requiring the coder to follow a strict output contract.
    """
    metrics = {}
    for name, value in METRIC_LINE.findall(stdout):
        # MLflow metric keys can't contain spaces or most punctuation
        key = re.sub(r"[^A-Za-z0-9_]", "_", name.strip().lower())
        try:
            metrics[key] = float(value)
        except ValueError:
            pass
    return metrics


def log_run_to_mlflow(result: RunResult, workspace: Path, goal: str,
                      experiment_name: str = "auto-data-scientist"):
    """Log one full pipeline run as an MLflow run: params, metrics, and artifacts."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_param("goal", goal)
        mlflow.log_param("problem_type", result.plan.problem_type)
        mlflow.log_param("target_column", result.plan.target_column)
        mlflow.log_param("num_steps", len(result.plan.steps))
        mlflow.log_param("success", result.success)
        if not result.success:
            mlflow.log_param("failed_at_step", result.failed_at_step)

        # Log each step's own printed metrics, tagged with the step number
        # so you can tell "step 4's accuracy" apart from "step 5's accuracy"
        for log in result.logs:
            mlflow.log_param(f"step{log.step_id}_title", log.title)
            mlflow.log_metric(f"step{log.step_id}_coder_attempts", log.coder_attempts)
            mlflow.log_metric(f"step{log.step_id}_approved", int(log.approved))
            for key, value in extract_metrics(log.stdout).items():
                mlflow.log_metric(f"step{log.step_id}_{key}", value)

        # Log every file the pipeline produced as an artifact (charts, models, csvs)
        mlflow.log_artifacts(str(workspace))

        run_id = mlflow.active_run().info.run_id
    return run_id
