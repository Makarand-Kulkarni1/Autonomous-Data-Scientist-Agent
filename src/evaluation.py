from dataclasses import dataclass
from pathlib import Path


@dataclass
class Benchmark:
    """One dataset to evaluate the pipeline against."""
    name: str
    csv_path: Path
    goal: str
    baseline_metric_name: str
    baseline_value: float
    higher_is_better: bool = True


def get_agent_metric(result):
    """Use the orchestrator's own validated metric instead of re-parsing stdout."""
    source = f"step {result.metric_step}" if result.metric_step else None
    return result.metric, source


def run_evaluation_suite(benchmarks: list, run_pipeline_fn, runs_dir, verbose: bool = True):
    """Run the pipeline against every benchmark and collect results for comparison."""
    import pickle

    eval_results = []

    for bench in benchmarks:
        if verbose:
            print(f"\n{'=' * 60}\nEvaluating: {bench.name}\n{'=' * 60}")

        workspace = Path(runs_dir) / "eval" / bench.name.split(" (")[0].replace(" ", "_").lower() / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        try:
            result = run_pipeline_fn(csv_path=bench.csv_path, goal=bench.goal,
                                     workspace=workspace, verbose=verbose)
        except Exception as e:
            eval_results.append({
                "benchmark": bench.name, "pipeline_success": False,
                "agent_metric": None, "metric_source_step": None,
                "baseline_metric_name": bench.baseline_metric_name,
                "baseline_value": bench.baseline_value, "error": str(e),
            })
            continue

        with open(workspace / "run_result.pkl", "wb") as f:
            pickle.dump(result, f)

        if not result.success:
            eval_results.append({
                "benchmark": bench.name, "pipeline_success": False,
                "agent_metric": result.metric, "metric_source_step": result.metric_step,
                "baseline_metric_name": bench.baseline_metric_name,
                "baseline_value": bench.baseline_value,
                "error": f"Failed at step {result.failed_at_step}",
            })
            continue

        agent_value, source_step = get_agent_metric(result)
        eval_results.append({
            "benchmark": bench.name, "pipeline_success": True,
            "agent_metric": agent_value, "metric_source_step": source_step,
            "baseline_metric_name": bench.baseline_metric_name,
            "baseline_value": bench.baseline_value, "error": None,
        })

    return eval_results
