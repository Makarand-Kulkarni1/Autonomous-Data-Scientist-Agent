import pickle
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from src.agents.coder import solve_step
from src.agents.planner import make_plan
from src.agents.reviewer import review_step
from src.profiler import quick_profile
from src.metrics_contract import validate_metrics, read_metric, METRICS_FILE

STATE_FILE = "pipeline_state.pkl"


SPLIT_RULES = (
    "\n\nDATA RULES: Split train/test exactly once (stratified, random_state=42) and save "
    "train.csv and test.csv including the target column. Fit imputers, encoders, scalers and "
    "feature selectors on train only. Never fit anything on test data. Reuse existing "
    "train.csv/test.csv from earlier steps instead of re-splitting. Whatever transformations "
    "you apply (scaling, selection, engineered features), apply the identical fitted "
    "transformations to the test set, and save the final model together with the name of the "
    "processed test CSV it expects. If the task text asks for a train/test split and "
    "train.csv/test.csv already exist, do NOT re-split: state that the existing split is "
    "being reused, and carve any validation set out of train.csv only."
)

EVAL_RULES = (
    f"\n\nMETRIC RULE: Compute ROC AUC with predict_proba on the held-out test set only. "
    f"Write {METRICS_FILE} in the working directory as "
    f'{{"roc_auc": <float>, "split": "test", "model": "<name>"}}. '
    f"The step is not complete until this file exists."
)

PERSIST_RULES = (
    "\n\nPERSISTENCE RULE: Do not retrain and do not re-split. Load the final model with joblib "
    "(see selected_best_model.txt or best_model.joblib) and confirm it loads. "
    "To sanity-check it, find the processed test CSV named in a *_expected_test.txt file, "
    "or otherwise a processed test CSV whose column count equals model.n_features_in_ "
    "(after dropping 'target'). Use getattr(model, 'feature_names_in_', None): if it exists, "
    "select those columns; if it is None, use the CSV columns in their existing order. "
    "NEVER raise an error because an attribute is missing or columns do not align. If no matching "
    "CSV is found, print a warning and skip the prediction check. "
    "Predict on a few rows if possible, then print the model file path and a short confirmation."
)

MODEL_RULES = (
    "\n\nMODEL RULE: Compare models using train data only; do not load test.csv in this step. "
    "A processed training file from an earlier feature-engineering step, typically named "
    "processed_train.csv, should exist in the working directory — you MUST load and train "
    "on that exact file, using every one of its feature columns (all columns except the "
    "target). Do NOT use the raw train.csv, and do NOT drop, reselect, or recompute any "
    "columns from the processed file. "
    "Put preprocessing (scaler, etc.) inside an sklearn Pipeline together with each classifier "
    "and compare with stratified cross-validation on the training data "
    "(cross_val_score or cross_validate, scoring='roc_auc'), so preprocessing is refit inside "
    "every fold. Never call fit or fit_transform on test data and never scale before "
    "cross-validation. Pick the best model by CV score, refit it on the full training data "
    "using the SAME feature columns as processed_train.csv, save it as best_model.joblib, "
    "write its name to selected_best_model.txt, and print the CV ROC AUC of every model."
)

FEATURE_RULES = (
    "\n\nFEATURE ENGINEERING RULE: This step's job is to produce DATA, not a model. Do not "
    "train, fit, or save any predictive model (e.g. XGBClassifier, RandomForestClassifier, "
    "LogisticRegression) in this step, and do not save any file ending in a model-like name "
    "such as *_model.joblib. Only save: the processed train/test CSVs (with the target column "
    "included), and any fitted preprocessing objects (scaler, encoder, selector) needed to "
    "reproduce the same transformation later. Model training and comparison happens in a "
    "separate, later step — leave it to that step."
)

REVIEW_CONTEXT = (
    "\n\nREVIEWER CONTEXT: The data was already split once, in an earlier step, into "
    "train.csv and test.csv (stratified 80/20, random_state=42). Reusing those files satisfies "
    "any split requirement in the task, so do not reject only because this step does not "
    "re-split. Do reject real leakage: fitting anything on test data, or preprocessing fitted "
    "outside the cross-validation folds."
)


def _is_eval_step(step) -> bool:
    return "evaluat" in step.title.lower()


@dataclass
class StepLog:
    """A record of what happened on one plan step, for the final report."""
    step_id: int
    title: str
    task: str
    coder_success: bool
    coder_attempts: int
    approved: bool
    review_reason: str
    leakage_flags: list
    code: str
    stdout: str
    stderr: str
    review_history: list = field(default_factory=list)


@dataclass
class RunResult:
    """Everything about one full pipeline run, from goal to finish (or failure)."""
    success: bool
    plan: object                 # the Plan object from the planner
    logs: list = field(default_factory=list)
    failed_at_step: int = None   # None if the whole plan completed
    metric: float = None         # ROC AUC read from metrics.json
    metric_step: int = None      # step id that produced it
    warnings: list = field(default_factory=list)  # non-fatal problems (e.g. persistence)


def _files_in(workspace: Path) -> list:
    """List files currently in the workspace, so we can tell the coder what already exists."""
    return sorted(p.name for p in workspace.iterdir() if p.is_file())


def _save_state(path: Path, plan, logs, metric, metric_step):
    """Checkpoint after every approved step so a crash can resume instead of restarting."""
    with open(path, "wb") as f:
        pickle.dump({"plan": plan, "logs": logs,
                     "metric": metric, "metric_step": metric_step}, f)


def run_pipeline(csv_path, goal: str, workspace: Path,
                 max_step_retries: int = 2, coder_attempts_per_try: int = 4,
                 verbose: bool = True, resume: bool = False) -> RunResult:
    """Run the full agent pipeline: plan the goal, then execute and review each step.

    resume=True: if an unfinished run's checkpoint exists in the workspace, reuse its
    plan and skip steps that were already approved.
    """
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    shutil.copy(csv_path, workspace / "data.csv")
    profile = quick_profile(workspace / "data.csv")

    state_path = workspace / STATE_FILE
    logs, metric, metric_step, done = [], None, None, set()
    warnings = []

    if resume and state_path.exists():
        with open(state_path, "rb") as f:
            state = pickle.load(f)
        plan = state["plan"]
        logs = state["logs"]
        metric, metric_step = state["metric"], state["metric_step"]
        done = {l.step_id for l in logs}
        if verbose:
            print(f"[orchestrator] Resuming: steps {sorted(done)} already done, reusing saved plan.")
    else:
        state_path.unlink(missing_ok=True)
        if verbose:
            print("[orchestrator] Planning...")
        plan = make_plan(goal, profile)

    if verbose:
        print(f"[orchestrator] Plan has {len(plan.steps)} steps, problem_type={plan.problem_type}")

    for step in plan.steps:
        if step.id in done:
            if verbose:
                print(f"\n[orchestrator] --- Step {step.id}: {step.title} --- (skipped, already done)")
            continue

        if verbose:
            print(f"\n[orchestrator] --- Step {step.id}: {step.title} ---")

        title = step.title.lower()
        is_eval = _is_eval_step(step)
        is_persist = "persist" in title or "save" in title
        is_train = ("train" in title or "compar" in title) and not is_eval and not is_persist
        is_feature = ("feature" in title or "selection" in title or "engineer" in title) and not is_train
        skip_split = is_persist or "explor" in title

        extra = "" if skip_split else SPLIT_RULES
        if is_eval:
            extra += EVAL_RULES
        if is_persist:
            extra += PERSIST_RULES
        if is_train:
            extra += MODEL_RULES
        if is_feature:
            extra += FEATURE_RULES

        existing_files = [f for f in _files_in(workspace) if f not in ("step.py", STATE_FILE)]
        base_task = (
            f"{step.task}\n\n"
            f"IMPORTANT: If you save any processed/transformed dataset to CSV, you MUST include "
            f"the target column (keep its original name) alongside the features — later steps "
            f"will need to load labels from these files, not just the feature values.\n\n"
            f"IMPORTANT: If earlier steps already saved a processed/engineered dataset (e.g. "
            f"processed_train.csv, processed_test.csv), you MUST build on that file, not on the "
            f"original raw data. Do not re-derive features from scratch or ignore prior feature "
            f"engineering — train and evaluate using the already-processed data and feature set."
            f"{extra}\n\n"
        )
        task_with_context = (
            base_task +
            f"Files already present in the working directory from earlier steps "
            f"(read/reuse them as needed, e.g. via pd.read_csv or joblib.load): {existing_files}"
        )

        review = None
        outcome = None
        review_history = []
        metric_error = None

        for retry in range(max_step_retries + 1):
            if is_eval:
                (workspace / METRICS_FILE).unlink(missing_ok=True)  # no stale file can pass

            outcome = solve_step(task_with_context, profile, workspace,
                                 max_attempts=coder_attempts_per_try, verbose=verbose)

            if not outcome.success:
                break

            if is_eval:
                ok, msg = validate_metrics(workspace)
                if not ok:
                    metric_error = msg
                    review = None
                    review_history.append({"attempt": retry + 1, "approved": False,
                                           "reason": f"Metric check failed: {msg}"})
                    if verbose:
                        print(f"[orchestrator] Metric check failed "
                              f"(retry {retry + 1}/{max_step_retries}): {msg}")
                    task_with_context = (
                        base_task +
                        f"Your previous attempt did not satisfy the metric rule: {msg}\n"
                        f"Write a corrected version.\n\n"
                        f"Files already present in the working directory: {existing_files}"
                    )
                    continue
                metric_error = None

            review = review_step(step.task + REVIEW_CONTEXT, outcome.code, outcome.result.stdout)
            review_history.append({
                "attempt": retry + 1,
                "approved": review.approved,
                "reason": review.reason,
            })

            if review.approved:
                break

            if verbose:
                print(f"[orchestrator] Reviewer rejected (retry {retry + 1}/{max_step_retries}): "
                      f"{review.reason}")
            task_with_context = (
                base_task +
                f"Your previous attempt was rejected by a reviewer for this reason: {review.reason}\n"
                f"Leakage/quality flags: {review.leakage_flags}\n"
                f"Write a corrected version that avoids this problem.\n\n"
                f"Files already present in the working directory: {existing_files}"
            )

        log = StepLog(
            step_id=step.id, title=step.title, task=step.task,
            coder_success=outcome.success, coder_attempts=outcome.attempts,
            approved=bool(review and review.approved),
            review_reason=(review.reason if review
                           else (f"Metric check failed: {metric_error}" if metric_error
                                 else "Coder never produced working code.")),
            leakage_flags=review.leakage_flags if review else [],
            code=outcome.code, stdout=outcome.result.stdout,
            stderr=outcome.result.stderr,
            review_history=review_history,
        )

        if not log.approved:
            logs.append(log)
            # Persistence is the last step and the metric is already measured by then,
            # so a failure here is recorded as a warning instead of failing the run.
            if is_persist and metric is not None:
                warnings.append(f"Step {step.id} ({step.title}) failed: {log.review_reason}")
                if verbose:
                    print(f"[orchestrator] Step {step.id} failed but the metric is already "
                          f"measured; continuing with a warning.")
                continue
            if verbose:
                print(f"[orchestrator] Step {step.id} FAILED. Stopping pipeline "
                      f"(rerun with resume=True to continue from here).")
            return RunResult(success=False, plan=plan, logs=logs, failed_at_step=step.id,
                             metric=metric, metric_step=metric_step, warnings=warnings)

        logs.append(log)
        if is_eval:
            metric, metric_step = read_metric(workspace), step.id

        _save_state(state_path, plan, logs, metric, metric_step)

    state_path.unlink(missing_ok=True)   # finished: next run starts fresh
    if verbose:
        print("\n[orchestrator] All steps completed" +
              (" (with warnings)." if warnings else " and approved."))
    return RunResult(success=True, plan=plan, logs=logs,
                     metric=metric, metric_step=metric_step, warnings=warnings)
