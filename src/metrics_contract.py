import json
from pathlib import Path

METRICS_FILE = "metrics.json"

def validate_metrics(workspace):
    path = Path(workspace) / METRICS_FILE
    if not path.exists():
        return False, f"{METRICS_FILE} missing. Write it with keys roc_auc, split, model."
    try:
        m = json.loads(path.read_text())
    except Exception as e:
        return False, f"{METRICS_FILE} is not valid JSON: {e}"
    auc = m.get("roc_auc")
    if not isinstance(auc, (int, float)) or not (0.0 <= auc <= 1.0):
        return False, f"invalid roc_auc: {auc}"
    if m.get("split") != "test":
        return False, "metric must be computed on the held-out test split"
    return True, ""

def read_metric(workspace):
    return json.loads((Path(workspace) / METRICS_FILE).read_text())["roc_auc"]