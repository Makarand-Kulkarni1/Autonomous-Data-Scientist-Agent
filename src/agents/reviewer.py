import re

from pydantic import BaseModel, Field

from src.llm import chat


class ReviewResult(BaseModel):
    """The reviewer's verdict on one completed step."""
    approved: bool
    reason: str = Field(description="One or two sentences explaining the verdict")
    leakage_flags: list[str] = Field(default_factory=list)


# Patterns that are near-certain signs of data leakage in scikit-learn style code.
# Each is (compiled regex, human-readable explanation).
LEAKAGE_PATTERNS = [
    (re.compile(r"\.fit\w*\(\s*X\s*(?:,|\))"), "Called fit on the full X before splitting, instead of X_train"),
    (re.compile(r"scaler\.fit\w*\(\s*X_test"), "Fit a scaler on X_test (should only ever call .transform on test data)"),
    (re.compile(r"\.fit\w*\(\s*X_test"), "Called fit (or fit_transform) using X_test"),
]


def check_leakage_patterns(code: str) -> list[str]:
    """Cheap, deterministic scan for the most common leakage mistakes.
    Returns a list of human-readable warnings (empty list = nothing found).
    """
    # A final .fit(X, y) after proper cross-validation is standard practice,
    # not leakage -- cross_val_score already evaluated the model honestly.
    used_cross_validation = bool(re.search(r"cross_val_score\(|cross_validate\(", code))

    flags = []
    for pattern, msg in LEAKAGE_PATTERNS:
        if used_cross_validation and "before splitting" in msg:
            continue   # skip only this specific pattern when CV was used
        if pattern.search(code):
            flags.append(msg)
    return flags


SYSTEM_PROMPT = """You are a strict senior data scientist reviewing a junior colleague's completed step.

You will see: the task they were given, the code they wrote, and what the code printed when run.

Check for:
- Data leakage (fitting scalers/encoders/models on data that includes the test set, or using the
  target column as a feature, or any information from the future/test leaking into training).
- Whether the output actually satisfies what the task asked for.
- Suspicious results (e.g. accuracy or ROC-AUC extremely close to 1.0 on a non-trivial task is often
  a sign of leakage, not a good model).

Reply with ONLY a JSON object with exactly these keys:
- "approved": true or false
- "reason": one or two sentences explaining your verdict
- "leakage_flags": a list of short strings describing any leakage concerns (empty list if none)
"""


def _extract_json_obj(reply: str) -> str:
    reply = reply.strip().strip("`")
    if reply.lower().startswith("json"):
        reply = reply[4:]
    start, end = reply.find("{"), reply.rfind("}")
    return reply[start:end + 1] if start != -1 and end != -1 else reply


def review_step(task: str, code: str, stdout: str) -> ReviewResult:
    """Judge one completed step. Combines a fast pattern check with an LLM judgment."""
    import json

    pattern_flags = check_leakage_patterns(code)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"TASK:\n{task}\n\nCODE:\n{code}\n\nOUTPUT:\n{stdout}\n\n"
            f"Automated pattern scan already found these potential issues (may be empty): {pattern_flags}"
        )},
    ]
    reply = chat(messages)

    try:
        data = json.loads(_extract_json_obj(reply))
        result = ReviewResult(**data)
    except Exception:
        # If the judge's reply is unusable, fail safe: don't silently approve.
        result = ReviewResult(approved=False, reason="Reviewer reply could not be parsed.",
                              leakage_flags=pattern_flags)

    # Merge in anything the pattern scan found that the LLM might have missed
    for flag in pattern_flags:
        if flag not in result.leakage_flags:
            result.leakage_flags.append(flag)
    if pattern_flags:
        result.approved = False   # a hard pattern match always overrides the LLM's opinion

    return result
