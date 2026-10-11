import re
from dataclasses import dataclass, field
from pathlib import Path

from src.llm import chat
from src.sandbox import run_in_sandbox, SandboxResult

# Three backticks, built this way so they can't be mistaken for Markdown formatting
FENCE = "`" * 3

# The "system" message: standing rules the LLM must follow on every call
SYSTEM_PROMPT = f"""You are an expert data scientist. You write Python code that is executed automatically in a sandbox.

ENVIRONMENT RULES
- There is no internet access. Never try to download anything.
- The dataset is the file 'data.csv' in the current directory.
- Available libraries: pandas (3.x), numpy, scikit-learn, xgboost, matplotlib, seaborn, joblib.
- Only what you print() is returned to you, so print key results (metrics, shapes, etc.).
- For charts: call matplotlib.use("Agg") before importing pyplot, save with plt.savefig("name.png"), and never call plt.show().
- Save any files (charts, models) in the current directory.
- pandas 3.x: never use chained assignment or inplace=True on a column. Write df["col"] = df["col"].fillna(value).
- scikit-learn: use OneHotEncoder(..., sparse_output=True) or sparse_output=False -- NOT sparse=True/False, which was removed in recent scikit-learn versions and raises a TypeError.

OUTPUT FORMAT
- Reply with exactly ONE complete Python script inside a single code block that starts with {FENCE}python and ends with {FENCE}.
- Do not write explanations outside the code block. Use short comments inside the code instead.
"""

# Pattern that finds code blocks in the LLM's reply
CODE_BLOCK = re.compile(FENCE + r"(?:python|py)?[ \t]*\n(.*?)" + FENCE, re.DOTALL)


def extract_code(reply: str) -> str:
    """Pull the Python code out of the LLM's reply.

    LLMs wrap code in a fenced block and sometimes add chatty text around it.
    If there are several blocks we take the longest one (the real script).
    If there are none, we assume the whole reply is code.
    """
    blocks = CODE_BLOCK.findall(reply)
    if blocks:
        return max(blocks, key=len).strip()
    return reply.strip()


@dataclass
class StepOutcome:
    """What the agent hands back after working on one task."""
    success: bool
    attempts: int
    code: str
    result: SandboxResult
    history: list = field(default_factory=list)   # every attempt, for debugging


def solve_step(task: str, data_profile: str, workspace: Path,
               max_attempts: int = 4, verbose: bool = True) -> StepOutcome:
    """Write code for `task`, run it in the sandbox, and repair it if it fails."""
    base = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"TASK:\n{task}\n\nDATA PROFILE (file: data.csv):\n{data_profile}"},
    ]
    messages = list(base)
    history = []

    for attempt in range(1, max_attempts + 1):
        reply = chat(messages)                 # 1) ask the LLM for code
        code = extract_code(reply)

        if code:                               # 2) run it safely
            result = run_in_sandbox(code, workspace)
        else:                                  # the model returned nothing usable
            result = SandboxResult(ok=False, stdout="", stderr="Your reply contained no code.",
                                   returncode=-2, timed_out=False)

        history.append({"attempt": attempt, "code": code, "summary": result.summary()})
        if verbose:
            print(f"[coder] attempt {attempt}/{max_attempts}: {'OK' if result.ok else 'FAILED'}")

        if result.ok:                          # 3) success: stop here
            return StepOutcome(True, attempt, code, result, history)

        # 4) failure: rebuild the conversation as
        #    original request + its last answer + the error.
        #    We do NOT keep every past attempt, which saves tokens on free tiers.
        messages = base + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": (
                "Your script failed when executed.\n\n"
                f"{result.summary()}\n\n"
                "Read the error carefully, fix the root cause, and reply with the "
                "full corrected script in one code block."
            )},
        ]

    return StepOutcome(False, max_attempts, code, result, history)
