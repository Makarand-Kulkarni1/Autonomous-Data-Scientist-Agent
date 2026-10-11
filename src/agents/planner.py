import json

from pydantic import ValidationError

from src.llm import chat
from src.schemas import Plan

SYSTEM_PROMPT = """You are a senior data scientist creating a plan for a junior colleague to execute step by step.

RULES
- Reply with ONLY a single JSON object. No explanations, no markdown code fences.
- The JSON must have exactly these keys: "target_column", "problem_type", "steps".
- "problem_type" must be exactly "classification" or "regression".
- "steps" is a list of 4 to 7 objects, each with: "id" (int, starting at 1), "title" (short), "task" (a precise instruction).
- Always include, in this order: (1) explore/profile the data, (2) handle missing values and encode categorical columns,
  (3) engineer or select features if useful, (4) train and compare at least two models, (5) evaluate the best model
  with appropriate metrics and save a chart, (6) save the trained model to disk with joblib.
- Each task must be understandable on its own, without seeing the other steps.
Wrap the JSON in a code block, and add one sentence of explanation before it.
"""


def _extract_json(reply: str) -> str:
    """Pull a JSON object out of a reply that might have extra text around it."""
    reply = reply.strip()
    # Handle models that wrap JSON in ```json ... ``` anyway, despite instructions
    if reply.startswith("```"):
        reply = reply.strip("`")
        if reply.lower().startswith("json"):
            reply = reply[4:]
    # Take the text between the first { and the last }
    start, end = reply.find("{"), reply.rfind("}")
    if start == -1 or end == -1:
        return reply
    return reply[start:end + 1]


def make_plan(goal: str, data_profile: str, max_attempts: int = 3) -> Plan:
    """Ask the LLM for a plan and return it as a validated Plan object.

    Retries if the JSON is malformed or doesn't match the Plan shape,
    feeding the specific error back to the LLM each time.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"GOAL: {goal}\n\nDATA PROFILE:\n{data_profile}"},
    ]

    for attempt in range(1, max_attempts + 1):
        reply = chat(messages)
        json_text = _extract_json(reply)

        try:
            data = json.loads(json_text)             # 1) is it valid JSON at all?
            plan = Plan(**data)                       # 2) does it match our schema?

            # 3) a check Pydantic can't express on its own
            if plan.problem_type not in ("classification", "regression"):
                raise ValueError(f"problem_type must be 'classification' or 'regression', "
                                 f"got '{plan.problem_type}'")
            return plan

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            print(f"[planner] attempt {attempt}/{max_attempts} failed: {e}")
            if attempt == max_attempts:
                raise
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": (
                    f"That was invalid: {e}\n"
                    "Reply again with ONLY a corrected JSON object, no other text."
                )},
            ]


def _extract_json(reply: str) -> str:
    """Pull a JSON object out of a reply that might have extra text around it."""
    reply = reply.strip()
    # Handle models that wrap JSON in ```json ... ``` anyway, despite instructions
    if reply.startswith("```"):
        reply = reply.strip("`")
        if reply.lower().startswith("json"):
            reply = reply[4:]
    # Take the text between the first { and the last }
    start, end = reply.find("{"), reply.rfind("}")
    if start == -1 or end == -1:
        return reply
    return reply[start:end + 1]


def make_plan(goal: str, data_profile: str, max_attempts: int = 3) -> Plan:
    """Ask the LLM for a plan and return it as a validated Plan object.

    Retries if the JSON is malformed or doesn't match the Plan shape,
    feeding the specific error back to the LLM each time.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"GOAL: {goal}\n\nDATA PROFILE:\n{data_profile}"},
    ]

    for attempt in range(1, max_attempts + 1):
        reply = chat(messages)
        json_text = _extract_json(reply)

        try:
            data = json.loads(json_text)             # 1) is it valid JSON at all?
            plan = Plan(**data)                       # 2) does it match our schema?

            # 3) a check Pydantic can't express on its own
            if plan.problem_type not in ("classification", "regression"):
                raise ValueError(f"problem_type must be 'classification' or 'regression', "
                                 f"got '{plan.problem_type}'")
            return plan

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            print(f"[planner] attempt {attempt}/{max_attempts} failed: {e}")
            if attempt == max_attempts:
                raise
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": (
                    f"That was invalid: {e}\n"
                    "Reply again with ONLY a corrected JSON object, no other text."
                )},
            ]
