import os
import time

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, APIStatusError

load_dotenv()

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
            timeout=180,
        )
    return _client


def chat(messages: list, temperature: float | None = None, max_retries: int = 5) -> str:
    client = get_client()
    model = os.environ["LLM_MODEL"]

    kwargs = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""

        except (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError) as e:
            # APIStatusError covers 503 "service unavailable" and similar transient
            # server-side errors that aren't specifically rate limits.
            if attempt == max_retries - 1:
                raise
            wait = min(60, 5 * 2 ** attempt)
            print(f"[llm] {type(e).__name__}: retrying in {wait}s "
                  f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
