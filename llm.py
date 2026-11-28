"""The only code that talks to the model server (Ollama, through its OpenAI-compatible API)."""
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()  # before importing langfuse so the keys are picked up

from langfuse import get_client
from langfuse.openai import OpenAI  # traced by Langfuse
from openai import OpenAI as PlainOpenAI  # not traced

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


@lru_cache
def _get_client(base_url, api_key):
    return OpenAI(base_url=base_url, api_key=api_key)


def run_llm(
    name,
    model,
    system_prompt,
    user_prompt,
    metadata=None,
    schema=None,
    run_id=None,
    base_url=OLLAMA_URL,
    api_key="ollama",
    **params,  # temperature, top_p, max_tokens, reasoning_effort, ...
):
    """Return the full response text. If `schema` (a dict) is given, the output is constrained to it."""
    if schema is not None:
        params["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": name, "schema": schema},
        }

    metadata = {"langfuse_user_id": os.getenv("USERNAME"), **(metadata or {})}
    if run_id is not None:
        metadata.update(
            run_id=run_id,
            langfuse_tags=[f"run:{run_id}"],
        )

    response = _get_client(base_url, api_key).chat.completions.create(  # pyright: ignore[reportCallIssue]
        name=name,
        metadata=metadata,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **params,
    )
    get_client().flush()
    return response.choices[0].message.content


def embed(texts):
    """Return one embedding vector per text (not traced in Langfuse)."""
    client = PlainOpenAI(base_url=OLLAMA_URL, api_key="ollama")
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in res.data]
