"""Shared factory for the chat-completion client used for generation,
reranking, citation verification, and eval judging.

Points at Groq's free, OpenAI-compatible API by default (get a free key at
https://console.groq.com/keys). To use a different OpenAI-compatible
provider or endpoint instead (OpenAI itself, a local Ollama/vLLM server
with the OpenAI shim, etc.), just set GROQ_BASE_URL and GROQ_API_KEY in
.env -- no call sites need to change.
"""
from __future__ import annotations

from openai import OpenAI

from app.config import settings
from app.rate_limiter import get_rate_limiter

_client: OpenAI | None = None


def get_chat_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
                "and put it in your .env file before running generation, reranking, "
                "citation verification, or eval."
            )
        _client = OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)
    return _client


def rate_limited_chat_completion(client: OpenAI, **kwargs):
    """Thin wrapper around client.chat.completions.create that proactively
    throttles to the configured RPM limit for the requested model before
    making the call. Every call site in the pipeline should go through
    this instead of calling client.chat.completions.create directly, so
    the rate limit is enforced project-wide from one place."""
    get_rate_limiter(kwargs["model"]).acquire()
    return client.chat.completions.create(**kwargs)
