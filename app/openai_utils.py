"""Shared retry policy for OpenAI calls.

Distinguishes transient errors (rate limits with room left, connection
blips, timeouts) -- worth retrying with backoff -- from permanent errors
(insufficient_quota, invalid_api_key, model_not_found) that will never
succeed on retry and should fail immediately with a clear message instead
of being retried 3-4 times and then buried inside a tenacity.RetryError.
"""
from __future__ import annotations

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    NotFoundError,
    OpenAIError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


class OpenAIConfigError(RuntimeError):
    """Raised for errors that retrying will never fix (bad key, no quota,
    unknown model). Caller should surface this message directly to the user."""


def _error_code(exc: Exception) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return (body.get("error") or {}).get("code")
    return None


def _friendly_message(exc: Exception) -> str:
    code = _error_code(exc)
    if isinstance(exc, RateLimitError) and code == "insufficient_quota":
        return (
            "The configured account has no available quota (error code: insufficient_quota). "
            "If you're on Groq's free tier this is unusual -- check "
            "https://console.groq.com/settings/limits. If you've pointed this at OpenAI "
            "instead, add credits at https://platform.openai.com/settings/organization/billing/overview."
        )
    if isinstance(exc, AuthenticationError):
        return (
            "The chat API rejected the API key (authentication error). Double-check "
            "GROQ_API_KEY in your .env file -- get a free one at https://console.groq.com/keys."
        )
    if isinstance(exc, NotFoundError):
        return (
            f"The chat API returned 'not found' for the requested model. Check "
            f"GENERATION_MODEL / LLM_JUDGE_MODEL in your .env file match a model your "
            f"provider actually serves (see https://console.groq.com/docs/models for Groq). ({exc})"
        )
    return str(exc)


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitError):
        # A rate limit WITH quota remaining is transient (back off and retry).
        # insufficient_quota never recovers on its own -- don't retry it.
        return _error_code(exc) != "insufficient_quota"
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    return False


def raise_clear_error(exc: Exception):
    """Converts a non-transient OpenAI error into an OpenAIConfigError with
    a clear, actionable message. Call this in an `except OpenAIError` block."""
    raise OpenAIConfigError(_friendly_message(exc)) from exc


# Shared decorator: retries transient errors with backoff, and via
# reraise=True + the retry predicate above, immediately re-raises (rather
# than retrying or wrapping in RetryError) anything non-transient so the
# caller's `except OpenAIError` can turn it into a clear message.
#
# This is a safety net behind the proactive RateLimiter in
# app/rate_limiter.py, not the primary defense -- it exists for the rare
# case a 429 slips through anyway (e.g. another process sharing the same
# key). stop_after_attempt(6) + max wait of 20s gives ~60s of total
# backoff, enough to cover a full RPM window reset.
openai_retry = retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception(_is_transient),
    reraise=True,
)
