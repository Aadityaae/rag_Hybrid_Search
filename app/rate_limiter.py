"""Proactive rate limiting for chat-completion calls.

Groq's free tier caps requests-per-minute *per model* (e.g. 30 RPM for
llama-3.1-8b-instant). The reranker, citation verifier, and confidence
scorer each make one call per candidate/claim, so a single question can
easily fire 10-20 judge calls back to back -- enough to blow past a 30
RPM cap within a couple of questions. Rather than bursting and relying on
retries to recover (slow, and can still exhaust the retry budget), this
throttles calls proactively to stay under the limit in the first place.

Limits are tracked per model name, since different models have different
RPM caps and the pipeline uses at least two (generation_model, judge_model).
"""
from __future__ import annotations

import threading
import time
from collections import deque

from app.config import settings

_limiters: dict[str, "RateLimiter"] = {}
_registry_lock = threading.Lock()


class RateLimiter:
    """Sliding-window limiter: blocks `acquire()` until a call would stay
    under `max_calls` within the trailing `period_seconds`."""

    def __init__(self, max_calls: int, period_seconds: float = 60.0):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float):
        while self._calls and now - self._calls[0] >= self.period_seconds:
            self._calls.popleft()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            if len(self._calls) >= self.max_calls:
                wait = self.period_seconds - (now - self._calls[0]) + 0.1
                if wait > 0:
                    print(f"[rate limit] pausing {wait:.1f}s to stay under the RPM cap...", flush=True)
                    time.sleep(wait)
                now = time.monotonic()
                self._prune(now)
            self._calls.append(now)


def get_rate_limiter(model: str) -> RateLimiter:
    with _registry_lock:
        if model not in _limiters:
            _limiters[model] = RateLimiter(max_calls=settings.groq_requests_per_minute)
        return _limiters[model]
