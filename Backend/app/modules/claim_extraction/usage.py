"""In-process, per-model Groq usage accounting and conservative throttling."""
from collections import defaultdict, deque
from contextlib import contextmanager
import logging
import re
import threading
import time

logger = logging.getLogger("uvicorn.error")


def parse_duration(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    matches = list(re.finditer(r"(\d+(?:\.\d+)?)(ms|h|m|s)", value))
    if not matches or "".join(m.group(0) for m in matches) != value.strip():
        return None
    scale = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
    return sum(float(m.group(1)) * scale[m.group(2)] for m in matches)


class UsageMonitor:
    """One serialized request per model per process; never prints credentials."""

    def __init__(self, clock=time.monotonic, sleep=time.sleep):
        self.clock, self.sleep = clock, sleep
        self._guard = threading.Lock()
        self._locks = defaultdict(threading.Lock)
        self._state = {}

    @contextmanager
    def slot(self, model, estimated_tokens, settings):
        with self._guard:
            lock = self._locks[model]
        with lock:
            state = self._state.setdefault(model, {"events": deque(), "requests": 0, "reported_tokens": 0, "estimated_tokens": 0})
            is_verifier = model == settings.verifier_model_name
            limit = settings.verifier_tokens_per_minute if is_verifier else settings.llm_tokens_per_minute
            rpm = settings.verifier_requests_per_minute if is_verifier else settings.llm_requests_per_minute
            fraction = settings.groq_rate_safety_fraction
            if limit and estimated_tokens > limit * fraction:
                raise ValueError("Estimated request exceeds the configured safe per-minute token budget.")
            waited = 0.0
            while True:
                now = self.clock()
                while state["events"] and now - state["events"][0][0] >= 60:
                    state["events"].popleft()
                delays = []
                total = sum(event[1] for event in state["events"])
                if state["events"] and ((limit and total + estimated_tokens > limit * fraction) or
                                        (rpm and len(state["events"]) >= max(1, int(rpm * fraction)))):
                    delays.append(max(0, 60 - (now - state["events"][0][0])))
                remaining = state.get("remaining_tokens")
                reset_at = state.get("token_reset_at", 0)
                margin = (state.get("token_limit") or 0) * (1 - fraction)
                if remaining is not None and now < reset_at and remaining < estimated_tokens + margin:
                    delays.append(reset_at - now)
                if state.get("remaining_requests") == 0 and now < state.get("request_reset_at", 0):
                    delays.append(state["request_reset_at"] - now)
                if now < state.get("retry_after_at", 0):
                    delays.append(state["retry_after_at"] - now)
                if not delays:
                    break
                delay = max(delays)
                if waited + delay > settings.groq_max_throttle_wait_seconds:
                    raise ValueError("Provider quota reset exceeds configured maximum throttle wait.")
                logger.warning("Groq model=%s throttling for %.1fs; local estimate=%s tokens", model, delay, estimated_tokens)
                self.sleep(delay)
                waited += delay
            yield

    def observe(self, model, headers, usage, estimated_tokens):
        state = self._state.setdefault(model, {"events": deque(), "requests": 0, "reported_tokens": 0, "estimated_tokens": 0})
        actual = usage.get("total_tokens") if isinstance(usage, dict) else None
        used = actual if isinstance(actual, int) and actual >= 0 else estimated_tokens
        state["requests"] += 1
        state["reported_tokens" if actual is not None else "estimated_tokens"] += used
        state["events"].append((self.clock(), used))
        def integer(name):
            try:
                return int(headers.get(name))
            except (ValueError, TypeError):
                return None
        remaining_requests = integer("x-ratelimit-remaining-requests")
        remaining_tokens = integer("x-ratelimit-remaining-tokens")
        total_requests = integer("x-ratelimit-limit-requests")
        total_tokens = integer("x-ratelimit-limit-tokens")
        reset_requests = parse_duration(headers.get("x-ratelimit-reset-requests"))
        reset_tokens = parse_duration(headers.get("x-ratelimit-reset-tokens"))
        state["remaining_tokens"] = remaining_tokens
        state["token_reset_at"] = self.clock() + (reset_tokens or 0)
        state["token_limit"] = total_tokens
        state["remaining_requests"] = remaining_requests
        state["request_reset_at"] = self.clock() + (reset_requests or 0)
        retry_after = parse_duration(headers.get("retry-after"))
        if retry_after is not None:
            state["retry_after_at"] = self.clock() + retry_after
        logger.info(
            "Groq model=%s local_requests=%d provider_reported_tokens=%d locally_estimated_tokens=%d; provider daily_requests_remaining=%s "
            "minute_tokens_remaining=%s request_reset=%s token_reset=%s",
            model, state["requests"], state["reported_tokens"], state["estimated_tokens"],
            remaining_requests, remaining_tokens, headers.get("x-ratelimit-reset-requests"),
            headers.get("x-ratelimit-reset-tokens"),
        )
        for label, remaining, total in (("daily requests", remaining_requests, total_requests),
                                        ("minute tokens", remaining_tokens, total_tokens)):
            if total and remaining is not None:
                percent = 100 * (1 - remaining / total)
                if percent >= 80:
                    logger.warning("Groq model=%s provider %s quota %.0f%% consumed%s", model, label, percent,
                                   " (90% threshold)" if percent >= 90 else " (80% threshold)")


usage_monitor = UsageMonitor()
