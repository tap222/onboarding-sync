"""Resilience primitives: error taxonomy, retry with backoff, circuit breaker, bulkheads.

Composition for every downstream call (outermost first):
    bulkhead semaphore -> retry -> circuit breaker -> connector call
The breaker sits *inside* the retry so every attempt counts toward tripping it,
and an open breaker stops the retry loop immediately instead of hammering a dead API.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")


class TransientError(Exception):
    """Retryable: 429, 5xx, timeouts, connection resets."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class PermanentError(Exception):
    """Not retryable: 400 validation, 404, 409 conflict. Retrying only makes noise."""


class CircuitOpenError(Exception):
    """The breaker is open; the call was not attempted. Defer, don't fail."""


class CallTimeout(TransientError):
    """A call exceeded its deadline. Retryable - but the write may have landed,
    which is why every downstream write carries an idempotency key."""


async def with_timeout(fn: Callable[[], Awaitable[T]], seconds: float, what: str) -> T:
    try:
        return await asyncio.wait_for(fn(), timeout=seconds)
    except asyncio.TimeoutError:
        raise CallTimeout(f"{what} timed out after {seconds:g}s") from None


@dataclass(frozen=True)
class Timeouts:
    """Every wait in the system has an upper bound. Defaults are production-ish;
    the demo and tests shrink them."""
    call: dict = None              # per-system downstream call timeout (s)
    call_default: float = 10.0
    retry_budget: float = 60.0     # total time one write may spend retrying
    handler: float = 120.0         # one event handler (must be < worker lease)
    lease: float = 300.0           # worker lease on a claimed event
    claude: float = 90.0           # one triage/extraction model call, incl. tool loop
    approval_sla: float = 4 * 3600  # escalate a pending approval after this
    stuck_run: float = 600.0       # a 'running' run untouched this long is re-driven

    def for_system(self, name: str) -> float:
        return (self.call or {}).get(name, self.call_default)

    def __post_init__(self):
        if self.handler >= self.lease:
            raise ValueError("handler timeout must be shorter than the worker lease, or a slow handler "
                             "would be delivered twice concurrently")


@dataclass
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 0.2
    max_delay: float = 5.0
    jitter: bool = True

    def delay(self, attempt: int) -> float:
        # Exponential backoff with full jitter (AWS architecture blog): spreads retries
        # from many workers so they don't hit a recovering API in lockstep.
        backoff = min(self.max_delay, self.base_delay * 2 ** (attempt - 1))
        return random.uniform(0, backoff) if self.jitter else backoff


async def retry_async(fn: Callable[[], Awaitable[T]], policy: RetryPolicy,
                      on_retry: Optional[Callable[..., None]] = None,
                      budget_s: Optional[float] = None) -> tuple[T, int]:
    """Returns (result, attempts). Retries only TransientError, stopping at
    max_attempts or when the next wait would exceed the time budget."""
    attempt = 0
    started = time.monotonic()
    while True:
        attempt += 1
        try:
            return await fn(), attempt
        except TransientError as exc:
            wait = exc.retry_after if exc.retry_after is not None else policy.delay(attempt)
            over_budget = budget_s is not None and time.monotonic() - started + wait > budget_s
            if attempt >= policy.max_attempts or over_budget:
                exc.attempts = attempt  # type: ignore[attr-defined]
                if over_budget:
                    exc.args = (f"{exc.args[0]} (retry budget {budget_s:g}s exhausted)",)
                raise
            if on_retry:
                try:
                    on_retry(attempt, exc, wait)
                except TypeError:
                    on_retry(attempt, exc)
            await asyncio.sleep(wait)


class CircuitBreaker:
    """closed -> (N consecutive failures) -> open -> (cooldown) -> half_open -> 1 probe
    -> success: closed | failure: open again."""

    def __init__(self, name: str, failure_threshold: int = 3, reset_timeout: float = 2.0,
                 clock: Callable[[], float] = time.monotonic):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.clock = clock
        self.state = "closed"
        self.failures = 0
        self.opened_at = 0.0
        self._probe_in_flight = False

    def _current_state(self) -> str:
        if self.state == "open" and self.clock() - self.opened_at >= self.reset_timeout:
            self.state = "half_open"
        return self.state

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        state = self._current_state()
        if state == "open":
            raise CircuitOpenError(f"{self.name} circuit open")
        if state == "half_open":
            if self._probe_in_flight:
                raise CircuitOpenError(f"{self.name} circuit half-open, probe in flight")
            self._probe_in_flight = True
        try:
            result = await fn()
        except TransientError:
            self._on_failure()
            raise
        except PermanentError:
            # A 400 says nothing about the health of the service: don't trip on it.
            self._release_probe()
            raise
        else:
            self._on_success()
            return result

    def _release_probe(self) -> None:
        self._probe_in_flight = False

    def _on_success(self) -> None:
        self.state, self.failures = "closed", 0
        self._release_probe()

    def _on_failure(self) -> None:
        self.failures += 1
        if self.state == "half_open" or self.failures >= self.failure_threshold:
            self.state, self.opened_at = "open", self.clock()
        self._release_probe()

    def snapshot(self) -> dict:
        return {"system": self.name, "state": self._current_state(), "failures": self.failures}


class Bulkheads:
    """Per-system concurrency caps (e.g. Okta's API rate limits) plus a global cap,
    so one slow system can't take every worker hostage."""

    def __init__(self, per_system: dict[str, int], global_limit: int = 8):
        self._global = asyncio.Semaphore(global_limit)
        self._per = {name: asyncio.Semaphore(n) for name, n in per_system.items()}
        self.in_flight: dict[str, int] = {name: 0 for name in per_system}
        self.peak: dict[str, int] = {name: 0 for name in per_system}

    async def run(self, system: str, fn: Callable[[], Awaitable[T]]) -> T:
        async with self._global, self._per[system]:
            self.in_flight[system] += 1
            self.peak[system] = max(self.peak[system], self.in_flight[system])
            try:
                return await fn()
            finally:
                self.in_flight[system] -= 1
