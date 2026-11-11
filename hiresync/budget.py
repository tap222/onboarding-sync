"""Token and cost guardrails for every model call.

Checks, in order, before each model turn (the tool loop re-sends history, so
every turn is checked, not just the first):
  1. per-call input size   estimated tokens of system + messages + tools
  2. per-call output cap   max_tokens is clamped to the limit
  3. per-run budget        tokens and USD already spent by this run + this turn
  4. daily budget          USD spent today across the service; when exceeded the
                           agent falls back to the deterministic heuristic, and
                           Slack/webhooks are alerted once per day
  5. concurrency           at most N model calls in flight (rate-limit friendly)
After each turn the real usage from the API response is priced and stored on
the llm_calls row, which is the ledger the run and daily checks read.

Estimates are deliberately conservative (3 chars/token). Production can call
the count_tokens endpoint for exact preflight counts on large inputs; the
decision logic stays the same.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

if TYPE_CHECKING:
    from .store import Store

# USD per million tokens (Anthropic first-party list prices).
PRICES = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.10},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01},
}


def price(model: str, usage: dict) -> float:
    p = PRICES.get(model)
    if p is None:
        raise BudgetExceeded("UNKNOWN_MODEL_PRICE", f"no price for {model}; refusing to spend blind")
    cached = usage.get("cache_read_input_tokens", 0) or 0  # billed separately from input_tokens
    return ((usage.get("input_tokens", 0) or 0) * p["input"] + cached * p["cache_read"]
            + (usage.get("output_tokens", 0) or 0) * p["output"]) / 1e6


def estimate_tokens(payload: Any) -> int:
    text = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    return len(text) // 3 + 1


@dataclass(frozen=True)
class Limits:
    max_input_tokens_per_call: int = 40_000
    max_output_tokens_per_call: int = 8_000
    max_tokens_per_run: int = 120_000
    max_usd_per_run: float = 1.00
    max_usd_per_day: float = 25.00
    max_concurrent_calls: int = 4


class BudgetExceeded(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


class CostGuard:
    def __init__(self, store: "Store", limits: Optional[Limits] = None,
                 alert: Optional[Callable[[str, dict], Awaitable[None]]] = None):
        self.store = store
        self.limits = limits or Limits()
        self.alert = alert
        self._sem = asyncio.Semaphore(self.limits.max_concurrent_calls)
        self._alerted_day: Optional[str] = None
        self.blocked = 0

    @property
    def slot(self) -> asyncio.Semaphore:
        return self._sem

    def max_tokens(self, requested: int) -> int:
        return min(requested, self.limits.max_output_tokens_per_call)

    async def preflight(self, model: str, request_payload: Any, run_id: Optional[str],
                        spent_this_call: dict) -> None:
        """Raise BudgetExceeded if this turn may not be sent."""
        L = self.limits
        est_in = estimate_tokens(request_payload)
        if est_in > L.max_input_tokens_per_call:
            self._block("INPUT_TOO_LARGE")
            raise BudgetExceeded("INPUT_TOO_LARGE", f"~{est_in} input tokens > {L.max_input_tokens_per_call} per call")
        worst_case = price(model, {"input_tokens": est_in, "output_tokens": L.max_output_tokens_per_call})
        if run_id:
            run_tokens, run_usd = self.store.llm_spend_for_run(run_id)
            run_tokens += spent_this_call.get("tokens", 0)
            run_usd += spent_this_call.get("usd", 0.0)
            if run_tokens + est_in > L.max_tokens_per_run:
                self._block("RUN_TOKENS")
                raise BudgetExceeded("RUN_TOKENS", f"run {run_id} would exceed {L.max_tokens_per_run} tokens")
            if run_usd + worst_case > L.max_usd_per_run:
                self._block("RUN_COST")
                raise BudgetExceeded("RUN_COST", f"run {run_id} would exceed ${L.max_usd_per_run:.2f} "
                                                 f"(spent ${run_usd:.4f}, next turn up to ${worst_case:.4f})")
        today = datetime.now(timezone.utc).date().isoformat()
        day_usd = self.store.llm_spend_since(today) + spent_this_call.get("usd", 0.0)
        if day_usd + worst_case > L.max_usd_per_day:
            self._block("DAILY_COST")
            if self.alert and self._alerted_day != today:
                self._alerted_day = today
                await self.alert("budget.daily_exceeded", {"day": today, "spent_usd": round(day_usd, 4),
                                                           "limit_usd": L.max_usd_per_day})
            raise BudgetExceeded("DAILY_COST", f"daily model spend ${day_usd:.2f} reached the "
                                               f"${L.max_usd_per_day:.2f} cap; using deterministic fallback")

    def _block(self, code: str = "") -> None:
        self.blocked += 1
        from . import telemetry
        telemetry.count("budget_blocked", code=code)
