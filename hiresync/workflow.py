"""A small durable workflow engine on the existing database.

Same concepts as Temporal, so a later move is mechanical (see README), without
adding a cluster to run:

  workflow     a state machine class; one row in `workflows` holds its current state
               name, status and data (its working memory)
  step         `async def on_<state>(ctx)` returns Next(state) | Wait(...) | Done(status);
               after every step the engine checkpoints atomically: state, data,
               consumed signals, timer changes, history (store.wf_save)
  activity     ctx.activity(name, fn, compensate=...): a side effect, recorded by name so
               a re-run step skips it; activities must be idempotent (at-least-once, as
               in Temporal), which every connector call here is via idempotency keys
  timer        ctx.set_timer(name, when): a row in wf_timers; the timer loop fires due
               timers exactly once (flag flip + signal + wake in one transaction);
               survives restarts; re-arming a timer with a new time resets it
  signal       engine.signal(wf_id, name, payload): row in wf_signals + a wake event;
               handlers read them with ctx.take()/ctx.take_all()
  compensation each activity may register an undo; ctx.compensations() returns them
               newest first (saga: undo in reverse order)
  execution    every wake is a `workflow.wake` event on the worker queue keyed by wf_id,
               so one workflow advances strictly one step at a time, workflows run in
               parallel, and a failing step gets the queue's backoff, timeout and DLQ.

The engine's clock is injectable, so tests and the demo can fast-forward weeks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from . import telemetry

if TYPE_CHECKING:
    from .app import HireSync

log = logging.getLogger(__name__)


@dataclass
class Next:
    state: str
    note: str = ""


@dataclass
class Wait:
    note: str = ""


@dataclass
class Done:
    status: str = "completed"
    note: str = ""


class Ctx:
    def __init__(self, engine: "WorkflowEngine", wf: dict, signals: list[dict]):
        self.engine, self.app = engine, engine.app
        self.wf_id, self.key = wf["wf_id"], wf["key"]
        self.data: dict = wf["data"]
        self._signals = signals
        self.consumed: list[int] = []
        self.set_timers: dict[str, float] = {}
        self.clear_timers: list[str] = []

    def now(self) -> datetime:
        return self.engine.clock()

    # signals
    def take(self, name: str) -> Optional[dict]:
        for s in self._signals:
            if s["id"] not in self.consumed and s["name"] == name:
                self.consumed.append(s["id"])
                return s["payload"]
        return None

    def take_all(self, prefix: str = "") -> list[tuple[str, dict]]:
        out = []
        for s in self._signals:
            if s["id"] not in self.consumed and s["name"].startswith(prefix):
                self.consumed.append(s["id"])
                out.append((s["name"], s["payload"]))
        return out

    # timers
    def set_timer(self, name: str, when: datetime) -> None:
        self.set_timers[name] = when.timestamp()
        if name in self.clear_timers:
            self.clear_timers.remove(name)

    def clear_timer(self, name: str) -> None:
        self.clear_timers.append(name)
        self.set_timers.pop(name, None)

    # activities + compensation
    async def activity(self, name: str, fn: Callable[[], Awaitable[Any]],
                       compensate: Optional[dict] = None) -> Any:
        done = self.data.setdefault("activities", {})
        if name in done:
            return done[name]  # replayed step: don't repeat the side effect
        with telemetry.span(f"hiresync.workflow.activity {name}", **{"hiresync.wf_id": self.wf_id}):
            result = await fn()
        done[name] = result
        if compensate:
            self.data.setdefault("compensations", []).append({"activity": name, **compensate})
        return result

    def compensations(self) -> list[dict]:
        return list(reversed(self.data.get("compensations", [])))

    async def rollback(self, handlers: Optional[dict[str, Callable[[dict], Awaitable[Any]]]] = None,
                       retry_policy: Optional[Any] = None) -> list[dict]:
        """Execute registered compensations in reverse order (saga) with exponential backoff on retries and state tracking."""
        from .resilience import RetryPolicy, retry_async

        policy = retry_policy or RetryPolicy(max_attempts=3, base_delay=0.05, max_delay=0.5)
        results = []
        state_tracking = self.data.setdefault("compensation_state", {"status": "in_progress", "results": []})

        for comp in self.compensations():
            act_name = comp.get("activity")
            handler = (handlers or {}).get(act_name) or comp.get("fn")
            comp_entry: dict[str, Any] = {"activity": act_name, "status": "pending", "retries": []}
            if handler:
                async def run_comp():
                    return await handler(comp)

                def on_retry(att, exc, wait_s=None):
                    comp_entry["retries"].append({"attempt": att, "error": str(exc), "wait_s": wait_s, "strategy": "exponential_backoff"})

                try:
                    res, attempts = await retry_async(run_comp, policy, on_retry=on_retry)
                    comp_entry.update({"status": "succeeded", "result": res, "attempts": attempts})
                except Exception as exc:
                    comp_entry.update({"status": "failed", "error": str(exc)})
            else:
                comp_entry.update({"status": "no_handler"})
            results.append(comp_entry)
            state_tracking["results"].append(comp_entry)

        all_ok = all(r["status"] in ("succeeded", "no_handler") for r in results)
        state_tracking["status"] = "completed" if all_ok else "partial"
        return results


class Workflow:
    type = "base"
    initial = "start"

    def __init__(self, app: "HireSync"):
        self.app = app


class WorkflowEngine:
    MAX_STEPS_PER_WAKE = 25  # guards against a state machine that loops forever

    def __init__(self, app: "HireSync", clock: Callable[[], datetime]):
        self.app = app
        self.store = app.store
        self.clock = clock
        self.types: dict[str, type[Workflow]] = {}
        app.bus.subscribe("workflow.wake", "workflows", self._on_wake)

    def register(self, cls: type[Workflow]) -> None:
        self.types[cls.type] = cls

    def start(self, wf_type: str, key: str, data: Optional[dict] = None, wf_id: Optional[str] = None) -> str:
        wf_id = wf_id or f"{wf_type}:{key}"
        if self.store.wf_create(wf_id, wf_type, key, self.types[wf_type].initial, data or {}):
            telemetry.count("workflows_started", type=wf_type)
        return wf_id

    def signal(self, wf_id: str, name: str, payload: Optional[dict] = None) -> bool:
        if not self.store.wf_get(wf_id):
            return False
        self.store.wf_signal(wf_id, name, payload or {})
        return True

    def tick(self) -> int:
        """Fire due timers (the timer loop calls this; tests/demo call it after moving the clock)."""
        return self.store.fire_due_timers(self.clock().timestamp())

    async def _on_wake(self, msg: dict) -> None:
        await self.advance(msg["wf_id"])

    async def advance(self, wf_id: str) -> None:
        wf = self.store.wf_get(wf_id)
        if not wf or wf["status"] in ("completed", "cancelled", "failed"):
            return
        inst = self.types[wf["type"]](self.app)
        ctx = Ctx(self, wf, self.store.wf_pending_signals(wf_id))
        state, status, history = wf["state"], "running", []
        with telemetry.span(f"hiresync.workflow.advance {wf['type']}", **{
                "hiresync.wf_id": wf_id, "hiresync.wf.from_state": state}) as sp:
            for _ in range(self.MAX_STEPS_PER_WAKE):
                result = await getattr(inst, f"on_{state}")(ctx)
                if isinstance(result, Next):
                    history.append((state, result.state, result.note))
                    state = result.state
                    continue
                if isinstance(result, Wait):
                    status = "waiting"
                    if result.note:
                        history.append((state, state, result.note))
                    break
                if isinstance(result, Done):
                    status = result.status
                    history.append((state, result.status, result.note))
                    ctx.clear_timers.extend(t["name"] for t in self.store.wf_timers(wf_id))
                    break
                raise TypeError(f"{wf['type']}.on_{state} returned {result!r}")
            else:
                raise RuntimeError(f"{wf_id}: more than {self.MAX_STEPS_PER_WAKE} transitions in one wake")
            sp.set_attribute("hiresync.wf.to_state", state)
            sp.set_attribute("hiresync.wf.status", status)
        self.store.wf_save(wf_id, state, status, ctx.data, ctx.consumed, ctx.set_timers,
                           sorted(set(ctx.clear_timers)), history)


def utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass
class FakeClock:
    """Deterministic clock for tests and the demo: advance() moves time forward."""
    current: datetime = field(default_factory=lambda: datetime(2026, 11, 10, 9, 0, tzinfo=timezone.utc))

    def __call__(self) -> datetime:
        return self.current

    def advance(self, **delta) -> datetime:
        from datetime import timedelta
        self.current = self.current + timedelta(**delta)
        return self.current
