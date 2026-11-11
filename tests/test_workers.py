"""Async worker pool, dead-letter queue and timeouts."""
import asyncio
import json
import random
import time

import pytest

from conftest import AUDITOR, PEOPLE_OPS, hire, ingest, make_app
from hiresync import tools as toolreg
from hiresync.connectors import default_systems
from hiresync.events import DbQueueBus
from hiresync.resilience import CallTimeout, RetryPolicy, Timeouts, retry_async, with_timeout
from hiresync.store import Store


def fast_timeouts(**over):
    base = dict(call={}, call_default=0.2, retry_budget=2.0, handler=2.0, lease=3.0, claude=1.0,
                approval_sla=3600, stuck_run=600)
    base.update(over)
    return Timeouts(**base)


# ------------------------------------------------------------ worker pool
async def test_parallel_runs_cannot_race_to_the_same_okta_login(tmp_path):
    """Regression: with parallel workers, same-name hires all passed validation (none
    could see the others' in-flight Okta writes) and failed at Okta with 409s.
    The DB login reservation makes exactly one win; the rest are held for a human."""
    app = make_app(str(tmp_path / "t.db"), workers=4)
    await app.start()
    for i in range(6):
        await app.submit_change(f"H{i}", hire(), source="ashby", actor="t", event_id=f"e{i}")
    await app.drain()
    statuses = sorted(app.store.get_run(f"H{i}:v1")["status"] for i in range(6))
    assert statuses == ["awaiting_approval"] * 5 + ["completed"]
    assert app.systems["okta"].writes == 1
    await app.stop()



async def test_workers_run_hires_in_parallel_and_keep_per_hire_order():
    store = Store(":memory:")
    bus = DbQueueBus(store, workers=4, lease_s=5, handler_timeout_s=2, poll_interval=0.01)
    seen: dict[str, list[int]] = {}
    active, peak = 0, 0

    async def handler(v):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(random.uniform(0.005, 0.03))
        seen.setdefault(v["hire"], []).append(v["n"])
        active -= 1

    bus.subscribe("t", "g", handler)
    for n in range(5):
        for h in ("A", "B", "C", "D"):
            store.enqueue("t", h, {"hire": h, "n": n})
    await bus.start()
    await bus.drain()
    await bus.stop()
    assert all(seq == [0, 1, 2, 3, 4] for seq in seen.values())  # order per hire across 4 workers
    assert peak >= 3  # different hires really ran concurrently
    assert sum(1 for c in bus.per_worker.values() if c) >= 2


async def test_more_workers_finish_faster(tmp_path):
    async def run(workers):
        systems = default_systems()
        for s in systems.values():
            s.latency = 0.03
        app = make_app(str(tmp_path / f"w{workers}.db"), systems=systems, workers=workers)
        await app.start()
        t = time.monotonic()
        names = ["Ana", "Ben", "Cleo", "Dev", "Eli", "Fay", "Gus", "Hana"]
        for i in range(8):
            await app.submit_change(f"H{i}", hire(names[i]), source="ashby", actor="t", event_id=f"e{i}")
        await app.drain()
        elapsed = time.monotonic() - t
        assert all(app.store.get_run(f"H{i}:v1")["status"] == "completed" for i in range(8))
        await app.stop()
        return elapsed

    one, four = await run(1), await run(4)
    assert four < one * 0.6, (one, four)


async def test_expired_lease_is_taken_over():
    store = Store(":memory:")
    store.enqueue("t", "A", {"x": 1})
    assert store.claim_next("crashed-worker", lease_s=0.05)  # claimed, never acked
    assert store.claim_next("other", lease_s=5) is None  # still leased
    await asyncio.sleep(0.06)
    row = store.claim_next("other", lease_s=5)
    assert row and row["lease_owner"] == "other"
    assert not store.ack(row["id"], "crashed-worker")  # fenced: old owner can't ack
    assert store.ack(row["id"], "other")


async def test_handler_timeout_backoff_then_dead_letter_then_redrive():
    store = Store(":memory:")
    dead = []

    async def on_dead(row):
        dead.append(row)

    bus = DbQueueBus(store, workers=2, lease_s=1, handler_timeout_s=0.05, max_deliveries=3,
                     backoff_base_s=0.01, poll_interval=0.01, on_dead=on_dead)
    behaviour = {"hang": True}
    done = []

    async def handler(v):
        if behaviour["hang"] and v["n"] == 1:  # only event 1 is poison
            await asyncio.sleep(1)
        done.append(v)

    bus.subscribe("t", "g", handler)
    store.enqueue("t", "A", {"n": 1})
    store.enqueue("t", "A", {"n": 2})  # behind the poison event, same hire
    await bus.start()
    await bus.drain()
    assert bus.stats["timeouts"] == 3 and len(dead) == 1 and "handler timeout" in dead[0]["last_error"]
    assert done == [{"n": 2}]  # dead event stopped blocking its hire
    dlq = store.dead_events()
    assert len(dlq) == 1 and dlq[0]["attempts"] == 3
    behaviour["hang"] = False
    assert store.redrive_event(dlq[0]["id"], "u-maya")
    await bus.drain()
    await bus.stop()
    assert done == [{"n": 2}, {"n": 1}] and store.dead_events() == []


# --------------------------------------------------------------- timeouts
async def test_with_timeout_raises_retryable_call_timeout():
    with pytest.raises(CallTimeout):
        await with_timeout(lambda: asyncio.sleep(1), 0.01, "x")


async def test_retry_budget_stops_retrying_early():
    calls = 0

    async def flaky():
        nonlocal calls
        calls += 1
        raise CallTimeout("slow")

    t = time.monotonic()
    with pytest.raises(CallTimeout, match="budget"):
        await retry_async(flaky, RetryPolicy(max_attempts=50, base_delay=0.05, max_delay=0.05), budget_s=0.12)
    # Full jitter makes the *number* of attempts random; the budget bounds the *time*.
    assert time.monotonic() - t < 0.12 + 0.05 + 0.03 and calls < 50


async def test_hanging_system_times_out_and_is_dead_lettered(tmp_path):
    systems = default_systems()
    app = make_app(str(tmp_path / "t.db"), systems=systems, timeouts=fast_timeouts())
    await app.start()
    systems["expoit"].hang_next(10, seconds=5)
    await ingest(app, "H1", hire(), "e1")
    run = app.store.get_run("H1:v1")
    assert run["status"] == "partial"
    assert "timed out" in run["state"]["results"]["expoit"]["error"]
    assert [d["system"] for d in app.store.list_dead_letters()] == ["expoit"]
    await app.stop()


async def test_timeout_after_write_does_not_double_write(tmp_path):
    """The write lands, the response is lost, the retry replays the idempotency key."""
    systems = default_systems()
    app = make_app(str(tmp_path / "t.db"), systems=systems, timeouts=fast_timeouts())
    await app.start()
    systems["workday"].hang_next(1, seconds=5, after_write=True)
    await ingest(app, "H1", hire(), "e1")
    run = app.store.get_run("H1:v1")
    assert run["status"] == "completed" and run["state"]["results"]["workday"]["attempts"] == 2
    assert systems["workday"].writes == 1
    await app.stop()


async def test_claude_timeout_falls_back_to_heuristic(app):
    from hiresync.agent import TriageAgent
    from hiresync.memory import WorkingMemory

    class Slow:
        def __init__(self):
            self.beta = self
            self.messages = self

        async def create(self, **kw):
            await asyncio.sleep(5)

    agent = TriageAgent(mode="heuristic")
    agent.use_claude, agent.client, agent.app, agent.timeout_s = True, Slow(), app, 0.05
    wm = WorkingMemory(run_id="r", hire_id="H1", from_version=1, to_version=2,
                       issues=[{"code": "COUNTRY_MISMATCH", "severity": "warn", "message": "m"}])
    rec, _ = await agent.triage(wm, [])
    assert rec.source == "heuristic" and "CallTimeout" in rec.rationale


# ---------------------------------------------------------------- sweeper
async def test_sweeper_escalates_stale_approvals_once(tmp_path):
    app = make_app(str(tmp_path / "t.db"), timeouts=fast_timeouts(approval_sla=0.0), sweep_interval_s=0)
    await app.start()
    await ingest(app, "H1", hire(shipping_address={**hire()["shipping_address"], "country": "CA",
                                                   "postal_code": "M5V 2T6"}), "e1")
    assert app.store.list_approvals(status="pending")
    first = await app.orchestrator.sweep()
    second = await app.orchestrator.sweep()
    assert first["approvals_escalated"] == 1 and second["approvals_escalated"] == 0
    assert any("waited over" in m["text"] for m in app.slack.messages)
    await app.stop()


# -------------------------------------------------------------------- DLQ
async def test_dlq_tools_respect_permissions(tmp_path):
    systems = default_systems()
    app = make_app(str(tmp_path / "t.db"), systems=systems, timeouts=fast_timeouts())
    await app.start()
    systems["tracker"].fail_next(1, kind="permanent")
    await ingest(app, "H1", hire(), "e1")
    listed, err = await toolreg.execute(app, "list_dead_letters", {}, principal=AUDITOR)
    entries = json.loads(listed)
    assert not err and entries[0]["kind"] == "write" and entries[0]["payload"]["system"] == "tracker"
    _, err = await toolreg.execute(app, "redrive_dead_letter", {"kind": "write", "id": entries[0]["id"]},
                                   principal=AUDITOR)
    assert err  # auditors can look, not act
    out, err = await toolreg.execute(app, "redrive_dead_letter", {"kind": "write", "id": entries[0]["id"]},
                                     principal=PEOPLE_OPS)
    assert not err and json.loads(out)["status"] == "completed"
    assert app.store.list_dead_letters() == []
    _, err = await toolreg.execute(app, "discard_dead_letter", {"kind": "event", "id": 999}, principal=PEOPLE_OPS)
    assert err  # a note is required
    await app.stop()
