import asyncio
import json
import time
from datetime import date, timedelta
import pytest

from conftest import PEOPLE_OPS, RECRUITER, TODAY, hire, ingest, make_app
from hiresync import tools as toolreg
from hiresync.connectors import CohortTracker, ExpoIT, Lumos, MockSystem, Okta, Workday, default_systems, topo_levels
from hiresync.models import RunStatus, Step
from hiresync.resilience import CallTimeout, PermanentError, RetryPolicy, Timeouts, TransientError, retry_async
from hiresync.workflow import Ctx, Workflow, WorkflowEngine, FakeClock


# ==============================================================================
# 1. Connectors: Idempotency, Semantic Current-State Dependency, Faults
# ==============================================================================

async def test_mock_system_compensate_idempotency():
    system = MockSystem(latency=0.001)
    system.records["H-1"] = {"name": "Test User", "status": "active"}

    # First compensation
    res1 = await system.compensate("H-1", idempotency_key="undo-h1", today=TODAY)
    assert res1["action"] == "withdrawn"
    assert res1["status"] == "withdrawn"
    assert not res1.get("replayed")
    assert system.records["H-1"]["status"] == "withdrawn"
    assert system.writes == 1
    assert system.compensations == 1

    # Replayed compensation with same key
    res2 = await system.compensate("H-1", idempotency_key="undo-h1", today=TODAY)
    assert res2["action"] == "withdrawn"
    assert res2.get("replayed") is True
    # Does not double-write
    assert system.writes == 1
    assert system.compensations == 1


async def test_compensate_when_nothing_to_undo():
    system = MockSystem(latency=0.001)
    res = await system.compensate("NONEXISTENT", idempotency_key="k1", today=TODAY)
    assert res["action"] == "nothing_to_undo"
    assert res["status"] == "not_found"


async def test_workday_compensate_semantic_state():
    wd = Workday(latency=0.001)
    wd.records["H-10"] = {"legal_name": {"first": "Jane", "last": "Doe"}, "status": "Active"}

    res = await wd.compensate("H-10", idempotency_key="wd-undo-1", today=TODAY)
    assert res["action"] == "hire_rescinded"
    assert res["status"] == "Rescinded"
    assert wd.records["H-10"]["status"] == "Rescinded"


async def test_okta_compensate_captures_previous_state():
    okta = Okta(latency=0.001)
    # Stage 1: staged user
    okta.records["H-20"] = {"login": "jane.doe@betterup.co", "status": "STAGED"}
    res1 = await okta.compensate("H-20", idempotency_key="okta-undo-1", today=TODAY)
    assert res1["action"] == "deprovisioned"
    assert res1["was"] == "STAGED"
    assert res1["status"] == "DEPROVISIONED"
    assert okta.records["H-20"]["status"] == "DEPROVISIONED"

    # Stage 2: user was ACTIVE
    okta.records["H-21"] = {"login": "john.smith@betterup.co", "status": "ACTIVE"}
    res2 = await okta.compensate("H-21", idempotency_key="okta-undo-2", today=TODAY)
    assert res2["action"] == "deprovisioned"
    assert res2["was"] == "ACTIVE"
    assert res2["status"] == "DEPROVISIONED"
    assert okta.records["H-21"]["status"] == "DEPROVISIONED"


async def test_lumos_compensate_revokes_bundle():
    lumos = Lumos(latency=0.001)
    lumos.records["H-30"] = {"bundle": ["slack", "google-workspace", "github"], "status": "active"}

    res = await lumos.compensate("H-30", idempotency_key="lumos-undo-1", today=TODAY)
    assert res["action"] == "access_revoked"
    assert res["apps"] == 3
    assert res["status"] == "revoked"
    assert lumos.records["H-30"]["bundle"] == []
    assert lumos.records["H-30"]["status"] == "revoked"


async def test_expoit_semantic_compensation_based_on_ship_by_date():
    expoit = ExpoIT(latency=0.001)
    ship_by = TODAY + timedelta(days=5)  # in the future
    expoit.records["H-40"] = {
        "recipient": "Jane Doe",
        "ship_to": {"city": "Austin", "state": "TX"},
        "ship_by": ship_by.isoformat(),
        "status": "pending_shipment",
    }

    # Case A: today is BEFORE ship_by date -> order is cancelled
    res_before = await expoit.compensate("H-40", idempotency_key="expo-undo-1", today=TODAY)
    assert res_before["action"] == "order_cancelled"
    assert res_before["status"] == "cancelled"
    assert expoit.records["H-40"]["status"] == "cancelled"

    # Case B: today is ON or AFTER ship_by date -> return label sent
    expoit.records["H-41"] = {
        "recipient": "John Smith",
        "ship_to": {"city": "Denver", "state": "CO"},
        "ship_by": (TODAY - timedelta(days=1)).isoformat(),  # in the past
        "status": "shipped",
    }
    res_after = await expoit.compensate("H-41", idempotency_key="expo-undo-2", today=TODAY)
    assert res_after["action"] == "return_label_sent"
    assert res_after["to"] == "Denver"
    assert res_after["status"] == "return_requested"
    assert expoit.records["H-41"]["status"] == "return_requested"


async def test_cohort_tracker_compensate_marks_withdrawn():
    tracker = CohortTracker(latency=0.001)
    tracker.records["H-50"] = {"name": "Jane Doe", "status": "active"}

    res = await tracker.compensate("H-50", idempotency_key="tracker-undo-1", today=TODAY)
    assert res["action"] == "cohort_withdrawn"
    assert res["status"] == "withdrawn"
    assert tracker.records["H-50"]["status"] == "withdrawn"


# ==============================================================================
# 2. Fault Injection on Compensation: Transient, Permanent, Hang After Write
# ==============================================================================

async def test_compensate_fault_transient_retries_with_exponential_backoff():
    system = MockSystem(latency=0.001)
    system.records["H-60"] = {"status": "active"}
    system.fail_next(2, kind="transient")

    policy = RetryPolicy(max_attempts=4, base_delay=0.02, max_delay=0.2, jitter=False)
    retries = []

    def on_retry(attempt, exc, wait_s=None):
        retries.append({"attempt": attempt, "error": str(exc), "wait_s": wait_s})

    async def do_compensate():
        return await system.compensate("H-60", "key-retry-test", TODAY)

    res, attempts = await retry_async(do_compensate, policy, on_retry=on_retry)
    assert res["action"] == "withdrawn"
    assert attempts == 3
    assert len(retries) == 2
    # Check exponential backoff delays: wait1 = base_delay * 2^0 = 0.02, wait2 = base_delay * 2^1 = 0.04
    assert retries[0]["wait_s"] == pytest.approx(0.02)
    assert retries[1]["wait_s"] == pytest.approx(0.04)


async def test_compensate_fault_permanent_fails_immediately():
    system = MockSystem(latency=0.001)
    system.records["H-61"] = {"status": "active"}
    system.fail_next(1, kind="permanent")

    policy = RetryPolicy(max_attempts=4, base_delay=0.01)
    retries = []

    async def do_compensate():
        return await system.compensate("H-61", "key-perm-test", TODAY)

    with pytest.raises(PermanentError, match="400 Bad Request"):
        await retry_async(do_compensate, policy, on_retry=lambda a, e, w=None: retries.append(a))

    assert len(retries) == 0  # not retried!


async def test_compensate_hang_after_write_simulates_lost_ack_and_replays_cleanly():
    system = MockSystem(latency=0.001)
    system.records["H-62"] = {"status": "active"}
    # Hang after the write lands in state
    system.hang_next(1, seconds=2.0, after_write=True)

    # First attempt times out waiting for response
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(system.compensate("H-62", "hang-key-1", TODAY), timeout=0.05)

    # State was mutated
    assert system.records["H-62"]["status"] == "withdrawn"
    assert system.writes == 1
    assert system.compensations == 1

    # Retry with same idempotency key succeeds immediately without double-writing
    res_retry = await system.compensate("H-62", "hang-key-1", TODAY)
    assert res_retry["replayed"] is True
    assert system.writes == 1
    assert system.compensations == 1


# ==============================================================================
# 3. State Tracking on Connector
# ==============================================================================

async def test_connector_state_tracking_history():
    wd = Workday(latency=0.001)
    await wd.upsert("H-70", {"position": "SE", "status": "Active"}, "wd-up-1", {})

    state1 = wd.get_state("H-70")
    assert state1["status"] == "Active"
    assert len(state1["history"]) == 1
    assert state1["history"][0]["action"] == "upsert"

    await wd.compensate("H-70", "wd-comp-1", TODAY)
    state2 = wd.get_state("H-70")
    assert state2["status"] == "Rescinded"
    assert len(state2["history"]) == 2
    assert state2["history"][1]["from"] == "Active"
    assert state2["history"][1]["to"] == "Rescinded"
    assert state2["history"][1]["action"] == "hire_rescinded"


# ==============================================================================
# 4. Orchestrator Saga Compensation: Reverse Topological Order & State Tracking
# ==============================================================================

async def test_orchestrator_compensate_run_reverse_topological_order(tmp_path):
    app = make_app(str(tmp_path / "comp.db"))
    await app.start()

    # Ingest a new hire and let it propagate to all 5 systems
    await ingest(app, "H-100", hire(), "evt-comp-1")
    run = app.store.get_run("H-100:v1")
    assert run["status"] == "completed"

    # All systems are currently written
    assert app.systems["workday"].records["H-100"]["status"] == "active"
    assert app.systems["okta"].records["H-100"]["status"] == "STAGED"

    # Trigger compensation for the run
    res = await app.orchestrator.compensate_run("H-100:v1", reason="candidate_withdrew")
    assert res["status"] == "compensated"

    # Verify reverse topological order:
    # Topo levels are: [[expoit, workday], [okta], [lumos, tracker]]
    # Reverse execution must execute [lumos, tracker] before [okta], and [okta] before [expoit, workday]
    order = res["reverse_order"]
    lumos_idx = order.index("lumos")
    tracker_idx = order.index("tracker")
    okta_idx = order.index("okta")
    wd_idx = order.index("workday")
    expo_idx = order.index("expoit")

    assert max(lumos_idx, tracker_idx) < okta_idx
    assert okta_idx < min(wd_idx, expo_idx)

    # Verify all connector states were updated
    assert app.systems["workday"].records["H-100"]["status"] == "Rescinded"
    assert app.systems["okta"].records["H-100"]["status"] == "DEPROVISIONED"
    assert app.systems["lumos"].records["H-100"]["status"] == "revoked"
    assert app.systems["expoit"].records["H-100"]["status"] == "cancelled"
    assert app.systems["tracker"].records["H-100"]["status"] == "withdrawn"

    # Verify run status is COMPENSATED in store
    run_after = app.store.get_run("H-100:v1")
    assert run_after["status"] == "compensated"
    assert run_after["state"]["compensation_status"] == "compensated"
    assert all(c["status"] == "compensated" for c in run_after["state"]["compensations"].values())

    # Verify side_effects table has records of compensation
    side_effects = app.store.side_effects_for_run("H-100:v1")
    compensated_se = [s for s in side_effects if s["status"] == "compensated"]
    assert len(compensated_se) == 5

    await app.stop()


async def test_orchestrator_compensate_run_with_retry_and_exponential_backoff(tmp_path):
    app = make_app(str(tmp_path / "comp_retry.db"))
    await app.start()

    await ingest(app, "H-101", hire(), "evt-comp-2")
    assert app.store.get_run("H-101:v1")["status"] == "completed"

    # Inject transient failures on Lumos during compensation
    app.systems["lumos"].fail_next(2, kind="transient")

    res = await app.orchestrator.compensate_run("H-101:v1", reason="offer_rescinded")
    assert res["status"] == "compensated"

    lumos_res = res["compensations"]["lumos"]
    assert lumos_res["status"] == "compensated"
    assert lumos_res["attempts"] == 3
    assert len(lumos_res["retries"]) == 2
    # Verify exponential backoff strategy is tracked
    assert all(r["strategy"] == "exponential_backoff" for r in lumos_res["retries"])

    await app.stop()


async def test_cancel_hire_and_tool_execution(tmp_path):
    app = make_app(str(tmp_path / "cancel.db"))
    await app.start()

    await ingest(app, "H-102", hire(), "evt-comp-3")
    assert app.store.get_run("H-102:v1")["status"] == "completed"

    # Execute cancel_hire via toolreg
    out, err = await toolreg.execute(app, "compensate_run", {"run_id": "H-102:v1", "reason": "background_check_failed"},
                                    principal=PEOPLE_OPS)
    assert not err
    res = json.loads(out)
    assert res["status"] == "compensated"

    # Verify hire cancellation state in store
    await app.orchestrator.cancel_hire("H-102", reason="declined_offer", principal=PEOPLE_OPS)
    assert app.store.hire_status("H-102") == "cancelled"

    await app.stop()


# ==============================================================================
# 5. Workflow Engine Saga Rollback with Exponential Backoff
# ==============================================================================

async def test_workflow_rollback_with_exponential_backoff():
    clock = FakeClock()
    store = None

    class DummyEngine:
        def __init__(self):
            self.clock = clock
            self.app = None

    wf_data = {"activities": {}, "compensations": []}
    ctx = Ctx(DummyEngine(), {"wf_id": "wf-1", "key": "k1", "data": wf_data}, [])

    # Record two activities with compensating actions
    await ctx.activity("step1", lambda: asyncio.sleep(0.001), compensate={"system": "sys1", "param": 10})
    await ctx.activity("step2", lambda: asyncio.sleep(0.001), compensate={"system": "sys2", "param": 20})

    calls = []
    attempts_seen = []

    async def h_sys2(payload):
        calls.append("undo_sys2")
        return {"undone": "sys2"}

    fail_count = 2

    async def h_sys1(payload):
        nonlocal fail_count
        attempts_seen.append(1)
        if fail_count > 0:
            fail_count -= 1
            raise TransientError("sys1 temporary error")
        calls.append("undo_sys1")
        return {"undone": "sys1"}

    handlers = {
        "step2": h_sys2,
        "step1": h_sys1,
    }

    results = await ctx.rollback(handlers=handlers, retry_policy=RetryPolicy(max_attempts=4, base_delay=0.01))

    # Reverse order: step2 undone before step1
    assert calls == ["undo_sys2", "undo_sys1"]
    assert results[0]["activity"] == "step2"
    assert results[0]["status"] == "succeeded"
    assert results[1]["activity"] == "step1"
    assert results[1]["status"] == "succeeded"
    assert results[1]["attempts"] == 3
    assert len(results[1]["retries"]) == 2
    assert ctx.data["compensation_state"]["status"] == "completed"
