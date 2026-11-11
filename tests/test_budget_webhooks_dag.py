"""Token/cost guardrails, webhooks (both directions) and the dependency DAG."""
import json
import time
from types import SimpleNamespace

import pytest

from conftest import AUDITOR, PEOPLE_OPS, RECRUITER, hire, ingest, make_app
from hiresync import tools as toolreg
from hiresync.agent import TriageAgent
from hiresync.budget import BudgetExceeded, CostGuard, Limits, price
from hiresync.connectors import DependencyGraphError, default_systems, topo_levels
from hiresync.memory import WorkingMemory
from hiresync.store import Store
from hiresync.webhooks import RecordingTransport, check_url, sign, verify

ISSUE = [{"code": "COUNTRY_MISMATCH", "severity": "warn", "message": "m"}]


class CheapFake:
    def __init__(self, input_tokens=2000, output_tokens=300):
        self.calls = 0
        self.usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens, cache_read_input_tokens=0)

    async def create(self, **kw):
        self.calls += 1
        body = {"action": "hold_for_approval", "confidence": 0.7, "rationale": "COUNTRY_MISMATCH, no precedent",
                "evidence": ["issue:COUNTRY_MISMATCH"], "proposed_patch": [], "approver_message": "H1: please review"}
        return SimpleNamespace(stop_reason="end_turn", stop_details=None, usage=self.usage,
                               content=[SimpleNamespace(type="text", text=json.dumps(body))])


def claude_agent(app, fake):
    agent = TriageAgent(mode="heuristic")
    agent.use_claude, agent.app = True, app
    agent.client = SimpleNamespace(beta=SimpleNamespace(messages=fake))
    return agent


def wm(run="H1:v2", user_blob=""):
    return WorkingMemory(run_id=run, hire_id="H1", from_version=1, to_version=2, issues=ISSUE,
                         new={"note": user_blob} if user_blob else {})


# ------------------------------------------------------------ token / cost
def test_price_uses_model_table_and_refuses_unknown_models():
    assert price("claude-opus-5-5", {"input_tokens": 1_000_000, "output_tokens": 100_000}) == pytest.approx(6.0)
    with pytest.raises(BudgetExceeded, match="UNKNOWN_MODEL_PRICE"):
        price("some-new-model", {"input_tokens": 1})


async def test_oversized_input_is_refused_before_any_call(tmp_path):
    app = make_app(str(tmp_path / "t.db"), limits=Limits(max_input_tokens_per_call=500))
    await app.start()
    fake = CheapFake()
    w = wm()
    w.changes = [{"path": "x", "op": "replace", "old": "", "new": "word " * 2000}]  # huge diff
    rec, _ = await claude_agent(app, fake).triage(w, [])
    assert fake.calls == 0 and rec.source == "heuristic" and "INPUT_TOO_LARGE" in rec.rationale
    await app.stop()


async def test_per_run_cost_cap_stops_further_calls(tmp_path):
    app = make_app(str(tmp_path / "t.db"), limits=Limits(max_usd_per_run=0.25))
    await app.start()
    fake = CheapFake(input_tokens=20_000, output_tokens=2_000)  # $0.12 per call
    agent = claude_agent(app, fake)
    results = [(await agent.triage(wm("H1:v2"), []))[0] for _ in range(3)]
    assert [r.source for r in results] == ["claude", "heuristic", "heuristic"]
    assert "RUN_COST" in results[1].rationale
    tokens, usd = app.store.llm_spend_for_run("H1:v2")
    assert usd == pytest.approx(0.12) and tokens == 22_000
    await app.stop()


async def test_daily_cap_alerts_once_and_falls_back(tmp_path):
    app = make_app(str(tmp_path / "t.db"), limits=Limits(max_usd_per_day=0.20, max_usd_per_run=10))
    await app.start()
    sub = app.webhooks.subscribe("https://hooks.zapier.com/x", ["budget.daily_exceeded"], "u-maya")
    agent = claude_agent(app, CheapFake(input_tokens=20_000, output_tokens=2_000))
    sources = [(await agent.triage(wm(f"H1:v{i}"), []))[0].source for i in range(4)]
    await app.drain()
    assert sources[0] == "claude" and sources[-1] == "heuristic"
    alerts = [m for m in app.slack.messages if "budget.daily_exceeded" in m["text"]]
    assert len(alerts) == 1  # once per day, not per blocked call
    assert [r["body"]["type"] for r in app.webhooks.transport.received] == ["budget.daily_exceeded"]
    spend, _ = await toolreg.execute(app, "get_model_spend", {}, principal=AUDITOR)
    assert json.loads(spend)["today_usd"] == pytest.approx(0.12)
    await app.stop()


async def test_concurrency_slot_limits_parallel_model_calls():
    import asyncio
    guard = CostGuard(Store(":memory:"), Limits(max_concurrent_calls=2))
    active, peak = 0, 0

    async def call():
        nonlocal active, peak
        async with guard.slot:
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.gather(*(call() for _ in range(8)))
    assert peak == 2


# ---------------------------------------------------------------- webhooks
def test_signature_roundtrip_and_replay_window():
    body = b'{"a":1}'
    header = sign("s3cret", body)
    assert verify("s3cret", body, header)
    assert not verify("other", body, header)
    assert not verify("s3cret", b'{"a":2}', header)
    assert not verify("s3cret", body, sign("s3cret", body, ts=int(time.time()) - 3600))  # replayed


@pytest.mark.parametrize("url", ["http://hooks.zapier.com/x", "https://10.0.0.5/hook", "https://localhost/x",
                                 "https://evil.example.com/x", "https://metadata.internal/x"])
def test_webhook_urls_are_allowlisted(url):
    with pytest.raises(ValueError):
        check_url(url)


async def test_events_are_delivered_signed_and_retried(tmp_path):
    transport = RecordingTransport()
    app = make_app(str(tmp_path / "t.db"), webhook_transport=transport, queue_backoff_s=0.01)
    await app.start()
    out, err = await toolreg.execute(app, "create_webhook_subscription",
                                     {"url": "https://acme.app.n8n.cloud/webhook/hiresync",
                                      "events": ["run.completed", "approval.requested"]}, principal=PEOPLE_OPS)
    sub = json.loads(out)
    assert not err and sub["secret"].startswith("whsec_")
    _, err = await toolreg.execute(app, "create_webhook_subscription",
                                   {"url": "https://hooks.zapier.com/x", "events": ["run.completed"]},
                                   principal=RECRUITER)
    assert err  # needs webhooks:manage
    transport.fail_next = [503, 503]  # receiver flaps, then recovers
    await ingest(app, "H1", hire(), "e1")
    await app.drain()
    got = transport.received
    assert [g["body"]["type"] for g in got] == ["run.completed"]
    raw = json.dumps(got[0]["body"], sort_keys=True).encode()
    assert verify(sub["secret"], raw, got[0]["headers"]["HireSync-Signature"])
    assert got[0]["body"]["data"]["systems"]["okta"] == "succeeded"
    codes = [d["status_code"] for d in app.store.webhook_deliveries(sub["id"])]
    assert codes == [503, 503, 200]
    await app.stop()


async def test_rejected_webhook_dead_letters_without_feedback_loop(tmp_path):
    transport = RecordingTransport()
    app = make_app(str(tmp_path / "t.db"), webhook_transport=transport, queue_backoff_s=0.01)
    await app.start()
    app.webhooks.subscribe("https://hooks.zapier.com/x", ["run.completed", "dlq.dead"], "u-maya")
    transport.fail_next = [410]  # receiver gone: permanent
    await ingest(app, "H1", hire(), "e1")
    await app.drain()
    dead = app.store.dead_events()
    assert len(dead) == 1 and dead[0]["topic"] == "webhook.deliver" and dead[0]["attempts"] == 1
    assert "permanent" in dead[0]["last_error"]
    assert not any(r["body"]["type"] == "dlq.dead" for r in transport.received)  # no webhook-about-webhook
    await app.stop()


# --------------------------------------------------------------------- DAG
def test_dependency_graph_is_acyclic_and_cycles_are_named():
    systems = default_systems()
    assert topo_levels(systems) == [["expoit", "workday"], ["okta"], ["lumos", "tracker"]]
    systems["workday"].depends_on = ("lumos",)  # workday -> lumos -> okta -> workday
    with pytest.raises(DependencyGraphError, match="okta -> workday -> lumos -> okta|lumos -> okta -> workday -> lumos"):
        topo_levels(systems)
    systems = default_systems()
    systems["tracker"].depends_on = ("salesforce",)
    with pytest.raises(DependencyGraphError, match="unknown"):
        topo_levels(systems)


async def test_evaluator_rejects_a_patch_that_does_not_fix_the_issue(app):
    from hiresync.agent import Recommendation
    from hiresync.evaluator import deterministic_checks
    await ingest(app, "H1", hire(), "e1")
    w = WorkingMemory(run_id="r", hire_id="H1", from_version=1, to_version=2, new=app.store.get_version("H1")["data"],
                      issues=[{"code": "POSTAL_CODE_INVALID", "severity": "block", "message": "m",
                               "field": "shipping_address.postal_code"}])
    base = dict(action="hold_for_approval", confidence=0.9, rationale="POSTAL_CODE_INVALID",
                evidence=["issue:POSTAL_CODE_INVALID"], approver_message="H1: fix postal code")
    w.new = {**w.new, "shipping_address": {**w.new["shipping_address"], "postal_code": "7870"}}
    still_bad = Recommendation(**base, proposed_patch=[{"path": "shipping_address.postal_code", "value": "787"}])
    good = Recommendation(**base, proposed_patch=[{"path": "shipping_address.postal_code", "value": "78701"}])
    assert any("does not resolve" in p for p in deterministic_checks(still_bad, w, app))
    assert deterministic_checks(good, w, app) == []
