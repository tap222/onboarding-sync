"""OpenTelemetry: span tree, trace propagation through the queue, error analysis,
token/cost attributes and metrics - including the real Anthropic SDK's own spans."""
import json

import httpx2 as httpx  # the Anthropic SDK 2.x HTTP client
import pytest
from opentelemetry.trace import SpanKind

from conftest import MANAGER, hire, ingest, make_app
from hiresync import telemetry
from hiresync import tools as toolreg
from hiresync.agent import TriageAgent
from hiresync.connectors import default_systems
from hiresync.memory import WorkingMemory
from hiresync.resilience import Timeouts
from hiresync.webhooks import RecordingTransport


@pytest.fixture(autouse=True)
def fresh_spans():
    telemetry.setup_telemetry(force_memory=True)
    telemetry.reset_memory()
    yield


def spans():
    return list(telemetry._state["spans"].get_finished_spans())


def by_name(prefix):
    return [s for s in spans() if s.name.startswith(prefix)]


async def test_span_tree_and_trace_id_follow_a_change_through_the_queue(app):
    await ingest(app, "H1", hire(), "e1")
    ingest_span = by_name("hiresync.ingest")[0]
    event = next(s for s in by_name("hiresync.event hire.changed"))
    assert event.kind == SpanKind.CONSUMER
    assert event.links and event.links[0].context.trace_id == ingest_span.context.trace_id  # linked to producer
    trace_hex = format(ingest_span.context.trace_id, "032x")
    assert app.store.get_run("H1:v1")["state"]["trace_id"] == trace_hex  # pipeline trace_id == OTel trace id
    steps = {s.name for s in spans() if s.context.trace_id == event.context.trace_id}
    assert {"hiresync.run.validate", "hiresync.run.triage", "hiresync.run.propagate", "hiresync.run.reconcile",
            "hiresync.system.write okta"} <= steps
    write = by_name("hiresync.system.write okta")[0]
    assert write.kind == SpanKind.CLIENT and write.attributes["hiresync.outcome"] == "ok"


async def test_error_analysis_groups_timeouts_by_system(tmp_path):
    systems = default_systems()
    app = make_app(str(tmp_path / "t.db"), systems=systems,
                   timeouts=Timeouts(call={}, call_default=0.1, retry_budget=2, handler=5, lease=10))
    await app.start()
    systems["expoit"].hang_next(10, seconds=5)
    systems["okta"].fail_next(1)  # one transient blip, recovered by retry
    await ingest(app, "H1", hire(), "e1")
    rep = telemetry.error_report()
    errs = {(e["span"], e["where"], e["error.type"]): e["count"] for e in rep["errors"]}
    assert errs[("hiresync.system.write", "expoit", "CallTimeout")] == 3
    assert errs[("hiresync.system.write", "okta", "TransientError")] == 1
    assert rep["systems"]["expoit"]["outcomes"] == {"timeout": 3}
    assert rep["systems"]["okta"]["outcomes"] == {"transient": 1, "ok": 1}
    assert rep["systems"]["expoit"]["p95_ms"] >= 100
    snap = telemetry.metric_snapshot()
    timeouts = sum(v for k, v in snap["hiresync.system.calls"].items() if '"outcome": "timeout"' in k)
    assert timeouts >= 3
    assert any('"kind": "write"' in k for k in snap["hiresync.dlq.dead"])
    await app.stop()


def anthropic_mock(input_tokens=1500, output_tokens=200):
    body = {"action": "hold_for_approval", "confidence": 0.7, "rationale": "COUNTRY_MISMATCH without precedent",
            "evidence": ["issue:COUNTRY_MISMATCH"], "proposed_patch": [], "approver_message": "H1: please review"}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        assert "traceparent" in request.headers  # SDK propagates our trace to the API
        return httpx.Response(200, headers={"request-id": "req_test123"}, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": json.dumps(body)}], "stop_reason": "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}})
    return handler


def total_cost() -> float:
    return sum(telemetry.metric_snapshot().get("hiresync.llm.cost", {}).values())


async def test_real_sdk_spans_nest_under_ours_with_usage_and_cost(app):
    import anthropic
    cost_before = total_cost()  # counters are cumulative for the process
    agent = TriageAgent(mode="heuristic")
    agent.use_claude, agent.app = True, app
    agent.client = anthropic.AsyncAnthropic(api_key="test-key", max_retries=0,
                                            http_client=httpx.AsyncClient(transport=httpx.MockTransport(anthropic_mock())))
    wm = WorkingMemory(run_id="H1:v2", hire_id="H1", from_version=1, to_version=2,
                       issues=[{"code": "COUNTRY_MISMATCH", "severity": "warn", "message": "m"}])
    rec, _ = await agent.triage(wm, [])
    assert rec.source == "claude"
    llm = by_name("hiresync.llm triage")[0]
    sdk = [s for s in spans() if s.name == "anthropic.messages.create"]
    assert sdk, "Anthropic SDK emitted no span"
    assert sdk[0].kind == SpanKind.CLIENT and sdk[0].parent.span_id == llm.context.span_id
    assert sdk[0].attributes["gen_ai.usage.input_tokens"] == 1500
    a = llm.attributes
    assert a["hiresync.llm.input_tokens"] == 1500 and a["hiresync.llm.output_tokens"] == 200
    assert a["hiresync.llm.cost_usd"] == pytest.approx((1500 * 4 + 200 * 20) / 1e6)
    assert a["hiresync.prompt"] == "triage@v1" and "gen_ai.usage.input_tokens" not in a  # no double counting
    rep = telemetry.error_report()
    assert rep["llm"]["triage:triage@v1"]["input_tokens"] == 1500
    assert rep["anthropic_sdk_spans"] == {"input": 1500, "output": 200, "requests": 1}
    assert total_cost() - cost_before == pytest.approx(0.01)


async def test_denials_and_budget_blocks_are_counted_not_alarmed(app):
    await ingest(app, "H1", hire(manager_email="sam@betterup.co"), "e1")
    _, err = await toolreg.execute(app, "get_hire", {"hire_id": "H1"}, principal=MANAGER)
    assert err
    tool = by_name("hiresync.tool get_hire")[-1]
    assert tool.attributes["error.type"] == "Forbidden" and tool.status.is_unset  # a decision, not an outage
    assert any('"tool": "get_hire"' in k for k in telemetry.metric_snapshot()["hiresync.authz.denied"])


async def test_outbound_webhooks_carry_traceparent(tmp_path):
    transport = RecordingTransport()
    app = make_app(str(tmp_path / "t.db"), webhook_transport=transport)
    await app.start()
    app.webhooks.subscribe("https://hooks.zapier.com/x", ["run.completed"], "u-maya")
    await ingest(app, "H1", hire(), "e1")
    await app.drain()
    assert transport.received[0]["headers"]["traceparent"].startswith("00-")
    await app.stop()
