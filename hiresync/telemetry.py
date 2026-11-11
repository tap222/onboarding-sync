"""OpenTelemetry: traces + metrics for error analysis, latency, tokens and cost.

Setup (`setup_telemetry`):
  * OTEL_EXPORTER_OTLP_ENDPOINT set -> OTLP/HTTP exporters (traces + metrics), batch
    processor. Any OTLP backend: Datadog, Honeycomb, Grafana, Langfuse, ... .
  * otherwise -> in-memory span exporter + in-memory metric reader, which the demo,
    the tests and `error_report()` read. HIRESYNC_OTEL_CONSOLE=1 also prints spans.
  * Registered globally *before* the first Claude call, so the Anthropic SDK's own
    spans (anthropic.messages.create, CLIENT, with gen_ai.usage.*) nest under ours.
    The SDK runs in its default metadata_only mode: no prompts or responses leave
    the process - they contain hire data.

Span tree of one change (indentation = parent/child):

  hiresync.ingest                         submit_change (webhook / MCP / GraphQL)
    └ (traceparent carried in the outbox row)
  hiresync.event hire.changed             worker; links to the producer's trace context
    hiresync.run.validate
    hiresync.run.triage
      hiresync.llm triage                 one structured call: all turns, re-asks, eval retries
        anthropic.messages.create         Anthropic SDK span (CLIENT), per turn
        hiresync.tool get_hire            tool calls made by the agent
    hiresync.run.propagate
      hiresync.system.write okta          one per attempt: timeout / breaker / error.type
    hiresync.run.reconcile
  hiresync.event webhook.deliver          outbound webhook (traceparent header sent on)

Token counting: the SDK span carries gen_ai.usage.* per request. Our hiresync.llm span
aggregates a whole tool loop under *different* names (hiresync.llm.*), so backends
summing gen_ai.usage.* count each token exactly once.

Metrics (instruments below): tokens, cost, model calls by outcome, system calls by
outcome and error type with latency histograms, events processed, dead letters,
runs by status, authz denials, budget blocks.
"""
from __future__ import annotations

import json
import logging
import os
import statistics
from collections import Counter, defaultdict
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from opentelemetry import metrics, propagate, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Link, SpanKind, Status, StatusCode

log = logging.getLogger(__name__)
SCOPE = "hiresync"

_state: dict[str, Any] = {"configured": False, "spans": None, "metrics": None}


def setup_telemetry(service_name: str = "hiresync", force_memory: bool = False) -> dict:
    """Idempotent. Returns {"spans": InMemorySpanExporter|None, "metrics": InMemoryMetricReader|None}."""
    if _state["configured"]:
        return _state
    resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", service_name),
                                "service.version": "0.3.0",
                                "deployment.environment": os.getenv("HIRESYNC_ENV", "dev")})
    tp = TracerProvider(resource=resource)
    readers = []
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint and not force_memory:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))  # reads OTEL_EXPORTER_OTLP_* env
        readers.append(PeriodicExportingMetricReader(OTLPMetricExporter(), export_interval_millis=15_000))
    else:
        mem = InMemorySpanExporter()
        tp.add_span_processor(SimpleSpanProcessor(mem))
        _state["spans"] = mem
        reader = InMemoryMetricReader()
        readers.append(reader)
        _state["metrics"] = reader
    if os.getenv("HIRESYNC_OTEL_CONSOLE") == "1":
        tp.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(tp)
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=readers))
    _state["configured"] = True
    _init_instruments()
    return _state


def tracer() -> trace.Tracer:
    return trace.get_tracer(SCOPE)


# -------------------------------------------------------------- instruments
M: dict[str, Any] = {}


def _init_instruments() -> None:
    m = metrics.get_meter(SCOPE)
    M["llm_tokens"] = m.create_counter("hiresync.llm.tokens", unit="{token}",
                                       description="Model tokens by model, purpose, prompt, direction")
    M["llm_cost"] = m.create_counter("hiresync.llm.cost", unit="USD", description="Model spend")
    M["llm_calls"] = m.create_counter("hiresync.llm.calls", description="Structured model calls by outcome")
    M["llm_duration"] = m.create_histogram("hiresync.llm.duration", unit="ms")
    M["sys_calls"] = m.create_counter("hiresync.system.calls", description="Downstream attempts by outcome/error")
    M["sys_duration"] = m.create_histogram("hiresync.system.duration", unit="ms")
    M["events"] = m.create_counter("hiresync.events.processed", description="Queue deliveries by topic/outcome")
    M["event_duration"] = m.create_histogram("hiresync.event.duration", unit="ms")
    M["dlq"] = m.create_counter("hiresync.dlq.dead", description="Dead letters by kind/topic")
    M["runs"] = m.create_counter("hiresync.runs.finished", description="Runs by final status")
    M["authz_denied"] = m.create_counter("hiresync.authz.denied", description="Authorization denials by tool")
    M["budget_blocked"] = m.create_counter("hiresync.budget.blocked", description="Model calls refused by guardrail")


def count(name: str, value: float = 1, **attrs) -> None:
    if name in M:
        M[name].add(value, {k: v for k, v in attrs.items() if v is not None})


def observe(name: str, value: float, **attrs) -> None:
    if name in M:
        M[name].record(value, {k: v for k, v in attrs.items() if v is not None})


# ------------------------------------------------------------------- spans
@contextmanager
def span(name: str, kind: SpanKind = SpanKind.INTERNAL, links: Optional[list] = None,
         **attrs) -> Iterator[trace.Span]:
    """A span that records exceptions with error.type and ERROR status, and re-raises."""
    with tracer().start_as_current_span(name, kind=kind, links=links or [],
                                        record_exception=False, set_status_on_exception=False) as sp:
        for k, v in attrs.items():
            if v is not None:
                sp.set_attribute(k, v)
        try:
            yield sp
        except BaseException as exc:
            mark_error(sp, exc)
            raise


def mark_error(sp: trace.Span, exc: BaseException, expected: bool = False) -> None:
    sp.set_attribute("error.type", type(exc).__name__)
    sp.record_exception(exc)
    if not expected:
        sp.set_status(Status(StatusCode.ERROR, str(exc)[:200]))


# ---------------------------------------------- context through the queue
def inject_carrier() -> dict:
    """W3C trace context of the current span, to store with an outbox row."""
    carrier: dict = {}
    propagate.inject(carrier)
    return carrier


def links_from(carrier: Optional[dict]) -> list:
    """The consumer span *links* to the producer instead of becoming its child: a
    queued event can be handled minutes later or retried several times, which
    span links (not parent/child) model correctly per the messaging conventions."""
    if not carrier:
        return []
    ctx = propagate.extract(carrier)
    sc = trace.get_current_span(ctx).get_span_context()
    return [Link(sc)] if sc.is_valid else []


def current_trace_id() -> Optional[str]:
    sc = trace.get_current_span().get_span_context()
    return format(sc.trace_id, "032x") if sc.is_valid else None


class TraceContextLogFilter(logging.Filter):
    """Adds trace_id/span_id to log records so logs and traces join up."""

    def filter(self, record: logging.LogRecord) -> bool:
        sc = trace.get_current_span().get_span_context()
        record.trace_id = format(sc.trace_id, "032x") if sc.is_valid else "-"
        record.span_id = format(sc.span_id, "016x") if sc.is_valid else "-"
        return True


# ------------------------------------------------------------ error analysis
def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return round(values[min(len(values) - 1, int(p * len(values)))], 1)


def error_report(spans: Optional[list] = None) -> dict:
    """Error analysis from finished spans (in-memory exporter, or spans exported
    from a backend in the same shape). Production does this in the backend; this
    function is the same analysis, runnable offline and in CI."""
    if spans is None:
        mem = _state.get("spans")
        spans = list(mem.get_finished_spans()) if mem else []
    errors: Counter = Counter()
    examples: dict = {}
    sys_latency: dict[str, list[float]] = defaultdict(list)
    sys_outcomes: dict[str, Counter] = defaultdict(Counter)
    llm: dict[str, dict] = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                                                 "reasks": 0, "eval_retries": 0, "errors": Counter()})
    sdk_tokens = {"input": 0, "output": 0, "requests": 0}
    events: Counter = Counter()
    for s in spans:
        a = dict(s.attributes or {})
        dur = (s.end_time - s.start_time) / 1e6 if s.end_time else 0.0
        etype = a.get("error.type")
        if etype:
            key = (s.name.split(" ")[0], a.get("hiresync.system") or a.get("hiresync.purpose")
                   or a.get("messaging.destination.name") or "-", etype)
            errors[key] += 1
            if key not in examples:
                ev = next((e for e in s.events if e.name == "exception"), None)
                examples[key] = (ev.attributes.get("exception.message", "") if ev else "")[:120]
        if s.name.startswith("hiresync.system.write"):
            name = a.get("hiresync.system", "?")
            sys_latency[name].append(dur)
            sys_outcomes[name][a.get("hiresync.outcome", "?")] += 1
        elif s.name.startswith("hiresync.llm"):
            k = f"{a.get('hiresync.purpose')}:{a.get('hiresync.prompt')}"
            d = llm[k]
            d["calls"] += 1
            d["input_tokens"] += a.get("hiresync.llm.input_tokens", 0)
            d["output_tokens"] += a.get("hiresync.llm.output_tokens", 0)
            d["cost_usd"] = round(d["cost_usd"] + a.get("hiresync.llm.cost_usd", 0.0), 6)
            d["reasks"] += a.get("hiresync.llm.reasks", 0)
            d["eval_retries"] += a.get("hiresync.llm.eval_retries", 0)
            if etype:
                d["errors"][etype] += 1
        elif s.name == "anthropic.messages.create" and s.kind == SpanKind.CLIENT:
            sdk_tokens["requests"] += 1
            sdk_tokens["input"] += a.get("gen_ai.usage.input_tokens", 0)
            sdk_tokens["output"] += a.get("gen_ai.usage.output_tokens", 0)
        elif s.name.startswith("hiresync.event"):
            events[(a.get("messaging.destination.name"), a.get("hiresync.outcome"))] += 1
    return {
        "spans": len(spans),
        "errors": [{"span": k[0], "where": k[1], "error.type": k[2], "count": n, "example": examples[k]}
                   for k, n in errors.most_common()],
        "systems": {n: {"attempts": len(v), "p50_ms": _pct(v, .5), "p95_ms": _pct(v, .95),
                        "outcomes": dict(sys_outcomes[n])} for n, v in sorted(sys_latency.items())},
        "llm": {k: {**v, "errors": dict(v["errors"])} for k, v in llm.items()},
        "anthropic_sdk_spans": sdk_tokens,
        "events": [{"topic": t, "outcome": o, "count": n} for (t, o), n in sorted(events.items(), key=str)],
    }


def metric_snapshot() -> dict:
    """Current metric sums from the in-memory reader: {metric: {attrs_json: value}}."""
    reader = _state.get("metrics")
    if not reader:
        return {}
    data = reader.get_metrics_data()
    out: dict = {}
    for rm in data.resource_metrics if data else []:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                pts = {}
                for p in metric.data.data_points:
                    key = json.dumps(dict(p.attributes), sort_keys=True)
                    pts[key] = getattr(p, "value", None) if hasattr(p, "value") else {
                        "count": p.count, "sum": round(p.sum, 2)}
                out[metric.name] = pts
    return out


def reset_memory() -> None:
    mem = _state.get("spans")
    if mem:
        mem.clear()
