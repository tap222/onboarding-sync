"""Composition root: wires store, Redis idempotency, bus, outbox relay, systems,
agent, orchestrator and document service. Used by the API, the MCP server, the
demo and the tests."""
from __future__ import annotations

import asyncio
import logging

from opentelemetry.trace import SpanKind
from datetime import date
from typing import Callable, Optional

from pydantic import ValidationError

from . import telemetry
from .agent import TriageAgent
from .connectors import MockSystem, SlackNotifier, default_systems
from .documents import DocumentService
from .events import DbQueueBus, OutboxRelay, make_bus
from .idempotency import IN_FLIGHT, make_idempotency
from .memory import EpisodicMemory
from .models import canonicalize
from .orchestrator import Orchestrator
from .budget import CostGuard, Limits
from .evaluator import Evaluator
from .resilience import Timeouts
from .webhooks import Webhooks
from .store import Store
from .vectors import VectorStore
from .versioning import deep_merge

log = logging.getLogger(__name__)


class IngestError(ValueError):
    pass


class HireSync:
    def __init__(self, db_path: str = ":memory:", systems: Optional[dict[str, MockSystem]] = None,
                 today: Optional[Callable[[], date]] = None, agent_mode: Optional[str] = None,
                 redis=None, bus=None, slack: Optional[SlackNotifier] = None,
                 timeouts: Optional[Timeouts] = None, workers: int = 4, queue_backoff_s: float = 0.5,
                 max_deliveries: int = 5, sweep_interval_s: float = 30.0, limits: Optional[Limits] = None,
                 webhook_transport=None):
        telemetry.setup_telemetry()  # before any Claude client exists, so SDK spans nest under ours
        self.timeouts = timeouts or Timeouts()
        self.sweep_interval_s = sweep_interval_s
        self._sweeper: Optional[asyncio.Task] = None
        self.store = Store(db_path)
        self.systems = systems if systems is not None else default_systems()
        self.slack = slack or SlackNotifier()
        self.idem = make_idempotency(self.store, redis)  # DB by default; Redis only if given / REDIS_URL
        self.bus = bus if bus is not None else make_bus(  # DB-backed worker pool by default
            self.store, workers=workers, lease_s=self.timeouts.lease, handler_timeout_s=self.timeouts.handler,
            max_deliveries=max_deliveries, backoff_base_s=queue_backoff_s, on_dead=self._on_dead_event)
        # The DB queue reads the outbox directly; other buses need the relay.
        self.relay = None if isinstance(self.bus, DbQueueBus) else OutboxRelay(self.store, self.bus)
        self.vectors = VectorStore(self.store)
        self.cost_guard = CostGuard(self.store, limits, alert=self._budget_alert)
        self.evaluator = Evaluator(self)
        self.webhooks = Webhooks(self, webhook_transport)
        self.agent = TriageAgent(agent_mode)
        self.agent.app = self
        self.agent.timeout_s = self.timeouts.claude
        self.episodic = EpisodicMemory(self.store, self.vectors)
        self.orchestrator = Orchestrator(self, today=today or date.today, timeouts=self.timeouts)
        self.documents = DocumentService(self, self.idem)
        self.bus.subscribe("hire.changed", "orchestrator", self.orchestrator.on_hire_changed)
        self.bus.subscribe("run.resume", "orchestrator", self.orchestrator.on_run_resume)
        self.bus.subscribe("webhook.deliver", "webhooks", self.webhooks.deliver)

    async def start(self, relay_loop: bool = True) -> None:
        self.vectors.rebuild()  # HNSW indexes are derived from the DB
        await self.bus.start()
        if relay_loop and self.relay:
            self.relay.start()
        recovered = await self.orchestrator.recover()
        if recovered:
            log.info("recovered %d in-flight runs from checkpoints", recovered)
        if self.sweep_interval_s:
            self._sweeper = asyncio.create_task(self._sweep_loop())

    async def _sweep_loop(self) -> None:
        while True:
            await asyncio.sleep(self.sweep_interval_s)
            try:
                await self.orchestrator.sweep()
            except Exception:  # noqa: BLE001
                log.exception("sweep failed")

    async def _on_dead_event(self, row: dict) -> None:
        await self.slack.post("#people-ops-alerts",
                              f"☠ event #{row['id']} ({row['topic']}, {row['key']}) moved to the dead-letter queue "
                              f"after {row['attempts'] + 1} attempts: {row['last_error'][:160]}")
        if row["topic"] != "webhook.deliver":  # a dead webhook must not emit webhooks: no feedback loop
            self.webhooks.emit("dlq.dead", {"outbox_id": row["id"], "topic": row["topic"], "key": row["key"],
                                            "error": row["last_error"][:300]})

    async def _budget_alert(self, kind: str, data: dict) -> None:
        await self.slack.post("#people-ops-alerts", f"💸 {kind}: {data}")
        self.webhooks.emit(kind, data)

    async def stop(self) -> None:
        if self._sweeper:
            self._sweeper.cancel()
            await asyncio.gather(self._sweeper, return_exceptions=True)
            self._sweeper = None
        if self.relay:
            await self.relay.stop()
        await self.bus.stop()

    async def drain(self) -> None:
        """Run until every committed event has been handled (demo, tests, MCP)."""
        for _ in range(100):
            if self.relay:
                await self.relay.flush()
            await self.bus.drain()
            pending = self.store.pending_events(limit=1) if not self.relay else self.store.unpublished()
            if not pending:
                return

    async def compensate_run(self, run_id: str, reason: str = "cancelled") -> dict:
        return await self.orchestrator.compensate_run(run_id, reason=reason)

    async def cancel_hire(self, hire_id: str, reason: str = "cancelled",
                          principal: Optional[Principal] = None) -> dict:
        return await self.orchestrator.cancel_hire(hire_id, reason=reason, principal=principal)

    async def submit_change(self, hire_id: str, patch: dict, *, source: str, actor: str,
                            event_id: Optional[str], expected_version: Optional[int] = None,
                            reason: Optional[str] = None) -> dict:
        """Ingest a change from Ashby, Workday (off-cycle) or a human.

        Dedupe: Redis claim on event_id (fast, cross-worker) then the DB's
        inbound_events primary key (authoritative) inside append_version's transaction."""
        with telemetry.span("hiresync.ingest", kind=SpanKind.SERVER, **{
                "hiresync.hire_id": hire_id, "hiresync.source": source, "hiresync.event_id": event_id}) as sp:
            res = await self._submit_change(hire_id, patch, source=source, actor=actor, event_id=event_id,
                                            expected_version=expected_version, reason=reason)
            sp.set_attribute("hiresync.version", res.get("version") or 0)
            sp.set_attribute("hiresync.duplicate", bool(res.get("duplicate")))
            return res

    async def _submit_change(self, hire_id: str, patch: dict, *, source: str, actor: str,
                             event_id: Optional[str], expected_version: Optional[int] = None,
                             reason: Optional[str] = None) -> dict:
        if event_id:
            owned, cached = await self.idem.claim("event", event_id)
            if not owned:
                if cached == IN_FLIGHT:
                    return {"hire_id": hire_id, "status": "in_flight", "duplicate": True}
                return {**cached, "duplicate": True}
        try:
            current = self.store.get_version(hire_id)
            record = deep_merge(current["data"], patch) if current else {"hire_id": hire_id, **patch}
            try:
                record = canonicalize(record)  # structural check + defaults at the boundary
            except ValidationError as exc:
                raise IngestError(f"invalid record for {hire_id}: {exc.errors()[0]['msg']} "
                                  f"at {'.'.join(map(str, exc.errors()[0]['loc']))}") from exc
            # The OTel trace id IS the pipeline's trace_id: search the backend with it.
            res = self.store.append_version(hire_id, record, source=source, actor=actor, event_id=event_id,
                                            expected_version=expected_version, trace_id=telemetry.current_trace_id())
            if reason:
                self.store.audit(hire_id, None, actor, "change.reason", {"reason": reason, "version": res["version"]})
            result = {"hire_id": hire_id, **res}
            if event_id:
                await self.idem.complete("event", event_id, result)
            return result
        except Exception:
            if event_id:
                await self.idem.release("event", event_id)
            raise


