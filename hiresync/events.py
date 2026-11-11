"""Event bus + transactional outbox relay.

Delivery contract (same for every bus): at-least-once, ordered per key (hire_id).
  * Per-key ordering: events for one hire are handled in order, so v3 is never
    processed before v2. Different hires run in parallel.
  * At-least-once: an event is acknowledged only after its handler succeeds;
    handlers are idempotent (run_id = hire:version), so redelivery is harmless.
  * Poison messages: retried a bounded number of times, then dead-lettered so one
    bad event can't block a hire forever.

DbQueueBus (default) - the outbox table is the queue. Existing stack only.
InMemoryBus (BUS=memory) - tests of bus mechanics; loses in-flight events on crash.
KafkaBus (BUS=kafka) - same contract, if a shared event stream is ever needed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
import zlib
from typing import Awaitable, Callable, Optional, Protocol

from opentelemetry.trace import SpanKind

from . import telemetry
from .resilience import PermanentError, RetryPolicy
from .store import Store

log = logging.getLogger(__name__)
Handler = Callable[[dict], Awaitable[None]]


class Bus(Protocol):
    async def publish(self, topic: str, key: str, value: dict) -> None: ...
    def subscribe(self, topic: str, group: str, handler: Handler) -> None: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def drain(self) -> None: ...


def partition_for(key: str, n: int) -> int:
    return zlib.crc32(key.encode()) % n  # stable across processes, unlike hash()


class InMemoryBus:
    def __init__(self, partitions: int = 4, max_deliveries: int = 3, redelivery_delay: float = 0.05):
        self.partitions = partitions
        self.max_deliveries = max_deliveries
        self.redelivery_delay = redelivery_delay
        self._queues: dict[str, list[asyncio.Queue]] = {}
        self._handlers: dict[str, list[tuple[str, Handler]]] = {}
        self._tasks: list[asyncio.Task] = []
        self.dead: list[dict] = []
        self.delivered = 0

    def _topic(self, topic: str) -> list[asyncio.Queue]:
        if topic not in self._queues:
            self._queues[topic] = [asyncio.Queue() for _ in range(self.partitions)]
        return self._queues[topic]

    async def publish(self, topic: str, key: str, value: dict) -> None:
        await self._topic(topic)[partition_for(key, self.partitions)].put({"key": key, "value": value})

    def subscribe(self, topic: str, group: str, handler: Handler) -> None:
        self._handlers.setdefault(topic, []).append((group, handler))
        self._topic(topic)

    async def start(self) -> None:
        for topic, handlers in self._handlers.items():
            for p, q in enumerate(self._queues[topic]):
                self._tasks.append(asyncio.create_task(self._consume(topic, p, q, handlers)))

    async def _consume(self, topic: str, p: int, q: asyncio.Queue, handlers) -> None:
        while True:
            msg = await q.get()
            try:
                for group, handler in handlers:
                    await self._deliver(topic, group, handler, msg)
            finally:
                q.task_done()

    async def _deliver(self, topic: str, group: str, handler: Handler, msg: dict) -> None:
        value = {k: v for k, v in msg["value"].items() if k != "_otel"}
        for attempt in range(1, self.max_deliveries + 1):
            try:
                await handler(value)
                self.delivered += 1
                return
            except Exception as exc:  # noqa: BLE001 - the bus must survive any handler bug
                log.warning("%s/%s delivery %d failed: %s", topic, group, attempt, exc)
                if attempt == self.max_deliveries:
                    self.dead.append({"topic": topic, "group": group, "msg": msg, "error": repr(exc)})
                    return
                await asyncio.sleep(self.redelivery_delay * attempt)

    async def drain(self) -> None:
        """Wait until every partition is empty and idle (used by the demo and tests)."""
        while True:
            for queues in self._queues.values():
                for q in queues:
                    await q.join()
            if all(q.empty() for qs in self._queues.values() for q in qs):
                return

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()


class KafkaBus:
    """Same contract on Kafka: idempotent producer (acks=all), keyed by hire_id,
    manual commits after the handler succeeds, DLQ topic for poison messages."""

    def __init__(self, bootstrap: str, max_deliveries: int = 3):
        self.bootstrap = bootstrap
        self.max_deliveries = max_deliveries
        self._subs: list[tuple[str, str, Handler]] = []
        self._consumers = []
        self._tasks: list[asyncio.Task] = []
        self._producer = None

    async def start(self) -> None:
        from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

        self._producer = AIOKafkaProducer(bootstrap_servers=self.bootstrap, enable_idempotence=True, acks="all")
        await self._producer.start()
        for topic, group, handler in self._subs:
            consumer = AIOKafkaConsumer(topic, bootstrap_servers=self.bootstrap, group_id=group,
                                        enable_auto_commit=False, auto_offset_reset="earliest")
            await consumer.start()
            self._consumers.append(consumer)
            self._tasks.append(asyncio.create_task(self._consume(consumer, topic, handler)))

    async def publish(self, topic: str, key: str, value: dict) -> None:
        assert self._producer, "start() first"
        await self._producer.send_and_wait(topic, json.dumps(value).encode(), key=key.encode())

    def subscribe(self, topic: str, group: str, handler: Handler) -> None:
        self._subs.append((topic, group, handler))

    async def _consume(self, consumer, topic: str, handler: Handler) -> None:
        async for record in consumer:
            value = json.loads(record.value)
            for attempt in range(1, self.max_deliveries + 1):
                try:
                    await handler(value)
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt == self.max_deliveries:
                        await self.publish(f"{topic}.dlq", record.key.decode(),
                                           {"value": value, "error": repr(exc)})
                    else:
                        await asyncio.sleep(0.2 * attempt)
            await consumer.commit()  # commit only after handled (or dead-lettered)

    async def drain(self) -> None:
        await asyncio.sleep(1.0)

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for c in self._consumers:
            await c.stop()
        if self._producer:
            await self._producer.stop()


class DbQueueBus:
    """Default bus, existing stack only: the outbox table is the queue, consumed by
    a pool of N async workers (competing consumers).

    Each worker loops: claim -> handle (with timeout) -> ack | nack.
      claim    store.claim_next: atomic lease on the head event of some hire's
               queue. Only heads are claimable, so workers never process two
               events of one hire at once or out of order; different hires run
               in parallel, up to N at a time.
      timeout  handler runs under asyncio.wait_for(handler_timeout); timeout <
               lease, so a slow handler is never re-delivered while still running.
      ack      fenced by lease owner.
      nack     exponential backoff with jitter via available_at; after
               max_deliveries the event is dead-lettered (dead=1): it stops
               blocking its hire, an on_dead callback alerts (Slack), and it
               waits in the DLQ for redrive or discard.
      crash    an expired lease makes the event claimable by another worker;
               handlers are idempotent (checkpointed runs), so redelivery is safe.
      stop     graceful: stop claiming, let in-flight handlers finish (bounded by
               grace_s), then cancel.
    Postgres with several processes: claim_next becomes SELECT .. FOR UPDATE SKIP
    LOCKED; nothing else changes.
    """

    def __init__(self, store: Store, workers: int = 4, lease_s: float = 300.0, handler_timeout_s: float = 120.0,
                 max_deliveries: int = 5, backoff_base_s: float = 0.5, backoff_max_s: float = 60.0,
                 poll_interval: float = 0.05, on_dead: Optional[Callable[[dict], Awaitable[None]]] = None):
        if handler_timeout_s >= lease_s:
            raise ValueError("handler_timeout_s must be < lease_s")
        self.store = store
        self.n_workers, self.lease_s, self.handler_timeout_s = workers, lease_s, handler_timeout_s
        self.max_deliveries, self.poll_interval = max_deliveries, poll_interval
        self.backoff = RetryPolicy(max_attempts=max_deliveries, base_delay=backoff_base_s, max_delay=backoff_max_s)
        self.on_dead = on_dead
        self._handlers: dict[str, list[tuple[str, Handler]]] = {}
        self._tasks: list[asyncio.Task] = []
        self._stopping = asyncio.Event()
        self._wake = asyncio.Event()
        self.in_flight = 0
        self.peak_in_flight = 0
        self.stats = {"delivered": 0, "failed": 0, "timeouts": 0, "dead": 0, "lease_lost": 0}
        self.per_worker: dict[str, int] = {}
        self.owner_prefix = f"w{uuid.uuid4().hex[:6]}"

    async def publish(self, topic: str, key: str, value: dict) -> None:
        self.store.enqueue(topic, key, value)
        self._wake.set()

    def subscribe(self, topic: str, group: str, handler: Handler) -> None:
        self._handlers.setdefault(topic, []).append((group, handler))

    async def start(self) -> None:
        self._stopping.clear()
        for i in range(self.n_workers):
            owner = f"{self.owner_prefix}-{i}"
            self.per_worker[owner] = 0
            self._tasks.append(asyncio.create_task(self._worker(owner), name=owner))

    async def _worker(self, owner: str) -> None:
        while not self._stopping.is_set():
            try:
                row = self.store.claim_next(owner, self.lease_s)
            except Exception:  # noqa: BLE001 - DB hiccup: back off, keep the worker alive
                log.exception("claim failed")
                row = None
            if row is None:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=self.poll_interval)
                except asyncio.TimeoutError:
                    pass
                continue
            await self._handle(row, owner)

    async def _handle(self, row: dict, owner: str) -> None:
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        value = json.loads(row["payload"])
        carrier = value.pop("_otel", None)  # transport metadata: handlers see only the event
        started = time.monotonic()
        attrs = {"messaging.system": "hiresync.outbox", "messaging.operation.type": "process",
                 "messaging.destination.name": row["topic"], "messaging.message.id": str(row["id"]),
                 "hiresync.key": row["key"], "hiresync.attempt": row["attempts"] + 1, "hiresync.worker": owner}
        try:
            with telemetry.span(f"hiresync.event {row['topic']}", kind=SpanKind.CONSUMER,
                                links=telemetry.links_from(carrier), **attrs) as sp:
                outcome = await self._run(row, owner, value, sp)
                sp.set_attribute("hiresync.outcome", outcome)
            telemetry.count("events", topic=row["topic"], outcome=outcome)
            telemetry.observe("event_duration", (time.monotonic() - started) * 1000, topic=row["topic"])
        finally:
            self.in_flight -= 1

    async def _run(self, row: dict, owner: str, value: dict, sp) -> str:
        try:
            await asyncio.wait_for(self._dispatch(row["topic"], value), timeout=self.handler_timeout_s)
        except asyncio.TimeoutError as exc:
            self.stats["timeouts"] += 1
            telemetry.mark_error(sp, exc)
            return await self._fail(row, owner, f"handler timeout after {self.handler_timeout_s:g}s", "timeout")
        except asyncio.CancelledError:
            raise
        except PermanentError as exc:  # retrying a 400 can't help: dead-letter now
            telemetry.mark_error(sp, exc)
            return await self._fail(row, owner, f"permanent: {exc}", "failed", permanent=True)
        except Exception as exc:  # noqa: BLE001
            telemetry.mark_error(sp, exc)
            return await self._fail(row, owner, repr(exc), "failed")
        if self.store.ack(row["id"], owner):
            self.stats["delivered"] += 1
            self.per_worker[owner] = self.per_worker.get(owner, 0) + 1
            return "ok"
        self.stats["lease_lost"] += 1
        return "lease_lost"

    async def _dispatch(self, topic: str, value: dict) -> None:
        for _group, handler in self._handlers.get(topic, []):
            await handler(value)

    async def _fail(self, row: dict, owner: str, error: str, outcome: str = "failed",
                    permanent: bool = False) -> str:
        self.stats["failed"] += 1
        max_attempts = row["attempts"] + 1 if permanent else self.max_deliveries
        dead = self.store.nack(row["id"], owner, error, self.backoff.delay(row["attempts"] + 1), max_attempts)
        if dead is None:
            self.stats["lease_lost"] += 1
            return "lease_lost"
        if dead:
            self.stats["dead"] += 1
            telemetry.count("dlq", kind="event", topic=row["topic"])
            log.warning("event %s (%s, key %s) dead-lettered: %s", row["id"], row["topic"], row["key"], error)
            if self.on_dead:
                try:
                    await self.on_dead({**row, "last_error": error})
                except Exception:  # noqa: BLE001
                    log.exception("on_dead callback failed")
            return "dead"
        log.info("event %s failed (attempt %d), retrying later: %s", row["id"], row["attempts"] + 1, error)
        return outcome

    async def drain(self, timeout_s: float = 60.0) -> None:
        """Wait until nothing is pending or in flight (events in backoff count as pending)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._wake.set()
            if not self.store.pending_events(limit=1) and self.in_flight == 0:
                return
            await asyncio.sleep(0.01)
        raise TimeoutError(f"queue not drained after {timeout_s}s")

    async def stop(self, grace_s: float = 5.0) -> None:
        self._stopping.set()
        self._wake.set()
        if self._tasks:
            _done, pending = await asyncio.wait(self._tasks, timeout=grace_s)
            for t in pending:
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()


def make_bus(store: Store, **db_queue_opts) -> Bus:
    kind = os.getenv("BUS", "db")
    if kind == "kafka":
        return KafkaBus(os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"))
    if kind == "memory":
        return InMemoryBus()
    return DbQueueBus(store, workers=int(os.getenv("WORKERS", db_queue_opts.pop("workers", 4))), **db_queue_opts)


class OutboxRelay:
    """Moves committed outbox rows onto the bus. A crash between publish and
    mark_published re-publishes the row: that's the at-least-once half, and the
    idempotent consumer is the other half."""

    def __init__(self, store: Store, bus: Bus, poll_interval: float = 0.05):
        self.store, self.bus, self.poll_interval = store, bus, poll_interval
        self._task: Optional[asyncio.Task] = None

    async def flush(self) -> int:
        rows = self.store.unpublished()
        for row in rows:
            await self.bus.publish(row["topic"], row["key"], json.loads(row["payload"]))
            self.store.mark_published(row["id"])
        return len(rows)

    async def _loop(self) -> None:
        while True:
            try:
                await self.flush()
            except Exception:  # noqa: BLE001 - keep relaying; rows stay unpublished
                log.exception("outbox relay error")
            await asyncio.sleep(self.poll_interval)

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
