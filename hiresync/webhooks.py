"""Webhooks, both directions.

Inbound (api.py /webhooks/ashby): HMAC over "<timestamp>.<body>", timestamp within
5 minutes (replay protection), plus Idempotency-Key dedupe.

Outbound: how the existing tools (n8n, Zapier, Slack workflows) hear about what
happens - run.completed, run.partial, approval.requested, approval.decided,
dlq.dead, budget.daily_exceeded - without polling and without new infrastructure.

  emit(event)      one outbox row per matching subscription, in the same DB, keyed
                   "webhook:<sub_id>" so each subscriber gets events in order
  deliver          a worker-pool handler: POST with a timeout; non-2xx or timeout
                   raises, so the pool's backoff -> dead-letter path applies
  signature        HireSync-Signature: t=<unix>,v1=<hex hmac_sha256(secret, f"{t}.{body}")>
                   HireSync-Event-Id lets receivers dedupe (delivery is at-least-once)
  SSRF guardrail   https only, host must be on the allowlist (n8n / Zapier / Slack
                   by default), no IP literals, no localhost/private names
  secrets          generated server-side, returned once at creation
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import time
import uuid
from typing import TYPE_CHECKING, Awaitable, Callable, Optional, Protocol
from urllib.parse import urlparse

from opentelemetry import propagate

from .resilience import PermanentError, TransientError, with_timeout
from .store import now

if TYPE_CHECKING:
    from .app import HireSync

EVENT_TYPES = {"run.completed", "run.partial", "approval.requested", "approval.decided", "dlq.dead",
               "budget.daily_exceeded", "run.compensated"}
DEFAULT_ALLOWED_HOSTS = ("hooks.zapier.com", ".n8n.cloud", "hooks.slack.com")
REPLAY_TOLERANCE_S = 300


def allowed_hosts() -> tuple[str, ...]:
    extra = tuple(h.strip() for h in os.getenv("WEBHOOK_ALLOWED_HOSTS", "").split(",") if h.strip())
    return DEFAULT_ALLOWED_HOSTS + extra


def check_url(url: str) -> None:
    u = urlparse(url)
    if u.scheme != "https":
        raise ValueError("webhook URLs must be https")
    host = (u.hostname or "").lower()
    try:
        ipaddress.ip_address(host)
        raise ValueError("IP-literal webhook hosts are not allowed")
    except ValueError as exc:
        if "not allowed" in str(exc):
            raise
    if host in ("localhost",) or host.endswith((".local", ".internal")):
        raise ValueError("internal hosts are not allowed")
    if not any(host == h or (h.startswith(".") and host.endswith(h)) for h in allowed_hosts()):
        raise ValueError(f"{host} is not on the webhook allowlist {list(allowed_hosts())}")


def sign(secret: str, body: bytes, ts: Optional[int] = None) -> str:
    ts = ts if ts is not None else int(time.time())
    mac = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={ts},v1={mac}"


def verify(secret: str, body: bytes, header: str, tolerance_s: int = REPLAY_TOLERANCE_S) -> bool:
    """What a receiver (or our own inbound endpoint) runs."""
    try:
        parts = dict(p.split("=", 1) for p in header.split(","))
        ts = int(parts["t"])
    except (ValueError, KeyError):
        return False
    if abs(time.time() - ts) > tolerance_s:
        return False  # replayed or badly skewed
    expected = sign(secret, body, ts).split("v1=", 1)[1]
    return hmac.compare_digest(expected, parts.get("v1", ""))


class Transport(Protocol):
    async def post(self, url: str, body: bytes, headers: dict) -> int: ...


class RecordingTransport:
    """Offline stand-in for the receivers (n8n, Zapier); records what arrived."""

    def __init__(self):
        self.received: list[dict] = []
        self.fail_next: list[int] = []  # status codes to return next
        self.hang_next = 0

    async def post(self, url: str, body: bytes, headers: dict) -> int:
        import asyncio
        if self.hang_next:
            self.hang_next -= 1
            await asyncio.sleep(30)
        if self.fail_next:
            return self.fail_next.pop(0)
        self.received.append({"url": url, "body": json.loads(body), "headers": headers})
        return 200


class HttpxTransport:
    async def post(self, url: str, body: bytes, headers: dict) -> int:
        import httpx
        async with httpx.AsyncClient(follow_redirects=False) as client:  # redirects could bypass the allowlist
            r = await client.post(url, content=body, headers=headers)
            return r.status_code


class Webhooks:
    def __init__(self, app: "HireSync", transport: Optional[Transport] = None, timeout_s: float = 5.0):
        self.app = app
        self.transport = transport or (HttpxTransport() if os.getenv("WEBHOOK_TRANSPORT") == "http"
                                       else RecordingTransport())
        self.timeout_s = timeout_s

    # ----------------------------------------------------------- subscriptions
    def subscribe(self, url: str, events: list[str], created_by: str) -> dict:
        check_url(url)
        unknown = set(events) - EVENT_TYPES
        if unknown:
            raise ValueError(f"unknown event types {sorted(unknown)}; known: {sorted(EVENT_TYPES)}")
        sub_id, secret = f"wh_{uuid.uuid4().hex[:10]}", f"whsec_{secrets.token_urlsafe(24)}"
        self.app.store.add_webhook(sub_id, url, events, secret, created_by)
        return {"id": sub_id, "url": url, "events": events, "secret": secret}  # secret shown once

    # ------------------------------------------------------------------ emit
    def emit(self, event_type: str, data: dict) -> int:
        """Enqueue one delivery per matching subscription. Returns how many."""
        assert event_type in EVENT_TYPES, event_type
        event = {"id": f"evt_{uuid.uuid4().hex[:16]}", "type": event_type, "created": now(), "data": data}
        n = 0
        for sub in self.app.store.webhooks_for(event_type):
            self.app.store.enqueue("webhook.deliver", f"webhook:{sub['id']}", {"sub_id": sub["id"], "event": event})
            n += 1
        return n

    # --------------------------------------------------------------- deliver
    async def deliver(self, msg: dict) -> None:
        sub = self.app.store.get_webhook(msg["sub_id"])
        if not sub or not sub["active"]:
            return  # unsubscribed since: drop quietly
        body = json.dumps(msg["event"], sort_keys=True).encode()
        headers = {"Content-Type": "application/json", "HireSync-Event-Id": msg["event"]["id"],
                   "HireSync-Event-Type": msg["event"]["type"], "HireSync-Signature": sign(sub["secret"], body)}
        propagate.inject(headers)  # traceparent: an instrumented receiver (n8n) joins the same trace
        status = await with_timeout(lambda: self.transport.post(sub["url"], body, headers), self.timeout_s,
                                    f"webhook {sub['id']}")
        self.app.store.record_webhook_delivery(sub["id"], msg["event"]["id"], msg["event"]["type"], status)
        if status >= 500 or status in (408, 429):
            raise TransientError(f"webhook {sub['id']} answered {status}")
        if status >= 400:
            raise PermanentError(f"webhook {sub['id']} rejected the event with {status}")
