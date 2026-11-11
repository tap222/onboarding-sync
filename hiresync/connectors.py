"""Mocked downstream systems + the projection each one needs from the canonical record.

Each mock behaves like the real API in the ways that matter to the orchestrator:
  * it honours an Idempotency-Key (a replayed key returns the stored response, no 2nd write)
  * it can fail transiently (429/503) or permanently (400/409) on demand
  * it has its own state that can drift (someone edits Workday by hand)
Real adapters would swap `_apply` for HTTP calls; the projection functions stay.
"""
from __future__ import annotations

import asyncio
import re
import time
import unicodedata
from collections import deque
from datetime import date, timedelta
from typing import Any, Optional

from .resilience import PermanentError, TransientError

COMPANY_DOMAIN = "betterup.co"


# ------------------------------------------------------------------ helpers
def ascii_slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def okta_login(rec: dict) -> str:
    first = rec.get("preferred_first_name") or rec["legal_first_name"]
    return f"{ascii_slug(first)}.{ascii_slug(rec['legal_last_name'])}@{COMPANY_DOMAIN}"


def business_days_before(d: date, n: int) -> date:
    while n > 0:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def cohort_week(start: date) -> str:
    return (start - timedelta(days=start.weekday())).isoformat()


def display_name(rec: dict) -> str:
    return f"{rec.get('preferred_first_name') or rec['legal_first_name']} {rec['legal_last_name']}"


SHIP_LEAD_BUSINESS_DAYS = 5


# --------------------------------------------------------------- base mock
class MockSystem:
    name = "base"
    depends_on: tuple[str, ...] = ()
    # Which canonical fields this system's projection reads. Used to explain
    # (and test) why a given diff does or does not touch a system.
    reads: tuple[str, ...] = ()

    def __init__(self, latency: float = 0.01):
        self.records: dict[str, dict] = {}
        self._idem: dict[str, dict] = {}
        self._faults: deque[Exception | tuple[str, float]] = deque()
        self.latency = latency
        self.calls = 0
        self.writes = 0
        self.compensations = 0
        self.state_history: dict[str, list[dict]] = {}

    # fault injection (demo / tests)
    def fail_next(self, count: int = 1, kind: str = "transient") -> None:
        for _ in range(count):
            self._faults.append(
                TransientError(f"{self.name}: 503 Service Unavailable") if kind == "transient"
                else PermanentError(f"{self.name}: 400 Bad Request")
            )

    def hang_next(self, count: int = 1, seconds: float = 30.0, after_write: bool = False) -> None:
        """Simulate an unresponsive API. after_write=True: the write lands but the
        response never arrives - the case idempotency keys exist for."""
        for _ in range(count):
            self._faults.append(("hang_after_write" if after_write else "hang", seconds))

    def clear_faults(self) -> None:
        self._faults.clear()

    def project(self, rec: dict, upstream: dict[str, dict]) -> dict:
        raise NotImplementedError

    async def upsert(self, hire_id: str, payload: dict, idempotency_key: str,
                     upstream: dict[str, dict]) -> dict:
        await asyncio.sleep(self.latency)
        self.calls += 1
        if idempotency_key in self._idem:
            return {**self._idem[idempotency_key], "replayed": True}
        fault = self._faults.popleft() if self._faults else None
        if isinstance(fault, Exception):
            raise fault
        if isinstance(fault, tuple) and fault[0] == "hang":
            await asyncio.sleep(fault[1])
        result = self._apply(hire_id, payload, upstream)
        self.writes += 1
        self._idem[idempotency_key] = result
        if isinstance(fault, tuple) and fault[0] == "hang_after_write":
            await asyncio.sleep(fault[1])  # write landed; caller times out waiting
        return result

    def _apply(self, hire_id: str, payload: dict, upstream: dict[str, dict]) -> dict:
        old_status = self.records.get(hire_id, {}).get("status")
        self.records[hire_id] = dict(payload)
        new_status = self.records[hire_id].get("status", "active")
        self.records[hire_id].setdefault("status", "active")
        self.state_history.setdefault(hire_id, []).append({
            "from": old_status, "to": new_status, "action": "upsert", "timestamp": time.time()
        })
        return {"id": f"{self.name}-{hire_id}"}

    async def get(self, hire_id: str) -> Optional[dict]:
        await asyncio.sleep(self.latency)
        if self._faults and isinstance(self._faults[0], TransientError):
            raise self._faults.popleft()
        return self.records.get(hire_id)

    # ---------------------------------------------------------- compensation
    async def compensate(self, hire_id: str, idempotency_key: str, today: date) -> dict:
        """Undo this system's effect for a cancelled hire. Idempotent by key, subject
        to the same fault injection as writes. Semantic, not a delete: the record
        stays (audit), its state says it was undone."""
        await asyncio.sleep(self.latency)
        self.calls += 1
        if idempotency_key in self._idem:
            return {**self._idem[idempotency_key], "replayed": True}
        fault = self._faults.popleft() if self._faults else None
        if isinstance(fault, Exception):
            raise fault
        if isinstance(fault, tuple) and fault[0] == "hang":
            await asyncio.sleep(fault[1])
        if hire_id not in self.records:
            result = {"action": "nothing_to_undo", "status": "not_found"}
        else:
            old_status = self.records[hire_id].get("status")
            result = self._undo(hire_id, today)
            new_status = self.records[hire_id].get("status")
            self.state_history.setdefault(hire_id, []).append({
                "from": old_status, "to": new_status, "action": result.get("action"),
                "idempotency_key": idempotency_key, "timestamp": time.time()
            })
        self.writes += 1
        self.compensations += 1
        self._idem[idempotency_key] = result
        if isinstance(fault, tuple) and fault[0] == "hang_after_write":
            await asyncio.sleep(fault[1])  # undo landed; caller times out waiting
        return result

    def _undo(self, hire_id: str, today: date) -> dict:
        self.records[hire_id]["status"] = "withdrawn"
        return {"action": "withdrawn", "status": "withdrawn"}

    def get_state(self, hire_id: str) -> Optional[dict]:
        rec = self.records.get(hire_id)
        if rec is None:
            return None
        return {
            "system": self.name,
            "status": rec.get("status"),
            "record": dict(rec),
            "history": list(self.state_history.get(hire_id, [])),
        }

    def get_state_history(self, hire_id: str) -> list[dict]:
        return list(self.state_history.get(hire_id, []))


# ----------------------------------------------------------- concrete mocks
class Workday(MockSystem):
    name = "workday"
    reads = ("legal_first_name", "legal_last_name", "preferred_first_name", "start_date", "job_title",
             "department", "manager_email", "employment_type", "work_location", "personal_email")

    def project(self, rec, upstream):
        return {
            "worker_id": f"WD-{rec['hire_id']}",
            "legal_name": {"first": rec["legal_first_name"], "last": rec["legal_last_name"]},
            "preferred_name": rec.get("preferred_first_name"),
            "hire_date": rec["start_date"],
            "position": {"title": rec["job_title"], "org": rec["department"], "manager": rec["manager_email"]},
            "worker_type": "Employee" if rec["employment_type"] == "FTE" else "Contingent Worker",
            "location": rec["work_location"],
            "personal_email": rec["personal_email"],
        }

    def _undo(self, hire_id, today):
        self.records[hire_id]["status"] = "Rescinded"
        return {"action": "hire_rescinded", "status": "Rescinded"}

    def manual_edit(self, hire_id: str, path: list[str], value: Any) -> None:
        """Someone edits Workday directly - the drift case reconciliation must catch."""
        node = self.records[hire_id]
        for p in path[:-1]:
            node = node[p]
        node[path[-1]] = value


class Okta(MockSystem):
    name = "okta"
    depends_on = ("workday",)
    reads = ("legal_first_name", "legal_last_name", "preferred_first_name", "start_date",
             "department", "manager_email")

    def project(self, rec, upstream):
        return {
            "login": okta_login(rec),
            "firstName": rec.get("preferred_first_name") or rec["legal_first_name"],
            "lastName": rec["legal_last_name"],
            "department": rec["department"],
            "manager": rec["manager_email"],
            "activation_date": rec["start_date"],
            "employeeNumber": upstream.get("workday", {}).get("id"),
        }

    def _apply(self, hire_id, payload, upstream):
        for other_id, user in self.records.items():
            if other_id != hire_id and user["login"] == payload["login"]:
                raise PermanentError(f"okta: 409 login {payload['login']} already taken")
        existing = self.records.get(hire_id)
        status = existing["status"] if existing else "STAGED"
        self.records[hire_id] = {**payload, "status": status}
        return {"id": f"00u{hire_id}", "login": payload["login"], "status": status}

    def activate(self, hire_id: str) -> None:
        self.records[hire_id]["status"] = "ACTIVE"

    async def activate_user(self, hire_id: str) -> dict:
        """Day-1 activation as an idempotent activity."""
        await asyncio.sleep(self.latency)
        if self._faults:
            fault = self._faults.popleft()
            if isinstance(fault, Exception):
                raise fault
        user = self.records.get(hire_id)
        if not user:
            raise PermanentError(f"okta: no user for {hire_id}")
        if user["status"] == "DEPROVISIONED":
            raise PermanentError(f"okta: {hire_id} was deprovisioned")
        user["status"] = "ACTIVE"
        return {"status": "ACTIVE", "login": user["login"]}

    def _undo(self, hire_id, today):
        was = self.records[hire_id].get("status", "STAGED")
        self.records[hire_id]["status"] = "DEPROVISIONED"
        return {"action": "deprovisioned", "was": was, "status": "DEPROVISIONED"}

    def login_owner(self, login: str) -> Optional[str]:
        return next((h for h, u in self.records.items() if u["login"] == login), None)


ACCESS_BUNDLES = {
    "Engineering": ["github", "aws-dev", "datadog", "slack", "google-workspace"],
    "People": ["workday-admin", "ashby", "slack", "google-workspace"],
}


class Lumos(MockSystem):
    name = "lumos"
    depends_on = ("okta",)
    reads = ("department", "start_date", "employment_type")

    def project(self, rec, upstream):
        return {
            "okta_user_id": upstream.get("okta", {}).get("id"),
            "bundle": ACCESS_BUNDLES.get(rec["department"], ["slack", "google-workspace"]),
            "grant_on": rec["start_date"],
            "expires": None if rec["employment_type"] == "FTE" else "contract-end",
        }

    def _apply(self, hire_id, payload, upstream):
        if not payload.get("okta_user_id"):
            raise PermanentError("lumos: 422 okta user id required")
        self.records[hire_id] = payload
        return {"id": f"lumos-{hire_id}", "apps": len(payload["bundle"])}

    def _undo(self, hire_id, today):
        revoked = self.records[hire_id].get("bundle", [])
        self.records[hire_id] = {**self.records[hire_id], "bundle": [], "status": "revoked"}
        return {"action": "access_revoked", "apps": len(revoked), "status": "revoked"}


class ExpoIT(MockSystem):
    """Equipment: laptop order and shipping label."""
    name = "expoit"
    reads = ("legal_first_name", "legal_last_name", "preferred_first_name", "start_date",
             "department", "shipping_address")

    def project(self, rec, upstream):
        start = date.fromisoformat(str(rec["start_date"]))
        return {
            "recipient": f"{rec['legal_first_name']} {rec['legal_last_name']}",
            "ship_to": rec["shipping_address"],
            "device": "MacBook Pro 16" if rec["department"] == "Engineering" else "MacBook Air 15",
            "ship_by": business_days_before(start, SHIP_LEAD_BUSINESS_DAYS).isoformat(),
        }

    def _undo(self, hire_id, today):
        """Semantic compensation: what 'undo' means depends on where the order is."""
        order = self.records[hire_id]
        if order.get("status") == "cancelled":
            return {"action": "order_cancelled", "status": "cancelled"}
        if order.get("status") == "return_requested":
            return {"action": "return_label_sent", "to": order["ship_to"].get("city"), "status": "return_requested"}
        if today < date.fromisoformat(order["ship_by"]):
            order["status"] = "cancelled"
            return {"action": "order_cancelled", "status": "cancelled"}
        order["status"] = "return_requested"
        return {"action": "return_label_sent", "to": order["ship_to"].get("city"), "status": "return_requested"}


class CohortTracker(MockSystem):
    """The onboarding cohort tracker (a Google Sheet today)."""
    name = "tracker"
    depends_on = ("okta",)  # shows the Okta login
    reads = ("legal_first_name", "legal_last_name", "preferred_first_name", "start_date",
             "department", "manager_email")

    def project(self, rec, upstream):
        start = date.fromisoformat(str(rec["start_date"]))
        return {
            "name": display_name(rec),
            "cohort_week": cohort_week(start),
            "start_date": rec["start_date"],
            "department": rec["department"],
            "manager": rec["manager_email"],
            "okta_login": upstream.get("okta", {}).get("login"),
        }

    def _undo(self, hire_id, today):
        self.records[hire_id]["status"] = "withdrawn"
        return {"action": "cohort_withdrawn", "status": "withdrawn"}


class SlackNotifier:
    """Mock Slack: in production a chat.postMessage with Block Kit approve/reject buttons."""

    def __init__(self):
        self.messages: list[dict] = []

    async def post(self, channel: str, text: str) -> None:
        self.messages.append({"channel": channel, "text": text})


def default_systems() -> dict[str, MockSystem]:
    systems = [Workday(), Okta(), Lumos(), ExpoIT(), CohortTracker()]
    return {s.name: s for s in systems}


class DependencyGraphError(ValueError):
    pass


def validate_dag(systems: dict[str, MockSystem]) -> None:
    """The propagation graph must be a DAG over known systems. Reports the actual
    cycle (e.g. okta -> lumos -> okta), not just 'there is one'."""
    for name, s in systems.items():
        unknown = [d for d in s.depends_on if d not in systems]
        if unknown:
            raise DependencyGraphError(f"{name} depends on unknown system(s) {unknown}")
    WHITE, GREY, BLACK = 0, 1, 2
    color = {n: WHITE for n in systems}
    stack: list[str] = []

    def visit(n: str) -> None:
        color[n] = GREY
        stack.append(n)
        for d in systems[n].depends_on:
            if color[d] == GREY:
                cycle = stack[stack.index(d):] + [d]
                raise DependencyGraphError("dependency cycle: " + " -> ".join(cycle))
            if color[d] == WHITE:
                visit(d)
        stack.pop()
        color[n] = BLACK

    for n in sorted(systems):
        if color[n] == WHITE:
            visit(n)


def topo_levels(systems: dict[str, MockSystem]) -> list[list[str]]:
    """Group systems into dependency levels (Kahn); each level runs in parallel.
    Validated first, so a misconfigured graph fails at startup, not mid-run."""
    validate_dag(systems)
    remaining = dict(systems)
    done: set[str] = set()
    levels: list[list[str]] = []
    while remaining:
        level = sorted(n for n, s in remaining.items() if all(d in done for d in s.depends_on))
        levels.append(level)
        done.update(level)
        for n in level:
            remaining.pop(n)
    return levels
