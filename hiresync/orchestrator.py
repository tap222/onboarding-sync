"""The orchestrator: one durable, checkpointed run per (hire, version).

    START -> VALIDATED -> TRIAGED -> [AWAITING_APPROVAL] -> APPROVED -> PROPAGATED -> RECONCILED -> DONE

* run_id = "<hire_id>:v<version>" - redelivered events land on the same run.
* After every step the whole WorkingMemory is checkpointed to the DB. A crash, a
  deploy, or a 3-day wait for an approver resumes from the last checkpoint.
* A Redis lock (SET NX) stops two workers driving the same run at once.
* Every downstream write has idempotency key "<hire>:v<n>:<system>", stored in
  side_effects and sent to the system - on resume, finished writes are skipped,
  and a write that landed but wasn't recorded is deduped by the system.
* Targets are computed against what each system last *accepted*, not the previous
  version, so a superseded or partial run never leaves a field behind.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import date
from typing import TYPE_CHECKING, Callable, Optional

from opentelemetry.trace import SpanKind

from . import telemetry
from .agent import apply_policy
from .auth import Forbidden, Principal, require, require_hire
from .connectors import okta_login, topo_levels
from .memory import WorkingMemory
from .models import Issue, RunStatus, Severity, Step, canonicalize
from .resilience import (Bulkheads, CircuitBreaker, CircuitOpenError, PermanentError, RetryPolicy, Timeouts,
                         TransientError, retry_async, with_timeout)
from .validation import validate
from .versioning import diff, unflatten_set

if TYPE_CHECKING:
    from .app import HireSync

log = logging.getLogger(__name__)

TERMINAL = {RunStatus.COMPLETED.value, RunStatus.REJECTED.value, RunStatus.SUPERSEDED.value, RunStatus.COMPENSATED.value}


class Orchestrator:
    def __init__(self, app: "HireSync", today: Callable[[], date] = date.today,
                 retry_policy: Optional[RetryPolicy] = None, timeouts: Optional[Timeouts] = None):
        self.app = app
        self.store = app.store
        self.systems = app.systems
        self.today = today
        self.retry_policy = retry_policy or RetryPolicy(max_attempts=3, base_delay=0.05, max_delay=0.3)
        self.timeouts = timeouts or Timeouts()
        self.breakers = {n: CircuitBreaker(n, failure_threshold=3, reset_timeout=0.5) for n in self.systems}
        self.bulkheads = Bulkheads({"workday": 2, "okta": 1, "lumos": 2, "expoit": 2, "tracker": 4})
        self.levels = topo_levels(self.systems)

    # ------------------------------------------------------------- consumers
    async def on_hire_changed(self, msg: dict) -> None:
        run_id = f"{msg['hire_id']}:v{msg['version']}"
        if not self.store.get_run(run_id):
            wm = WorkingMemory(run_id=run_id, hire_id=msg["hire_id"], from_version=msg.get("from_version"),
                               to_version=msg["version"], trace_id=msg.get("trace_id"))
            self._checkpoint(wm, RunStatus.RUNNING, Step.START)
        await self.drive(run_id)

    async def on_run_resume(self, msg: dict) -> None:
        await self.drive(msg["run_id"])

    # ----------------------------------------------------------------- drive
    async def drive(self, run_id: str) -> None:
        owned, _ = await self.app.idem.claim("runlock", run_id)
        if not owned:
            log.info("%s is being driven by another worker", run_id)
            return
        try:
            await self._drive(run_id)
        finally:
            await self.app.idem.release("runlock", run_id)

    async def _drive(self, run_id: str) -> None:
        run = self.store.get_run(run_id)
        if not run or run["status"] in TERMINAL or run["status"] == RunStatus.AWAITING_APPROVAL.value:
            return
        wm = WorkingMemory.from_dict(run["state"])
        step = Step(run["step"])
        ids = {"hiresync.run_id": wm.run_id, "hiresync.hire_id": wm.hire_id, "hiresync.version": wm.to_version,
               "hiresync.trace_id": wm.trace_id}

        if step == Step.START:
            if self._superseded(wm):
                return
            with telemetry.span("hiresync.run.validate", **ids) as sp:
                self._load(wm)
                sp.set_attribute("hiresync.issues", [i["code"] for i in wm.issues])
                sp.set_attribute("hiresync.targets", wm.targets)
            step = self._checkpoint(wm, RunStatus.RUNNING, Step.VALIDATED)
        if step == Step.VALIDATED:
            with telemetry.span("hiresync.run.triage", **ids) as sp:
                decision = await self._triage(wm)
                sp.set_attribute("hiresync.decision", decision)
                if wm.recommendation:
                    sp.set_attribute("hiresync.triage.source", wm.recommendation.get("source", ""))
                    sp.set_attribute("hiresync.triage.prompt", wm.recommendation.get("prompt", ""))
            if decision == "hold":
                return  # parked in AWAITING_APPROVAL; a human decision resumes it
            step = Step.APPROVED
        if step == Step.APPROVED:
            if self._superseded(wm):
                return
            with telemetry.span("hiresync.run.propagate", **ids) as sp:
                await self._propagate(wm)
                bad = sorted(n for n, r in wm.results.items() if r.get("status") != "succeeded")
                sp.set_attribute("hiresync.failed_systems", bad)
            step = self._checkpoint(wm, RunStatus.RUNNING, Step.PROPAGATED)
        if step == Step.PROPAGATED:
            with telemetry.span("hiresync.run.reconcile", **ids):
                await self._reconcile(wm)
            step = self._checkpoint(wm, RunStatus.RUNNING, Step.RECONCILED)
        if step == Step.RECONCILED:
            await self._finish(wm)

    # ----------------------------------------------------------------- steps
    def _load(self, wm: WorkingMemory) -> None:
        new = self.store.get_version(wm.hire_id, wm.to_version)
        old = self.store.get_version(wm.hire_id, wm.from_version) if wm.from_version else None
        wm.new, wm.old = new["data"], (old["data"] if old else None)
        wm.changes = [c.model_dump() for c in diff(wm.old, wm.new)]
        issues = validate(wm.new, wm.old, diff(wm.old, wm.new), self.systems, self.today())
        issues += self._document_issues(wm)
        issues += self._reserve_login(wm)
        wm.issues = [i.model_dump(mode="json") for i in issues]
        wm.targets = self._targets(wm)
        # A newer version makes any older run still waiting on a human moot.
        for r in self.store.list_runs(wm.hire_id, RunStatus.AWAITING_APPROVAL.value):
            if r["to_version"] < wm.to_version:
                self._supersede(WorkingMemory.from_dict(r["state"]), by=wm.run_id)

    def _reserve_login(self, wm: WorkingMemory) -> list[Issue]:
        """Validation reads Okta, but a parallel run may be creating the same login
        right now (check-then-act race). Reserve it atomically in our DB first."""
        try:
            login = okta_login(wm.new)
        except KeyError:
            return []
        owner = self.store.reserve_login(login, wm.hire_id)
        if owner != wm.hire_id:
            return [Issue(code="OKTA_LOGIN_COLLISION", severity=Severity.BLOCK, field="legal_last_name",
                          message=f"{login} is reserved by {owner} (concurrent or earlier hire)", systems=["okta"])]
        return []

    def _document_issues(self, wm: WorkingMemory) -> list[Issue]:
        issues = []
        for h in self._doc_hashes(wm):
            doc = self.store.get_document(h)
            if doc and doc["scan"]["suspicious"]:
                issues.append(Issue(code="DOCUMENT_INJECTION_SUSPECTED", severity=Severity.BLOCK,
                                    message=f"source document {h[:12]} contains instruction-like text: "
                                            f"{doc['scan']['injection_hits']}"))
        return issues

    def _doc_hashes(self, wm: WorkingMemory) -> list[str]:
        return sorted({p["doc_hash"] for p in self.store.provenance(wm.hire_id)
                       if p["version"] == wm.to_version and p["doc_hash"]})

    def _targets(self, wm: WorkingMemory) -> list[str]:
        targets = []
        for name, system in self.systems.items():
            last = self.store.last_pushed(wm.hire_id, name)
            upstream = {d: (self.store.last_pushed(wm.hire_id, d) or {}).get("response", {})
                        for d in system.depends_on}
            if last is None or system.project(wm.new, upstream) != last["request"]:
                targets.append(name)
        return sorted(targets)

    async def _triage(self, wm: WorkingMemory) -> str:
        if not wm.issues:
            wm.decision = "auto"
            self._checkpoint(wm, RunStatus.RUNNING, Step.APPROVED)
            return "proceed"
        wm.recalled = self.app.episodic.recall(wm.features(), exclude_run=wm.run_id, text=wm.summary())
        rec, transcript = await self.app.agent.triage(wm, self._doc_hashes(wm))
        wm.recommendation = rec.model_dump()
        if transcript:
            wm.notes.append(f"agent tools: {[t['tool'] for t in transcript]}")
        decision = apply_policy(wm, rec)
        self.store.audit(wm.hire_id, wm.run_id, f"agent:{rec.source}", "triage.recommendation",
                         {"action": rec.action, "confidence": rec.confidence, "policy": decision,
                          "evidence": rec.evidence, "tools": transcript, "trace_id": wm.trace_id})
        if decision == "proceed":
            wm.decision, wm.approved_by = "agent_proceed", f"agent:{rec.source}"
            self._checkpoint(wm, RunStatus.RUNNING, Step.APPROVED)
            return "proceed"
        wm.approval_id = self.store.create_approval(
            wm.run_id, wm.hire_id, "; ".join(f"{i['code']}({i['severity']})" for i in wm.issues), wm.recommendation)
        self._checkpoint(wm, RunStatus.AWAITING_APPROVAL, Step.TRIAGED)
        self.app.webhooks.emit("approval.requested", {"approval_id": wm.approval_id, "run_id": wm.run_id,
                                                      "hire_id": wm.hire_id, "issues": [i["code"] for i in wm.issues]})
        await self.app.slack.post("#people-ops-approvals",
                                  f"[{wm.approval_id}] {rec.approver_message} (confidence {rec.confidence:.2f})")
        return "hold"

    async def _propagate(self, wm: WorkingMemory) -> None:
        upstream = {n: (self.store.last_pushed(wm.hire_id, n) or {}).get("response", {}) for n in self.systems}
        for level in self.levels:
            names = []
            for n in level:
                if n in wm.targets:
                    names.append(n)
                elif any(wm.results.get(d, {}).get("status") == "succeeded" for d in self.systems[n].depends_on):
                    # Upstream was just rewritten: re-push only if this system's projection
                    # actually changes (e.g. a new Okta id), not on every upstream write.
                    last = self.store.last_pushed(wm.hire_id, n)
                    if last is None or self.systems[n].project(wm.new, upstream) != last["request"]:
                        names.append(n)
                        wm.targets.append(n)
            for name in names:  # fail fast on broken dependencies, before spending calls
                broken = [d for d in self.systems[name].depends_on
                          if wm.results.get(d, {}).get("status") not in (None, "succeeded")]
                if broken:
                    wm.results[name] = {"status": "blocked", "error": f"dependency {broken} not written"}
            runnable = [n for n in names if wm.results.get(n, {}).get("status") != "blocked"]
            outcomes = await asyncio.gather(*(self._push(wm, n, upstream) for n in runnable))
            for name, out in zip(runnable, outcomes):
                wm.results[name] = out
                if out["status"] == "succeeded":
                    upstream[name] = out["response"]

    async def _push(self, wm: WorkingMemory, name: str, upstream: dict) -> dict:
        system = self.systems[name]
        key = f"{wm.hire_id}:v{wm.to_version}:{name}"
        done = self.store.get_side_effect(key)
        if done and done["status"] == "succeeded":
            return {"status": "succeeded", "response": json.loads(done["response"]), "resumed": True}
        payload = system.project(wm.new, upstream)
        breaker = self.breakers[name]

        call_timeout = self.timeouts.for_system(name)
        n_attempt = {"n": 0}

        async def attempt():
            # bulkhead (per attempt, so a backoff sleep doesn't hold a slot)
            #   -> breaker (a timeout counts as a failure) -> timeout -> call
            n_attempt["n"] += 1
            started = time.monotonic()
            with telemetry.tracer().start_as_current_span(
                    f"hiresync.system.write {name}", kind=SpanKind.CLIENT,
                    attributes={"hiresync.system": name, "hiresync.attempt": n_attempt["n"],
                                "hiresync.timeout_s": call_timeout, "hiresync.run_id": wm.run_id,
                                "hiresync.breaker.state": breaker.snapshot()["state"]},
                    record_exception=False, set_status_on_exception=False) as sp:
                outcome, err = "ok", None
                try:
                    return await self.bulkheads.run(name, lambda: breaker.call(lambda: with_timeout(
                        lambda: system.upsert(wm.hire_id, payload, key, upstream), call_timeout, f"{name}.upsert")))
                except BaseException as exc:
                    err = type(exc).__name__
                    outcome = {"CallTimeout": "timeout", "CircuitOpenError": "breaker_open",
                               "PermanentError": "permanent"}.get(err, "transient")
                    telemetry.mark_error(sp, exc, expected=outcome == "breaker_open")
                    raise
                finally:
                    sp.set_attribute("hiresync.outcome", outcome)
                    telemetry.count("sys_calls", system=name, outcome=outcome, **{"error.type": err})
                    telemetry.observe("sys_duration", (time.monotonic() - started) * 1000, system=name, outcome=outcome)

        try:
            response, attempts = await retry_async(attempt, self.retry_policy, budget_s=self.timeouts.retry_budget)
            self.store.record_side_effect(key, wm.run_id, name, "succeeded", payload, response, attempts)
            self.store.resolve_dead_letters(wm.run_id, name)
            return {"status": "succeeded", "response": response, "attempts": attempts}
        except CircuitOpenError as exc:
            self.store.record_side_effect(key, wm.run_id, name, "deferred", payload, None, 0, str(exc))
            return {"status": "deferred", "error": str(exc)}
        except (TransientError, PermanentError) as exc:
            kind = "transient" if isinstance(exc, TransientError) else "permanent"
            attempts = getattr(exc, "attempts", 1)
            self.store.record_side_effect(key, wm.run_id, name, "failed", payload, None, attempts, str(exc))
            self.store.dead_letter(wm.run_id, name, payload, f"{kind}: {exc}")
            telemetry.count("dlq", kind="write", topic=f"write:{name}")
            return {"status": "failed", "error": str(exc), "kind": kind, "attempts": attempts}

    async def _reconcile(self, wm: WorkingMemory) -> None:
        """Read back what each system now holds and compare with what we sent."""
        wm.reconciliation = []
        for name, res in wm.results.items():
            if res.get("status") != "succeeded":
                continue
            try:
                actual = await with_timeout(lambda: self.systems[name].get(wm.hire_id),
                                            self.timeouts.for_system(name), f"{name}.get")
            except TransientError as exc:
                wm.reconciliation.append({"system": name, "status": "unverified", "error": str(exc)})
                continue
            se = self.store.get_side_effect(f"{wm.hire_id}:v{wm.to_version}:{name}")
            expected = json.loads(se["request"])
            drift = sorted(k for k, v in expected.items() if (actual or {}).get(k) != v)
            wm.reconciliation.append({"system": name, "status": "drift" if drift else "ok", "fields": drift})

    async def _finish(self, wm: WorkingMemory) -> None:
        bad = {n: r for n, r in wm.results.items() if r.get("status") != "succeeded"}
        status = RunStatus.PARTIAL if bad else RunStatus.COMPLETED
        self._checkpoint(wm, status, Step.DONE)
        telemetry.count("runs", status=status.value, decision=wm.decision or "auto")
        self.app.webhooks.emit(f"run.{status.value}", {
            "run_id": wm.run_id, "hire_id": wm.hire_id, "version": wm.to_version, "trace_id": wm.trace_id,
            "systems": {n: r.get("status") for n, r in wm.results.items()}})
        if wm.issues and wm.decision in ("approved", "agent_proceed"):
            self.app.episodic.record(wm, wm.decision, wm.approved_by or "?", status.value)
        if bad:
            lines = ", ".join(f"{n}={r['status']}" for n, r in sorted(bad.items()))
            await self.app.slack.post("#people-ops-alerts",
                                      f"{wm.hire_id} v{wm.to_version} partially propagated: {lines}. "
                                      f"Retry with retry_run('{wm.run_id}').")

    # --------------------------------------------------------- human actions
    async def decide(self, approval_id: str, decision: str, principal: Principal,
                     note: Optional[str] = None, patch: Optional[dict] = None) -> dict:
        require(principal, "approval:decide", approval_id)
        pending = next((a for a in self.store.list_approvals() if a["approval_id"] == approval_id), None)
        if not pending:
            raise KeyError(f"unknown approval {approval_id}")
        current = self.store.get_version(pending["hire_id"])
        require_hire(principal, pending["hire_id"], current["data"] if current else None)
        run_row = self.store.get_run(pending["run_id"])
        changed_by = self.store.get_version(pending["hire_id"], run_row["to_version"])["actor"]
        if changed_by.lower() == principal.email.lower():
            # Separation of duties: nobody approves their own change.
            raise Forbidden(principal, "approve", approval_id, "you made this change; another approver must decide")
        actor = principal.email
        row = self.store.decide_approval(approval_id, decision, actor, note, patch)
        if row["decided_by"] == actor:
            self.app.webhooks.emit("approval.decided", {"approval_id": approval_id, "run_id": row["run_id"],
                                                        "hire_id": row["hire_id"], "decision": decision,
                                                        "decided_by": actor, "corrected": bool(patch)})
        run = self.store.get_run(row["run_id"])
        if row["decided_by"] != actor or run["status"] != RunStatus.AWAITING_APPROVAL.value:
            return {"approval_id": approval_id, "status": row["status"], "note": "already decided"}
        wm = WorkingMemory.from_dict(run["state"])
        if decision == "rejected":
            wm.decision = "rejected"
            self._checkpoint(wm, RunStatus.REJECTED, Step.DONE)
            self.app.episodic.record(wm, "rejected", actor, "rejected")
            return {"approval_id": approval_id, "status": "rejected"}
        if patch:
            # Correction = a new version through the full pipeline; this run is superseded.
            record = wm.new
            for path, value in _flatten_patch(patch).items():
                record = unflatten_set(record, path, value)
            try:
                record = canonicalize(record)
            except ValueError as exc:
                raise ValueError(f"correction produces an invalid record: {exc}") from exc
            wm.decision = "corrected"
            self.app.episodic.record(wm, "corrected", actor, "superseded by correction")
            res = self.store.append_version(wm.hire_id, record, source="human", actor=actor,
                                            event_id=f"{approval_id}:correction",
                                            provenance={p: {"citation": {"approval_id": approval_id, "note": note}}
                                                        for p in _flatten_patch(patch)},
                                            trace_id=wm.trace_id)
            self._supersede(wm, by=f"{wm.hire_id}:v{res['version']}")
            return {"approval_id": approval_id, "status": "approved", "corrected_version": res["version"]}
        wm.decision, wm.approved_by = "approved", actor
        self._checkpoint(wm, RunStatus.RUNNING, Step.APPROVED)
        self.store.enqueue("run.resume", wm.hire_id, {"run_id": wm.run_id})  # outbox -> bus -> resume
        return {"approval_id": approval_id, "status": "approved", "resuming": wm.run_id}

    async def retry_run(self, run_id: str) -> dict:
        run = self.store.get_run(run_id)
        if not run or run["status"] != RunStatus.PARTIAL.value:
            return {"run_id": run_id, "status": run["status"] if run else "unknown", "note": "nothing to retry"}
        wm = WorkingMemory.from_dict(run["state"])
        wm.results = {n: r for n, r in wm.results.items() if r.get("status") == "succeeded"}
        self._checkpoint(wm, RunStatus.RUNNING, Step.APPROVED)
        await self.drive(run_id)
        return {"run_id": run_id, "status": self.store.get_run(run_id)["status"]}

    # ---------------------------------------------------------- compensation
    async def _compensate_system(self, wm: WorkingMemory, name: str, reason: str, today: date) -> dict:
        """Compensate one downstream system with exponential backoff on retries and state tracking."""
        system = self.systems[name]
        key = f"{wm.hire_id}:v{wm.to_version}:undo:{name}"
        done = self.store.get_side_effect(key)
        if done and done["status"] == "compensated":
            return {"status": "compensated", "response": json.loads(done["response"]), "resumed": True}

        breaker = self.breakers[name]
        call_timeout = self.timeouts.for_system(name)
        n_attempt = {"n": 0}
        retry_history: list[dict] = []

        async def attempt():
            n_attempt["n"] += 1
            attempt_idx = n_attempt["n"]
            started = time.monotonic()
            with telemetry.tracer().start_as_current_span(
                    f"hiresync.system.compensate {name}", kind=SpanKind.CLIENT,
                    attributes={"hiresync.system": name, "hiresync.attempt": attempt_idx,
                                "hiresync.timeout_s": call_timeout, "hiresync.run_id": wm.run_id,
                                "hiresync.breaker.state": breaker.snapshot()["state"]},
                    record_exception=False, set_status_on_exception=False) as sp:
                outcome, err = "ok", None
                try:
                    return await self.bulkheads.run(name, lambda: breaker.call(lambda: with_timeout(
                        lambda: system.compensate(wm.hire_id, key, today), call_timeout, f"{name}.compensate")))
                except BaseException as exc:
                    err = type(exc).__name__
                    outcome = {"CallTimeout": "timeout", "CircuitOpenError": "breaker_open",
                               "PermanentError": "permanent"}.get(err, "transient")
                    telemetry.mark_error(sp, exc, expected=outcome == "breaker_open")
                    raise
                finally:
                    sp.set_attribute("hiresync.outcome", outcome)
                    telemetry.count("sys_calls", system=name, outcome=outcome, op="compensate", **{"error.type": err})
                    telemetry.observe("sys_duration", (time.monotonic() - started) * 1000, system=name, outcome=outcome)

        def on_retry(att: int, exc: Exception, wait_s: Optional[float] = None) -> None:
            retry_history.append({
                "attempt": att,
                "error": str(exc),
                "wait_s": wait_s,
                "strategy": "exponential_backoff"
            })

        try:
            response, attempts = await retry_async(
                attempt, self.retry_policy, on_retry=on_retry, budget_s=self.timeouts.retry_budget
            )
            self.store.record_side_effect(
                key, wm.run_id, name, "compensated",
                request={"action": "undo", "reason": reason},
                response=response, attempts=attempts
            )
            return {
                "status": "compensated",
                "response": response,
                "attempts": attempts,
                "retries": retry_history
            }
        except CircuitOpenError as exc:
            self.store.record_side_effect(
                key, wm.run_id, name, "deferred",
                request={"action": "undo", "reason": reason},
                error=str(exc)
            )
            return {"status": "deferred", "error": str(exc), "retries": retry_history}
        except (TransientError, PermanentError) as exc:
            kind = "transient" if isinstance(exc, TransientError) else "permanent"
            attempts = getattr(exc, "attempts", n_attempt["n"])
            self.store.record_side_effect(
                key, wm.run_id, name, "compensation_failed",
                request={"action": "undo", "reason": reason},
                attempts=attempts, error=str(exc)
            )
            self.store.dead_letter(wm.run_id, name, {"action": "undo", "key": key}, f"{kind}: {exc}")
            telemetry.count("dlq", kind="compensate", topic=f"compensate:{name}")
            return {
                "status": "failed",
                "error": str(exc),
                "kind": kind,
                "attempts": attempts,
                "retries": retry_history
            }

    async def compensate_run(self, run_id: str, reason: str = "cancelled",
                             systems: Optional[list[str]] = None) -> dict:
        """Saga compensation: undo side effects in REVERSE dependency order."""
        owned, _ = await self.app.idem.claim("runlock", run_id)
        if not owned:
            log.info("%s is being driven by another worker", run_id)
            return {"run_id": run_id, "status": "locked"}
        try:
            return await self._compensate_run(run_id, reason, systems)
        finally:
            await self.app.idem.release("runlock", run_id)

    async def _compensate_run(self, run_id: str, reason: str,
                              systems: Optional[list[str]] = None) -> dict:
        run = self.store.get_run(run_id)
        if not run:
            raise KeyError(f"unknown run {run_id}")
        wm = WorkingMemory.from_dict(run["state"])
        today = self.today()

        if systems is not None:
            to_undo = set(systems)
        else:
            to_undo = set()
            for n in self.systems:
                if wm.results.get(n, {}).get("status") == "succeeded":
                    to_undo.add(n)
                elif self.store.last_pushed(wm.hire_id, n):
                    to_undo.add(n)
            for se in self.store.side_effects_for_run(run_id):
                if se.get("status") == "succeeded":
                    to_undo.add(se["system"])

        self._checkpoint(wm, RunStatus.RUNNING, Step.COMPENSATING)
        executed_order: list[str] = []

        # Reverse topological order
        for level in reversed(self.levels):
            level_systems = [n for n in level if n in to_undo]
            if not level_systems:
                continue
            outcomes = await asyncio.gather(*(self._compensate_system(wm, n, reason, today) for n in level_systems))
            for name, out in zip(level_systems, outcomes):
                wm.compensations[name] = out
                executed_order.append(name)
            self._checkpoint(wm, RunStatus.RUNNING, Step.COMPENSATING)

        bad = {n: r for n, r in wm.compensations.items() if r.get("status") != "compensated"}
        final_status = RunStatus.PARTIAL if bad else RunStatus.COMPENSATED
        wm.compensation_status = final_status.value
        self._checkpoint(wm, final_status, Step.DONE)

        self.store.audit(wm.hire_id, wm.run_id, "orchestrator", "run.compensated", {
            "status": final_status.value, "reason": reason, "order": executed_order,
            "failed_systems": list(bad.keys())
        })
        self.app.webhooks.emit(f"run.{final_status.value}", {
            "run_id": wm.run_id, "hire_id": wm.hire_id, "reason": reason,
            "compensations": {n: r.get("status") for n, r in wm.compensations.items()}
        })
        return {
            "run_id": run_id,
            "hire_id": wm.hire_id,
            "status": final_status.value,
            "compensations": wm.compensations,
            "reverse_order": executed_order,
        }

    async def cancel_hire(self, hire_id: str, reason: str = "cancelled",
                          principal: Optional[Principal] = None) -> dict:
        """Cancel a hire: record cancellation in hire_status and trigger saga compensation."""
        actor = principal.email if principal else "system"
        self.store.set_hire_status(hire_id, "cancelled", reason, actor)
        latest_version = self.store.get_version(hire_id)
        if not latest_version:
            return {"hire_id": hire_id, "status": "cancelled", "note": "no versions found"}

        run_id = f"{hire_id}:v{latest_version['version']}"
        run = self.store.get_run(run_id)
        if run:
            comp_res = await self.compensate_run(run_id, reason=reason)
            return {"hire_id": hire_id, "status": "cancelled", "run_compensation": comp_res}

        runs = self.store.list_runs(hire_id=hire_id)
        if runs:
            comp_res = await self.compensate_run(runs[-1]["run_id"], reason=reason)
            return {"hire_id": hire_id, "status": "cancelled", "run_compensation": comp_res}

        return {"hire_id": hire_id, "status": "cancelled"}

    async def recover(self) -> int:
        """On startup: resume every run a crash left mid-flight."""
        stuck = self.store.list_runs(status=RunStatus.RUNNING.value)
        for r in stuck:
            await self.drive(r["run_id"])
        return len(stuck)

    async def sweep(self) -> dict:
        """Periodic housekeeping (the sweeper task): time-bound everything that
        waits on something outside a single handler."""
        redriven = 0
        for r in self.store.stale_runs(self.timeouts.stuck_run):
            self.store.audit(r["hire_id"], r["run_id"], "sweeper", "run.stuck_redrive", {"since": r["updated_at"]})
            await self.drive(r["run_id"])  # the run lock stops a double-drive
            redriven += 1
        escalated = 0
        for a in self.store.approvals_to_escalate(self.timeouts.approval_sla):
            if self.store.mark_escalated(a["approval_id"]):
                escalated += 1
                await self.app.slack.post("#people-ops-leads",
                                          f"⏰ [{a['approval_id']}] {a['hire_id']} has waited over "
                                          f"{self.timeouts.approval_sla / 3600:g}h for approval ({a['reason']}).")
        dead = self.store.dead_events()
        return {"stuck_runs_redriven": redriven, "approvals_escalated": escalated, "dead_events": len(dead),
                "dead_writes": len(self.store.list_dead_letters())}

    async def redrive_write(self, dl_id: int, principal: Principal) -> dict:
        """DLQ action for a failed downstream write: re-run the run's remaining writes."""
        require(principal, "dlq:manage", f"dead letter {dl_id}")
        dl = self.store.get_dead_letter(dl_id)
        if not dl or dl["resolved_at"]:
            return {"id": dl_id, "status": "nothing to redrive"}
        self.store.audit(None, dl["run_id"], principal.sub, "dlq.write_redriven", {"id": dl_id, "system": dl["system"]})
        return await self.retry_run(dl["run_id"])

    # --------------------------------------------------------------- helpers
    def _checkpoint(self, wm: WorkingMemory, status: RunStatus, step: Step) -> Step:
        self.store.checkpoint(wm.run_id, wm.hire_id, wm.from_version, wm.to_version,
                              status.value, step.value, wm.to_dict())
        return step

    def _superseded(self, wm: WorkingMemory) -> bool:
        latest = self.store.get_version(wm.hire_id)
        if latest and latest["version"] > wm.to_version:
            self._supersede(wm, by=f"{wm.hire_id}:v{latest['version']}")
            return True
        return False

    def _supersede(self, wm: WorkingMemory, by: str) -> None:
        wm.notes.append(f"superseded by {by}")
        self._checkpoint(wm, RunStatus.SUPERSEDED, Step.DONE)
        for a in self.store.list_approvals(status="pending", run_id=wm.run_id):
            self.store.decide_approval(a["approval_id"], "superseded", "system", f"superseded by {by}", None)


def _flatten_patch(patch: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in patch.items():
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_patch(v, p))
        else:
            out[p] = v
    return out

