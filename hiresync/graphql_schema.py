"""GraphQL API. The data is a graph - a hire has versions, each version has a diff
and field provenance pointing at documents, each version has a run with side
effects and maybe an approval - and the approver UI, a Slack bot and MCP each need
a different slice of it.

Every resolver authorizes against the request's Principal (Bearer token, see
auth.py) using the same functions as the MCP tools: RBAC permission, ABAC on the
hire, document ACL, PII masking. Lists are filtered, not errored, so a hiring
manager's `hires` simply contains their own hires.
"""
from __future__ import annotations

import base64
import json
from typing import Optional

import strawberry
from strawberry.scalars import JSON
from strawberry.types import Info

from . import tools as toolreg
from .auth import (ANONYMOUS, Forbidden, Principal, can_access_hire, can_read_document, mask_record, require,
                   require_hire)


def _app(info: Info):
    return info.context["app"]


def _p(info: Info) -> Principal:
    return info.context.get("principal") or ANONYMOUS


def _deny(info: Info, exc: Forbidden):
    _app(info).store.audit(None, None, exc.principal.sub, "authz.denied",
                           {"action": exc.action, "resource": exc.resource, "reason": exc.reason, "api": "graphql"})
    raise PermissionError(f"forbidden: {exc.reason}") from None


def _guard(info: Info, fn):
    try:
        return fn()
    except Forbidden as exc:
        _deny(info, exc)


def _hire_access(info: Info, hire_id: str) -> dict:
    v = _app(info).store.get_version(hire_id)
    _guard(info, lambda: require_hire(_p(info), hire_id, v["data"] if v else None))
    return v


@strawberry.type
class FieldChange:
    path: str
    op: str
    old: Optional[JSON]
    new: Optional[JSON]


def _fc(rows: list[dict]) -> list[FieldChange]:
    return [FieldChange(path=c["path"], op=c["op"], old=c["old"], new=c["new"]) for c in rows]


@strawberry.type
class Provenance:
    version: int
    path: str
    value: Optional[JSON]
    source: str
    actor: str
    doc_hash: Optional[str]
    citation: Optional[JSON]
    trace_id: Optional[str]


@strawberry.type
class HireVersion:
    hire_id: str
    version: int
    source: str
    actor: str
    created_at: str
    content_hash: str
    data: JSON
    changes: list[FieldChange]


@strawberry.type
class SideEffect:
    system: str
    status: str
    attempts: int
    last_error: Optional[str]
    idempotency_key: str


@strawberry.type
class Approval:
    approval_id: str
    run_id: str
    hire_id: str
    status: str
    reason: str
    recommendation: Optional[JSON]
    decided_by: Optional[str]
    note: Optional[str]


def _approval(r: dict) -> Approval:
    return Approval(approval_id=r["approval_id"], run_id=r["run_id"], hire_id=r["hire_id"], status=r["status"],
                    reason=r["reason"], recommendation=json.loads(r["recommendation"]) if r["recommendation"] else None,
                    decided_by=r["decided_by"], note=r["note"])


@strawberry.type
class Run:
    run_id: str
    hire_id: str
    from_version: Optional[int]
    to_version: int
    status: str
    step: str
    state: strawberry.Private[dict]

    @strawberry.field
    def issues(self) -> JSON:
        return self.state["issues"]

    @strawberry.field
    def targets(self) -> list[str]:
        return self.state["targets"]

    @strawberry.field
    def recommendation(self) -> Optional[JSON]:
        return self.state["recommendation"]

    @strawberry.field
    def results(self) -> JSON:
        return self.state["results"]

    @strawberry.field
    def reconciliation(self) -> JSON:
        return self.state["reconciliation"]

    @strawberry.field
    def trace_id(self) -> Optional[str]:
        return self.state.get("trace_id")

    @strawberry.field
    def notes(self) -> list[str]:
        return self.state["notes"]

    @strawberry.field
    def side_effects(self, info: Info) -> list[SideEffect]:
        return [SideEffect(system=s["system"], status=s["status"], attempts=s["attempts"],
                           last_error=s["last_error"], idempotency_key=s["idempotency_key"])
                for s in _app(info).store.side_effects_for_run(self.run_id)]

    @strawberry.field
    def approval(self, info: Info) -> Optional[Approval]:
        rows = _app(info).store.list_approvals(run_id=self.run_id)
        return _approval(rows[-1]) if rows else None


def _run(r: dict) -> Run:
    return Run(run_id=r["run_id"], hire_id=r["hire_id"], from_version=r["from_version"],
               to_version=r["to_version"], status=r["status"], step=r["step"], state=r["state"])


@strawberry.type
class Document:
    doc_hash: str
    hire_id: str
    doc_type: str
    classification: str
    status: str
    uploads: JSON
    scan: JSON
    extraction: Optional[JSON]


def _doc(d: dict) -> Document:
    return Document(doc_hash=d["doc_hash"], hire_id=d["hire_id"], doc_type=d["doc_type"],
                    classification=d["classification"], status=d["status"],
                    uploads=[{"uploaded_by": u["uploaded_by"], "filename": u["filename"], "at": u["uploaded_at"]}
                             for u in d["uploads"]], scan=d["scan"], extraction=d["extraction"])


@strawberry.type
class SystemState:
    system: str
    record: Optional[JSON]


@strawberry.type
class Hire:
    hire_id: str

    @strawberry.field
    def current_version(self, info: Info) -> int:
        return _app(info).store.get_version(self.hire_id)["version"]

    @strawberry.field
    def current(self, info: Info) -> JSON:
        return mask_record(_p(info), _app(info).store.get_version(self.hire_id)["data"])

    @strawberry.field
    def versions(self, info: Info) -> list[HireVersion]:
        p, out, prev = _p(info), [], None
        for v in _app(info).store.list_versions(self.hire_id):
            out.append(HireVersion(hire_id=self.hire_id, version=v["version"], source=v["source"], actor=v["actor"],
                                   created_at=v["created_at"], content_hash=v["content_hash"],
                                   data=mask_record(p, v["data"]),
                                   changes=_fc(toolreg._masked_changes(p, prev, v["data"]))))
            prev = v["data"]
        return out

    @strawberry.field
    def diff(self, info: Info, from_version: int, to_version: int) -> list[FieldChange]:
        s = _app(info).store
        old = s.get_version(self.hire_id, from_version) if from_version else None
        return _fc(toolreg._masked_changes(_p(info), old["data"] if old else None,
                                           s.get_version(self.hire_id, to_version)["data"]))

    @strawberry.field
    async def provenance(self, info: Info, path: Optional[str] = None) -> list[Provenance]:
        rows = await toolreg._provenance(_app(info), toolreg.ProvenanceIn(hire_id=self.hire_id, path=path), _p(info))
        return [Provenance(version=r["version"], path=r["path"], value=r["value"], source=r["source"],
                           actor=r["actor"], doc_hash=r["doc_hash"], citation=r["citation"], trace_id=r["trace_id"])
                for r in rows]

    @strawberry.field
    async def systems(self, info: Info) -> list[SystemState]:
        _guard(info, lambda: require(_p(info), "hire:read_pii", "downstream records"))
        app = _app(info)
        return [SystemState(system=n, record=await s.get(self.hire_id)) for n, s in app.systems.items()]

    @strawberry.field
    def runs(self, info: Info) -> list[Run]:
        return [_run(r) for r in _app(info).store.list_runs(self.hire_id)]

    @strawberry.field
    def documents(self, info: Info) -> list[Document]:
        s, p = _app(info).store, _p(info)
        return [_doc(d) for d in s.list_documents(self.hire_id)
                if can_read_document(p, d, s.uploaders(d["doc_hash"]))[0]]

    @strawberry.field
    def audit(self, info: Info, limit: int = 100) -> JSON:
        _guard(info, lambda: require(_p(info), "audit:read", "audit log"))
        return _app(info).store.audit_trail(self.hire_id, limit)


@strawberry.type
class ToolInfo:
    name: str
    description: str
    read_only: bool
    input_schema: JSON


@strawberry.type
class Me:
    sub: str
    email: str
    groups: list[str]
    permissions: list[str]


@strawberry.type
class Query:
    @strawberry.field
    def me(self, info: Info) -> Me:
        p = _p(info)
        return Me(sub=p.sub, email=p.email, groups=sorted(p.groups), permissions=sorted(p.permissions))

    @strawberry.field
    def hire(self, info: Info, hire_id: str) -> Optional[Hire]:
        return Hire(hire_id=hire_id) if _hire_access(info, hire_id) else None

    @strawberry.field
    def hires(self, info: Info) -> list[Hire]:
        s, p = _app(info).store, _p(info)
        return [Hire(hire_id=h) for h in s.list_hire_ids()
                if can_access_hire(p, h, s.get_version(h)["data"])[0]]

    @strawberry.field
    def run(self, info: Info, run_id: str) -> Optional[Run]:
        r = _app(info).store.get_run(run_id)
        if not r:
            return None
        _hire_access(info, r["hire_id"])
        return _run(r)

    @strawberry.field
    async def pending_approvals(self, info: Info) -> list[Approval]:
        rows = await toolreg._pending(_app(info), toolreg.EmptyIn(), _p(info)) if (
            _p(info).has("approval:decide") or _p(info).has("approval:read")) else []
        by_id = {a["approval_id"]: a for a in _app(info).store.list_approvals(status="pending")}
        return [_approval(by_id[r["approval_id"]]) for r in rows]

    @strawberry.field
    def document(self, info: Info, doc_hash: str) -> Optional[Document]:
        s = _app(info).store
        d = s.get_document(doc_hash)
        if not d or not can_read_document(_p(info), d, s.uploaders(doc_hash))[0]:
            return None  # same answer for "missing" and "not yours"
        return _doc(d)

    @strawberry.field
    async def search_documents(self, info: Info, query: str, hire_id: Optional[str] = None,
                               doc_types: Optional[list[str]] = None, k: int = 5) -> JSON:
        _guard(info, lambda: require(_p(info), "document:search", "documents"))
        if hire_id:
            _hire_access(info, hire_id)
        args = toolreg.SearchIn(query=query, hire_id=hire_id, doc_types=doc_types, k=k)
        return await toolreg._search(_app(info), args, _p(info))

    @strawberry.field
    def episodes(self, info: Info) -> JSON:
        _guard(info, lambda: require(_p(info), "episodes:read", "episodes"))
        return _app(info).store.all_episodes()

    @strawberry.field
    def circuit_breakers(self, info: Info) -> JSON:
        _guard(info, lambda: require(_p(info), "audit:read", "breakers"))
        return [b.snapshot() for b in _app(info).orchestrator.breakers.values()]

    @strawberry.field
    def dead_letters(self, info: Info) -> JSON:
        """Unified DLQ: dead queue events + failed downstream writes."""
        _guard(info, lambda: require(_p(info), "dlq:read", "dead letters"))
        return toolreg.dead_letter_view(_app(info))

    @strawberry.field
    def error_analysis(self, info: Info) -> JSON:
        """Errors by span/system/error.type, latency percentiles, tokens and cost by prompt,
        from this process's spans. With an OTLP backend configured, use the backend."""
        _guard(info, lambda: require(_p(info), "audit:read", "telemetry"))
        from . import telemetry
        return {**telemetry.error_report(), "metrics": telemetry.metric_snapshot()}

    @strawberry.field
    async def model_spend(self, info: Info) -> JSON:
        out, err = await toolreg.execute(_app(info), "get_model_spend", {}, principal=_p(info))
        if err:
            raise PermissionError(json.loads(out)["error"])
        return json.loads(out)

    @strawberry.field
    async def webhooks(self, info: Info) -> JSON:
        out, err = await toolreg.execute(_app(info), "list_webhook_subscriptions", {}, principal=_p(info))
        if err:
            raise PermissionError(json.loads(out)["error"])
        return json.loads(out)

    @strawberry.field
    def workers(self, info: Info) -> JSON:
        _guard(info, lambda: require(_p(info), "audit:read", "workers"))
        bus = _app(info).bus
        return {"workers": getattr(bus, "n_workers", None), "in_flight": getattr(bus, "in_flight", None),
                "peak_in_flight": getattr(bus, "peak_in_flight", None), "stats": getattr(bus, "stats", None),
                "per_worker": getattr(bus, "per_worker", None)}

    @strawberry.field
    def audit_chain_valid(self, info: Info) -> bool:
        _guard(info, lambda: require(_p(info), "audit:read", "audit log"))
        return _app(info).store.verify_audit_chain()

    @strawberry.field
    def tools(self, info: Info) -> list[ToolInfo]:
        """The tools *this caller* may use."""
        return [ToolInfo(name=t.name, description=t.description, read_only=t.read_only,
                         input_schema=t.input_model.model_json_schema()) for t in toolreg.tools_for(_p(info))]


@strawberry.input
class ChangeInput:
    hire_id: str
    patch: JSON
    expected_version: Optional[int] = None
    reason: Optional[str] = None


@strawberry.type
class ChangeResult:
    hire_id: str
    version: Optional[int]
    noop: bool
    duplicate: bool
    trace_id: Optional[str]


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def submit_change(self, info: Info, input: ChangeInput) -> ChangeResult:
        p = _p(info)
        _guard(info, lambda: require(p, "hire:write", input.hire_id))
        if _app(info).store.get_version(input.hire_id):
            _hire_access(info, input.hire_id)
        r = await _app(info).submit_change(input.hire_id, input.patch, source="human", actor=p.email,
                                           event_id=None, expected_version=input.expected_version,
                                           reason=input.reason)
        return ChangeResult(hire_id=input.hire_id, version=r.get("version"), noop=r.get("noop", False),
                            duplicate=r.get("duplicate", False), trace_id=r.get("trace_id"))

    @strawberry.mutation
    async def upload_document(self, info: Info, hire_id: str, filename: str, content_base64: str,
                              mime: str = "text/plain", doc_type: str = "other") -> JSON:
        try:
            return await _app(info).documents.upload(hire_id, filename, base64.b64decode(content_base64), mime,
                                                     _p(info), doc_type)
        except Forbidden as exc:
            _deny(info, exc)

    @strawberry.mutation
    async def decide_approval(self, info: Info, approval_id: str, decision: str,
                              note: Optional[str] = None, patch: Optional[JSON] = None) -> JSON:
        if decision not in ("approved", "rejected"):
            raise ValueError("decision must be approved or rejected")
        try:
            return await _app(info).orchestrator.decide(approval_id, decision, _p(info), note, patch)
        except Forbidden as exc:
            _deny(info, exc)

    @strawberry.mutation
    async def retry_run(self, info: Info, run_id: str) -> JSON:
        _guard(info, lambda: require(_p(info), "run:retry", run_id))
        return await _app(info).orchestrator.retry_run(run_id)

    @strawberry.mutation
    async def redrive_dead_letter(self, info: Info, kind: str, id: int) -> JSON:
        out, err = await toolreg.execute(_app(info), "redrive_dead_letter", {"kind": kind, "id": id}, principal=_p(info))
        if err:
            raise PermissionError(json.loads(out)["error"])
        return json.loads(out)

    @strawberry.mutation
    async def discard_dead_letter(self, info: Info, kind: str, id: int, note: str) -> JSON:
        out, err = await toolreg.execute(_app(info), "discard_dead_letter", {"kind": kind, "id": id, "note": note},
                                         principal=_p(info))
        if err:
            raise PermissionError(json.loads(out)["error"])
        return json.loads(out)

    @strawberry.mutation
    async def create_webhook(self, info: Info, url: str, events: list[str]) -> JSON:
        out, err = await toolreg.execute(_app(info), "create_webhook_subscription", {"url": url, "events": events},
                                         principal=_p(info))
        if err:
            raise PermissionError(json.loads(out)["error"])
        return json.loads(out)

    @strawberry.mutation
    def inject_fault(self, info: Info, system: str, count: int = 1, kind: str = "transient") -> bool:
        """Demo only: make a mock system fail its next `count` calls."""
        _guard(info, lambda: require(_p(info), "admin:faults", system))
        _app(info).systems[system].fail_next(count, kind)
        return True


schema = strawberry.Schema(query=Query, mutation=Mutation)
