"""SQLite persistence. In production this is Postgres; the schema and the
transaction boundaries are what matter, and they carry over unchanged.

Tables
  hire_versions   append-only history of the canonical record (never updated)
  inbound_events  dedupe of source events (exactly-once *ingest*)
  outbox          transactional outbox -> event bus (no lost or phantom events)
  runs            orchestration checkpoints, one row per (hire, version)
  side_effects    one row per downstream write, keyed by idempotency key
  approvals       human-in-the-loop gate
  audit_log       hash-chained, append-only audit trail
  episodes        episodic memory of past runs and how they were resolved
  dead_letters    writes that exhausted retries or failed permanently
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from .telemetry import inject_carrier
from .versioning import canonical_json, content_hash

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS hire_versions (
  hire_id TEXT NOT NULL, version INTEGER NOT NULL, data TEXT NOT NULL,
  content_hash TEXT NOT NULL, source TEXT NOT NULL, actor TEXT NOT NULL,
  event_id TEXT, created_at TEXT NOT NULL,
  PRIMARY KEY (hire_id, version)
);
CREATE TABLE IF NOT EXISTS inbound_events (
  event_id TEXT PRIMARY KEY, hire_id TEXT NOT NULL, version INTEGER, received_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outbox (
  id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT NOT NULL, key TEXT NOT NULL,
  payload TEXT NOT NULL, created_at TEXT NOT NULL, published_at TEXT,
  processed_at TEXT, attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT, dead INTEGER NOT NULL DEFAULT 0,
  lease_owner TEXT, lease_until REAL, available_at REAL NOT NULL DEFAULT 0, first_failed_at TEXT,
  dead_at TEXT, redrives INTEGER NOT NULL DEFAULT 0, resolved_at TEXT, resolution TEXT, resolved_by TEXT
);
CREATE INDEX IF NOT EXISTS outbox_pending ON outbox (processed_at, dead, id);
CREATE INDEX IF NOT EXISTS outbox_key ON outbox (key, id);
CREATE TABLE IF NOT EXISTS idempotency_keys (
  scope TEXT NOT NULL, key TEXT NOT NULL, status TEXT NOT NULL, result TEXT, owner TEXT NOT NULL,
  expires_at REAL NOT NULL, PRIMARY KEY (scope, key)
);
CREATE TABLE IF NOT EXISTS doc_chunks (
  doc_hash TEXT NOT NULL, idx INTEGER NOT NULL, hire_id TEXT NOT NULL, char_start INTEGER NOT NULL,
  char_end INTEGER NOT NULL, text TEXT NOT NULL, embed_model TEXT NOT NULL, embedding BLOB NOT NULL,
  kind TEXT, heading TEXT, PRIMARY KEY (doc_hash, idx)
);
CREATE TABLE IF NOT EXISTS webhook_subscriptions (
  id TEXT PRIMARY KEY, url TEXT NOT NULL, events TEXT NOT NULL, secret TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 1, created_by TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS webhook_deliveries (
  id INTEGER PRIMARY KEY AUTOINCREMENT, sub_id TEXT NOT NULL, event_id TEXT NOT NULL, event_type TEXT NOT NULL,
  status_code INTEGER, at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS workflows (
  wf_id TEXT PRIMARY KEY, type TEXT NOT NULL, key TEXT NOT NULL, state TEXT NOT NULL, status TEXT NOT NULL,
  data TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS wf_timers (
  wf_id TEXT NOT NULL, name TEXT NOT NULL, fire_at REAL NOT NULL, fired INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (wf_id, name)
);
CREATE INDEX IF NOT EXISTS wf_timers_due ON wf_timers (fired, fire_at);
CREATE TABLE IF NOT EXISTS wf_signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT, wf_id TEXT NOT NULL, name TEXT NOT NULL, payload TEXT NOT NULL,
  created_at TEXT NOT NULL, consumed_at TEXT
);
CREATE TABLE IF NOT EXISTS wf_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT, wf_id TEXT NOT NULL, ts TEXT NOT NULL, from_state TEXT,
  to_state TEXT NOT NULL, note TEXT
);
CREATE TABLE IF NOT EXISTS hire_status (
  hire_id TEXT PRIMARY KEY, status TEXT NOT NULL, reason TEXT, actor TEXT, at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS login_reservations (
  login TEXT PRIMARY KEY, hire_id TEXT NOT NULL, reserved_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS episode_vectors (
  episode_id TEXT PRIMARY KEY, embed_model TEXT NOT NULL, embedding BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, hire_id TEXT NOT NULL, from_version INTEGER, to_version INTEGER NOT NULL,
  status TEXT NOT NULL, step TEXT NOT NULL, state TEXT NOT NULL,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS side_effects (
  idempotency_key TEXT PRIMARY KEY, run_id TEXT NOT NULL, system TEXT NOT NULL,
  status TEXT NOT NULL, request TEXT, response TEXT, attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
  approval_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, hire_id TEXT NOT NULL, status TEXT NOT NULL,
  reason TEXT NOT NULL, recommendation TEXT, decided_by TEXT, note TEXT, patch TEXT,
  created_at TEXT NOT NULL, decided_at TEXT, escalated_at TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, hire_id TEXT, run_id TEXT,
  actor TEXT NOT NULL, action TEXT NOT NULL, detail TEXT NOT NULL,
  prev_hash TEXT NOT NULL, hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
  episode_id TEXT PRIMARY KEY, hire_id TEXT NOT NULL, run_id TEXT NOT NULL,
  features TEXT NOT NULL, situation TEXT NOT NULL, decision TEXT NOT NULL,
  decided_by TEXT NOT NULL, outcome TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
  doc_hash TEXT PRIMARY KEY, hire_id TEXT NOT NULL, doc_type TEXT NOT NULL, classification TEXT NOT NULL,
  mime TEXT NOT NULL,
  size_bytes INTEGER NOT NULL, text_redacted TEXT NOT NULL, scan TEXT NOT NULL,
  first_uploaded_by TEXT NOT NULL, extraction TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS document_uploads (
  doc_hash TEXT NOT NULL, uploaded_by TEXT NOT NULL, filename TEXT NOT NULL, uploaded_at TEXT NOT NULL,
  trace_id TEXT, UNIQUE (doc_hash, uploaded_by, filename)
);
CREATE TABLE IF NOT EXISTS field_provenance (
  hire_id TEXT NOT NULL, version INTEGER NOT NULL, path TEXT NOT NULL, value TEXT,
  source TEXT NOT NULL, actor TEXT NOT NULL, doc_hash TEXT, citation TEXT, trace_id TEXT,
  recorded_at TEXT NOT NULL, PRIMARY KEY (hire_id, version, path)
);
CREATE TABLE IF NOT EXISTS llm_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, purpose TEXT NOT NULL,
  prompt TEXT NOT NULL, prompt_sha TEXT NOT NULL, model TEXT NOT NULL, effort TEXT NOT NULL,
  tools_sha TEXT, input_sha TEXT NOT NULL, input TEXT NOT NULL, output TEXT, outcome TEXT NOT NULL,
  error TEXT, reasks INTEGER, turns INTEGER, tool_calls TEXT, usage TEXT, latency_ms INTEGER,
  run_id TEXT, trace_id TEXT, tokens INTEGER NOT NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0,
  eval_retries INTEGER NOT NULL DEFAULT 0, eval_problems TEXT
);
CREATE TABLE IF NOT EXISTS dead_letters (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, system TEXT NOT NULL,
  payload TEXT NOT NULL, error TEXT NOT NULL, created_at TEXT NOT NULL, resolved_at TEXT,
  resolution TEXT, resolved_by TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class ConcurrencyError(Exception):
    """expected_version did not match the current version (optimistic lock)."""


class Store:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._conn.executescript(SCHEMA)

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def _q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    # ---------------------------------------------------------------- versions
    def append_version(
        self,
        hire_id: str,
        data: dict,
        source: str,
        actor: str,
        event_id: Optional[str],
        expected_version: Optional[int] = None,
        provenance: Optional[dict[str, dict]] = None,
        trace_id: Optional[str] = None,
    ) -> dict:
        """Append a new version, its field provenance and its outbox event in ONE
        transaction - either all of it is visible or none of it is.

        Returns {"version", "noop", "duplicate"}. Idempotent twice over:
        a replayed event_id is ignored, and identical content is a no-op.
        provenance: {dotted_path: {"doc_hash", "citation"}} for fields this change set;
        every changed field gets a provenance row either way.
        """
        from .versioning import diff as _diff

        digest = content_hash(data)
        trace_id = trace_id or f"tr_{uuid.uuid4().hex[:12]}"
        with self.tx() as c:
            if event_id:
                seen = c.execute(
                    "SELECT version FROM inbound_events WHERE event_id=?", (event_id,)
                ).fetchone()
                if seen:
                    return {"version": seen["version"], "noop": True, "duplicate": True}
            cur = c.execute(
                "SELECT version, content_hash FROM hire_versions WHERE hire_id=? ORDER BY version DESC LIMIT 1",
                (hire_id,),
            ).fetchone()
            current = cur["version"] if cur else 0
            if expected_version is not None and expected_version != current:
                raise ConcurrencyError(f"{hire_id}: expected v{expected_version}, current is v{current}")
            if cur and cur["content_hash"] == digest:
                if event_id:
                    c.execute(
                        "INSERT INTO inbound_events VALUES (?,?,?,?)", (event_id, hire_id, current, now())
                    )
                return {"version": current, "noop": True, "duplicate": False}
            version = current + 1
            c.execute(
                "INSERT INTO hire_versions VALUES (?,?,?,?,?,?,?,?)",
                (hire_id, version, canonical_json(data), digest, source, actor, event_id, now()),
            )
            if event_id:
                c.execute("INSERT INTO inbound_events VALUES (?,?,?,?)", (event_id, hire_id, version, now()))
            prev = c.execute("SELECT data FROM hire_versions WHERE hire_id=? AND version=?",
                             (hire_id, current)).fetchone()
            provenance = provenance or {}
            for ch in _diff(json.loads(prev["data"]) if prev else None, data):
                p = provenance.get(ch.path, {})
                c.execute(
                    "INSERT INTO field_provenance VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (hire_id, version, ch.path, canonical_json(ch.new), source, actor,
                     p.get("doc_hash"), canonical_json(p["citation"]) if p.get("citation") else None,
                     trace_id, now()),
                )
            payload = {"hire_id": hire_id, "version": version, "from_version": current or None,
                       "trace_id": trace_id, "_otel": inject_carrier()}
            c.execute(
                "INSERT INTO outbox (topic, key, payload, created_at) VALUES (?,?,?,?)",
                ("hire.changed", hire_id, canonical_json(payload), now()),
            )
            self._audit(c, hire_id, None, actor, "version.appended",
                        {"version": version, "source": source, "trace_id": trace_id})
            return {"version": version, "noop": False, "duplicate": False, "trace_id": trace_id}

    def get_version(self, hire_id: str, version: Optional[int] = None) -> Optional[dict]:
        if version is None:
            rows = self._q(
                "SELECT * FROM hire_versions WHERE hire_id=? ORDER BY version DESC LIMIT 1", (hire_id,)
            )
        else:
            rows = self._q("SELECT * FROM hire_versions WHERE hire_id=? AND version=?", (hire_id, version))
        return self._version_row(rows[0]) if rows else None

    def list_versions(self, hire_id: str) -> list[dict]:
        rows = self._q("SELECT * FROM hire_versions WHERE hire_id=? ORDER BY version", (hire_id,))
        return [self._version_row(r) for r in rows]

    def list_hire_ids(self) -> list[str]:
        return [r["hire_id"] for r in self._q("SELECT DISTINCT hire_id FROM hire_versions ORDER BY hire_id")]

    @staticmethod
    def _version_row(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["data"] = json.loads(d["data"])
        return d

    # ------------------------------------------------------------------ outbox
    def unpublished(self, limit: int = 100) -> list[dict]:
        rows = self._q("SELECT * FROM outbox WHERE published_at IS NULL ORDER BY id LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def enqueue(self, topic: str, key: str, payload: dict) -> None:
        payload = {**payload, "_otel": inject_carrier()}  # producer's trace context rides with the event
        with self.tx() as c:
            c.execute(
                "INSERT INTO outbox (topic, key, payload, created_at) VALUES (?,?,?,?)",
                (topic, key, canonical_json(payload), now()),
            )

    def mark_published(self, outbox_id: int) -> None:
        with self.tx() as c:
            c.execute("UPDATE outbox SET published_at=? WHERE id=?", (now(), outbox_id))

    # ------------------------------------------------- outbox as a durable queue
    def pending_events(self, limit: int = 100) -> list[dict]:
        rows = self._q("SELECT * FROM outbox WHERE processed_at IS NULL AND dead=0 ORDER BY id LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def claim_next(self, owner: str, lease_s: float) -> Optional[dict]:
        """Atomically lease the next deliverable event. Only the *head* of each
        key's queue is claimable (no earlier pending row for the same hire), so
        any number of workers keeps per-hire order while hires run in parallel.
        An expired lease (crashed worker) makes the row claimable again.
        Postgres: the same query with FOR UPDATE SKIP LOCKED."""
        t = time.time()
        with self.tx() as c:
            row = c.execute(
                """SELECT o.* FROM outbox o
                   WHERE o.processed_at IS NULL AND o.dead = 0 AND o.available_at <= ?
                     AND (o.lease_until IS NULL OR o.lease_until < ?)
                     AND NOT EXISTS (SELECT 1 FROM outbox p WHERE p.key = o.key AND p.id < o.id
                                     AND p.processed_at IS NULL AND p.dead = 0)
                   ORDER BY o.id LIMIT 1""", (t, t)).fetchone()
            if not row:
                return None
            c.execute("UPDATE outbox SET lease_owner=?, lease_until=? WHERE id=?", (owner, t + lease_s, row["id"]))
            d = dict(row)
            d["lease_owner"], d["lease_until"] = owner, t + lease_s
            return d

    def ack(self, outbox_id: int, owner: str) -> bool:
        """Mark processed. Fenced by lease owner: a worker whose lease expired and
        was taken over can't ack someone else's delivery."""
        with self.tx() as c:
            cur = c.execute("""UPDATE outbox SET processed_at=?, lease_owner=NULL, lease_until=NULL
                               WHERE id=? AND lease_owner=?""", (now(), outbox_id, owner))
            return cur.rowcount == 1

    def nack(self, outbox_id: int, owner: str, error: str, backoff_s: float, max_attempts: int) -> Optional[bool]:
        """Failed delivery: back off, or dead-letter after max_attempts.
        Returns True if dead-lettered, False if scheduled for retry, None if the lease was lost."""
        with self.tx() as c:
            row = c.execute("SELECT attempts, lease_owner FROM outbox WHERE id=?", (outbox_id,)).fetchone()
            if not row or row["lease_owner"] != owner:
                return None
            attempts = row["attempts"] + 1
            dead = attempts >= max_attempts
            c.execute("""UPDATE outbox SET attempts=?, last_error=?, first_failed_at=COALESCE(first_failed_at, ?),
                         lease_owner=NULL, lease_until=NULL, available_at=?, dead=?, dead_at=?
                         WHERE id=?""",
                      (attempts, error[:2000], now(), time.time() + backoff_s, int(dead), now() if dead else None,
                       outbox_id))
            if dead:
                self._audit(c, None, None, "worker", "dlq.event_dead",
                            {"outbox_id": outbox_id, "attempts": attempts, "error": error[:300]})
            return dead

    # ------------------------------------------------------ dead-letter queue
    def dead_events(self, include_resolved: bool = False) -> list[dict]:
        sql = "SELECT * FROM outbox WHERE dead=1" + ("" if include_resolved else " AND resolved_at IS NULL")
        return [dict(r) for r in self._q(sql + " ORDER BY id")]

    def redrive_event(self, outbox_id: int, actor: str) -> bool:
        with self.tx() as c:
            cur = c.execute("""UPDATE outbox SET dead=0, attempts=0, available_at=0, lease_owner=NULL,
                               lease_until=NULL, dead_at=NULL, redrives=redrives+1
                               WHERE id=? AND dead=1 AND resolved_at IS NULL""", (outbox_id,))
            if cur.rowcount:
                self._audit(c, None, None, actor, "dlq.event_redriven", {"outbox_id": outbox_id})
            return cur.rowcount == 1

    def discard_event(self, outbox_id: int, actor: str, note: str) -> bool:
        with self.tx() as c:
            cur = c.execute("""UPDATE outbox SET resolved_at=?, resolution=?, resolved_by=?
                               WHERE id=? AND dead=1 AND resolved_at IS NULL""",
                            (now(), f"discarded: {note}", actor, outbox_id))
            if cur.rowcount:
                self._audit(c, None, None, actor, "dlq.event_discarded", {"outbox_id": outbox_id, "note": note})
            return cur.rowcount == 1

    # ------------------------------------------------- idempotency keys (DB)
    def idem_claim(self, scope: str, key: str, owner: str, lease_s: float) -> tuple[bool, Optional[dict]]:
        """Atomic claim: INSERT .. ON CONFLICT DO NOTHING; an expired in-flight lease
        (crashed owner) is taken over. Returns (owned, existing_row)."""
        import time as _t
        t = _t.time()
        with self.tx() as c:
            c.execute("DELETE FROM idempotency_keys WHERE scope=? AND key=? AND status='in_flight' AND expires_at<?",
                      (scope, key, t))
            cur = c.execute(
                "INSERT INTO idempotency_keys VALUES (?,?,?,?,?,?) ON CONFLICT(scope, key) DO NOTHING",
                (scope, key, "in_flight", None, owner, t + lease_s))
            if cur.rowcount == 1:
                return True, None
            row = c.execute("SELECT * FROM idempotency_keys WHERE scope=? AND key=?", (scope, key)).fetchone()
            return False, dict(row) if row else None

    def idem_complete(self, scope: str, key: str, result: Any, ttl_s: float) -> None:
        import time as _t
        with self.tx() as c:
            c.execute("UPDATE idempotency_keys SET status='done', result=?, expires_at=? WHERE scope=? AND key=?",
                      (canonical_json(result), _t.time() + ttl_s, scope, key))

    def idem_release(self, scope: str, key: str) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM idempotency_keys WHERE scope=? AND key=? AND status='in_flight'", (scope, key))

    # ------------------------------------------------------------- vectors
    def insert_chunks(self, rows: list[tuple]) -> int:
        """rows: (doc_hash, idx, hire_id, start, end, text, model, embedding_bytes, kind, heading). Deduped by PK."""
        n = 0
        with self.tx() as c:
            for r in rows:
                n += c.execute("INSERT INTO doc_chunks VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING", r).rowcount
        return n

    def all_chunks(self) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM doc_chunks ORDER BY doc_hash, idx")]

    def chunks_for(self, doc_hash: str) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM doc_chunks WHERE doc_hash=? ORDER BY idx", (doc_hash,))]

    def upsert_episode_vector(self, episode_id: str, model: str, embedding: bytes) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO episode_vectors VALUES (?,?,?)", (episode_id, model, embedding))

    def episode_vectors(self) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM episode_vectors")]

    def episode(self, episode_id: str) -> Optional[dict]:
        rows = self._q("SELECT * FROM episodes WHERE episode_id=?", (episode_id,))
        if not rows:
            return None
        d = dict(rows[0])
        d["features"] = json.loads(d["features"])
        return d

    # -------------------------------------------------------------------- runs
    def get_run(self, run_id: str) -> Optional[dict]:
        rows = self._q("SELECT * FROM runs WHERE run_id=?", (run_id,))
        if not rows:
            return None
        d = dict(rows[0])
        d["state"] = json.loads(d["state"])
        return d

    def list_runs(self, hire_id: Optional[str] = None, status: Optional[str] = None) -> list[dict]:
        sql, args = "SELECT * FROM runs WHERE 1=1", []
        if hire_id:
            sql += " AND hire_id=?"
            args.append(hire_id)
        if status:
            sql += " AND status=?"
            args.append(status)
        rows = self._q(sql + " ORDER BY created_at", tuple(args))
        out = []
        for r in rows:
            d = dict(r)
            d["state"] = json.loads(d["state"])
            out.append(d)
        return out

    def checkpoint(self, run_id: str, hire_id: str, from_v: Optional[int], to_v: int,
                   status: str, step: str, state: dict) -> None:
        """Durable checkpoint: the full working memory is saved at every step,
        so a crashed or restarted worker resumes exactly where it stopped."""
        with self.tx() as c:
            c.execute(
                """INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(run_id) DO UPDATE SET status=excluded.status, step=excluded.step,
                   state=excluded.state, updated_at=excluded.updated_at""",
                (run_id, hire_id, from_v, to_v, status, step, canonical_json(state), now(), now()),
            )
            self._audit(c, hire_id, run_id, "orchestrator", "run.checkpoint", {"status": status, "step": step})

    # ------------------------------------------------------------ side effects
    def get_side_effect(self, key: str) -> Optional[dict]:
        rows = self._q("SELECT * FROM side_effects WHERE idempotency_key=?", (key,))
        return dict(rows[0]) if rows else None

    def record_side_effect(self, key: str, run_id: str, system: str, status: str,
                           request: Any = None, response: Any = None, attempts: int = 0,
                           error: Optional[str] = None) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO side_effects VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(idempotency_key) DO UPDATE SET status=excluded.status,
                   response=excluded.response, attempts=side_effects.attempts+excluded.attempts,
                   last_error=excluded.last_error, updated_at=excluded.updated_at""",
                (key, run_id, system, status, canonical_json(request), canonical_json(response),
                 attempts, error, now()),
            )

    def last_pushed(self, hire_id: str, system: str) -> Optional[dict]:
        """The last payload a system accepted for this hire: the baseline a new
        version is compared against (not the previous version - that one may
        never have been propagated if it was superseded or failed)."""
        rows = self._q(
            """SELECT request, response FROM side_effects WHERE system=? AND status='succeeded'
               AND idempotency_key LIKE ? ORDER BY updated_at DESC, rowid DESC LIMIT 1""",
            (system, f"{hire_id}:%"),
        )
        if not rows:
            return None
        return {"request": json.loads(rows[0]["request"]), "response": json.loads(rows[0]["response"])}

    def side_effects_for_run(self, run_id: str) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM side_effects WHERE run_id=? ORDER BY system", (run_id,))]

    # --------------------------------------------------------------- approvals
    def create_approval(self, run_id: str, hire_id: str, reason: str, recommendation: dict) -> str:
        approval_id = f"apr_{uuid.uuid4().hex[:10]}"
        with self.tx() as c:
            existing = c.execute(
                "SELECT approval_id FROM approvals WHERE run_id=? AND status='pending'", (run_id,)
            ).fetchone()
            if existing:  # idempotent: re-entering the gate never creates a second ask
                return existing["approval_id"]
            c.execute(
                "INSERT INTO approvals VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (approval_id, run_id, hire_id, "pending", reason, canonical_json(recommendation),
                 None, None, None, now(), None, None),
            )
            self._audit(c, hire_id, run_id, "orchestrator", "approval.requested", {"approval_id": approval_id})
        return approval_id

    def decide_approval(self, approval_id: str, decision: str, actor: str,
                        note: Optional[str], patch: Optional[dict]) -> dict:
        with self.tx() as c:
            row = c.execute("SELECT * FROM approvals WHERE approval_id=?", (approval_id,)).fetchone()
            if not row:
                raise KeyError(approval_id)
            if row["status"] != "pending":
                return dict(row)  # idempotent: a double-click doesn't decide twice
            c.execute(
                "UPDATE approvals SET status=?, decided_by=?, note=?, patch=?, decided_at=? WHERE approval_id=?",
                (decision, actor, note, canonical_json(patch) if patch else None, now(), approval_id),
            )
            self._audit(c, row["hire_id"], row["run_id"], actor, f"approval.{decision}",
                        {"approval_id": approval_id, "note": note, "patch": patch})
            return dict(c.execute("SELECT * FROM approvals WHERE approval_id=?", (approval_id,)).fetchone())

    def list_approvals(self, status: Optional[str] = None, run_id: Optional[str] = None) -> list[dict]:
        sql, args = "SELECT * FROM approvals WHERE 1=1", []
        if status:
            sql += " AND status=?"
            args.append(status)
        if run_id:
            sql += " AND run_id=?"
            args.append(run_id)
        return [dict(r) for r in self._q(sql + " ORDER BY created_at", tuple(args))]

    # ------------------------------------------------------------------- audit
    def audit(self, hire_id: Optional[str], run_id: Optional[str], actor: str, action: str, detail: dict) -> None:
        with self.tx() as c:
            self._audit(c, hire_id, run_id, actor, action, detail)

    def _audit(self, c, hire_id, run_id, actor, action, detail) -> None:
        prev = c.execute("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
        prev_hash = prev["hash"] if prev else "genesis"
        ts = now()
        body = canonical_json([ts, hire_id, run_id, actor, action, detail, prev_hash])
        c.execute(
            "INSERT INTO audit_log (ts, hire_id, run_id, actor, action, detail, prev_hash, hash) VALUES (?,?,?,?,?,?,?,?)",
            (ts, hire_id, run_id, actor, action, canonical_json(detail), prev_hash, content_hash(body)),
        )

    def audit_trail(self, hire_id: Optional[str] = None, limit: int = 500) -> list[dict]:
        if hire_id:
            rows = self._q("SELECT * FROM audit_log WHERE hire_id=? ORDER BY id LIMIT ?", (hire_id, limit))
        else:
            rows = self._q("SELECT * FROM audit_log ORDER BY id LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def verify_audit_chain(self) -> bool:
        prev_hash = "genesis"
        for r in self._q("SELECT * FROM audit_log ORDER BY id"):
            body = canonical_json([r["ts"], r["hire_id"], r["run_id"], r["actor"], r["action"],
                                   json.loads(r["detail"]), r["prev_hash"]])
            if r["prev_hash"] != prev_hash or r["hash"] != content_hash(body):
                return False
            prev_hash = r["hash"]
        return True

    # ---------------------------------------------------------------- episodes
    def save_episode(self, hire_id: str, run_id: str, features: list[str], situation: str,
                     decision: str, decided_by: str, outcome: str) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?)",
                (f"ep_{run_id}", hire_id, run_id, canonical_json(sorted(features)), situation,
                 decision, decided_by, outcome, now()),
            )

    def all_episodes(self) -> list[dict]:
        rows = self._q("SELECT * FROM episodes ORDER BY created_at DESC")
        out = []
        for r in rows:
            d = dict(r)
            d["features"] = json.loads(d["features"])
            out.append(d)
        return out

    # --------------------------------------------------------------- documents
    def insert_document(self, doc_hash: str, hire_id: str, doc_type: str, classification: str, mime: str,
                        size: int, text_redacted: str, scan: dict, uploaded_by: str, filename: str,
                        trace_id: str) -> bool:
        """Atomic dedupe. Returns True if this call created the document.
        The upload row is always recorded (who uploaded what, when) but is itself
        unique per (hash, uploader, filename), so a double-click adds nothing."""
        with self.tx() as c:
            cur = c.execute(
                """INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(doc_hash) DO NOTHING""",
                (doc_hash, hire_id, doc_type, classification, mime, size, text_redacted, canonical_json(scan),
                 uploaded_by, None, "received", now()),
            )
            created = cur.rowcount == 1
            c.execute(
                "INSERT INTO document_uploads VALUES (?,?,?,?,?) ON CONFLICT DO NOTHING",
                (doc_hash, uploaded_by, filename, now(), trace_id),
            )
            self._audit(c, hire_id, None, uploaded_by,
                        "document.created" if created else "document.duplicate_upload",
                        {"doc_hash": doc_hash, "filename": filename, "trace_id": trace_id})
            return created

    def set_document_extraction(self, doc_hash: str, status: str, extraction: Optional[dict]) -> None:
        with self.tx() as c:
            c.execute("UPDATE documents SET status=?, extraction=? WHERE doc_hash=?",
                      (status, canonical_json(extraction) if extraction else None, doc_hash))

    def doc_owner_hire(self, doc_hash: str) -> Optional[str]:
        rows = self._q("SELECT hire_id FROM documents WHERE doc_hash=?", (doc_hash,))
        return rows[0]["hire_id"] if rows else None

    def uploaders(self, doc_hash: str) -> set[str]:
        return {r["uploaded_by"] for r in self._q("SELECT uploaded_by FROM document_uploads WHERE doc_hash=?",
                                                   (doc_hash,))}

    def get_document(self, doc_hash: str) -> Optional[dict]:
        rows = self._q("SELECT * FROM documents WHERE doc_hash=?", (doc_hash,))
        if not rows:
            return None
        d = dict(rows[0])
        d["scan"] = json.loads(d["scan"])
        d["extraction"] = json.loads(d["extraction"]) if d["extraction"] else None
        d["uploads"] = [dict(r) for r in self._q(
            "SELECT * FROM document_uploads WHERE doc_hash=? ORDER BY uploaded_at", (doc_hash,))]
        return d

    def list_documents(self, hire_id: Optional[str] = None) -> list[dict]:
        sql, args = "SELECT doc_hash FROM documents", ()
        if hire_id:
            sql, args = sql + " WHERE hire_id=?", (hire_id,)
        return [self.get_document(r["doc_hash"]) for r in self._q(sql + " ORDER BY created_at", args)]

    def provenance(self, hire_id: str, path: Optional[str] = None) -> list[dict]:
        sql, args = "SELECT * FROM field_provenance WHERE hire_id=?", [hire_id]
        if path:
            sql += " AND path=?"
            args.append(path)
        out = []
        for r in self._q(sql + " ORDER BY version, path", tuple(args)):
            d = dict(r)
            d["value"] = json.loads(d["value"]) if d["value"] else None
            d["citation"] = json.loads(d["citation"]) if d["citation"] else None
            out.append(d)
        return out

    # --------------------------------------------------------------- llm calls
    def record_llm_call(self, purpose: str, call: dict, run_id: Optional[str] = None,
                        trace_id: Optional[str] = None) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO llm_calls (ts, purpose, prompt, prompt_sha, model, effort, tools_sha, input_sha,
                   input, output, outcome, error, reasks, turns, tool_calls, usage, latency_ms, run_id, trace_id,
                   tokens, cost_usd, eval_retries, eval_problems)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (now(), purpose, call["prompt"], call["prompt_sha"], call["model"], call["effort"],
                 call.get("tools_sha"), call["input_sha"], call["input"],
                 canonical_json(call["output"]) if call.get("output") is not None else None, call["outcome"],
                 call.get("error"), call.get("reasks"), call.get("turns"), canonical_json(call.get("tool_calls", [])),
                 canonical_json(call.get("usage", {})), call.get("latency_ms"), run_id, trace_id,
                 call.get("tokens", 0), call.get("cost_usd", 0.0), call.get("eval_retries", 0),
                 canonical_json(call.get("eval_problems")) if call.get("eval_problems") else None),
            )

    def llm_spend_for_run(self, run_id: str) -> tuple[int, float]:
        r = self._q("SELECT COALESCE(SUM(tokens),0) t, COALESCE(SUM(cost_usd),0) c FROM llm_calls WHERE run_id=?",
                    (run_id,))[0]
        return int(r["t"]), float(r["c"])

    def llm_spend_since(self, iso_day: str) -> float:
        return float(self._q("SELECT COALESCE(SUM(cost_usd),0) c FROM llm_calls WHERE ts >= ?", (iso_day,))[0]["c"])

    def llm_calls(self, purpose: Optional[str] = None, prompt: Optional[str] = None) -> list[dict]:
        sql, args = "SELECT * FROM llm_calls WHERE 1=1", []
        if purpose:
            sql += " AND purpose=?"
            args.append(purpose)
        if prompt:
            sql += " AND prompt=?"
            args.append(prompt)
        out = []
        for r in self._q(sql + " ORDER BY id", tuple(args)):
            d = dict(r)
            for k in ("output", "tool_calls", "usage", "eval_problems"):
                d[k] = json.loads(d[k]) if d[k] else None
            out.append(d)
        return out

    # ------------------------------------------------------------ dead letters
    def dead_letter(self, run_id: str, system: str, payload: dict, error: str) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO dead_letters (run_id, system, payload, error, created_at) VALUES (?,?,?,?,?)",
                (run_id, system, canonical_json(payload), error, now()),
            )

    def resolve_dead_letters(self, run_id: str, system: str, resolution: str = "succeeded on retry",
                             actor: str = "orchestrator") -> None:
        with self.tx() as c:
            c.execute(
                """UPDATE dead_letters SET resolved_at=?, resolution=?, resolved_by=?
                   WHERE run_id=? AND system=? AND resolved_at IS NULL""",
                (now(), resolution, actor, run_id, system),
            )

    def get_dead_letter(self, dl_id: int) -> Optional[dict]:
        rows = self._q("SELECT * FROM dead_letters WHERE id=?", (dl_id,))
        return dict(rows[0]) if rows else None

    def discard_dead_letter(self, dl_id: int, actor: str, note: str) -> bool:
        with self.tx() as c:
            cur = c.execute("""UPDATE dead_letters SET resolved_at=?, resolution=?, resolved_by=?
                               WHERE id=? AND resolved_at IS NULL""", (now(), f"discarded: {note}", actor, dl_id))
            if cur.rowcount:
                self._audit(c, None, None, actor, "dlq.write_discarded", {"id": dl_id, "note": note})
            return cur.rowcount == 1

    # -------------------------------------------------------------- webhooks
    def add_webhook(self, sub_id: str, url: str, events: list[str], secret: str, created_by: str) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO webhook_subscriptions VALUES (?,?,?,?,1,?,?)",
                      (sub_id, url, canonical_json(sorted(events)), secret, created_by, now()))
            self._audit(c, None, None, created_by, "webhook.subscribed", {"id": sub_id, "url": url, "events": events})

    def webhooks_for(self, event_type: str) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM webhook_subscriptions WHERE active=1")
                if event_type in json.loads(r["events"])]

    def get_webhook(self, sub_id: str) -> Optional[dict]:
        rows = self._q("SELECT * FROM webhook_subscriptions WHERE id=?", (sub_id,))
        return dict(rows[0]) if rows else None

    def list_webhooks(self) -> list[dict]:
        return [{k: r[k] for k in ("id", "url", "events", "active", "created_by", "created_at")}
                for r in self._q("SELECT * FROM webhook_subscriptions ORDER BY created_at")]

    def record_webhook_delivery(self, sub_id: str, event_id: str, event_type: str, status: int) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO webhook_deliveries (sub_id, event_id, event_type, status_code, at) VALUES (?,?,?,?,?)",
                      (sub_id, event_id, event_type, status, now()))

    def webhook_deliveries(self, sub_id: Optional[str] = None) -> list[dict]:
        if sub_id:
            return [dict(r) for r in self._q("SELECT * FROM webhook_deliveries WHERE sub_id=? ORDER BY id", (sub_id,))]
        return [dict(r) for r in self._q("SELECT * FROM webhook_deliveries ORDER BY id")]

    # ------------------------------------------------------------- workflows
    def wf_create(self, wf_id: str, wf_type: str, key: str, state: str, data: dict) -> bool:
        """Idempotent start: the same wf_id never creates a second workflow."""
        with self.tx() as c:
            cur = c.execute("INSERT INTO workflows VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(wf_id) DO NOTHING",
                            (wf_id, wf_type, key, state, "running", canonical_json(data), now(), now()))
            if cur.rowcount:
                c.execute("INSERT INTO wf_history (wf_id, ts, from_state, to_state, note) VALUES (?,?,?,?,?)",
                          (wf_id, now(), None, state, "started"))
                c.execute("INSERT INTO outbox (topic, key, payload, created_at) VALUES (?,?,?,?)",
                          ("workflow.wake", wf_id, canonical_json({"wf_id": wf_id}), now()))
            return cur.rowcount == 1

    def wf_get(self, wf_id: str) -> Optional[dict]:
        rows = self._q("SELECT * FROM workflows WHERE wf_id=?", (wf_id,))
        if not rows:
            return None
        d = dict(rows[0])
        d["data"] = json.loads(d["data"])
        return d

    def wf_list(self, wf_type: Optional[str] = None, status: Optional[str] = None) -> list[dict]:
        sql, args = "SELECT * FROM workflows WHERE 1=1", []
        if wf_type:
            sql += " AND type=?"
            args.append(wf_type)
        if status:
            sql += " AND status=?"
            args.append(status)
        out = []
        for r in self._q(sql + " ORDER BY created_at", tuple(args)):
            d = dict(r)
            d["data"] = json.loads(d["data"])
            out.append(d)
        return out

    def wf_signal(self, wf_id: str, name: str, payload: dict) -> None:
        """Record a signal and wake the workflow, in one transaction."""
        with self.tx() as c:
            c.execute("INSERT INTO wf_signals (wf_id, name, payload, created_at) VALUES (?,?,?,?)",
                      (wf_id, name, canonical_json(payload), now()))
            c.execute("INSERT INTO outbox (topic, key, payload, created_at) VALUES (?,?,?,?)",
                      ("workflow.wake", wf_id, canonical_json({"wf_id": wf_id, "_otel": inject_carrier()}), now()))

    def wf_pending_signals(self, wf_id: str) -> list[dict]:
        rows = self._q("SELECT * FROM wf_signals WHERE wf_id=? AND consumed_at IS NULL ORDER BY id", (wf_id,))
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    def wf_save(self, wf_id: str, state: str, status: str, data: dict, consumed: list[int],
                set_timers: dict[str, float], clear_timers: list[str], history: list[tuple]) -> None:
        """Checkpoint a workflow step atomically: new state, consumed signals, timer
        changes and history either all commit or none do."""
        with self.tx() as c:
            c.execute("UPDATE workflows SET state=?, status=?, data=?, updated_at=? WHERE wf_id=?",
                      (state, status, canonical_json(data), now(), wf_id))
            for sid in consumed:
                c.execute("UPDATE wf_signals SET consumed_at=? WHERE id=?", (now(), sid))
            for name in clear_timers:
                c.execute("DELETE FROM wf_timers WHERE wf_id=? AND name=?", (wf_id, name))
            for name, at in set_timers.items():
                # Re-arming a timer (e.g. the start date moved) resets it, even if it had fired.
                c.execute("""INSERT INTO wf_timers VALUES (?,?,?,0) ON CONFLICT(wf_id, name)
                             DO UPDATE SET fire_at=excluded.fire_at, fired=0
                             WHERE wf_timers.fire_at != excluded.fire_at""", (wf_id, name, at))
            for frm, to, note in history:
                c.execute("INSERT INTO wf_history (wf_id, ts, from_state, to_state, note) VALUES (?,?,?,?,?)",
                          (wf_id, now(), frm, to, note))

    def wf_timers(self, wf_id: str) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM wf_timers WHERE wf_id=? ORDER BY fire_at", (wf_id,))]

    def wf_history(self, wf_id: str) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM wf_history WHERE wf_id=? ORDER BY id", (wf_id,))]

    def fire_due_timers(self, now_epoch: float, limit: int = 100) -> int:
        """Exactly-once firing: each due timer flips fired=0->1 in the same transaction
        that records its signal and wakes its workflow."""
        fired = 0
        with self.tx() as c:
            due = c.execute("SELECT wf_id, name, fire_at FROM wf_timers WHERE fired=0 AND fire_at<=? "
                            "ORDER BY fire_at LIMIT ?", (now_epoch, limit)).fetchall()
            for t in due:
                if c.execute("UPDATE wf_timers SET fired=1 WHERE wf_id=? AND name=? AND fired=0",
                             (t["wf_id"], t["name"])).rowcount:
                    c.execute("INSERT INTO wf_signals (wf_id, name, payload, created_at) VALUES (?,?,?,?)",
                              (t["wf_id"], f"timer:{t['name']}", canonical_json({"fire_at": t["fire_at"]}), now()))
                    c.execute("INSERT INTO outbox (topic, key, payload, created_at) VALUES (?,?,?,?)",
                              ("workflow.wake", t["wf_id"], canonical_json({"wf_id": t["wf_id"]}), now()))
                    fired += 1
        return fired

    def set_hire_status(self, hire_id: str, status: str, reason: str, actor: str) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO hire_status VALUES (?,?,?,?,?)", (hire_id, status, reason, actor, now()))
            self._audit(c, hire_id, None, actor, f"hire.{status}", {"reason": reason})

    def hire_status(self, hire_id: str) -> Optional[str]:
        rows = self._q("SELECT status FROM hire_status WHERE hire_id=?", (hire_id,))
        return rows[0]["status"] if rows else None

    # ---------------------------------------------------- login reservations
    def reserve_login(self, login: str, hire_id: str) -> Optional[str]:
        """Atomic first-come reservation of an Okta login. Returns the hire that
        owns it (this one if the reservation succeeded). Concurrent runs can't
        both pass validation and then race to create the same account."""
        with self.tx() as c:
            c.execute("INSERT INTO login_reservations VALUES (?,?,?) ON CONFLICT(login) DO NOTHING",
                      (login.lower(), hire_id, now()))
            return c.execute("SELECT hire_id FROM login_reservations WHERE login=?", (login.lower(),)).fetchone()[0]

    # ------------------------------------------------------- sweeper queries
    def stale_runs(self, older_than_s: float) -> list[dict]:
        cutoff = datetime.fromtimestamp(time.time() - older_than_s, timezone.utc).isoformat(timespec="milliseconds")
        return [dict(r) for r in self._q(
            "SELECT run_id, hire_id, updated_at FROM runs WHERE status='running' AND updated_at < ?", (cutoff,))]

    def approvals_to_escalate(self, older_than_s: float) -> list[dict]:
        cutoff = datetime.fromtimestamp(time.time() - older_than_s, timezone.utc).isoformat(timespec="milliseconds")
        return [dict(r) for r in self._q(
            "SELECT * FROM approvals WHERE status='pending' AND escalated_at IS NULL AND created_at < ?", (cutoff,))]

    def mark_escalated(self, approval_id: str) -> bool:
        with self.tx() as c:
            cur = c.execute("UPDATE approvals SET escalated_at=? WHERE approval_id=? AND escalated_at IS NULL",
                            (now(), approval_id))
            if cur.rowcount:
                self._audit(c, None, None, "sweeper", "approval.escalated", {"approval_id": approval_id})
            return cur.rowcount == 1

    def list_dead_letters(self, unresolved_only: bool = True) -> list[dict]:
        sql = "SELECT * FROM dead_letters" + (" WHERE resolved_at IS NULL" if unresolved_only else "")
        return [dict(r) for r in self._q(sql + " ORDER BY id")]
