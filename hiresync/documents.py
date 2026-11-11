"""Document ingestion: the same offer letter or IT form is often uploaded by the
recruiter, the coordinator and the manager. It must be stored, extracted and turned
into a record change exactly once, while every uploader is still recorded.

Pipeline for one upload
  0. authz                            document:upload + access to the hire; one hire per document
  1. guardrails.check_upload           size / MIME / UTF-8
  2. normalise + SHA-256               identity = content, not filename or uploader
  3. idempotency claim idem:doc:<hash> fast dedupe across workers (DB lease; Redis optional)
  4. DB INSERT .. ON CONFLICT          authoritative dedupe, + upload row per uploader
  5. scan + redact                     injection hits flagged, PII removed before any model
  5b. chunk + embed + HNSW index       offsets kept, so a search hit is a citation
  6. extract (Claude or heuristic)     every field must cite a verbatim quote
  7. verify citations in code          quote located in the text -> char offsets
  8. append_version(event_id=doc:<hash>, provenance=citations)
                                       a document can create at most one version
"""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, ValidationInfo, field_validator, model_validator

from . import guardrails
from .idempotency import IN_FLIGHT, Idempotency
from .auth import DOC_CLASSIFICATION, Principal, can_read_document, require, require_hire
from .models import canonicalize
from .prompt_registry import registry
from .structured import StructuredCallError, structured_call
from .versioning import flatten, unflatten_set

if TYPE_CHECKING:
    from .app import HireSync

log = logging.getLogger(__name__)

ExtractablePath = Literal[
    "legal_first_name", "legal_last_name", "preferred_first_name", "start_date", "job_title", "department",
    "shipping_address.line1", "shipping_address.city", "shipping_address.state",
    "shipping_address.postal_code", "shipping_address.country", "work_location.country",
]


def normalise(text: str) -> str:
    """Same document saved on Windows vs. Mac, or with/without a BOM, hashes the same."""
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip() + "\n"


def doc_hash(text: str) -> str:
    return hashlib.sha256(normalise(text).encode()).hexdigest()


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def parse_date(s: str) -> Optional[str]:
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%d %B %Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


class ExtractedField(BaseModel):
    path: ExtractablePath
    value: str
    quote: str  # verbatim span from the document supporting `value`

    @model_validator(mode="after")
    def cited(self, info: ValidationInfo):
        text = (info.context or {}).get("text")
        if text is None:
            return self
        if _squash(self.quote) not in _squash(text):
            raise ValueError(f"quote for {self.path} does not appear verbatim in the document: {self.quote!r}")
        if self.path == "start_date":
            iso = parse_date(self.value)
            if not iso:
                raise ValueError(f"start_date {self.value!r} must be an ISO date (YYYY-MM-DD)")
            self.value = iso
        elif self.path.endswith("country"):
            if not re.fullmatch(r"[A-Z]{2}", self.value):
                raise ValueError(f"{self.path} must be an ISO alpha-2 code, got {self.value!r}")
        elif _squash(self.value) not in _squash(self.quote):
            raise ValueError(f"value {self.value!r} for {self.path} is not contained in its quote {self.quote!r}")
        return self


class Extraction(BaseModel):
    fields: list[ExtractedField]
    notes: str = ""

    @field_validator("fields")
    @classmethod
    def unique_paths(cls, v):
        paths = [f.path for f in v]
        if len(paths) != len(set(paths)):
            raise ValueError("each path may appear at most once")
        return v


EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "fields": {"type": "array", "items": {
            "type": "object",
            "properties": {"path": {"type": "string", "enum": list(ExtractablePath.__args__)},
                           "value": {"type": "string"}, "quote": {"type": "string"}},
            "required": ["path", "value", "quote"], "additionalProperties": False}},
        "notes": {"type": "string"},
    },
    "required": ["fields", "notes"],
    "additionalProperties": False,
}

# The extraction prompt lives in hiresync/prompts/extraction/v<N>.md (versioned, locked).


LABELS: dict[str, ExtractablePath] = {
    "legal first name": "legal_first_name", "first name": "legal_first_name",
    "legal last name": "legal_last_name", "last name": "legal_last_name",
    "preferred name": "preferred_first_name", "start date": "start_date",
    "job title": "job_title", "title": "job_title", "department": "department",
    "street": "shipping_address.line1", "address line 1": "shipping_address.line1",
    "city": "shipping_address.city", "state": "shipping_address.state",
    "zip": "shipping_address.postal_code", "postal code": "shipping_address.postal_code",
    "country": "shipping_address.country",
}
COUNTRIES = {"united states": "US", "usa": "US", "us": "US", "united kingdom": "GB", "uk": "GB", "gb": "GB",
             "canada": "CA", "ca": "CA", "india": "IN", "in": "IN"}


def heuristic_extract(text: str) -> Extraction:
    """Offline extractor for 'Label: value' forms. Same output contract as Claude."""
    fields: list[ExtractedField] = []
    seen: set[str] = set()
    for line in text.split("\n"):
        m = re.match(r"^\s*([A-Za-z][A-Za-z0-9 ]{1,30}?)\s*:\s*(.+?)\s*$", line)
        if not m:
            continue
        path = LABELS.get(m.group(1).strip().lower())
        if not path or path in seen:
            continue
        value = m.group(2)
        if path == "start_date":
            value = parse_date(value) or value
        if path == "shipping_address.country":
            value = COUNTRIES.get(value.lower(), value)
        fields.append(ExtractedField(path=path, value=value, quote=line.strip()))
        seen.add(path)
    return Extraction(fields=fields, notes="heuristic extractor")


def locate(text: str, quote: str) -> dict:
    """Citation offsets computed by code - model offsets are never trusted."""
    i = text.find(quote)
    if i < 0:  # whitespace-insensitive fallback
        m = re.search(r"\s+".join(map(re.escape, quote.split())), text, re.IGNORECASE)
        if m:
            return {"quote": m.group(0), "char_start": m.start(), "char_end": m.end()}
        return {"quote": quote, "char_start": None, "char_end": None}
    return {"quote": quote, "char_start": i, "char_end": i + len(quote)}


class DocumentService:
    def __init__(self, app: "HireSync", idem: Idempotency):
        self.app = app
        self.idem = idem
        self.extractions = 0

    async def upload(self, hire_id: str, filename: str, content: bytes, mime: str,
                     principal: Principal, doc_type: str = "other") -> dict:
        """Authorization: needs document:upload and access to the hire. A duplicate
        upload records the uploader but only reveals the existing document's details
        to someone allowed to read it - a hash match must not become an oracle."""
        store = self.app.store
        require(principal, "document:upload", f"hire {hire_id}")
        current = store.get_version(hire_id)
        require_hire(principal, hire_id, current["data"] if current else None)
        uploaded_by = principal.email
        classification = DOC_CLASSIFICATION.get(doc_type, "restricted")
        trace_id = f"tr_{uuid.uuid4().hex[:12]}"
        text = normalise(guardrails.check_upload(content, mime))
        h = doc_hash(text)

        owner_hire = store.doc_owner_hire(h)
        if owner_hire is not None and owner_hire != hire_id:
            # Same bytes already belong to another hire: refuse, and don't say which.
            store.audit(hire_id, None, uploaded_by, "document.cross_hire_rejected", {"doc_hash": h[:12]})
            raise guardrails.GuardrailViolation("DOC_BELONGS_TO_ANOTHER_HIRE",
                                                "this document is already filed for a different hire")
        scan = guardrails.scan(text)
        redacted = guardrails.redact(text)
        args = (h, hire_id, doc_type, classification, mime, len(content), redacted, _scan_dict(scan),
                uploaded_by, filename, trace_id)

        owned, cached = await self.idem.claim("doc", h)
        if not owned:
            store.insert_document(*args)  # records *who* uploaded it (atomic, deduped)
            status = "processing" if cached == IN_FLIGHT else "duplicate"
            return self._duplicate_view(principal, h, status, cached if isinstance(cached, dict) else {})
        try:
            created = store.insert_document(*args)
            if not created:
                # The idempotency key expired or was lost, but the DB constraint caught it.
                existing = store.get_document(h)
                result = {"doc_hash": h, "version": (existing.get("extraction") or {}).get("version")}
                await self.idem.complete("doc", h, result)
                return self._duplicate_view(principal, h, "duplicate", result)

            chunks = self.app.vectors.index_document(h, hire_id, redacted, doc_type, classification)
            extraction = await self._extract(redacted, h)
            result = self._to_change(hire_id, h, redacted, extraction, uploaded_by, trace_id)
            store.set_document_extraction(h, "extracted", {**extraction.model_dump(), **result})
            await self.idem.complete("doc", h, {"doc_hash": h, **result})
            return {"doc_hash": h, **result, "duplicate": False, "status": "extracted", "chunks": chunks,
                    "classification": classification, "suspicious": scan.suspicious,
                    "pii_redacted": scan.pii_found}
        except Exception:
            await self.idem.release("doc", h)
            store.set_document_extraction(h, "failed", None)
            raise

    def _duplicate_view(self, p: Principal, h: str, status: str, cached: dict) -> dict:
        doc = self.app.store.get_document(h)
        # Decide on group membership only (uploaders=set()): this person was just
        # added as an uploader, and uploading a copy must not grant access to the
        # existing document's extraction, version or other uploaders.
        ok, _ = can_read_document(p, doc, set()) if doc else (False, "")
        if ok:
            return {**cached, "doc_hash": h, "duplicate": True, "status": status}
        return {"duplicate": True, "status": "received"}

    LONG_DOC_CHARS = 6000
    FIELD_QUERIES = ["legal first name last name preferred name", "start date",
                     "job title department", "shipping address street city state zip postal code country"]

    def extraction_context(self, text: str, h: str) -> str:
        """Short documents go whole. Long ones: retrieve the chunks relevant to the
        fields we extract (HNSW search filtered to this document) and send only
        those, with their offsets. Quotes are still verified against the full text."""
        if len(text) <= self.LONG_DOC_CHARS:
            return text
        picked: dict[int, dict] = {}
        for q in self.FIELD_QUERIES:
            for hit in self.app.vectors.search_chunks(q, k=3, where=lambda p: p["doc_hash"] == h):
                picked[hit["idx"]] = hit
        parts = [f"[chars {c['char_start']}-{c['char_end']}]\n{c['text']}" for _, c in sorted(picked.items())]
        return "Relevant excerpts of a longer document:\n\n" + "\n\n".join(parts)

    async def _extract(self, redacted: str, h: str) -> Extraction:
        agent = self.app.agent
        self.extractions += 1
        if agent.use_claude:
            try:
                return await structured_call(
                    agent.client, prompt=registry().get("extraction"), response_model=Extraction,
                    schema=EXTRACTION_SCHEMA, user=guardrails.quarantine(self.extraction_context(redacted, h)),
                    validation_context={"text": redacted},
                    record=lambda call: self.app.store.record_llm_call("extraction", call, None, None),
                    guard=self.app.cost_guard, purpose="extraction",
                )
            except (StructuredCallError, Exception) as exc:  # noqa: BLE001
                log.warning("claude extraction failed for %s (%s); heuristic fallback", h[:12], exc)
        return Extraction.model_validate(heuristic_extract(redacted).model_dump(), context={"text": redacted})

    def _to_change(self, hire_id: str, h: str, text: str, ex: Extraction, actor: str, trace_id: str) -> dict:
        current = self.app.store.get_version(hire_id)
        if not current:
            return {"version": None, "changed_fields": [], "note": "unknown hire; extraction stored only"}
        flat = flatten(current["data"])
        record, provenance, changed = current["data"], {}, []
        for f in ex.fields:
            if str(flat.get(f.path)) != f.value:
                record = unflatten_set(record, f.path, f.value)
                provenance[f.path] = {"doc_hash": h, "citation": {"doc_hash": h, **locate(text, f.quote)}}
                changed.append(f.path)
        if not changed:
            return {"version": current["version"], "changed_fields": [], "note": "document matches the record"}
        try:
            record = canonicalize(record)
        except ValueError:
            pass  # stored as-is; validation flags SCHEMA_INVALID (block) so a human sees it
        res = self.app.store.append_version(hire_id, record, source="document", actor=actor,
                                            event_id=f"doc:{h}", provenance=provenance, trace_id=trace_id)
        return {"version": res["version"], "changed_fields": changed, "trace_id": trace_id}


def _scan_dict(scan: guardrails.ScanResult) -> dict:
    return {"injection_hits": scan.injection_hits, "pii_found": scan.pii_found, "suspicious": scan.suspicious}
