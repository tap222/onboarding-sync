"""Guardrails around everything that crosses a trust boundary.

Input   (uploads, free text, tool results -> model)
  * upload limits: size, MIME allowlist, UTF-8 decodable
  * prompt-injection scan: documents are data, never instructions
  * PII redaction: SSN, bank/card numbers, DOB are stripped before any model call
Tools   (model -> our functions)
  * allowlist per caller: the pipeline agent gets read-only tools only
  * every tool input validated against its Pydantic schema before running
  * call budget per run, so a looping model can't run up cost
Output  (model -> pipeline)
  * schema-validated (structured.py), patch paths allowlisted (agent.sanitize)
  * citations must quote the source verbatim (checked in code, not trusted)
  * evidence refs must point at things that exist in this run
  * no PII in the Slack message
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
ALLOWED_MIME = {"text/plain", "text/markdown", "application/json"}

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
    r"disregard (the )?(system|previous) (prompt|instructions)",
    r"you are now\b",
    r"\bsystem prompt\b",
    r"\bact as\b.{0,40}\b(admin|developer|system)\b",
    r"(approve|auto-?approve) (this|all|every)",
    r"set (action|confidence)\s*[:=]",
    r"</?(system|assistant|instructions?)>",
    r"\bproceed without (approval|review)\b",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

PII_PATTERNS = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "CARD": r"\b(?:\d[ -]?){13,16}\b",
    "BANK_ACCOUNT": r"\b(?:account|acct|routing)\s*(?:no\.?|number|#)?\s*[:#]?\s*\d{6,17}\b",
    "DOB": r"\b(?:dob|date of birth)\s*[:\-]?\s*\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b",
}
_PII_RES = {k: re.compile(v, re.IGNORECASE) for k, v in PII_PATTERNS.items()}


class GuardrailViolation(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass
class ScanResult:
    injection_hits: list[str] = field(default_factory=list)
    pii_found: dict[str, int] = field(default_factory=dict)

    @property
    def suspicious(self) -> bool:
        return bool(self.injection_hits)


def check_upload(content: bytes, mime: str) -> str:
    if len(content) == 0:
        raise GuardrailViolation("EMPTY_UPLOAD", "file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise GuardrailViolation("UPLOAD_TOO_LARGE", f"{len(content)} bytes > {MAX_UPLOAD_BYTES}")
    if mime not in ALLOWED_MIME:
        raise GuardrailViolation("MIME_NOT_ALLOWED", f"{mime} not in {sorted(ALLOWED_MIME)}")
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GuardrailViolation("NOT_UTF8", str(exc)) from exc


def scan(text: str) -> ScanResult:
    res = ScanResult()
    res.injection_hits = sorted({m.group(0).lower() for m in _INJECTION_RE.finditer(text)})
    for name, rx in _PII_RES.items():
        n = len(rx.findall(text))
        if n:
            res.pii_found[name] = n
    return res


def redact(text: str) -> str:
    for name, rx in _PII_RES.items():
        text = rx.sub(f"[REDACTED_{name}]", text)
    return text


def quarantine(text: str) -> str:
    """Neutralise injection attempts while keeping the text readable to the model,
    and wrap it so the model can see where untrusted data starts and ends."""
    text = _INJECTION_RE.sub(lambda m: f"[removed:{len(m.group(0))}ch]", text)
    return f"<untrusted_document>\n{text}\n</untrusted_document>"


def assert_no_pii(text: str, where: str) -> None:
    found = scan(text).pii_found
    if found:
        raise GuardrailViolation("PII_IN_OUTPUT", f"{where} contains {sorted(found)}")


class ToolBudget:
    def __init__(self, max_calls: int = 8):
        self.max_calls = max_calls
        self.used = 0

    def spend(self) -> None:
        self.used += 1
        if self.used > self.max_calls:
            raise GuardrailViolation("TOOL_BUDGET", f"more than {self.max_calls} tool calls in one run")
