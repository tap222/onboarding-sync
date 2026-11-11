"""The exception-triage agent. Claude is called only when validation raised issues -
the happy path never touches the model.

Claude decides:  proceed vs. hold for a human; a proposed fix; the approver message.
                 It can investigate with READ-ONLY tools (history, provenance, the
                 downstream record, past episodes, the source document).
Claude never:    decides whether a rule fired, skips a BLOCK, or writes anywhere.
                 `apply_policy` (code) has the final word.

Output is validated Instructor-style (structured.py): schema, then semantic checks
with run context - every evidence ref must exist in this run, the patch must touch
only editable fields, and the Slack message must contain no PII. Failures are sent
back to the model to fix; if it still fails, the deterministic heuristic takes over.
"""
from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Literal, Optional

import anthropic
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from . import tools as toolreg
from .auth import service_principal
from .guardrails import GuardrailViolation, ToolBudget, assert_no_pii
from .memory import WorkingMemory
from .resilience import CircuitBreaker, CircuitOpenError, TransientError, with_timeout
from .prompt_registry import PromptVersion, registry
from .budget import BudgetExceeded
from .structured import EvaluationFailed, StructuredCallError, structured_call

if TYPE_CHECKING:
    from .app import HireSync

log = logging.getLogger(__name__)

ALLOWED_PATCH_PREFIXES = ("legal_first_name", "legal_last_name", "preferred_first_name", "start_date",
                          "shipping_address", "work_location", "job_title", "department")


class PatchOp(BaseModel):
    path: str
    value: str

    @field_validator("path")
    @classmethod
    def editable(cls, v: str) -> str:
        if v.split(".")[0] not in ALLOWED_PATCH_PREFIXES:
            raise ValueError(f"{v} is not an editable field; allowed roots: {ALLOWED_PATCH_PREFIXES}")
        return v


class Recommendation(BaseModel):
    action: Literal["proceed", "hold_for_approval", "reject"]
    confidence: float = Field(ge=0, le=1)
    rationale: str
    evidence: list[str] = Field(default_factory=list, description="issue:<CODE> | episode:<run_id> | doc:<hash> | tool:<name>")
    proposed_patch: list[PatchOp] = Field(default_factory=list)
    approver_message: str
    source: str = "claude"
    evaluation: dict = Field(default_factory=dict)  # set by the evaluator loop, not by the model
    prompt: str = ""  # name@version that produced it, e.g. triage@v1 or heuristic@builtin

    @field_validator("evidence")
    @classmethod
    def evidence_exists(cls, refs: list[str], info: ValidationInfo) -> list[str]:
        ctx = info.context or {}
        if not ctx:
            return refs
        bad = []
        for ref in refs:
            kind, _, val = ref.partition(":")
            ok = {
                "issue": val in ctx.get("issue_codes", ()),
                "episode": val in ctx.get("episode_ids", ()),
                "doc": val in ctx.get("doc_hashes", ()),
                "tool": val in ctx.get("tools_called", ()),
            }.get(kind, False)
            if not ok:
                bad.append(ref)
        if bad:
            raise ValueError(f"evidence refs not found in this run: {bad}. Cite only issue codes, "
                             "episode run_ids, doc hashes and tools that actually appear above.")
        return refs

    @field_validator("approver_message")
    @classmethod
    def no_pii(cls, v: str) -> str:
        try:
            assert_no_pii(v, "approver_message")
        except GuardrailViolation as exc:
            raise ValueError(str(exc)) from exc
        return v


RECOMMENDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["proceed", "hold_for_approval", "reject"]},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "proposed_patch": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"path": {"type": "string"}, "value": {"type": "string"}},
                      "required": ["path", "value"], "additionalProperties": False},
        },
        "approver_message": {"type": "string"},
    },
    "required": ["action", "confidence", "rationale", "evidence", "proposed_patch", "approver_message"],
    "additionalProperties": False,
}

# The triage system prompt lives in hiresync/prompts/triage/v<N>.md (versioned, locked).



def build_triage_input(wm: WorkingMemory, doc_hashes: list[str]) -> str:
    return (
        "Situation (working memory for this run):\n"
        f"{json.dumps(wm.prompt_view(), indent=2, default=str)}\n\n"
        f"Source documents behind this change: {doc_hashes or 'none'}\n\n"
        "Similar past episodes (most similar first):\n"
        f"{json.dumps(wm.recalled, indent=2) if wm.recalled else '(none)'}"
    )


class TriageAgent:
    def __init__(self, mode: Optional[str] = None, prompt_version: Optional[str] = None):
        mode = mode or os.getenv("AGENT_MODE", "auto")
        has_creds = any(os.getenv(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"))
        self.use_claude = mode == "claude" or (mode == "auto" and has_creds)
        self.client = anthropic.AsyncAnthropic(max_retries=2, timeout=120) if self.use_claude else None
        self.breaker = CircuitBreaker("claude", failure_threshold=3, reset_timeout=30)
        self.calls = 0
        self.timeout_s = 90.0  # set from Timeouts.claude by HireSync
        self.prompt: PromptVersion = registry().get("triage", prompt_version)
        self.app: Optional["HireSync"] = None  # set by HireSync for tool access

    async def triage(self, wm: WorkingMemory, doc_hashes: list[str]) -> tuple[Recommendation, list[dict]]:
        """Returns (recommendation, tool transcript)."""
        transcript: list[dict] = []
        if not self.use_claude:
            rec = heuristic_triage(wm)
            rec.prompt = "heuristic@builtin"
            return rec, transcript
        try:
            # One deadline for the whole tool loop: a slow model must not hold the run.
            rec = await self.breaker.call(lambda: with_timeout(
                lambda: self._ask_claude(wm, doc_hashes, transcript), self.timeout_s, "claude.triage"))
        except EvaluationFailed as exc:
            # Well-formed but still rejected by the evaluator after a retry: keep it
            # for the approver to read, but it can never auto-proceed.
            rec = exc.parsed
            rec.source, rec.prompt = "claude", self.prompt.ref
            rec.evaluation = {"passed": False, "problems": exc.problems}
        except BudgetExceeded as exc:
            log.warning("claude triage blocked by budget (%s); using heuristic", exc)
            rec = heuristic_triage(wm)
            rec.prompt = "heuristic@builtin"
            rec.rationale = f"[budget: {exc.code}] " + rec.rationale
        except (CircuitOpenError, TransientError, StructuredCallError, anthropic.APIError) as exc:
            # The model being unavailable must never stop onboarding.
            log.warning("claude triage unavailable (%s); using heuristic", exc)
            rec = heuristic_triage(wm)
            rec.prompt = "heuristic@builtin"
            rec.rationale = f"[Claude unavailable: {type(exc).__name__}] " + rec.rationale
        return rec, transcript

    async def _ask_claude(self, wm: WorkingMemory, doc_hashes: list[str], transcript: list[dict]) -> Recommendation:
        self.calls += 1
        budget = ToolBudget(max_calls=8)
        called: set[str] = set()

        # The agent acts as a service principal scoped to this one hire, and is
        # additionally limited to read-only tools (defense in depth).
        principal = service_principal("svc-triage", hire_id=wm.hire_id)

        async def run_tool(name: str, raw: dict) -> tuple[str, bool]:
            called.add(name)
            return await toolreg.execute(self.app, name, raw, principal=principal,
                                         allow=toolreg.READ_ONLY, budget=budget)

        user = build_triage_input(wm, doc_hashes)
        context = {
            "issue_codes": {i["code"] for i in wm.issues},
            "episode_ids": {e["run_id"] for e in wm.recalled},
            "doc_hashes": set(doc_hashes),
            "tools_called": called,  # same set object: grows as tools run
        }
        store = self.app.store if self.app else None
        eval_rounds = {"n": 0}

        async def evaluate(candidate: Recommendation) -> list[str]:
            eval_rounds["n"] += 1
            return await self.app.evaluator(candidate, wm) if self.app else []

        rec = await structured_call(
            self.client, prompt=self.prompt, response_model=Recommendation, schema=RECOMMENDATION_SCHEMA,
            user=user, tools=toolreg.specs(principal, toolreg.READ_ONLY),
            tool_executor=run_tool, validation_context=context, transcript=transcript, purpose="triage",
            record=(lambda call: store.record_llm_call("triage", call, wm.run_id, wm.trace_id)) if store else None,
            guard=self.app.cost_guard if self.app else None, run_id=wm.run_id,
            evaluate=evaluate, max_eval_retries=1,
        )
        rec.source, rec.prompt = "claude", self.prompt.ref
        rec.evaluation = {"passed": True, "rounds": eval_rounds["n"]}
        return rec


def heuristic_triage(wm: WorkingMemory) -> Recommendation:
    """Deterministic stand-in: offline mode and the fallback when Claude is down.
    Same contract, conservative policy: any block -> human; warns -> follow precedent."""
    blocks = [i for i in wm.issues if i["severity"] == "block"]
    warns = [i for i in wm.issues if i["severity"] == "warn"]
    patch: list[PatchOp] = []
    for i in wm.issues:
        if i["code"] in ("NAME_SUSPICIOUS", "NAME_CASING") and i.get("field"):
            v = wm.new.get(i["field"]) or ""
            fixed = "".join(ch for ch in " ".join(v.split()).title() if not ch.isdigit())
            if fixed and fixed != v:
                patch.append(PatchOp(path=i["field"], value=fixed))
    codes = sorted({i["code"] for i in wm.issues})
    evidence = [f"issue:{c}" for c in codes]
    if blocks:
        return Recommendation(
            action="hold_for_approval", confidence=0.9, source="heuristic", proposed_patch=patch,
            evidence=evidence, rationale=f"blocking issues present: {', '.join(codes)}",
            approver_message=f"{wm.hire_id}: change v{wm.from_version}->v{wm.to_version} is held. "
                             f"Blocking checks failed ({', '.join(codes)}). Please review before it reaches downstream systems.",
        )
    approved = [e for e in wm.recalled if e["decision"] in ("approved", "agent_proceed") and e["similarity"] >= 0.5]
    corrected = [e for e in wm.recalled if e["decision"] in ("rejected", "corrected")]
    if warns and approved and not corrected:
        return Recommendation(
            action="proceed", confidence=0.85, source="heuristic",
            evidence=evidence + [f"episode:{e['run_id']}" for e in approved],
            rationale=f"warnings ({', '.join(codes)}) match {len(approved)} past episode(s) a human approved",
            approver_message=f"{wm.hire_id}: auto-approved warnings ({', '.join(codes)}) based on prior approvals.",
        )
    return Recommendation(
        action="hold_for_approval", confidence=0.6, source="heuristic", proposed_patch=patch, evidence=evidence,
        rationale=f"warnings ({', '.join(codes)}) with no clear precedent",
        approver_message=f"{wm.hire_id}: change v{wm.from_version}->v{wm.to_version} raised warnings "
                         f"({', '.join(codes)}). Approve to propagate, or edit the record.",
    )


AUTO_PROCEED_CONFIDENCE = 0.8


def apply_policy(wm: WorkingMemory, rec: Recommendation) -> str:
    """Code, not the model, has the final word. Returns proceed | hold."""
    if any(i["severity"] == "block" for i in wm.issues):
        return "hold"  # even a model "reject" goes to a human, who can confirm it
    if rec.evaluation.get("passed") is False:
        return "hold"  # the evaluator still disagrees after a retry
    if rec.action == "proceed" and rec.confidence >= AUTO_PROCEED_CONFIDENCE:
        return "proceed"
    return "hold"
