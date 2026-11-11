"""Evaluator-optimizer loop for triage recommendations.

Schema validation (structured.py) proves the answer is well-formed. The evaluator
checks it is *right for this run*, and its critique drives a retry:

    generate -> validate schema -> evaluate -> (problems?) -> critique -> retry
                                               (max N)     -> still failing: the
                                               recommendation is kept but marked
                                               evaluation.passed=False, and
                                               apply_policy forces a human hold.

Deterministic checks (always on, free):
  * never "proceed" while a BLOCK issue exists
  * every BLOCK issue is cited in evidence
  * a proposed patch is re-validated with the same cross-system rules: it must
    resolve the issue it targets, not introduce new blocking issues, and touch
    only fields an issue named
  * "proceed" needs confidence >= the auto-proceed threshold
  * the approver message names the hire; the rationale references the issues

Optional LLM judge (EVALUATOR_LLM=1): a second model (claude-haiku-5-5, versioned
prompt judge@vN) reviews the rationale against the situation. It runs under the
same cost guard; if it can't run, the deterministic verdict stands.
"""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel

from .validation import validate
from .versioning import diff, unflatten_set

if TYPE_CHECKING:
    from .agent import Recommendation
    from .app import HireSync
    from .memory import WorkingMemory

AUTO_PROCEED_CONFIDENCE = 0.8


class JudgeVerdict(BaseModel):
    verdict: Literal["pass", "fail"]
    problems: list[str]


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["pass", "fail"]},
                   "problems": {"type": "array", "items": {"type": "string"}}},
    "required": ["verdict", "problems"], "additionalProperties": False,
}


def deterministic_checks(rec: "Recommendation", wm: "WorkingMemory", app: Optional["HireSync"]) -> list[str]:
    problems: list[str] = []
    blocks = {i["code"] for i in wm.issues if i["severity"] == "block"}
    all_codes = {i["code"] for i in wm.issues}
    if blocks and rec.action == "proceed":
        problems.append(f"action is 'proceed' but blocking issues {sorted(blocks)} require hold_for_approval or reject")
    missing = sorted(c for c in blocks if f"issue:{c}" not in rec.evidence)
    if missing:
        problems.append(f"evidence must cite every blocking issue; missing {['issue:' + c for c in missing]}")
    if rec.action == "proceed" and rec.confidence < AUTO_PROCEED_CONFIDENCE:
        problems.append(f"'proceed' with confidence {rec.confidence:.2f} < {AUTO_PROCEED_CONFIDENCE}: "
                        "either justify higher confidence or hold_for_approval")
    if wm.hire_id not in rec.approver_message:
        problems.append(f"approver_message must name the hire ({wm.hire_id})")
    if all_codes and not any(c in rec.rationale for c in all_codes) and "episode" not in rec.rationale.lower():
        problems.append("rationale should reference the issue codes or episodes the decision rests on")
    if rec.proposed_patch and wm.new and app is not None:
        issue_fields = {(i.get("field") or "").split(",")[0] for i in wm.issues}
        patched = wm.new
        for op in rec.proposed_patch:
            patched = unflatten_set(patched, op.path, op.value)
            if not any(op.path == f or op.path.startswith(f + ".") or f.startswith(op.path) for f in issue_fields if f):
                problems.append(f"patch touches {op.path}, which no issue names")
        after = validate(patched, wm.old, diff(wm.old, patched), app.systems, app.orchestrator.today())
        after_codes = {i.code for i in after}
        new_blocks = {i.code for i in after if i.severity.value == "block"} - blocks
        if new_blocks:
            problems.append(f"proposed patch would introduce blocking issues {sorted(new_blocks)}")
        patched_paths = {op.path for op in rec.proposed_patch}
        unresolved = sorted(i["code"] for i in wm.issues
                            if (i.get("field") or "") in patched_paths and i["code"] in after_codes)
        if unresolved:
            problems.append(f"proposed patch does not resolve {unresolved}; re-validated record still fails")
    return problems


class Evaluator:
    def __init__(self, app: Optional["HireSync"] = None, use_llm: Optional[bool] = None):
        self.app = app
        self.use_llm = use_llm if use_llm is not None else os.getenv("EVALUATOR_LLM") == "1"
        self.runs = 0
        self.rejections = 0

    async def __call__(self, rec: "Recommendation", wm: "WorkingMemory") -> list[str]:
        self.runs += 1
        problems = deterministic_checks(rec, wm, self.app)
        if not problems and self.use_llm and self.app and self.app.agent.use_claude:
            problems = await self._judge(rec, wm)
        if problems:
            self.rejections += 1
        return problems

    async def _judge(self, rec: "Recommendation", wm: "WorkingMemory") -> list[str]:
        from .budget import BudgetExceeded
        from .prompt_registry import registry
        from .structured import StructuredCallError, structured_call
        try:
            verdict = await structured_call(
                self.app.agent.client, prompt=registry().get("judge"), response_model=JudgeVerdict,
                schema=JUDGE_SCHEMA,
                user=json.dumps({"situation": wm.prompt_view(), "recommendation": rec.model_dump()}, default=str),
                guard=self.app.cost_guard, run_id=wm.run_id, purpose="judge",
                record=lambda call: self.app.store.record_llm_call("judge", call, wm.run_id, wm.trace_id))
        except (BudgetExceeded, StructuredCallError, Exception):  # noqa: BLE001 - judge is advisory
            return []
        return verdict.problems if verdict.verdict == "fail" else []
