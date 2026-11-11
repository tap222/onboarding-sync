"""Behavioural testing for versioned prompts.

A case describes a situation (issues, diff, precedent, optional seeded state for
tools) and *behavioural* expectations - not exact output, which varies run to run:

    action_in          allowed recommendation actions
    never              actions that must never be recommended
    policy             what apply_policy must decide (hold | proceed)
    evidence_includes  evidence refs that must be cited
    patch              {path: value} that must be proposed
    message_mentions   substrings the approver message must contain
    max_tool_calls     budget

Each case runs `repeats` times per prompt version (model output is not
deterministic): pass rate per case, flaky cases (pass sometimes), and a
version-vs-version table of cases whose behaviour changed. Reports carry the
prompt sha256, so a result is tied to exact prompt text + model + effort.

Replay mode re-runs recorded production inputs (llm_calls) against a candidate
version and compares its action with the recorded one and, where a human later
decided (episodes), with the human.

    python -m hiresync.behave --versions v1 v2 --repeats 3
    python -m hiresync.behave --replay path/to/hiresync.db --candidate v2
Needs Anthropic credentials (the heuristic ignores prompts, so testing it
against prompt versions would be meaningless).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .agent import RECOMMENDATION_SCHEMA, Recommendation, TriageAgent, apply_policy
from .app import HireSync
from .memory import WorkingMemory
from .prompt_registry import registry
from .structured import structured_call

CASES = Path(__file__).parent.parent / "evals" / "triage_cases.jsonl"
RESULTS = Path(__file__).parent.parent / "evals" / "results"


@dataclass
class CaseResult:
    case_id: str
    version: str
    passes: int = 0
    runs: int = 0
    actions: Counter = field(default_factory=Counter)
    failures: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.passes / self.runs if self.runs else 0.0

    @property
    def flaky(self) -> bool:
        return 0 < self.passes < self.runs


def load_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check(case: dict, rec: Recommendation, tool_calls: int) -> list[str]:
    exp, wm_issues, fails = case["expect"], case["wm"]["issues"], []
    wm = WorkingMemory(run_id="x", hire_id=case["wm"]["hire_id"], from_version=1, to_version=2, issues=wm_issues)
    if "action_in" in exp and rec.action not in exp["action_in"]:
        fails.append(f"action {rec.action} not in {exp['action_in']}")
    if rec.action in exp.get("never", []):
        fails.append(f"forbidden action {rec.action}")
    if "policy" in exp and apply_policy(wm, rec) != exp["policy"]:
        fails.append(f"policy {apply_policy(wm, rec)} != {exp['policy']}")
    for ref in exp.get("evidence_includes", []):
        if ref not in rec.evidence:
            fails.append(f"missing evidence {ref}")
    proposed = {p.path: p.value for p in rec.proposed_patch}
    for path, value in exp.get("patch", {}).items():
        if proposed.get(path) != value:
            fails.append(f"patch {path}={proposed.get(path)!r}, want {value!r}")
    for s in exp.get("message_mentions", []):
        if s not in rec.approver_message:
            fails.append(f"message lacks {s!r}")
    if tool_calls > exp.get("max_tool_calls", 8):
        fails.append(f"{tool_calls} tool calls > {exp['max_tool_calls']}")
    return fails


async def _case_app(case: dict) -> HireSync:
    """A throwaway app seeded with the case's state, so the agent's tools see it."""
    app = HireSync(db_path=":memory:", agent_mode="heuristic")
    seed = case.get("seed", {})
    hid = case["wm"]["hire_id"]
    for i, rec in enumerate(seed.get("versions", [])):
        app.store.append_version(hid, {"hire_id": hid, **rec}, source="seed", actor="seed", event_id=f"seed-{i}")
    if "okta" in seed:
        app.systems["okta"].records[hid] = seed["okta"]
    return app


async def run_case(case: dict, version: str, client, repeats: int) -> CaseResult:
    res = CaseResult(case["id"], version)
    for _ in range(repeats):
        app = await _case_app(case)
        agent = TriageAgent(mode="heuristic", prompt_version=version)
        agent.use_claude, agent.client, agent.app = True, client, app
        w = case["wm"]
        wm = WorkingMemory(run_id=f"behave:{case['id']}", hire_id=w["hire_id"], from_version=w.get("from_version", 1),
                           to_version=w.get("to_version", 2), new=w.get("new", {}), changes=w.get("changes", []),
                           issues=w["issues"], targets=w.get("targets", []), recalled=w.get("recalled", []))
        rec, transcript = await agent.triage(wm, w.get("doc_hashes", []))
        res.runs += 1
        res.actions[rec.action] += 1
        if rec.source != "claude":
            res.failures.append(f"model call failed, fell back: {rec.rationale[:120]}")
            continue
        fails = check(case, rec, len(transcript))
        if fails:
            res.failures.extend(fails)
        else:
            res.passes += 1
    return res


async def compare(versions: list[str], client, repeats: int = 3, cases: Optional[list[dict]] = None) -> dict:
    cases = cases or load_cases()
    reg = registry()
    report: dict[str, Any] = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "repeats": repeats, "versions": {}}
    for v in versions:
        pv = reg.get("triage", v)
        results = [await run_case(c, v, client, repeats) for c in cases]
        report["versions"][v] = {
            "prompt_sha": pv.sha256, "model": pv.model, "effort": pv.effort,
            "pass_rate": round(sum(r.rate for r in results) / len(results), 3),
            "cases": {r.case_id: {"rate": round(r.rate, 2), "flaky": r.flaky, "actions": dict(r.actions),
                                  "failures": sorted(set(r.failures))} for r in results},
        }
    if len(versions) >= 2:
        a, b = versions[0], versions[-1]
        ca, cb = report["versions"][a]["cases"], report["versions"][b]["cases"]
        report["changed"] = [{"case": cid, a: ca[cid]["rate"], b: cb[cid]["rate"],
                              "verdict": "improved" if cb[cid]["rate"] > ca[cid]["rate"] else "regressed"}
                             for cid in ca if ca[cid]["rate"] != cb[cid]["rate"]]
    return report


async def replay(db_path: str, candidate: str, client, limit: int = 50) -> dict:
    """Re-run recorded triage inputs against a candidate prompt version."""
    app = HireSync(db_path=db_path, agent_mode="heuristic")
    pv = registry().get("triage", candidate)
    human = {e["run_id"]: e["decision"] for e in app.store.all_episodes()}
    rows = [r for r in app.store.llm_calls(purpose="triage") if r["outcome"] == "ok"][-limit:]
    out = []
    for r in rows:
        rec = await structured_call(client, prompt=pv, response_model=Recommendation, schema=RECOMMENDATION_SCHEMA,
                                    user=r["input"])
        recorded = (r["output"] or {}).get("action")
        h = human.get(r["run_id"])
        out.append({"run_id": r["run_id"], "recorded_prompt": r["prompt"], "recorded": recorded,
                    "candidate": rec.action, "human": h,
                    "agrees_with_human": None if h is None else
                    (rec.action == "proceed") == (h in ("approved", "agent_proceed"))})
    judged = [o for o in out if o["agrees_with_human"] is not None]
    return {"candidate": pv.ref, "prompt_sha": pv.sha256, "replayed": len(out),
            "changed_vs_recorded": sum(o["recorded"] != o["candidate"] for o in out),
            "human_agreement": round(sum(o["agrees_with_human"] for o in judged) / len(judged), 3) if judged else None,
            "rows": out}


def print_report(rep: dict) -> None:
    for v, data in rep["versions"].items():
        print(f"\ntriage@{v}  sha={data['prompt_sha'][:12]}  {data['model']}/{data['effort']}  "
              f"pass rate {data['pass_rate']:.0%}")
        for cid, c in data["cases"].items():
            flag = " (flaky)" if c["flaky"] else ""
            print(f"  {'PASS' if c['rate'] == 1 else 'FAIL'} {cid:42} {c['rate']:.0%}{flag} {c['actions']}")
            for f in c["failures"][:3]:
                print(f"       - {f}")
    for ch in rep.get("changed", []):
        print(f"  changed: {ch}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--replay")
    ap.add_argument("--candidate")
    args = ap.parse_args()
    if not any(os.getenv(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")):
        raise SystemExit("behavioural tests call Claude: set ANTHROPIC_API_KEY (the offline heuristic ignores prompts)")
    import anthropic

    client = anthropic.AsyncAnthropic(max_retries=2, timeout=120)
    if args.replay:
        rep = asyncio.run(replay(args.replay, args.candidate, client))
        print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=2))
    else:
        versions = args.versions or [pv.version for pv in registry().versions("triage")]
        rep = asyncio.run(compare(versions, client, args.repeats))
        print_report(rep)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(rep, indent=2, default=str))
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
