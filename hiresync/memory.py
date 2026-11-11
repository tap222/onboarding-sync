"""Two kinds of memory, with different lifetimes.

WorkingMemory  - everything one run knows right now (versions, diff, issues, plan,
                 per-system results, agent output). Serialised into the run's
                 checkpoint after every step, so it *is* the resume state.
EpisodicMemory - what happened in past runs and how each was resolved (who decided,
                 what, and the outcome). Recalled by similarity and given to the
                 triage agent, so a human's past decisions shape the next one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .store import Store


@dataclass
class WorkingMemory:
    run_id: str
    hire_id: str
    from_version: Optional[int]
    to_version: int
    trace_id: Optional[str] = None
    decision: Optional[str] = None  # auto | agent_proceed | approved | corrected | rejected
    old: Optional[dict] = None
    new: dict = field(default_factory=dict)
    changes: list[dict] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)  # systems whose projection changed
    recalled: list[dict] = field(default_factory=list)  # episodes shown to the agent
    recommendation: Optional[dict] = None
    approval_id: Optional[str] = None
    approved_by: Optional[str] = None
    results: dict[str, dict] = field(default_factory=dict)  # system -> {status, response, error}
    reconciliation: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    compensation_status: Optional[str] = None
    compensations: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WorkingMemory":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def features(self) -> list[str]:
        """The signature used to find similar past episodes."""
        feats = {f"issue:{i['code']}:{i['severity']}" for i in self.issues}
        if self.from_version is None:
            feats.add("kind:new_hire")  # every field "changes" on creation; that's noise
        else:
            feats |= {f"field:{c['path'].split('.')[0]}" for c in self.changes}
        return sorted(feats)

    def summary(self) -> str:
        codes = sorted({i["code"] for i in self.issues})
        if self.from_version is None:
            return f"{self.hire_id} new hire (v{self.to_version}); issues {codes}"
        fields = sorted({c["path"] for c in self.changes})
        return f"{self.hire_id} v{self.from_version}->v{self.to_version}: changed {fields}; issues {codes}"

    def prompt_view(self) -> dict[str, Any]:
        """A bounded, PII-minimised view for the model: the diff and issues, not the whole record."""
        return {
            "hire_id": self.hire_id,
            "versions": {"from": self.from_version, "to": self.to_version},
            "changes": self.changes,
            "issues": self.issues,
            "target_systems": self.targets,
            "department": self.new.get("department"),
            "start_date": self.new.get("start_date"),
            "work_country": self.new.get("work_location", {}).get("country"),
            "ship_country": self.new.get("shipping_address", {}).get("country"),
        }


class EpisodicMemory:
    """Hybrid recall. Candidates come from the HNSW index over episode text
    (situation + features); each is scored 0.7 * Jaccard(features) + 0.3 * cosine.
    The exact-feature term keeps "same issue codes" dominant; the vector term
    ranks near-misses and scales past a full scan as episodes accumulate."""

    W_FEATURES, W_VECTOR = 0.7, 0.3

    def __init__(self, store: Store, vectors=None):
        self.store = store
        self.vectors = vectors

    def record(self, wm: WorkingMemory, decision: str, decided_by: str, outcome: str) -> None:
        self.store.save_episode(wm.hire_id, wm.run_id, wm.features(), wm.summary(),
                                decision, decided_by, outcome)
        if self.vectors is not None:
            self.vectors.index_episode(self.store.episode(f"ep_{wm.run_id}"))

    def recall(self, features: list[str], k: int = 3, min_similarity: float = 0.34,
               exclude_run: Optional[str] = None, text: Optional[str] = None) -> list[dict]:
        target = set(features)
        if not target:
            return []
        query = f"{text or ''} {' '.join(features)}"
        if self.vectors is not None and len(self.vectors.episodes):
            cos = self.vectors.search_episodes(query, k=50)
            candidates = [e for e in (self.store.episode(i) for i in cos) if e]
        else:
            cos, candidates = {}, self.store.all_episodes()
        scored = []
        for ep in candidates:
            if ep["run_id"] == exclude_run:
                continue
            jac = len(target & set(ep["features"])) / len(target | set(ep["features"]))
            score = self.W_FEATURES * jac + self.W_VECTOR * max(0.0, cos.get(ep["episode_id"], jac))
            if score >= min_similarity:
                scored.append((score, ep["created_at"], ep))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)  # newest wins ties
        return [
            {"run_id": e["run_id"], "similarity": round(sc, 2), "situation": e["situation"], "decision": e["decision"],
             "decided_by": e["decided_by"], "outcome": e["outcome"], "features": e["features"]}
            for sc, _, e in scored[:k]
        ]
