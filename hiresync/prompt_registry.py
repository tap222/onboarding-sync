"""Prompt versioning.

A prompt version = text + model + effort (behaviour depends on all three), stored
as hiresync/prompts/<name>/v<N>.md with a small front-matter header:

    ---
    description: what changed and why
    model: claude-opus-5-5
    effort: medium
    created: 2026-11-10
    ---
    <prompt text>

Rules that make behavioural testing trustworthy:
  * Versions are immutable. prompts.lock.json pins the sha256 of every version;
    tests fail if an existing version's file changes ("bump the version instead").
  * The active version per prompt is explicit (prompts/active.json), overridable
    per process with PROMPT_<NAME>_VERSION=v2 for A/B or shadow runs.
  * Every model call records name@version + sha + model + effort + tools sha in
    `llm_calls`, so any decision can be traced to the exact prompt that made it,
    and recorded inputs can be replayed against a candidate version.

CLI:
    python -m hiresync.prompt_registry list
    python -m hiresync.prompt_registry lock      # add new versions to the lock (never rewrites)
    python -m hiresync.prompt_registry diff triage v1 v2
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROMPT_DIR = Path(__file__).parent / "prompts"
LOCK_FILE = PROMPT_DIR / "prompts.lock.json"
ACTIVE_FILE = PROMPT_DIR / "active.json"


@dataclass(frozen=True)
class PromptVersion:
    name: str
    version: str
    text: str
    model: str
    effort: str
    description: str
    created: str

    @property
    def sha256(self) -> str:
        """Hash of everything that changes behaviour: text, model, effort."""
        return hashlib.sha256(f"{self.model}\n{self.effort}\n{self.text}".encode()).hexdigest()

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    def meta(self) -> dict:
        return {"prompt": self.ref, "prompt_sha": self.sha256[:16], "model": self.model, "effort": self.effort}


def _parse(path: Path) -> PromptVersion:
    raw = path.read_text()
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.DOTALL)
    if not m:
        raise ValueError(f"{path}: missing front matter")
    meta = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    for key in ("model", "effort", "description"):
        if key not in meta:
            raise ValueError(f"{path}: front matter needs '{key}'")
    return PromptVersion(name=path.parent.name, version=path.stem, text=m.group(2).strip(),
                         model=meta["model"], effort=meta["effort"], description=meta["description"],
                         created=meta.get("created", ""))


def _vkey(v: str) -> int:
    return int(v.lstrip("v"))


class PromptRegistry:
    def __init__(self, root: Path = PROMPT_DIR):
        self.root = root
        self._versions: dict[str, dict[str, PromptVersion]] = {}
        for path in sorted(root.glob("*/v*.md")):
            pv = _parse(path)
            self._versions.setdefault(pv.name, {})[pv.version] = pv
        active_file = root / "active.json"
        self._active: dict[str, str] = json.loads(active_file.read_text()) if active_file.exists() else {}

    def names(self) -> list[str]:
        return sorted(self._versions)

    def versions(self, name: str) -> list[PromptVersion]:
        return [self._versions[name][v] for v in sorted(self._versions[name], key=_vkey)]

    def active_version(self, name: str) -> str:
        return (os.getenv(f"PROMPT_{name.upper()}_VERSION") or self._active.get(name)
                or self.versions(name)[-1].version)

    def get(self, name: str, version: Optional[str] = None) -> PromptVersion:
        version = version or self.active_version(name)
        try:
            return self._versions[name][version]
        except KeyError:
            raise KeyError(f"unknown prompt {name}@{version}; have {sorted(self._versions.get(name, {}))}") from None

    # -------------------------------------------------------------- lock file
    def lock_status(self, lock_path: Optional[Path] = None) -> dict[str, list[str]]:
        """{'changed': [...], 'unlocked': [...]} - changed is the error case."""
        lock_path = lock_path or self.root / "prompts.lock.json"
        lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
        changed, unlocked = [], []
        for name in self.names():
            for pv in self.versions(name):
                pinned = lock.get(name, {}).get(pv.version)
                if pinned is None:
                    unlocked.append(pv.ref)
                elif pinned != pv.sha256:
                    changed.append(pv.ref)
        return {"changed": changed, "unlocked": unlocked}

    def write_lock(self, lock_path: Optional[Path] = None) -> list[str]:
        lock_path = lock_path or self.root / "prompts.lock.json"
        status = self.lock_status(lock_path)
        if status["changed"]:
            raise SystemExit(f"refusing to re-lock edited versions {status['changed']}: "
                             "restore them and create a new version instead")
        lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
        for name in self.names():
            for pv in self.versions(name):
                lock.setdefault(name, {}).setdefault(pv.version, pv.sha256)
        lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
        return status["unlocked"]

    def diff(self, name: str, a: str, b: str) -> str:
        pa, pb = self.get(name, a), self.get(name, b)
        head = [f"model {pa.model} -> {pb.model}\n" if pa.model != pb.model else "",
                f"effort {pa.effort} -> {pb.effort}\n" if pa.effort != pb.effort else ""]
        body = difflib.unified_diff(pa.text.splitlines(True), pb.text.splitlines(True),
                                    fromfile=pa.ref, tofile=pb.ref)
        return "".join(head) + "".join(body)


_registry: Optional[PromptRegistry] = None


def registry() -> PromptRegistry:
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry


def tools_sha(tool_specs: list[dict]) -> str:
    return hashlib.sha256(json.dumps(tool_specs, sort_keys=True).encode()).hexdigest()[:16]


if __name__ == "__main__":
    reg = PromptRegistry()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        for n in reg.names():
            for pv in reg.versions(n):
                mark = "*" if pv.version == reg.active_version(n) else " "
                print(f"{mark} {pv.ref:16} {pv.sha256[:12]} {pv.model}/{pv.effort}  {pv.description}")
        print(reg.lock_status())
    elif cmd == "lock":
        print("locked:", reg.write_lock() or "nothing new")
    elif cmd == "diff":
        print(reg.diff(sys.argv[2], sys.argv[3], sys.argv[4]))
