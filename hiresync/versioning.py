"""Append-only versioning helpers: canonical hashing, flattening, field-level diffs."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel


def canonical_json(data: Any) -> str:
    """Stable serialisation: sorted keys, no whitespace. Same data -> same bytes."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode()).hexdigest()


def flatten(data: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten(value, path))
        else:
            out[path] = value
    return out


def unflatten_set(data: dict, path: str, value: Any) -> dict:
    """Return a copy of `data` with dotted `path` set to `value`."""
    result = json.loads(json.dumps(data, default=str))
    node = result
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
    return result


def deep_merge(base: dict, patch: dict) -> dict:
    result = json.loads(json.dumps(base, default=str))
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class FieldChange(BaseModel):
    path: str
    op: str  # add | remove | replace
    old: Any = None
    new: Any = None


def diff(old: dict | None, new: dict) -> list[FieldChange]:
    """Field-level diff on flattened paths, sorted for deterministic output."""
    a = flatten(old or {})
    b = flatten(new)
    changes: list[FieldChange] = []
    for path in sorted(set(a) | set(b)):
        if path not in a:
            changes.append(FieldChange(path=path, op="add", new=b[path]))
        elif path not in b:
            changes.append(FieldChange(path=path, op="remove", old=a[path]))
        elif a[path] != b[path]:
            changes.append(FieldChange(path=path, op="replace", old=a[path], new=b[path]))
    return changes
