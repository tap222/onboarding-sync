"""Authentication and authorization.

Authentication: Okta-shaped JWTs (`sub`, `email`, `groups` claims). Okta is
already the identity provider, so groups come from Okta group membership.
  prototype   HS256 tokens from a local dev issuer (issue_dev_token)
  production  RS256, verified against Okta's JWKS (OKTA_ISSUER/.well-known),
              with iss, aud and exp checked. Only `_decode` changes.

Authorization, three layers, all evaluated here and nowhere else:
  1. RBAC: group -> permissions ("hire:read", "approval:decide", ...).
  2. ABAC on the hire: a hiring manager sees only hires whose manager_email is
     theirs; the triage agent's service principal is scoped to the one hire
     it is triaging.
  3. Document ACL: a document's classification (from its type) decides which
     groups may read it; the uploader can always read what they uploaded.
     The same predicate filters vector search (metadata filtering), so search
     can't leak a document that a direct read would deny.
Field masking: home address and personal email are masked unless the principal
has hire:read_pii.
Every deny is audited by the caller.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import jwt

DEV_SECRET = os.getenv("HIRESYNC_DEV_JWT_SECRET", "dev-only-secret-do-not-use-in-prod-0123456789")
ISSUER = os.getenv("HIRESYNC_JWT_ISSUER", "https://dev-issuer.hiresync.local")
AUDIENCE = "hiresync"

GROUP_PERMISSIONS: dict[str, set[str]] = {
    "people-ops": {"hire:read", "hire:read_pii", "hire:write", "approval:decide", "document:upload",
                   "document:search", "run:retry", "audit:read", "episodes:read", "admin:faults",
                   "dlq:read", "dlq:manage", "webhooks:manage", "budget:read"},
    "recruiting": {"hire:read", "hire:write", "document:upload", "document:search"},
    "it": {"hire:read", "hire:read_pii", "document:search"},  # needs the address to ship a laptop
    "hiring-manager": {"hire:read"},  # ABAC: own hires only, no PII
    "auditor": {"hire:read", "audit:read", "episodes:read", "approval:read", "dlq:read", "budget:read"},
    "svc-triage": {"hire:read", "hire:read_pii", "episodes:read", "document:search"},  # ABAC: one hire
    "svc-ashby": {"hire:write"},
}

# Document classification -> groups that may read it.
DOC_CLASSIFICATION = {"offer_letter": "confidential", "it_hardware_form": "internal",
                      "address_change": "restricted", "other": "internal"}
CLASSIFICATION_READERS = {
    "confidential": {"people-ops", "recruiting", "svc-triage"},  # compensation lives here
    "internal": {"people-ops", "recruiting", "it", "svc-triage"},
    "restricted": {"people-ops", "svc-triage"},
}
PII_PATHS = ("personal_email", "shipping_address")


@dataclass(frozen=True)
class Principal:
    sub: str
    email: str
    groups: frozenset[str] = field(default_factory=frozenset)
    kind: str = "user"  # user | service
    hire_scope: Optional[str] = None  # service principals scoped to one hire

    @property
    def permissions(self) -> set[str]:
        perms: set[str] = set()
        for g in self.groups:
            perms |= GROUP_PERMISSIONS.get(g, set())
        return perms

    def has(self, perm: str) -> bool:
        return perm in self.permissions


ANONYMOUS = Principal(sub="anonymous", email="", groups=frozenset())


def service_principal(name: str, hire_id: Optional[str] = None) -> Principal:
    return Principal(sub=name, email=f"{name}@hiresync.internal", groups=frozenset({name}),
                     kind="service", hire_scope=hire_id)


class AuthError(Exception):
    pass


class Forbidden(Exception):
    def __init__(self, principal: Principal, action: str, resource: str, reason: str):
        super().__init__(f"{principal.sub} may not {action} {resource}: {reason}")
        self.principal, self.action, self.resource, self.reason = principal, action, resource, reason


# --------------------------------------------------------------- tokens
def issue_dev_token(sub: str, email: str, groups: list[str], ttl_s: int = 3600) -> str:
    now = int(time.time())
    return jwt.encode({"iss": ISSUER, "aud": AUDIENCE, "sub": sub, "email": email, "groups": groups,
                       "iat": now, "exp": now + ttl_s}, DEV_SECRET, algorithm="HS256")


def _decode(token: str) -> dict:
    return jwt.decode(token, DEV_SECRET, algorithms=["HS256"], audience=AUDIENCE, issuer=ISSUER,
                      options={"require": ["exp", "sub", "iss", "aud"]})


def authenticate(authorization: Optional[str]) -> Principal:
    """Bearer token -> Principal. No/invalid token -> AuthError (never a default user)."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AuthError("missing bearer token")
    try:
        claims = _decode(authorization.split(" ", 1)[1].strip())
    except jwt.PyJWTError as exc:
        raise AuthError(f"invalid token: {exc}") from exc
    groups = frozenset(g for g in claims.get("groups", []) if not g.startswith("svc-"))  # users can't claim service groups
    return Principal(sub=claims["sub"], email=claims.get("email", ""), groups=groups)


# ---------------------------------------------------------- decisions
def require(p: Principal, perm: str, resource: str = "*") -> None:
    if not p.has(perm):
        raise Forbidden(p, perm, resource, f"needs '{perm}' (groups: {sorted(p.groups) or 'none'})")


def can_access_hire(p: Principal, hire_id: str, record: Optional[dict]) -> tuple[bool, str]:
    if not p.has("hire:read"):
        return False, "no hire:read"
    if p.hire_scope is not None:
        return (p.hire_scope == hire_id), f"scoped to {p.hire_scope}"
    if p.groups & {"people-ops", "recruiting", "it", "auditor"}:
        return True, "role"
    if "hiring-manager" in p.groups:
        ok = bool(record) and record.get("manager_email", "").lower() == p.email.lower()
        return ok, "manager of this hire" if ok else "not this hire's manager"
    return False, "no role grants hire access"


def require_hire(p: Principal, hire_id: str, record: Optional[dict]) -> None:
    ok, why = can_access_hire(p, hire_id, record)
    if not ok:
        raise Forbidden(p, "read", f"hire {hire_id}", why)


def can_read_document(p: Principal, doc: dict, uploaders: set[str]) -> tuple[bool, str]:
    """doc needs: hire_id, classification."""
    if p.hire_scope is not None and doc["hire_id"] != p.hire_scope:
        return False, f"scoped to {p.hire_scope}"
    if p.email and p.email.lower() in {u.lower() for u in uploaders}:
        return True, "uploader"
    readers = CLASSIFICATION_READERS.get(doc.get("classification") or "restricted", set())
    if p.groups & readers:
        return True, f"group may read {doc.get('classification')}"
    return False, f"{doc.get('classification')} documents are limited to {sorted(readers)}"


def doc_filter(p: Principal, hire_id: Optional[str] = None, doc_types: Optional[set[str]] = None,
               uploads_of: Optional[Callable[[str], set[str]]] = None) -> Callable[[dict], bool]:
    """Metadata filter for vector search: identical decision to can_read_document,
    plus optional caller filters (hire, doc type)."""
    def allowed(payload: dict) -> bool:
        if hire_id and payload["hire_id"] != hire_id:
            return False
        if doc_types and payload.get("doc_type") not in doc_types:
            return False
        ups = uploads_of(payload["doc_hash"]) if uploads_of else set()
        return can_read_document(p, payload, ups)[0]
    return allowed


def mask_record(p: Principal, record: dict) -> dict:
    if p.has("hire:read_pii"):
        return record
    masked = dict(record)
    for path in PII_PATHS:
        if path in masked:
            masked[path] = "[MASKED]"
    return masked


def mask_path_value(p: Principal, path: str, value):
    if p.has("hire:read_pii") or not path.startswith(PII_PATHS):
        return value
    return "[MASKED]"
