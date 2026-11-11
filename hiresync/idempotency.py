"""Idempotency claims: one interface, two backends.

DbIdempotency (default, existing stack only)
    INSERT INTO idempotency_keys .. ON CONFLICT DO NOTHING in a transaction is the
    atomic claim; a claim is a lease, so a crashed owner's key is taken over after
    `lease`. On Postgres this is the same statement; no extra infrastructure.

RedisIdempotency (optional, REDIS_URL set)
    SET idem:<scope>:<key> NX PX. Only worth adding when many workers hammer the
    same keys and the DB round-trip matters. Even then the DB unique constraints
    on the business tables stay the guarantee; Redis is a fast pre-check.

claim() -> (owned, cached):
    owned=True                  you do the work, then complete() or release()
    owned=False, IN_FLIGHT      someone else is doing it right now
    owned=False, <result>       already done; here is the stored result
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Any, Optional, Protocol

from .store import Store

IN_FLIGHT = "__in_flight__"


class Idempotency(Protocol):
    claims: int
    hits: int

    async def claim(self, scope: str, key: str) -> tuple[bool, Optional[Any]]: ...
    async def complete(self, scope: str, key: str, result: Any) -> None: ...
    async def release(self, scope: str, key: str) -> None: ...


class DbIdempotency:
    def __init__(self, store: Store, lease_s: float = 30.0, ttl_s: float = 7 * 24 * 3600):
        self.store, self.lease_s, self.ttl_s = store, lease_s, ttl_s
        self.owner = f"w_{uuid.uuid4().hex[:8]}"
        self.claims = self.hits = 0

    async def claim(self, scope: str, key: str) -> tuple[bool, Optional[Any]]:
        owned, row = self.store.idem_claim(scope, key, self.owner, self.lease_s)
        if owned:
            self.claims += 1
            return True, None
        self.hits += 1
        if row is None or row["status"] == "in_flight":
            return False, IN_FLIGHT
        return False, json.loads(row["result"]) if row["result"] else None

    async def complete(self, scope: str, key: str, result: Any) -> None:
        self.store.idem_complete(scope, key, result, self.ttl_s)

    async def release(self, scope: str, key: str) -> None:
        self.store.idem_release(scope, key)


class RedisIdempotency:
    def __init__(self, redis, ttl_ms: int = 24 * 3600 * 1000, lock_ms: int = 30_000):
        self.redis, self.ttl_ms, self.lock_ms = redis, ttl_ms, lock_ms
        self.claims = self.hits = 0

    @staticmethod
    def _k(scope: str, key: str) -> str:
        return f"idem:{scope}:{key}"

    async def claim(self, scope: str, key: str) -> tuple[bool, Optional[Any]]:
        ok = await self.redis.set(self._k(scope, key), f"{IN_FLIGHT}:{uuid.uuid4().hex}", nx=True, px=self.lock_ms)
        if ok:
            self.claims += 1
            return True, None
        self.hits += 1
        existing = await self.redis.get(self._k(scope, key))
        if existing is None:  # expired between SET and GET
            return await self.claim(scope, key)
        if existing.startswith(IN_FLIGHT):
            return False, IN_FLIGHT
        return False, json.loads(existing)

    async def complete(self, scope: str, key: str, result: Any) -> None:
        await self.redis.set(self._k(scope, key), json.dumps(result, default=str), px=self.ttl_ms)

    async def release(self, scope: str, key: str) -> None:
        await self.redis.delete(self._k(scope, key))


def make_idempotency(store: Store, redis=None) -> Idempotency:
    if redis is not None:
        return RedisIdempotency(redis)
    url = os.getenv("REDIS_URL")
    if url:
        import redis.asyncio as aioredis

        return RedisIdempotency(aioredis.from_url(url, decode_responses=True))
    return DbIdempotency(store)
