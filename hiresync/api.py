"""HTTP entrypoint: GraphQL at /graphql (Bearer token required), plus an Ashby-shaped
webhook authenticated by HMAC signature rather than a user token.

Run:  uvicorn hiresync.api:api --reload
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from strawberry.fastapi import GraphQLRouter

from . import webhooks
from .app import HireSync, IngestError
from .auth import AuthError, authenticate
from .graphql_schema import schema
from .store import ConcurrencyError

ASHBY_SECRET = os.getenv("ASHBY_WEBHOOK_SECRET", "dev-ashby-secret").encode()
app_state: Optional[HireSync] = None  # created at startup, not on import (tests inject one)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global app_state
    if app_state is None:
        app_state = HireSync(db_path=os.getenv("HIRESYNC_DB", "hiresync.db"))
    await app_state.start()
    yield
    await app_state.stop()


async def graphql_context(request: Request) -> dict:
    try:
        principal = authenticate(request.headers.get("authorization"))
    except AuthError as exc:
        raise HTTPException(401, str(exc), headers={"WWW-Authenticate": "Bearer"}) from exc
    return {"app": app_state, "principal": principal}


api = FastAPI(title="HireSync", lifespan=lifespan)
api.include_router(GraphQLRouter(schema, context_getter=graphql_context), prefix="/graphql")


def ashby_signature(body: bytes, ts: int | None = None) -> str:
    """t=<unix>,v1=<hmac_sha256(secret, f"{t}.{body}")>: the timestamp is signed too."""
    return webhooks.sign(ASHBY_SECRET.decode(), body, ts)


@api.post("/webhooks/ashby")
async def ashby_webhook(request: Request, idempotency_key: str = Header(..., alias="Idempotency-Key"),
                        signature: str = Header("", alias="Ashby-Signature")):
    body = await request.body()
    # Authenticity + replay protection (timestamp within 5 min); Idempotency-Key
    # then dedupes legitimate redeliveries.
    if not webhooks.verify(ASHBY_SECRET.decode(), body, signature):
        raise HTTPException(401, "bad or stale signature")
    payload = json.loads(body)
    try:
        return await app_state.submit_change(payload["hire_id"], payload["patch"], source="ashby",
                                             actor="svc-ashby", event_id=idempotency_key)
    except IngestError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ConcurrencyError as exc:
        raise HTTPException(409, str(exc)) from exc


@api.get("/healthz")
async def healthz():
    return {"ok": True}
