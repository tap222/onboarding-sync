"""MCP server: exposes the tool registry to MCP clients (Claude Desktop, Claude Code)
so People Ops can ask "why is H-1002 held?" or approve a change in plain language.

Authentication: the caller's token (MCP_TOKEN, an Okta-shaped JWT) -> Principal.
The tool LIST is per caller: a recruiter sees upload/search tools, an auditor sees
read tools, nobody sees a tool their groups don't grant. Calls go through the same
`tools.execute` choke point as GraphQL and the in-pipeline agent: permission, ABAC
on the hire, document ACL, masking, PII redaction, audit.

Extra guardrail: write tools are hidden unless MCP_ALLOW_WRITES=1, even for users
whose groups allow them (so a read-only deployment can't be talked into writing).
Tools carry MCP annotations (read_only_hint, idempotent_hint, destructive_hint) so
the client can confirm before writes.

Production: the MCP authorization spec (OAuth 2.1) with Okta as the authorization
server; the token arrives per session instead of via an environment variable.

Run (stdio):  MCP_TOKEN=<jwt> python -m hiresync.mcp_server
"""
from __future__ import annotations

import asyncio
import os

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from . import tools as toolreg
from .app import HireSync
from .auth import Principal, authenticate

ALLOW_WRITES = os.getenv("MCP_ALLOW_WRITES") == "1"
IDEMPOTENT = {"submit_change": False, "decide_approval": True, "upload_document": True, "retry_run": True,
              "redrive_dead_letter": True, "discard_dead_letter": True, "create_webhook_subscription": False}


def allowed_names() -> frozenset[str]:
    return frozenset(toolreg.REGISTRY) if ALLOW_WRITES else toolreg.READ_ONLY


def mcp_tools(principal: Principal) -> list[types.Tool]:
    out = []
    for t in sorted(toolreg.tools_for(principal, allowed_names()), key=lambda t: t.name):
        out.append(types.Tool(
            name=t.name, description=t.description, input_schema=t.input_model.model_json_schema(),
            annotations=types.ToolAnnotations(
                read_only_hint=t.read_only,
                destructive_hint=False,  # nothing deletes; every write is a new version
                idempotent_hint=True if t.read_only else IDEMPOTENT.get(t.name, False),
                open_world_hint=False,
            ),
        ))
    return out


def build_server(app: HireSync, principal: Principal) -> Server:
    async def on_list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=mcp_tools(principal))

    async def on_call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        text, is_error = await toolreg.execute(app, params.name, params.arguments or {},
                                               principal=principal, allow=allowed_names())
        if not is_error and not toolreg.REGISTRY[params.name].read_only:
            await app.drain()  # let the pipeline react before answering
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)], is_error=is_error)

    return Server("hiresync", version="0.2.0", on_list_tools=on_list_tools, on_call_tool=on_call_tool)


async def main() -> None:
    principal = authenticate(f"Bearer {os.environ.get('MCP_TOKEN', '')}")  # fails closed
    app = HireSync(db_path=os.getenv("HIRESYNC_DB", "hiresync.db"))
    await app.start()
    server = build_server(app, principal)
    try:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    finally:
        await app.stop()


if __name__ == "__main__":
    asyncio.run(main())
