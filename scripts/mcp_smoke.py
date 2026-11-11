"""Connect to the MCP server over stdio as two different users: list tools, call
read and write tools, and check that authorization holds over MCP."""
import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from hiresync.auth import issue_dev_token


def env_for(token):
    return {**os.environ, "MCP_ALLOW_WRITES": "1", "MCP_TOKEN": token, "HIRESYNC_DB": sys.argv[1],
            "AGENT_MODE": "heuristic"}


async def main():
    env = env_for(issue_dev_token("u-maya", "maya@betterup.co", ["people-ops"]))
    params = StdioServerParameters(command=sys.executable, args=["-m", "hiresync.mcp_server"], env=env)
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        tools = (await s.list_tools()).tools
        print("tools:", [(t.name, "ro" if t.annotations.read_only_hint else "rw") for t in tools])
        rec = {"legal_first_name": "Ana", "legal_last_name": "Silva", "personal_email": "a@example.com",
               "start_date": "2026-12-07", "job_title": "Designer", "department": "Design",
               "manager_email": "m@betterup.co", "work_location": {"country": "US"},
               "shipping_address": {"line1": "5 Main", "city": "Reno", "state": "NV", "postal_code": "89501",
                                    "country": "US"}}
        res = await s.call_tool("submit_change", {"hire_id": "H-9", "patch": rec, "reason": "off-cycle hire"})
        print("submit_change:", res.content[0].text)
        res = await s.call_tool("get_run", {"run_id": "H-9:v1"})
        print("get_run:", res.content[0].text[:200])
        res = await s.call_tool("get_system_record", {"system": "salesforce", "hire_id": "H-9"})
        print("bad input -> is_error:", res.is_error, res.content[0].text[:90])

    params = StdioServerParameters(command=sys.executable, args=["-m", "hiresync.mcp_server"],
                                   env=env_for(issue_dev_token("u-dana", "nobody@betterup.co", ["hiring-manager"])))
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        print("manager tools:", [t.name for t in (await s.list_tools()).tools])
        res = await s.call_tool("get_hire", {"hire_id": "H-9"})
        print("manager get_hire H-9 (not their hire):", res.is_error, res.content[0].text)
        res = await s.call_tool("decide_approval", {"approval_id": "x", "decision": "approved"})
        print("manager decide_approval:", res.is_error, res.content[0].text)

asyncio.run(main())
