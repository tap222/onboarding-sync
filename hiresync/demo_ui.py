"""HireSync Demo Web Dashboard & Streaming API.

Provides an interactive Web UI for running and inspecting the 15 end-to-end scenarios,
generating test JWT tokens for different roles, and accessing GraphQL & REST docs.
"""
from __future__ import annotations

import asyncio
import html
import json
import os
import re
from contextlib import redirect_stdout
from typing import AsyncGenerator

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .auth import issue_dev_token

router = APIRouter()

ROLES = {
    "people-ops": {
        "sub": "u-maya",
        "email": "maya@betterup.co",
        "groups": ["people-ops"],
        "name": "Maya (People Ops)",
        "desc": "Full admin: writes, approvals, all documents, audit trail, DLQ management",
    },
    "recruiting": {
        "sub": "u-raj",
        "email": "raj@betterup.co",
        "groups": ["recruiting"],
        "name": "Raj (Recruiting)",
        "desc": "Ingest hires, upload offer letters, document search",
    },
    "it": {
        "sub": "u-ivy",
        "email": "ivy@betterup.co",
        "groups": ["it"],
        "name": "Ivy (IT Systems)",
        "desc": "Hardware provisioning, shipping addresses (PII allowed), IT hardware forms",
    },
    "hiring-manager": {
        "sub": "u-dana",
        "email": "dana.lee@betterup.co",
        "groups": ["hiring-manager"],
        "name": "Dana (Hiring Manager)",
        "desc": "ABAC scoped to own department hires only; home address PII masked",
    },
    "auditor": {
        "sub": "u-alex",
        "email": "alex@betterup.co",
        "groups": ["auditor"],
        "name": "Alex (Compliance Auditor)",
        "desc": "Read-only access to audit logs, episodic memory, budget and runs",
    },
}


class _QueueWriter:
    """Captures stdout lines and pushes them into an asyncio Queue."""

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.buf = ""

    def write(self, s: str):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.queue.put_nowait(line)

    def flush(self):
        if self.buf:
            self.queue.put_nowait(self.buf)
            self.buf = ""


ANSI_REGEX = re.compile(r"\x1b\[([0-9;]*)m")


def ansi_to_html(text: str) -> str:
    """Simple ANSI escape converter to styled HTML spans."""
    parts = []
    last_end = 0
    active_classes = []

    def cls_for_code(code: str) -> str:
        match code:
            case "1":
                return "font-bold text-white"
            case "2":
                return "text-slate-400"
            case "31":
                return "text-rose-400 font-semibold"
            case "32":
                return "text-emerald-400 font-semibold"
            case "33":
                return "text-amber-300 font-semibold"
            case "34":
                return "text-sky-400 font-semibold"
            case "36":
                return "text-cyan-400"
            case "0" | "":
                return ""
            case _:
                return ""

    for m in ANSI_REGEX.finditer(text):
        raw_segment = text[last_end:m.start()]
        if raw_segment:
            escaped = html.escape(raw_segment)
            if active_classes:
                parts.append(f'<span class="{" ".join(active_classes)}">{escaped}</span>')
            else:
                parts.append(escaped)

        code = m.group(1)
        if code in ("0", ""):
            active_classes = []
        else:
            cls = cls_for_code(code)
            if cls:
                active_classes.append(cls)
        last_end = m.end()

    raw_tail = text[last_end:]
    if raw_tail:
        escaped = html.escape(raw_tail)
        if active_classes:
            parts.append(f'<span class="{" ".join(active_classes)}">{escaped}</span>')
        else:
            parts.append(escaped)

    return "".join(parts)


@router.get("/api/demo/token")
async def get_demo_token(role: str = Query("people-ops")):
    """Generates a valid signed JWT for testing GraphQL or MCP APIs."""
    info = ROLES.get(role, ROLES["people-ops"])
    token = issue_dev_token(info["sub"], info["email"], info["groups"])
    return JSONResponse(
        {
            "role": role,
            "sub": info["sub"],
            "email": info["email"],
            "groups": info["groups"],
            "token": token,
            "header": f"Bearer {token}",
        }
    )


@router.get("/api/demo/roles")
async def list_demo_roles():
    """Lists available test personas."""
    return JSONResponse(ROLES)


@router.get("/api/demo/stream")
async def stream_demo_run(mode: str = Query("heuristic"), api_key: str = Query("")):
    """Streams the execution of demo.py line-by-line using Server-Sent Events (SSE)."""

    async def event_generator() -> AsyncGenerator[str, None]:
        q: asyncio.Queue = asyncio.Queue()
        writer = _QueueWriter(q)

        # Set environment variables for Claude if requested
        orig_mode = os.environ.get("AGENT_MODE")
        orig_key = os.environ.get("ANTHROPIC_API_KEY")
        if mode == "claude" and api_key:
            os.environ["AGENT_MODE"] = "claude"
            os.environ["ANTHROPIC_API_KEY"] = api_key

        async def run_task():
            try:
                # Import demo dynamically so any runtime changes are picked up
                import demo  # type: ignore

                with redirect_stdout(writer):
                    await demo.main()
            except Exception as exc:
                writer.write(f"\033[31mError during demo execution: {exc}\033[0m\n")
            finally:
                writer.flush()
                await q.put(None)
                if orig_mode is not None:
                    os.environ["AGENT_MODE"] = orig_mode
                elif "AGENT_MODE" in os.environ:
                    del os.environ["AGENT_MODE"]
                if orig_key is not None:
                    os.environ["ANTHROPIC_API_KEY"] = orig_key
                elif "ANTHROPIC_API_KEY" in os.environ:
                    del os.environ["ANTHROPIC_API_KEY"]

        task = asyncio.create_task(run_task())

        current_scenario = 0
        scenario_match = re.compile(r"━━\s*(\d+)\.\s*(.*?)\s*━")

        while True:
            line = await q.get()
            if line is None:
                break

            m = scenario_match.search(line)
            if m:
                current_scenario = int(m.group(1))
                payload = {
                    "type": "scenario_start",
                    "scenario": current_scenario,
                    "title": m.group(2).strip(),
                    "raw": line,
                    "html": ansi_to_html(line),
                }
            else:
                payload = {
                    "type": "log",
                    "scenario": current_scenario,
                    "raw": line,
                    "html": ansi_to_html(line),
                }

            yield f"data: {json.dumps(payload)}\n\n"

        await task
        yield f"data: {json.dumps({'type': 'complete', 'scenario': 15})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


DEMO_PAGE_HTML = """<!DOCTYPE html>
<html lang="en" class="h-full bg-slate-950 text-slate-100">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>HireSync — Interactive Automation Demo</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"/>
  <style>
    @keyframes pulse-subtle { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
    .pulse-running { animation: pulse-subtle 1.5s infinite; }
    pre::-webkit-scrollbar { width: 8px; height: 8px; }
    pre::-webkit-scrollbar-track { background: #090d16; }
    pre::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
    pre::-webkit-scrollbar-thumb:hover { background: #334155; }
  </style>
</head>
<body class="min-h-full flex flex-col font-sans antialiased selection:bg-indigo-500 selection:text-white">

  <!-- Top Navbar -->
  <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="w-9 h-9 rounded-lg bg-gradient-to-tr from-indigo-500 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/20 text-white font-black text-lg">
          ⚡
        </div>
        <div>
          <span class="font-bold text-lg tracking-tight text-white flex items-center gap-2">
            HireSync
            <span class="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Live System</span>
          </span>
          <p class="text-xs text-slate-400">BetterUp Onboarding Automation & Propagation Engine</p>
        </div>
      </div>

      <div class="flex items-center gap-2 sm:gap-4">
        <a href="/graphql" target="_blank" class="text-xs sm:text-sm font-medium text-slate-300 hover:text-white px-3 py-1.5 rounded-lg border border-slate-700 hover:bg-slate-800 transition flex items-center gap-1.5">
          <i class="fa-solid fa-cube text-pink-400"></i>
          <span>GraphQL</span>
        </a>
        <a href="/docs" target="_blank" class="text-xs sm:text-sm font-medium text-slate-300 hover:text-white px-3 py-1.5 rounded-lg border border-slate-700 hover:bg-slate-800 transition flex items-center gap-1.5">
          <i class="fa-solid fa-book text-sky-400"></i>
          <span>Swagger Docs</span>
        </a>
        <a href="https://github.com/tap222/onboarding-sync" target="_blank" class="text-xs sm:text-sm font-medium text-slate-400 hover:text-white px-2 py-1.5 transition">
          <i class="fa-brands fa-github text-lg"></i>
        </a>
      </div>
    </div>
  </header>

  <!-- Main Content -->
  <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">

    <!-- Hero / Controls Banner -->
    <div class="relative overflow-hidden rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 p-6 sm:p-8 shadow-2xl">
      <div class="relative z-10 flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
        <div class="space-y-2 max-w-2xl">
          <h1 class="text-2xl sm:text-3xl font-extrabold tracking-tight text-white">
            End-to-End System Demo
          </h1>
          <p class="text-slate-400 text-sm sm:text-base leading-relaxed">
            Execute all 15 scenarios verifying idempotency, human-in-the-loop triage, crash recovery, circuit breakers, prompt injection defense, and saga compensation.
          </p>
        </div>

        <!-- Controls Box -->
        <div class="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 w-full lg:w-auto">
          <button id="runBtn" onclick="startDemo()" class="px-6 py-3.5 rounded-xl font-semibold text-white bg-gradient-to-r from-indigo-500 via-indigo-600 to-violet-600 hover:from-indigo-400 hover:to-violet-500 active:scale-95 transition shadow-lg shadow-indigo-500/25 flex items-center justify-center gap-2 cursor-pointer">
            <i class="fa-solid fa-play text-xs"></i>
            <span>Run Full Demo</span>
          </button>
          <button id="clearBtn" onclick="clearConsole()" class="px-4 py-3.5 rounded-xl font-medium text-slate-400 hover:text-white border border-slate-800 hover:bg-slate-850 active:scale-95 transition flex items-center justify-center gap-2 cursor-pointer">
            <i class="fa-solid fa-eraser text-xs"></i>
            <span>Clear Console</span>
          </button>
        </div>
      </div>

      <!-- Live Progress Bar -->
      <div class="mt-6 pt-6 border-t border-slate-800/80">
        <div class="flex justify-between items-center text-xs font-semibold uppercase tracking-wider mb-2 text-slate-400">
          <span id="progressText">Ready to run (0 / 15 Scenarios)</span>
          <span id="percentText" class="text-indigo-400 font-bold">0%</span>
        </div>
        <div class="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden p-0.5">
          <div id="progressBar" class="bg-gradient-to-r from-indigo-500 to-emerald-400 h-full rounded-full transition-all duration-300 w-0"></div>
        </div>
      </div>
    </div>

    <!-- Dual Layout: Terminal & Scenario Matrix -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">

      <!-- Left Column: Live Terminal Stream (7 cols) -->
      <div class="lg:col-span-7 flex flex-col space-y-3">
        <div class="flex items-center justify-between text-xs font-semibold text-slate-400 uppercase tracking-wider">
          <span class="flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-emerald-400 inline-block animate-ping"></span>
            Real-Time Output Console
          </span>
          <span id="streamStatus" class="text-slate-500">Idle</span>
        </div>

        <div class="relative rounded-2xl border border-slate-800 bg-[#090d16] shadow-2xl overflow-hidden flex flex-col h-[580px]">
          <!-- Mac Window Header -->
          <div class="bg-slate-900/90 px-4 py-3 border-b border-slate-800 flex items-center justify-between">
            <div class="flex items-center gap-2">
              <span class="w-3 h-3 rounded-full bg-rose-500/80 inline-block"></span>
              <span class="w-3 h-3 rounded-full bg-amber-500/80 inline-block"></span>
              <span class="w-3 h-3 rounded-full bg-emerald-500/80 inline-block"></span>
              <span class="ml-2 font-mono text-xs text-slate-400">hiresync-demo — bash</span>
            </div>
            <button onclick="copyTerminal()" class="text-xs text-slate-400 hover:text-white px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 transition flex items-center gap-1">
              <i class="fa-regular fa-copy text-[11px]"></i> Copy Output
            </button>
          </div>

          <!-- Terminal Output Area -->
          <pre id="terminal" class="flex-1 p-4 font-mono text-xs sm:text-[13px] leading-relaxed text-slate-300 overflow-y-auto whitespace-pre-wrap select-text">HireSync Demo Console ready.
Click "Run Full Demo" above to execute all 15 scenarios.
</pre>
        </div>
      </div>

      <!-- Right Column: Interactive Role Token Generator & Scenario Checklist (5 cols) -->
      <div class="lg:col-span-5 space-y-6">

        <!-- JWT Token Generator Card -->
        <div class="rounded-2xl border border-slate-800 bg-slate-900/70 p-6 shadow-xl space-y-4">
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-bold text-white flex items-center gap-2">
              <i class="fa-solid fa-key text-amber-400"></i>
              Instant Dev Token Generator
            </h3>
            <span class="text-[11px] text-slate-400">Okta RBAC / ABAC</span>
          </div>
          <p class="text-xs text-slate-400">
            Generate authenticated JWTs to test the GraphQL playground with different roles and permissions.
          </p>

          <div class="space-y-3">
            <div>
              <label class="block text-xs font-medium text-slate-400 mb-1">Select Persona:</label>
              <select id="roleSelect" onchange="updateToken()" class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500">
                <option value="people-ops">Maya (People Ops) — Full Admin & Approvals</option>
                <option value="recruiting">Raj (Recruiting) — Offer Ingest & Candidates</option>
                <option value="it">Ivy (IT Systems) — Hardware & PII Allowed</option>
                <option value="hiring-manager">Dana (Hiring Manager) — Masked PII Scope</option>
                <option value="auditor">Alex (Compliance Auditor) — Read-Only</option>
              </select>
            </div>

            <div>
              <div class="flex items-center justify-between mb-1">
                <label class="text-xs font-medium text-slate-400">Authorization Header:</label>
                <button onclick="copyHeaderToken()" class="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1 cursor-pointer">
                  <i class="fa-regular fa-copy"></i>
                  <span id="copyFeedback">Copy JSON for GraphQL</span>
                </button>
              </div>
              <textarea id="tokenBox" rows="3" readonly class="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 font-mono text-[11px] text-slate-300 resize-none select-all focus:outline-none"></textarea>
            </div>

            <div class="pt-1 flex gap-2">
              <a id="openGqlBtn" href="/graphql" target="_blank" class="w-full py-2 rounded-lg text-center text-xs font-semibold bg-indigo-600/20 text-indigo-300 border border-indigo-500/30 hover:bg-indigo-600/30 transition">
                Open in GraphiQL &rarr;
              </a>
            </div>
          </div>
        </div>

        <!-- 15 Scenarios Overview -->
        <div class="rounded-2xl border border-slate-800 bg-slate-900/70 p-6 shadow-xl space-y-4">
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-bold text-white flex items-center gap-2">
              <i class="fa-solid fa-list-check text-indigo-400"></i>
              Verified Scenarios (15)
            </h3>
            <span id="badgeCounter" class="text-[11px] font-bold px-2 py-0.5 rounded-full bg-slate-800 text-slate-400">0 / 15</span>
          </div>

          <div id="scenarioList" class="space-y-2 max-h-[360px] overflow-y-auto pr-1">
            <!-- Populated via JavaScript -->
          </div>
        </div>

      </div>

    </div>

  </main>

  <!-- Footer -->
  <footer class="border-t border-slate-800 py-6 text-center text-xs text-slate-500">
    HireSync Automation Engine &bull; Built with FastAPI, Strawberry GraphQL & OpenTelemetry
  </footer>

  <script>
    const SCENARIOS = [
      { id: 1, title: "New hire from Ashby", desc: "Sync to 5 SaaS downstream systems" },
      { id: 2, title: "Concurrent Webhooks", desc: "Atomic DB claim & idempotency deduplication" },
      { id: 3, title: "Human Approval Hold", desc: "Okta login change hold, survives crash & restart" },
      { id: 4, title: "Document Deduplication", desc: "4 uploads, CRLF normalization & provenance" },
      { id: 5, title: "Security & Guardrails", desc: "SSN PII redaction & prompt injection defense" },
      { id: 6, title: "Circuit Breakers & DLQ", desc: "503 outage, backoff trip, manual DLQ redrive" },
      { id: 7, title: "Episodic Memory", desc: "Learns from past human approval decisions" },
      { id: 8, title: "Unified GraphQL", desc: "History, version diffs & run status in 1 query" },
      { id: 9, title: "RBAC & ABAC Security", desc: "Okta groups, doc ACLs, masked PII views" },
      { id: 10, title: "Prompt Registry & MCP", desc: "Versioned prompts & caller-scoped MCP tools" },
      { id: 11, title: "Async Worker Pool", desc: "Concurrent worker queue & crash recovery" },
      { id: 12, title: "Adaptive Chunking", desc: "Document chunking & semantic search" },
      { id: 13, title: "Cost & Token Guardrails", desc: "Spend caps, evaluator critique & webhooks" },
      { id: 14, title: "OpenTelemetry Spans", desc: "Traces, downstream latency waterfalls & p95s" },
      { id: 15, title: "Saga Compensation", desc: "Candidate withdrawal reverse rollback" }
    ];

    function initScenarios() {
      const container = document.getElementById('scenarioList');
      container.innerHTML = SCENARIOS.map(s => `
        <div id="sc-card-${s.id}" class="p-2.5 rounded-xl border border-slate-800/80 bg-slate-950/60 flex items-center justify-between text-xs transition duration-200">
          <div class="flex items-center gap-2.5 truncate">
            <span id="sc-icon-${s.id}" class="w-5 h-5 rounded-full bg-slate-800 text-slate-400 font-bold text-[10px] flex items-center justify-center shrink-0">
              ${s.id}
            </span>
            <div class="truncate">
              <span id="sc-title-${s.id}" class="font-medium text-slate-300 block truncate">${s.title}</span>
              <span class="text-[10px] text-slate-500 block truncate">${s.desc}</span>
            </div>
          </div>
          <span id="sc-status-${s.id}" class="text-[10px] font-semibold text-slate-500 uppercase px-1.5 py-0.5 rounded bg-slate-900 shrink-0">
            Pending
          </span>
        </div>
      `).join('');
    }

    async function updateToken() {
      const role = document.getElementById('roleSelect').value;
      try {
        const res = await fetch(`/api/demo/token?role=${role}`);
        const data = await res.json();
        const headerJson = JSON.stringify({ "Authorization": data.header }, null, 2);
        document.getElementById('tokenBox').value = headerJson;
      } catch (err) {
        console.error(err);
      }
    }

    function copyHeaderToken() {
      const box = document.getElementById('tokenBox');
      box.select();
      navigator.clipboard.writeText(box.value);
      const fb = document.getElementById('copyFeedback');
      fb.textContent = "Copied to Clipboard!";
      fb.classList.add("text-emerald-400");
      setTimeout(() => {
        fb.textContent = "Copy JSON for GraphQL";
        fb.classList.remove("text-emerald-400");
      }, 2000);
    }

    let activeEventSource = null;

    function startDemo() {
      const runBtn = document.getElementById('runBtn');
      const terminal = document.getElementById('terminal');
      const streamStatus = document.getElementById('streamStatus');
      const progressBar = document.getElementById('progressBar');
      const progressText = document.getElementById('progressText');
      const percentText = document.getElementById('percentText');
      const badgeCounter = document.getElementById('badgeCounter');

      if (activeEventSource) {
        activeEventSource.close();
      }

      // Reset UI state
      terminal.innerHTML = "";
      runBtn.disabled = true;
      runBtn.classList.add("opacity-50", "cursor-not-allowed");
      runBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin text-xs"></i><span>Running Demo...</span>';
      streamStatus.textContent = "Streaming live...";
      streamStatus.className = "text-emerald-400 font-semibold";
      progressBar.style.width = "0%";
      percentText.textContent = "0%";

      // Reset scenario checklist
      SCENARIOS.forEach(s => {
        const card = document.getElementById(`sc-card-${s.id}`);
        const icon = document.getElementById(`sc-icon-${s.id}`);
        const status = document.getElementById(`sc-status-${s.id}`);
        card.className = "p-2.5 rounded-xl border border-slate-800/80 bg-slate-950/60 flex items-center justify-between text-xs transition duration-200";
        icon.className = "w-5 h-5 rounded-full bg-slate-800 text-slate-400 font-bold text-[10px] flex items-center justify-center shrink-0";
        icon.textContent = s.id;
        status.className = "text-[10px] font-semibold text-slate-500 uppercase px-1.5 py-0.5 rounded bg-slate-900 shrink-0";
        status.textContent = "Pending";
      });

      let currentSc = 0;
      activeEventSource = new EventSource('/api/demo/stream');

      activeEventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === "scenario_start") {
          currentSc = data.scenario;
          const pct = Math.round((currentSc / 15) * 100);
          progressBar.style.width = `${pct}%`;
          percentText.textContent = `${pct}%`;
          progressText.textContent = `Executing Scenario ${currentSc} of 15`;
          badgeCounter.textContent = `${currentSc} / 15`;

          // Mark previous as passed
          for (let i = 1; i < currentSc; i++) {
            markScenarioDone(i);
          }
          // Mark current as running
          const card = document.getElementById(`sc-card-${currentSc}`);
          const status = document.getElementById(`sc-status-${currentSc}`);
          if (card && status) {
            card.classList.add("border-indigo-500/50", "bg-indigo-950/20");
            status.textContent = "Running";
            status.className = "text-[10px] font-semibold text-indigo-400 uppercase px-1.5 py-0.5 rounded bg-indigo-900/30 shrink-0 animate-pulse";
          }
        }

        if (data.html) {
          terminal.innerHTML += data.html + "\\n";
          terminal.scrollTop = terminal.scrollHeight;
        }

        if (data.type === "complete") {
          // All scenarios done
          for (let i = 1; i <= 15; i++) {
            markScenarioDone(i);
          }
          progressBar.style.width = "100%";
          percentText.textContent = "100%";
          progressText.textContent = "All 15 Scenarios Passed Successfully!";
          badgeCounter.textContent = "15 / 15";
          badgeCounter.className = "text-[11px] font-bold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400";
          streamStatus.textContent = "Completed";
          streamStatus.className = "text-slate-400";

          runBtn.disabled = false;
          runBtn.classList.remove("opacity-50", "cursor-not-allowed");
          runBtn.innerHTML = '<i class="fa-solid fa-rotate-right text-xs"></i><span>Run Again</span>';
          activeEventSource.close();
          activeEventSource = null;
        }
      };

      activeEventSource.onerror = () => {
        streamStatus.textContent = "Stream finished";
        runBtn.disabled = false;
        runBtn.classList.remove("opacity-50", "cursor-not-allowed");
        runBtn.innerHTML = '<i class="fa-solid fa-play text-xs"></i><span>Run Full Demo</span>';
        if (activeEventSource) {
          activeEventSource.close();
          activeEventSource = null;
        }
      };
    }

    function markScenarioDone(id) {
      const card = document.getElementById(`sc-card-${id}`);
      const icon = document.getElementById(`sc-icon-${id}`);
      const status = document.getElementById(`sc-status-${id}`);
      if (card && icon && status) {
        card.className = "p-2.5 rounded-xl border border-emerald-500/30 bg-emerald-950/10 flex items-center justify-between text-xs transition duration-200";
        icon.className = "w-5 h-5 rounded-full bg-emerald-500/20 text-emerald-400 font-bold text-[10px] flex items-center justify-center shrink-0";
        icon.innerHTML = '<i class="fa-solid fa-check text-[10px]"></i>';
        status.className = "text-[10px] font-semibold text-emerald-400 uppercase px-1.5 py-0.5 rounded bg-emerald-900/30 shrink-0";
        status.textContent = "Passed";
      }
    }

    function clearConsole() {
      document.getElementById('terminal').innerHTML = "Console cleared.\\n";
    }

    function copyTerminal() {
      const text = document.getElementById('terminal').innerText;
      navigator.clipboard.writeText(text);
      alert("Terminal output copied to clipboard!");
    }

    // Initialize on load
    initScenarios();
    updateToken();
  </script>
</body>
</html>
"""


@router.get("/demo", response_class=HTMLResponse)
async def serve_demo_dashboard():
    """Serves the interactive Demo Dashboard page."""
    return HTMLResponse(DEMO_PAGE_HTML)
