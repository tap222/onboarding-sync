"""Instructor-style structured calls on Claude's native structured outputs.

Instructor's core idea: the model's output is a Pydantic model; if validation fails,
send the validation error back and ask again ("re-ask"), up to N times. Validators
can use runtime context (here: the source document text, the run's issue codes), so
they enforce *semantic* rules - "this citation must quote the document verbatim" -
not just types.

Why not the `instructor` package itself: its default Anthropic mode forces
tool_choice={"type":"tool"} (rejected with a 400 by claude-opus-5-5), and its
JSON_SCHEMA mode sends the deprecated `output_format` parameter and strips `tools`,
which the triage agent needs. This loop keeps the pattern on the current API:
output_config.format + client tools + validated re-ask.

Every call is tied to a PromptVersion (text + model + effort) and reported to an
optional `record` callback with the outcome, so behaviour can be traced to and
compared across prompt versions.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Awaitable, Callable, Optional, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from . import telemetry
from .budget import CostGuard, price
from .prompt_registry import PromptVersion, tools_sha
from .resilience import TransientError

log = logging.getLogger(__name__)
M = TypeVar("M", bound=BaseModel)
ToolExecutor = Callable[[str, dict], Awaitable[tuple[str, bool]]]  # (name, input) -> (result, is_error)
Recorder = Callable[[dict], None]


class StructuredCallError(Exception):
    pass


class EvaluationFailed(Exception):
    """The answer is schema-valid but the evaluator still rejects it after the
    allowed retries. Carries the last answer so the caller can decide (we hold it
    for a human rather than discard it)."""

    def __init__(self, parsed: BaseModel, problems: list[str]):
        super().__init__("; ".join(problems))
        self.parsed, self.problems = parsed, problems


async def structured_call(
    client: anthropic.AsyncAnthropic,
    *,
    prompt: PromptVersion,
    response_model: type[M],
    schema: dict,
    user: str,
    tools: Optional[list[dict]] = None,
    tool_executor: Optional[ToolExecutor] = None,
    validation_context: Optional[dict] = None,
    max_reasks: int = 2,
    max_turns: int = 10,
    transcript: Optional[list[dict]] = None,
    record: Optional[Recorder] = None,
    guard: Optional["CostGuard"] = None,
    run_id: Optional[str] = None,
    evaluate: Optional[Callable[[M], Awaitable[list[str]]]] = None,
    max_eval_retries: int = 1,
    purpose: str = "llm",
) -> M:
    """One span (hiresync.llm <purpose>) around the whole call: every turn, re-ask and
    evaluator retry. The Anthropic SDK's anthropic.messages.create spans nest inside."""
    with telemetry.tracer().start_as_current_span(
            f"hiresync.llm {purpose}", attributes={
                "hiresync.purpose": purpose, "hiresync.prompt": prompt.ref, "hiresync.prompt_sha": prompt.sha256[:16],
                "gen_ai.operation.name": "chat", "gen_ai.provider.name": "anthropic",
                "gen_ai.request.model": prompt.model, "hiresync.effort": prompt.effort, "hiresync.run_id": run_id},
            record_exception=False, set_status_on_exception=False) as sp:
        return await _structured_call(
            client, prompt=prompt, response_model=response_model, schema=schema, user=user, tools=tools,
            tool_executor=tool_executor, validation_context=validation_context, max_reasks=max_reasks,
            max_turns=max_turns, transcript=transcript, record=record, guard=guard, run_id=run_id,
            evaluate=evaluate, max_eval_retries=max_eval_retries, purpose=purpose, span=sp)


async def _structured_call(client, *, prompt, response_model, schema, user, tools, tool_executor,
                           validation_context, max_reasks, max_turns, transcript, record, guard, run_id,
                           evaluate, max_eval_retries, purpose, span):
    """guard: token/cost guardrails checked before every turn. evaluate: returns
    problems with a schema-valid answer; non-empty -> critique and retry."""
    started = time.monotonic()
    call = {**prompt.meta(), "tools_sha": tools_sha(tools) if tools else None,
            "input_sha": hashlib.sha256(user.encode()).hexdigest()[:16], "input": user,
            "reasks": 0, "turns": 0, "tool_calls": [], "usage": {"input_tokens": 0, "output_tokens": 0},
            "tokens": 0, "cost_usd": 0.0, "eval_retries": 0, "eval_problems": None}

    def finish(outcome: str, output: Any = None, error: Optional[str] = None) -> None:
        latency = int((time.monotonic() - started) * 1000)
        if record:
            record({**call, "outcome": outcome, "output": output, "error": error, "latency_ms": latency})
        # Aggregates under hiresync.* names: the SDK spans already carry gen_ai.usage.* per request.
        for k, v in {"hiresync.llm.input_tokens": call["usage"].get("input_tokens", 0),
                     "hiresync.llm.output_tokens": call["usage"].get("output_tokens", 0),
                     "hiresync.llm.cost_usd": round(call["cost_usd"], 6), "hiresync.llm.turns": call["turns"],
                     "hiresync.llm.reasks": call["reasks"], "hiresync.llm.eval_retries": call["eval_retries"],
                     "hiresync.llm.tool_calls": len(call["tool_calls"]), "hiresync.outcome": outcome}.items():
            span.set_attribute(k, v)
        dims = {"model": prompt.model, "purpose": purpose, "prompt": prompt.ref}
        telemetry.count("llm_tokens", call["usage"].get("input_tokens", 0), direction="input", **dims)
        telemetry.count("llm_tokens", call["usage"].get("output_tokens", 0), direction="output", **dims)
        telemetry.count("llm_cost", call["cost_usd"], **dims)
        telemetry.count("llm_calls", outcome=outcome, **{**dims, "error.type": error.split(":")[0] if error else None})
        telemetry.observe("llm_duration", latency, **dims)

    messages: list[Any] = [{"role": "user", "content": user}]
    try:
        for _turn in range(max_turns):
            call["turns"] += 1
            if guard is not None:
                await guard.preflight(prompt.model, {"system": prompt.text, "messages": messages, "tools": tools},
                                      run_id, {"tokens": call["tokens"], "usd": call["cost_usd"]})
            kwargs: dict[str, Any] = dict(
                model=prompt.model,
                max_tokens=guard.max_tokens(8000) if guard else 8000,
                system=prompt.text,
                messages=messages,
                output_config={"effort": prompt.effort, "format": {"type": "json_schema", "schema": schema}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            if tools:
                kwargs["tools"] = tools
            try:
                if guard is not None:
                    async with guard.slot:
                        response = await client.beta.messages.create(**kwargs)
                else:
                    response = await client.beta.messages.create(**kwargs)
            except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as exc:
                raise TransientError(str(exc)) from exc
            usage = getattr(response, "usage", None)
            turn = {k: getattr(usage, k, 0) or 0 for k in ("input_tokens", "output_tokens", "cache_read_input_tokens")}
            for k, v in turn.items():
                call["usage"][k] = call["usage"].get(k, 0) + v
            call["tokens"] += turn["input_tokens"] + turn["output_tokens"]
            call["cost_usd"] += price(prompt.model, turn)

            if response.stop_reason == "refusal":
                raise StructuredCallError(f"model declined: {getattr(response.stop_details, 'category', None)}")
            # Thinking blocks must go back unchanged, so append the full content list.
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                if tool_executor is None:
                    raise StructuredCallError("model called a tool but no executor was given")
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    out, is_error = await tool_executor(block.name, block.input)
                    entry = {"tool": block.name, "input": block.input, "is_error": is_error}
                    call["tool_calls"].append(entry)
                    if transcript is not None:
                        transcript.append(entry)
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": out,
                                    "is_error": is_error})
                messages.append({"role": "user", "content": results})  # all results in ONE message
                continue

            if response.stop_reason == "max_tokens":
                raise StructuredCallError("output truncated at max_tokens")

            text = next((b.text for b in response.content if b.type == "text"), "")
            try:
                parsed = response_model.model_validate(json.loads(text), context=validation_context or {})
            except (json.JSONDecodeError, ValidationError) as exc:
                if call["reasks"] >= max_reasks:
                    raise StructuredCallError(f"invalid after {call['reasks']} re-asks: {exc}") from exc
                call["reasks"] += 1
                log.info("re-asking (%d/%d): %s", call["reasks"], max_reasks, exc)
                messages.append({"role": "user", "content":
                                 "Your JSON failed validation. Fix exactly these problems and reply with the "
                                 f"corrected JSON only:\n{_errors(exc)}"})
                continue
            if evaluate is not None:
                problems = await evaluate(parsed)
                if problems:
                    call["eval_problems"] = problems
                    if call["eval_retries"] >= max_eval_retries:
                        finish("eval_failed", parsed.model_dump(mode="json"), "; ".join(problems))
                        raise EvaluationFailed(parsed, problems)
                    call["eval_retries"] += 1
                    messages.append({"role": "user", "content":
                                     "A reviewer checked your answer against this run and found problems. "
                                     "Revise and reply with the corrected JSON only:\n"
                                     + "\n".join(f"- {p}" for p in problems)})
                    continue
            finish("ok", parsed.model_dump(mode="json"))
            return parsed
        raise StructuredCallError(f"no final answer after {max_turns} turns")
    except EvaluationFailed as exc:
        telemetry.mark_error(span, exc, expected=True)  # handled: held for a human
        raise
    except Exception as exc:
        outcome = "budget_blocked" if type(exc).__name__ == "BudgetExceeded" else "error"
        finish(outcome, error=f"{type(exc).__name__}: {exc}")
        telemetry.mark_error(span, exc, expected=outcome == "budget_blocked")
        raise


def _errors(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "\n".join(f"- {'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
    return f"- {exc}"
