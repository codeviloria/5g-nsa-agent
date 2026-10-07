"""
5G NSA troubleshooting agent, "under the hood": a zero-dependency tool-calling loop against Ollama.

Only the Python standard library (LangSmith tracing is optional). Shows what frameworks
like LangChain do for you:
  1. turn Python functions into JSON schemas,
  2. send them with the conversation,
  3. execute whatever tool_calls the model returns,
  4. feed results back as `tool` messages until the model answers in plain text.

Usage:
    uv run python check_env.py      # first time: validates .env, Ollama, model and tools
    uv run python agent_raw.py "EN-DC ratio on BAQ_021_A is very low. Why?"
"""
from __future__ import annotations

import inspect
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

from check_env import load_dotenv
from nsa_tools import TOOLS

try:  # optional: LangSmith tracing without LangChain. Without the package, it's a no-op.
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        return args[0] if args and callable(args[0]) else (lambda fn: fn)

load_dotenv(Path(__file__).parent / ".env")   # stdlib-only .env loader
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.getenv("MODEL", "qwen3:8b")
THINK = os.getenv("THINK", "false").lower() == "true"   # gemma4/qwen3 think by default: slow on CPU
MAX_STEPS = 6

SYSTEM_PROMPT = """You are a 5G NSA (EN-DC) troubleshooting assistant for an operator's RAN team.

Rules:
- Use the tools to get facts. Never invent KPI or parameter values.
- Run the tools yourself. Never ask the user to run a tool, and never describe or simulate
  a tool call in text: call it.
- If a tool returns an error, fix the arguments and call it again.
- Before answering, always complete this checklist:
  1. get_endc_kpis for the anchor (it returns the anchor's NR neighbors).
  2. audit_endc_config for the anchor.
  3. For EVERY NR neighbor: check_nr_pci_conflicts and ssb_beam_report.
- Report every finding, not only the first one. Several causes can add up.
- Only recommend a change that a tool finding supports, and cite that tool as evidence.
  If a check passed (for example a B1 threshold with no finding, or healthy SSB beams),
  list it under "Checks OK" and do not propose a change for it.
- Tool arguments and assumptions (tx_power_dbm, ssb_beam_gain_dbi, distance_m) are not
  network parameters. Never propose them as changes.
- For each action give: parameter, current value, proposed value, why, evidence (tool).
- You only recommend changes; a human approves them.
- Be concise. Answer in the same language as the user."""

PY_TO_JSON = {float: "number", int: "integer", str: "string", bool: "boolean"}


def function_to_schema(fn) -> dict:
    """Build an Ollama/OpenAI-style tool schema from type hints + Google-style docstring."""
    doc = inspect.getdoc(fn) or ""
    description = doc.split("\n\n")[0].replace("\n", " ")
    arg_docs = dict(re.findall(r"^\s{0,8}(\w+): (.+)$", doc.split("Args:")[-1], re.M)) if "Args:" in doc else {}
    hints = {k: eval(v) if isinstance(v, str) else v for k, v in fn.__annotations__.items()}

    props, required = {}, []
    for name, param in inspect.signature(fn).parameters.items():
        props[name] = {"type": PY_TO_JSON.get(hints.get(name), "string"),
                       "description": arg_docs.get(name, "")}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "function",
            "function": {"name": fn.__name__, "description": description,
                         "parameters": {"type": "object", "properties": props, "required": required}}}


@traceable(run_type="llm", name="ollama_chat")
def chat(messages: list[dict], tools: list[dict]) -> dict:
    """One call to Ollama's /api/chat (non-streaming)."""
    body = json.dumps({"model": MODEL, "messages": messages, "tools": tools, "think": THINK,
                       "stream": False, "options": {"temperature": 0, "num_ctx": 8192}}).encode()
    req = urllib.request.Request(f"{OLLAMA_HOST}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["message"]


REGISTRY = {fn.__name__: fn for fn in TOOLS}


@traceable(run_type="tool", name="tool_call")
def run_tool(name: str, args: dict) -> dict:
    """Execute one tool call; errors go back to the model as data."""
    try:
        return REGISTRY[name](**args)
    except Exception as exc:        # bad tool name or arguments -> tell the model
        return {"error": f"{type(exc).__name__}: {exc}"}


@traceable(run_type="chain", name="5g-nsa-agent (raw)")
def run_agent(question: str, verbose: bool = True) -> str:
    schemas = [function_to_schema(fn) for fn in TOOLS]
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]

    for _ in range(MAX_STEPS):
        msg = chat(messages, schemas)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:                       # plain answer -> done
            return msg.get("content", "")
        for call in calls:                  # model asked for one or more tools
            name = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args or "{}")
            result = run_tool(name, args)
            if verbose:
                print(f"  -> {name}({args})")
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
    return "Stopped: too many tool steps."


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or ("Anchor BAQ_021_A: EN-DC time ratio is only 2% although n78 coverage "
                                   "looks good in drive tests. Diagnose and propose actions.")
    print(f"Model: {MODEL} @ {OLLAMA_HOST}\nQ: {q}\n")
    print("\n" + run_agent(q))
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        try:                              # make sure traces are sent before the script exits
            from langsmith.run_trees import get_cached_client
            get_cached_client().flush()
        except Exception:
            pass
