"""
The same 5G NSA agent with LangChain v1 `create_agent`: the framework builds the schemas,
runs the tool loop and (optionally) traces every step to LangSmith.

Usage:
    uv run python agent_langchain.py "Anchor BAQ_034_A has 7.8% SCG failures. Diagnose."
"""
from __future__ import annotations

import os
import re
import sys

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

from nsa_tools import TOOLS
from agent_raw import SYSTEM_PROMPT   # also loads .env

MODEL = os.getenv("MODEL", "qwen3:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# parse_docstring=True reads the "Args:" section into per-argument descriptions
lc_tools = [tool(fn, parse_docstring=True) for fn in TOOLS]
TOOL_NAMES = [fn.__name__ for fn in TOOLS]


def build_agent(model=None):
    model = model or ChatOllama(model=MODEL, base_url=OLLAMA_HOST, temperature=0,
                                reasoning=False, num_ctx=8192)   # no hidden thinking
    return create_agent(model=model, tools=lc_tools, system_prompt=SYSTEM_PROMPT)


if __name__ == "__main__":
    import time
    q = " ".join(sys.argv[1:]) or ("Anchor BAQ_021_A: EN-DC time ratio is only 2% although n78 coverage "
                                   "looks good in drive tests. Diagnose and propose actions.")
    agent = build_agent()
    t0, answered, tool_calls = time.time(), False, 0
    for step in agent.stream({"messages": [{"role": "user", "content": q}]}, stream_mode="updates"):
        for node, update in step.items():
            for m in (update or {}).get("messages", []):
                for tc in getattr(m, "tool_calls", []) or []:
                    tool_calls += 1
                    print(f"  -> {tc['name']}({tc['args']})")
                if node != "model":
                    continue
                # Silent failures: make them visible instead of ending with no output
                for bad in getattr(m, "invalid_tool_calls", []) or []:
                    print(f"  !! invalid tool call: {bad.get('name')} args={bad.get('args')!r} "
                          f"error={bad.get('error')}")
                if not getattr(m, "tool_calls", None):
                    if m.content:
                        print("\n" + m.content)
                        leaked = [n for n in TOOL_NAMES if re.search(rf"{n}\s*[({{]", m.content)]
                        if leaked:   # the model "called" a tool in plain text: the parser missed it
                            print(f"  !! tool call written as text, not executed: {leaked} "
                                  f"(model/parser issue; try another model or update Ollama)")
                        else:
                            answered = True
                    else:
                        meta = getattr(m, "response_metadata", {}) or {}
                        print(f"  !! model returned an empty final message "
                              f"(done_reason={meta.get('done_reason')}, "
                              f"eval_count={meta.get('eval_count')})")
    print(f"\n[{MODEL} | {tool_calls} tool calls | {time.time() - t0:.0f} s"
          f"{'' if answered else ' | NO FINAL ANSWER: check the LangSmith trace'}]")
