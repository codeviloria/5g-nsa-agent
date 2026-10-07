"""
Pre-flight check: validates everything the agent needs before you run it.

    uv run python check_env.py

Checks, in order: Python version, installed packages, .env variables,
Ollama reachable, model downloaded, model supports tools, LangSmith (optional).
Exit code 0 = ready, 1 = something blocking is missing.
"""
from __future__ import annotations

import importlib.metadata as md
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

OK, WARN, FAIL = "\033[32m✔\033[0m", "\033[33m!\033[0m", "\033[31m✘\033[0m"
errors = 0


def report(status: str, msg: str, hint: str = "") -> None:
    global errors
    if status == FAIL:
        errors += 1
    print(f" {status} {msg}" + (f"\n     → {hint}" if hint else ""))


def load_dotenv(path: Path) -> None:
    """Minimal .env loader (stdlib only), without overwriting existing variables."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.split(" #")[0].strip().strip("'\""))


def http_json(url: str, body: dict | None = None, timeout: int = 10) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main() -> int:
    root = Path(__file__).parent
    print("\n5G NSA agent: pre-flight check\n")

    # 1. Python
    v = sys.version_info
    report(OK if v >= (3, 10) else FAIL, f"Python {v.major}.{v.minor}.{v.micro}",
           "" if v >= (3, 10) else "Python >= 3.10 is required (uv python install 3.12)")

    # 2. Packages (only needed by agent_langchain.py; agent_raw.py is stdlib-only)
    for pkg in ("langchain", "langchain-ollama", "langchain-core", "python-dotenv", "langsmith"):
        try:
            report(OK, f"{pkg} {md.version(pkg)}")
        except md.PackageNotFoundError:
            report(FAIL, f"{pkg} not installed", "uv sync")

    # 3. .env
    env_file = root / ".env"
    load_dotenv(env_file)
    report(OK if env_file.exists() else WARN, ".env found" if env_file.exists() else ".env not found",
           "" if env_file.exists() else "cp .env.example .env and edit it")
    host = os.getenv("OLLAMA_HOST", "").rstrip("/")
    model = os.getenv("MODEL", "")
    report(OK if host else FAIL, f"OLLAMA_HOST = {host or '(empty)'}",
           "" if host else "set OLLAMA_HOST=http://192.168.1.7:11434 in .env")
    report(OK if model else FAIL, f"MODEL = {model or '(empty)'}",
           "" if model else "set MODEL=qwen3:8b (or gemma4:e2b) in .env")
    if not host or not model:
        return finish()

    # 4. Ollama reachable
    try:
        tags = http_json(f"{host}/api/tags")
        version = http_json(f"{host}/api/version").get("version", "?")
        report(OK, f"Ollama reachable at {host} (v{version})")
    except (urllib.error.URLError, OSError) as exc:
        report(FAIL, f"Ollama not reachable at {host}: {exc}",
               "server on? OLLAMA_HOST=0.0.0.0 on the server? firewall port 11434?")
        return finish()

    # 5. Model downloaded
    names = {m["name"] for m in tags.get("models", [])}
    wanted = model if ":" in model else f"{model}:latest"
    if wanted not in names:
        report(FAIL, f"model '{model}' not downloaded on the server",
               f"on the server: ollama pull {model}   (available: {', '.join(sorted(names))})")
        return finish()
    report(OK, f"model '{model}' downloaded")

    # 6. Tools capability
    try:
        caps = http_json(f"{host}/api/show", {"model": model}).get("capabilities")
    except (urllib.error.URLError, OSError):
        caps = None
    if caps is None:
        report(WARN, "could not read model capabilities (old Ollama?)",
               f"check manually: ollama show {model}  → Capabilities must list 'tools'")
    elif "tools" in caps:
        report(OK, f"'{model}' supports tools  {caps}")
        if "thinking" in caps and os.getenv("THINK", "false").lower() == "true":
            report(WARN, "THINK=true: the model will reason before every tool call (slow on CPU)",
                   "set THINK=false in .env unless you want to see its reasoning")
    else:
        report(FAIL, f"'{model}' does NOT support tools  {caps}",
               "use qwen3:8b, qwen3:4b or gemma4:e2b (gemma3 has no tool calling)")

    # 7. LangSmith (optional)
    key = os.getenv("LANGSMITH_API_KEY", "")
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    if not tracing:
        report(WARN, "LangSmith tracing off (optional)")
    elif key.startswith("lsv2_") and len(key) == 51:
        report(OK, "LangSmith key format looks right (lsv2_…, 51 chars)")
    else:
        report(FAIL, f"LANGSMITH_TRACING=true but key looks wrong (len {len(key)})",
               "fix LANGSMITH_API_KEY or set LANGSMITH_TRACING=false")
    return finish()


def finish() -> int:
    print("\n" + ("Ready: uv run python agent_raw.py" if errors == 0
                  else f"{errors} blocking problem(s): fix them and run again.") + "\n")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
