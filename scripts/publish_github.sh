#!/usr/bin/env bash
# Publishes this folder as a public GitHub repo, in one go.
#   bash scripts/publish_github.sh            # repo name: 5g-nsa-agent
#   bash scripts/publish_github.sh my-name    # custom repo name
# Requires: git, gh (GitHub CLI) logged in (gh auth login).
set -euo pipefail

REPO="${1:-5g-nsa-agent}"
DESC="PoC: 5G NSA (EN-DC) troubleshooting agent with tool calling on local LLMs (Ollama), traced in LangSmith"
TOPICS="5g,ran,telecom,llm,agents,langchain,ollama,langsmith,tool-calling"
cd "$(dirname "$0")/.."

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*"; exit 1; }

say "Checking tools"
command -v git >/dev/null || die "git not installed: sudo apt install -y git"
command -v gh  >/dev/null || die "gh not installed: sudo apt install -y gh"
gh auth status >/dev/null 2>&1 || die "gh not logged in: run 'gh auth login'"
USER_GH="$(gh api user -q .login)"
echo "GitHub user: $USER_GH"

say "Running offline tests"
LANGSMITH_TRACING=false uv run python test_agents.py

say "Badges and links: TU-USUARIO -> $USER_GH"
sed -i "s/TU-USUARIO/$USER_GH/g" README.md

say "CI workflow (.github/workflows/tests.yml)"
mkdir -p .github/workflows
cat > .github/workflows/tests.yml <<'EOF'
name: tests
on: [push, pull_request]
jobs:
  offline-tests:
    runs-on: ubuntu-latest
    env:
      LANGSMITH_TRACING: "false"
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --locked
      - run: uv run python test_agents.py
EOF

say "Git commit"
[ -d .git ] || git init -b main
git add .
if git diff --cached --name-only | grep -qx ".env"; then
  git reset -q .env
  die ".env was staged. Check .gitignore; nothing was pushed."
fi
if git diff --cached | grep -Eq "lsv2_pt_[A-Za-z0-9]{20,}|tvly-[A-Za-z0-9]{20,}"; then
  die "A real API key appears in the staged files. Remove it; nothing was pushed."
fi
git diff --cached --quiet || git commit -q -m "PoC: 5G NSA troubleshooting agent with tool calling on local LLMs"
git log --oneline -1

say "Create GitHub repo and push"
if gh repo view "$USER_GH/$REPO" >/dev/null 2>&1; then
  git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://github.com/$USER_GH/$REPO.git"
  git push -u origin main
else
  gh repo create "$REPO" --public --source=. --push --description "$DESC"
fi
gh repo edit "$USER_GH/$REPO" --add-topic "$TOPICS" >/dev/null

say "Done: https://github.com/$USER_GH/$REPO"
echo "GitHub Actions will run the tests in a few minutes (badge in the README)."
