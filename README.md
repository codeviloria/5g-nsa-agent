# 5G NSA Troubleshooting Agent (PoC)

[![tests](https://github.com/codeviloria/5g-nsa-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/codeviloria/5g-nsa-agent/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![uv](https://img.shields.io/badge/deps-uv-6e40c9)
![LangChain](https://img.shields.io/badge/LangChain-create__agent-1c3c3c)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20(local)-black)
![license](https://img.shields.io/badge/license-MIT-green)

A proof of concept of a **tool-calling AI agent for RAN operations**. It diagnoses **5G NSA (EN-DC, option 3x)** problems on LTE anchors and n78 cells, runs **100 % locally** on [Ollama](https://ollama.com), and is **traced end to end in LangSmith**.

> Write-up: *Teaching a Local LLM to Troubleshoot 5G NSA: What the Traces Taught Me* (TODO: link to the blog post)

> **All network data is fictional.** No operator data is used. The propagation model is real but simplified.

---

## What it does

Ask a question like *"Users at site BAQ_021 barely see the 5G icon. What would you check first?"*. The agent:

1. reads the anchor's EN-DC KPIs,
2. audits the anchor's NR measurement config against the real gNB SSB config,
3. checks NR PCI conflicts and SSB beam quality for every NR neighbor,
4. returns root causes and actions, **each one citing the tool finding behind it**. A human approves any change.

```
question ─► create_agent (LangChain) ─► ChatOllama (qwen3:8b / gemma4:e2b)
                ▲                             │ tool_calls
                └──── tool results ◄──── nsa_tools.py (Python, mock OSS)
                         every step traced in LangSmith
```

## Tools

| Tool | Type | What it does |
|---|---|---|
| `nr_link_budget` | Physics | 3GPP TR 38.901 UMa path loss → n78 SS-RSRP and margin vs B1 |
| `get_endc_kpis` | Mock OSS (PM) | B1 reports, SgNB addition, SCG failures, EN-DC time ratio |
| `audit_endc_config` | Mock OSS (CM) + rules | Anchor `measObjectNR` vs gNB SSB: NR-ARFCN, SCS, **SMTC vs SSB burst timing**, B1, X2 |
| `check_nr_pci_conflicts` | Mock OSS + rules | NR PCI collision, mod-3 (PSS), mod-30 (UL DMRS/SRS) |
| `ssb_beam_report` | Mock DT / L3 MR | Per-SSB-beam SS-RSRP, SS-SINR and sample share (L_max = 8) |

Beam-level data is **not** in standard per-cell PM counters. In a real network it comes from drive-test scanners that decode the SSB index, or from L3 measurement reports with per-SSB results (`rsIndexResults`) collected via MR/MDT or vendor traces.

## Scenarios

| Anchor | Symptom | Root causes (ground truth) |
|---|---|---|
| `BAQ_021_A` | EN-DC time ratio 2 %, 3 B1 reports per 1,000 UEs, good n78 coverage | SMTC window (0–5 ms) misses the SSB burst (10 ms); NR PCI mod-3 with `BAQ_034_N78_A` |
| `BAQ_034_A` | 7.8 % SCG failures | B1 at −124 dBm (too permissive); NR PCI mod-3; weak SSB beams 6–7 (18 % of samples) |

## Results

Same questions, same tools, four iterations driven by LangSmith traces. Full logs in [`runs/`](runs/).

| Run | Change | Model | Outcome |
|---|---|---|---|
| 1 | Basic prompt | gemma4:e2b | Stopped at the first finding; asked the user to run a tool |
| 2 | Checklist in the system prompt | gemma4:e2b | SMTC case 2/2 causes; other cases ended silently or *simulated* tool calls in text |
| 3 | Actionable tool errors, ID resolution, runner that surfaces silent failures | gemma4:e2b → **qwen3:8b** | gemma4:e2b could not hold a 4-tool loop on Ollama; qwen3:8b found **all** real causes but padded answers with unsupported actions |
| 4 | Evidence required for every action | **qwen3:8b** | **2/2 and 2/2 causes, 0 unsupported actions** |

Latency on CPU (qwen3:8b): tools take milliseconds; **61–77 % of each ~225 s run is the final answer being written**. Remaining issues (misquoted numbers, wrong cell name) are the target of the next step: automated evaluation.

## Quick start

Requirements: Python ≥ 3.10, [uv](https://docs.astral.sh/uv/), an Ollama server with a model that supports **tool calling**.

```bash
git clone https://github.com/codeviloria/5g-nsa-agent.git
cd 5g-nsa-agent
uv sync                                   # installs locked dependencies (uv.lock)
cp .env.example .env                      # set OLLAMA_HOST and MODEL
uv run python check_env.py                # pre-flight: packages, .env, Ollama, model, tools capability

uv run python agent_langchain.py          # default question (BAQ_021_A)
uv run python agent_langchain.py "Anchor BAQ_034_A has 7.8% SCG failures. Diagnose and propose actions."
uv run python test_agents.py              # offline tests, no Ollama needed
```

| Model | Tool calling | Notes |
|---|---|---|
| `qwen3:8b` | ✅ | **Recommended.** Holds the 4-tool loop (~3–4 min per diagnosis on CPU) |
| `gemma4:e2b` | ✅ | Fast, but did not hold the loop on Ollama in these tests |
| `gemma3:*` | ❌ | No tool calling (`does not support tools (400)`) |

Check a model: `ollama show <model>` → *Capabilities* must list `tools`. Thinking is disabled by default (`THINK=false`), because on CPU it multiplies latency.

## Tracing (LangSmith)

Set in `.env`:
```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=5g-nsa-agent
```
`agent_langchain.py` is traced automatically (`model → tools → model` graph). `agent_raw.py` is traced with `@traceable`, without LangChain.

## Project layout

```
nsa_tools.py         5 tools as plain Python functions + the fictional network
agent_langchain.py   LangChain create_agent + ChatOllama (main agent)
agent_raw.py         the same loop in plain Python (stdlib + optional @traceable); shared SYSTEM_PROMPT
check_env.py         pre-flight checks
test_agents.py       offline tests with a scripted fake model
runs/                run logs: what failed, why, and what changed
```

## Design rules

- **Tools own the numbers; the model chooses and explains.**
- **Errors are data the model can act on** ("call the tool again with an exact anchor ID from …").
- **Every recommended action cites its evidence.** Passed checks are reported as OK, not turned into actions.
- **Read-only.** The agent recommends; a human approves. Writing to a live network would need approvals, audit logs and rollback.

## Roadmap

- [ ] Evaluation dataset in LangSmith (anchors with known root causes) and automated graders: numbers in the answer vs tool outputs, cell/parameter consistency, LLM-as-judge
- [ ] Shorter answer format to cut final-answer latency
- [ ] Replace mocks with CM/PM exports (anonymized)

## License

MIT. See [LICENSE](LICENSE).
