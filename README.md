# 🏠 microids — AI Device Coordinator

<p align="center">
  <strong>Tell your home what to do. It figures out the rest.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License"></a>
  <a href="https://microids.cloudcomps.net"><img src="https://img.shields.io/badge/Live_Demo-microids.cloudcomps.net-4ade80?style=for-the-badge" alt="Live Demo"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+"></a>
</p>

**microids** takes natural language goals, decomposes them into device tasks using an LLM, and executes them against real hardware through Home Assistant. It checks device state before sending commands, retries on failure, and shows animated results in real-time.

Think [OpenClaw](https://github.com/openclaw/openclaw) but for physical devices instead of chat channels.

[Live Simulator](https://microids.cloudcomps.net) · [Architecture](#how-it-works) · [Quick Start](#quick-start) · [Models](#models) · [Security](#security)

## Quick start

```bash
pip install -e .
cp .env.example .env          # add your GROQ_API_KEY
microids setup                 # interactive wizard
microids models set groq       # set default model (one time)
microids agent -m "Clean the house"
```

```
✓ task-1 → vacuum done
✓ task-2 → vacuum done (mop)
completed · 1.2s
```

## Simulator (no hardware needed)

The web simulator lets you try microids instantly — animated device cards respond in real-time as the LLM plans and executes.

```bash
MICROIDS_SIMULATOR=1 uvicorn microids.server:app --host 0.0.0.0 --port 8200
```

Open `http://localhost:8200`. Type "going to bed" and watch 5 devices animate.

## Models

microids is model-agnostic. Set your default once, use it everywhere. Swap anytime.

```bash
microids models list           # see available models
microids models set groq       # set default (persists to .env)
microids agent -m "Water the garden"   # uses default model
microids agent -m "Water the garden" --model bedrock  # one-off override
```

| Provider | Model | Speed | Cost |
|----------|-------|-------|------|
| `groq` | Llama 3.3 70B | ~280 t/s | Free/Dev tier |
| `groq` | Llama 4 Scout 17B | ~400 t/s | Free/Dev tier |
| `groq` | GPT-OSS 20B | ~300 t/s | Free/Dev tier |
| `bedrock` | Claude 3.5 Haiku | ~50 t/s | AWS pricing |
| `ollama` | Qwen3 8B (local) | ~15 t/s | Free (local) |

All models share the same prompt pipeline via `BaseAgent`. Few-shot ICL examples are built dynamically from your actual device fleet — no model-specific tuning needed.

### Web UI model switching

The simulator UI has a dropdown to hot-swap models without restart:

```
POST /models/switch {"model_id": "openai/gpt-oss-20b"}
GET  /models         # list available + current
```

## How it works

```
CLI / Web UI
     │
     ▼
┌─────────────────────────────────┐
│           Gateway               │
│       (control plane)           │
└──────────┬──────────────────────┘
           │
     ┌─────┼──────────┐
     ▼     ▼          ▼
  Planner  Executor   Recovery
  (LLM)   (DAG)      Engine
     │     │          │
     ▼     ▼          ▼
  BaseAgent  Router → Channel → Devices
  ├─ GroqAgent
  ├─ BedrockAgent
  ├─ OllamaAgent
  └─ MockAgent
```

1. **Planner** sends goal + device fleet + few-shot examples to the LLM
2. **LLM** returns a task plan (JSON with subtasks, dependencies, device IDs)
3. **Executor** runs tasks in topological order through the Recovery Engine
4. **Recovery Engine** retries with configurable backoff per device category
5. **Channel** checks device state before commands (won't vacuum if already cleaning)

## CLI commands

```bash
# Goals
microids agent -m "Leaving for work"        # execute a goal
microids agent -m "Goodnight" --model bedrock  # override model

# Interactive
microids tui                                 # terminal chat with /help, /devices, /model

# Models (OpenClaw-style set-once)
microids models list                         # show available, ● marks active
microids models set groq                     # persist to .env
microids models status --probe               # test connectivity

# Devices
microids devices list --channel ha           # list real HA devices
microids channels status --probe             # test channel connectivity

# Server
microids gateway run                         # start the server
microids gateway status                      # check if running

# Ops
microids setup                               # interactive wizard
microids doctor --fix                        # diagnose + auto-fix
microids status                              # system overview
microids logs -f                             # tail server logs
microids config get                          # show all config
microids config set GROQ_API_KEY sk-...      # set a value
```

## ROUTINES.md

Define routines in `ROUTINES.md`. The LLM reads this file to match goals to device actions:

```markdown
## Leaving for work
1. Open the garage door (device_id: cover.garage_door, capability: open)
2. Start the vacuum (device_id: vacuum.robot_vacuum, capability: start)
3. Water the front lawn (device_id: switch.front_sprinkler, capability: turn_on)
```

## State-aware commands

microids checks device state before sending commands:

- Won't send `return_to_base` if vacuum is already docked
- Won't send `open_cover` if garage is already open
- Skipped commands show as ⊘ in the UI

## Security

Built with [OpenClaw's security lessons](https://agenteer.com/blog/security-analysis-of-openclaw-and-the-ai-agent-era) from day 0:

- **Capability validation** on every command — devices can only do what they declare
- **Constant-time comparison** for secrets via `hmac.compare_digest()`
- **Canonical device IDs** — `channel:native_id` format prevents confusion
- **Untrusted input boundaries** in LLM prompts
- **Secret sanitization** before logging

## Deployment

```bash
# On your server (alongside Home Assistant)
pip install -e .
uvicorn microids.server:app --host 127.0.0.1 --port 8200
```

Put behind Nginx + Cloudflare Access for secure remote access with MFA.

## Project structure

```
microids/
├── microids/
│   ├── agents/          # LLM providers (base.py + groq/bedrock/ollama/mock)
│   ├── channels/        # Device channels (homeassistant.py, mock.py)
│   ├── core/            # Engine (gateway, executor, planner, recovery, registry, router, events, security)
│   ├── cli.py           # OpenClaw-style CLI
│   ├── models.py        # 22 data types, DAG validation
│   └── server.py        # FastAPI server + simulator UI
├── tests/               # 146 unit + integration tests
├── ROUTINES.md          # Your home routines
├── .env.example         # Config template
└── pyproject.toml
```

## Development

```bash
pip install -e ".[dev]"
pytest                         # 146 tests
```

## Research basis

The LLM prompt pipeline is informed by:

- **[SAGE](https://arxiv.org/abs/2311.00772)** (CMU 2023) — two-stage device planning with tool disambiguation
- **[Sasha](https://arxiv.org/abs/2305.09802)** (2023) — few-shot examples for under-specified smart home commands
- **[home-llm](https://github.com/acon96/home-llm)** — 98% accuracy with domain-qualified capabilities + fine-tuned models
- **[2026 Prompting Benchmark](https://markaicode.com/chain-of-thought-few-shot-self-consistency-prompting-benchmark/)** — Few-Shot CoT is the best single-pass technique for structured output

## License

MIT
