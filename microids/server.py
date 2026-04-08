"""microids API server — runs on Hetzner alongside Home Assistant.

All device commands go localhost → HA, bypassing corporate proxy.
LLM calls go direct to Groq API from the server.

Supports two modes:
- Production: real HA channel + LLM agent (default)
- Simulator: mock channel + mock agent with animated device UI

Usage:
    uvicorn microids.server:app --host 127.0.0.1 --port 8200
    MICROIDS_SIMULATOR=1 uvicorn microids.server:app --host 127.0.0.1 --port 8200
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Gateway components (initialized on startup)
_gateway = None
_registry = None
_event_bus = None
_mock_channel = None  # Only set in simulator mode
_ws_clients: set[WebSocket] = set()


class GoalRequest(BaseModel):
    goal: str
    agent: str = "groq"

# Max goal length to prevent token abuse
_MAX_GOAL_LENGTH = 500


class GoalResponse(BaseModel):
    goal: str
    status: str
    duration_seconds: float
    results: list[dict[str, Any]]
    error: str | None = None


class DeviceInfo(BaseModel):
    id: str
    device_type: str
    category: str
    zone: str
    capabilities: list[str]
    status: str


def _load_env() -> None:
    """Load .env from server working directory."""
    env_file = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k, v)


def _is_simulator() -> bool:
    """Check if we should run in simulator mode."""
    return os.environ.get("MICROIDS_SIMULATOR", "").lower() in ("1", "true", "yes")


def _build_gateway(agent_type: str = "groq"):
    from microids.core.events import EventBus
    from microids.core.gateway import Gateway
    from microids.core.recovery import RecoveryEngine
    from microids.core.registry import Registry
    from microids.core.router import Router

    if agent_type == "groq":
        from microids.agents.groq import GroqAgent
        agent = GroqAgent()
    elif agent_type == "bedrock":
        from microids.agents.bedrock import BedrockAgent
        agent = BedrockAgent()
    elif agent_type == "ollama":
        from microids.agents.ollama import OllamaAgent
        agent = OllamaAgent()
    else:
        from microids.agents.mock import MockAgent
        agent = MockAgent()

    # HA channel talks to localhost — no Cloudflare headers needed on server
    from microids.channels.homeassistant import HomeAssistantChannel
    channel = HomeAssistantChannel()
    registry = Registry()
    event_bus = EventBus()
    router = Router(default_channel=channel)
    recovery_engine = RecoveryEngine(
        registry=registry, router=router, event_bus=event_bus,
    )
    gateway = Gateway(
        agent=agent, router=router, registry=registry,
        event_bus=event_bus, recovery_engine=recovery_engine,
    )
    return gateway, registry, event_bus, None


def _build_simulator_gateway():
    """Build gateway with Groq LLM + mock channel for simulator mode.

    Uses the real LLM for natural language understanding but executes
    commands against simulated devices with animated state machines.
    Falls back to MockAgent if GROQ_API_KEY is not set.
    """
    from microids.channels.mock import MockChannel
    from microids.core.events import EventBus
    from microids.core.gateway import Gateway
    from microids.core.recovery import RecoveryEngine
    from microids.core.registry import Registry
    from microids.core.router import Router

    # Use real LLM if available, mock agent as fallback
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        from microids.agents.groq import GroqAgent
        agent = GroqAgent()
    else:
        from microids.agents.mock import MockAgent
        agent = MockAgent()

    channel = MockChannel(simulator=True)
    registry = Registry()
    event_bus = EventBus()
    router = Router(default_channel=channel)
    recovery_engine = RecoveryEngine(
        registry=registry, router=router, event_bus=event_bus,
    )
    gateway = Gateway(
        agent=agent, router=router, registry=registry,
        event_bus=event_bus, recovery_engine=recovery_engine,
    )
    return gateway, registry, event_bus, channel


async def _broadcast_state(device_key: str, state: str, state_info: dict) -> None:
    """Broadcast device state change to all WebSocket clients."""
    global _ws_clients
    msg = json.dumps({
        "type": "state_change",
        "device": device_key,
        "state": state,
        "label": state_info.get("label", state),
        "color": state_info.get("color", "#6b7280"),
        "animated": state_info.get("animated", False),
        "transient": state_info.get("transient", False),
    })
    dead: set[WebSocket] = set()
    for ws in _ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    _ws_clients -= dead


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _gateway, _registry, _event_bus, _mock_channel
    _load_env()

    if _is_simulator():
        _gateway, _registry, _event_bus, _mock_channel = _build_simulator_gateway()
        # Wire up WebSocket broadcast for state changes
        _mock_channel.on_state_change(
            lambda dk, s, si: asyncio.ensure_future(_broadcast_state(dk, s, si))
        )
    else:
        _gateway, _registry, _event_bus, _mock_channel = _build_gateway()

    await _gateway.start()
    yield
    await _gateway.stop()


app = FastAPI(title="microids", version="0.1.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def web_ui():
    if _mock_channel is not None:
        return _SIMULATOR_HTML
    return _PRODUCTION_HTML


@app.get("/health")
async def health():
    devices = await _registry.list_all() if _registry else []
    mode = "simulator" if _mock_channel else "production"
    agent = "unknown"
    if _gateway and _gateway._agent:
        agent = _gateway._agent.model_name() if hasattr(_gateway._agent, 'model_name') else type(_gateway._agent).__name__
    return {"status": "ok", "devices": len(devices), "mode": mode, "agent": agent}


# Available Groq models for the simulator model switcher
_GROQ_MODELS = [
    {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "provider": "groq"},
    {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B", "provider": "groq"},
    {"id": "qwen/qwen3-32b", "name": "Qwen3 32B", "provider": "groq"},
    {"id": "meta-llama/llama-4-scout-17b-16e-instruct", "name": "Llama 4 Scout 17B", "provider": "groq"},
    {"id": "openai/gpt-oss-20b", "name": "GPT-OSS 20B", "provider": "groq"},
]


@app.get("/models")
async def list_models():
    """List available models and the currently active one."""
    current = "unknown"
    if _gateway and _gateway._agent:
        current = _gateway._agent.model_name() if hasattr(_gateway._agent, 'model_name') else ""
    return {"models": _GROQ_MODELS, "current": current}


class ModelSwitch(BaseModel):
    model_id: str


@app.post("/models/switch")
async def switch_model(req: ModelSwitch):
    """Hot-swap the LLM model without restarting the server."""
    if not _gateway:
        raise HTTPException(503, "Gateway not initialized")

    from microids.agents.groq import GroqAgent
    new_agent = GroqAgent(model=req.model_id)
    _gateway._agent = new_agent
    # Rewire the planner with the new agent
    from microids.core.planner import Planner
    _gateway._planner = Planner(new_agent)
    return {"status": "ok", "agent": new_agent.model_name()}


@app.post("/goal", response_model=GoalResponse)
async def execute_goal(req: GoalRequest):
    if not _gateway:
        raise HTTPException(503, "Gateway not initialized")

    # Input validation
    goal = req.goal.strip()
    if not goal:
        raise HTTPException(400, "Goal cannot be empty")
    if len(goal) > _MAX_GOAL_LENGTH:
        raise HTTPException(400, f"Goal too long ({len(goal)} chars, max {_MAX_GOAL_LENGTH})")

    # In simulator mode, always use the pre-configured gateway (Groq + mock channel)
    if _mock_channel:
        gateway = _gateway
    else:
        gateway = _gateway
        if req.agent not in ("groq", "mock"):
            gw, reg, eb, _ = _build_gateway(req.agent)
            await gw.start()
            gateway = gw

    try:
        result = await gateway.goal(req.goal)

        task_results = []
        if result.execution_report:
            for tr in result.execution_report.results:
                skipped = False
                skip_reason = ""
                if tr.response and isinstance(tr.response, dict):
                    if tr.response.get("status") == "skipped":
                        skipped = True
                        skip_reason = tr.response.get("reason", "already in state")

                task_results.append({
                    "subtask_id": tr.subtask_id,
                    "device_id": tr.device_id or "",
                    "success": tr.success,
                    "error": tr.error,
                    "skipped": skipped,
                    "skip_reason": skip_reason,
                })

        return GoalResponse(
            goal=result.goal,
            status=result.status.value,
            duration_seconds=result.duration_seconds,
            results=task_results,
            error=result.error,
        )
    finally:
        if not _mock_channel and gateway is not _gateway:
            await gateway.stop()


@app.get("/devices", response_model=list[DeviceInfo])
async def list_devices():
    if not _registry:
        raise HTTPException(503, "Registry not initialized")

    devices = await _registry.list_all()
    return [
        DeviceInfo(
            id=d.id,
            device_type=d.spec.device_type,
            category=d.spec.category.value,
            zone=d.spec.context.zone,
            capabilities=[c.name for c in d.spec.capabilities],
            status=d.status.value,
        )
        for d in devices
    ]


@app.get("/simulator/state")
async def simulator_state():
    """Get current simulator device states (simulator mode only)."""
    if not _mock_channel:
        raise HTTPException(404, "Not in simulator mode")
    return _mock_channel.get_simulator_state()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket for real-time device state updates."""
    await ws.accept()
    _ws_clients.add(ws)
    try:
        # Send initial state
        if _mock_channel:
            await ws.send_text(json.dumps({
                "type": "full_state",
                "devices": _mock_channel.get_simulator_state(),
            }))
        # Keep alive — listen for pings
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


# ---------------------------------------------------------------------------
# Production UI (original — for real HA devices)
# ---------------------------------------------------------------------------
_PRODUCTION_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>microids</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
.header { padding: 16px 24px; border-bottom: 1px solid #2a2a4a; display: flex; align-items: center; gap: 12px; }
.header h1 { font-size: 18px; font-weight: 600; color: #fff; }
.header .dot { width: 8px; height: 8px; border-radius: 50%; background: #4ade80; }
.header .status { font-size: 12px; color: #888; }
.messages { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
.msg { max-width: 720px; width: 100%; margin: 0 auto; }
.msg .bubble { padding: 12px 16px; border-radius: 12px; line-height: 1.5; font-size: 14px; }
.msg.user .bubble { background: #2a2a4a; color: #fff; }
.msg.system .bubble { background: #16213e; color: #e0e0e0; border: 1px solid #2a2a4a; }
.msg .label { font-size: 11px; color: #666; margin-bottom: 4px; padding-left: 4px; }
.result-row { display: flex; align-items: center; gap: 8px; padding: 4px 0; font-size: 13px; font-family: monospace; }
.result-row .icon { font-size: 14px; }
.result-row.ok .icon { color: #4ade80; }
.result-row.fail .icon { color: #f87171; }
.result-row.skip .icon { color: #facc15; }
.result-meta { font-size: 12px; color: #888; margin-top: 8px; }
.input-area { padding: 16px 24px; border-top: 1px solid #2a2a4a; }
.input-wrap { max-width: 720px; margin: 0 auto; display: flex; gap: 8px; }
.input-wrap input { flex: 1; background: #16213e; border: 1px solid #2a2a4a; border-radius: 24px; padding: 12px 20px; color: #fff; font-size: 14px; outline: none; }
.input-wrap input:focus { border-color: #4a6fa5; }
.input-wrap input::placeholder { color: #555; }
.input-wrap button { background: #4a6fa5; border: none; border-radius: 24px; padding: 12px 24px; color: #fff; font-size: 14px; cursor: pointer; font-weight: 500; }
.input-wrap button:hover { background: #5a7fb5; }
.input-wrap button:disabled { background: #333; cursor: not-allowed; }
.devices-bar { padding: 8px 24px; display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }
.device-chip { font-size: 11px; padding: 4px 10px; border-radius: 12px; background: #16213e; border: 1px solid #2a2a4a; color: #aaa; }
.spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid #444; border-top-color: #4a6fa5; border-radius: 50%; animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="header">
  <div class="dot" id="statusDot"></div>
  <h1>microids</h1>
  <span class="status" id="statusText">connecting...</span>
</div>
<div class="devices-bar" id="devicesBar"></div>
<div class="messages" id="messages"></div>
<div class="input-area">
  <div class="input-wrap">
    <input type="text" id="goalInput" placeholder="What do you want to do?" autocomplete="off" />
    <button id="sendBtn" onclick="sendGoal()">Send</button>
  </div>
</div>
<script>
const msgs = document.getElementById('messages');
const input = document.getElementById('goalInput');
const btn = document.getElementById('sendBtn');
input.addEventListener('keydown', e => { if (e.key === 'Enter' && !btn.disabled) sendGoal(); });
async function init() {
  try {
    const r = await fetch('/health');
    const d = await r.json();
    document.getElementById('statusDot').style.background = '#4ade80';
    document.getElementById('statusText').textContent = d.devices + ' devices connected';
    const dr = await fetch('/devices');
    const devices = await dr.json();
    const bar = document.getElementById('devicesBar');
    const icons = {vacuum:'🤖', cover:'🚪', switch:'💧', sensor:'📡', binary_sensor:'📡'};
    devices.forEach(d => {
      const name = d.id.replace('homeassistant:','').replace(/_/g,' ');
      bar.innerHTML += '<span class="device-chip">' + (icons[d.device_type]||'📟') + ' ' + name + '</span>';
    });
  } catch(e) {
    document.getElementById('statusDot').style.background = '#f87171';
    document.getElementById('statusText').textContent = 'disconnected';
  }
}
function addMsg(type, html) {
  const label = type === 'user' ? 'You' : 'microids';
  msgs.innerHTML += '<div class="msg ' + type + '"><div class="label">' + label + '</div><div class="bubble">' + html + '</div></div>';
  msgs.scrollTop = msgs.scrollHeight;
}
async function sendGoal() {
  const goal = input.value.trim();
  if (!goal) return;
  input.value = '';
  btn.disabled = true;
  addMsg('user', goal);
  addMsg('system', '<span class="spinner"></span> Working on it...');
  try {
    const r = await fetch('/goal', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({goal: goal, agent: 'groq'}) });
    const d = await r.json();
    const bubbles = msgs.querySelectorAll('.msg.system .bubble');
    const last = bubbles[bubbles.length - 1];
    let html = '';
    if (d.results) {
      d.results.forEach(t => {
        let cls, icon, status;
        if (t.skipped) { cls = 'skip'; icon = '⊘'; status = t.skip_reason; }
        else if (t.success) { cls = 'ok'; icon = '✓'; status = 'OK'; }
        else { cls = 'fail'; icon = '✗'; status = t.error || 'failed'; }
        html += '<div class="result-row ' + cls + '"><span class="icon">' + icon + '</span> ' + t.subtask_id + ' → ' + (t.device_id||'').replace('homeassistant:','') + ' ' + status + '</div>';
      });
    }
    const color = d.status === 'completed' ? '#4ade80' : '#f87171';
    html += '<div class="result-meta">Status: <span style="color:' + color + '">' + d.status + '</span> · ' + d.duration_seconds.toFixed(2) + 's</div>';
    if (d.error) html += '<div class="result-meta" style="color:#f87171">' + d.error + '</div>';
    last.innerHTML = html;
  } catch(e) {
    const bubbles = msgs.querySelectorAll('.msg.system .bubble');
    const last = bubbles[bubbles.length - 1];
    last.innerHTML = '<span style="color:#f87171">Error: ' + e.message + '</span>';
  }
  btn.disabled = false;
  input.focus();
}
init();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Simulator UI — split layout with animated device cards
# ---------------------------------------------------------------------------
_SIMULATOR_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>microids simulator</title>
<style>
:root {
  --bg: #0f0f1a;
  --surface: #1a1a2e;
  --surface2: #16213e;
  --border: #2a2a4a;
  --text: #e0e0e0;
  --text-dim: #888;
  --accent: #4a6fa5;
  --green: #4ade80;
  --red: #f87171;
  --yellow: #facc15;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); height: 100vh; overflow: hidden; }

/* Layout */
.app { display: flex; height: 100vh; }
.chat-panel { flex: 1; display: flex; flex-direction: column; border-right: 1px solid var(--border); min-width: 0; }
.device-panel { width: 380px; display: flex; flex-direction: column; background: var(--surface); flex-shrink: 0; }

/* Header */
.header { padding: 14px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 10px; background: var(--surface); }
.header h1 { font-size: 16px; font-weight: 600; color: #fff; }
.header .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); flex-shrink: 0; }
.header .badge { font-size: 10px; padding: 2px 8px; border-radius: 10px; background: var(--accent); color: #fff; margin-left: auto; }

/* Chat */
.messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
.msg { max-width: 100%; }
.msg .bubble { padding: 10px 14px; border-radius: 12px; line-height: 1.5; font-size: 13px; }
.msg.user .bubble { background: var(--border); color: #fff; margin-left: 40px; }
.msg.system .bubble { background: var(--surface2); color: var(--text); border: 1px solid var(--border); margin-right: 40px; }
.msg .label { font-size: 10px; color: var(--text-dim); margin-bottom: 3px; padding-left: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
.result-row { display: flex; align-items: center; gap: 6px; padding: 3px 0; font-size: 12px; font-family: 'SF Mono', 'Fira Code', monospace; }
.result-row .icon { font-size: 13px; }
.result-row.ok .icon { color: var(--green); }
.result-row.fail .icon { color: var(--red); }
.result-row.skip .icon { color: var(--yellow); }
.result-meta { font-size: 11px; color: var(--text-dim); margin-top: 6px; }
.welcome { text-align: center; color: var(--text-dim); font-size: 13px; padding: 40px 20px; line-height: 1.8; }
.welcome .logo { font-size: 32px; margin-bottom: 8px; }
.welcome .hint { font-size: 11px; color: #555; margin-top: 12px; }
.welcome .examples { display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; margin-top: 12px; }
.welcome .example-btn { font-size: 11px; padding: 5px 12px; border-radius: 16px; background: var(--surface2); border: 1px solid var(--border); color: var(--text-dim); cursor: pointer; transition: all 0.2s; }
.welcome .example-btn:hover { border-color: var(--accent); color: #fff; }

/* Input */
.input-area { padding: 12px 20px; border-top: 1px solid var(--border); background: var(--surface); }
.input-wrap { display: flex; gap: 8px; }
.input-wrap input { flex: 1; background: var(--surface2); border: 1px solid var(--border); border-radius: 20px; padding: 10px 16px; color: #fff; font-size: 13px; outline: none; transition: border-color 0.2s; }
.input-wrap input:focus { border-color: var(--accent); }
.input-wrap input::placeholder { color: #555; }
.input-wrap button { background: var(--accent); border: none; border-radius: 20px; padding: 10px 20px; color: #fff; font-size: 13px; cursor: pointer; font-weight: 500; transition: all 0.2s; }
.input-wrap button:hover { background: #5a7fb5; transform: scale(1.02); }
.input-wrap button:disabled { background: #333; cursor: not-allowed; transform: none; }

/* Device Panel */
.device-header { padding: 14px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 8px; }
.device-header h2 { font-size: 14px; font-weight: 600; color: #fff; }
.device-header .count { font-size: 11px; color: var(--text-dim); margin-left: auto; }
.device-list { flex: 1; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 10px; }

/* Device Cards */
.device-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 16px;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;
}
.device-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: var(--card-color, var(--border));
  transition: background 0.4s;
}
.device-card.active { border-color: var(--card-color, var(--accent)); }
.device-card.active::before { height: 3px; }
.card-top { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.card-emoji {
  font-size: 28px;
  width: 44px; height: 44px;
  display: flex; align-items: center; justify-content: center;
  background: var(--surface2);
  border-radius: 12px;
  transition: all 0.3s;
}
.card-emoji.animated { animation: pulse-emoji 1.5s ease-in-out infinite; }
.card-info { flex: 1; min-width: 0; }
.card-name { font-size: 13px; font-weight: 600; color: #fff; }
.card-zone { font-size: 10px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }
.card-state {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 11px; font-weight: 500;
  padding: 3px 10px;
  border-radius: 10px;
  background: rgba(255,255,255,0.05);
  color: var(--card-color, var(--text-dim));
  transition: all 0.3s;
}
.card-state .state-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--card-color, var(--text-dim));
  transition: background 0.3s;
}
.card-state .state-dot.animated { animation: blink-dot 1s ease-in-out infinite; }
.card-caps { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }
.cap-tag { font-size: 9px; padding: 2px 6px; border-radius: 6px; background: var(--surface2); color: var(--text-dim); }

/* Animations */
@keyframes pulse-emoji {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.15); }
}
@keyframes blink-dot {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}
@keyframes flash-card {
  0% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.4); }
  50% { box-shadow: 0 0 20px 4px rgba(74, 222, 128, 0.2); }
  100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); }
}
.device-card.flash { animation: flash-card 0.6s ease-out; }

/* Spinner */
.spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid #444; border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; vertical-align: middle; }
@keyframes spin { to { transform: rotate(360deg); } }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3a3a5a; }

/* Responsive */
@media (max-width: 768px) {
  .app { flex-direction: column-reverse; }
  .chat-panel { border-right: none; min-height: 55vh; }
  .device-panel { width: 100%; height: auto; max-height: 40vh; border-bottom: 1px solid var(--border); }
  .device-list { flex-direction: row; flex-wrap: wrap; padding: 8px; gap: 6px; overflow-x: auto; }
  .device-card { min-width: 140px; flex: 1; padding: 10px; }
  .card-emoji { font-size: 22px; width: 36px; height: 36px; }
  .card-name { font-size: 12px; }
  .card-zone { font-size: 9px; }
  .card-state { font-size: 10px; }
  .header h1 { font-size: 14px; }
  .header .badge { font-size: 9px; }
  .device-header h2 { font-size: 12px; }
  .input-wrap input { font-size: 16px; padding: 10px 14px; }
  .input-wrap button { padding: 10px 16px; font-size: 12px; }
  .welcome { padding: 20px 12px; }
  .welcome .logo { font-size: 24px; }
  .msg .bubble { font-size: 12px; padding: 8px 12px; }
  .msg.user .bubble { margin-left: 20px; }
  .msg.system .bubble { margin-right: 20px; }
  #modelSelect { max-width: 120px; font-size: 9px; }
}
</style>
</head>
<body>
<div class="app">
  <!-- Chat Panel -->
  <div class="chat-panel">
    <div class="header">
      <div class="dot" id="statusDot"></div>
      <h1>microids</h1>
      <span class="badge">simulator</span>
      <select id="modelSelect" class="badge" style="background:#333;border:1px solid #444;color:#fff;font-size:10px;padding:2px 6px;border-radius:10px;cursor:pointer;outline:none;max-width:200px;"></select>
    </div>
    <div class="messages" id="messages">
      <div class="welcome">
        <div class="logo">🏠</div>
        <div>Welcome to the microids simulator</div>
        <div style="font-size:12px; margin-top:4px;">Send a goal and watch your devices respond in real-time</div>
        <div class="examples">
          <button class="example-btn" onclick="useExample(this)">Clean the house</button>
          <button class="example-btn" onclick="useExample(this)">Water the garden</button>
          <button class="example-btn" onclick="useExample(this)">Open the garage</button>
          <button class="example-btn" onclick="useExample(this)">Turn on the lights</button>
          <button class="example-btn" onclick="useExample(this)">Secure the house</button>
        </div>
        <div class="hint">Devices on the right will animate as commands execute</div>
      </div>
    </div>
    <div class="input-area">
      <div class="input-wrap">
        <input type="text" id="goalInput" placeholder="Tell your home what to do..." autocomplete="off" maxlength="500" />
        <button id="sendBtn" onclick="sendGoal()">Send</button>
      </div>
    </div>
  </div>

  <!-- Device Panel -->
  <div class="device-panel">
    <div class="device-header">
      <span style="font-size:16px;">📡</span>
      <h2>Devices</h2>
      <span class="count" id="deviceCount">0 devices</span>
    </div>
    <div class="device-list" id="deviceList"></div>
  </div>
</div>

<script>
const msgs = document.getElementById('messages');
const input = document.getElementById('goalInput');
const btn = document.getElementById('sendBtn');
let ws = null;
let deviceStates = {};

input.addEventListener('keydown', e => { if (e.key === 'Enter' && !btn.disabled) sendGoal(); });

function useExample(el) {
  input.value = el.textContent;
  input.focus();
}

// --- WebSocket ---
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/ws');

  ws.onopen = () => {
    document.getElementById('statusDot').style.background = 'var(--green)';
  };

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'full_state') {
      deviceStates = msg.devices;
      renderDevices();
    } else if (msg.type === 'state_change') {
      if (deviceStates[msg.device]) {
        deviceStates[msg.device].state = msg.state;
        deviceStates[msg.device].label = msg.label;
        deviceStates[msg.device].color = msg.color;
        deviceStates[msg.device].animated = msg.animated;
        deviceStates[msg.device].transient = msg.transient;
        updateDeviceCard(msg.device);
      }
    }
  };

  ws.onclose = () => {
    document.getElementById('statusDot').style.background = 'var(--red)';
    setTimeout(connectWS, 2000);
  };
}

// --- Device Rendering ---
function renderDevices() {
  const list = document.getElementById('deviceList');
  const keys = Object.keys(deviceStates);
  document.getElementById('deviceCount').textContent = keys.length + ' devices';

  list.innerHTML = keys.map(key => {
    const d = deviceStates[key];
    return buildCardHTML(key, d);
  }).join('');
}

function buildCardHTML(key, d) {
  const isActive = d.state !== d.initial && d.color !== '#6b7280';
  const activeClass = isActive ? ' active' : '';
  const animClass = d.animated ? ' animated' : '';
  const dotAnim = (d.animated || d.transient) ? ' animated' : '';

  return '<div class="device-card' + activeClass + '" id="card-' + key + '" style="--card-color:' + d.color + '">' +
    '<div class="card-top">' +
      '<div class="card-emoji' + animClass + '">' + d.emoji + '</div>' +
      '<div class="card-info">' +
        '<div class="card-name">' + d.name + '</div>' +
        '<div class="card-zone">' + d.zone + '</div>' +
      '</div>' +
    '</div>' +
    '<div class="card-state">' +
      '<span class="state-dot' + dotAnim + '"></span>' +
      '<span>' + d.label + '</span>' +
    '</div>' +
  '</div>';
}

function updateDeviceCard(key) {
  const d = deviceStates[key];
  const card = document.getElementById('card-' + key);
  if (!card) return;

  // Replace card content with animation
  const isActive = d.color !== '#6b7280';
  card.style.setProperty('--card-color', d.color);
  card.className = 'device-card' + (isActive ? ' active' : '');

  const animClass = d.animated ? ' animated' : '';
  const dotAnim = (d.animated || d.transient) ? ' animated' : '';

  card.innerHTML =
    '<div class="card-top">' +
      '<div class="card-emoji' + animClass + '">' + d.emoji + '</div>' +
      '<div class="card-info">' +
        '<div class="card-name">' + d.name + '</div>' +
        '<div class="card-zone">' + d.zone + '</div>' +
      '</div>' +
    '</div>' +
    '<div class="card-state">' +
      '<span class="state-dot' + dotAnim + '"></span>' +
      '<span>' + d.label + '</span>' +
    '</div>';

  // Flash effect
  card.classList.add('flash');
  setTimeout(() => card.classList.remove('flash'), 600);
}

// --- Chat ---
function clearWelcome() {
  const welcome = msgs.querySelector('.welcome');
  if (welcome) welcome.remove();
}

function addMsg(type, html) {
  clearWelcome();
  const label = type === 'user' ? 'You' : 'microids';
  const div = document.createElement('div');
  div.className = 'msg ' + type;
  div.innerHTML = '<div class="label">' + label + '</div><div class="bubble">' + html + '</div>';
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return div;
}

async function sendGoal() {
  const goal = input.value.trim();
  if (!goal) return;
  input.value = '';
  btn.disabled = true;

  addMsg('user', escapeHtml(goal));
  const sysMsg = addMsg('system', '<span class="spinner"></span> Planning and executing...');

  try {
    const r = await fetch('/goal', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({goal: goal, agent: 'groq'})
    });
    const d = await r.json();
    const bubble = sysMsg.querySelector('.bubble');

    let html = '';
    if (d.results && d.results.length) {
      d.results.forEach(t => {
        let cls, icon, status;
        const device = (t.device_id || '').replace('mock:', '');
        if (t.skipped) { cls = 'skip'; icon = '⊘'; status = t.skip_reason; }
        else if (t.success) { cls = 'ok'; icon = '✓'; status = 'done'; }
        else { cls = 'fail'; icon = '✗'; status = t.error || 'failed'; }
        html += '<div class="result-row ' + cls + '"><span class="icon">' + icon + '</span> ' + escapeHtml(t.subtask_id) + (device ? ' → ' + device : '') + ' ' + escapeHtml(status) + '</div>';
      });
    }
    const color = d.status === 'completed' ? 'var(--green)' : 'var(--red)';
    html += '<div class="result-meta">' + d.status + ' · ' + d.duration_seconds.toFixed(2) + 's</div>';
    if (d.error) html += '<div class="result-meta" style="color:var(--red)">' + escapeHtml(d.error) + '</div>';
    bubble.innerHTML = html;
  } catch(e) {
    const bubble = sysMsg.querySelector('.bubble');
    bubble.innerHTML = '<span style="color:var(--red)">Error: ' + escapeHtml(e.message) + '</span>';
  }
  btn.disabled = false;
  input.focus();
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// --- Init ---
connectWS();

// Load models dropdown
fetch('/models').then(r=>r.json()).then(d=>{
  const sel = document.getElementById('modelSelect');
  if(!sel) return;
  d.models.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = m.name;
    if(d.current && d.current.includes(m.id)) opt.selected = true;
    sel.appendChild(opt);
  });
  sel.onchange = async () => {
    sel.disabled = true;
    try {
      const r = await fetch('/models/switch', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({model_id: sel.value})
      });
      const data = await r.json();
      if(data.status !== 'ok') alert('Switch failed');
    } catch(e) { alert('Switch failed: ' + e.message); }
    sel.disabled = false;
  };
}).catch(()=>{});
</script>
</body>
</html>"""
