"""microids CLI — OpenClaw-style command interface.

Command structure mirrors OpenClaw:
    microids setup                          # Interactive setup wizard
    microids doctor [--fix]                 # Diagnose and fix issues
    microids status [--json]                # System status
    microids tui                            # Interactive terminal chat
    microids agent --message "..."          # Execute a goal
    microids gateway run                    # Run server in foreground
    microids gateway start/stop/restart     # Manage systemd service
    microids gateway status                 # Gateway health
    microids channels list                  # List channels
    microids channels status [--probe]      # Channel health
    microids devices list [--json]          # List devices
    microids models list                    # List available models
    microids models status [--probe]        # Model connectivity
    microids config get [KEY]               # Get config
    microids config set KEY VALUE           # Set config
    microids logs [-f] [-n 50]              # Tail logs
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(stderr=True)
JSON_MODE = False


def _load_env() -> None:
    """Load .env file if present."""
    env_file = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k, v)


def _get_channel(channel_type: str):
    if channel_type == "ha":
        from microids.channels.homeassistant import HomeAssistantChannel
        return HomeAssistantChannel()
    else:
        from microids.channels.mock import MockChannel
        return MockChannel()


def _get_agent(agent_type: str):
    if agent_type == "groq":
        from microids.agents.groq import GroqAgent
        return GroqAgent()
    elif agent_type == "bedrock":
        from microids.agents.bedrock import BedrockAgent
        return BedrockAgent()
    elif agent_type == "ollama":
        from microids.agents.ollama import OllamaAgent
        return OllamaAgent()
    else:
        from microids.agents.mock import MockAgent
        return MockAgent()


def _build_gateway(channel_type: str = "mock", agent_type: str = "mock"):
    from microids.core.events import EventBus
    from microids.core.gateway import Gateway
    from microids.core.recovery import RecoveryEngine
    from microids.core.registry import Registry
    from microids.core.router import Router

    channel = _get_channel(channel_type)
    agent = _get_agent(agent_type)
    registry = Registry()
    event_bus = EventBus()
    router = Router(default_channel=channel)
    recovery_engine = RecoveryEngine(
        registry=registry, router=router, event_bus=event_bus,
    )
    gw = Gateway(
        agent=agent, router=router, registry=registry,
        event_bus=event_bus, recovery_engine=recovery_engine,
    )
    return gw, registry, event_bus


def _output(data: Any) -> None:
    if JSON_MODE:
        click.echo(json.dumps(data, indent=2, default=str))


def _check(label: str, ok: bool, detail: str = "") -> None:
    icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
    suffix = f": {detail}" if detail else ""
    console.print(f"  {icon} {label}{suffix}")


# ═══════════════════════════════════════════════════════════════════════════
# Root group + global flags
# ═══════════════════════════════════════════════════════════════════════════

@click.group()
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON")
def main(json_mode: bool) -> None:
    """microids — AI-coordinated device workflows."""
    global JSON_MODE
    JSON_MODE = json_mode
    _load_env()


# ═══════════════════════════════════════════════════════════════════════════
# microids setup  (OpenClaw: openclaw onboard)
# ═══════════════════════════════════════════════════════════════════════════

@main.command()
def setup() -> None:
    """Interactive setup wizard — Apple Home-inspired device onboarding."""
    console.print(Panel("🏠 microids setup", subtitle="Let's connect your home"))
    console.print()

    env_path = os.path.join(os.getcwd(), ".env")

    # Ensure .env exists
    if not os.path.exists(env_path):
        env_example = os.path.join(os.getcwd(), ".env.example")
        if os.path.exists(env_example):
            import shutil
            shutil.copy(env_example, env_path)
        else:
            _write_default_env(env_path)
    _load_env()

    # ── Step 1: Find Home Assistant ──────────────────────────────
    console.print("[bold]Step 1:[/bold] Finding Home Assistant\n")
    ha_url = os.environ.get("HA_URL", "")
    ha_found = False

    if ha_url:
        ha_found = _probe_ha(ha_url)
        if ha_found:
            console.print(f"  [green]✓[/green] Found Home Assistant at {ha_url}")
        else:
            console.print(f"  [yellow]![/yellow] Configured URL {ha_url} is not reachable")
            ha_url = ""

    if not ha_found:
        # Auto-discover on common addresses
        console.print("  🔍 Searching for Home Assistant...")
        for candidate in ["http://localhost:8123", "http://127.0.0.1:8123",
                          "http://homeassistant.local:8123", "http://homeassistant:8123"]:
            if _probe_ha(candidate):
                ha_url = candidate
                ha_found = True
                console.print(f"  [green]✓[/green] Found Home Assistant at {ha_url}")
                break

    if not ha_found:
        ha_url = click.prompt("  Enter your Home Assistant URL", default="http://localhost:8123")
        if _probe_ha(ha_url):
            ha_found = True
            console.print(f"  [green]✓[/green] Connected to {ha_url}")
        else:
            console.print(f"  [red]✗[/red] Cannot reach {ha_url}")
            console.print("  Make sure Home Assistant is running and accessible.")
            console.print("  You can continue with the simulator: [bold]microids gateway run[/bold]")
            _save_env_key(env_path, "HA_URL", ha_url)
            return

    _save_env_key(env_path, "HA_URL", ha_url)

    # ── Step 2: Get access token ─────────────────────────────────
    console.print(f"\n[bold]Step 2:[/bold] Connecting to Home Assistant\n")
    ha_token = os.environ.get("HA_TOKEN", "")

    if ha_token and _verify_ha_token(ha_url, ha_token):
        console.print("  [green]✓[/green] Existing token is valid")
    else:
        token_url = ha_url + "/profile"
        console.print("  microids needs an access token to control your devices.\n")
        console.print(f"  [bold]→ Open this link:[/bold] {token_url}")
        console.print("    1. Scroll to \"Long-Lived Access Tokens\"")
        console.print("    2. Click \"Create Token\" → name it \"microids\"")
        console.print("    3. Copy the token and paste it below\n")

        while True:
            ha_token = click.prompt("  Paste your token", hide_input=True)
            if _verify_ha_token(ha_url, ha_token):
                console.print("  [green]✓[/green] Token verified")
                break
            else:
                console.print("  [red]✗[/red] Invalid token — try again")

    _save_env_key(env_path, "HA_TOKEN", ha_token)
    os.environ["HA_TOKEN"] = ha_token

    # ── Step 3: Discover devices ─────────────────────────────────
    console.print(f"\n[bold]Step 3:[/bold] Discovering your devices\n")
    devices = _discover_ha_devices(ha_url, ha_token)

    if devices:
        EMOJIS = {"vacuum": "🤖", "cover": "🚪", "light": "💡", "switch": "💧",
                  "camera": "📷", "climate": "🌡️", "fan": "🌀"}
        for d in devices:
            emoji = EMOJIS.get(d["domain"], "📟")
            console.print(f"  {emoji} {d['name']} = {d['state']}")
        console.print(f"\n  [green]✓[/green] Found {len(devices)} controllable devices")
    else:
        console.print("  [yellow]![/yellow] No controllable devices found")
        console.print("  Add devices in Home Assistant first, then re-run setup.")

    # ── Step 4: Choose LLM ───────────────────────────────────────
    console.print(f"\n[bold]Step 4:[/bold] Choose your AI model\n")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    model = os.environ.get("MICROIDS_MODEL", "")

    if groq_key and model:
        console.print(f"  [green]✓[/green] Model: {model} (API key set)")
    else:
        console.print("  [cyan]groq[/cyan]    — Llama 3.3 70B, fast, free tier (recommended)")
        console.print("  [cyan]bedrock[/cyan] — Claude 3.5 Haiku via AWS")
        console.print("  [cyan]ollama[/cyan]  — Local model, no cloud")
        console.print("  [cyan]mock[/cyan]    — No LLM, hardcoded responses\n")
        model = click.prompt("  Choose a model", default="groq",
                             type=click.Choice(["groq", "bedrock", "ollama", "mock"]))
        _save_env_key(env_path, "MICROIDS_MODEL", model)
        os.environ["MICROIDS_MODEL"] = model

        if model == "groq" and not groq_key:
            console.print("\n  Get a free API key at [bold]https://console.groq.com[/bold]")
            groq_key = click.prompt("  Paste your Groq API key", hide_input=True)
            _save_env_key(env_path, "GROQ_API_KEY", groq_key)
            os.environ["GROQ_API_KEY"] = groq_key

    # ── Step 5: Create ROUTINES.md ───────────────────────────────
    routines_path = os.path.join(os.getcwd(), "ROUTINES.md")
    if not os.path.exists(routines_path):
        _write_default_routines(routines_path)
        console.print(f"\n  [green]✓[/green] Created ROUTINES.md — customize your home routines")

    # ── Done ─────────────────────────────────────────────────────
    console.print(f"\n[bold green]✓ Setup complete![/bold green]\n")
    console.print("  Verify:  [bold]microids doctor[/bold]")
    if devices:
        console.print(f"  Try it:  [bold]microids agent -m \"turn on the lights\"[/bold]")
    else:
        console.print(f"  Try it:  [bold]MICROIDS_SIMULATOR=1 microids gateway run[/bold]")


def _probe_ha(url: str) -> bool:
    """Check if Home Assistant is reachable at the given URL."""
    import socket
    import urllib.request
    try:
        req = urllib.request.Request(url + "/api/", method="GET")
        urllib.request.urlopen(req, timeout=3)
        return True
    except urllib.error.HTTPError as e:
        return e.code == 401  # 401 = HA is there, just needs auth
    except Exception:
        return False


def _verify_ha_token(url: str, token: str) -> bool:
    """Verify a HA token works."""
    import urllib.request
    try:
        req = urllib.request.Request(
            url + "/api/",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp = urllib.request.urlopen(req, timeout=5)
        return resp.status == 200
    except Exception:
        return False


def _discover_ha_devices(url: str, token: str) -> list[dict]:
    """Discover controllable devices from HA."""
    import urllib.request
    try:
        req = urllib.request.Request(
            url + "/api/states",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp = urllib.request.urlopen(req, timeout=10)
        states = json.loads(resp.read())
    except Exception:
        return []

    controllable = ["vacuum", "light", "switch", "cover", "camera", "climate", "fan"]
    skip_prefixes = ("backup_", "sun_", "hacs_", "homeassistant_")
    devices = []
    for s in states:
        eid = s["entity_id"]
        domain = eid.split(".")[0]
        if domain not in controllable:
            continue
        obj_id = eid.split(".", 1)[1]
        if any(obj_id.startswith(p) for p in skip_prefixes):
            continue
        devices.append({
            "entity_id": eid,
            "domain": domain,
            "name": s.get("attributes", {}).get("friendly_name", eid),
            "state": s["state"],
        })
    return devices


def _save_env_key(env_path: str, key: str, value: str) -> None:
    """Write or update a key in the .env file."""
    lines, found = [], False
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith(f"{key}="):
                    lines.append(f"{key}={value}\n")
                    found = True
                else:
                    lines.append(line)
    if not found:
        lines.append(f"{key}={value}\n")
    with open(env_path, "w") as f:
        f.writelines(lines)
    os.environ[key] = value


def _write_default_env(path: str) -> None:
    with open(path, "w") as f:
        f.write("# Home Assistant\nHA_URL=http://localhost:8123\nHA_TOKEN=\n\n"
                "# Groq (free at console.groq.com)\nGROQ_API_KEY=\nGROQ_MODEL_ID=llama-3.3-70b-versatile\n")


def _write_default_routines(path: str) -> None:
    with open(path, "w") as f:
        f.write("# My Home Routines\n\n## Example\nWhen the user says \"example\", execute ALL tasks:\n"
                "1. Turn on the light (device_id: light.living_room, capability: turn_on)\n\n"
                "IMPORTANT: Always execute ALL tasks listed in a routine. Never skip tasks.\n")


# ═══════════════════════════════════════════════════════════════════════════
# microids agent  (OpenClaw: openclaw agent --message)
# ═══════════════════════════════════════════════════════════════════════════

@main.command()
@click.option("--message", "-m", required=True, help="Goal to execute")
@click.option("--channel", "channel_type", default="ha", help="Channel: mock or ha")
@click.option("--model", "agent_type", default=None, help="Agent: mock, ollama, bedrock, or groq (default: from config)")
@click.option("--remote", is_flag=True, help="Execute on remote server")
def agent(message: str, channel_type: str, agent_type: str | None, remote: bool) -> None:
    """Execute a goal via the agent."""
    # OpenClaw-style: read default model from config, --model overrides
    if agent_type is None:
        agent_type = os.environ.get("MICROIDS_MODEL", "groq")
    if remote:
        asyncio.run(_run_agent_remote(message, agent_type))
    else:
        asyncio.run(_run_agent_local(message, channel_type, agent_type))


async def _run_agent_remote(message: str, agent_type: str) -> None:
    import aiohttp
    import subprocess

    server_url = os.environ.get("MICROIDS_SERVER_URL", "https://ha.cloudcomps.net/microids")
    cf_id = os.environ.get("CF_ACCESS_CLIENT_ID", "")
    cf_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET", "")

    headers = {"Content-Type": "application/json"}
    if cf_id and cf_secret:
        headers["CF-Access-Client-Id"] = cf_id
        headers["CF-Access-Client-Secret"] = cf_secret

    console.print("[dim]Connecting...[/dim]")

    # Try direct
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{server_url}/health", headers=headers,
                             timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status == 200 and "status" in (await r.json()):
                    return await _send_agent_remote(server_url, headers, message, agent_type)
    except Exception:
        pass

    # SSH tunnel fallback
    console.print("[dim]Tunneling via SSH...[/dim]")
    ssh_key = os.environ.get("MICROIDS_SSH_KEY", os.path.expanduser("~/.ssh/microids_hetzner"))
    server_ip = os.environ.get("MICROIDS_SERVER_IP", "178.156.229.79")
    tunnel = subprocess.Popen(
        ["ssh", "-i", ssh_key, "-N", "-L", "18200:127.0.0.1:8200",
         f"root@{server_ip}", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        await asyncio.sleep(2)
        return await _send_agent_remote(
            "http://127.0.0.1:18200", {"Content-Type": "application/json"}, message, agent_type)
    finally:
        tunnel.terminate()
        tunnel.wait(timeout=5)


async def _send_agent_remote(url: str, headers: dict, message: str, agent_type: str) -> None:
    import aiohttp
    async with aiohttp.ClientSession() as s:
        async with s.post(f"{url}/goal", json={"goal": message, "agent": agent_type},
                          headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as r:
            if r.status != 200:
                console.print(f"[red]Error {r.status}:[/red] {(await r.text())[:300]}")
                return
            data = await r.json()
    _display_result(data)


async def _run_agent_local(message: str, channel_type: str, agent_type: str) -> None:
    gw, registry, event_bus = _build_gateway(channel_type, agent_type)
    if channel_type == "ha":
        console.print("[dim]Connecting to Home Assistant...[/dim]")

    await gw.start()
    try:
        devs = await registry.list_all()
        console.print(f"[dim]{len(devs)} device(s) discovered[/dim]")

        result = await gw.goal(message)

        data = {
            "goal": result.goal, "status": result.status.value,
            "duration_seconds": result.duration_seconds, "results": [], "error": result.error,
        }
        if result.execution_report:
            for tr in result.execution_report.results:
                skipped = (tr.response or {}).get("status") == "skipped" if isinstance(tr.response, dict) else False
                data["results"].append({
                    "subtask_id": tr.subtask_id, "device_id": tr.device_id or "",
                    "success": tr.success, "error": tr.error, "skipped": skipped,
                    "skip_reason": (tr.response or {}).get("reason", "") if skipped else "",
                })

        _output(data)
        if not JSON_MODE:
            _display_result(data)

        evts = event_bus.get_history()
        if evts and not JSON_MODE:
            console.print(f"\n[dim]{len(evts)} events captured[/dim]")
    finally:
        await gw.stop()


def _display_result(data: dict) -> None:
    sc = {"completed": "green", "failed": "red", "suspended": "yellow"}.get(data["status"], "white")
    console.print(Panel(
        f"[bold]Goal:[/bold] {data['goal']}\n"
        f"[bold]Status:[/bold] [{sc}]{data['status']}[/{sc}]\n"
        f"[bold]Duration:[/bold] {data['duration_seconds']:.2f}s",
        title="microids",
    ))
    for tr in data.get("results", []):
        if tr.get("skipped"):
            console.print(f"  [yellow]⊘[/yellow] {tr['subtask_id']} → {tr.get('device_id','N/A')} {tr.get('skip_reason','skipped')}")
        elif tr.get("success"):
            console.print(f"  [green]✓[/green] {tr['subtask_id']} → {tr.get('device_id','N/A')} OK")
        else:
            console.print(f"  [red]✗[/red] {tr['subtask_id']} → {tr.get('device_id','N/A')} {tr.get('error','failed')}")
    if data.get("error"):
        console.print(f"\n[red]Error:[/red] {data['error']}")


# ═══════════════════════════════════════════════════════════════════════════
# microids tui  (OpenClaw: openclaw tui)
# ═══════════════════════════════════════════════════════════════════════════

@main.command()
@click.option("--channel", "channel_type", default="ha", help="Channel: mock or ha")
@click.option("--model", "agent_type", default=None, help="Agent: mock, ollama, bedrock, or groq (default: from config)")
def tui(channel_type: str, agent_type: str | None) -> None:
    """Interactive terminal chat."""
    if agent_type is None:
        agent_type = os.environ.get("MICROIDS_MODEL", "groq")
    asyncio.run(_run_tui(channel_type, agent_type))


async def _run_tui(channel_type: str, agent_type: str) -> None:
    gw, registry, event_bus = _build_gateway(channel_type, agent_type)

    console.print(Panel(
        f"Channel: {channel_type} · Model: {agent_type}\n"
        f"Type a goal and press Enter. /help for commands. /exit to quit.",
        title="microids tui",
    ))

    await gw.start()
    try:
        devs = await registry.list_all()
        console.print(f"[dim]{len(devs)} device(s) connected[/dim]\n")

        while True:
            try:
                text = console.input("[bold cyan]>[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not text:
                continue

            if text.startswith("/"):
                cmd = text.lower().split()[0]
                if cmd in ("/exit", "/quit", "/q"):
                    break
                elif cmd == "/help":
                    console.print("  /devices  — list devices\n  /status   — system status\n"
                                  "  /model    — current model\n  /exit     — quit")
                elif cmd == "/devices":
                    for d in devs:
                        caps = ", ".join(c.name for c in d.spec.capabilities)
                        console.print(f"  {d.id} ({d.spec.device_type}) [{caps}]")
                elif cmd == "/status":
                    console.print(f"  Channel: {channel_type} · Model: {agent_type} · "
                                  f"Devices: {len(devs)} · Events: {len(event_bus.get_history())}")
                elif cmd == "/model":
                    console.print(f"  {_get_agent(agent_type).model_name()}")
                else:
                    console.print(f"  [yellow]Unknown: {cmd}[/yellow] — try /help")
                continue

            console.print("[dim]Working...[/dim]")
            try:
                result = await gw.goal(text)
                sc = {"completed": "green", "failed": "red", "suspended": "yellow"}.get(result.status.value, "white")
                if result.execution_report:
                    for tr in result.execution_report.results:
                        skipped = (tr.response or {}).get("status") == "skipped" if isinstance(tr.response, dict) else False
                        if skipped:
                            console.print(f"  [yellow]⊘[/yellow] {tr.subtask_id} → {tr.device_id} {(tr.response or {}).get('reason','skipped')}")
                        elif tr.success:
                            console.print(f"  [green]✓[/green] {tr.subtask_id} → {tr.device_id} OK")
                        else:
                            console.print(f"  [red]✗[/red] {tr.subtask_id} → {tr.device_id} {tr.error or 'failed'}")
                console.print(f"  [{sc}]{result.status.value}[/{sc}] · {result.duration_seconds:.2f}s\n")
                if result.error:
                    console.print(f"  [red]{result.error}[/red]\n")
            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]\n")
    finally:
        await gw.stop()
        console.print("[dim]Disconnected.[/dim]")


# ═══════════════════════════════════════════════════════════════════════════
# microids gateway  (OpenClaw: openclaw gateway run/start/stop/status)
# ═══════════════════════════════════════════════════════════════════════════

@main.group()
def gateway() -> None:
    """Manage the microids gateway server."""
    pass


@gateway.command("run")
@click.option("--host", default="127.0.0.1", help="Bind address (use 127.0.0.1 for security)")
@click.option("--port", default=8200, help="Port number")
def gateway_run(host: str, port: int) -> None:
    """Run gateway in foreground."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed.[/red] Run: pip install microids[server]")
        return
    console.print(f"[dim]Starting gateway on {host}:{port}...[/dim]")
    uvicorn.run("microids.server:app", host=host, port=port, log_level="info")


@gateway.command("status")
def gateway_status() -> None:
    """Check gateway health."""
    asyncio.run(_gateway_status())


async def _gateway_status() -> None:
    import aiohttp
    server_url = os.environ.get("MICROIDS_SERVER_URL", "http://127.0.0.1:8200")
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{server_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                data = await r.json()
                if JSON_MODE:
                    _output(data)
                else:
                    _check("Gateway", True, f"{data.get('devices', 0)} devices connected")
    except Exception as e:
        if JSON_MODE:
            _output({"status": "offline", "error": str(e)})
        else:
            _check("Gateway", False, str(e))


@gateway.command("start")
def gateway_start() -> None:
    """Start gateway service (systemd)."""
    ssh_key = os.environ.get("MICROIDS_SSH_KEY", os.path.expanduser("~/.ssh/microids_hetzner"))
    server_ip = os.environ.get("MICROIDS_SERVER_IP", "178.156.229.79")
    os.system(f"ssh -i {ssh_key} root@{server_ip} 'systemctl start microids'")
    console.print("  [green]✓[/green] Gateway started")


@gateway.command("stop")
def gateway_stop() -> None:
    """Stop gateway service (systemd)."""
    ssh_key = os.environ.get("MICROIDS_SSH_KEY", os.path.expanduser("~/.ssh/microids_hetzner"))
    server_ip = os.environ.get("MICROIDS_SERVER_IP", "178.156.229.79")
    os.system(f"ssh -i {ssh_key} root@{server_ip} 'systemctl stop microids'")
    console.print("  [green]✓[/green] Gateway stopped")


@gateway.command("restart")
def gateway_restart() -> None:
    """Restart gateway service (systemd)."""
    ssh_key = os.environ.get("MICROIDS_SSH_KEY", os.path.expanduser("~/.ssh/microids_hetzner"))
    server_ip = os.environ.get("MICROIDS_SERVER_IP", "178.156.229.79")
    os.system(f"ssh -i {ssh_key} root@{server_ip} 'systemctl restart microids'")
    console.print("  [green]✓[/green] Gateway restarted")


# ═══════════════════════════════════════════════════════════════════════════
# microids channels  (OpenClaw: openclaw channels list/status)
# ═══════════════════════════════════════════════════════════════════════════

@main.group()
def channels() -> None:
    """Manage device channels."""
    pass


@channels.command("list")
def channels_list() -> None:
    """List configured channels."""
    available = [
        {"name": "mock", "type": "mock", "description": "Simulated devices for testing"},
        {"name": "homeassistant", "type": "ha", "description": "Home Assistant REST API"},
    ]
    if JSON_MODE:
        _output(available)
        return
    table = Table(title="Channels")
    table.add_column("Name", style="cyan")
    table.add_column("Type")
    table.add_column("Description")
    for ch in available:
        table.add_row(ch["name"], ch["type"], ch["description"])
    console.print(table)


@channels.command("status")
@click.option("--probe", is_flag=True, help="Test connectivity")
def channels_status(probe: bool) -> None:
    """Check channel health."""
    asyncio.run(_channels_status(probe))


async def _channels_status(probe: bool) -> None:
    _check("mock", True, "always available")

    ha_url = os.environ.get("HA_URL", "")
    ha_token = os.environ.get("HA_TOKEN", "")

    if not ha_url or not ha_token:
        _check("homeassistant", False, "HA_URL or HA_TOKEN not set")
        return

    if not probe:
        _check("homeassistant", True, f"configured ({ha_url})")
        return

    try:
        from microids.channels.homeassistant import HomeAssistantChannel
        ch = HomeAssistantChannel(ha_url, ha_token)
        await ch.connect({})
        specs = await ch.discover_devices()
        _check("homeassistant", True, f"connected, {len(specs)} devices")
        await ch.disconnect()
    except Exception as e:
        _check("homeassistant", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# microids devices  (OpenClaw: openclaw devices list)
# ═══════════════════════════════════════════════════════════════════════════

@main.group()
def devices() -> None:
    """Manage devices."""
    pass


@devices.command("list")
@click.option("--channel", "channel_type", default="mock", help="Channel: mock or ha")
def devices_list(channel_type: str) -> None:
    """List all discovered devices."""
    asyncio.run(_devices_list(channel_type))


async def _devices_list(channel_type: str) -> None:
    gw, registry, _ = _build_gateway(channel_type)
    await gw.start()
    try:
        all_devs = await registry.list_all()
        if JSON_MODE:
            _output([{
                "id": d.id, "device_type": d.spec.device_type,
                "category": d.spec.category.value, "zone": d.spec.context.zone,
                "capabilities": [c.name for c in d.spec.capabilities],
                "status": d.status.value,
                "battery": d.spec.context.battery_level,
            } for d in all_devs])
            return

        table = Table(title=f"Devices ({channel_type})")
        table.add_column("ID", style="cyan")
        table.add_column("Type")
        table.add_column("Category", style="magenta")
        table.add_column("Zone", style="green")
        table.add_column("Capabilities")
        table.add_column("Status")

        for d in all_devs:
            caps = ", ".join(c.name for c in d.spec.capabilities)
            friendly = d.spec.metadata.get("friendly_name", "")
            name = f"{d.id}" + (f" ({friendly})" if friendly and friendly != d.id else "")
            table.add_row(name, d.spec.device_type, d.spec.category.value,
                          d.spec.context.zone, caps, d.status.value)
        console.print(table)
        console.print(f"\n{len(all_devs)} device(s) total")
    finally:
        await gw.stop()


# ═══════════════════════════════════════════════════════════════════════════
# microids models  (OpenClaw: openclaw models list/status)
# ═══════════════════════════════════════════════════════════════════════════

@main.group()
def models() -> None:
    """Manage AI models."""
    pass


@models.command("list")
def models_list() -> None:
    """List available models and show the active one."""
    current = os.environ.get("MICROIDS_MODEL", "groq")
    available = [
        {"id": "groq", "model": "llama-3.3-70b-versatile", "speed": "280 t/s", "cost": "free/dev tier"},
        {"id": "bedrock", "model": "claude-3.5-haiku", "speed": "~50 t/s", "cost": "AWS pricing"},
        {"id": "ollama", "model": "qwen3:8b", "speed": "~15 t/s", "cost": "free (local)"},
        {"id": "mock", "model": "mock-agent-v1", "speed": "instant", "cost": "free"},
    ]
    if JSON_MODE:
        _output({"current": current, "models": available})
        return
    table = Table(title="Available Models")
    table.add_column("", width=2)
    table.add_column("Agent ID", style="cyan")
    table.add_column("Model")
    table.add_column("Speed")
    table.add_column("Cost")
    for m in available:
        active = "●" if m["id"] == current else " "
        style = "bold green" if m["id"] == current else ""
        table.add_row(active, m["id"], m["model"], m["speed"], m["cost"], style=style)
    console.print(table)
    console.print(f"\n  Active: [green]{current}[/green]  (change with: microids models set <agent>)")


@models.command("set")
@click.argument("agent_id")
def models_set(agent_id: str) -> None:
    """Set the default model. Usage: microids models set groq"""
    valid = {"groq", "bedrock", "ollama", "mock"}
    if agent_id not in valid:
        console.print(f"  [red]Unknown agent: {agent_id}[/red]")
        console.print(f"  Valid options: {', '.join(sorted(valid))}")
        return

    # Write to .env via the same mechanism as config set
    env_path = os.path.join(os.getcwd(), ".env")
    lines, found = [], False
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith("MICROIDS_MODEL="):
                    lines.append(f"MICROIDS_MODEL={agent_id}\n")
                    found = True
                else:
                    lines.append(line)
    if not found:
        lines.append(f"MICROIDS_MODEL={agent_id}\n")
    with open(env_path, "w") as f:
        f.writelines(lines)

    os.environ["MICROIDS_MODEL"] = agent_id
    console.print(f"  [green]✓[/green] Default model set to [cyan]{agent_id}[/cyan]")
    console.print(f"  All future commands will use {agent_id} unless --model is specified.")


@models.command("status")
@click.option("--probe", is_flag=True, help="Test model connectivity")
def models_status(probe: bool) -> None:
    """Check model availability."""
    asyncio.run(_models_status(probe))


async def _models_status(probe: bool) -> None:
    groq_key = os.environ.get("GROQ_API_KEY", "")
    bedrock_profile = os.environ.get("BEDROCK_AWS_PROFILE", "")

    _check("groq", bool(groq_key), "API key set" if groq_key else "GROQ_API_KEY not set")
    _check("bedrock", bool(bedrock_profile), f"profile: {bedrock_profile}" if bedrock_profile else "no profile")
    _check("ollama", True, "localhost:11434 (check manually)")
    _check("mock", True, "always available")

    if probe and groq_key:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                async with s.get("https://api.groq.com/openai/v1/models",
                                 headers={"Authorization": f"Bearer {groq_key}"},
                                 timeout=aiohttp.ClientTimeout(total=5)) as r:
                    _check("groq connectivity", r.status == 200, f"HTTP {r.status}")
        except Exception as e:
            _check("groq connectivity", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# microids config  (OpenClaw: openclaw config get/set)
# ═══════════════════════════════════════════════════════════════════════════

@main.group()
def config() -> None:
    """Manage configuration."""
    pass


@config.command("get")
@click.argument("key", required=False)
def config_get(key: Optional[str]) -> None:
    """Get a config value (or all config)."""
    env_path = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(env_path):
        console.print("[red].env not found[/red] — run microids setup")
        return

    data = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                data[k] = v

    if key:
        val = data.get(key)
        if val is not None:
            if JSON_MODE:
                _output({key: val})
            else:
                console.print(f"  {key} = {val}")
        else:
            console.print(f"  [yellow]{key} not found[/yellow]")
    else:
        safe = {}
        for k, v in data.items():
            safe[k] = "***" if any(s in k.upper() for s in ["TOKEN", "KEY", "SECRET", "PASSWORD"]) else v
        if JSON_MODE:
            _output(safe)
        else:
            for k, v in safe.items():
                console.print(f"  {k} = {v}")


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a config value."""
    env_path = os.path.join(os.getcwd(), ".env")
    lines, found = [], False
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith(f"{key}="):
                    lines.append(f"{key}={value}\n")
                    found = True
                else:
                    lines.append(line)
    if not found:
        lines.append(f"{key}={value}\n")
    with open(env_path, "w") as f:
        f.writelines(lines)
    os.environ[key] = value
    console.print(f"  [green]✓[/green] {key} updated")


# ═══════════════════════════════════════════════════════════════════════════
# microids doctor  (OpenClaw: openclaw doctor [--fix])
# ═══════════════════════════════════════════════════════════════════════════

@main.command()
@click.option("--fix", is_flag=True, help="Auto-fix issues where possible")
def doctor(fix: bool) -> None:
    """Diagnose setup and optionally fix issues."""
    asyncio.run(_doctor(fix))


async def _doctor(fix: bool) -> None:
    issues = []
    console.print(Panel("microids doctor", subtitle="Diagnosing your setup"))

    env_exists = os.path.exists(".env")
    _check(".env file", env_exists)
    if not env_exists:
        issues.append("missing .env")
        if fix:
            _write_default_env(".env")
            console.print("    [green]→ Created .env[/green]")

    routines_exists = os.path.exists("ROUTINES.md")
    _check("ROUTINES.md", routines_exists)
    if not routines_exists:
        issues.append("missing ROUTINES.md")
        if fix:
            _write_default_routines("ROUTINES.md")
            console.print("    [green]→ Created ROUTINES.md[/green]")

    ha_url = os.environ.get("HA_URL", "")
    ha_token = os.environ.get("HA_TOKEN", "")
    _check("HA_URL", bool(ha_url), ha_url or "not set")
    _check("HA_TOKEN", bool(ha_token), "***" if ha_token else "not set")
    if not ha_url:
        issues.append("HA_URL not set")
    if not ha_token:
        issues.append("HA_TOKEN not set")

    if ha_url and ha_token:
        try:
            from microids.channels.homeassistant import HomeAssistantChannel
            ch = HomeAssistantChannel(ha_url, ha_token)
            await ch.connect({})
            _check("HA connection", True, ha_url)
            specs = await ch.discover_devices()
            _check("Device discovery", len(specs) > 0, f"{len(specs)} device(s)")
            by_type: dict[str, int] = {}
            for s in specs:
                by_type[s.device_type] = by_type.get(s.device_type, 0) + 1
            for dtype, count in sorted(by_type.items()):
                _check(f"  {dtype}", True, f"{count}")
            await ch.disconnect()
        except Exception as e:
            _check("HA connection", False, str(e))
            issues.append(f"HA: {e}")
            if "SSL" in str(e) or "certificate" in str(e).lower():
                console.print("  [yellow]SSL issue — likely corporate proxy. Install pip-system-certs.[/yellow]")

    groq_key = os.environ.get("GROQ_API_KEY", "")
    _check("Groq API key", bool(groq_key))
    if not groq_key:
        issues.append("GROQ_API_KEY not set")

    if groq_key:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                async with s.get("https://api.groq.com/openai/v1/models",
                                 headers={"Authorization": f"Bearer {groq_key}"},
                                 timeout=aiohttp.ClientTimeout(total=5)) as r:
                    _check("Groq API", r.status == 200, f"HTTP {r.status}")
                    if r.status != 200:
                        issues.append("Groq API error")
        except Exception as e:
            _check("Groq API", False, str(e))
            issues.append(f"Groq: {e}")

    if issues:
        console.print(f"\n[yellow]{len(issues)} issue(s)[/yellow]")
        if not fix:
            console.print("  Run [bold]microids doctor --fix[/bold] to auto-fix")
    else:
        console.print("\n[green]All systems operational.[/green]")

    if JSON_MODE:
        _output({"issues": issues, "fix_applied": fix})


# ═══════════════════════════════════════════════════════════════════════════
# microids status  (OpenClaw: openclaw status)
# ═══════════════════════════════════════════════════════════════════════════

@main.command()
def status() -> None:
    """Show system status."""
    data = {
        "env_file": os.path.exists(".env"),
        "routines_file": os.path.exists("ROUTINES.md"),
        "ha_url": os.environ.get("HA_URL", "") or None,
        "ha_token_set": bool(os.environ.get("HA_TOKEN", "")),
        "groq_key_set": bool(os.environ.get("GROQ_API_KEY", "")),
        "bedrock_profile": os.environ.get("BEDROCK_AWS_PROFILE", "") or None,
        "groq_model": os.environ.get("GROQ_MODEL_ID", "llama-3.3-70b-versatile"),
    }
    if JSON_MODE:
        _output(data)
        return
    console.print(Panel("microids status", subtitle="System overview"))
    _check("Config (.env)", data["env_file"])
    _check("Routines", data["routines_file"])
    _check("HA URL", bool(data["ha_url"]), data["ha_url"] or "not set")
    _check("HA Token", data["ha_token_set"])
    _check("Groq API Key", data["groq_key_set"])
    _check("Groq Model", True, data["groq_model"])
    if data["bedrock_profile"]:
        _check("Bedrock Profile", True, data["bedrock_profile"])


# ═══════════════════════════════════════════════════════════════════════════
# microids logs  (OpenClaw: openclaw logs)
# ═══════════════════════════════════════════════════════════════════════════

@main.command()
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
@click.option("--limit", "-n", default=50, help="Number of lines")
def logs(follow: bool, limit: int) -> None:
    """Tail gateway logs."""
    ssh_key = os.environ.get("MICROIDS_SSH_KEY", os.path.expanduser("~/.ssh/microids_hetzner"))
    server_ip = os.environ.get("MICROIDS_SERVER_IP", "178.156.229.79")
    follow_flag = "-f" if follow else ""
    cmd = f"ssh -i {ssh_key} root@{server_ip} \"journalctl -u microids --no-pager {follow_flag} -n {limit}\""
    console.print(f"[dim]Fetching logs from {server_ip}...[/dim]")
    os.system(cmd)
