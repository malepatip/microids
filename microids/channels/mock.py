"""MockChannel — simulated device channel for development and testing.

Supports two modes:
1. Basic mode (original): instant responses for unit tests
2. Simulator mode: realistic state transitions with timing for the web UI

No BaseChannel ABC yet — just a concrete class with the right method signatures.
The ABC will be extracted in Phase 3 when the second channel (HA) demands it.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Optional

from microids.core.security import validate_command
from microids.models import (
    Device,
    DeviceCapability,
    DeviceCategory,
    DeviceContext,
    DeviceSpec,
    DeviceStatus,
)


# Simulator device state model — richer than DeviceStatus for UI animations
SIMULATOR_DEVICES = {
    "garage": {
        "name": "Garage Door",
        "emoji": "🚪",
        "type": "cover",
        "category": DeviceCategory.CONTROLLER,
        "zone": "garage",
        "capabilities": ["open", "close", "stop"],
        "states": {
            "closed": {"label": "Closed", "color": "#6b7280"},
            "opening": {"label": "Opening...", "color": "#f59e0b", "transient": True, "next": "open", "duration": 2.0},
            "open": {"label": "Open", "color": "#10b981"},
            "closing": {"label": "Closing...", "color": "#f59e0b", "transient": True, "next": "closed", "duration": 2.0},
        },
        "actions": {
            "open": {"from": ["closed"], "transition": "opening"},
            "close": {"from": ["open"], "transition": "closing"},
            "stop": {"from": ["opening", "closing"], "transition": "stopped"},
        },
        "initial": "closed",
    },
    "vacuum": {
        "name": "Robot Vacuum",
        "emoji": "🤖",
        "type": "vacuum",
        "category": DeviceCategory.ACTUATOR,
        "zone": "downstairs",
        "capabilities": ["vacuum", "return_to_base", "stop", "pause", "start", "mop"],
        "states": {
            "docked": {"label": "Docked", "color": "#6b7280"},
            "undocking": {"label": "Undocking...", "color": "#f59e0b", "transient": True, "next": "cleaning", "duration": 1.5},
            "cleaning": {"label": "Cleaning", "color": "#3b82f6", "animated": True},
            "mopping": {"label": "Mopping", "color": "#8b5cf6", "animated": True},
            "returning": {"label": "Returning...", "color": "#f59e0b", "transient": True, "next": "docked", "duration": 3.0},
            "paused": {"label": "Paused", "color": "#eab308"},
        },
        "actions": {
            "vacuum": {"from": ["docked", "paused"], "transition": "undocking"},
            "start": {"from": ["docked", "paused"], "transition": "undocking"},
            "mop": {"from": ["docked", "paused", "cleaning"], "transition": "mopping"},
            "return_to_base": {"from": ["cleaning", "mopping", "paused"], "transition": "returning"},
            "dock": {"from": ["cleaning", "mopping", "paused"], "transition": "returning"},
            "pause": {"from": ["cleaning", "mopping"], "transition": "paused"},
            "stop": {"from": ["cleaning", "mopping"], "transition": "docked"},
        },
        "initial": "docked",
    },
    "sprinkler": {
        "name": "Garden Sprinkler",
        "emoji": "💧",
        "type": "sprinkler",
        "category": DeviceCategory.CONTROLLER,
        "zone": "garden",
        "capabilities": ["water", "schedule", "turn_on", "turn_off"],
        "states": {
            "off": {"label": "Off", "color": "#6b7280"},
            "starting": {"label": "Starting...", "color": "#06b6d4", "transient": True, "next": "watering", "duration": 1.0},
            "watering": {"label": "Watering", "color": "#06b6d4", "animated": True},
            "stopping": {"label": "Stopping...", "color": "#f59e0b", "transient": True, "next": "off", "duration": 1.0},
        },
        "actions": {
            "water": {"from": ["off"], "transition": "starting"},
            "turn_on": {"from": ["off"], "transition": "starting"},
            "schedule": {"from": ["off", "watering"], "transition": "watering"},
            "turn_off": {"from": ["watering"], "transition": "stopping"},
        },
        "initial": "off",
    },
    "camera": {
        "name": "Front Camera",
        "emoji": "📷",
        "type": "camera",
        "category": DeviceCategory.SENSOR,
        "zone": "front_yard",
        "capabilities": ["snapshot", "detect_motion", "stop"],
        "states": {
            "idle": {"label": "Idle", "color": "#6b7280"},
            "recording": {"label": "Recording", "color": "#ef4444", "animated": True},
            "snapshot": {"label": "Snapshot!", "color": "#f59e0b", "transient": True, "next": "idle", "duration": 1.5},
        },
        "actions": {
            "snapshot": {"from": ["idle", "recording"], "transition": "snapshot"},
            "detect_motion": {"from": ["idle"], "transition": "recording"},
            "stop": {"from": ["recording"], "transition": "idle"},
        },
        "initial": "idle",
    },
    "lights": {
        "name": "Living Room Lights",
        "emoji": "💡",
        "type": "light",
        "category": DeviceCategory.CONTROLLER,
        "zone": "living_room",
        "capabilities": ["turn_on", "turn_off", "toggle"],
        "states": {
            "off": {"label": "Off", "color": "#6b7280"},
            "on": {"label": "On", "color": "#fbbf24"},
        },
        "actions": {
            "turn_on": {"from": ["off"], "transition": "on"},
            "turn_off": {"from": ["on"], "transition": "off"},
            "toggle": {"from": ["on", "off"], "transition": "_toggle"},
        },
        "initial": "off",
    },
}


class MockChannel:
    """Simulated device channel for development/testing.

    In simulator mode, devices have realistic state machines with
    timed transitions. State changes are broadcast to listeners
    for real-time UI updates.
    """

    def __init__(self, simulator: bool = False) -> None:
        self._devices: dict[str, Device] = {}
        self._simulator = simulator
        # Simulator state: device_key → current state string
        self._sim_states: dict[str, str] = {}
        # Listeners for state changes (for WebSocket broadcast)
        self._state_listeners: list[Callable] = []
        # Background tasks for transient state transitions
        self._transition_tasks: dict[str, asyncio.Task] = {}

    def on_state_change(self, callback: Callable) -> None:
        """Register a callback for state changes: callback(device_key, new_state, state_info)."""
        self._state_listeners.append(callback)

    async def _notify_state_change(self, device_key: str, state: str) -> None:
        """Notify all listeners of a state change."""
        device_def = SIMULATOR_DEVICES.get(device_key, {})
        state_info = device_def.get("states", {}).get(state, {})
        for listener in self._state_listeners:
            try:
                result = listener(device_key, state, state_info)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

    async def connect(self, config: dict[str, Any]) -> None:
        """No-op for mock."""

    async def disconnect(self) -> None:
        """Cancel any pending transition tasks."""
        for task in self._transition_tasks.values():
            task.cancel()
        self._transition_tasks.clear()

    async def discover_devices(self) -> list[DeviceSpec]:
        """Return simulated devices. In simulator mode, uses the richer device set."""
        if self._simulator:
            return await self._discover_simulator_devices()
        return await self._discover_basic_devices()

    async def _discover_basic_devices(self) -> list[DeviceSpec]:
        """Original basic device set for unit tests."""
        specs = [
            DeviceSpec(
                name="vacuum",
                device_type="vacuum",
                category=DeviceCategory.ACTUATOR,
                capabilities=[
                    DeviceCapability(name="vacuum"),
                    DeviceCapability(name="mop"),
                    DeviceCapability(name="dock"),
                ],
                context=DeviceContext(
                    zone="downstairs",
                    reachable_zones=["downstairs", "hallway"],
                ),
            ),
            DeviceSpec(
                name="camera",
                device_type="camera",
                category=DeviceCategory.SENSOR,
                capabilities=[
                    DeviceCapability(name="snapshot"),
                    DeviceCapability(name="detect_motion"),
                ],
                context=DeviceContext(
                    zone="front_yard",
                    reachable_zones=["front_yard", "driveway"],
                ),
            ),
            DeviceSpec(
                name="sprinkler",
                device_type="sprinkler",
                category=DeviceCategory.CONTROLLER,
                capabilities=[
                    DeviceCapability(name="water"),
                    DeviceCapability(name="schedule"),
                ],
                context=DeviceContext(
                    zone="garden",
                    reachable_zones=["garden"],
                ),
            ),
            DeviceSpec(
                name="mop",
                device_type="mop",
                category=DeviceCategory.ACTUATOR,
                capabilities=[
                    DeviceCapability(name="mop"),
                ],
                context=DeviceContext(
                    zone="downstairs",
                    reachable_zones=["downstairs"],
                ),
            ),
        ]

        self._devices.clear()
        for spec in specs:
            device = Device(id=spec.name, spec=spec, status=DeviceStatus.IDLE)
            self._devices[spec.name] = device
            self._devices[f"mock:{spec.name}"] = device

        return specs

    async def _discover_simulator_devices(self) -> list[DeviceSpec]:
        """Rich device set for the simulator UI."""
        specs: list[DeviceSpec] = []
        self._devices.clear()
        self._sim_states.clear()

        for key, dev_def in SIMULATOR_DEVICES.items():
            caps = [DeviceCapability(name=c) for c in dev_def["capabilities"]]
            spec = DeviceSpec(
                name=key,
                device_type=dev_def["type"],
                category=dev_def["category"],
                capabilities=caps,
                context=DeviceContext(
                    zone=dev_def["zone"],
                    reachable_zones=[dev_def["zone"]],
                ),
                metadata={
                    "friendly_name": dev_def["name"],
                    "emoji": dev_def["emoji"],
                },
            )
            specs.append(spec)

            device = Device(id=key, spec=spec, status=DeviceStatus.IDLE)
            self._devices[key] = device
            self._devices[f"mock:{key}"] = device
            self._sim_states[key] = dev_def["initial"]

        return specs

    async def send_command(
        self, device_id: str, action: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Simulate command execution with capability validation.

        In simulator mode, triggers state machine transitions with timing.
        """
        # Normalize device_id — strip channel prefix
        bare_id = device_id.replace("mock:", "")
        device = self._devices.get(device_id) or self._devices.get(bare_id)
        if device is None:
            raise ValueError(f"Unknown device: {device_id}")

        if not validate_command(device, action):
            raise ValueError(
                f"Action '{action}' is not a declared capability of device '{device_id}'"
            )

        if self._simulator:
            return await self._sim_execute(bare_id, action, parameters)

        await asyncio.sleep(0.1)
        return {"status": "ok", "device_id": device_id, "action": action}

    async def _sim_execute(
        self, device_key: str, action: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a command through the simulator state machine.

        Adds a realistic delay so the UI can show transitions happening.
        """
        dev_def = SIMULATOR_DEVICES.get(device_key)
        if not dev_def:
            return {"status": "error", "reason": f"No simulator def for {device_key}"}

        action_def = dev_def["actions"].get(action)
        if not action_def:
            # Action is a valid capability but no state transition defined — just ack
            await asyncio.sleep(0.3)
            return {"status": "ok", "device_id": device_key, "action": action}

        current = self._sim_states.get(device_key, dev_def["initial"])

        # Check if already in target state
        transition = action_def["transition"]
        if transition == "_toggle":
            transition = "off" if current == "on" else "on"

        valid_from = action_def.get("from", [])
        if valid_from and current not in valid_from:
            return {
                "status": "skipped",
                "reason": f"already {current}",
                "device_id": device_key,
            }

        # Cancel any pending transition for this device
        if device_key in self._transition_tasks:
            self._transition_tasks[device_key].cancel()
            del self._transition_tasks[device_key]

        # Small delay so the UI sees each command arrive sequentially
        await asyncio.sleep(0.4)

        # Apply the transition
        self._sim_states[device_key] = transition
        await self._notify_state_change(device_key, transition)

        # If this is a transient state, schedule the next transition
        state_def = dev_def["states"].get(transition, {})
        if state_def.get("transient"):
            next_state = state_def["next"]
            duration = state_def.get("duration", 2.0)
            task = asyncio.create_task(
                self._delayed_transition(device_key, next_state, duration)
            )
            self._transition_tasks[device_key] = task

        return {"status": "ok", "device_id": device_key, "action": action, "state": transition}

    async def _sim_execute(
        self, device_key: str, action: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a command through the simulator state machine."""
        dev_def = SIMULATOR_DEVICES.get(device_key)
        if not dev_def:
            return {"status": "error", "reason": f"No simulator def for {device_key}"}

        action_def = dev_def["actions"].get(action)
        if not action_def:
            # Action is a valid capability but no state transition defined — just ack
            return {"status": "ok", "device_id": device_key, "action": action}

        current = self._sim_states.get(device_key, dev_def["initial"])

        # Check if already in target state
        transition = action_def["transition"]
        if transition == "_toggle":
            transition = "off" if current == "on" else "on"

        valid_from = action_def.get("from", [])
        if valid_from and current not in valid_from:
            # Already in desired state or incompatible — skip
            return {
                "status": "skipped",
                "reason": f"already {current}",
                "device_id": device_key,
            }

        # Cancel any pending transition for this device
        if device_key in self._transition_tasks:
            self._transition_tasks[device_key].cancel()
            del self._transition_tasks[device_key]

        # Apply the transition
        self._sim_states[device_key] = transition
        await self._notify_state_change(device_key, transition)

        # If this is a transient state, schedule the next transition
        state_def = dev_def["states"].get(transition, {})
        if state_def.get("transient"):
            next_state = state_def["next"]
            duration = state_def.get("duration", 2.0)
            task = asyncio.create_task(
                self._delayed_transition(device_key, next_state, duration)
            )
            self._transition_tasks[device_key] = task

        return {"status": "ok", "device_id": device_key, "action": action, "state": transition}

    async def _delayed_transition(self, device_key: str, next_state: str, delay: float) -> None:
        """After a delay, transition to the next state (for transient states)."""
        try:
            await asyncio.sleep(delay)
            self._sim_states[device_key] = next_state
            await self._notify_state_change(device_key, next_state)
            self._transition_tasks.pop(device_key, None)
        except asyncio.CancelledError:
            pass

    def get_simulator_state(self) -> dict[str, Any]:
        """Return full simulator state for the UI."""
        result = {}
        for key, dev_def in SIMULATOR_DEVICES.items():
            current = self._sim_states.get(key, dev_def["initial"])
            state_info = dev_def["states"].get(current, {})
            result[key] = {
                "name": dev_def["name"],
                "emoji": dev_def["emoji"],
                "type": dev_def["type"],
                "zone": dev_def["zone"],
                "state": current,
                "label": state_info.get("label", current),
                "color": state_info.get("color", "#6b7280"),
                "animated": state_info.get("animated", False),
                "transient": state_info.get("transient", False),
            }
        return result

    async def get_device_status(self, device_id: str) -> DeviceStatus:
        """Return simulated device status."""
        bare_id = device_id.replace("mock:", "")
        device = self._devices.get(device_id) or self._devices.get(bare_id)
        if device is None:
            raise ValueError(f"Unknown device: {device_id}")
        return device.status

    async def wait_for_state(
        self,
        device_id: str,
        expected_state: DeviceStatus,
        timeout_seconds: float = 30.0,
        poll_interval: float = 2.0,
    ) -> bool:
        """Mock always returns True immediately (simulated instant transition)."""
        return True

    async def discover_room_zones(self, device_id: str) -> list[dict[str, Any]]:
        """Return empty list (mock doesn't have real rooms)."""
        return []
