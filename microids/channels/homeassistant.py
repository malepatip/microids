"""Home Assistant Channel — real device communication via HA REST API.

Early implementation (pre-Phase 3) to enable real hardware testing
from the CLI without throwaway scripts.

Usage via CLI:
    microids goal "Clean the house" --channel ha
    microids devices --channel ha
    microids doctor
"""

from __future__ import annotations

import os
from typing import Any, Optional

import aiohttp

from microids.core.security import validate_command
from microids.models import (
    Device,
    DeviceCapability,
    DeviceCategory,
    DeviceContext,
    DeviceSpec,
    DeviceStatus,
)

# HA domain → DeviceCategory mapping
_DOMAIN_CATEGORY: dict[str, DeviceCategory] = {
    "vacuum": DeviceCategory.ACTUATOR,
    "light": DeviceCategory.CONTROLLER,
    "switch": DeviceCategory.CONTROLLER,
    "climate": DeviceCategory.CONTROLLER,
    "fan": DeviceCategory.CONTROLLER,
    "cover": DeviceCategory.CONTROLLER,
    "sensor": DeviceCategory.SENSOR,
    "binary_sensor": DeviceCategory.SENSOR,
    "camera": DeviceCategory.SENSOR,
}

# HA domain → default capabilities
_DOMAIN_CAPABILITIES: dict[str, list[str]] = {
    "vacuum": ["vacuum", "return_to_base", "stop", "pause", "start"],
    "light": ["turn_on", "turn_off", "toggle"],
    "switch": ["turn_on", "turn_off", "toggle"],
    "climate": ["set_temperature", "turn_on", "turn_off"],
    "fan": ["turn_on", "turn_off", "set_speed"],
    "cover": ["open", "close", "stop"],
    "camera": ["snapshot", "detect_motion"],
}


class HomeAssistantChannel:
    """Channel that talks to Home Assistant via REST API.

    Discovers real devices, sends real commands, reads real status.
    Capability validation enforced on every command (Security Rule S2).
    """

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None) -> None:
        self._url = url or os.environ.get("HA_URL", "http://localhost:8123")
        self._token = token or os.environ.get("HA_TOKEN", "")
        self._cf_client_id = os.environ.get("CF_ACCESS_CLIENT_ID", "")
        self._cf_client_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET", "")
        self._session: Optional[aiohttp.ClientSession] = None
        self._devices: dict[str, Device] = {}

    def _build_headers(self) -> dict[str, str]:
        """Build request headers including HA auth and optional Cloudflare Access."""
        headers = {"Authorization": f"Bearer {self._token}"}
        if self._cf_client_id and self._cf_client_secret:
            headers["CF-Access-Client-Id"] = self._cf_client_id
            headers["CF-Access-Client-Secret"] = self._cf_client_secret
        return headers

    async def connect(self, config: dict[str, Any]) -> None:
        """Connect to Home Assistant and verify."""
        url = config.get("url", self._url)
        token = config.get("token", self._token)
        self._url = url
        self._token = token

        if not self._token:
            raise RuntimeError(
                "No HA token. Set HA_TOKEN env var or pass token in config.\n"
                "Get one at: http://localhost:8123/profile → Long-Lived Access Tokens"
            )

        self._session = aiohttp.ClientSession(
            headers=self._build_headers(),
        )

        try:
            async with self._session.get(f"{self._url}/api/") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    version = data.get("version", "unknown")
                    return
                elif resp.status == 401:
                    raise RuntimeError("HA token is invalid or expired")
                else:
                    raise RuntimeError(f"HA returned HTTP {resp.status}")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Cannot connect to HA at {self._url}: {e}")

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def discover_devices(self) -> list[DeviceSpec]:
        """Discover real devices from Home Assistant.
        
        Groups HA entities by physical device. Multiple entities (vacuum.robovac,
        switch.robovac_auto_empty, etc.) belonging to the same physical device
        are merged into one DeviceSpec with combined capabilities.
        """
        if not self._session:
            raise RuntimeError("Not connected. Call connect() first.")

        async with self._session.get(f"{self._url}/api/states") as resp:
            states = await resp.json()

        # Group entities by device prefix (e.g., "robovac" from "vacuum.robovac")
        device_groups: dict[str, list[dict]] = {}
        for state in states:
            entity_id: str = state["entity_id"]
            domain = entity_id.split(".")[0]
            if domain not in _DOMAIN_CATEGORY:
                continue

            # Skip HA system/internal entities that aren't real devices
            object_id = entity_id.split(".", 1)[1]
            _SYSTEM_PREFIXES = ("backup_", "sun_", "hacs_", "homeassistant_")
            if any(object_id.startswith(p) for p in _SYSTEM_PREFIXES):
                continue

            # Extract device name: "vacuum.robovac" → "robovac", "switch.robovac_auto_empty" → "robovac"
            object_id = entity_id.split(".", 1)[1]
            attrs = state.get("attributes", {})
            friendly = attrs.get("friendly_name", object_id)

            # Find the root device name by checking if this entity's object_id
            # starts with a known device prefix
            device_key = self._find_device_key(object_id, device_groups)
            if device_key is None:
                device_key = object_id

            if device_key not in device_groups:
                device_groups[device_key] = []
            device_groups[device_key].append({
                "entity_id": entity_id,
                "domain": domain,
                "object_id": object_id,
                "state": state["state"],
                "attrs": attrs,
                "friendly_name": friendly,
            })

        # Build one DeviceSpec per physical device
        specs: list[DeviceSpec] = []
        self._devices.clear()

        for device_key, entities in device_groups.items():
            # Find the primary entity (vacuum > switch > sensor > light)
            primary = self._pick_primary(entities)
            domain = primary["domain"]
            entity_id = primary["entity_id"]
            attrs = primary["attrs"]
            friendly = primary["friendly_name"]

            # Merge capabilities from all entities in this device group
            all_caps: list[DeviceCapability] = []
            for ent in entities:
                ent_domain = ent["domain"]
                for cap_name in _DOMAIN_CAPABILITIES.get(ent_domain, []):
                    if not any(c.name == cap_name for c in all_caps):
                        all_caps.append(DeviceCapability(name=cap_name))

                # Add entity-specific capabilities from switches
                if ent["domain"] == "switch" and ent["object_id"] != device_key:
                    suffix = ent["object_id"].replace(device_key + "_", "")
                    if suffix and not any(c.name == suffix for c in all_caps):
                        all_caps.append(DeviceCapability(name=suffix))

            category = _DOMAIN_CATEGORY.get(domain, DeviceCategory.CONTROLLER)
            zone = attrs.get("area", "home") or "home"

            spec = DeviceSpec(
                name=entity_id,
                device_type=domain,
                category=category,
                capabilities=all_caps if all_caps else [DeviceCapability(name="unknown")],
                context=DeviceContext(
                    zone=zone,
                    reachable_zones=[zone],
                    is_mobile=(domain == "vacuum"),
                    battery_level=_parse_battery(attrs),
                ),
                channel_config={"type": "homeassistant"},
                metadata={
                    "friendly_name": friendly,
                    "ha_state": primary["state"],
                    "entity_count": len(entities),
                    "entities": [e["entity_id"] for e in entities],
                },
            )
            specs.append(spec)

            device = Device(id=entity_id, spec=spec, status=_map_state(primary["state"]))
            self._devices[entity_id] = device
            self._devices[f"homeassistant:{entity_id}"] = device
            # Also index by all child entity IDs so send_command works with any of them
            for ent in entities:
                self._devices[ent["entity_id"]] = device
                self._devices[f"homeassistant:{ent['entity_id']}"] = device

        return specs

    def _find_device_key(self, object_id: str, existing_groups: dict) -> str | None:
        """Check if object_id belongs to an existing device group."""
        for key in existing_groups:
            if object_id.startswith(key + "_") or object_id == key:
                return key
            if key.startswith(object_id + "_") or key == object_id:
                return object_id
        return None

    def _pick_primary(self, entities: list[dict]) -> dict:
        """Pick the primary entity from a group (vacuum > climate > light > switch > sensor)."""
        priority = ["vacuum", "climate", "cover", "fan", "light", "switch", "camera", "sensor", "binary_sensor"]
        for domain in priority:
            for ent in entities:
                if ent["domain"] == domain:
                    return ent
        return entities[0]

    async def send_command(
        self, device_id: str, action: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Send a command to a device via HA REST API.

        Security Rule S2: validates capability before dispatch.
        Pre-checks device state to avoid redundant/harmful commands.
        """
        device = self._devices.get(device_id)
        if device is None:
            raise ValueError(f"Unknown device: {device_id}")

        if not validate_command(device, action):
            raise ValueError(
                f"Action '{action}' is not a declared capability of '{device_id}'"
            )

        entity_id = device.spec.name
        domain = entity_id.split(".")[0]

        # Pre-command state check — skip if device is already in desired state
        skip_reason = await self._check_already_in_state(entity_id, domain, action)
        if skip_reason:
            return {"status": "skipped", "reason": skip_reason, "entity_id": entity_id}

        # Map capability names to HA service names
        _SERVICE_MAP = {
            "vacuum": "start",
            "dock": "return_to_base",
            "open": "open_cover",
            "close": "close_cover",
        }
        service = _SERVICE_MAP.get(action, action)
        payload: dict[str, Any] = {"entity_id": entity_id}
        # Add parameters but strip internal fields (device_id is for Executor, not HA)
        clean_params = {k: v for k, v in parameters.items() if k != "device_id"}
        payload.update(clean_params)

        async with self._session.post(
            f"{self._url}/api/services/{domain}/{service}",
            json=payload,
        ) as resp:
            if resp.status == 200:
                return {"status": "ok", "service": f"{domain}.{service}", "entity_id": entity_id}
            else:
                text = await resp.text()
                raise RuntimeError(f"HA API error {resp.status}: {text}")

    async def _check_already_in_state(
        self, entity_id: str, domain: str, action: str,
    ) -> str | None:
        """Check if device is already in the target state. Returns skip reason or None."""
        # Map action → state that means "already done"
        _ALREADY_DONE: dict[str, dict[str, list[str]]] = {
            "cover": {
                "open": ["open", "opening"],
                "close": ["closed", "closing"],
            },
            "vacuum": {
                "return_to_base": ["docked"],
                "start": ["cleaning"],
                "stop": ["docked", "idle"],
            },
            "switch": {
                "turn_on": ["on"],
                "turn_off": ["off"],
            },
        }

        target_states = _ALREADY_DONE.get(domain, {}).get(action)
        if not target_states:
            return None

        try:
            async with self._session.get(f"{self._url}/api/states/{entity_id}") as resp:
                data = await resp.json()
                current = data.get("state", "unknown")
                if current in target_states:
                    return f"already {current}"
        except Exception:
            pass  # If state check fails, proceed with command anyway

        return None

    async def get_device_status(self, device_id: str) -> DeviceStatus:
        device = self._devices.get(device_id)
        if device is None:
            raise ValueError(f"Unknown device: {device_id}")

        entity_id = device.spec.name
        async with self._session.get(f"{self._url}/api/states/{entity_id}") as resp:
            data = await resp.json()
            return _map_state(data.get("state", "unknown"))

    async def wait_for_state(
        self,
        device_id: str,
        expected_state: DeviceStatus,
        timeout_seconds: float = 30.0,
        poll_interval: float = 2.0,
    ) -> bool:
        """Poll until device reaches expected state or timeout."""
        import asyncio
        import time

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            current = await self.get_device_status(device_id)
            if current == expected_state:
                return True
            await asyncio.sleep(poll_interval)
        return False

    async def discover_room_zones(self, device_id: str) -> list[dict[str, Any]]:
        """Read room/zone data from device attributes if available."""
        device = self._devices.get(device_id)
        if device is None:
            return []

        entity_id = device.spec.name
        async with self._session.get(f"{self._url}/api/states/{entity_id}") as resp:
            data = await resp.json()
            attrs = data.get("attributes", {})

            rooms = attrs.get("rooms", attrs.get("segments", []))
            if isinstance(rooms, list):
                return [{"id": str(i), "name": str(r)} for i, r in enumerate(rooms)]
            elif isinstance(rooms, dict):
                return [{"id": k, "name": v} for k, v in rooms.items()]
            return []


def _map_state(ha_state: str) -> DeviceStatus:
    """Map HA state string to DeviceStatus."""
    mapping = {
        "on": DeviceStatus.IDLE,
        "off": DeviceStatus.OFFLINE,
        "docked": DeviceStatus.IDLE,
        "cleaning": DeviceStatus.BUSY,
        "returning": DeviceStatus.BUSY,
        "paused": DeviceStatus.IDLE,
        "idle": DeviceStatus.IDLE,
        "error": DeviceStatus.ERROR,
        "unavailable": DeviceStatus.OFFLINE,
        "unknown": DeviceStatus.OFFLINE,
    }
    return mapping.get(ha_state, DeviceStatus.IDLE)


def _parse_battery(attrs: dict) -> Optional[float]:
    """Parse battery level from HA attributes."""
    level = attrs.get("battery_level")
    if level is not None:
        try:
            val = float(level)
            return val / 100.0 if val > 1.0 else val
        except (ValueError, TypeError):
            pass
    return None
