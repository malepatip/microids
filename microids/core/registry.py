"""Device registry with capability-based and zone-aware lookup.

Thread-safe via asyncio.Lock (Security Rule S11).
Device IDs stored in canonical format (Security Rule S6).
"""

from __future__ import annotations

import asyncio

from microids.core.security import normalize_device_id
from microids.models import Device, DeviceCategory, DeviceSpec, DeviceStatus


class Registry:
    """In-memory device registry with capability-based and zone-aware lookup.

    Thread-safe via asyncio.Lock (Security Rule S11).
    Device IDs stored in canonical format (Security Rule S6).
    """

    def __init__(self) -> None:
        self._devices: dict[str, Device] = {}
        self._lock = asyncio.Lock()
        # Index device name → device_id for duplicate detection
        self._name_index: dict[str, str] = {}

    async def register(self, spec: DeviceSpec, channel_type: str = "mock") -> Device:
        """Register a device. Validates spec, assigns canonical ID, returns Device.

        - Validates DeviceSpec (calls spec.validate())
        - Generates canonical ID using normalize_device_id(channel_type, spec.name)
        - If device with same name exists: update it (never create duplicates — Property 7)
        - If device with same name exists but different category: reject (Property 21 — category immutability)
        - Returns the registered Device
        """
        if not spec.validate():
            raise ValueError(f"Invalid DeviceSpec: name={spec.name!r}")

        device_id = normalize_device_id(channel_type, spec.name)

        async with self._lock:
            # Check for existing device with same name
            if spec.name in self._name_index:
                existing_id = self._name_index[spec.name]
                existing_device = self._devices[existing_id]

                # Category immutability — Property 21
                if existing_device.spec.category != spec.category:
                    raise ValueError(
                        f"Cannot change category of device {spec.name!r} "
                        f"from {existing_device.spec.category.value} "
                        f"to {spec.category.value}"
                    )

                # Update existing device spec (idempotent registration — Property 7)
                existing_device.spec = spec
                return existing_device

            device = Device(id=device_id, spec=spec)
            self._devices[device_id] = device
            self._name_index[spec.name] = device_id
            return device

    async def unregister(self, device_id: str) -> bool:
        """Remove a device. Returns True if found and removed, False if not found."""
        async with self._lock:
            device = self._devices.pop(device_id, None)
            if device is None:
                return False
            self._name_index.pop(device.spec.name, None)
            return True

    async def get(self, device_id: str) -> Device | None:
        """Get device by ID."""
        async with self._lock:
            return self._devices.get(device_id)

    async def get_by_capability(self, capability: str) -> list[Device]:
        """Return all devices that have the given capability name."""
        async with self._lock:
            return [
                device
                for device in self._devices.values()
                if any(cap.name == capability for cap in device.spec.capabilities)
            ]

    async def get_by_capability_and_zone(
        self, capability: str, zone: str
    ) -> list[Device]:
        """Return devices with matching capability whose reachable_zones include the target zone.

        This is the zone-aware allocation method. A vacuum on 'upstairs' with
        reachable_zones=['upstairs', 'hallway'] will be returned for zone='upstairs'
        but NOT for zone='garden'.
        """
        async with self._lock:
            return [
                device
                for device in self._devices.values()
                if any(cap.name == capability for cap in device.spec.capabilities)
                and zone in device.spec.context.reachable_zones
            ]

    async def get_by_type(self, device_type: str) -> list[Device]:
        """Return all devices matching the given device_type string."""
        async with self._lock:
            return [
                device
                for device in self._devices.values()
                if device.spec.device_type == device_type
            ]

    async def list_all(self) -> list[Device]:
        """Return all registered devices."""
        async with self._lock:
            return list(self._devices.values())

    async def update_status(self, device_id: str, status: DeviceStatus) -> None:
        """Update a device's status. Raises KeyError if device not found."""
        async with self._lock:
            if device_id not in self._devices:
                raise KeyError(f"Device not found: {device_id!r}")
            self._devices[device_id].status = status

    async def get_capabilities_schema(self) -> list[dict]:
        """Return fleet capabilities in a format suitable for LLM prompts.

        Capabilities shared across multiple devices (e.g., turn_on, turn_off)
        are domain-qualified as "{device_type}.{capability}" so the LLM treats
        them as distinct actions per device. Unique capabilities keep their
        original names.

        Includes current device state (inspired by home-llm's 98% accuracy
        approach) so the LLM can reason about what needs to change.
        """
        async with self._lock:
            # First pass: count how many devices have each capability
            cap_count: dict[str, int] = {}
            for device in self._devices.values():
                for cap in device.spec.capabilities:
                    cap_count[cap.name] = cap_count.get(cap.name, 0) + 1

            # Second pass: qualify shared capabilities with device_type prefix
            return [
                {
                    "device_id": device.id,
                    "device_type": device.spec.device_type,
                    "category": device.spec.category.value,
                    "zone": device.spec.context.zone,
                    "state": device.status.value,
                    "capabilities": [
                        f"{device.spec.device_type}.{cap.name}"
                        if cap_count.get(cap.name, 0) > 1
                        else cap.name
                        for cap in device.spec.capabilities
                    ],
                }
                for device in self._devices.values()
            ]
