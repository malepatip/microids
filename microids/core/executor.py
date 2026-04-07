"""Executor — dependency-aware task execution.

Phase 1: sequential execution within each wave.
Phase 2: parallel execution via asyncio.gather within waves.

Dispatches through RecoveryEngine, NOT directly to Router.
"""

from __future__ import annotations

import time
from typing import Any

from microids.core.events import DeviceEvent, EventBus
from microids.core.recovery import RecoveryEngine
from microids.core.registry import Registry
from microids.core.router import Router
from microids.models import (
    Device,
    DeviceStatus,
    ExecutionReport,
    Subtask,
    TaskPlan,
    TaskResult,
)


class AllocationError(Exception):
    """Raised when no suitable device is available for a subtask."""


class Executor:
    """Dependency-aware task executor — sequential for Phase 1."""

    def __init__(
        self,
        registry: Registry,
        router: Router,
        recovery_engine: RecoveryEngine,
        event_bus: EventBus,
    ) -> None:
        self._registry = registry
        self._router = router
        self._recovery_engine = recovery_engine
        self._event_bus = event_bus

    async def execute(self, plan: TaskPlan) -> ExecutionReport:
        """Execute subtasks in topological order (sequential for Phase 1).

        For each wave:
          For each subtask in wave:
            1. Check if dependencies failed → skip with error
            2. allocate_device(subtask) → get IDLE device with matching capability
            3. recovery_engine.execute_with_recovery(subtask, device) → TaskResult
            4. Update device status back to IDLE
            5. Publish task_complete or task_failed event to EventBus
        """
        start = time.monotonic()
        waves = plan.topological_order()
        results: list[TaskResult] = []
        failed_ids: set[str] = set()

        # Build dependency map for failure cascading
        dep_map: dict[str, list[str]] = {s.id: s.dependencies for s in plan.subtasks}

        for wave in waves:
            for subtask in wave:
                # Check if any dependency failed → skip
                if any(dep_id in failed_ids for dep_id in dep_map.get(subtask.id, [])):
                    result = TaskResult(
                        subtask_id=subtask.id,
                        device_id="",
                        success=False,
                        error=f"Skipped: dependency failed",
                    )
                    results.append(result)
                    failed_ids.add(subtask.id)
                    await self._publish_event(subtask, result)
                    continue

                # Allocate device
                try:
                    device = await self.allocate_device(subtask)
                except AllocationError as exc:
                    result = TaskResult(
                        subtask_id=subtask.id,
                        device_id="",
                        success=False,
                        error=str(exc),
                    )
                    results.append(result)
                    failed_ids.add(subtask.id)
                    await self._publish_event(subtask, result)
                    continue

                # Execute through RecoveryEngine
                try:
                    result = await self._recovery_engine.execute_with_recovery(
                        subtask, device
                    )
                except ValueError as exc:
                    # Capability validation failure from security.py
                    result = TaskResult(
                        subtask_id=subtask.id,
                        device_id=device.id,
                        success=False,
                        error=str(exc),
                    )

                results.append(result)

                # Update device status back to IDLE
                await self._registry.update_status(device.id, DeviceStatus.IDLE)

                if not result.success:
                    failed_ids.add(subtask.id)

                await self._publish_event(subtask, result)

        elapsed = time.monotonic() - start
        return ExecutionReport(
            goal=plan.goal,
            results=results,
            total_duration_seconds=elapsed,
        )

    async def allocate_device(self, subtask: Subtask) -> Device:
        """Find best available IDLE device with required capability.

        If subtask.parameters contains a "device_id", target that specific device.
        Otherwise, find the first IDLE device with the required capability.
        Handles domain-qualified capabilities (e.g., "light.turn_off" → "turn_off").
        Sets device to BUSY. Raises AllocationError if none available.
        """
        cap = subtask.required_capability
        # Strip domain prefix if present (e.g., "light.turn_off" → "turn_off")
        base_cap = cap.split(".", 1)[1] if "." in cap else cap

        # Check if LLM specified a target device
        target_id = subtask.parameters.get("device_id", "")
        if target_id:
            for lookup_id in [target_id, f"homeassistant:{target_id}"]:
                device = await self._registry.get(lookup_id)
                if device and device.status == DeviceStatus.IDLE:
                    await self._registry.update_status(device.id, DeviceStatus.BUSY)
                    # Rewrite the capability to the base name for the channel
                    subtask.required_capability = base_cap
                    return device

        # Fallback: find any IDLE device with the capability
        devices = await self._registry.get_by_capability(base_cap)
        for device in devices:
            if device.status == DeviceStatus.IDLE:
                await self._registry.update_status(device.id, DeviceStatus.BUSY)
                subtask.required_capability = base_cap
                return device

        raise AllocationError(
            f"No IDLE device with capability '{cap}'"
        )

    async def _publish_event(
        self, subtask: Subtask, result: TaskResult
    ) -> None:
        """Publish task_complete or task_failed event."""
        event_type = "task_complete" if result.success else "task_failed"
        await self._event_bus.publish(
            DeviceEvent(
                source_device_id=result.device_id or "executor",
                event_type=event_type,
                payload={
                    "subtask_id": subtask.id,
                    "success": result.success,
                    "error": result.error,
                },
            )
        )
