"""Recovery Engine — composable pipeline of recovery strategies.

Phase 1: only RetryStrategy wired into the default pipeline.
Phase 2: full 8-strategy pipeline with category-specific defaults.

Architecture position:
    Executor → RecoveryEngine → Router → Channel → Device

Security Rules enforced:
    S2 — validate_command() before every command dispatch
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from microids.core.events import DeviceEvent, EventBus
from microids.core.security import validate_command
from microids.models import (
    Device,
    DeviceCategory,
    DeviceStatus,
    RecoveryAction,
    RecoveryContext,
    Subtask,
    TaskResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy ABC
# ---------------------------------------------------------------------------


class RecoveryStrategy(ABC):
    """Base class for composable recovery strategies."""

    @abstractmethod
    async def attempt(self, context: RecoveryContext) -> RecoveryAction:
        """Evaluate context and return the recovery action to take."""
        ...


# ---------------------------------------------------------------------------
# RetryStrategy (Phase 1)
# ---------------------------------------------------------------------------


class RetryStrategy(RecoveryStrategy):
    """Retry the failed command with configurable backoff."""

    def __init__(
        self,
        max_attempts: int = 3,
        backoff: str = "exponential",
        base_delay: float = 30.0,
    ) -> None:
        self.max_attempts = max_attempts
        self.backoff = backoff
        self.base_delay = base_delay

    async def attempt(self, context: RecoveryContext) -> RecoveryAction:
        """Return RETRY if under max_attempts, else FAIL."""
        if context.attempt_number < self.max_attempts:
            return RecoveryAction.RETRY
        return RecoveryAction.FAIL

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number."""
        if self.backoff == "exponential":
            return self.base_delay * (2 ** attempt)
        elif self.backoff == "linear":
            return self.base_delay * (attempt + 1)
        else:  # fixed
            return self.base_delay


# ---------------------------------------------------------------------------
# Category Pipeline Defaults (Phase 1: all get RetryStrategy only)
# ---------------------------------------------------------------------------


def _default_pipelines() -> dict[DeviceCategory, list[RecoveryStrategy]]:
    """Phase 1: all categories get a simple RetryStrategy."""
    return {
        DeviceCategory.ACTUATOR: [RetryStrategy(max_attempts=3)],
        DeviceCategory.SENSOR: [RetryStrategy(max_attempts=3)],
        DeviceCategory.CONTROLLER: [RetryStrategy(max_attempts=3)],
        DeviceCategory.MOBILE_AUTONOMOUS: [RetryStrategy(max_attempts=3)],
    }


# ---------------------------------------------------------------------------
# RecoveryEngine
# ---------------------------------------------------------------------------


class RecoveryEngine:
    """Category-aware recovery pipeline engine.

    Wraps every subtask execution with the device's category-specific
    recovery pipeline. On failure, strategies execute in declared order.
    """

    def __init__(
        self,
        registry: Any,
        router: Any,
        event_bus: EventBus,
        config: Optional[Any] = None,
    ) -> None:
        self._registry = registry
        self._router = router
        self._event_bus = event_bus
        self._config = config
        self._pipelines = _default_pipelines()

    async def execute_with_recovery(
        self, subtask: Subtask, device: Device
    ) -> TaskResult:
        """Execute subtask with category-specific recovery pipeline.

        1. First attempt: validate_command, route to channel, send_command
        2. On success: return TaskResult(success=True)
        3. On failure: build RecoveryContext, walk pipeline
        4. For Phase 1: pipeline is just [RetryStrategy] for all categories
        5. Publish recovery_action events to EventBus
        """
        start = time.monotonic()
        action = subtask.required_capability
        channel = self._router.route(device)

        # --- First attempt ---
        result = await self._try_command(device, action, subtask.parameters, channel)
        if result is not None:
            elapsed = time.monotonic() - start
            return TaskResult(
                subtask_id=subtask.id,
                device_id=device.id,
                success=True,
                response=result,
                duration_seconds=elapsed,
            )

        # --- Recovery pipeline ---
        pipeline = self.get_pipeline(device)
        attempt = 1
        last_error: Optional[str] = None

        for strategy in pipeline:
            context = RecoveryContext(
                subtask=subtask,
                device=device,
                error=Exception(f"Command failed on attempt {attempt}"),
                attempt_number=attempt,
                category=device.spec.category,
                zone=device.spec.context.zone,
                fleet_state={},
                previous_actions=[],
            )

            recovery_action = await strategy.attempt(context)

            # Publish recovery event
            await self._event_bus.publish(
                DeviceEvent(
                    source_device_id=device.id,
                    event_type="recovery_action",
                    payload={
                        "subtask_id": subtask.id,
                        "action": recovery_action.value,
                        "attempt": attempt,
                    },
                )
            )

            if recovery_action == RecoveryAction.RETRY:
                # Calculate delay from strategy if it supports it
                delay = 0.0
                if isinstance(strategy, RetryStrategy):
                    delay = strategy.get_delay(attempt)
                if delay > 0:
                    await asyncio.sleep(min(delay, 0.1))  # cap for tests

                result = await self._try_command(
                    device, action, subtask.parameters, channel
                )
                if result is not None:
                    elapsed = time.monotonic() - start
                    return TaskResult(
                        subtask_id=subtask.id,
                        device_id=device.id,
                        success=True,
                        response=result,
                        duration_seconds=elapsed,
                    )
                attempt += 1
                last_error = f"Retry {attempt} failed"
                continue

            elif recovery_action == RecoveryAction.FAIL:
                last_error = f"Recovery exhausted after {attempt} attempts"
                break

            elif recovery_action == RecoveryAction.SUSPEND:
                elapsed = time.monotonic() - start
                return TaskResult(
                    subtask_id=subtask.id,
                    device_id=device.id,
                    success=False,
                    error="SUSPENDED",
                    duration_seconds=elapsed,
                )

            else:
                # Other actions (Phase 2)
                last_error = f"Unhandled recovery action: {recovery_action.value}"
                break

        elapsed = time.monotonic() - start
        return TaskResult(
            subtask_id=subtask.id,
            device_id=device.id,
            success=False,
            error=last_error or "Command failed",
            duration_seconds=elapsed,
        )

    async def _try_command(
        self,
        device: Device,
        action: str,
        parameters: dict[str, Any],
        channel: Any,
    ) -> Optional[dict[str, Any]]:
        """Attempt a single command dispatch. Returns result dict or None on failure.

        Calls validate_command() before dispatch (Security Rule S2).
        """
        if not validate_command(device, action):
            raise ValueError(
                f"Action '{action}' is not a declared capability of device '{device.id}'"
            )

        try:
            result = await channel.send_command(device.id, action, parameters)
            return result
        except Exception as exc:
            logger.warning("Command failed for %s action=%s: %s", device.id, action, exc)
            return None

    def get_pipeline(self, device: Device) -> list[RecoveryStrategy]:
        """Get recovery pipeline for device category.

        Phase 1: all categories get [RetryStrategy(max_attempts=3)].
        Phase 2: three-layer override model.
        """
        return self._pipelines.get(
            device.spec.category,
            [RetryStrategy(max_attempts=3)],
        )
