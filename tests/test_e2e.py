"""End-to-end integration tests for the microids control plane.

Submits goals through Gateway with MockChannel + MockAgent + RecoveryEngine
and verifies the full coordination loop.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from microids.agents.mock import MockAgent
from microids.channels.mock import MockChannel
from microids.core.events import EventBus
from microids.core.gateway import Gateway
from microids.core.recovery import RecoveryEngine
from microids.core.registry import Registry
from microids.core.router import Router
from microids.models import DeviceStatus, GoalStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_gateway(
    channel: MockChannel | None = None,
    agent: MockAgent | None = None,
) -> Gateway:
    """Wire up a full Gateway with all components."""
    channel = channel or MockChannel()
    agent = agent or MockAgent()
    registry = Registry()
    event_bus = EventBus()
    router = Router(default_channel=channel)
    recovery_engine = RecoveryEngine(
        registry=registry,
        router=router,
        event_bus=event_bus,
    )
    gateway = Gateway(
        agent=agent,
        router=router,
        registry=registry,
        event_bus=event_bus,
        recovery_engine=recovery_engine,
    )
    await gateway.start()
    return gateway


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_goal_execution():
    """Submit a goal, verify COMPLETED status, all devices IDLE after."""
    gateway = await _make_gateway()
    try:
        result = await gateway.goal("Clean the house")

        assert result.status == GoalStatus.COMPLETED
        assert result.execution_report is not None
        assert result.execution_report.success
        assert result.duration_seconds >= 0

        # Verify all devices back to IDLE
        devices = await gateway._registry.list_all()
        for device in devices:
            assert device.status == DeviceStatus.IDLE, (
                f"Device {device.id} should be IDLE, got {device.status}"
            )

        # Verify EventBus has task_complete events
        events = gateway._event_bus.get_history(event_type="task_complete")
        assert len(events) > 0, "Expected at least one task_complete event"

        # Each successful subtask should have a corresponding event
        successful_ids = {
            r.subtask_id
            for r in result.execution_report.results
            if r.success
        }
        event_ids = {e.payload["subtask_id"] for e in events}
        assert successful_ids == event_ids
    finally:
        await gateway.stop()


@pytest.mark.asyncio
async def test_retry_recovery():
    """Simulate transient failure, verify RetryStrategy recovers."""

    class FailOnceMockChannel(MockChannel):
        """MockChannel that fails the first send_command call, then succeeds."""

        def __init__(self) -> None:
            super().__init__()
            self._call_count: dict[str, int] = {}

        async def send_command(
            self, device_id: str, action: str, parameters: dict[str, Any]
        ) -> dict[str, Any]:
            key = f"{device_id}:{action}"
            self._call_count[key] = self._call_count.get(key, 0) + 1
            if self._call_count[key] == 1:
                raise RuntimeError(f"Transient failure for {key}")
            return await super().send_command(device_id, action, parameters)

    channel = FailOnceMockChannel()
    gateway = await _make_gateway(channel=channel)
    try:
        result = await gateway.goal("Clean the house")

        # Should recover via retry and complete
        assert result.status == GoalStatus.COMPLETED
        assert result.execution_report is not None
        assert result.execution_report.success

        # Verify recovery events were published
        recovery_events = gateway._event_bus.get_history(
            event_type="recovery_action"
        )
        assert len(recovery_events) > 0, "Expected recovery_action events"
    finally:
        await gateway.stop()


@pytest.mark.asyncio
async def test_capability_validation_rejects_invalid():
    """Submit command for undeclared capability, verify rejection."""
    channel = MockChannel()
    registry = Registry()
    event_bus = EventBus()
    router = Router(default_channel=channel)
    recovery_engine = RecoveryEngine(
        registry=registry,
        router=router,
        event_bus=event_bus,
    )

    # Use a custom agent that returns a subtask with an invalid capability
    class BadCapabilityAgent(MockAgent):
        async def decompose(self, goal, device_capabilities, constraints=None):
            return {
                "goal": goal,
                "subtasks": [
                    {
                        "id": "task-1",
                        "description": "Do something impossible",
                        "required_capability": "teleport",
                        "parameters": {},
                        "dependencies": [],
                        "priority": 1,
                    }
                ],
                "reasoning": "Testing invalid capability",
            }

    agent = BadCapabilityAgent()
    gateway = Gateway(
        agent=agent,
        router=router,
        registry=registry,
        event_bus=event_bus,
        recovery_engine=recovery_engine,
    )
    await gateway.start()
    try:
        result = await gateway.goal("Teleport to Mars")

        # Should fail because no device has "teleport" capability
        assert result.status == GoalStatus.FAILED
        assert result.execution_report is not None
        assert not result.execution_report.success
    finally:
        await gateway.stop()


@pytest.mark.asyncio
async def test_goal_result_includes_plan_and_report():
    """Verify GoalResult contains both TaskPlan and ExecutionReport."""
    gateway = await _make_gateway()
    try:
        result = await gateway.goal("Water the garden")

        assert result.task_plan is not None
        assert len(result.task_plan.subtasks) > 0
        assert result.execution_report is not None
        assert len(result.execution_report.results) == len(
            result.task_plan.subtasks
        )
    finally:
        await gateway.stop()
