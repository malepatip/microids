"""Gateway — central control plane for microids.

Equivalent to OpenClaw's Gateway. Orchestrates the full goal lifecycle:
receives goals from CLI/API, invokes the Planner for decomposition via
the Agent Runtime, manages the device fleet through the Registry, and
drives task execution.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from microids.core.events import EventBus
from microids.core.executor import Executor
from microids.core.planner import Planner, PlanningError
from microids.core.recovery import RecoveryEngine
from microids.core.registry import Registry
from microids.core.router import Router
from microids.models import GoalResult, GoalStatus


class Gateway:
    """Central control plane — wires everything together."""

    def __init__(
        self,
        config: Optional[Any] = None,
        agent: Optional[Any] = None,
        router: Optional[Router] = None,
        registry: Optional[Registry] = None,
        event_bus: Optional[EventBus] = None,
        recovery_engine: Optional[RecoveryEngine] = None,
    ) -> None:
        self._config = config
        self._agent = agent
        self._router = router or Router()
        self._registry = registry or Registry()
        self._event_bus = event_bus or EventBus()
        self._recovery_engine = recovery_engine

        # Internal components wired lazily
        self._planner: Optional[Planner] = None
        self._executor: Optional[Executor] = None
        self._channel: Optional[Any] = None

    async def start(self) -> None:
        """Connect channel, discover devices, register in registry.

        1. Connect the channel (await channel.connect({}))
        2. Discover devices (await channel.discover_devices())
        3. Register each device in the registry
        """
        if self._router._default_channel is not None:
            self._channel = self._router._default_channel
        elif self._router._channels:
            self._channel = next(iter(self._router._channels.values()))

        if self._channel is not None:
            await self._channel.connect({})
            specs = await self._channel.discover_devices()
            for spec in specs:
                try:
                    await self._registry.register(spec, channel_type="homeassistant" if hasattr(self._channel, '_url') else "mock")
                except (ValueError, Exception):
                    pass  # Skip devices that fail validation — real HA has messy data

        # Wire internal components
        if self._agent is not None:
            self._planner = Planner(self._agent)

        if self._recovery_engine is not None:
            self._executor = Executor(
                registry=self._registry,
                router=self._router,
                recovery_engine=self._recovery_engine,
                event_bus=self._event_bus,
            )

    async def stop(self) -> None:
        """Disconnect channel."""
        if self._channel is not None:
            await self._channel.disconnect()

    async def goal(self, goal_text: str) -> GoalResult:
        """Full coordination loop: decompose → validate → execute → report."""
        start = time.monotonic()

        if self._planner is None:
            return GoalResult(
                goal=goal_text,
                status=GoalStatus.FAILED,
                error="No agent configured — cannot plan",
                duration_seconds=time.monotonic() - start,
            )

        if self._executor is None:
            return GoalResult(
                goal=goal_text,
                status=GoalStatus.FAILED,
                error="No recovery engine configured — cannot execute",
                duration_seconds=time.monotonic() - start,
            )

        # 1. Get capabilities schema from registry
        capabilities = await self._registry.get_capabilities_schema()

        # Enrich with simulator state if available (richer than DeviceStatus)
        if self._channel and hasattr(self._channel, 'get_simulator_state'):
            sim_state = self._channel.get_simulator_state()
            for cap in capabilities:
                dev_key = cap["device_id"].replace("mock:", "")
                if dev_key in sim_state:
                    cap["state"] = sim_state[dev_key]["state"]

        # 2. Decompose goal via Planner
        try:
            plan = await self._planner.decompose(goal_text, capabilities)
        except PlanningError as exc:
            return GoalResult(
                goal=goal_text,
                status=GoalStatus.FAILED,
                error=str(exc),
                duration_seconds=time.monotonic() - start,
            )

        # 3. Validate the task plan
        if not plan.validate():
            return GoalResult(
                goal=goal_text,
                status=GoalStatus.FAILED,
                task_plan=plan,
                error="Task plan validation failed: cyclic dependencies or invalid structure",
                duration_seconds=time.monotonic() - start,
            )

        # 4. Execute via Executor
        report = await self._executor.execute(plan)

        # 5. Determine status
        # Check for SUSPENDED results
        suspended = any(
            r.error == "SUSPENDED" for r in report.results if not r.success
        )
        if suspended:
            status = GoalStatus.SUSPENDED
        elif report.success:
            status = GoalStatus.COMPLETED
        else:
            status = GoalStatus.FAILED

        elapsed = time.monotonic() - start
        return GoalResult(
            goal=goal_text,
            status=status,
            task_plan=plan,
            execution_report=report,
            duration_seconds=elapsed,
        )

    async def resume(self, goal_id: str) -> GoalResult:
        """Stub for Phase 2."""
        raise NotImplementedError("resume() available in Phase 2")

    async def skip(self, goal_id: str, subtask_id: str) -> GoalResult:
        """Stub for Phase 2."""
        raise NotImplementedError("skip() available in Phase 2")

    async def replan(self, goal_id: str) -> GoalResult:
        """Stub for Phase 2."""
        raise NotImplementedError("replan() available in Phase 2")
