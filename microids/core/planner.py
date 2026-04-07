"""Planner — goal decomposition via Agent Runtime.

Phase 1: direct in-process function call to agent.handle_rpc().
Phase 2+: JSON-RPC transport layer.

Security Rules enforced:
  S7 — Token budget awareness (truncate capabilities if too large)
  S8 — wrap_untrusted() on all user-supplied content before LLM
"""

from __future__ import annotations

import json
from typing import Any, Optional

from microids.core.security import wrap_untrusted
from microids.models import Subtask, TaskPlan


class PlanningError(Exception):
    """Raised when the LLM fails to produce a valid task plan."""


# Rough token budget for device capability schema (Phase 1: simple char limit)
_MAX_CAPABILITIES_CHARS = 50_000


class Planner:
    """Goal decomposition via Agent Runtime (in-process for Phase 1)."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    async def decompose(
        self,
        goal: str,
        device_capabilities: list[dict[str, Any]],
        constraints: Optional[dict[str, Any]] = None,
    ) -> TaskPlan:
        """Ask the agent to decompose a goal into subtasks.

        1. Wrap goal with wrap_untrusted (Security Rule S8)
        2. Wrap capabilities with wrap_untrusted (Security Rule S8)
        3. Truncate capabilities if exceeding token budget (Security Rule S7)
        4. Call agent.handle_rpc("decompose", params) directly (in-process)
        5. Parse response dict into TaskPlan with Subtask objects
        6. If response is unparseable, raise PlanningError
        """
        # S8 — boundary markers for LLM prompts
        wrapped_goal = wrap_untrusted("user goal", goal)

        caps_json = json.dumps(device_capabilities)
        # S7 — truncate if exceeding token budget
        if len(caps_json) > _MAX_CAPABILITIES_CHARS:
            caps_json = caps_json[:_MAX_CAPABILITIES_CHARS]
        wrapped_caps = wrap_untrusted("device data", caps_json)

        params: dict[str, Any] = {
            "goal": wrapped_goal,
            "device_capabilities": device_capabilities,
            "constraints": constraints,
        }

        try:
            response = await self._agent.handle_rpc("decompose", params)
        except Exception as exc:
            raise PlanningError(f"Agent decompose failed: {exc}") from exc

        return self._parse_response(response, goal)

    def _parse_response(self, response: Any, original_goal: str) -> TaskPlan:
        """Parse agent response dict into a TaskPlan."""
        if not isinstance(response, dict):
            raise PlanningError(
                f"Expected dict from agent, got {type(response).__name__}"
            )

        subtasks_raw = response.get("subtasks")
        if not isinstance(subtasks_raw, list) or not subtasks_raw:
            raise PlanningError("Agent returned no subtasks")

        subtasks: list[Subtask] = []
        for raw in subtasks_raw:
            if not isinstance(raw, dict):
                raise PlanningError(f"Invalid subtask entry: {raw!r}")
            try:
                subtasks.append(
                    Subtask(
                        id=raw["id"],
                        description=raw.get("description", ""),
                        required_capability=raw.get("required_capability", ""),
                        parameters=raw.get("parameters", {}),
                        dependencies=raw.get("dependencies", []),
                        priority=raw.get("priority", 5),
                    )
                )
            except (KeyError, TypeError) as exc:
                raise PlanningError(
                    f"Failed to parse subtask: {exc}"
                ) from exc

        return TaskPlan(
            goal=response.get("goal", original_goal),
            subtasks=subtasks,
            reasoning=response.get("reasoning", ""),
            estimated_duration_seconds=response.get(
                "estimated_duration_seconds"
            ),
        )
