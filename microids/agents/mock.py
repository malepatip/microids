"""MockAgent — returns task plans for testing and simulation.

No BaseAgent ABC yet — just a concrete class with the right method signatures.
The ABC will be extracted in Phase 4 when the second agent (OpenAI) demands it.
"""

from __future__ import annotations

from typing import Any, Optional


# Goal → capability mapping for the simulator.
# Keywords in the goal text trigger specific device capabilities.
_GOAL_PATTERNS: list[tuple[list[str], list[dict[str, Any]]]] = [
    # Cleaning
    (
        ["clean", "vacuum", "sweep", "floor"],
        [
            {"cap": "vacuum", "desc": "Vacuum the floors", "priority": 1},
            {"cap": "mop", "desc": "Mop the floors", "priority": 2, "depends_on": "vacuum"},
        ],
    ),
    # Vacuuming only
    (
        ["just vacuum", "only vacuum"],
        [
            {"cap": "vacuum", "desc": "Vacuum the floors", "priority": 1},
        ],
    ),
    # Mopping only
    (
        ["mop"],
        [
            {"cap": "mop", "desc": "Mop the floors", "priority": 1},
        ],
    ),
    # Watering
    (
        ["water", "sprinkler", "irrigat", "garden"],
        [
            {"cap": "water", "desc": "Water the garden", "priority": 1},
        ],
    ),
    # Garage
    (
        ["garage", "open the door", "close the door"],
        [
            {"cap": "open", "desc": "Open the garage door", "priority": 1},
        ],
    ),
    # Close garage
    (
        ["close the garage", "shut the garage"],
        [
            {"cap": "close", "desc": "Close the garage door", "priority": 1},
        ],
    ),
    # Lights on
    (
        ["lights on", "turn on the light", "turn on light", "illuminate", "bright"],
        [
            {"cap": "turn_on", "desc": "Turn on the lights", "priority": 1},
        ],
    ),
    # Lights off
    (
        ["lights off", "turn off the light", "turn off light", "dark"],
        [
            {"cap": "turn_off", "desc": "Turn off the lights", "priority": 1},
        ],
    ),
    # Camera / security
    (
        ["secure", "security", "camera", "snapshot", "check outside", "watch", "monitor"],
        [
            {"cap": "snapshot", "desc": "Take a security snapshot", "priority": 1},
            {"cap": "detect_motion", "desc": "Enable motion detection", "priority": 2},
        ],
    ),
    # Dock / return vacuum
    (
        ["dock", "return", "come back", "stop clean"],
        [
            {"cap": "return_to_base", "desc": "Return vacuum to dock", "priority": 1},
        ],
    ),
    # Everything / goodnight / leaving / going to bed / going to office
    (
        ["everything", "goodnight", "good night", "leaving", "away",
         "lock down", "going to bed", "bedtime", "sleep",
         "going to office", "heading out", "bye"],
        [
            {"cap": "close", "desc": "Close the garage door", "priority": 1},
            {"cap": "turn_off", "desc": "Turn off the lights", "priority": 2},
            {"cap": "detect_motion", "desc": "Enable motion detection", "priority": 3},
            {"cap": "return_to_base", "desc": "Dock the vacuum", "priority": 4},
        ],
    ),
    # Good morning / arriving / coming home
    (
        ["good morning", "wake up", "arriving", "i'm home", "coming home",
         "home", "back home", "hello"],
        [
            {"cap": "open", "desc": "Open the garage door", "priority": 1},
            {"cap": "turn_on", "desc": "Turn on the lights", "priority": 2},
        ],
    ),
    # Party / movie / entertain
    (
        ["party", "movie", "entertain", "chill", "relax", "cozy"],
        [
            {"cap": "turn_on", "desc": "Set the mood lighting", "priority": 1},
        ],
    ),
    # Reset / stop everything
    (
        ["stop", "reset", "cancel", "off", "shut down", "all off"],
        [
            {"cap": "turn_off", "desc": "Turn off the lights", "priority": 1},
            {"cap": "close", "desc": "Close the garage", "priority": 2},
            {"cap": "return_to_base", "desc": "Dock the vacuum", "priority": 3},
        ],
    ),
]


class MockAgent:
    """Mock LLM agent that returns goal-aware task plans.

    In simulator mode, parses the goal text to create relevant tasks.
    Falls back to capability-based planning for unit tests.
    """

    async def handle_rpc(self, method: str, params: dict[str, Any]) -> Any:
        """Dispatch JSON-RPC method to handler."""
        if method == "decompose":
            return await self.decompose(
                goal=params["goal"],
                device_capabilities=params["device_capabilities"],
                constraints=params.get("constraints"),
            )
        raise ValueError(f"Unknown method: {method}")

    async def decompose(
        self,
        goal: str,
        device_capabilities: list[Any],
        constraints: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Return a task plan based on the goal text and available capabilities."""
        # Flatten all capability names from the fleet
        available: set[str] = set()
        for device_caps in device_capabilities:
            if isinstance(device_caps, dict):
                for cap in device_caps.get("capabilities", []):
                    if isinstance(cap, str):
                        available.add(cap)
                    elif isinstance(cap, dict):
                        available.add(cap.get("name", ""))
            elif isinstance(device_caps, list):
                for cap in device_caps:
                    if isinstance(cap, str):
                        available.add(cap)
                    elif isinstance(cap, dict):
                        available.add(cap.get("name", ""))

        # Try goal-aware planning first
        subtasks = self._plan_from_goal(goal, available)

        # Fallback for unit tests: if no goal pattern matched AND the
        # capabilities are the basic test set (vacuum/mop/water), use
        # the old all-capabilities behavior for backward compatibility.
        # For the simulator (which has richer capabilities like open/close/
        # turn_on/turn_off), return a "don't understand" message instead.
        _BASIC_TEST_CAPS = {"vacuum", "mop", "water", "dock", "snapshot",
                            "detect_motion", "schedule"}
        if not subtasks:
            if available <= _BASIC_TEST_CAPS:
                subtasks = self._plan_all_capabilities(available)
            # else: leave subtasks empty → final fallback below

        # Final fallback
        if not subtasks:
            subtasks = [{
                "id": "task-1",
                "description": f"Not sure how to: {goal}. Try 'clean the house', 'water the garden', 'open the garage', or 'goodnight'.",
                "required_capability": "unknown",
                "parameters": {},
                "dependencies": [],
                "priority": 5,
            }]

        return {
            "goal": goal,
            "subtasks": subtasks,
            "reasoning": f"Planned {len(subtasks)} task(s) for: {goal}",
            "estimated_duration_seconds": len(subtasks) * 10.0,
        }

    def _plan_from_goal(
        self, goal: str, available: set[str]
    ) -> list[dict[str, Any]]:
        """Match goal text against known patterns and build tasks.

        Supports compound goals like "Clean house and water garden" by
        collecting tasks from ALL matching patterns, deduplicating by capability.
        """
        goal_lower = goal.lower()

        # Collect tasks from all matching patterns
        matched_tasks: list[dict[str, Any]] = []
        seen_caps: set[str] = set()

        for keywords, task_defs in _GOAL_PATTERNS:
            if any(kw in goal_lower for kw in keywords):
                for td in task_defs:
                    if td["cap"] in available and td["cap"] not in seen_caps:
                        matched_tasks.append(td)
                        seen_caps.add(td["cap"])

        if not matched_tasks:
            return []

        # Build subtask list
        subtasks: list[dict[str, Any]] = []
        task_id = 0
        id_map: dict[str, str] = {}  # cap → task_id for dependency resolution

        for td in matched_tasks:
            task_id += 1
            tid = f"task-{task_id}"
            id_map[td["cap"]] = tid

            deps = []
            dep_cap = td.get("depends_on")
            if dep_cap and dep_cap in id_map:
                deps = [id_map[dep_cap]]

            subtasks.append({
                "id": tid,
                "description": td["desc"],
                "required_capability": td["cap"],
                "parameters": {},
                "dependencies": deps,
                "priority": td.get("priority", 5),
            })

        return subtasks

    def _plan_all_capabilities(
        self, available: set[str]
    ) -> list[dict[str, Any]]:
        """Fallback: plan tasks for all available capabilities (unit test compat)."""
        subtasks: list[dict[str, Any]] = []
        task_id = 0

        if "vacuum" in available:
            task_id += 1
            subtasks.append({
                "id": f"task-{task_id}",
                "description": "Vacuum all reachable zones",
                "required_capability": "vacuum",
                "parameters": {},
                "dependencies": [],
                "priority": 1,
            })

        vacuum_id = f"task-{task_id}" if "vacuum" in available else None

        if "mop" in available:
            task_id += 1
            deps = [vacuum_id] if vacuum_id else []
            subtasks.append({
                "id": f"task-{task_id}",
                "description": "Mop all reachable zones",
                "required_capability": "mop",
                "parameters": {},
                "dependencies": deps,
                "priority": 2,
            })

        if "water" in available:
            task_id += 1
            subtasks.append({
                "id": f"task-{task_id}",
                "description": "Water the garden",
                "required_capability": "water",
                "parameters": {},
                "dependencies": [],
                "priority": 3,
            })

        return subtasks

    def model_name(self) -> str:
        return "mock-agent-v1"
