"""Core data models for the microids framework.

All dataclasses, enums, and type definitions used across the framework.
This module is the single source of truth for all types.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import uuid


# ---------------------------------------------------------------------------
# Device Categories
# ---------------------------------------------------------------------------


class DeviceCategory(Enum):
    """Device classification for category-specific recovery pipelines."""

    ACTUATOR = "actuator"
    SENSOR = "sensor"
    CONTROLLER = "controller"
    MOBILE_AUTONOMOUS = "mobile_autonomous"


# ---------------------------------------------------------------------------
# Device Context
# ---------------------------------------------------------------------------


@dataclass
class DeviceContext:
    """Spatial and physical context for zone-aware device allocation and recovery."""

    zone: str
    reachable_zones: list[str]
    is_mobile: bool = False
    battery_level: Optional[float] = None


# ---------------------------------------------------------------------------
# Device Domain
# ---------------------------------------------------------------------------


@dataclass
class DeviceCapability:
    """A named capability with parameters and constraints."""

    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)


class DeviceStatus(Enum):
    """Current operational status of a device."""

    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class DeviceSpec:
    """Full specification of a device including capabilities and context."""

    name: str
    device_type: str
    category: DeviceCategory
    capabilities: list[DeviceCapability]
    context: DeviceContext
    channel_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate the device spec against all rules.

        Rules:
        - name: non-empty, max 255 chars
        - device_type: non-empty
        - capabilities: non-empty list
        - category: valid DeviceCategory
        - context.zone: non-empty
        - context.reachable_zones: non-empty, must include own zone
        - context.battery_level: if present, in [0.0, 1.0]
        """
        if not self.name or len(self.name) > 255:
            return False
        if not self.device_type:
            return False
        if not self.capabilities:
            return False
        if not isinstance(self.category, DeviceCategory):
            return False
        if not self.context.zone:
            return False
        if not self.context.reachable_zones:
            return False
        if self.context.zone not in self.context.reachable_zones:
            return False
        if self.context.battery_level is not None:
            if not (0.0 <= self.context.battery_level <= 1.0):
                return False
        return True


@dataclass
class Device:
    """A registered device instance with runtime state."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    spec: DeviceSpec = field(default=None)  # type: ignore[assignment]
    status: DeviceStatus = DeviceStatus.IDLE
    current_task: Optional[str] = None


# ---------------------------------------------------------------------------
# Task Planning Domain
# ---------------------------------------------------------------------------


@dataclass
class Subtask:
    """An individual unit of work within a TaskPlan."""

    id: str
    description: str
    required_capability: str
    parameters: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    priority: int = 5


@dataclass
class TaskPlan:
    """A structured plan containing subtasks with dependencies (DAG)."""

    goal: str
    subtasks: list[Subtask]
    reasoning: str
    estimated_duration_seconds: Optional[float] = None

    def validate(self) -> bool:
        """Validate the task plan DAG using Kahn's algorithm.

        Returns True iff:
        - subtasks list is non-empty
        - all subtask IDs are unique
        - all dependency references point to existing subtask IDs
        - the dependency graph contains no cycles
        """
        if not self.subtasks:
            return False

        ids = {s.id for s in self.subtasks}
        if len(ids) != len(self.subtasks):
            return False

        for subtask in self.subtasks:
            for dep_id in subtask.dependencies:
                if dep_id not in ids:
                    return False

        # Kahn's algorithm for cycle detection
        in_degree: dict[str, int] = {s.id: 0 for s in self.subtasks}
        adjacency: dict[str, list[str]] = {s.id: [] for s in self.subtasks}
        for subtask in self.subtasks:
            for dep_id in subtask.dependencies:
                adjacency[dep_id].append(subtask.id)
                in_degree[subtask.id] += 1

        queue = deque(sid for sid, deg in in_degree.items() if deg == 0)
        processed_count = 0
        while queue:
            node = queue.popleft()
            processed_count += 1
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return processed_count == len(self.subtasks)

    def topological_order(self) -> list[list[Subtask]]:
        """Return subtasks grouped by execution wave (topological layers).

        Each wave contains subtasks whose dependencies are all in prior waves.
        Subtasks within a wave can execute in parallel.
        """
        if not self.subtasks:
            return []

        subtask_map = {s.id: s for s in self.subtasks}
        in_degree: dict[str, int] = {s.id: 0 for s in self.subtasks}
        adjacency: dict[str, list[str]] = {s.id: [] for s in self.subtasks}

        for subtask in self.subtasks:
            for dep_id in subtask.dependencies:
                adjacency[dep_id].append(subtask.id)
                in_degree[subtask.id] += 1

        current_wave = [sid for sid, deg in in_degree.items() if deg == 0]
        waves: list[list[Subtask]] = []

        while current_wave:
            waves.append([subtask_map[sid] for sid in current_wave])
            next_wave: list[str] = []
            for sid in current_wave:
                for neighbor in adjacency[sid]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_wave.append(neighbor)
            current_wave = next_wave

        return waves


# ---------------------------------------------------------------------------
# Execution Domain
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    """The outcome of a single subtask execution."""

    subtask_id: str
    device_id: str
    success: bool
    response: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class ExecutionReport:
    """Aggregate results for all subtasks in a plan."""

    goal: str
    results: list[TaskResult]
    total_duration_seconds: float = 0.0

    @property
    def success(self) -> bool:
        """True iff all results have success=True."""
        return all(r.success for r in self.results)

    @property
    def failed_tasks(self) -> list[TaskResult]:
        """List of TaskResults where success is False."""
        return [r for r in self.results if not r.success]


# ---------------------------------------------------------------------------
# Goal Lifecycle
# ---------------------------------------------------------------------------


class GoalStatus(Enum):
    """Status of a goal through its lifecycle."""

    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"


@dataclass
class GoalResult:
    """The outcome of a goal execution."""

    goal: str
    status: GoalStatus
    task_plan: Optional[TaskPlan] = None
    execution_report: Optional[ExecutionReport] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Recovery Domain
# ---------------------------------------------------------------------------


class RecoveryAction(Enum):
    """Possible outcomes from a recovery strategy attempt."""

    RETRY = "retry"
    CIRCUIT_BREAK = "circuit_break"
    ALTERNATIVE_DEVICE = "alternative_device"
    REPLAN = "replan"
    SUSPEND = "suspend"
    SAFE_STATE = "safe_state"
    SKIP = "skip"
    FAIL = "fail"


@dataclass
class RecoveryContext:
    """Context passed to each recovery strategy in the pipeline."""

    subtask: Subtask
    device: Device
    error: Exception
    attempt_number: int
    category: DeviceCategory
    zone: str
    fleet_state: dict[str, DeviceStatus]
    previous_actions: list[RecoveryAction] = field(default_factory=list)


@dataclass
class ExecutionCheckpoint:
    """Snapshot of goal execution state when entering SUSPENDED status."""

    goal_id: str
    completed_subtasks: list[str]
    failed_subtasks: list[str]
    blocked_subtasks: list[str]
    pending_subtasks: list[str]
    suspended_device_id: str
    suspended_subtask_id: str
    suspension_reason: str
    suspended_at: datetime
    timeout_seconds: int


# ---------------------------------------------------------------------------
# JSON-RPC Protocol
# ---------------------------------------------------------------------------


@dataclass
class RPCRequest:
    """A JSON-RPC 2.0 request message."""

    jsonrpc: str = "2.0"
    id: Optional[int] = None
    method: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RPCResponse:
    """A JSON-RPC 2.0 response message."""

    jsonrpc: str = "2.0"
    id: Optional[int] = None
    result: Optional[Any] = None
    error: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Configuration Domain
# ---------------------------------------------------------------------------


@dataclass
class DeviceRecoveryConfig:
    """Per-device recovery overrides (Layer 2 — YAML)."""

    max_retries: Optional[int] = None
    allow_alternative: Optional[bool] = None
    on_failure: Optional[str] = None
    cooldown_seconds: Optional[int] = None


@dataclass
class RecoveryConfig:
    """Global recovery configuration."""

    default_suspension_timeout: int = 1800
    notification_intervals: list[float] = field(
        default_factory=lambda: [0.0, 0.5, 0.9, 1.0]
    )
    enable_circuit_breaker: bool = True
    circuit_breaker_cooldown: int = 300


@dataclass
class ChannelConfig:
    """Channel connection configuration."""

    type: str
    url: Optional[str] = None
    token: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """Agent runtime configuration."""

    type: str
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    transport: str = "in-process"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class MicroidsConfig:
    """Top-level configuration for the microids framework."""

    channel: ChannelConfig
    agent: AgentConfig
    devices: list[DeviceSpec] = field(default_factory=list)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    workflows_dir: str = "./workflows"
    log_level: str = "INFO"
