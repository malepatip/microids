"""Unit tests for microids.models — all core data models."""

from datetime import datetime, timezone

import pytest

from microids.models import (
    AgentConfig,
    ChannelConfig,
    Device,
    DeviceCapability,
    DeviceCategory,
    DeviceContext,
    DeviceRecoveryConfig,
    DeviceSpec,
    DeviceStatus,
    ExecutionCheckpoint,
    ExecutionReport,
    GoalResult,
    GoalStatus,
    MicroidsConfig,
    RPCRequest,
    RPCResponse,
    RecoveryAction,
    RecoveryConfig,
    RecoveryContext,
    Subtask,
    TaskPlan,
    TaskResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(zone: str = "kitchen", reachable: list[str] | None = None) -> DeviceContext:
    return DeviceContext(zone=zone, reachable_zones=reachable or [zone])


def _make_cap(name: str = "vacuum") -> DeviceCapability:
    return DeviceCapability(name=name)


def _make_spec(
    name: str = "test-vacuum",
    device_type: str = "vacuum",
    category: DeviceCategory = DeviceCategory.ACTUATOR,
    capabilities: list[DeviceCapability] | None = None,
    context: DeviceContext | None = None,
) -> DeviceSpec:
    return DeviceSpec(
        name=name,
        device_type=device_type,
        category=category,
        capabilities=[_make_cap()] if capabilities is None else capabilities,
        context=context or _make_context(),
    )


# ---------------------------------------------------------------------------
# DeviceCategory
# ---------------------------------------------------------------------------

class TestDeviceCategory:
    def test_all_members(self):
        assert DeviceCategory.ACTUATOR.value == "actuator"
        assert DeviceCategory.SENSOR.value == "sensor"
        assert DeviceCategory.CONTROLLER.value == "controller"
        assert DeviceCategory.MOBILE_AUTONOMOUS.value == "mobile_autonomous"

    def test_member_count(self):
        assert len(DeviceCategory) == 4


# ---------------------------------------------------------------------------
# DeviceContext
# ---------------------------------------------------------------------------

class TestDeviceContext:
    def test_defaults(self):
        ctx = DeviceContext(zone="garden", reachable_zones=["garden"])
        assert ctx.is_mobile is False
        assert ctx.battery_level is None

    def test_with_battery(self):
        ctx = DeviceContext(zone="a", reachable_zones=["a"], battery_level=0.85)
        assert ctx.battery_level == 0.85


# ---------------------------------------------------------------------------
# DeviceSpec.validate()
# ---------------------------------------------------------------------------

class TestDeviceSpecValidation:
    def test_valid_spec(self):
        assert _make_spec().validate() is True

    def test_empty_name(self):
        assert _make_spec(name="").validate() is False

    def test_long_name(self):
        assert _make_spec(name="x" * 256).validate() is False

    def test_max_name(self):
        assert _make_spec(name="x" * 255).validate() is True

    def test_empty_device_type(self):
        assert _make_spec(device_type="").validate() is False

    def test_empty_capabilities(self):
        assert _make_spec(capabilities=[]).validate() is False

    def test_empty_zone(self):
        ctx = DeviceContext(zone="", reachable_zones=[""])
        assert _make_spec(context=ctx).validate() is False

    def test_empty_reachable_zones(self):
        ctx = DeviceContext(zone="kitchen", reachable_zones=[])
        assert _make_spec(context=ctx).validate() is False

    def test_zone_not_in_reachable(self):
        ctx = DeviceContext(zone="kitchen", reachable_zones=["hallway"])
        assert _make_spec(context=ctx).validate() is False

    def test_battery_too_high(self):
        ctx = DeviceContext(zone="a", reachable_zones=["a"], battery_level=1.1)
        assert _make_spec(context=ctx).validate() is False

    def test_battery_too_low(self):
        ctx = DeviceContext(zone="a", reachable_zones=["a"], battery_level=-0.1)
        assert _make_spec(context=ctx).validate() is False

    def test_battery_boundary_zero(self):
        ctx = DeviceContext(zone="a", reachable_zones=["a"], battery_level=0.0)
        assert _make_spec(context=ctx).validate() is True

    def test_battery_boundary_one(self):
        ctx = DeviceContext(zone="a", reachable_zones=["a"], battery_level=1.0)
        assert _make_spec(context=ctx).validate() is True


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

class TestDevice:
    def test_default_status(self):
        dev = Device(spec=_make_spec())
        assert dev.status == DeviceStatus.IDLE
        assert dev.current_task is None

    def test_auto_id(self):
        d1 = Device(spec=_make_spec())
        d2 = Device(spec=_make_spec())
        assert d1.id != d2.id


# ---------------------------------------------------------------------------
# TaskPlan.validate()
# ---------------------------------------------------------------------------

class TestTaskPlanValidate:
    def test_valid_linear_chain(self):
        s1 = Subtask(id="1", description="a", required_capability="x")
        s2 = Subtask(id="2", description="b", required_capability="y", dependencies=["1"])
        plan = TaskPlan(goal="g", subtasks=[s1, s2], reasoning="r")
        assert plan.validate() is True

    def test_valid_no_deps(self):
        s1 = Subtask(id="1", description="a", required_capability="x")
        s2 = Subtask(id="2", description="b", required_capability="y")
        plan = TaskPlan(goal="g", subtasks=[s1, s2], reasoning="r")
        assert plan.validate() is True

    def test_empty_subtasks(self):
        plan = TaskPlan(goal="g", subtasks=[], reasoning="r")
        assert plan.validate() is False

    def test_duplicate_ids(self):
        s1 = Subtask(id="1", description="a", required_capability="x")
        s2 = Subtask(id="1", description="b", required_capability="y")
        plan = TaskPlan(goal="g", subtasks=[s1, s2], reasoning="r")
        assert plan.validate() is False

    def test_invalid_dep_ref(self):
        s1 = Subtask(id="1", description="a", required_capability="x", dependencies=["999"])
        plan = TaskPlan(goal="g", subtasks=[s1], reasoning="r")
        assert plan.validate() is False

    def test_cycle_two_nodes(self):
        s1 = Subtask(id="a", description="x", required_capability="x", dependencies=["b"])
        s2 = Subtask(id="b", description="y", required_capability="y", dependencies=["a"])
        plan = TaskPlan(goal="g", subtasks=[s1, s2], reasoning="r")
        assert plan.validate() is False

    def test_cycle_three_nodes(self):
        s1 = Subtask(id="a", description="x", required_capability="x", dependencies=["c"])
        s2 = Subtask(id="b", description="y", required_capability="y", dependencies=["a"])
        s3 = Subtask(id="c", description="z", required_capability="z", dependencies=["b"])
        plan = TaskPlan(goal="g", subtasks=[s1, s2, s3], reasoning="r")
        assert plan.validate() is False

    def test_self_dependency(self):
        s1 = Subtask(id="a", description="x", required_capability="x", dependencies=["a"])
        plan = TaskPlan(goal="g", subtasks=[s1], reasoning="r")
        assert plan.validate() is False

    def test_single_subtask_no_deps(self):
        s1 = Subtask(id="1", description="a", required_capability="x")
        plan = TaskPlan(goal="g", subtasks=[s1], reasoning="r")
        assert plan.validate() is True


# ---------------------------------------------------------------------------
# TaskPlan.topological_order()
# ---------------------------------------------------------------------------

class TestTaskPlanTopologicalOrder:
    def test_linear_chain(self):
        s1 = Subtask(id="1", description="a", required_capability="x")
        s2 = Subtask(id="2", description="b", required_capability="y", dependencies=["1"])
        s3 = Subtask(id="3", description="c", required_capability="z", dependencies=["2"])
        plan = TaskPlan(goal="g", subtasks=[s1, s2, s3], reasoning="r")
        waves = plan.topological_order()
        assert len(waves) == 3
        assert [w[0].id for w in waves] == ["1", "2", "3"]

    def test_parallel_tasks(self):
        s1 = Subtask(id="1", description="a", required_capability="x")
        s2 = Subtask(id="2", description="b", required_capability="y")
        plan = TaskPlan(goal="g", subtasks=[s1, s2], reasoning="r")
        waves = plan.topological_order()
        assert len(waves) == 1
        assert len(waves[0]) == 2

    def test_diamond_dag(self):
        s1 = Subtask(id="1", description="a", required_capability="x")
        s2 = Subtask(id="2", description="b", required_capability="y", dependencies=["1"])
        s3 = Subtask(id="3", description="c", required_capability="z", dependencies=["1"])
        s4 = Subtask(id="4", description="d", required_capability="w", dependencies=["2", "3"])
        plan = TaskPlan(goal="g", subtasks=[s1, s2, s3, s4], reasoning="r")
        waves = plan.topological_order()
        assert len(waves) == 3
        assert waves[0][0].id == "1"
        assert {w.id for w in waves[1]} == {"2", "3"}
        assert waves[2][0].id == "4"

    def test_empty_plan(self):
        plan = TaskPlan(goal="g", subtasks=[], reasoning="r")
        assert plan.topological_order() == []


# ---------------------------------------------------------------------------
# ExecutionReport
# ---------------------------------------------------------------------------

class TestExecutionReport:
    def test_success_all_pass(self):
        r1 = TaskResult(subtask_id="1", device_id="d1", success=True)
        r2 = TaskResult(subtask_id="2", device_id="d2", success=True)
        report = ExecutionReport(goal="g", results=[r1, r2])
        assert report.success is True
        assert report.failed_tasks == []

    def test_success_with_failure(self):
        r1 = TaskResult(subtask_id="1", device_id="d1", success=True)
        r2 = TaskResult(subtask_id="2", device_id="d2", success=False, error="oops")
        report = ExecutionReport(goal="g", results=[r1, r2])
        assert report.success is False
        assert len(report.failed_tasks) == 1
        assert report.failed_tasks[0].subtask_id == "2"

    def test_empty_results(self):
        report = ExecutionReport(goal="g", results=[])
        assert report.success is True
        assert report.failed_tasks == []


# ---------------------------------------------------------------------------
# GoalStatus & GoalResult
# ---------------------------------------------------------------------------

class TestGoalStatus:
    def test_suspended_exists(self):
        assert GoalStatus.SUSPENDED.value == "suspended"

    def test_all_statuses(self):
        assert len(GoalStatus) == 7


class TestGoalResult:
    def test_defaults(self):
        r = GoalResult(goal="test", status=GoalStatus.COMPLETED)
        assert r.task_plan is None
        assert r.execution_report is None
        assert r.error is None
        assert r.duration_seconds == 0.0


# ---------------------------------------------------------------------------
# Recovery Domain
# ---------------------------------------------------------------------------

class TestRecoveryAction:
    def test_all_actions(self):
        assert len(RecoveryAction) == 8
        assert RecoveryAction.RETRY.value == "retry"
        assert RecoveryAction.SAFE_STATE.value == "safe_state"
        assert RecoveryAction.FAIL.value == "fail"


class TestRecoveryContext:
    def test_default_previous_actions(self):
        s = Subtask(id="1", description="a", required_capability="x")
        d = Device(spec=_make_spec())
        ctx = RecoveryContext(
            subtask=s, device=d, error=RuntimeError("fail"),
            attempt_number=1, category=DeviceCategory.ACTUATOR,
            zone="kitchen", fleet_state={},
        )
        assert ctx.previous_actions == []


# ---------------------------------------------------------------------------
# JSON-RPC
# ---------------------------------------------------------------------------

class TestRPCRequest:
    def test_defaults(self):
        req = RPCRequest()
        assert req.jsonrpc == "2.0"
        assert req.id is None
        assert req.method == ""
        assert req.params == {}

    def test_with_values(self):
        req = RPCRequest(id=1, method="decompose", params={"goal": "test"})
        assert req.id == 1
        assert req.method == "decompose"


class TestRPCResponse:
    def test_defaults(self):
        resp = RPCResponse()
        assert resp.jsonrpc == "2.0"
        assert resp.result is None
        assert resp.error is None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestConfigModels:
    def test_channel_config(self):
        c = ChannelConfig(type="mock")
        assert c.url is None
        assert c.extra == {}

    def test_agent_config_defaults(self):
        a = AgentConfig(type="openai")
        assert a.model == "gpt-4o"
        assert a.transport == "in-process"

    def test_recovery_config_defaults(self):
        r = RecoveryConfig()
        assert r.default_suspension_timeout == 1800
        assert r.notification_intervals == [0.0, 0.5, 0.9, 1.0]
        assert r.enable_circuit_breaker is True

    def test_device_recovery_config_defaults(self):
        d = DeviceRecoveryConfig()
        assert d.max_retries is None
        assert d.on_failure is None

    def test_microids_config(self):
        cfg = MicroidsConfig(
            channel=ChannelConfig(type="mock"),
            agent=AgentConfig(type="openai"),
        )
        assert cfg.workflows_dir == "./workflows"
        assert cfg.log_level == "INFO"
        assert cfg.devices == []
        assert isinstance(cfg.recovery, RecoveryConfig)


# ---------------------------------------------------------------------------
# ExecutionCheckpoint
# ---------------------------------------------------------------------------

class TestExecutionCheckpoint:
    def test_creation(self):
        cp = ExecutionCheckpoint(
            goal_id="g1",
            completed_subtasks=["1"],
            failed_subtasks=[],
            blocked_subtasks=[],
            pending_subtasks=["2"],
            suspended_device_id="d1",
            suspended_subtask_id="s1",
            suspension_reason="stuck",
            suspended_at=datetime.now(timezone.utc),
            timeout_seconds=1800,
        )
        assert cp.goal_id == "g1"
        assert cp.timeout_seconds == 1800
