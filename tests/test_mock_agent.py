"""Tests for microids.agents.mock — MockAgent with capability-aware decomposition."""

from __future__ import annotations

import pytest

from microids.agents.mock import MockAgent


@pytest.fixture
def agent() -> MockAgent:
    return MockAgent()


def _caps(*names: str) -> list[dict]:
    """Build a device_capabilities list with the given capability names."""
    return [{"capabilities": [{"name": n} for n in names]}]


@pytest.mark.asyncio
class TestDecompose:
    async def test_vacuum_only(self, agent: MockAgent) -> None:
        result = await agent.decompose(
            goal="Clean the house",
            device_capabilities=_caps("vacuum"),
        )
        assert result["goal"] == "Clean the house"
        assert len(result["subtasks"]) == 1
        assert result["subtasks"][0]["required_capability"] == "vacuum"

    async def test_vacuum_and_mop(self, agent: MockAgent) -> None:
        result = await agent.decompose(
            goal="Clean the house",
            device_capabilities=_caps("vacuum", "mop"),
        )
        subtasks = result["subtasks"]
        assert len(subtasks) == 2
        caps = {s["required_capability"] for s in subtasks}
        assert caps == {"vacuum", "mop"}

        # Mop depends on vacuum
        mop_task = next(s for s in subtasks if s["required_capability"] == "mop")
        vacuum_task = next(s for s in subtasks if s["required_capability"] == "vacuum")
        assert vacuum_task["id"] in mop_task["dependencies"]

    async def test_vacuum_mop_water(self, agent: MockAgent) -> None:
        result = await agent.decompose(
            goal="Clean house and water garden",
            device_capabilities=_caps("vacuum", "mop", "water"),
        )
        subtasks = result["subtasks"]
        assert len(subtasks) == 3
        caps = {s["required_capability"] for s in subtasks}
        assert caps == {"vacuum", "mop", "water"}

        # Water is independent (no dependencies)
        water_task = next(s for s in subtasks if s["required_capability"] == "water")
        assert water_task["dependencies"] == []

    async def test_water_only(self, agent: MockAgent) -> None:
        result = await agent.decompose(
            goal="Water the garden",
            device_capabilities=_caps("water"),
        )
        assert len(result["subtasks"]) == 1
        assert result["subtasks"][0]["required_capability"] == "water"
        assert result["subtasks"][0]["dependencies"] == []

    async def test_no_matching_capabilities(self, agent: MockAgent) -> None:
        result = await agent.decompose(
            goal="Do something",
            device_capabilities=_caps("fly"),
        )
        assert len(result["subtasks"]) == 1
        assert result["subtasks"][0]["required_capability"] == "unknown"

    async def test_empty_capabilities(self, agent: MockAgent) -> None:
        result = await agent.decompose(
            goal="Do something",
            device_capabilities=[],
        )
        assert len(result["subtasks"]) == 1
        assert result["subtasks"][0]["required_capability"] == "unknown"

    async def test_has_reasoning(self, agent: MockAgent) -> None:
        result = await agent.decompose(
            goal="Clean the house",
            device_capabilities=_caps("vacuum"),
        )
        assert "reasoning" in result
        assert len(result["reasoning"]) > 0

    async def test_has_estimated_duration(self, agent: MockAgent) -> None:
        result = await agent.decompose(
            goal="Clean",
            device_capabilities=_caps("vacuum", "mop"),
        )
        assert result["estimated_duration_seconds"] > 0

    async def test_unique_subtask_ids(self, agent: MockAgent) -> None:
        result = await agent.decompose(
            goal="Everything",
            device_capabilities=_caps("vacuum", "mop", "water"),
        )
        ids = [s["id"] for s in result["subtasks"]]
        assert len(ids) == len(set(ids))

    async def test_mop_without_vacuum_has_no_deps(self, agent: MockAgent) -> None:
        """If only mop is available (no vacuum), mop has no dependencies."""
        result = await agent.decompose(
            goal="Mop the floor",
            device_capabilities=_caps("mop"),
        )
        assert len(result["subtasks"]) == 1
        assert result["subtasks"][0]["dependencies"] == []

    async def test_capabilities_as_string_list(self, agent: MockAgent) -> None:
        """Support capabilities passed as plain string lists."""
        result = await agent.decompose(
            goal="Clean",
            device_capabilities=[{"capabilities": ["vacuum", "mop"]}],
        )
        assert len(result["subtasks"]) == 2


@pytest.mark.asyncio
class TestHandleRpc:
    async def test_decompose_method(self, agent: MockAgent) -> None:
        result = await agent.handle_rpc(
            "decompose",
            {
                "goal": "Clean the house",
                "device_capabilities": _caps("vacuum"),
            },
        )
        assert "subtasks" in result

    async def test_unknown_method_raises(self, agent: MockAgent) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            await agent.handle_rpc("unknown_method", {})

    async def test_decompose_with_constraints(self, agent: MockAgent) -> None:
        result = await agent.handle_rpc(
            "decompose",
            {
                "goal": "Clean",
                "device_capabilities": _caps("vacuum"),
                "constraints": {"max_duration": 60},
            },
        )
        assert "subtasks" in result


class TestModelName:
    def test_returns_mock_name(self, agent: MockAgent) -> None:
        assert agent.model_name() == "mock-agent-v1"
