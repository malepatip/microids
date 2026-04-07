"""Tests for microids.channels.mock — MockChannel with capability validation."""

from __future__ import annotations

import pytest
import pytest_asyncio

from microids.channels.mock import MockChannel
from microids.models import DeviceCategory, DeviceStatus


@pytest.fixture
def channel() -> MockChannel:
    return MockChannel()


@pytest.mark.asyncio
class TestDiscoverDevices:
    async def test_returns_four_devices(self, channel: MockChannel) -> None:
        specs = await channel.discover_devices()
        assert len(specs) == 4

    async def test_device_names(self, channel: MockChannel) -> None:
        specs = await channel.discover_devices()
        names = {s.name for s in specs}
        assert names == {"vacuum", "camera", "sprinkler", "mop"}

    async def test_vacuum_is_actuator(self, channel: MockChannel) -> None:
        specs = await channel.discover_devices()
        vacuum = next(s for s in specs if s.name == "vacuum")
        assert vacuum.category == DeviceCategory.ACTUATOR
        assert vacuum.context.zone == "downstairs"
        assert "downstairs" in vacuum.context.reachable_zones
        assert "hallway" in vacuum.context.reachable_zones

    async def test_camera_is_sensor(self, channel: MockChannel) -> None:
        specs = await channel.discover_devices()
        camera = next(s for s in specs if s.name == "camera")
        assert camera.category == DeviceCategory.SENSOR
        assert camera.context.zone == "front_yard"

    async def test_sprinkler_is_controller(self, channel: MockChannel) -> None:
        specs = await channel.discover_devices()
        sprinkler = next(s for s in specs if s.name == "sprinkler")
        assert sprinkler.category == DeviceCategory.CONTROLLER
        assert sprinkler.context.zone == "garden"

    async def test_all_specs_are_valid(self, channel: MockChannel) -> None:
        specs = await channel.discover_devices()
        for spec in specs:
            assert spec.validate(), f"{spec.name} failed validation"

    async def test_populates_internal_devices(self, channel: MockChannel) -> None:
        await channel.discover_devices()
        # 4 devices stored by native name + 4 by canonical ID (mock:{name})
        assert len(channel._devices) == 8


@pytest.mark.asyncio
class TestSendCommand:
    async def test_valid_command_succeeds(self, channel: MockChannel) -> None:
        await channel.discover_devices()
        result = await channel.send_command("vacuum", "vacuum", {})
        assert result["status"] == "ok"
        assert result["device_id"] == "vacuum"
        assert result["action"] == "vacuum"

    async def test_invalid_capability_raises(self, channel: MockChannel) -> None:
        """Security Rule S2: undeclared capabilities MUST be rejected."""
        await channel.discover_devices()
        with pytest.raises(ValueError, match="not a declared capability"):
            await channel.send_command("vacuum", "fly", {})

    async def test_unknown_device_raises(self, channel: MockChannel) -> None:
        await channel.discover_devices()
        with pytest.raises(ValueError, match="Unknown device"):
            await channel.send_command("nonexistent", "vacuum", {})

    async def test_mop_capability_on_vacuum(self, channel: MockChannel) -> None:
        """Vacuum device has mop capability."""
        await channel.discover_devices()
        result = await channel.send_command("vacuum", "mop", {})
        assert result["status"] == "ok"

    async def test_mop_device_cannot_vacuum(self, channel: MockChannel) -> None:
        """Mop device only has 'mop' capability, not 'vacuum'."""
        await channel.discover_devices()
        with pytest.raises(ValueError, match="not a declared capability"):
            await channel.send_command("mop", "vacuum", {})

    async def test_camera_snapshot(self, channel: MockChannel) -> None:
        await channel.discover_devices()
        result = await channel.send_command("camera", "snapshot", {})
        assert result["status"] == "ok"

    async def test_camera_cannot_vacuum(self, channel: MockChannel) -> None:
        await channel.discover_devices()
        with pytest.raises(ValueError, match="not a declared capability"):
            await channel.send_command("camera", "vacuum", {})

    async def test_command_before_discover_raises(self, channel: MockChannel) -> None:
        """No devices registered before discover_devices is called."""
        with pytest.raises(ValueError, match="Unknown device"):
            await channel.send_command("vacuum", "vacuum", {})


@pytest.mark.asyncio
class TestGetDeviceStatus:
    async def test_returns_idle_by_default(self, channel: MockChannel) -> None:
        await channel.discover_devices()
        status = await channel.get_device_status("vacuum")
        assert status == DeviceStatus.IDLE

    async def test_unknown_device_raises(self, channel: MockChannel) -> None:
        await channel.discover_devices()
        with pytest.raises(ValueError, match="Unknown device"):
            await channel.get_device_status("nonexistent")


@pytest.mark.asyncio
class TestWaitForState:
    async def test_always_returns_true(self, channel: MockChannel) -> None:
        result = await channel.wait_for_state("any", DeviceStatus.IDLE)
        assert result is True


@pytest.mark.asyncio
class TestDiscoverRoomZones:
    async def test_returns_empty_list(self, channel: MockChannel) -> None:
        result = await channel.discover_room_zones("vacuum")
        assert result == []


@pytest.mark.asyncio
class TestConnectDisconnect:
    async def test_connect_is_noop(self, channel: MockChannel) -> None:
        await channel.connect({})  # should not raise

    async def test_disconnect_is_noop(self, channel: MockChannel) -> None:
        await channel.disconnect()  # should not raise
