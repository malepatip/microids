"""Tests for the device Registry with zone-aware queries."""

from __future__ import annotations

import pytest

from microids.core.registry import Registry
from microids.models import (
    Device,
    DeviceCapability,
    DeviceCategory,
    DeviceContext,
    DeviceSpec,
    DeviceStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_spec(
    name: str = "vacuum",
    device_type: str = "vacuum",
    category: DeviceCategory = DeviceCategory.ACTUATOR,
    capabilities: list[DeviceCapability] | None = None,
    zone: str = "downstairs",
    reachable_zones: list[str] | None = None,
) -> DeviceSpec:
    """Build a valid DeviceSpec with sensible defaults."""
    if capabilities is None:
        capabilities = [DeviceCapability(name="vacuum")]
    if reachable_zones is None:
        reachable_zones = [zone]
    return DeviceSpec(
        name=name,
        device_type=device_type,
        category=category,
        capabilities=capabilities,
        context=DeviceContext(zone=zone, reachable_zones=reachable_zones),
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_valid_device():
    reg = Registry()
    spec = make_spec()
    device = await reg.register(spec)

    assert isinstance(device, Device)
    assert device.id == "mock:vacuum"
    assert device.spec is spec
    assert device.status == DeviceStatus.IDLE


@pytest.mark.asyncio
async def test_register_invalid_spec_empty_name():
    reg = Registry()
    spec = make_spec(name="")
    with pytest.raises(ValueError, match="Invalid DeviceSpec"):
        await reg.register(spec)


@pytest.mark.asyncio
async def test_register_invalid_spec_empty_capabilities():
    reg = Registry()
    spec = make_spec(capabilities=[])
    with pytest.raises(ValueError, match="Invalid DeviceSpec"):
        await reg.register(spec)


@pytest.mark.asyncio
async def test_register_invalid_spec_empty_device_type():
    reg = Registry()
    spec = make_spec(device_type="")
    with pytest.raises(ValueError, match="Invalid DeviceSpec"):
        await reg.register(spec)


# ---------------------------------------------------------------------------
# Duplicate / Idempotent Registration (Property 7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_name_updates_existing():
    reg = Registry()
    spec1 = make_spec(name="vac", capabilities=[DeviceCapability(name="vacuum")])
    spec2 = make_spec(name="vac", capabilities=[DeviceCapability(name="vacuum"), DeviceCapability(name="mop")])

    d1 = await reg.register(spec1)
    d2 = await reg.register(spec2)

    # Same device object, updated spec
    assert d1 is d2
    assert len(d2.spec.capabilities) == 2

    # Only one device in registry
    all_devices = await reg.list_all()
    assert len(all_devices) == 1


# ---------------------------------------------------------------------------
# Category Immutability (Property 21)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_name_different_category_raises():
    reg = Registry()
    spec1 = make_spec(name="thing", category=DeviceCategory.ACTUATOR)
    spec2 = make_spec(name="thing", category=DeviceCategory.SENSOR)

    await reg.register(spec1)
    with pytest.raises(ValueError, match="Cannot change category"):
        await reg.register(spec2)


# ---------------------------------------------------------------------------
# get_by_capability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_by_capability_returns_matching():
    reg = Registry()
    await reg.register(make_spec(name="vac", capabilities=[DeviceCapability(name="vacuum")]))
    await reg.register(make_spec(name="mop", capabilities=[DeviceCapability(name="mop")]))
    await reg.register(
        make_spec(name="combo", capabilities=[DeviceCapability(name="vacuum"), DeviceCapability(name="mop")])
    )

    vacuums = await reg.get_by_capability("vacuum")
    assert len(vacuums) == 2
    names = {d.spec.name for d in vacuums}
    assert names == {"vac", "combo"}


@pytest.mark.asyncio
async def test_get_by_capability_no_match():
    reg = Registry()
    await reg.register(make_spec(name="vac", capabilities=[DeviceCapability(name="vacuum")]))
    result = await reg.get_by_capability("fly")
    assert result == []


# ---------------------------------------------------------------------------
# get_by_capability_and_zone (zone-aware)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_by_capability_and_zone_respects_reachable():
    reg = Registry()
    # Vacuum can reach upstairs and hallway
    await reg.register(
        make_spec(
            name="upstairs_vac",
            zone="upstairs",
            reachable_zones=["upstairs", "hallway"],
            capabilities=[DeviceCapability(name="vacuum")],
        )
    )
    # Vacuum can only reach downstairs
    await reg.register(
        make_spec(
            name="downstairs_vac",
            zone="downstairs",
            reachable_zones=["downstairs"],
            capabilities=[DeviceCapability(name="vacuum")],
        )
    )

    # Query for upstairs — only upstairs_vac should match
    result = await reg.get_by_capability_and_zone("vacuum", "upstairs")
    assert len(result) == 1
    assert result[0].spec.name == "upstairs_vac"

    # Query for hallway — upstairs_vac can reach it
    result = await reg.get_by_capability_and_zone("vacuum", "hallway")
    assert len(result) == 1
    assert result[0].spec.name == "upstairs_vac"

    # Query for garden — nobody can reach it
    result = await reg.get_by_capability_and_zone("vacuum", "garden")
    assert result == []


@pytest.mark.asyncio
async def test_get_by_capability_and_zone_requires_both():
    """Device must have BOTH the capability AND the zone."""
    reg = Registry()
    # Has vacuum capability but NOT in garden zone
    await reg.register(
        make_spec(
            name="vac",
            zone="downstairs",
            reachable_zones=["downstairs"],
            capabilities=[DeviceCapability(name="vacuum")],
        )
    )
    # Has garden zone but NOT vacuum capability
    await reg.register(
        make_spec(
            name="sprinkler",
            device_type="sprinkler",
            category=DeviceCategory.CONTROLLER,
            zone="garden",
            reachable_zones=["garden"],
            capabilities=[DeviceCapability(name="water")],
        )
    )

    result = await reg.get_by_capability_and_zone("vacuum", "garden")
    assert result == []


# ---------------------------------------------------------------------------
# get_by_type
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_by_type_returns_matching():
    reg = Registry()
    await reg.register(make_spec(name="v1", device_type="vacuum"))
    await reg.register(make_spec(name="m1", device_type="mop"))
    await reg.register(make_spec(name="v2", device_type="vacuum"))

    vacuums = await reg.get_by_type("vacuum")
    assert len(vacuums) == 2
    names = {d.spec.name for d in vacuums}
    assert names == {"v1", "v2"}


@pytest.mark.asyncio
async def test_get_by_type_no_match():
    reg = Registry()
    await reg.register(make_spec(name="v1", device_type="vacuum"))
    result = await reg.get_by_type("drone")
    assert result == []


# ---------------------------------------------------------------------------
# Unregister
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unregister_existing_returns_true():
    reg = Registry()
    device = await reg.register(make_spec(name="vac"))
    result = await reg.unregister(device.id)
    assert result is True

    # Device is gone
    assert await reg.get(device.id) is None
    assert await reg.list_all() == []


@pytest.mark.asyncio
async def test_unregister_nonexistent_returns_false():
    reg = Registry()
    result = await reg.unregister("mock:nonexistent")
    assert result is False


# ---------------------------------------------------------------------------
# update_status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_status_works():
    reg = Registry()
    device = await reg.register(make_spec(name="vac"))
    assert device.status == DeviceStatus.IDLE

    await reg.update_status(device.id, DeviceStatus.BUSY)
    fetched = await reg.get(device.id)
    assert fetched is not None
    assert fetched.status == DeviceStatus.BUSY


@pytest.mark.asyncio
async def test_update_status_nonexistent_raises():
    reg = Registry()
    with pytest.raises(KeyError, match="Device not found"):
        await reg.update_status("mock:ghost", DeviceStatus.BUSY)


# ---------------------------------------------------------------------------
# get_capabilities_schema
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_capabilities_schema_format():
    reg = Registry()
    await reg.register(
        make_spec(
            name="vacuum",
            device_type="vacuum",
            category=DeviceCategory.ACTUATOR,
            zone="downstairs",
            reachable_zones=["downstairs"],
            capabilities=[
                DeviceCapability(name="vacuum"),
                DeviceCapability(name="mop"),
                DeviceCapability(name="dock"),
            ],
        )
    )

    schema = await reg.get_capabilities_schema()
    assert len(schema) == 1
    entry = schema[0]
    assert entry["device_id"] == "mock:vacuum"
    assert entry["device_type"] == "vacuum"
    assert entry["category"] == "actuator"
    assert entry["zone"] == "downstairs"
    assert entry["capabilities"] == ["vacuum", "mop", "dock"]


# ---------------------------------------------------------------------------
# list_all
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_all_returns_all():
    reg = Registry()
    await reg.register(make_spec(name="a"))
    await reg.register(make_spec(name="b"))
    await reg.register(make_spec(name="c"))

    all_devices = await reg.list_all()
    assert len(all_devices) == 3
    names = {d.spec.name for d in all_devices}
    assert names == {"a", "b", "c"}


@pytest.mark.asyncio
async def test_list_all_empty():
    reg = Registry()
    assert await reg.list_all() == []


# ---------------------------------------------------------------------------
# Canonical ID format (Security Rule S6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_canonical_id_format():
    reg = Registry()
    device = await reg.register(make_spec(name="my_device"), channel_type="ha")
    assert device.id == "ha:my_device"


@pytest.mark.asyncio
async def test_get_by_canonical_id():
    reg = Registry()
    device = await reg.register(make_spec(name="vac"), channel_type="mqtt")
    fetched = await reg.get("mqtt:vac")
    assert fetched is device
