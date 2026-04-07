"""Tests for Router — channel resolution based on device channel_config."""

from __future__ import annotations

import pytest

from microids.core.router import Router
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


def _make_device(
    name: str = "test-device",
    channel_config: dict | None = None,
) -> Device:
    """Create a minimal Device with optional channel_config."""
    return Device(
        id=name,
        spec=DeviceSpec(
            name=name,
            device_type="vacuum",
            category=DeviceCategory.ACTUATOR,
            capabilities=[DeviceCapability(name="vacuum")],
            context=DeviceContext(zone="room", reachable_zones=["room"]),
            channel_config=channel_config or {},
        ),
        status=DeviceStatus.IDLE,
    )


class _FakeChannel:
    """Minimal stand-in for a channel instance."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __repr__(self) -> str:
        return f"FakeChannel({self.label!r})"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRouterRouting:
    """Route device with explicit channel_config to correct channel."""

    def test_route_device_with_explicit_channel_config(self) -> None:
        mock_ch = _FakeChannel("mock")
        ha_ch = _FakeChannel("ha")

        router = Router(default_channel=mock_ch)
        router.register_channel("mock", mock_ch)
        router.register_channel("homeassistant", ha_ch)

        device = _make_device("cam", channel_config={"type": "homeassistant"})
        assert router.route(device) is ha_ch

    def test_route_device_without_channel_config_uses_default(self) -> None:
        mock_ch = _FakeChannel("mock")
        router = Router(default_channel=mock_ch)

        device = _make_device("vac")  # no channel_config
        assert router.route(device) is mock_ch

    def test_route_with_no_default_raises_runtime_error(self) -> None:
        router = Router()  # no default, no channels

        device = _make_device("vac")
        with pytest.raises(RuntimeError, match="No channel found"):
            router.route(device)

    def test_register_channel_sets_first_as_default(self) -> None:
        router = Router()  # no default at construction
        first = _FakeChannel("first")
        second = _FakeChannel("second")

        router.register_channel("first", first)
        router.register_channel("second", second)

        # Device with no channel_config should get the first-registered channel
        device = _make_device("vac")
        assert router.route(device) is first

    def test_multiple_channels_correct_one_selected(self) -> None:
        ch_a = _FakeChannel("a")
        ch_b = _FakeChannel("b")
        ch_c = _FakeChannel("c")

        router = Router()
        router.register_channel("alpha", ch_a)
        router.register_channel("beta", ch_b)
        router.register_channel("gamma", ch_c)

        dev_a = _make_device("d1", channel_config={"type": "alpha"})
        dev_b = _make_device("d2", channel_config={"type": "beta"})
        dev_c = _make_device("d3", channel_config={"type": "gamma"})

        assert router.route(dev_a) is ch_a
        assert router.route(dev_b) is ch_b
        assert router.route(dev_c) is ch_c

    def test_unknown_channel_type_falls_back_to_default(self) -> None:
        default_ch = _FakeChannel("default")
        router = Router(default_channel=default_ch)

        device = _make_device("vac", channel_config={"type": "nonexistent"})
        assert router.route(device) is default_ch

    def test_empty_channel_config_uses_default(self) -> None:
        default_ch = _FakeChannel("default")
        router = Router(default_channel=default_ch)

        device = _make_device("vac", channel_config={})
        assert router.route(device) is default_ch
