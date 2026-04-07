"""Tests for the EventBus with recovery and suspension events."""

from __future__ import annotations

import asyncio

import pytest

from microids.core.events import DeviceEvent, EventBus


# ---------------------------------------------------------------------------
# Publish with no subscribers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_no_subscribers_no_error():
    bus = EventBus()
    event = DeviceEvent(source_device_id="dev-1", event_type="task_complete")
    await bus.publish(event)  # should not raise


# ---------------------------------------------------------------------------
# Subscribe and receive event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscribe_and_receive_event():
    bus = EventBus()
    received: list[DeviceEvent] = []

    async def handler(e: DeviceEvent) -> None:
        received.append(e)

    await bus.subscribe("task_complete", handler)
    event = DeviceEvent(source_device_id="dev-1", event_type="task_complete", payload={"zone": "kitchen"})
    await bus.publish(event)

    assert len(received) == 1
    assert received[0] is event
    assert received[0].payload == {"zone": "kitchen"}


# ---------------------------------------------------------------------------
# Multiple subscribers receive same event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_subscribers_receive_same_event():
    bus = EventBus()
    received_a: list[DeviceEvent] = []
    received_b: list[DeviceEvent] = []

    async def handler_a(e: DeviceEvent) -> None:
        received_a.append(e)

    async def handler_b(e: DeviceEvent) -> None:
        received_b.append(e)

    await bus.subscribe("sensor_reading", handler_a)
    await bus.subscribe("sensor_reading", handler_b)

    event = DeviceEvent(source_device_id="cam-1", event_type="sensor_reading")
    await bus.publish(event)

    assert len(received_a) == 1
    assert len(received_b) == 1
    assert received_a[0] is event
    assert received_b[0] is event


# ---------------------------------------------------------------------------
# Subscriber isolation: one handler raises, others still called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscriber_isolation_on_error():
    bus = EventBus()
    received: list[DeviceEvent] = []

    async def bad_handler(e: DeviceEvent) -> None:
        raise RuntimeError("boom")

    async def good_handler(e: DeviceEvent) -> None:
        received.append(e)

    await bus.subscribe("recovery_action", bad_handler)
    await bus.subscribe("recovery_action", good_handler)

    event = DeviceEvent(source_device_id="dev-1", event_type="recovery_action")
    await bus.publish(event)  # should not raise

    assert len(received) == 1
    assert received[0] is event


# ---------------------------------------------------------------------------
# Unsubscribe stops receiving events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsubscribe_stops_receiving():
    bus = EventBus()
    received: list[DeviceEvent] = []

    async def handler(e: DeviceEvent) -> None:
        received.append(e)

    sub_id = await bus.subscribe("goal_suspended", handler)
    await bus.publish(DeviceEvent(source_device_id="gw", event_type="goal_suspended"))
    assert len(received) == 1

    await bus.unsubscribe(sub_id)
    await bus.publish(DeviceEvent(source_device_id="gw", event_type="goal_suspended"))
    assert len(received) == 1  # no new event received


# ---------------------------------------------------------------------------
# History is bounded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_history_bounded():
    bus = EventBus(max_history=5)
    for i in range(10):
        await bus.publish(
            DeviceEvent(source_device_id=f"dev-{i}", event_type="status_changed")
        )

    history = bus.get_history()
    assert len(history) == 5
    # Oldest events (dev-0 through dev-4) should be dropped
    ids = [e.source_device_id for e in history]
    assert ids == [f"dev-{i}" for i in range(5, 10)]


# ---------------------------------------------------------------------------
# get_history with event_type filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_history_filter_by_event_type():
    bus = EventBus()
    await bus.publish(DeviceEvent(source_device_id="a", event_type="task_complete"))
    await bus.publish(DeviceEvent(source_device_id="b", event_type="task_failed"))
    await bus.publish(DeviceEvent(source_device_id="c", event_type="task_complete"))

    filtered = bus.get_history(event_type="task_complete")
    assert len(filtered) == 2
    assert all(e.event_type == "task_complete" for e in filtered)

    failed = bus.get_history(event_type="task_failed")
    assert len(failed) == 1
    assert failed[0].source_device_id == "b"


# ---------------------------------------------------------------------------
# get_history with limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_history_with_limit():
    bus = EventBus()
    for i in range(20):
        await bus.publish(
            DeviceEvent(source_device_id=f"dev-{i}", event_type="sensor_reading")
        )

    limited = bus.get_history(limit=5)
    assert len(limited) == 5
    # Should return the 5 most recent
    ids = [e.source_device_id for e in limited]
    assert ids == [f"dev-{i}" for i in range(15, 20)]
