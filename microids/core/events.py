"""In-process pub/sub EventBus for device-to-device communication.

Inspired by ROS2 topics and OpenClaw's agent-to-agent messaging.
Thread-safe via asyncio.Lock (Security Rule S11).
Events are in-process only — no network exposure (Security Rule S10).
Bounded event history for debugging (OpenClaw memory lesson).

Supported event types:
    task_complete, task_failed, status_changed, sensor_reading,
    recovery_action, circuit_breaker_open, circuit_breaker_close,
    goal_suspended, suspension_notification, goal_resumed, suspension_timeout
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# Callback type: async function that receives a DeviceEvent
EventHandler = Callable[["DeviceEvent"], Awaitable[None]]


@dataclass
class DeviceEvent:
    """An event emitted by a device or the framework."""

    source_device_id: str
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EventBus:
    """In-process pub/sub for device-to-device communication.

    Thread-safe via asyncio.Lock (Security Rule S11).
    Events are in-process only — no network exposure (Security Rule S10).
    Bounded event history for debugging (OpenClaw memory lesson).
    """

    def __init__(self, max_history: int = 1000) -> None:
        # event_type -> {subscription_id: handler}
        self._subscribers: dict[str, dict[str, EventHandler]] = {}
        self._lock = asyncio.Lock()
        self._history: deque[DeviceEvent] = deque(maxlen=max_history)

    async def publish(self, event: DeviceEvent) -> None:
        """Publish an event. All matching subscribers notified concurrently.

        Subscriber isolation: one failing handler doesn't block others.
        Uses asyncio.gather with return_exceptions=True.
        """
        async with self._lock:
            self._history.append(event)
            handlers = list(self._subscribers.get(event.event_type, {}).values())

        if not handlers:
            return

        results = await asyncio.gather(
            *(h(event) for h in handlers), return_exceptions=True
        )
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "EventBus handler error for event_type=%s: %s",
                    event.event_type,
                    result,
                )

    async def subscribe(self, event_type: str, handler: EventHandler) -> str:
        """Subscribe to an event type. Returns a unique subscription ID."""
        sub_id = str(uuid.uuid4())
        async with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = {}
            self._subscribers[event_type][sub_id] = handler
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """Remove a subscription by ID."""
        async with self._lock:
            for event_type, subs in self._subscribers.items():
                if subscription_id in subs:
                    del subs[subscription_id]
                    # Clean up empty event type entries
                    if not subs:
                        del self._subscribers[event_type]
                    return

    def get_history(
        self, event_type: Optional[str] = None, limit: int = 100
    ) -> list[DeviceEvent]:
        """Query recent events for debugging. Optionally filter by event_type."""
        if event_type is not None:
            events = [e for e in self._history if e.event_type == event_type]
        else:
            events = list(self._history)
        # Return the most recent `limit` events
        return events[-limit:]
