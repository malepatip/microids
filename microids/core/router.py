"""Router — routes device commands to the correct channel.

No ABC yet — concrete class. The ABC will be extracted in Phase 3
when multi-channel dispatch demands it.
"""

from __future__ import annotations

from typing import Any

from microids.models import Device


class Router:
    """Routes device commands to the correct channel.

    Resolves which channel handles a device based on
    device.spec.channel_config, with fallback to default channel.
    """

    def __init__(self, default_channel: Any = None) -> None:
        self._channels: dict[str, Any] = {}  # name -> channel instance
        self._default_channel: Any = default_channel

    def register_channel(self, name: str, channel: Any) -> None:
        """Register a channel under a name.

        The first channel registered becomes the default if none was
        set at construction time.
        """
        self._channels[name] = channel
        if self._default_channel is None:
            self._default_channel = channel

    def route(self, device: Device) -> Any:
        """Resolve which channel handles this device.

        Resolution order:
        1. Check device.spec.channel_config for a "type" key.
        2. If found, look up that channel by name.
        3. If not found or no channel_config, return default channel.
        4. If no default channel, raise RuntimeError.
        """
        channel_type = device.spec.channel_config.get("type") if device.spec.channel_config else None

        if channel_type and channel_type in self._channels:
            return self._channels[channel_type]

        if self._default_channel is not None:
            return self._default_channel

        raise RuntimeError(
            f"No channel found for device '{device.id}' and no default channel configured"
        )
