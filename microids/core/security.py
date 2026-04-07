"""Security foundations for the microids framework.

OpenClaw lessons — day 0, not day 60. Every security rule is enforced
from the first line of code, not bolted on after the first incident.

Security Rules implemented:
  S2 — validate_command: capability whitelist before dispatch
  S4 — safe_compare: timing-safe string comparison
  S6 — normalize_device_id: canonical device ID format
  S8 — wrap_untrusted: boundary markers for LLM prompts
  S9 — sanitize_for_log: strip secrets before logging
"""

from __future__ import annotations

import hmac
import re

from microids.models import Device


# ---------------------------------------------------------------------------
# S2 — Command Validation
# ---------------------------------------------------------------------------


def validate_command(device: Device, action: str) -> bool:
    """Check *action* against the device's declared capabilities.

    Every channel MUST call this before dispatching a command.
    Returns True only if *action* matches a capability name in
    ``device.spec.capabilities``.
    """
    return any(cap.name == action for cap in device.spec.capabilities)


# ---------------------------------------------------------------------------
# S4 — Timing-Safe Comparison
# ---------------------------------------------------------------------------


def safe_compare(a: str, b: str) -> bool:
    """Timing-safe string comparison — never use ``==`` for secrets.

    Wraps :func:`hmac.compare_digest` so callers don't need to remember.
    """
    return hmac.compare_digest(a, b)


# ---------------------------------------------------------------------------
# S6 — Canonical Device ID
# ---------------------------------------------------------------------------


def normalize_device_id(channel_type: str, native_id: str) -> str:
    """Return canonical device ID in ``{channel_type}:{native_id}`` format.

    Both parts must be non-empty strings; raises :class:`ValueError` otherwise.
    """
    if not channel_type:
        raise ValueError("channel_type must be non-empty")
    if not native_id:
        raise ValueError("native_id must be non-empty")
    return f"{channel_type}:{native_id}"


# ---------------------------------------------------------------------------
# S8 — Untrusted Content Wrapping
# ---------------------------------------------------------------------------


def wrap_untrusted(label: str, content: str) -> str:
    """Wrap user-supplied content with boundary markers before sending to LLM.

    Returns::

        [UNTRUSTED: {label}]
        {content}
        [/UNTRUSTED: {label}]
    """
    return f"[UNTRUSTED: {label}]\n{content}\n[/UNTRUSTED: {label}]"


# ---------------------------------------------------------------------------
# S9 — Log Sanitization
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Bearer tokens (typically JWT starting with ey)
    (re.compile(r"Bearer\s+\S+", re.IGNORECASE), "Bearer [REDACTED]"),
    # OpenAI-style API keys
    (re.compile(r"sk-\S+"), "[REDACTED]"),
    # Generic key=value secrets
    (re.compile(r"(api_key=)\S+", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"(token=)\S+", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"(password=)\S+", re.IGNORECASE), r"\1[REDACTED]"),
]


def sanitize_for_log(text: str) -> str:
    """Replace common secret patterns with ``[REDACTED]`` before logging."""
    result = text
    for pattern, replacement in _SECRET_PATTERNS:
        result = pattern.sub(replacement, result)
    return result
