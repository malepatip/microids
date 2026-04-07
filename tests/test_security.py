"""Tests for microids.core.security — Security Rules S2, S4, S6, S8, S9."""

from __future__ import annotations

import pytest

from microids.core.security import (
    normalize_device_id,
    safe_compare,
    sanitize_for_log,
    validate_command,
    wrap_untrusted,
)
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


def _make_device(*capability_names: str) -> Device:
    """Build a Device with the given capability names."""
    caps = [DeviceCapability(name=n) for n in capability_names]
    spec = DeviceSpec(
        name="test-device",
        device_type="vacuum",
        category=DeviceCategory.ACTUATOR,
        capabilities=caps,
        context=DeviceContext(zone="living_room", reachable_zones=["living_room"]),
    )
    return Device(id="dev-001", spec=spec, status=DeviceStatus.IDLE)


# ---------------------------------------------------------------------------
# S2 — validate_command
# ---------------------------------------------------------------------------


class TestValidateCommand:
    def test_valid_action(self) -> None:
        device = _make_device("vacuum", "return_to_base")
        assert validate_command(device, "vacuum") is True

    def test_invalid_action(self) -> None:
        device = _make_device("vacuum", "return_to_base")
        assert validate_command(device, "mop") is False

    def test_empty_action(self) -> None:
        device = _make_device("vacuum")
        assert validate_command(device, "") is False

    def test_no_capabilities(self) -> None:
        device = _make_device()  # no capabilities
        assert validate_command(device, "vacuum") is False

    def test_case_sensitive(self) -> None:
        device = _make_device("Vacuum")
        assert validate_command(device, "vacuum") is False


# ---------------------------------------------------------------------------
# S4 — safe_compare
# ---------------------------------------------------------------------------


class TestSafeCompare:
    def test_equal_strings(self) -> None:
        assert safe_compare("secret123", "secret123") is True

    def test_unequal_strings(self) -> None:
        assert safe_compare("secret123", "secret456") is False

    def test_empty_strings(self) -> None:
        assert safe_compare("", "") is True

    def test_one_empty(self) -> None:
        assert safe_compare("abc", "") is False


# ---------------------------------------------------------------------------
# S6 — normalize_device_id
# ---------------------------------------------------------------------------


class TestNormalizeDeviceId:
    def test_canonical_format(self) -> None:
        assert normalize_device_id("ha", "light.kitchen") == "ha:light.kitchen"

    def test_empty_channel_type_raises(self) -> None:
        with pytest.raises(ValueError, match="channel_type"):
            normalize_device_id("", "abc")

    def test_empty_native_id_raises(self) -> None:
        with pytest.raises(ValueError, match="native_id"):
            normalize_device_id("mqtt", "")

    def test_both_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_device_id("", "")


# ---------------------------------------------------------------------------
# S8 — wrap_untrusted
# ---------------------------------------------------------------------------


class TestWrapUntrusted:
    def test_basic_wrapping(self) -> None:
        result = wrap_untrusted("user_goal", "Clean the house")
        assert result == (
            "[UNTRUSTED: user_goal]\n"
            "Clean the house\n"
            "[/UNTRUSTED: user_goal]"
        )

    def test_empty_content(self) -> None:
        result = wrap_untrusted("label", "")
        assert "[UNTRUSTED: label]" in result
        assert "[/UNTRUSTED: label]" in result


# ---------------------------------------------------------------------------
# S9 — sanitize_for_log
# ---------------------------------------------------------------------------


class TestSanitizeForLog:
    def test_bearer_token(self) -> None:
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig"
        assert "eyJ" not in sanitize_for_log(text)
        assert "[REDACTED]" in sanitize_for_log(text)

    def test_openai_key(self) -> None:
        text = "Using key sk-abc123def456"
        assert "sk-abc" not in sanitize_for_log(text)
        assert "[REDACTED]" in sanitize_for_log(text)

    def test_api_key_param(self) -> None:
        text = "url?api_key=supersecret&other=1"
        sanitized = sanitize_for_log(text)
        assert "supersecret" not in sanitized
        assert "api_key=[REDACTED]" in sanitized

    def test_token_param(self) -> None:
        text = "token=my_secret_token"
        sanitized = sanitize_for_log(text)
        assert "my_secret_token" not in sanitized

    def test_password_param(self) -> None:
        text = "password=hunter2"
        sanitized = sanitize_for_log(text)
        assert "hunter2" not in sanitized

    def test_no_secrets_unchanged(self) -> None:
        text = "Normal log message with no secrets"
        assert sanitize_for_log(text) == text
