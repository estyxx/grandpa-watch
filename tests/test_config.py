"""Unit tests for Config.render().

Pure-Python: no camera, no model, no Telegram.
"""

import dataclasses

import pytest

from fall_watch.config import (
    _SENSITIVE_FIELDS,
    Config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RTSP_WITH_CREDS = "rtsp://admin:secret123@192.168.1.100:554/stream"
_RTSP_NO_CREDS = "rtsp://192.168.1.100/stream"
_FLOOR_ROI: tuple[tuple[int, int], ...] = ((0, 0), (100, 0), (100, 100), (0, 100))
_BED_ROI: tuple[tuple[int, int], ...] = ((10, 10), (50, 10), (50, 50), (10, 50))


def _make_config(**kwargs: object) -> Config:
    defaults: dict[str, object] = {
        "rtsp_url": _RTSP_WITH_CREDS,
        "telegram_token": "bot123:ABC-DEF1234",
        "telegram_chat_id": "-100123456789",
    }
    defaults.update(kwargs)
    return Config(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Sensitive field masking
# ---------------------------------------------------------------------------


def test_sensitive_fields_masked() -> None:
    config = _make_config()
    rendered = config.render()
    assert "bot123:ABC-DEF1234" not in rendered
    assert "-100123456789" not in rendered
    assert rendered.count("***") >= len(_SENSITIVE_FIELDS)


def test_sensitive_field_set_covers_token_and_chat_id() -> None:
    assert "telegram_token" in _SENSITIVE_FIELDS
    assert "telegram_chat_id" in _SENSITIVE_FIELDS


# ---------------------------------------------------------------------------
# RTSP URL redaction
# ---------------------------------------------------------------------------


def test_rtsp_password_redacted_host_preserved() -> None:
    config = _make_config(rtsp_url=_RTSP_WITH_CREDS)
    rendered = config.render()
    assert "secret123" not in rendered
    assert "192.168.1.100" in rendered
    assert "admin" in rendered
    assert "554" in rendered


def test_rtsp_without_credentials_unchanged() -> None:
    config = _make_config(rtsp_url=_RTSP_NO_CREDS)
    rendered = config.render()
    assert "192.168.1.100" in rendered
    assert "***" not in rendered.split("rtsp_url")[1].split("\n")[0]


# ---------------------------------------------------------------------------
# ROI formatting
# ---------------------------------------------------------------------------


def test_roi_shows_point_count_when_set() -> None:
    config = _make_config(floor_roi=_FLOOR_ROI, bed_roi=_BED_ROI)
    rendered = config.render()
    assert rendered.count("4 points") == 2


def test_roi_shows_not_set_when_absent() -> None:
    config = _make_config(floor_roi=None, bed_roi=None)
    rendered = config.render()
    assert rendered.count("not set") == 2


# ---------------------------------------------------------------------------
# Duration units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(  # type: ignore[misc]
    ("field", "value", "expected_fragment"),
    [
        ("fall_threshold_minutes", 3.0, "3.0 min"),
        ("alert_cooldown_minutes", 15.0, "15.0 min"),
        ("climb_alert_cooldown_minutes", 5.0, "5.0 min"),
        ("frame_interval_seconds", 5, "5 s"),
        ("climb_threshold_seconds", 10, "10 s"),
        ("reader_poll_interval", 0.01, "0.01 s"),
    ],
)
def test_duration_field_has_unit(field: str, value: object, expected_fragment: str) -> None:
    config = _make_config(**{field: value})
    rendered = config.render()
    assert expected_fragment in rendered


# ---------------------------------------------------------------------------
# Completeness and size
# ---------------------------------------------------------------------------


def test_all_fields_appear_in_output() -> None:
    config = _make_config()
    rendered = config.render()
    for field in dataclasses.fields(config):
        assert field.name in rendered, f"Field {field.name!r} missing from render output"


def test_output_under_telegram_limit() -> None:
    config = _make_config(floor_roi=_FLOOR_ROI, bed_roi=_BED_ROI)
    rendered = config.render()
    assert len(rendered) < 3500, f"render() output is {len(rendered)} chars — too long for Telegram"


def test_output_is_code_block() -> None:
    config = _make_config()
    rendered = config.render()
    assert rendered.startswith("```")
    assert rendered.endswith("```")
