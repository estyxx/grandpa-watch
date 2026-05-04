"""Unit tests for ClimbWatcher supervision suppression.

These tests are pure-Python: no camera, no model, no Telegram.
"""

from datetime import datetime, timedelta
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from fall_watch.climb_watcher import ClimbWatcher
from fall_watch.config import Config
from fall_watch.detector import FrameAnalysis, PersonDetection
from fall_watch.notifier import Notifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BLANK_FRAME = np.zeros((100, 100, 3), dtype=np.uint8)
_BLANK_KPS = np.zeros((17, 3), dtype=np.float32)
_T0 = datetime(2026, 5, 4, 8, 0, 0)


def _make_config(*, suppress: bool = True) -> Config:
    return Config(
        rtsp_url="rtsp://fake",
        telegram_token="tok",
        telegram_chat_id="123",
        climb_threshold_seconds=10,
        climb_alert_cooldown_minutes=5.0,
        climb_suppress_when_supervised=suppress,
    )


def _make_person(*, climbing_out: bool = False, on_floor: bool = False) -> PersonDetection:
    return PersonDetection(
        keypoints=_BLANK_KPS,
        box=(0, 0, 10, 10),
        box_confidence=0.9,
        on_floor=on_floor,
        climbing_out=climbing_out,
    )


def _make_notifier() -> MagicMock:
    """Return a MagicMock typed as Notifier for mock-method access without type ignores."""
    return MagicMock(spec=Notifier)


def _make_watcher(notifier: MagicMock, *, suppress: bool = True) -> ClimbWatcher:
    return ClimbWatcher(_make_config(suppress=suppress), cast(Notifier, notifier))


# ---------------------------------------------------------------------------
# Tests — suppression when supervised (two people in frame)
# ---------------------------------------------------------------------------


def test_no_alert_when_supervised_and_climb_detected() -> None:
    """With two people in frame, a climbing detection must NOT fire an alert,
    and the streak counter (climbing_since) must be reset to None."""
    notifier = _make_notifier()
    watcher = _make_watcher(notifier, suppress=True)

    climber = _make_person(climbing_out=True)
    bystander = _make_person()
    analysis = FrameAnalysis(people=(climber, bystander))

    now = _T0
    for _ in range(5):
        watcher.observe(analysis, _BLANK_FRAME, now)
        now += timedelta(seconds=5)

    notifier.send_climbing_alert.assert_not_called()
    assert watcher._state.climbing_since is None


def test_streak_resets_when_supervised() -> None:
    """After a solo climbing detection starts the timer, a supervised frame
    must reset climbing_since to None so the threshold restarts cleanly."""
    notifier = _make_notifier()
    watcher = _make_watcher(notifier, suppress=True)

    solo = FrameAnalysis(people=(_make_person(climbing_out=True),))
    supervised = FrameAnalysis(people=(_make_person(climbing_out=True), _make_person()))

    # Start the timer with a solo detection.
    watcher.observe(solo, _BLANK_FRAME, _T0)
    assert watcher._state.climbing_since is not None

    # A supervised frame must clear the streak.
    watcher.observe(supervised, _BLANK_FRAME, _T0 + timedelta(seconds=5))
    assert watcher._state.climbing_since is None
    notifier.send_climbing_alert.assert_not_called()


# ---------------------------------------------------------------------------
# Tests — suppression disabled / single person (regression guards)
# ---------------------------------------------------------------------------


def test_alert_fires_for_single_person_after_threshold() -> None:
    """A solo climbing detection must still fire after the threshold — the
    suppression path must not break the happy path."""
    notifier = _make_notifier()
    watcher = _make_watcher(notifier, suppress=True)

    solo = FrameAnalysis(people=(_make_person(climbing_out=True),))

    now = _T0
    # Feed frames spanning exactly the threshold (10 s) in 5-second steps.
    for _ in range(3):  # 0 s, 5 s, 10 s — third frame crosses the threshold
        watcher.observe(solo, _BLANK_FRAME, now)
        now += timedelta(seconds=5)

    notifier.send_climbing_alert.assert_called_once()


def test_suppression_flag_off_allows_alert_with_two_people() -> None:
    """When CLIMB_SUPPRESS_WHEN_SUPERVISED=false, two people in frame must
    NOT suppress the alert."""
    notifier = _make_notifier()
    watcher = _make_watcher(notifier, suppress=False)

    climber = _make_person(climbing_out=True)
    bystander = _make_person()
    analysis = FrameAnalysis(people=(climber, bystander))

    now = _T0
    for _ in range(3):
        watcher.observe(analysis, _BLANK_FRAME, now)
        now += timedelta(seconds=5)

    notifier.send_climbing_alert.assert_called_once()


# ---------------------------------------------------------------------------
# Tests — FrameAnalysis.is_supervised property
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(  # type: ignore[misc, unused-ignore]
    ("n_people", "expected"),
    [
        (0, False),
        (1, False),
        (2, True),
        (3, True),
    ],
)
def test_is_supervised(n_people: int, expected: bool) -> None:
    people = tuple(_make_person() for _ in range(n_people))
    assert FrameAnalysis(people=people).is_supervised is expected
