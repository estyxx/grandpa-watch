import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from fall_watch.config import Config
from fall_watch.notifier import Notifier

logger = logging.getLogger(__name__)


@dataclass
class _FallState:
    on_floor_since: datetime | None = None
    alert_sent_at: datetime | None = None
    last_floor_frame: np.ndarray | None = field(default=None, repr=False)
    latest_frame: np.ndarray | None = field(default=None, repr=False)
    was_on_floor: bool = False
    not_on_floor_streak: int = 0


class FallWatcher:
    def __init__(self, config: Config, notifier: Notifier) -> None:
        self._config = config
        self._notifier = notifier
        self._state = _FallState()

    def observe(self, person_on_floor: bool, frame: np.ndarray, now: datetime) -> None:
        """Feed one detection signal into the state machine.

        Updates internal state, may trigger a fall alert or all-clear via the notifier.
        """
        self._state.latest_frame = frame
        if person_on_floor:
            self._on_floor(frame, now)
        else:
            self._off_floor()

    def handle_status_request(self, chat_id: str) -> None:
        """Reply to a /stato command using the current internal state and last frame."""
        self._notifier.send_status_reply(
            chat_id, self._state.latest_frame, self._state.on_floor_since
        )

    def _on_floor(self, frame: np.ndarray, now: datetime) -> None:
        self._state.not_on_floor_streak = 0

        if self._state.on_floor_since is None:
            self._state.on_floor_since = now
            logger.warning("⚠️  Person on floor — timer started")

        self._state.last_floor_frame = frame.copy()
        minutes_on_floor = (now - self._state.on_floor_since).total_seconds() / 60
        cooldown_ok = self._state.alert_sent_at is None or (
            now - self._state.alert_sent_at > timedelta(minutes=self._config.alert_cooldown_minutes)
        )

        if minutes_on_floor >= self._config.fall_threshold_minutes and cooldown_ok:
            logger.warning("🚨 Alerting! On floor for %.1fmin", minutes_on_floor)
            self._notifier.send_fall_alert(minutes_on_floor, frame)
            self._state.alert_sent_at = now

        self._state.was_on_floor = True

    def _off_floor(self) -> None:
        if not self._state.was_on_floor:
            return

        self._state.not_on_floor_streak += 1
        logger.info(
            "📊 Off floor streak: %d/%d",
            self._state.not_on_floor_streak,
            self._config.not_on_floor_streak_max,
        )

        if self._state.not_on_floor_streak >= self._config.not_on_floor_streak_max:
            logger.info("✅ Person got up")
            if self._state.alert_sent_at is not None:
                self._notifier.send_all_clear(self._state.last_floor_frame)
            self._state = _FallState()
