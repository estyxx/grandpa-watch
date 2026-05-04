import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from fall_watch.config import Config
from fall_watch.detector import FrameAnalysis
from fall_watch.notifier import Notifier

logger = logging.getLogger(__name__)


@dataclass
class _ClimbState:
    climbing_since: datetime | None = None
    alert_sent_at: datetime | None = None
    last_climb_frame: np.ndarray | None = field(default=None, repr=False)


class ClimbWatcher:
    def __init__(self, config: Config, notifier: Notifier) -> None:
        self._config = config
        self._notifier = notifier
        self._state = _ClimbState()

    def observe(self, analysis: FrameAnalysis, frame: np.ndarray, now: datetime) -> None:
        """Feed one frame analysis into the climb-out state machine."""
        if self._config.climb_suppress_when_supervised and analysis.is_supervised:
            self._state.climbing_since = None
            logger.debug("climb suppressed: %d people in frame", len(analysis.people))
            return

        if analysis.any_climbing_out:
            self._on_climbing(frame, now)
        else:
            self._state.climbing_since = None

    def _on_climbing(self, frame: np.ndarray, now: datetime) -> None:
        if self._state.climbing_since is None:
            self._state.climbing_since = now
            logger.warning("⚠️  Climb-out posture detected — timer started")

        self._state.last_climb_frame = frame.copy()
        seconds_climbing = (now - self._state.climbing_since).total_seconds()
        cooldown_ok = self._state.alert_sent_at is None or (
            now - self._state.alert_sent_at
            > timedelta(minutes=self._config.climb_alert_cooldown_minutes)
        )

        if seconds_climbing >= self._config.climb_threshold_seconds and cooldown_ok:
            logger.warning("🚨 Climb-out alert! Posture held for %.0fs", seconds_climbing)
            self._notifier.send_climbing_alert(frame)
            self._state.alert_sent_at = now
