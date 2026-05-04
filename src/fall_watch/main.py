import logging
import logging.config
import os
import time
from datetime import datetime
from datetime import time as dt_time

from dotenv import load_dotenv

# Must run before local imports so Config.load() reads populated os.environ
load_dotenv()

import numpy as np  # noqa: E402

from fall_watch.camera import FrameReader, FreshFrameCapture  # noqa: E402
from fall_watch.climb_watcher import ClimbWatcher  # noqa: E402
from fall_watch.config import Config  # noqa: E402
from fall_watch.detector import (  # noqa: E402
    FrameAnalysis,
    analyse_frame,
    draw_debug_overlay,
    load_model,
)
from fall_watch.fall_watcher import FallWatcher  # noqa: E402
from fall_watch.notifier import Notifier, TelegramNotifier  # noqa: E402

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    log_file = os.getenv("LOG_FILE", "fall-watch.log")
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "console": {
                    "format": "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
                    "datefmt": "%H:%M:%S",
                },
                "file": {
                    "format": "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {"class": "logging.StreamHandler", "formatter": "console"},
                "file": {
                    "class": "logging.handlers.TimedRotatingFileHandler",
                    "filename": log_file,
                    "when": "D",  # daily, time set by atTime
                    "atTime": dt_time(20, 0),  # rotate at 20:00
                    "backupCount": 7,  # keep one week of daily logs
                    "encoding": "utf-8",
                    "formatter": "file",
                },
            },
            "root": {"level": log_level, "handlers": ["console", "file"]},
        }
    )
    logger.info("📝 Logging to %s at level %s", log_file, log_level)


def _reconnect(config: Config, cap: FrameReader) -> FreshFrameCapture:
    logger.warning("⚠️  Lost camera connection, retrying in 10s...")
    cap.release()
    while True:
        time.sleep(10)
        try:
            return FreshFrameCapture(config.rtsp_url, config.reader_poll_interval)
        except RuntimeError as e:
            logger.error("❌ Reconnect failed: %s — will retry in 10s", e)


def _handle_commands(
    notifier: Notifier,
    watcher: FallWatcher,
    offset: int,
    config: Config,
    last_frame: np.ndarray | None,
    last_analysis: FrameAnalysis | None,
) -> int:
    """Poll Telegram for commands and reply to /stato, /debug, and /config requests."""
    commands, new_offset = notifier.poll_commands(offset)
    for chat_id, cmd in commands:
        match cmd:
            case "/stato":
                logger.info("📲 /stato requested by chat %s", chat_id)
                watcher.handle_status_request(chat_id)
            case "/debug":
                logger.info("📲 /debug requested by chat %s", chat_id)
                _handle_debug(notifier, config, chat_id, last_frame, last_analysis)
            case "/config":
                logger.info("📲 /config requested by chat %s", chat_id)
                notifier.send_config_reply(chat_id, config.render())
            case _:
                logger.warning("⚙️  Unknown command '%s' from chat %s — ignored", cmd, chat_id)
    return new_offset


def _handle_debug(
    notifier: Notifier,
    config: Config,
    chat_id: str,
    last_frame: np.ndarray | None,
    last_analysis: FrameAnalysis | None,
) -> None:
    if last_frame is None or last_analysis is None:
        notifier.send_debug_reply(
            chat_id, np.zeros((100, 400, 3), dtype=np.uint8), "⏳ No frame captured yet."
        )
        return

    annotated = draw_debug_overlay(last_frame, last_analysis, config.floor_roi, config.bed_roi)

    n = len(last_analysis.people)
    person_lines = "\n".join(
        f"  👤 Person {i + 1}: {'🔴 LYING DOWN' if p.on_floor else '🟡 CLIMBING' if p.climbing_out else '🟢 OK'}  ({p.box_confidence:.0%})"
        for i, p in enumerate(last_analysis.people)
    )
    timestamp = datetime.now().strftime("%H:%M:%S")
    summary = (
        f"🔍 <b>Debug snapshot</b> — {timestamp}\n\n"
        f"👁 {n} person{'s' if n != 1 else ''} detected\n"
        f"{person_lines}\n\n"
        f"📐 FLOOR_ROI: {'✅ set' if config.floor_roi else '❌ not set'}\n"
        f"📐 BED_ROI: {'✅ set' if config.bed_roi else '❌ not set'}"
    )
    notifier.send_debug_reply(chat_id, annotated, summary)


def main() -> None:
    _setup_logging()
    logger.info("🟢 OcchioSuNonno starting...")

    config = Config.load()
    model = load_model()
    notifier = TelegramNotifier(config)
    watcher = FallWatcher(config, notifier)
    climb_watcher = ClimbWatcher(config, notifier)
    notifier.send_startup()
    cap: FrameReader = FreshFrameCapture(config.rtsp_url, config.reader_poll_interval)

    update_offset = 0
    last_frame: np.ndarray | None = None
    last_analysis: FrameAnalysis | None = None

    try:
        while True:
            update_offset = _handle_commands(
                notifier, watcher, update_offset, config, last_frame, last_analysis
            )

            if cap.failed:
                cap = _reconnect(config, cap)
                continue

            frame = cap.read_latest()
            if frame is None:
                time.sleep(config.frame_interval_seconds)
                continue

            now = datetime.now()
            last_analysis = analyse_frame(
                model, frame, floor_roi=config.floor_roi, bed_polygon=config.bed_roi
            )
            last_frame = frame
            watcher.observe(last_analysis.any_on_floor, frame, now)
            climb_watcher.observe(last_analysis, frame, now)

            time.sleep(config.frame_interval_seconds)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
