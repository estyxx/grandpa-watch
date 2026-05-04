import logging
from datetime import datetime
from typing import Any, Protocol

import cv2
import numpy as np
import requests

from fall_watch.config import Config

logger = logging.getLogger(__name__)


class Notifier(Protocol):
    def send_startup(self) -> bool: ...
    def send_fall_alert(self, minutes_on_floor: float, frame: np.ndarray | None = None) -> bool: ...
    def send_all_clear(self, frame: np.ndarray | None = None) -> bool: ...
    def send_climbing_alert(self, frame: np.ndarray | None = None) -> bool: ...
    def send_status_reply(
        self, chat_id: str, frame: np.ndarray | None, on_floor_since: datetime | None
    ) -> bool: ...
    def send_debug_reply(self, chat_id: str, annotated_frame: np.ndarray, caption: str) -> bool: ...
    def send_config_reply(self, chat_id: str, rendered: str) -> bool: ...
    def poll_commands(self, offset: int) -> tuple[list[tuple[str, str]], int]: ...


class TelegramNotifier:
    def __init__(self, config: Config) -> None:
        self._config = config

    def _now(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _send_text(
        self, text: str, to_chat_id: str | None = None, parse_mode: str = "HTML"
    ) -> bool:
        chat_id = to_chat_id or self._config.telegram_chat_id
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{self._config.telegram_token}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode},
                timeout=10,
            )
            r.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error("❌ Telegram error: %s", e)
            return False

    def _send_photo(
        self,
        frame: np.ndarray | None,
        caption: str,
        to_chat_id: str | None = None,
    ) -> bool:
        """Encode frame as JPEG and send it to Telegram with a caption."""
        if frame is None:
            return self._send_text(caption, to_chat_id)

        chat_id = to_chat_id or self._config.telegram_chat_id
        try:
            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return self._send_text(caption, to_chat_id)

            r = requests.post(
                f"https://api.telegram.org/bot{self._config.telegram_token}/sendPhoto",
                data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"},
                files={"photo": ("alert.jpg", buffer.tobytes(), "image/jpeg")},
                timeout=15,
            )
            r.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error("❌ Telegram photo error: %s", e)
            return self._send_text(caption, to_chat_id)

    def poll_commands(self, offset: int) -> tuple[list[tuple[str, str]], int]:
        """
        Poll Telegram getUpdates (non-blocking, timeout=0).

        Returns a list of (chat_id, command) pairs for any bot commands found,
        and the next offset to pass on the following call to avoid reprocessing.
        """
        try:
            # POST is accepted by the Bot API and avoids params serialisation issues
            r = requests.post(
                f"https://api.telegram.org/bot{self._config.telegram_token}/getUpdates",
                json={"offset": offset, "timeout": 0, "allowed_updates": ["message"]},
                timeout=5,
            )
            r.raise_for_status()
            data: Any = r.json()  # untyped Bot API response
        except requests.RequestException as e:
            logger.error("❌ Telegram poll error: %s", e)
            return [], offset

        commands: list[tuple[str, str]] = []
        new_offset = offset

        for update in data.get("result", []):
            update_id: int = update["update_id"]
            new_offset = max(new_offset, update_id + 1)

            msg: Any = update.get("message", {})
            text: str = str(msg.get("text", ""))
            chat_id: str = str(msg.get("chat", {}).get("id", ""))

            if text.startswith("/") and chat_id:
                # Strip bot @mention: /stato@MyBot → /stato
                cmd = text.split("@")[0].split()[0]
                commands.append((chat_id, cmd))

        return commands, new_offset

    def send_status_reply(
        self,
        chat_id: str,
        frame: np.ndarray | None,
        on_floor_since: datetime | None,
    ) -> bool:
        """Reply to a /stato command with the latest frame and current state."""
        if on_floor_since is not None:
            minutes = (datetime.now() - on_floor_since).total_seconds() / 60
            status_line = (
                f"⚠️ <b>Nonno è a terra da {minutes:.0f} minut{'o' if minutes < 2 else 'i'}!</b>"
            )
        else:
            status_line = "✅ <b>Nonno sta bene.</b>"

        caption = f"{status_line}\n🕐 {self._now()}"
        return self._send_photo(frame, caption, chat_id)

    def send_fall_alert(self, minutes_on_floor: float, frame: np.ndarray | None = None) -> bool:
        caption = (
            f"🚨 <b>ATTENZIONE — Nonno a terra!</b>\n\n"
            f"A terra da <b>{minutes_on_floor:.0f} minuti</b>. Controllare subito!\n\n"
            f"🕐 {self._now()}"
        )
        return self._send_photo(frame, caption)

    def send_all_clear(self, frame: np.ndarray | None = None) -> bool:
        caption = f"✅ <b>Tutto ok</b> — il nonno si è rialzato.\n🕐 {self._now()}"
        return self._send_photo(frame, caption)

    def send_climbing_alert(self, frame: np.ndarray | None = None) -> bool:
        caption = (
            f"🚨 <b>ATTENZIONE — Nonno sta scavalcando la sponda!</b>\n\n"
            f"Controllare subito.\n\n"
            f"🕐 {self._now()}"
        )
        return self._send_photo(frame, caption)

    def send_debug_reply(self, chat_id: str, annotated_frame: np.ndarray, caption: str) -> bool:
        """Send an annotated debug frame to the requesting chat."""
        return self._send_photo(annotated_frame, caption, chat_id)

    def send_config_reply(self, chat_id: str, rendered: str) -> bool:
        """Reply to a /config command with the current effective configuration."""
        return self._send_text(rendered, to_chat_id=chat_id, parse_mode="MarkdownV2")

    def send_startup(self) -> bool:
        sent = self._send_text(
            "👋 <b>OcchioSuNonno attivo!</b>\nIl sistema di monitoraggio è operativo. 🟢\n"
            "Invia /stato per ricevere uno screenshot live.",
        )
        logger.info("✅ Startup message sent")
        return sent
