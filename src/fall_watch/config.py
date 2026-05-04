import dataclasses
import os
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

_SENSITIVE_FIELDS: frozenset[str] = frozenset({"telegram_token", "telegram_chat_id"})
_DURATION_MIN_FIELDS: frozenset[str] = frozenset(
    {"fall_threshold_minutes", "alert_cooldown_minutes", "climb_alert_cooldown_minutes"}
)
_DURATION_SEC_FIELDS: frozenset[str] = frozenset(
    {"frame_interval_seconds", "climb_threshold_seconds", "reader_poll_interval"}
)
_ROI_FIELDS: frozenset[str] = frozenset({"floor_roi", "bed_roi"})


def _redact_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.password:
        return url
    user = parsed.username or ""
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    return urlunparse(parsed._replace(netloc=f"{user}:***@{host}{port}"))


def _render_value(name: str, value: object) -> str:
    if name in _SENSITIVE_FIELDS:
        return "***"
    if name == "rtsp_url":
        return _redact_url(str(value))
    if name in _ROI_FIELDS:
        if not isinstance(value, tuple):
            return "not set"
        return f"{len(value)} points"
    if name in _DURATION_MIN_FIELDS:
        return f"{value} min"
    if name in _DURATION_SEC_FIELDS:
        return f"{value} s"
    return str(value)


@dataclass(frozen=True)
class Config:
    rtsp_url: str
    telegram_token: str
    telegram_chat_id: str
    fall_threshold_minutes: float = 3.0
    alert_cooldown_minutes: float = 15.0
    frame_interval_seconds: int = 5
    not_on_floor_streak_max: int = 3
    floor_roi: tuple[tuple[int, int], ...] | None = None
    bed_roi: tuple[tuple[int, int], ...] | None = None
    climb_threshold_seconds: int = 10
    climb_alert_cooldown_minutes: float = 5.0
    climb_suppress_when_supervised: bool = True
    reader_poll_interval: float = 0.01

    def render(self) -> str:
        """Render config as a human-readable, multi-line string with secrets redacted."""
        fields = dataclasses.fields(self)
        pad = max(len(f.name) for f in fields)
        lines = ["fall-watch config", "─" * (pad + 14)]
        for field in fields:
            value_str = _render_value(field.name, getattr(self, field.name))
            lines.append(f"  {field.name:<{pad}}  {value_str}")
        return "```\n" + "\n".join(lines) + "\n```"

    @classmethod
    def load(cls) -> "Config":
        return cls(
            rtsp_url=os.environ["RTSP_URL"],
            telegram_token=os.environ["TELEGRAM_TOKEN"],
            telegram_chat_id=os.environ["TELEGRAM_CHAT_ID"],
            fall_threshold_minutes=float(os.getenv("FALL_THRESHOLD_MINUTES", "3")),
            alert_cooldown_minutes=float(os.getenv("ALERT_COOLDOWN_MINUTES", "15")),
            frame_interval_seconds=int(os.getenv("FRAME_INTERVAL_SECONDS", "5")),
            not_on_floor_streak_max=int(os.getenv("NOT_ON_FLOOR_STREAK_MAX", "3")),
            floor_roi=parse_polygon(os.getenv("FLOOR_ROI")),
            bed_roi=parse_polygon(os.getenv("BED_ROI")),
            climb_threshold_seconds=int(os.getenv("CLIMB_THRESHOLD_SECONDS", "10")),
            climb_alert_cooldown_minutes=float(os.getenv("CLIMB_ALERT_COOLDOWN_MINUTES", "5")),
            climb_suppress_when_supervised=os.getenv(
                "CLIMB_SUPPRESS_WHEN_SUPERVISED", "true"
            ).lower()
            not in ("0", "false", "no"),
            reader_poll_interval=float(os.getenv("READER_POLL_INTERVAL", "0.01")),
        )


def parse_polygon(raw: str | None) -> tuple[tuple[int, int], ...] | None:
    """Parse a polygon string of the form "x1,y1;x2,y2;...".

    Returns None for empty/missing input. Raises ValueError for malformed input
    so misconfiguration fails loudly at startup rather than silently disabling
    the ROI.
    """
    if not raw:
        return None
    points: list[tuple[int, int]] = []
    for pair in raw.split(";"):
        x_str, y_str = pair.split(",")
        points.append((int(x_str.strip()), int(y_str.strip())))
    if len(points) < 3:
        raise ValueError(f"Polygon needs at least 3 points, got {len(points)}: {raw!r}")
    return tuple(points)
