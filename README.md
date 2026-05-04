# 👁️ fall-watch

Real-time monitoring for vulnerable individuals using a camera stream. Analyses
pose via YOLOv8 and sends Telegram alerts in two situations:

- **Fall detection** — someone has been lying on the floor for longer than a
  configurable threshold.
- **Climb-out detection** — someone appears to be climbing over the bedrail
  (upright posture with one hip on the bed and one ankle off it).

No wearables required — works with any RTSP-compatible IP camera (tested with EZVIZ).

## How it works

1. A background thread keeps the RTSP buffer drained so the analyser always
   sees the latest frame, never a 5-second-stale one (see `camera.py`).
2. Every few seconds, YOLOv8-pose runs once on the current frame and produces
   a per-person `FrameAnalysis` with both `on_floor` and `climbing_out` flags.
3. Two state machines consume the same analysis:
   - `FallWatcher` — fires once the floor state persists past
     `FALL_THRESHOLD_MINUTES`, with hysteresis on the all-clear.
   - `ClimbWatcher` — fires when the climbing posture holds for
     `CLIMB_THRESHOLD_SECONDS`. Suppressed automatically when more than one
     person is in the frame (a helper is present, so it isn't a real climb-out).
4. Two polygon zones (`FLOOR_ROI`, `BED_ROI`) restrict where each signal
   counts, so the bed isn't read as the floor and only the bedrail crossing
   triggers a climb-out.
5. Both alerts and the all-clear include a JPEG snapshot, not just text.

False positives are minimised by requiring each state to persist over time
rather than alerting on any brief motion.

## Stack

- **YOLOv8-pose** — real-time body pose estimation
- **OpenCV** — RTSP stream capture
- **Telegram Bot API** — group alerts
- **uv** — package management
- **ruff** — linting & formatting
- **mypy** — type checking

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/youruser/fall-watch
cd fall-watch

# 2. Copy and fill in your secrets
cp .env.example .env

# 3. Install dependencies (uv will pin Python 3.13 automatically)
uv sync

# 4. Define the floor and bed ROI zones — see "Setting up ROIs" below.
#    This populates FLOOR_ROI and BED_ROI in .env.

# 5. Run the monitor
uv run fall-watch
```

You can run the monitor without `FLOOR_ROI` / `BED_ROI`, but you'll get more
false positives (any "lying down" pose anywhere in frame counts as a fall, and
climb-out detection is disabled until `BED_ROI` is set).

### Setting up ROIs

ROI polygons are stored as **absolute pixel coordinates** on the live RTSP
frame. Pixel `(88, 718)` means very different things on a 1280×720 snapshot
versus a 1920×1080 stream — so the image you click on **must match the
resolution of the live stream**. If it doesn't, the polygon lands in the
wrong place and detection silently breaks (this is a bug we hit in practice
when drawing ROIs on a 1280×725 Telegram snapshot of a 1920×1080 EZVIZ feed).

`scripts/setup_roi.py` opens a window and waits for clicks, so it must run
on a host with a display. The reliable flow is: capture the frame on the
host that has stream access (typically the Pi), copy it to a desktop
machine, run the ROI tool there, then copy the resulting env-vars back.

```bash
# 1. On the host with RTSP access (e.g. the Pi).
#    Load the env first — bash: `set -a; source .env; set +a`
ffmpeg -rtsp_transport tcp -i "$RTSP_URL" -frames:v 1 -q:v 2 -y /tmp/live_snapshot.jpg

# Optional sanity check — confirm the stream resolution matches the snapshot:
ffprobe -v error -select_streams v:0 -show_entries stream=width,height "$RTSP_URL"

# 2. Pull it to the desktop machine (skip if everything runs on one host):
scp pi@my-pi.local:/tmp/live_snapshot.jpg ~/Downloads/

# 3. Click the four corners of each zone — this writes back to the local .env:
uv run python scripts/setup_roi.py ~/Downloads/live_snapshot.jpg --zone floor
uv run python scripts/setup_roi.py ~/Downloads/live_snapshot.jpg --zone bed

# 4. If the tool ran on a different host than production, copy the new
#    FLOOR_ROI / BED_ROI lines into the production .env:
grep -E '^(FLOOR|BED)_ROI=' .env
# paste them on the Pi, then restart the service:
sudo systemctl restart fall-watch
```

Avoid VLC's "Take Snapshot" feature for this step — it tends to save at the
display resolution rather than the stream's native resolution, which is
exactly how the resolution-mismatch bug happens. `ffmpeg` directly against
the RTSP URL gives you a frame at the same resolution the monitor will see.

## Configuration

Copy `.env.example` to `.env` and fill in your values. All knobs live in
`config.py` and are loaded from the environment — nothing is hardcoded.

**Required**

| Variable | Description |
|---|---|
| `TELEGRAM_TOKEN` | Token from [@BotFather](https://t.me/botfather) |
| `TELEGRAM_CHAT_ID` | Group chat ID (negative number — get it via the [getUpdates API](https://core.telegram.org/bots/api#getupdates)) |
| `RTSP_URL` | Camera stream — typically `rtsp://admin:PASSWORD@CAMERA_IP:554/h264/ch01/main/av_stream` |

**Fall detection**

| Variable | Default | Description |
|---|---|---|
| `FALL_THRESHOLD_MINUTES` | `3` | Minutes on floor before alerting |
| `ALERT_COOLDOWN_MINUTES` | `15` | Minutes between repeated fall alerts |
| `NOT_ON_FLOOR_STREAK_MAX` | `3` | Consecutive "not on floor" frames before all-clear (hysteresis) |
| `FLOOR_ROI` | unset | Floor polygon, e.g. `"x1,y1;x2,y2;x3,y3;x4,y4"` — usually set by `scripts/setup_roi.py` |

**Climb-out detection**

| Variable | Default | Description |
|---|---|---|
| `CLIMB_THRESHOLD_SECONDS` | `10` | Seconds of climbing posture before alerting |
| `CLIMB_ALERT_COOLDOWN_MINUTES` | `5` | Minutes between repeated climb-out alerts |
| `CLIMB_SUPPRESS_WHEN_SUPERVISED` | `true` | When `true`, no alert fires if a second person is in the frame (helper present) |
| `BED_ROI` | unset | Bed polygon — same format as `FLOOR_ROI`, set via `scripts/setup_roi.py --zone bed`. Climb-out detection is disabled until this is set. |

**Sampling and runtime**

| Variable | Default | Description |
|---|---|---|
| `FRAME_INTERVAL_SECONDS` | `5` | How often to sample a frame for analysis |
| `READER_POLL_INTERVAL` | `0.01` | Back-off (s) between failed reads in the background camera thread |
| `LOG_FILE` | `fall-watch.log` | Path of the rotating log file |
| `LOG_LEVEL` | `INFO` | Root log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### Getting the RTSP URL (EZVIZ)

The verification code is printed on the camera label. The URL format is:
```
rtsp://admin:VERIFICATION_CODE@CAMERA_IP:554/h264/ch01/main/av_stream
```

Find the camera IP from your router's device list or the EZVIZ app under
*Device Settings → Device Info*.

## Telegram commands

| Command | Audience | Description |
|---|---|---|
| `/status` | everyone | Live screenshot with current monitoring state |
| `/debug` | admin | Annotated frame showing pose keypoints, ROI zones, and per-person detection flags |
| `/config` | admin | Current effective configuration with secrets redacted — useful for remote diagnostics without SSH |

The `Audience` column reflects what's shown in each user's command menu in
Telegram. The split is purely a UI hint — the bot itself accepts any of these
commands from any group member, since the family group is private and trusted.

To configure the menu so that non-admins only see `/status` while admins see
all three, push two scoped command lists to Telegram once with
[`setMyCommands`](https://core.telegram.org/bots/api#setmycommands). BotFather
only sets the global default; per-admin scoping requires a one-shot API call.

```fish
# load TELEGRAM_TOKEN and TELEGRAM_CHAT_ID from .env first

curl -X POST "https://api.telegram.org/bot$TELEGRAM_TOKEN/setMyCommands" \
  -H "Content-Type: application/json" \
  -d "{
    \"commands\": [
      {\"command\": \"status\", \"description\": \"📷 Come sta nonno adesso\"},
      {\"command\": \"config\", \"description\": \"⚙️ Mostra configurazione attiva\"},
      {\"command\": \"debug\", \"description\": \"🔍 Frame con keypoints YOLO\"}
    ],
    \"scope\": {\"type\": \"chat_administrators\", \"chat_id\": $TELEGRAM_CHAT_ID}
  }"
```

The default (member-facing) list is set via BotFather → `Edit Commands` — list
only the public commands there (just `/status`).

## Dev

```bash
uv run ruff check src/ tests/    # lint
uv run ruff format src/ tests/   # format
uv run mypy src/                 # type check
uv run pytest                    # unit tests (camera, climb watcher, config render)
```

Two additional manual scripts under `tests/` exist for hands-on verification —
they aren't run by `pytest` because they touch the camera or open a window:

```bash
uv run python tests/test_detector.py             # webcam + YOLO + Telegram, end-to-end
uv run python tests/test_frame.py path/to/img.jpg  # detection on a single still image
```

Both wrap their side-effecting code in an `if __name__ == "__main__":` guard
so pytest's collection scan won't trigger them. If you add a new manual
script under `tests/`, follow the same pattern — otherwise pytest will run
its top-level code (and, in the case of Telegram, spam the family group)
every time someone calls `uv run pytest`.

## Deploying to a Raspberry Pi

The script is designed to run 24/7 on a Raspberry Pi 5 (4 GB). `uv` manages
its own Python 3.13 build, so the host system's Python version doesn't matter.

```bash
sudo nano /etc/systemd/system/fall-watch.service
```

```ini
[Unit]
Description=Fall Watch Monitor
After=network.target

[Service]
ExecStart=/home/pi/fall-watch/.venv/bin/fall-watch
WorkingDirectory=/home/pi/fall-watch
Restart=always
EnvironmentFile=/home/pi/fall-watch/.env

[Install]
WantedBy=multi-user.target
```

`WorkingDirectory` matters: `LOG_FILE` defaults to a relative path
(`fall-watch.log`) and the YOLOv8 weights are cached in the cwd on first run.

```bash
sudo systemctl enable fall-watch
sudo systemctl start fall-watch
sudo journalctl -u fall-watch -f   # tail logs
```

## Limitations

- Detection accuracy depends on camera angle — top-down or angled works better than flat-on
- IR night vision cameras work fine; the model handles grayscale well
- The pose model may struggle with heavy blankets or obstructions covering the person
