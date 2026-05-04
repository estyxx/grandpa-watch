# 👁️ fall-watch

Real-time fall detection for vulnerable individuals using a camera stream. Analyses
pose via YOLOv8 and sends a Telegram alert if someone has been lying on the floor
for longer than a configurable threshold.

No wearables required — works with any RTSP-compatible IP camera (tested with EZVIZ).

## How it works

1. Reads an RTSP stream from a local IP camera
2. Runs YOLOv8-pose on a frame every few seconds
3. Detects if a person's keypoints indicate a horizontal/floor position
4. If that state persists beyond `FALL_THRESHOLD_MINUTES` → sends a Telegram alert
5. Sends an all-clear when the person gets up

False positives are minimised by requiring the fallen state to persist over time
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

# 4. Test Telegram alerts first (no camera needed)
uv run python tests/test_telegram.py

# 5. Run the monitor
uv run fall-watch
```

## Configuration

Copy `.env.example` to `.env` and fill in your values:

| Variable | Description |
|---|---|
| `TELEGRAM_TOKEN` | Token from [@BotFather](https://t.me/botfather) |
| `TELEGRAM_CHAT_ID` | Group chat ID (negative number — get it via the [getUpdates API](https://core.telegram.org/bots/api#getupdates)) |
| `RTSP_URL` | Camera stream — typically `rtsp://admin:PASSWORD@CAMERA_IP:554/h264/ch01/main/av_stream` |
| `FALL_THRESHOLD_MINUTES` | Minutes on floor before alerting (default: `3`) |
| `ALERT_COOLDOWN_MINUTES` | Minutes between repeated alerts (default: `15`) |

### Getting the RTSP URL (EZVIZ)

The verification code is printed on the camera label. The URL format is:
```
rtsp://admin:VERIFICATION_CODE@CAMERA_IP:554/h264/ch01/main/av_stream
```

Find the camera IP from your router's device list or the EZVIZ app under
*Device Settings → Device Info*.

## Telegram commands

| Command | Description |
|---|---|
| `/status` | Live screenshot with current monitoring state |
| `/debug` | Annotated frame showing pose keypoints, ROI zones, and per-person detection flags |
| `/config` | Current effective configuration with secrets redacted — useful for remote diagnostics without SSH |

## Dev

```bash
uv run ruff check src/    # lint
uv run ruff format src/   # format
uv run mypy src/          # type check
```

## Deploying to a Raspberry Pi

The script is designed to run 24/7 on a Raspberry Pi 4 (4GB recommended).

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

```bash
sudo systemctl enable fall-watch
sudo systemctl start fall-watch
```

## Limitations

- Detection accuracy depends on camera angle — top-down or angled works better than flat-on
- IR night vision cameras work fine; the model handles grayscale well
- The pose model may struggle with heavy blankets or obstructions covering the person
