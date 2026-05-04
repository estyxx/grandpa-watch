"""
Full integration test — webcam + YOLOv8 detection + Telegram alerts + /stato command.

- Green box = standing/sitting (ok)
- Red box = ON FLOOR (triggers alert after threshold)

Press Q to quit.
"""

from dotenv import load_dotenv

load_dotenv()

from datetime import datetime, timedelta  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from fall_watch.config import Config  # noqa: E402
from fall_watch.detector import _is_lying_down, load_model  # noqa: E402
from fall_watch.notifier import TelegramNotifier  # noqa: E402

FALL_THRESHOLD_SECONDS = 10  # shorter than production for testing
ALERT_COOLDOWN_SECONDS = 30
NOT_ON_FLOOR_STREAK_MAX = 8  # consecutive "ok" frames before declaring all clear


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def main() -> None:
    config = Config.load()
    notifier = TelegramNotifier(config)
    model = load_model()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open webcam — try running from Warp or grant camera permissions")
        return

    _log("✅ Webcam open")
    notifier.send_startup()
    _log("✅ Startup message sent to Telegram")

    on_floor_since: datetime | None = None
    alert_sent_at: datetime | None = None
    latest_frame: np.ndarray | None = None
    was_on_floor = False
    not_on_floor_streak = 0
    update_offset = 0

    while True:
        # --- Poll Telegram for commands ---
        commands, update_offset = notifier.poll_commands(update_offset)
        for chat_id, cmd in commands:
            match cmd:
                case "/stato":
                    _log(f"📲 /stato from chat {chat_id}")
                    notifier.send_status_reply(chat_id, latest_frame, on_floor_since)
                case _:
                    _log(f"⚙️  Unknown command '{cmd}' — ignored")

        ret, frame = cap.read()
        if not ret:
            break

        latest_frame = frame
        now = datetime.now()
        results = model(frame, verbose=False)
        display = frame.copy()
        person_on_floor = False

        for result in results:
            if result.keypoints is None:
                continue

            keypoints_data = result.keypoints.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []

            for i, person_kps in enumerate(keypoints_data):
                on_floor = _is_lying_down(person_kps, frame.shape[0])
                if on_floor:
                    person_on_floor = True

                color = (0, 0, 255) if on_floor else (0, 200, 0)
                label = "ON FLOOR" if on_floor else "ok"

                if i < len(boxes):
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                    )

                for kp in person_kps:
                    x, y, conf = kp
                    if conf > 0.3:
                        cv2.circle(display, (int(x), int(y)), 4, color, -1)

        # --- State machine with hysteresis ---
        if person_on_floor:
            not_on_floor_streak = 0

            if on_floor_since is None:
                on_floor_since = now
                _log("⚠️  On floor — timer started")

            seconds_on_floor = (now - on_floor_since).total_seconds()
            cooldown_ok = alert_sent_at is None or (
                now - alert_sent_at > timedelta(seconds=ALERT_COOLDOWN_SECONDS)
            )

            if seconds_on_floor >= FALL_THRESHOLD_SECONDS and cooldown_ok:
                _log(f"🚨 Sending alert! On floor for {seconds_on_floor:.0f}s")
                notifier.send_fall_alert(seconds_on_floor / 60, frame)
                alert_sent_at = now

            remaining = max(0, FALL_THRESHOLD_SECONDS - seconds_on_floor)
            label_text = f"Alert in {remaining:.0f}s" if alert_sent_at is None else "ALERT SENT"
            cv2.putText(
                display, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
            )

            was_on_floor = True

        else:
            if was_on_floor:
                not_on_floor_streak += 1
                cv2.putText(
                    display,
                    f"Checking... {not_on_floor_streak}/{NOT_ON_FLOOR_STREAK_MAX}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2,
                )

                if not_on_floor_streak >= NOT_ON_FLOOR_STREAK_MAX:
                    _log("✅ Got up — sending all clear")
                    if alert_sent_at is not None:
                        notifier.send_all_clear(frame)
                    on_floor_since = None
                    alert_sent_at = None
                    was_on_floor = False
                    not_on_floor_streak = 0

        status = "ON FLOOR 🔴" if person_on_floor else "ok 🟢"
        cv2.putText(
            display,
            status,
            (10, display.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("fall-watch — integration test (Q to quit)", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    _log("👋 Test ended")


if __name__ == "__main__":
    main()
