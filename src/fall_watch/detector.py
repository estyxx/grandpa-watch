import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

logger = logging.getLogger(__name__)

# COCO keypoint indices
_LEFT_SHOULDER = 5
_RIGHT_SHOULDER = 6
_LEFT_HIP = 11
_RIGHT_HIP = 12
_LEFT_ANKLE = 15
_RIGHT_ANKLE = 16

# BGR colours for the debug overlay
_COLOR_OK = (50, 200, 50)
_COLOR_ON_FLOOR = (30, 30, 220)
_COLOR_CLIMBING = (0, 200, 240)
_COLOR_FLOOR_ROI = (200, 200, 0)
_COLOR_BED_ROI = (0, 140, 255)
_COLOR_SHOULDER = (255, 120, 0)
_COLOR_HIP = (0, 165, 255)
_COLOR_ANKLE = (180, 0, 180)


@dataclass(frozen=True)
class PersonDetection:
    """Detection result for a single person in a frame."""

    keypoints: np.ndarray  # shape (17, 3): x, y, confidence — read-only by convention
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in original image coordinates
    box_confidence: float
    on_floor: bool
    climbing_out: bool


@dataclass(frozen=True)
class FrameAnalysis:
    """All per-person detections derived from a single YOLO inference."""

    people: tuple[PersonDetection, ...]

    @property
    def any_on_floor(self) -> bool:
        return any(p.on_floor for p in self.people)

    @property
    def any_climbing_out(self) -> bool:
        return any(p.climbing_out for p in self.people)

    @property
    def is_supervised(self) -> bool:
        """True when more than one person is in frame — assume nonno isn't alone."""
        return len(self.people) > 1


def load_model() -> YOLO:
    """Load YOLOv8 nano pose model — downloads ~6MB on first run."""
    model = YOLO("yolov8n-pose.pt")
    logger.info("✅ AI model loaded")
    return model


def _keypoint(kps: np.ndarray, idx: int, min_conf: float = 0.3) -> np.ndarray | None:
    """Return (x, y) for a keypoint if confidence is high enough, else None."""
    kp = kps[idx]
    return kp[:2] if kp[2] >= min_conf else None


def _is_lying_down(kps: np.ndarray, frame_height: int) -> bool:
    """
    Heuristic: person is on the floor when their body keypoints are more
    spread horizontally than vertically, OR shoulders and hips are at a
    similar vertical level (flat body).
    """
    left_shoulder, right_shoulder, left_hip, right_hip = (
        _keypoint(kps, _LEFT_SHOULDER),
        _keypoint(kps, _RIGHT_SHOULDER),
        _keypoint(kps, _LEFT_HIP),
        _keypoint(kps, _RIGHT_HIP),
    )
    visible = [p for p in (left_shoulder, right_shoulder, left_hip, right_hip) if p is not None]

    if len(visible) < 2:
        logger.debug("  lying_down: only %d/4 keypoints visible — skip", len(visible))
        return False

    y_coords = [p[1] for p in visible]
    x_coords = [p[0] for p in visible]
    vertical_spread = float(max(y_coords) - min(y_coords))
    horizontal_spread = float(max(x_coords) - min(x_coords))

    is_horizontal = horizontal_spread > vertical_spread * 1.5

    shoulder_ys = [p[1] for p in (left_shoulder, right_shoulder) if p is not None]
    hip_ys = [p[1] for p in (left_hip, right_hip) if p is not None]
    shoulder_hip_diff = (
        abs(float(np.mean(shoulder_ys)) - float(np.mean(hip_ys)))
        if shoulder_ys and hip_ys
        else None
    )
    flat_threshold = frame_height * 0.15
    is_flat = shoulder_hip_diff is not None and shoulder_hip_diff < flat_threshold

    logger.debug(
        "  lying_down: h_spread=%.0f v_spread=%.0f is_horizontal=%s | "
        "sh_hip_diff=%s flat_thresh=%.0f is_flat=%s → result=%s",
        horizontal_spread,
        vertical_spread,
        is_horizontal,
        f"{shoulder_hip_diff:.0f}" if shoulder_hip_diff is not None else "n/a",
        flat_threshold,
        is_flat,
        is_horizontal or is_flat,
    )
    return is_horizontal or is_flat


def _hip_in_zone(kps: np.ndarray, polygon: tuple[tuple[int, int], ...]) -> bool:
    """True if at least one visible hip keypoint is inside the polygon.

    Fails safe: returns False when no hip keypoints are visible, so an
    ambiguous frame never triggers a false positive in bed.
    """
    contour = np.array(polygon, dtype=np.int32)
    for idx in (_LEFT_HIP, _RIGHT_HIP):
        if (point := _keypoint(kps, idx)) is not None:
            if cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0:
                return True
    return False


def _is_climbing_out(
    kps: np.ndarray,
    frame_height: int,
    bed_polygon: tuple[tuple[int, int], ...] | None,
) -> bool:
    """Heuristic: person is climbing over the bedrail when posture is upright,
    a hip is inside the bed polygon, and at least one ankle is outside it."""
    if bed_polygon is None:
        logger.debug("  climbing: no BED_ROI configured — skip")
        return False

    visible_shoulders = [
        p for i in (_LEFT_SHOULDER, _RIGHT_SHOULDER) if (p := _keypoint(kps, i)) is not None
    ]
    visible_hips = [p for i in (_LEFT_HIP, _RIGHT_HIP) if (p := _keypoint(kps, i)) is not None]
    visible_ankles = [
        p for i in (_LEFT_ANKLE, _RIGHT_ANKLE) if (p := _keypoint(kps, i)) is not None
    ]

    if not (visible_shoulders and visible_hips and visible_ankles):
        logger.debug(
            "  climbing: insufficient keypoints — shoulders=%d hips=%d ankles=%d",
            len(visible_shoulders),
            len(visible_hips),
            len(visible_ankles),
        )
        return False

    mean_shoulder_y = float(np.mean([s[1] for s in visible_shoulders]))
    mean_hip_y = float(np.mean([h[1] for h in visible_hips]))
    upright_threshold = frame_height * 0.10

    # In image coords, smaller y = higher up. Upright means shoulders well above hips.
    is_upright = (mean_hip_y - mean_shoulder_y) > upright_threshold
    logger.debug(
        "  climbing: mean_shoulder_y=%.0f mean_hip_y=%.0f diff=%.0f threshold=%.0f is_upright=%s",
        mean_shoulder_y,
        mean_hip_y,
        mean_hip_y - mean_shoulder_y,
        upright_threshold,
        is_upright,
    )
    if not is_upright:
        return False

    bed_contour = np.array(bed_polygon, dtype=np.int32)
    hip_inside = any(
        cv2.pointPolygonTest(bed_contour, (float(h[0]), float(h[1])), False) >= 0
        for h in visible_hips
    )
    ankle_outside = any(
        cv2.pointPolygonTest(bed_contour, (float(a[0]), float(a[1])), False) < 0
        for a in visible_ankles
    )
    logger.debug(
        "  climbing: hip_inside_bed=%s ankle_outside_bed=%s → result=%s",
        hip_inside,
        ankle_outside,
        hip_inside and ankle_outside,
    )
    return hip_inside and ankle_outside


def _is_person_on_floor(
    person_kps: np.ndarray,
    frame_height: int,
    floor_roi: tuple[tuple[int, int], ...] | None,
) -> bool:
    if not _is_lying_down(person_kps, frame_height):
        logger.debug("  on_floor: not lying down → False")
        return False
    if floor_roi is None:
        logger.debug("  on_floor: lying down, no FLOOR_ROI → True")
        return True
    hip_ok = _hip_in_zone(person_kps, floor_roi)
    logger.debug("  on_floor: lying down, hip_in_floor_roi=%s → %s", hip_ok, hip_ok)
    return hip_ok


def _to_numpy(data: Any) -> np.ndarray:  # Any: ultralytics returns Tensor | ndarray
    """Convert a Tensor or ndarray to a numpy array."""
    if isinstance(data, np.ndarray):
        return data
    arr: np.ndarray = data.cpu().numpy()
    return arr


def analyse_frame(
    model: YOLO,
    frame: np.ndarray,
    floor_roi: tuple[tuple[int, int], ...] | None = None,
    bed_polygon: tuple[tuple[int, int], ...] | None = None,
) -> FrameAnalysis:
    """Run YOLO once on the frame and return a per-person analysis.

    `on_floor` and `climbing_out` per person share the same inference result,
    so callers always agree on what counts and the model runs exactly once.
    """
    results: list[Results] = model(frame, verbose=False)
    people: list[PersonDetection] = []

    for result in results:
        if result.keypoints is None or result.boxes is None:
            continue

        kps_array = _to_numpy(result.keypoints.data)
        boxes_xyxy = _to_numpy(result.boxes.xyxy)
        confidences = _to_numpy(result.boxes.conf)

        for person_kps, box_xyxy, conf in zip(kps_array, boxes_xyxy, confidences, strict=True):
            x1, y1, x2, y2 = (int(v) for v in box_xyxy)
            box_conf = float(conf)
            logger.debug("👤 Person conf=%.2f box=(%d,%d,%d,%d)", box_conf, x1, y1, x2, y2)
            on_floor = _is_person_on_floor(person_kps, frame.shape[0], floor_roi)
            climbing_out = _is_climbing_out(person_kps, frame.shape[0], bed_polygon)
            people.append(
                PersonDetection(
                    keypoints=person_kps,
                    box=(x1, y1, x2, y2),
                    box_confidence=box_conf,
                    on_floor=on_floor,
                    climbing_out=climbing_out,
                )
            )

    analysis = FrameAnalysis(people=tuple(people))
    logger.info(
        "🎞  Frame: %d person(s) detected | on_floor=%s climbing=%s",
        len(people),
        analysis.any_on_floor,
        analysis.any_climbing_out,
    )
    return analysis


def draw_debug_overlay(
    frame: np.ndarray,
    analysis: FrameAnalysis,
    floor_roi: tuple[tuple[int, int], ...] | None = None,
    bed_roi: tuple[tuple[int, int], ...] | None = None,
) -> np.ndarray:
    """Return an annotated copy of frame showing ROI zones, keypoints, and detection labels."""
    out = frame.copy()

    _draw_roi(out, floor_roi, _COLOR_FLOOR_ROI, "FLOOR ROI")
    _draw_roi(out, bed_roi, _COLOR_BED_ROI, "BED ROI")

    for person in analysis.people:
        _draw_person(out, person)

    if not analysis.people:
        _put_label(out, "NO PERSON DETECTED", (10, 30), _COLOR_ON_FLOOR)

    _draw_status_banner(out, analysis)

    return out


def _draw_status_banner(out: np.ndarray, analysis: FrameAnalysis) -> None:
    """Draw a coloured status banner at the bottom of the frame."""
    if analysis.any_on_floor:
        color = _COLOR_ON_FLOOR
        text = "  ON FLOOR"
    elif analysis.any_climbing_out:
        color = _COLOR_CLIMBING
        text = "  CLIMBING OUT"
    elif analysis.people:
        color = _COLOR_OK
        text = "  OK"
    else:
        color = (80, 80, 80)
        text = "  NO PERSON"

    h, w = out.shape[:2]
    banner_h = 36
    y0 = h - banner_h

    overlay = out.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), color, cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    cv2.putText(out, text, (4, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def _draw_roi(
    out: np.ndarray,
    polygon: tuple[tuple[int, int], ...] | None,
    color: tuple[int, int, int],
    label: str,
) -> None:
    if polygon is None:
        return
    pts = np.array(polygon, dtype=np.int32)

    # Filled semi-transparent zone
    overlay = out.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)

    # Solid border
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)

    # Centroid label
    cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
    _put_label(out, label, (cx - 40, cy), color)


def _draw_person(out: np.ndarray, person: PersonDetection) -> None:
    x1, y1, x2, y2 = person.box

    if person.on_floor:
        box_color = _COLOR_ON_FLOOR
        status = "LYING DOWN"
    elif person.climbing_out:
        box_color = _COLOR_CLIMBING
        status = "CLIMBING"
    else:
        box_color = _COLOR_OK
        status = "OK"

    cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)

    tag = f"{status}  {person.box_confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), box_color, cv2.FILLED)
    cv2.putText(out, tag, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    kps = person.keypoints
    for idx, kp_color in [
        (_LEFT_SHOULDER, _COLOR_SHOULDER),
        (_RIGHT_SHOULDER, _COLOR_SHOULDER),
        (_LEFT_HIP, _COLOR_HIP),
        (_RIGHT_HIP, _COLOR_HIP),
        (_LEFT_ANKLE, _COLOR_ANKLE),
        (_RIGHT_ANKLE, _COLOR_ANKLE),
    ]:
        if (pt := _keypoint(kps, idx)) is not None:
            cv2.circle(out, (int(pt[0]), int(pt[1])), 6, kp_color, cv2.FILLED)
            cv2.circle(out, (int(pt[0]), int(pt[1])), 6, (255, 255, 255), 1)


def _put_label(
    out: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    cv2.putText(out, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(out, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
