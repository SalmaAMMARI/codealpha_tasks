"""
Task 4: Object Detection and Tracking
Real-time pipeline using YOLOv8 for detection and SORT for multi-object tracking.
Supports webcam input, video file input, and single-image inference.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Optional imports with graceful messages
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from sort_tracker import SORTTracker  # local implementation below
    SORT_AVAILABLE = True
except ImportError:
    SORT_AVAILABLE = False


# ===========================================================================
# CONSTANTS
# ===========================================================================

# YOLO model size: "yolov8n" (fastest) -> "yolov8s" -> "yolov8m" -> "yolov8l" -> "yolov8x" (most accurate)
DEFAULT_MODEL    = "yolov8n.pt"
CONF_THRESHOLD   = 0.40        # minimum detection confidence
IOU_THRESHOLD    = 0.45        # NMS IoU threshold
MAX_TRACK_AGE    = 30          # frames to keep a lost track alive
MIN_HITS         = 3           # frames before confirming a track
IOU_TRACK        = 0.30        # IoU threshold for track association
DEFAULT_FPS_CAP  = 30          # cap on display FPS


# COCO class colors - one distinct color per class index (wraps at 80)
_PALETTE = [
    (56,  56,  255), (151, 157, 255), (31,  112, 255), (29,  178, 255),
    (49,  210, 207), (10,  249, 72),  (23,  204, 146), (134, 219, 61),
    (52,  147, 26),  (187, 212, 0),   (168, 153, 44),  (255, 194, 0),
    (147, 69,  52),  (255, 115, 100), (236, 24,  0),   (255, 56,  132),
    (133, 0,   82),  (255, 56,  203), (200, 149, 255), (199, 55,  255),
]


def class_color(class_id):
    """Return a consistent BGR color for a given class index."""
    return _PALETTE[int(class_id) % len(_PALETTE)]


def track_color(track_id):
    """Return a consistent BGR color for a given track ID."""
    np.random.seed(int(track_id) % 2**16)
    return tuple(int(c) for c in np.random.randint(80, 220, 3).tolist())


# ===========================================================================
# DRAWING UTILITIES
# ===========================================================================

def draw_detection_box(frame, x1, y1, x2, y2, label, color, thickness=2):
    """
    Draw a bounding box with a filled label badge on the frame in-place.

    Parameters
    ----------
    frame : np.ndarray
        BGR image.
    x1, y1, x2, y2 : int
        Box coordinates.
    label : str
        Text to display (class name + confidence or class name + track ID).
    color : tuple of int
        BGR color.
    thickness : int
        Box line thickness.
    """
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thick = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)

    # Badge background
    badge_y1 = max(y1 - th - baseline - 4, 0)
    badge_y2 = max(y1, th + baseline + 4)
    cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 6, badge_y2), color, -1)

    # Label text
    cv2.putText(
        frame, label,
        (x1 + 3, badge_y2 - baseline - 2),
        font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA
    )


def draw_fps(frame, fps):
    """Overlay FPS counter in the top-left corner."""
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
    )


def draw_counts(frame, n_detections, n_tracks):
    """Overlay detection and track counts in the top-right corner."""
    h, w = frame.shape[:2]
    text = f"Det: {n_detections}  Track: {n_tracks}"
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(
        frame, text,
        (w - tw - 10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA
    )


# ===========================================================================
# SORT TRACKER (self-contained implementation)
# ===========================================================================

"""
SORT (Simple Online and Realtime Tracking) implementation.
Paper: Bewley et al. (2016), https://arxiv.org/abs/1602.00763

Uses a Kalman filter per track and the Hungarian algorithm for assignment.
No deep features (those would be Deep SORT). Suitable for real-time use.
"""

from scipy.optimize import linear_sum_assignment


class KalmanBoxTracker:
    """
    Represents a single tracked object via a constant-velocity Kalman filter.

    State vector: [x_center, y_center, scale, aspect_ratio, dx, dy, ds]
    Measurement : [x_center, y_center, scale, aspect_ratio]
    """

    _count = 0   # class-level counter for unique IDs

    def __init__(self, bbox):
        """
        Parameters
        ----------
        bbox : array-like, shape (4,)
            Detection bounding box [x1, y1, x2, y2].
        """
        # --- Kalman filter matrices (7-state, 4-measurement) ---
        self.kf = cv2.KalmanFilter(7, 4)

        # Measurement matrix H
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)

        # State transition matrix F (constant velocity)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        # Process noise Q
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 1e-2
        self.kf.processNoiseCov[4:, 4:] *= 10.0   # higher uncertainty on velocity

        # Measurement noise R
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 10.0
        self.kf.measurementNoiseCov[2:, 2:] *= 10.0

        # Initial state covariance P
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)
        self.kf.errorCovPost[4:, 4:] *= 1000.0    # high initial velocity uncertainty

        # Initialize state from the first detection
        self.kf.statePost = self._bbox_to_z(bbox)

        KalmanBoxTracker._count += 1
        self.id          = KalmanBoxTracker._count
        self.hits        = 1
        self.hit_streak  = 1
        self.age         = 0
        self.time_since_update = 0
        self.history     = []

    @staticmethod
    def _bbox_to_z(bbox):
        """Convert [x1, y1, x2, y2] to Kalman state [cx, cy, s, r, 0, 0, 0]."""
        x1, y1, x2, y2 = bbox
        w = float(x2 - x1)
        h = float(y2 - y1)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        s  = w * h          # scale = area
        r  = w / h if h > 0 else 1.0
        return np.array([[cx], [cy], [s], [r], [0.], [0.], [0.]], dtype=np.float32)

    @staticmethod
    def _z_to_bbox(z):
        """Convert Kalman state to [x1, y1, x2, y2]."""
        cx, cy, s, r = float(z[0]), float(z[1]), float(z[2]), float(z[3])
        if s <= 0 or r <= 0:
            return np.zeros(4)
        w = np.sqrt(s * r)
        h = s / w
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predict(self):
        """Run Kalman prediction step and return predicted bbox."""
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        pred = self._z_to_bbox(self.kf.statePost)
        self.history.append(pred)
        return pred

    def update(self, bbox):
        """Correct the Kalman filter with a matched detection."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.correct(self._bbox_to_z(bbox))
        self.history = []

    def get_state(self):
        """Return the current bbox estimate [x1, y1, x2, y2]."""
        return self._z_to_bbox(self.kf.statePost)


def _iou_batch(bboxes_a, bboxes_b):
    """
    Compute pairwise IoU matrix between two sets of boxes.

    Parameters
    ----------
    bboxes_a : np.ndarray, shape (M, 4)  [x1,y1,x2,y2]
    bboxes_b : np.ndarray, shape (N, 4)  [x1,y1,x2,y2]

    Returns
    -------
    np.ndarray, shape (M, N)
        IoU values.
    """
    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])

    inter_x1 = np.maximum(bboxes_a[:, None, 0], bboxes_b[None, :, 0])
    inter_y1 = np.maximum(bboxes_a[:, None, 1], bboxes_b[None, :, 1])
    inter_x2 = np.minimum(bboxes_a[:, None, 2], bboxes_b[None, :, 2])
    inter_y2 = np.minimum(bboxes_a[:, None, 3], bboxes_b[None, :, 3])

    inter_w  = np.maximum(0, inter_x2 - inter_x1)
    inter_h  = np.maximum(0, inter_y2 - inter_y1)
    inter    = inter_w * inter_h

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.where(union > 0, union, 1e-6)


class SORTTracker:
    """
    SORT multi-object tracker.

    Manages a pool of KalmanBoxTracker instances and uses the Hungarian
    algorithm on IoU scores to associate detections with existing tracks.
    """

    def __init__(self, max_age=MAX_TRACK_AGE, min_hits=MIN_HITS, iou_threshold=IOU_TRACK):
        """
        Parameters
        ----------
        max_age : int
            Frames a track can go unmatched before deletion.
        min_hits : int
            Consecutive detections required before a track is confirmed.
        iou_threshold : float
            Minimum IoU for a detection-track association.
        """
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.trackers      = []
        self.frame_count   = 0

    def update(self, detections):
        """
        Update tracks with new detections for the current frame.

        Parameters
        ----------
        detections : np.ndarray, shape (N, 5)
            Each row is [x1, y1, x2, y2, confidence].
            Pass an empty array (shape (0,5)) when no objects are detected.

        Returns
        -------
        np.ndarray, shape (M, 5)
            Active confirmed tracks: [x1, y1, x2, y2, track_id].
        """
        self.frame_count += 1

        # Predict new locations for all existing trackers
        predicted = []
        dead = []
        for t in self.trackers:
            pred = t.predict()
            if np.any(np.isnan(pred)):
                dead.append(t)
            else:
                predicted.append(pred)

        for t in dead:
            self.trackers.remove(t)

        pred_array = np.array(predicted) if predicted else np.empty((0, 4))
        det_array  = detections[:, :4] if len(detections) > 0 else np.empty((0, 4))

        # --- Hungarian assignment ---
        if len(pred_array) > 0 and len(det_array) > 0:
            iou_matrix  = _iou_batch(pred_array, det_array)
            row_inds, col_inds = linear_sum_assignment(-iou_matrix)
            matched_pairs = [
                (r, c) for r, c in zip(row_inds, col_inds)
                if iou_matrix[r, c] >= self.iou_threshold
            ]
        else:
            matched_pairs = []

        matched_rows = {r for r, _ in matched_pairs}
        matched_cols = {c for _, c in matched_pairs}

        # Update matched trackers
        for r, c in matched_pairs:
            self.trackers[r].update(det_array[c])

        # Create new trackers for unmatched detections
        for c in range(len(det_array)):
            if c not in matched_cols:
                self.trackers.append(KalmanBoxTracker(det_array[c]))

        # Remove stale trackers
        self.trackers = [
            t for t in self.trackers
            if t.time_since_update <= self.max_age
        ]

        # Collect confirmed tracks
        results = []
        for t in self.trackers:
            if t.time_since_update < 1 and (t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                bbox = t.get_state()
                results.append([*bbox, t.id])

        return np.array(results) if results else np.empty((0, 5))

    def reset(self):
        """Reset all tracks and the frame counter."""
        self.trackers   = []
        self.frame_count = 0
        KalmanBoxTracker._count = 0


# ===========================================================================
# DETECTION PIPELINE
# ===========================================================================

class DetectionTracker:
    """
    Combines YOLOv8 detection with SORT tracking in a single convenient class.
    """

    def __init__(
        self,
        model_path=DEFAULT_MODEL,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        max_age=MAX_TRACK_AGE,
        min_hits=MIN_HITS,
        iou_track=IOU_TRACK,
        classes=None,
        device="",
    ):
        """
        Parameters
        ----------
        model_path : str
            Path to a YOLOv8 .pt file. If the file is absent, Ultralytics
            will download it automatically from the YOLO model hub.
        conf : float
            Detection confidence threshold.
        iou : float
            NMS IoU threshold for detection.
        max_age : int
            SORT max track age (frames).
        min_hits : int
            SORT minimum confirmed hits.
        iou_track : float
            SORT IoU threshold for assignment.
        classes : list of int or None
            If set, only detect these COCO class IDs.
        device : str
            Inference device: "" (auto), "cpu", "cuda", "mps".
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics is required. Install with: pip install ultralytics"
            )

        print(f"[MODEL] Loading YOLO model: {model_path}")
        self.model   = YOLO(model_path)
        self.conf    = conf
        self.iou     = iou
        self.classes = classes
        self.device  = device
        self.tracker = SORTTracker(max_age=max_age, min_hits=min_hits, iou_threshold=iou_track)
        self.names   = self.model.names   # {class_id: class_name}
        print(f"[MODEL] Model loaded. Classes: {len(self.names)}")

    def detect(self, frame):
        """
        Run YOLO inference on a single BGR frame.

        Returns
        -------
        np.ndarray, shape (N, 6)
            Each row: [x1, y1, x2, y2, confidence, class_id]
        """
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6))

        xyxy    = boxes.xyxy.cpu().numpy()
        confs   = boxes.conf.cpu().numpy().reshape(-1, 1)
        cls_ids = boxes.cls.cpu().numpy().reshape(-1, 1)
        return np.hstack([xyxy, confs, cls_ids])

    def track(self, detections):
        """
        Pass detections [x1,y1,x2,y2,conf] to SORT.

        Returns
        -------
        np.ndarray, shape (M, 5)
            [x1, y1, x2, y2, track_id]
        """
        if len(detections) == 0:
            return self.tracker.update(np.empty((0, 5)))
        return self.tracker.update(detections[:, :5])

    def process_frame(self, frame, show_detections=True, show_tracks=True):
        """
        Detect objects, update tracks, and annotate the frame.

        Parameters
        ----------
        frame : np.ndarray
            Input BGR frame.
        show_detections : bool
            If True, draw raw detection boxes in class color.
        show_tracks : bool
            If True, draw track boxes with track IDs.

        Returns
        -------
        annotated_frame : np.ndarray
            Copy of the frame with all annotations drawn.
        detections : np.ndarray, shape (N, 6)
            Raw detections: [x1,y1,x2,y2,conf,class_id].
        tracks : np.ndarray, shape (M, 5)
            Active tracks: [x1,y1,x2,y2,track_id].
        """
        annotated = frame.copy()
        detections = self.detect(frame)
        tracks     = self.track(detections)

        # Build a quick class-id lookup from detection: key = rounded bbox
        det_cls = {}
        for d in detections:
            key = tuple(np.round(d[:4]).astype(int))
            det_cls[key] = int(d[5])

        if show_detections:
            for d in detections:
                x1, y1, x2, y2 = map(int, d[:4])
                cls_id = int(d[5])
                conf   = d[4]
                color  = class_color(cls_id)
                label  = f"{self.names.get(cls_id, cls_id)} {conf:.2f}"
                draw_detection_box(annotated, x1, y1, x2, y2, label, color, thickness=1)

        if show_tracks:
            for tr in tracks:
                x1, y1, x2, y2 = map(int, tr[:4])
                track_id = int(tr[4])
                # Try to find matching class
                best_cls = None
                best_iou = 0.0
                for d in detections:
                    dx1, dy1, dx2, dy2 = map(int, d[:4])
                    inter = max(0, min(x2, dx2) - max(x1, dx1)) * max(0, min(y2, dy2) - max(y1, dy1))
                    union = (x2-x1)*(y2-y1) + (dx2-dx1)*(dy2-dy1) - inter
                    if union > 0 and inter/union > best_iou:
                        best_iou = inter/union
                        best_cls = int(d[5])

                color = track_color(track_id)
                cls_name = self.names.get(best_cls, "?") if best_cls is not None else "?"
                label    = f"ID:{track_id} {cls_name}"
                draw_detection_box(annotated, x1, y1, x2, y2, label, color, thickness=2)

        return annotated, detections, tracks

    def reset_tracker(self):
        """Reset SORT state (call between video files)."""
        self.tracker.reset()


# ===========================================================================
# VIDEO PROCESSING
# ===========================================================================

class VideoProcessor:
    """
    Wraps DetectionTracker to handle video streams (webcam or file).
    Supports display, saving, and per-frame statistics.
    """

    def __init__(self, detector_tracker: DetectionTracker):
        self.dt = detector_tracker

    def run(
        self,
        source=0,
        output_path=None,
        display=True,
        fps_cap=DEFAULT_FPS_CAP,
        show_detections=True,
        show_tracks=True,
        frame_limit=None,
    ):
        """
        Run the detection+tracking pipeline on a video source.

        Parameters
        ----------
        source : int or str
            0 for webcam, or path to a video file.
        output_path : str or None
            If set, write annotated video to this path.
        display : bool
            If True, show the live annotated window.
        fps_cap : int
            Maximum display FPS (real-time cap).
        show_detections : bool
            Draw raw detection boxes.
        show_tracks : bool
            Draw tracking boxes with IDs.
        frame_limit : int or None
            Stop after this many frames (useful for testing).
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source: {source}")
            return

        src_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # -1 for webcam

        print(f"[VIDEO] Source   : {source}")
        print(f"[VIDEO] Resolution: {w}x{h} @ {src_fps:.1f} FPS")
        if total > 0:
            print(f"[VIDEO] Total frames: {total}")

        # Video writer setup
        writer = None
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, src_fps, (w, h))
            print(f"[VIDEO] Writing output to: {output_path}")

        self.dt.reset_tracker()

        frame_idx   = 0
        fps_display = 0.0
        t_prev      = time.perf_counter()
        delay_ms    = max(1, int(1000 / fps_cap))

        print("[VIDEO] Processing... Press 'q' to quit, 'p' to pause.")
        paused = False

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[VIDEO] End of stream.")
                    break

                # --- Inference ---
                annotated, dets, tracks = self.dt.process_frame(
                    frame,
                    show_detections=show_detections,
                    show_tracks=show_tracks,
                )

                # --- FPS calculation ---
                t_now       = time.perf_counter()
                fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
                t_prev      = t_now

                # --- Overlays ---
                draw_fps(annotated, fps_display)
                draw_counts(annotated, len(dets), len(tracks))

                # --- Progress for file input ---
                if total > 0:
                    pct = 100.0 * frame_idx / total
                    bar_w   = int(w * pct / 100)
                    cv2.rectangle(annotated, (0, h - 4), (bar_w, h), (0, 200, 0), -1)

                if writer:
                    writer.write(annotated)

                frame_idx += 1
                if frame_limit and frame_idx >= frame_limit:
                    print(f"[VIDEO] Frame limit ({frame_limit}) reached.")
                    break

            # --- Display ---
            if display:
                cv2.imshow("Object Detection & Tracking", annotated if not paused else annotated)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key == ord("q"):
                    print("[VIDEO] Quit by user.")
                    break
                elif key == ord("p"):
                    paused = not paused
                    print("[VIDEO] Paused." if paused else "[VIDEO] Resumed.")

        cap.release()
        if writer:
            writer.release()
            print(f"[VIDEO] Output saved to: {os.path.abspath(output_path)}")
        if display:
            cv2.destroyAllWindows()

        print(f"[VIDEO] Processed {frame_idx} frames.")

    def process_image(self, image_path, output_path=None, display=True):
        """
        Run detection and tracking on a single image file.

        Parameters
        ----------
        image_path : str
            Path to the input image.
        output_path : str or None
            If set, save the annotated image here.
        display : bool
            If True, display the result in a window.
        """
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ERROR] Cannot read image: {image_path}")
            return

        annotated, dets, tracks = self.dt.process_frame(frame)
        print(f"[IMAGE] Detections: {len(dets)} | Tracks: {len(tracks)}")

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            cv2.imwrite(output_path, annotated)
            print(f"[IMAGE] Annotated image saved to: {os.path.abspath(output_path)}")

        if display:
            cv2.imshow("Detection Result", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time object detection and tracking with YOLOv8 + SORT."
    )

    # Input
    parser.add_argument(
        "--source", type=str, default="0",
        help=(
            "Video source. '0' = default webcam, '1','2',... = other cameras, "
            "or path to a video/image file."
        )
    )

    # Model
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"YOLOv8 model: yolov8n/s/m/l/x.pt (default: {DEFAULT_MODEL})."
    )
    parser.add_argument(
        "--conf", type=float, default=CONF_THRESHOLD,
        help=f"Detection confidence threshold (default: {CONF_THRESHOLD})."
    )
    parser.add_argument(
        "--iou", type=float, default=IOU_THRESHOLD,
        help=f"NMS IoU threshold (default: {IOU_THRESHOLD})."
    )
    parser.add_argument(
        "--classes", type=int, nargs="+", default=None,
        help="COCO class IDs to detect (e.g. 0 for person, 2 for car). Default: all."
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="Inference device: '' (auto), 'cpu', 'cuda', 'mps'."
    )

    # Tracker
    parser.add_argument(
        "--max_age", type=int, default=MAX_TRACK_AGE,
        help=f"SORT: max frames to keep a lost track (default: {MAX_TRACK_AGE})."
    )
    parser.add_argument(
        "--min_hits", type=int, default=MIN_HITS,
        help=f"SORT: min hits to confirm a track (default: {MIN_HITS})."
    )
    parser.add_argument(
        "--iou_track", type=float, default=IOU_TRACK,
        help=f"SORT: IoU threshold for assignment (default: {IOU_TRACK})."
    )

    # Output / display
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save the annotated output video (e.g. output/result.mp4)."
    )
    parser.add_argument(
        "--no_display", action="store_true",
        help="Disable live display window (useful for headless servers)."
    )
    parser.add_argument(
        "--no_detections", action="store_true",
        help="Hide raw detection boxes; show tracking boxes only."
    )
    parser.add_argument(
        "--no_tracks", action="store_true",
        help="Hide tracking boxes; show raw detections only."
    )
    parser.add_argument(
        "--fps_cap", type=int, default=DEFAULT_FPS_CAP,
        help=f"Maximum display FPS (default: {DEFAULT_FPS_CAP})."
    )
    parser.add_argument(
        "--frame_limit", type=int, default=None,
        help="Stop after this many frames (useful for testing)."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse source - webcam index or file path
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    dt = DetectionTracker(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_track=args.iou_track,
        classes=args.classes,
        device=args.device,
    )

    vp = VideoProcessor(dt)

    # Image mode detection (by extension)
    if isinstance(source, str) and source.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    ):
        out = args.output or "output/result_image.jpg"
        vp.process_image(source, output_path=out, display=not args.no_display)
    else:
        vp.run(
            source=source,
            output_path=args.output,
            display=not args.no_display,
            fps_cap=args.fps_cap,
            show_detections=not args.no_detections,
            show_tracks=not args.no_tracks,
            frame_limit=args.frame_limit,
        )


if __name__ == "__main__":
    main()
