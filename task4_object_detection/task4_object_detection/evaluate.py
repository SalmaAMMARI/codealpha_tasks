"""
evaluate.py - Batch evaluation and benchmarking for the detection+tracking pipeline.

Computes per-frame detection metrics (precision, recall at a fixed IoU threshold)
and tracking statistics (track count, average lifespan) over a video file or
image folder. Optionally writes a CSV report.

Usage:
    python evaluate.py --source video.mp4 --ground_truth gt.json
    python evaluate.py --source video.mp4 --report output/metrics.csv
"""

import os
import csv
import json
import time
import argparse
import numpy as np
import cv2

from object_detection_tracking import DetectionTracker, VideoProcessor, DEFAULT_MODEL


# ===========================================================================
# METRIC HELPERS
# ===========================================================================

def bbox_iou(box_a, box_b):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_detections_to_gt(detections, gt_boxes, iou_threshold=0.5):
    """
    Match detected boxes to ground-truth boxes at a given IoU threshold.

    Parameters
    ----------
    detections : list of [x1,y1,x2,y2]
    gt_boxes   : list of [x1,y1,x2,y2]
    iou_threshold : float

    Returns
    -------
    tp : int  - true positives
    fp : int  - false positives
    fn : int  - false negatives
    """
    matched_gt = set()
    tp = 0
    for det in detections:
        best_iou   = 0.0
        best_gt_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = bbox_iou(det, gt)
            if iou > best_iou:
                best_iou    = iou
                best_gt_idx = i
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(detections) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


# ===========================================================================
# BENCHMARK (no ground truth)
# ===========================================================================

def benchmark_video(source, model_path=DEFAULT_MODEL, frame_limit=None):
    """
    Run inference on a video and report throughput, detection counts,
    and tracking statistics. No ground truth required.

    Parameters
    ----------
    source : str or int
        Video file path or webcam index.
    model_path : str
        Path to YOLO model weights.
    frame_limit : int or None
        Stop after this many frames.

    Returns
    -------
    dict
        Summary statistics.
    """
    dt  = DetectionTracker(model_path=model_path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return {}

    frame_times   = []
    det_counts    = []
    track_counts  = []
    track_lifespans = {}   # track_id -> first/last seen frame

    frame_idx = 0
    print(f"[BENCH] Benchmarking on '{source}' ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        _, dets, tracks = dt.process_frame(frame, show_detections=False, show_tracks=False)
        elapsed = time.perf_counter() - t0

        frame_times.append(elapsed)
        det_counts.append(len(dets))
        track_counts.append(len(tracks))

        for tr in tracks:
            tid = int(tr[4])
            if tid not in track_lifespans:
                track_lifespans[tid] = [frame_idx, frame_idx]
            else:
                track_lifespans[tid][1] = frame_idx

        frame_idx += 1
        if frame_limit and frame_idx >= frame_limit:
            break

    cap.release()

    lifespans = [v[1] - v[0] + 1 for v in track_lifespans.values()]

    stats = {
        "total_frames"      : frame_idx,
        "avg_fps"           : 1.0 / np.mean(frame_times) if frame_times else 0,
        "min_fps"           : 1.0 / np.max(frame_times) if frame_times else 0,
        "max_fps"           : 1.0 / np.min(frame_times) if frame_times else 0,
        "avg_detections"    : float(np.mean(det_counts)) if det_counts else 0,
        "avg_tracks"        : float(np.mean(track_counts)) if track_counts else 0,
        "total_tracks"      : len(track_lifespans),
        "avg_track_lifespan": float(np.mean(lifespans)) if lifespans else 0,
    }

    print("\n--- Benchmark Results ---")
    for k, v in stats.items():
        print(f"  {k:<25}: {v:.2f}" if isinstance(v, float) else f"  {k:<25}: {v}")
    print()
    return stats


# ===========================================================================
# EVALUATION WITH GROUND TRUTH
# ===========================================================================

def evaluate_with_gt(source, gt_path, model_path=DEFAULT_MODEL, iou_threshold=0.5, report_path=None):
    """
    Evaluate detections against COCO-format ground truth annotations.

    Parameters
    ----------
    source : str
        Path to the video file.
    gt_path : str
        Path to a JSON file in COCO format (only 'annotations' and 'images' used).
    model_path : str
        YOLO weights path.
    iou_threshold : float
        IoU threshold for TP/FP assignment.
    report_path : str or None
        If set, write per-frame CSV here.

    Returns
    -------
    dict
        Aggregated precision, recall, and F1.
    """
    with open(gt_path, "r") as f:
        gt_data = json.load(f)

    # Index GT boxes by image_id (frame index for video)
    gt_by_frame = {}
    for ann in gt_data.get("annotations", []):
        fid = ann["image_id"]
        x, y, w, h = ann["bbox"]
        box = [x, y, x + w, y + h]
        gt_by_frame.setdefault(fid, []).append(box)

    dt  = DetectionTracker(model_path=model_path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return {}

    total_tp = total_fp = total_fn = 0
    rows = [["frame", "tp", "fp", "fn", "precision", "recall"]]

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, dets, _ = dt.process_frame(frame, show_detections=False, show_tracks=False)
        det_boxes  = [list(map(int, d[:4])) for d in dets]
        gt_boxes   = gt_by_frame.get(frame_idx, [])

        tp, fp, fn = match_detections_to_gt(det_boxes, gt_boxes, iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rows.append([frame_idx, tp, fp, fn, f"{prec:.3f}", f"{rec:.3f}"])
        frame_idx += 1

    cap.release()

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    if report_path:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"[EVAL] Per-frame report saved to: {report_path}")

    results = {
        "precision": precision,
        "recall"   : recall,
        "f1"       : f1,
        "iou_thr"  : iou_threshold,
        "frames"   : frame_idx,
    }
    print("\n--- Evaluation Results ---")
    for k, v in results.items():
        print(f"  {k:<12}: {v:.4f}" if isinstance(v, float) else f"  {k:<12}: {v}")
    print()
    return results


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark or evaluate the detection+tracking pipeline.")
    parser.add_argument("--source",       type=str, required=True, help="Video file path or webcam index.")
    parser.add_argument("--model",        type=str, default=DEFAULT_MODEL)
    parser.add_argument("--ground_truth", type=str, default=None, help="Path to COCO-format GT JSON.")
    parser.add_argument("--iou_threshold",type=float, default=0.5)
    parser.add_argument("--report",       type=str, default=None, help="CSV report output path.")
    parser.add_argument("--frame_limit",  type=int, default=None)
    args = parser.parse_args()

    try:
        src = int(args.source)
    except ValueError:
        src = args.source

    if args.ground_truth:
        evaluate_with_gt(
            source=src,
            gt_path=args.ground_truth,
            model_path=args.model,
            iou_threshold=args.iou_threshold,
            report_path=args.report,
        )
    else:
        benchmark_video(
            source=src,
            model_path=args.model,
            frame_limit=args.frame_limit,
        )
