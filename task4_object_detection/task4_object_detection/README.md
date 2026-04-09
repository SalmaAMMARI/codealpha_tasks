# Task 4: Object Detection and Tracking

## Overview

A real-time pipeline combining:
- **YOLOv8** (Ultralytics) for frame-level object detection
- **SORT** (Simple Online and Realtime Tracking) for multi-object tracking

Both the detector and tracker are production-quality, fully self-contained, and
optimized for real-time throughput on CPU and GPU alike.

---

## Project Structure

```
task4_object_detection/
    object_detection_tracking.py  - Main pipeline (detector, tracker, video processor, CLI)
    evaluate.py                   - Benchmark and optional COCO-format GT evaluation
    requirements.txt              - Python dependencies
    output/                       - Auto-created when saving annotated video
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Webcam (default)
```bash
python object_detection_tracking.py
```
Press `q` to quit, `p` to pause.

### 3. Video file
```bash
python object_detection_tracking.py --source /path/to/video.mp4
```

### 4. Save annotated output
```bash
python object_detection_tracking.py \
    --source /path/to/video.mp4 \
    --output output/result.mp4
```

### 5. Detect only people and cars
```bash
python object_detection_tracking.py --classes 0 2
```
(COCO class 0 = person, 2 = car)

### 6. Headless mode (no display, save only)
```bash
python object_detection_tracking.py \
    --source video.mp4 \
    --output output/result.mp4 \
    --no_display
```

### 7. Single image
```bash
python object_detection_tracking.py \
    --source image.jpg \
    --output output/result.jpg
```

---

## CLI Arguments

| Argument          | Default    | Description                                              |
|-------------------|------------|----------------------------------------------------------|
| `--source`        | `0`        | `0`=webcam, `1`=second cam, or file path                 |
| `--model`         | `yolov8n.pt` | YOLO model: `yolov8n/s/m/l/x.pt`                       |
| `--conf`          | `0.40`     | Detection confidence threshold                           |
| `--iou`           | `0.45`     | NMS IoU threshold                                        |
| `--classes`       | All        | COCO class IDs to detect (e.g. `0 2 7`)                  |
| `--device`        | Auto       | `cpu`, `cuda`, `mps`, or `""` for auto                   |
| `--max_age`       | `30`       | SORT: max frames to keep a lost track                    |
| `--min_hits`      | `3`        | SORT: confirmed hits before showing track                |
| `--iou_track`     | `0.30`     | SORT: IoU threshold for track-detection assignment       |
| `--output`        | None       | Output video path                                        |
| `--no_display`    | False      | Disable live window                                      |
| `--no_detections` | False      | Hide raw detection boxes                                 |
| `--no_tracks`     | False      | Hide tracking boxes                                      |
| `--fps_cap`       | `30`       | Maximum display FPS                                      |
| `--frame_limit`   | None       | Stop after N frames                                      |

---

## YOLO Model Sizes

| Model       | Size  | Speed (CPU) | Accuracy |
|-------------|-------|-------------|----------|
| yolov8n.pt  | 6 MB  | Fastest     | Good     |
| yolov8s.pt  | 22 MB | Fast        | Better   |
| yolov8m.pt  | 50 MB | Moderate    | High     |
| yolov8l.pt  | 84 MB | Slow        | Higher   |
| yolov8x.pt  | 131 MB| Slowest     | Best     |

Models are downloaded automatically on first use.

---

## SORT Tracker Architecture

```
Frame N detections
      |
  IoU Matrix (detections x active tracks)
      |
Hungarian Algorithm (optimal 1-to-1 assignment)
      |
  Matched pairs -> Kalman filter correction (update step)
  Unmatched dets -> new KalmanBoxTracker (born track)
  Unmatched tracks -> Kalman prediction only (age track)
  Stale tracks (age > max_age) -> delete
      |
  Confirmed tracks (hit_streak >= min_hits)
      |
Output: [x1, y1, x2, y2, track_id]
```

Each tracker maintains a 7-state Kalman filter:
- State: [cx, cy, scale, aspect_ratio, dx, dy, ds]
- Measurement: [cx, cy, scale, aspect_ratio]

---

## Evaluation & Benchmarking

### Throughput benchmark (no GT needed)
```bash
python evaluate.py --source video.mp4
```

### Evaluate against COCO-format ground truth
```bash
python evaluate.py \
    --source video.mp4 \
    --ground_truth annotations.json \
    --iou_threshold 0.5 \
    --report output/metrics.csv
```

---

## Display Overlay Legend

| Visual element     | Meaning                                            |
|--------------------|----------------------------------------------------|
| Thin colored box   | Raw YOLO detection (color = class)                 |
| Thick colored box  | Confirmed SORT track (color = unique track ID)     |
| Badge text (thin)  | Class name + confidence                            |
| Badge text (thick) | `ID:N ClassName` (tracking)                        |
| Top-left FPS       | Real-time inference + rendering FPS                |
| Top-right counts   | Current frame detection and active track count     |
| Bottom progress bar| Video playback progress (file input only)          |

---

## Common COCO Class IDs

| ID | Name        | ID | Name        |
|----|-------------|----|-------------|
| 0  | person      | 14 | bird        |
| 1  | bicycle     | 15 | cat         |
| 2  | car         | 16 | dog         |
| 3  | motorcycle  | 17 | horse       |
| 4  | airplane    | 24 | backpack    |
| 5  | bus         | 26 | handbag     |
| 6  | train       | 39 | bottle      |
| 7  | truck       | 41 | cup         |
| 9  | boat        | 67 | cell phone  |
| 11 | stop sign   | 73 | laptop      |

---

## GPU Acceleration

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python object_detection_tracking.py --source 0 --device cuda
```

On Apple Silicon:
```bash
python object_detection_tracking.py --source 0 --device mps
```
