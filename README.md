# Social Distance Detector

A real-time social distancing monitor that uses a **pretrained YOLOv3** model (trained on the COCO dataset) to detect people in video footage and flags those who are too close to each other. No custom model training is required.

## How it works

1. On the first frame you manually mark a **region of interest** (4 points for the ground plane) and a **reference distance** (4 more points representing a known real-world measurement).
2. A **perspective transform** maps the camera view to a bird's-eye view so that pixel distances correspond to real-world distances.
3. YOLOv3 detects every person in each frame (class 0 of COCO).
4. Each detected person is assigned a **risk level**:
   - 🔴 **High risk** — closer than 150 cm
   - 🟡 **Low risk** — 150–180 cm
   - 🟢 **Safe** — beyond 180 cm
5. Two output videos are written: an annotated main view and a bird's-eye view.

## Project structure

```
├── src/
│   ├── main.py                    # Entry point — perspective-transform pipeline
│   ├── utils.py                   # Distance calculation and perspective helpers
│   ├── plot.py                    # Drawing: bird's-eye view and annotated frame
│   └── social_distance_detector.py # Alternate simpler pipeline (Euclidean distance)
├── models/
│   └── yolov3.cfg                 # YOLOv3 architecture config
├── requirements.txt
└── .gitignore
```

> **Note:** `yolov3.weights` is not included in this repository because of its size (~236 MB). Download it separately (see below).

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/social-distance-detector.git
cd social-distance-detector
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Download YOLOv3 weights

```bash
wget https://pjreddie.com/media/files/yolov3.weights
# or on Windows (PowerShell):
# Invoke-WebRequest -Uri https://pjreddie.com/media/files/yolov3.weights -OutFile yolov3.weights
```

Place `yolov3.weights` in the `models/` directory (or pass its path with `-m`).

### 4. Create output directories

```bash
mkdir output output_vid output/bird_eye_view
```

## Usage

```bash
python src/main.py -v <path_to_video>
```

### All arguments

| Flag | Default | Description |
|------|---------|-------------|
| `-v` / `--video_path` | `./data/example.mp4` | Input video |
| `-o` / `--output_dir` | `./output/` | Directory for per-frame images |
| `-O` / `--output_vid` | `./output_vid/` | Directory for output videos |
| `-m` / `--model` | `./models/` | Directory containing `yolov3.weights` and `yolov3.cfg` |

### Calibration (interactive)

When the video opens you will be prompted to click **8 points**:

- **Points 1–4**: corners of a rectangular region on the ground plane (your ROI).
- **Points 5–8**: two pairs of points that define a known real-world distance (e.g. lane markings 180 cm apart — click each pair).

After 8 clicks the window closes and processing starts automatically.

## Alternate pipeline

`social_distance_detector.py` is a simpler alternative that uses straight Euclidean pixel distance (no perspective transform). It relies on the `pyimagesearch` helper package:

```bash
pip install pyimagesearch
python social_distance_detector.py --input pedestrians.mp4 --output output.avi
```

A desktop notification is sent when violation count exceeds the configured threshold.

## Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Video I/O, DNN inference, drawing |
| `numpy` | Array math |
| `imutils` | Frame resizing helpers |
| `scipy` | Pairwise distance matrix |
| `plyer` | Desktop notifications |

## Model

This project uses **YOLOv3** by Joseph Redmon et al., pretrained on the **COCO dataset** (80 classes). Only the `person` class (class ID 0) is used.

- Paper: [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
- Weights: [pjreddie.com/darknet/yolo](https://pjreddie.com/darknet/yolo/)

## License

MIT
