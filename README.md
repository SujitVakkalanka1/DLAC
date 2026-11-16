# DLAC — Deep Learning Academic Coursework

This repository contains academic lab coursework, practical assignments, and deep learning implementations.

---

## Coursework Index

### 📁 [VisionPort Practical 1](./VisionPort%20Practical%201)
**Intelligent Vehicle Detection, Tracking, and Traffic Monitoring System** (covering all 20 syllabus tasks):
- **Tasks 1–5**: Environment setup, physical/bundled image capture, image display, video playback, and 1-second video frame extraction.
- **Tasks 6–9**: YOLO11 high-precision vehicle detection, classification (Cars, Buses, Trucks, Motorcycles), confidence reporting, and frame counts.
- **Tasks 10–11**: Traffic density heuristics and sustained congestion detection.
- **Task 12**: ByteTrack multi-object tracking across video frames with unique vehicle IDs.
- **Tasks 13–19**: Speed approximation, traffic light recognition, waiting time estimation, directional entry/exit counting, wrong turn, wrong-way driving, and restricted parking monitoring.
- **Task 20**: Candidate motorcycle triple-riding detection.

#### Features Included in Practical 1:
- 🖥️ **Desktop GUI Dashboard (`gui_app.py`)**: Full interactive Tkinter interface with live visual video streaming, model selectors, confidence sliders, and real-time statistics.
- ⚡ **Automated Test Runner (`run_all.py`)**: Sequential execution of all 15 task modules with a summary report.
- 🏋️ **Model Training Pipeline (`train.py`)**: Script to fine-tune or train YOLO on custom traffic datasets.
- 🎯 **Pretrained YOLO11 Weights**: Bundled `yolo11m.pt` (medium/accurate), `yolo11s.pt` (small), and `yolo11n.pt` (nano).
