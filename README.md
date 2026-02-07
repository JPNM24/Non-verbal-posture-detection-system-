# 🎯 Non-Verbal Posture Detection System

A lightweight, real-time non-verbal behavior analysis system for interview assessment. Uses computer vision to analyze posture, eye contact, facial engagement, and stability — **no machine learning required**.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange.svg)

---

## ✨ Key Features

### 🔬 Deterministic Analysis
- **No ML/DL dependencies** — Pure mathematical computations using trigonometry and running statistics
- **O(1) memory footprint** — Constant per-frame cost, no frame buffers
- **100% reproducible** — Same input produces identical output

### 📊 Real-Time Metrics

| Metric | Description |
|--------|-------------|
| **Eye Contact** | Gaze direction relative to camera center |
| **Facial Expression** | Micro-movement engagement scoring |
| **Posture** | Shoulder alignment + torso forward lean |
| **Stability** | Head movement variance over time |

### ⚡ Fast Score Response
- **Rolling window averaging** (last 30 frames / ~1 second)
- Scores adapt quickly to posture changes
- No sluggish full-history averaging

### 🛡️ Interview Integrity
- **Multi-face detection** — Session auto-cancels if multiple faces appear
- **Blink exclusion** — Eye contact not penalized during natural blinks
- **Single-person enforcement** — 15-frame threshold before cancellation

---

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Demo
```bash
python run_demo.py
```

Press `q` to quit the demo.

---

## 📁 Project Structure

```
non-verbal-module/
├── src/non_verbal_analysis/
│   ├── analyzer.py              # Main orchestrator (MediaPipe-based)
│   ├── deterministic_analyzer.py # Pure math analyzer (spec-compliant)
│   ├── pipeline.py              # Processing stages
│   ├── eye_contact_analyzer.py  # Gaze detection
│   ├── session_manager.py       # Session state management
│   ├── models.py                # Output schemas (Pydantic)
│   ├── utils.py                 # Normalization utilities
│   └── validators.py            # Input validation
├── run_demo.py                  # Live webcam demo
├── ANALYZER.md                  # Technical documentation
└── requirements.txt             # Dependencies
```

---

## 🔧 Configuration

### Constants (in `deterministic_analyzer.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAMERA_FOV_DEG` | 60° | Camera field of view |
| `BASELINE_SMOOTH_K` | 300 | Baseline adaptation speed |
| `STATE_SMOOTH_ALPHA` | 0.9 | Latent state temporal smoothing |
| `SD_YAW_DEG` | 3.3° | Standard deviation for yaw normalization |
| `SD_PITCH_DEG` | 3.9° | Standard deviation for pitch normalization |

---

## 📈 Output Format

```python
{
    "session_status": "active",  # or "cancelled", "insufficient_data"
    "non_verbal_scores": {
        "eye_contact": 85.5,
        "facial_expression": 72.3,
        "posture": 88.1,
        "stability": 91.2,
        "final_non_verbal_score": 84.7
    },
    "insights": ["Posture needs improvement"]  # Empty if all good
}
```

---

## 🎯 Score Weights

| Component | Weight |
|-----------|--------|
| Eye Contact | 35% |
| Facial Expression | 25% |
| Posture | 25% |
| Stability | 15% |

---

## 🧮 Algorithm Highlights

### Head Orientation (Geometry-based)
```
yaw   = atan(nose_dx / focal_length) × (180/π)
pitch = atan(nose_dy / focal_length) × (180/π)
roll  = atan(shoulder_dy / shoulder_dx) × (180/π)
```

### Z-Score Normalization
```
Z_signal = (current - baseline) / standard_deviation
```

### Evidence Scoring
```
slouch     = 0.4×Z_shoulder + 0.4×Z_torso + 0.2×Z_pitch
nervous    = 0.5×|Z_roll| + 0.3×|Z_yaw| + 0.2×variance
engagement = -|Z_torso| + forward_lean_bonus
```

---

## 📋 Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- MediaPipe
- NumPy
- Pydantic

---

## 🔗 Related

This module is designed to work alongside a **verbal speech analysis module** for complete interview assessment.

---

## 📄 License

**Proprietary** — All Rights Reserved. See [LICENSE](LICENSE) for details.

---

## 👤 Author

Built for interview preparation and assessment systems.
