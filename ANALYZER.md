# Deterministic Non-Verbal Behavior Analysis Module

## Overview

A lightweight, deterministic module for real-time non-verbal behavior analysis from facial and pose landmarks. Uses **only basic arithmetic and trigonometry** — no machine learning or deep learning.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FrameInput (per frame)                   │
│  • Eye centers (L/R)  • Nose tip  • Shoulders  • Hips       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              DeterministicNonVerbalAnalyzer                 │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Baseline    │  │  Variance    │  │   Latent     │      │
│  │   State      │  │   State      │  │   State      │      │
│  │  (means)     │  │  (Welford)   │  │  (0-1)       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     AnalysisOutput                          │
│  • posture_deviation_scores (Z-scores)                      │
│  • latent_state_vector (engagement, confidence, etc.)       │
│  • stability_indices (variance metrics)                     │
│  • evidence_scores (slouch, nervousness, engagement)        │
└─────────────────────────────────────────────────────────────┘
```

---

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `CAMERA_FOV_DEG` | 60° | Camera field of view |
| `GLOBAL_IPD_MM` | 63.0 mm | Average inter-pupillary distance |
| `SD_YAW_DEG` | 3.3° | Std dev for yaw normalization |
| `SD_PITCH_DEG` | 3.9° | Std dev for pitch normalization |
| `SD_ROLL_DEG` | 2.8° | Std dev for roll normalization |
| `SD_SHOULDER_SLOPE_DEG` | 4.0° | Std dev for shoulder slope |
| `SD_TORSO_ANGLE_DEG` | 5.0° | Std dev for torso angle |
| `BASELINE_SMOOTH_K` | 300 | Smoothing constant for baseline |
| `STATE_SMOOTH_ALPHA` | 0.9 | Temporal smoothing for latent states |

---

## Processing Pipeline

### 1. Precomputation (once)
```
focal_length_px = (frame_width / 2) / tan(FOV / 2)
```

### 2. Per-Frame Processing

| Step | Operation | Formula |
|------|-----------|---------|
| **Scale** | IPD normalization | `scale = 63mm / IPD_pixels` |
| **Yaw** | Left-right rotation | `atan(nose_dx / focal_length)` |
| **Pitch** | Up-down rotation | `atan(nose_dy / focal_length)` |
| **Roll** | Head tilt | `atan(shoulder_dy / shoulder_dx)` |
| **Shoulder Slope** | Posture | `abs(roll)` |
| **Torso Angle** | Forward lean | `atan(torso_dx / torso_dy)` |

### 3. Baseline Update
```
baseline += (current - baseline) / 300
```

### 4. Z-Score Normalization
```
Z_signal = (signal - baseline) / SD_signal
```

### 5. Evidence Scores
```python
slouch     = 0.4×Z_shoulder + 0.4×Z_torso + 0.2×Z_pitch
nervous    = 0.5×|Z_roll| + 0.3×|Z_yaw| + 0.2×var_shoulder
engagement = -|Z_torso| + 0.1
```

### 6. Latent State Update
```
state_new = 0.9×state_old + 0.1×sigmoid(evidence)
```

---

## Usage

```python
from src.non_verbal_analysis.deterministic_analyzer import (
    DeterministicNonVerbalAnalyzer,
    FrameInput
)

# Initialize
analyzer = DeterministicNonVerbalAnalyzer(
    frame_width_px=640,
    frame_height_px=480,
    camera_fov_deg=60.0
)

# Per frame
output = analyzer.process_frame(FrameInput(
    left_eye_center_px=(280, 200),
    right_eye_center_px=(360, 200),
    nose_tip_px=(320, 260),
    left_shoulder_px=(200, 400),
    right_shoulder_px=(440, 400),
    left_hip_px=(220, 600),
    right_hip_px=(420, 600)
))

# Access results
print(output.posture_deviation_scores)  # Z-scores
print(output.latent_state_vector)       # [0-1] states
print(output.stability_indices)         # Variance metrics
```

---

## Output Structure

```python
AnalysisOutput(
    posture_deviation_scores={
        "z_yaw": float,      # Left-right deviation
        "z_pitch": float,    # Up-down deviation
        "z_roll": float,     # Tilt deviation
        "z_shoulder": float, # Shoulder slope deviation
        "z_torso": float     # Forward lean deviation
    },
    latent_state_vector={
        "engagement": float,    # [0-1]
        "confidence": float,    # [0-1]
        "nervousness": float,   # [0-1]
        "attentiveness": float  # [0-1]
    },
    stability_indices={
        "variance_yaw": float,
        "variance_pitch": float,
        "variance_roll": float,
        "variance_shoulder": float
    },
    evidence_scores={
        "slouch_score": float,
        "nervous_score": float,
        "engagement_score": float
    }
)
```

---

## Design Constraints

| Constraint | Implementation |
|------------|----------------|
| **O(1) Memory** | No frame buffers, only scalar state |
| **Deterministic** | Same input → same output |
| **No ML/DL** | Pure trigonometry + running statistics |
| **Numerical Stability** | Division-by-zero guards throughout |
| **Camera Geometry** | FOV-based focal length computation |

---

## File Structure

```
src/non_verbal_analysis/
├── deterministic_analyzer.py  ← Main implementation
├── analyzer.py                ← Legacy MediaPipe-based analyzer
├── pipeline.py                ← Pipeline stages
├── models.py                  ← Pydantic output schemas
└── ...
```
