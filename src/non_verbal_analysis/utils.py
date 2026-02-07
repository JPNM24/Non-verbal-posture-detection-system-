import numpy as np
from typing import List, Any
from .validators import validate_normalization_inputs

def calculate_ear(eye_landmarks: List[Any]) -> float:
    if len(eye_landmarks) != 6:
        raise ValueError(f"Expected 6 eye landmarks, got {len(eye_landmarks)}")

    def dist(p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    v1 = dist(eye_landmarks[1], eye_landmarks[5])
    v2 = dist(eye_landmarks[2], eye_landmarks[4])

    h = dist(eye_landmarks[0], eye_landmarks[3])

    if h == 0:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return ear

def normalize_movement(raw_movement: float, face_width: float, metric_name: str = "movement") -> float:
    validate_normalization_inputs(raw_movement, face_width, metric_name)
    return raw_movement / face_width

def normalize_posture_alignment(alignment_error: float, face_width: float) -> float:
    return normalize_movement(alignment_error, face_width, "posture_alignment")

def get_face_width(landmarks: Any) -> float:

    left_side = landmarks[234]
    right_side = landmarks[454]

    width = np.sqrt(
        (left_side.x - right_side.x)**2 +
        (left_side.y - right_side.y)**2 +
        (left_side.z - right_side.z)**2
    )

    return width

def calculate_landmark_variance(current_landmarks: Any, previous_landmarks: Any, face_width: float) -> float:

    key_indices = [1, 33, 263, 61, 291]

    diffs = []
    for i in key_indices:
        curr = current_landmarks[i]
        prev = previous_landmarks[i]

        d = np.sqrt(
            (curr.x - prev.x)**2 +
            (curr.y - prev.y)**2 +
            (curr.z - prev.z)**2
        )
        diffs.append(d)

    avg_movement = np.mean(diffs)

    return normalize_movement(avg_movement, face_width, "facial_engagement")

def calculate_nose_movement(current_nose: Any, previous_nose_pos: tuple, face_width: float) -> float:

    if len(previous_nose_pos) == 2:

        raw_movement = np.sqrt(
            (current_nose.x - previous_nose_pos[0])**2 +
            (current_nose.y - previous_nose_pos[1])**2
        )
    else:

        raw_movement = np.sqrt(
            (current_nose.x - previous_nose_pos[0])**2 +
            (current_nose.y - previous_nose_pos[1])**2 +
            (current_nose.z - previous_nose_pos[2])**2
        )

    return normalize_movement(raw_movement, face_width, "stability")
