from typing import Any, Optional

def is_blink_frame(ear: float, threshold: float = 0.21) -> bool:
    return ear < threshold

def is_multi_face_violation(face_count: int, consecutive_count: int, threshold: int) -> bool:
    return face_count > 1 and consecutive_count >= threshold

def is_valid_landmark(landmark: Any) -> bool:
    if landmark is None:
        return False
    return hasattr(landmark, 'x') and hasattr(landmark, 'y')

def is_valid_face_width(face_width: float) -> bool:

    return 0.0 < face_width < 1.0

def validate_score_range(score: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    return max(min_val, min(max_val, score))

def validate_normalization_inputs(raw_value: float, face_width: float, metric_name: str) -> None:
    if not is_valid_face_width(face_width):
        raise ValueError(
            f"Invalid face_width for {metric_name}: {face_width}. "
            f"Face width must be positive and < 1.0"
        )
