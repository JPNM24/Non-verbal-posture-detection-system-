"""
Validation rules for non-verbal analysis.
Centralizes all hard rules and thresholds for enforceable validation.
"""
from typing import Any, Optional


def is_blink_frame(ear: float, threshold: float = 0.21) -> bool:
    """
    Determine if a frame represents a blink based on Eye Aspect Ratio.
    
    Threshold of 0.21 is research-backed (Soukupová & Čech, 2016) to avoid
    false positives from concentration-related eye narrowing.
    
    Args:
        ear: Eye Aspect Ratio (vertical/horizontal eye opening ratio)
        threshold: EAR threshold below which a blink is detected
        
    Returns:
        True if blinking, False otherwise
    """
    return ear < threshold


def is_multi_face_violation(face_count: int, consecutive_count: int, threshold: int) -> bool:
    """
    Check if multi-face detection threshold has been exceeded.
    
    Args:
        face_count: Number of faces detected in current frame
        consecutive_count: Number of consecutive frames with multiple faces
        threshold: Maximum allowed consecutive multi-face frames
        
    Returns:
        True if violation threshold exceeded, False otherwise
    """
    return face_count > 1 and consecutive_count >= threshold


def is_valid_landmark(landmark: Any) -> bool:
    """
    Validate that a landmark has required attributes.
    
    Args:
        landmark: MediaPipe landmark object
        
    Returns:
        True if landmark is valid, False otherwise
    """
    if landmark is None:
        return False
    return hasattr(landmark, 'x') and hasattr(landmark, 'y')


def is_valid_face_width(face_width: float) -> bool:
    """
    Validate that face width is positive and reasonable.
    
    Args:
        face_width: Computed face width in normalized coordinates
        
    Returns:
        True if face width is valid, False otherwise
    """
    # Face width should be positive and less than 1.0 (normalized coordinates)
    return 0.0 < face_width < 1.0


def validate_score_range(score: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """
    Clamp score to valid range.
    
    Args:
        score: Raw score value
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Clamped score within [min_val, max_val]
    """
    return max(min_val, min(max_val, score))


def validate_normalization_inputs(raw_value: float, face_width: float, metric_name: str) -> None:
    """
    Validate inputs for normalization to prevent division by zero or invalid values.
    
    Args:
        raw_value: Raw measurement value
        face_width: Face width for normalization
        metric_name: Name of metric being normalized (for error messages)
        
    Raises:
        ValueError: If face_width is invalid
    """
    if not is_valid_face_width(face_width):
        raise ValueError(
            f"Invalid face_width for {metric_name}: {face_width}. "
            f"Face width must be positive and < 1.0"
        )
