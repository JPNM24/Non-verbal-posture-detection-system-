"""
Utility functions for non-verbal analysis.
Provides normalization, landmark calculations, and geometric utilities.
"""
import numpy as np
from typing import List, Any
from .validators import validate_normalization_inputs


def calculate_ear(eye_landmarks: List[Any]) -> float:
    """
    Compute Eye Aspect Ratio (EAR) for blink detection.
    
    EAR formula: (v1 + v2) / (2.0 * h)
    where v1, v2 are vertical distances and h is horizontal distance.
    
    Args:
        eye_landmarks: List of 6 MediaPipe landmarks for one eye
                      Expected order: [p1, p2, p3, p4, p5, p6]
                      where p1-p4 is horizontal, p2-p6 and p3-p5 are vertical
    
    Returns:
        Eye Aspect Ratio (higher = more open, lower = more closed)
    
    Raises:
        ValueError: If eye_landmarks doesn't have exactly 6 landmarks
    """
    if len(eye_landmarks) != 6:
        raise ValueError(f"Expected 6 eye landmarks, got {len(eye_landmarks)}")
    
    def dist(p1, p2):
        """Calculate Euclidean distance between two landmarks."""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    # Vertical distances (indices based on MediaPipe eye landmark structure)
    v1 = dist(eye_landmarks[1], eye_landmarks[5])
    v2 = dist(eye_landmarks[2], eye_landmarks[4])
    
    # Horizontal distance
    h = dist(eye_landmarks[0], eye_landmarks[3])
    
    if h == 0:
        return 0.0
    
    ear = (v1 + v2) / (2.0 * h)
    return ear


def normalize_movement(raw_movement: float, face_width: float, metric_name: str = "movement") -> float:
    """
    Normalize movement by face width to be camera-distance invariant.
    
    Formula: normalized_value = raw_value / face_width
    
    This ensures that the same physical movement produces the same normalized value
    regardless of camera distance.
    
    Args:
        raw_movement: Raw pixel movement value
        face_width: Face width in same units as raw_movement
        metric_name: Name of metric being normalized (for error messages)
    
    Returns:
        Normalized movement value
    
    Raises:
        ValueError: If face_width is invalid
    """
    validate_normalization_inputs(raw_movement, face_width, metric_name)
    return raw_movement / face_width


def normalize_posture_alignment(alignment_error: float, face_width: float) -> float:
    """
    Normalize posture alignment error by face width.
    
    Args:
        alignment_error: Raw alignment error (e.g., nose-to-shoulder-center distance)
        face_width: Face width for normalization
    
    Returns:
        Normalized alignment error
    
    Raises:
        ValueError: If face_width is invalid
    """
    return normalize_movement(alignment_error, face_width, "posture_alignment")


def get_face_width(landmarks: Any) -> float:
    """
    Estimate face width using lateral landmarks.
    
    Uses MediaPipe Face Mesh landmarks:
    - 454: Right side of face
    - 234: Left side of face
    
    Args:
        landmarks: MediaPipe face landmarks
    
    Returns:
        Face width in normalized coordinates
    
    Raises:
        IndexError: If landmarks don't contain required indices
    """
    # MediaPipe Face Mesh: 454 (right side), 234 (left side)
    left_side = landmarks[234]
    right_side = landmarks[454]
    
    # Calculate Euclidean distance
    width = np.sqrt(
        (left_side.x - right_side.x)**2 + 
        (left_side.y - right_side.y)**2 + 
        (left_side.z - right_side.z)**2
    )
    
    return width


def calculate_landmark_variance(current_landmarks: Any, previous_landmarks: Any, face_width: float) -> float:
    """
    Calculate normalized variance between current and previous landmarks.
    
    Measures movement of key facial features (nose, eye corners, mouth corners).
    Result is normalized by face width for camera-distance invariance.
    
    Args:
        current_landmarks: Current frame landmarks
        previous_landmarks: Previous frame landmarks
        face_width: Face width for normalization
    
    Returns:
        Normalized landmark variance (engagement metric)
    """
    # Key facial feature indices
    # 1: Nose tip, 33: Right eye outer corner, 263: Left eye outer corner
    # 61: Right mouth corner, 291: Left mouth corner
    key_indices = [1, 33, 263, 61, 291]
    
    diffs = []
    for i in key_indices:
        curr = current_landmarks[i]
        prev = previous_landmarks[i]
        
        # Calculate Euclidean distance
        d = np.sqrt(
            (curr.x - prev.x)**2 + 
            (curr.y - prev.y)**2 + 
            (curr.z - prev.z)**2
        )
        diffs.append(d)
    
    # Average movement
    avg_movement = np.mean(diffs)
    
    # Normalize by face width
    return normalize_movement(avg_movement, face_width, "facial_engagement")


def calculate_nose_movement(current_nose: Any, previous_nose_pos: tuple, face_width: float) -> float:
    """
    Calculate normalized nose movement for stability metric.
    
    Uses 3D coordinates (X, Y, Z) to capture all head movement including
    forward/backward motion which indicates instability in interviews.
    
    Args:
        current_nose: Current nose landmark
        previous_nose_pos: Previous nose position (x, y, z) tuple
        face_width: Face width for normalization
    
    Returns:
        Normalized nose movement
    """
    # Use 3D movement to capture forward/backward head instability
    if len(previous_nose_pos) == 2:
        # Backward compatibility: 2D position
        raw_movement = np.sqrt(
            (current_nose.x - previous_nose_pos[0])**2 + 
            (current_nose.y - previous_nose_pos[1])**2
        )
    else:
        # 3D position (recommended)
        raw_movement = np.sqrt(
            (current_nose.x - previous_nose_pos[0])**2 + 
            (current_nose.y - previous_nose_pos[1])**2 +
            (current_nose.z - previous_nose_pos[2])**2
        )
    
    return normalize_movement(raw_movement, face_width, "stability")

