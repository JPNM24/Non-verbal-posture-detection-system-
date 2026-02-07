"""
Eye contact analysis with strict blink exclusion.
Isolates eye contact logic to ensure blink frames are completely excluded.
"""
import numpy as np
from typing import List, Any
from .pipeline import PipelineResult, PipelineContext
from .utils import calculate_ear
from .validators import is_blink_frame


# MediaPipe Face Mesh landmark indices for eyes
# Left eye: 362, 385, 387, 263, 373, 380
# Right eye: 33, 160, 158, 133, 153, 144
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Iris landmarks for gaze estimation
LEFT_IRIS_INDEX = 468
RIGHT_IRIS_INDEX = 473

# Eye corner landmarks for gaze ratio calculation
LEFT_EYE_INNER_CORNER = 362
LEFT_EYE_OUTER_CORNER = 263
RIGHT_EYE_OUTER_CORNER = 33
RIGHT_EYE_INNER_CORNER = 133


def analyze_eye_contact(context: PipelineContext, blink_threshold: float = 0.2) -> PipelineResult:
    """
    Stage 6: Analyze eye contact with strict blink exclusion.
    
    CRITICAL RULES:
    1. Blink frames are COMPLETELY EXCLUDED from all counters
    2. Gaze estimation uses normalized iris positions
    3. No assumptions about previous frames
    
    Args:
        context: Pipeline context with landmarks and face_width
        blink_threshold: EAR threshold for blink detection
        
    Returns:
        PipelineResult with eye contact analysis or skip signal
    """
    if context.landmarks is None:
        return PipelineResult.error_result("Landmarks not available for eye contact analysis")
    
    if context.face_width is None:
        return PipelineResult.error_result("Face width not available for normalization")
    
    landmarks = context.landmarks
    
    # Step 1: Extract eye landmarks
    try:
        left_eye = [landmarks[i] for i in LEFT_EYE_INDICES]
        right_eye = [landmarks[i] for i in RIGHT_EYE_INDICES]
    except IndexError as e:
        return PipelineResult.error_result(f"Invalid eye landmark indices: {str(e)}")
    
    # Step 2: Calculate Eye Aspect Ratio (EAR)
    try:
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
    except Exception as e:
        return PipelineResult.error_result(f"Failed to calculate EAR: {str(e)}")
    
    # Step 3: HARD RULE - Skip if blinking
    if is_blink_frame(avg_ear, blink_threshold):
        # DO NOT update any counters
        # DO NOT penalize the user
        # COMPLETELY SKIP this frame
        return PipelineResult.skip_frame_result("blink_detected")
    
    # Step 4: Estimate gaze (normalized by face width)
    try:
        gaze_detected = _estimate_gaze_normalized(landmarks, context.face_width)
    except Exception as e:
        return PipelineResult.error_result(f"Failed to estimate gaze: {str(e)}")
    
    # Step 5: Update counters ONLY if not blinking
    if gaze_detected:
        context.session_state.eye_contact_frames += 1
    
    context.session_state.total_processed_frames += 1
    context.eye_contact_detected = gaze_detected
    
    return PipelineResult.success_result({
        "gaze_detected": gaze_detected,
        "ear": avg_ear
    })


def _estimate_gaze_normalized(landmarks: Any, face_width: float) -> bool:
    """
    Estimate if user is looking at camera using normalized iris positions.
    
    This is a heuristic approach:
    - Calculate ratio of distances from iris to eye corners
    - Normalize by face width to be camera-distance invariant
    - If iris is centered (ratio close to 1.0), assume eye contact
    
    Args:
        landmarks: MediaPipe face landmarks
        face_width: Face width for normalization
        
    Returns:
        True if gaze is toward camera, False otherwise
    """
    def get_gaze_ratio(iris_idx: int, corner1_idx: int, corner2_idx: int) -> float:
        """Calculate normalized gaze ratio for one eye."""
        iris = landmarks[iris_idx]
        c1 = landmarks[corner1_idx]
        c2 = landmarks[corner2_idx]
        
        # Distance from iris to corners (normalized by face width)
        d1 = np.sqrt((iris.x - c1.x)**2 + (iris.y - c1.y)**2) / face_width
        d2 = np.sqrt((iris.x - c2.x)**2 + (iris.y - c2.y)**2) / face_width
        
        if d2 == 0:
            return 0.5
        
        return d1 / d2
    
    # Calculate gaze ratio for both eyes
    ratio_left = get_gaze_ratio(LEFT_IRIS_INDEX, LEFT_EYE_INNER_CORNER, LEFT_EYE_OUTER_CORNER)
    ratio_right = get_gaze_ratio(RIGHT_IRIS_INDEX, RIGHT_EYE_OUTER_CORNER, RIGHT_EYE_INNER_CORNER)
    
    # If ratio is close to 1.0, iris is centered
    # Gaze is toward camera if ratio is within [0.5, 2.0]
    # This threshold accounts for natural variation and head pose
    return 0.5 < ratio_left < 2.0 and 0.5 < ratio_right < 2.0


def get_eye_contact_score(session_state: Any) -> float:
    """
    Calculate eye contact score from session state.
    
    Args:
        session_state: Session state with counters
        
    Returns:
        Eye contact percentage (0-100)
    """
    if session_state.total_processed_frames == 0:
        return 0.0
    
    return (session_state.eye_contact_frames / session_state.total_processed_frames) * 100.0
