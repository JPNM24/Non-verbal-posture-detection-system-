"""
Strict processing pipeline for non-verbal analysis.
Enforces stage execution order and provides clear contracts for each stage.
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Any, Dict
from dataclasses import dataclass
from enum import Enum


class PipelineStage(Enum):
    """Enumeration of pipeline stages in execution order."""
    VALIDATE_FRAME = 1
    DETECT_FACES = 2
    ENFORCE_SINGLE_FACE = 3
    EXTRACT_LANDMARKS = 4
    NORMALIZE_FACE_SIZE = 5
    ANALYZE_EYE_CONTACT = 6
    ANALYZE_FACIAL_EXPRESSION = 7
    ANALYZE_POSTURE = 8
    ANALYZE_STABILITY = 9
    ACCUMULATE_SCORES = 10
    GENERATE_OUTPUT = 11


@dataclass
class PipelineResult:
    """
    Result contract for pipeline stages.
    
    Attributes:
        success: Whether the stage completed successfully
        data: Stage output data (None if failed)
        error: Error message if failed (None if successful)
        should_skip_frame: Whether to skip this frame (e.g., blink detected)
        should_cancel_session: Whether to cancel the entire session
        cancellation_reason: Reason for session cancellation
    """
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    should_skip_frame: bool = False
    should_cancel_session: bool = False
    cancellation_reason: Optional[str] = None
    
    @staticmethod
    def success_result(data: Any = None) -> 'PipelineResult':
        """Create a successful result."""
        return PipelineResult(success=True, data=data)
    
    @staticmethod
    def error_result(error: str) -> 'PipelineResult':
        """Create an error result."""
        return PipelineResult(success=False, error=error)
    
    @staticmethod
    def skip_frame_result(reason: str) -> 'PipelineResult':
        """Create a result that skips the current frame."""
        return PipelineResult(
            success=True,
            should_skip_frame=True,
            data={"skip_reason": reason}
        )
    
    @staticmethod
    def cancel_session_result(reason: str) -> 'PipelineResult':
        """Create a result that cancels the session."""
        return PipelineResult(
            success=True,
            should_cancel_session=True,
            cancellation_reason=reason
        )


class PipelineContext:
    """
    Context object passed through pipeline stages.
    Accumulates data from each stage for use by subsequent stages.
    """
    def __init__(self, frame: np.ndarray, session_state: Any):
        self.frame = frame
        self.session_state = session_state
        
        # Stage outputs
        self.rgb_frame: Optional[np.ndarray] = None
        self.face_results: Optional[Any] = None
        self.landmarks: Optional[Any] = None
        self.face_width: Optional[float] = None
        self.pose_results: Optional[Any] = None
        
        # Analysis results
        self.eye_contact_detected: Optional[bool] = None
        self.facial_engagement: Optional[float] = None
        self.posture_score: Optional[float] = None
        self.stability_score: Optional[float] = None


def validate_frame(frame: np.ndarray) -> PipelineResult:
    """
    Stage 1: Validate that frame is a valid numpy array.
    
    Args:
        frame: Input frame
        
    Returns:
        PipelineResult with validation status
    """
    if frame is None:
        return PipelineResult.error_result("Frame is None")
    
    if not isinstance(frame, np.ndarray):
        return PipelineResult.error_result(f"Frame must be numpy array, got {type(frame)}")
    
    if frame.size == 0:
        return PipelineResult.error_result("Frame is empty")
    
    if len(frame.shape) != 3:
        return PipelineResult.error_result(f"Frame must be 3D array (H, W, C), got shape {frame.shape}")
    
    return PipelineResult.success_result({"frame_shape": frame.shape})


def convert_to_rgb(context: PipelineContext) -> PipelineResult:
    """
    Convert BGR frame to RGB for MediaPipe processing.
    
    Args:
        context: Pipeline context
        
    Returns:
        PipelineResult with RGB frame
    """
    try:
        context.rgb_frame = cv2.cvtColor(context.frame, cv2.COLOR_BGR2RGB)
        return PipelineResult.success_result()
    except Exception as e:
        return PipelineResult.error_result(f"Failed to convert frame to RGB: {str(e)}")


def detect_faces(context: PipelineContext, face_mesh) -> PipelineResult:
    """
    Stage 2: Run MediaPipe face detection.
    
    Args:
        context: Pipeline context
        face_mesh: MediaPipe FaceMesh instance
        
    Returns:
        PipelineResult with face detection results
    """
    if context.rgb_frame is None:
        return PipelineResult.error_result("RGB frame not available")
    
    try:
        face_results = face_mesh.process(context.rgb_frame)
        context.face_results = face_results
        
        if not face_results.multi_face_landmarks:
            return PipelineResult.error_result("No face detected")
        
        return PipelineResult.success_result({
            "face_count": len(face_results.multi_face_landmarks)
        })
    except Exception as e:
        return PipelineResult.error_result(f"Face detection failed: {str(e)}")


def enforce_single_face_rule(context: PipelineContext, multi_face_threshold: int) -> PipelineResult:
    """
    Stage 3: Enforce single-face rule and check for violations.
    
    Args:
        context: Pipeline context
        multi_face_threshold: Maximum consecutive frames with multiple faces
        
    Returns:
        PipelineResult with enforcement status
    """
    if context.face_results is None:
        return PipelineResult.error_result("Face results not available")
    
    face_count = len(context.face_results.multi_face_landmarks) if context.face_results.multi_face_landmarks else 0
    
    if face_count > 1:
        context.session_state.multi_face_counter += 1
        if context.session_state.multi_face_counter >= multi_face_threshold:
            return PipelineResult.cancel_session_result("multiple_faces_detected")
    else:
        context.session_state.multi_face_counter = 0
    
    return PipelineResult.success_result({"face_count": face_count})


def extract_landmarks(context: PipelineContext) -> PipelineResult:
    """
    Stage 4: Extract facial landmarks from first detected face.
    
    Args:
        context: Pipeline context
        
    Returns:
        PipelineResult with landmarks
    """
    if context.face_results is None or not context.face_results.multi_face_landmarks:
        return PipelineResult.error_result("No face landmarks available")
    
    # Use first face only
    context.landmarks = context.face_results.multi_face_landmarks[0].landmark
    
    return PipelineResult.success_result({
        "landmark_count": len(context.landmarks)
    })


def normalize_by_face_size(context: PipelineContext) -> PipelineResult:
    """
    Stage 5: Compute face width for normalization.
    
    Args:
        context: Pipeline context
        
    Returns:
        PipelineResult with face width
    """
    from .utils import get_face_width
    from .validators import is_valid_face_width
    
    if context.landmarks is None:
        return PipelineResult.error_result("Landmarks not available")
    
    try:
        face_width = get_face_width(context.landmarks)
        
        if not is_valid_face_width(face_width):
            return PipelineResult.error_result(f"Invalid face width: {face_width}")
        
        context.face_width = face_width
        context.session_state.face_width_history.append(face_width)
        
        return PipelineResult.success_result({"face_width": face_width})
    except Exception as e:
        return PipelineResult.error_result(f"Failed to compute face width: {str(e)}")


def analyze_facial_expression(context: PipelineContext) -> PipelineResult:
    """
    Stage 7: Analyze facial expression (engagement) with normalization.
    
    Measures movement of key facial features between frames.
    Normalized by face width for camera-distance invariance.
    
    Args:
        context: Pipeline context
        
    Returns:
        PipelineResult with engagement score
    """
    from .utils import calculate_landmark_variance
    
    if context.landmarks is None:
        return PipelineResult.error_result("Landmarks not available")
    
    if context.face_width is None:
        return PipelineResult.error_result("Face width not available")
    
    # Check if we have previous landmarks for comparison
    if not context.session_state.has_sufficient_data_for_engagement():
        # First frame - cannot compute engagement yet
        return PipelineResult.success_result({"engagement": None, "reason": "first_frame"})
    
    try:
        # Calculate normalized landmark variance
        normalized_engagement = calculate_landmark_variance(
            context.landmarks,
            context.session_state.previous_landmarks,
            context.face_width
        )
        
        # Store for accumulation
        context.session_state.facial_engagement_scores.append(normalized_engagement)
        context.facial_engagement = normalized_engagement
        
        return PipelineResult.success_result({"engagement": normalized_engagement})
    except Exception as e:
        return PipelineResult.error_result(f"Failed to analyze facial expression: {str(e)}")


def analyze_posture(context: PipelineContext, pose_detector) -> PipelineResult:
    """
    Stage 8: Analyze posture with normalization.
    
    Checks alignment of nose with shoulder midpoint and shoulder levelness.
    Normalized by face width for camera-distance invariance.
    
    Args:
        context: Pipeline context
        pose_detector: MediaPipe Pose instance
        
    Returns:
        PipelineResult with posture score
    """
    from .utils import normalize_posture_alignment
    from .validators import validate_score_range
    
    if context.rgb_frame is None:
        return PipelineResult.error_result("RGB frame not available")
    
    if context.face_width is None:
        return PipelineResult.error_result("Face width not available for normalization")
    
    try:
        pose_results = pose_detector.process(context.rgb_frame)
        context.pose_results = pose_results
        
        if not pose_results.pose_landmarks:
            # No pose detected - return None score
            return PipelineResult.success_result({"posture_score": None, "reason": "no_pose_detected"})
        
        landmarks = pose_results.pose_landmarks.landmark
        
        # Nose: 0, L Shoulder: 11, R Shoulder: 12
        nose = landmarks[0]
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        
        # Calculate shoulder midpoint
        shoulder_mid_x = (l_shoulder.x + r_shoulder.x) / 2
        
        # Alignment error (raw)
        raw_alignment_error = abs(nose.x - shoulder_mid_x)
        
        # Shoulder tilt (raw)
        raw_shoulder_tilt = abs(l_shoulder.y - r_shoulder.y)
        
        # CRITICAL: Normalize by face width
        normalized_alignment = normalize_posture_alignment(raw_alignment_error, context.face_width)
        normalized_tilt = normalize_posture_alignment(raw_shoulder_tilt, context.face_width)
        
        # Compute score (100 = perfect alignment)
        # Scaling factors tuned for normalized values (200x is reasonable for typical posture variation)
        score = 100 - (normalized_alignment * 200 + normalized_tilt * 200)
        score = validate_score_range(score, 0, 100)
        
        context.session_state.posture_scores.append(score)
        context.posture_score = score
        
        return PipelineResult.success_result({"posture_score": score})
    except Exception as e:
        return PipelineResult.error_result(f"Failed to analyze posture: {str(e)}")


def analyze_stability(context: PipelineContext) -> PipelineResult:
    """
    Stage 9: Analyze stability (head movement) with normalization.
    
    Measures nose movement between frames.
    Normalized by face width for camera-distance invariance.
    
    Args:
        context: Pipeline context
        
    Returns:
        PipelineResult with stability score
    """
    from .utils import calculate_nose_movement
    from .validators import validate_score_range
    
    if context.landmarks is None:
        return PipelineResult.error_result("Landmarks not available")
    
    if context.face_width is None:
        return PipelineResult.error_result("Face width not available")
    
    # Check if we have previous nose position
    if not context.session_state.has_sufficient_data_for_stability():
        # First frame - cannot compute stability yet
        return PipelineResult.success_result({"stability_score": None, "reason": "first_frame"})
    
    try:
        # Get current nose position
        nose = context.landmarks[1]  # Nose tip
        
        # Calculate normalized nose movement
        normalized_movement = calculate_nose_movement(
            nose,
            context.session_state.previous_nose_pos,
            context.face_width
        )
        
        # Stability is inverse of movement
        # Scaling factor: 1000 (tuned for normalized values)
        # Higher movement = lower stability
        stability_score = 100 - (normalized_movement * 1000)
        stability_score = validate_score_range(stability_score, 0, 100)
        
        context.session_state.stability_scores.append(stability_score)
        context.stability_score = stability_score
        
        return PipelineResult.success_result({"stability_score": stability_score})
    except Exception as e:
        return PipelineResult.error_result(f"Failed to analyze stability: {str(e)}")


def update_session_state(context: PipelineContext) -> PipelineResult:
    """
    Stage 10: Update session state with current frame data.
    
    This must be called after all analysis stages to prepare for next frame.
    
    Args:
        context: Pipeline context
        
    Returns:
        PipelineResult indicating success
    """
    if context.landmarks is None:
        return PipelineResult.error_result("Landmarks not available")
    
    # Update temporal state for next frame
    context.session_state.previous_landmarks = context.landmarks
    
    # Update nose position for stability tracking (3D for depth detection)
    nose = context.landmarks[1]  # Nose tip
    context.session_state.previous_nose_pos = (nose.x, nose.y, nose.z)
    
    return PipelineResult.success_result({"state_updated": True})
