import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Any, Dict
from dataclasses import dataclass
from enum import Enum

class PipelineStage(Enum):
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
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    should_skip_frame: bool = False
    should_cancel_session: bool = False
    cancellation_reason: Optional[str] = None

    @staticmethod
    def success_result(data: Any = None) -> 'PipelineResult':
        return PipelineResult(success=True, data=data)

    @staticmethod
    def error_result(error: str) -> 'PipelineResult':
        return PipelineResult(success=False, error=error)

    @staticmethod
    def skip_frame_result(reason: str) -> 'PipelineResult':
        return PipelineResult(
            success=True,
            should_skip_frame=True,
            data={"skip_reason": reason}
        )

    @staticmethod
    def cancel_session_result(reason: str) -> 'PipelineResult':
        return PipelineResult(
            success=True,
            should_cancel_session=True,
            cancellation_reason=reason
        )

class PipelineContext:
    def __init__(self, frame: np.ndarray, session_state: Any):
        self.frame = frame
        self.session_state = session_state

        self.rgb_frame: Optional[np.ndarray] = None
        self.face_results: Optional[Any] = None
        self.landmarks: Optional[Any] = None
        self.face_width: Optional[float] = None
        self.pose_results: Optional[Any] = None

        self.eye_contact_detected: Optional[bool] = None
        self.facial_engagement: Optional[float] = None
        self.posture_score: Optional[float] = None
        self.stability_score: Optional[float] = None

def validate_frame(frame: np.ndarray) -> PipelineResult:
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
    try:
        context.rgb_frame = cv2.cvtColor(context.frame, cv2.COLOR_BGR2RGB)
        return PipelineResult.success_result()
    except Exception as e:
        return PipelineResult.error_result(f"Failed to convert frame to RGB: {str(e)}")

def detect_faces(context: PipelineContext, face_mesh) -> PipelineResult:
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
    if context.face_results is None or not context.face_results.multi_face_landmarks:
        return PipelineResult.error_result("No face landmarks available")

    context.landmarks = context.face_results.multi_face_landmarks[0].landmark

    return PipelineResult.success_result({
        "landmark_count": len(context.landmarks)
    })

def normalize_by_face_size(context: PipelineContext) -> PipelineResult:
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
    from .utils import calculate_landmark_variance

    if context.landmarks is None:
        return PipelineResult.error_result("Landmarks not available")

    if context.face_width is None:
        return PipelineResult.error_result("Face width not available")

    if not context.session_state.has_sufficient_data_for_engagement():

        return PipelineResult.success_result({"engagement": None, "reason": "first_frame"})

    try:

        normalized_engagement = calculate_landmark_variance(
            context.landmarks,
            context.session_state.previous_landmarks,
            context.face_width
        )

        context.session_state.facial_engagement_scores.append(normalized_engagement)
        context.facial_engagement = normalized_engagement

        return PipelineResult.success_result({"engagement": normalized_engagement})
    except Exception as e:
        return PipelineResult.error_result(f"Failed to analyze facial expression: {str(e)}")

def analyze_posture(context: PipelineContext, pose_detector) -> PipelineResult:
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

            return PipelineResult.success_result({"posture_score": None, "reason": "no_pose_detected"})

        landmarks = pose_results.pose_landmarks.landmark

        nose = landmarks[0]
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]

        shoulder_mid_x = (l_shoulder.x + r_shoulder.x) / 2

        raw_alignment_error = abs(nose.x - shoulder_mid_x)

        raw_shoulder_tilt = abs(l_shoulder.y - r_shoulder.y)

        normalized_alignment = normalize_posture_alignment(raw_alignment_error, context.face_width)
        normalized_tilt = normalize_posture_alignment(raw_shoulder_tilt, context.face_width)

        score = 100 - (normalized_alignment * 200 + normalized_tilt * 200)
        score = validate_score_range(score, 0, 100)

        context.session_state.posture_scores.append(score)
        context.posture_score = score

        return PipelineResult.success_result({"posture_score": score})
    except Exception as e:
        return PipelineResult.error_result(f"Failed to analyze posture: {str(e)}")

def analyze_stability(context: PipelineContext) -> PipelineResult:
    from .utils import calculate_nose_movement
    from .validators import validate_score_range

    if context.landmarks is None:
        return PipelineResult.error_result("Landmarks not available")

    if context.face_width is None:
        return PipelineResult.error_result("Face width not available")

    if not context.session_state.has_sufficient_data_for_stability():

        return PipelineResult.success_result({"stability_score": None, "reason": "first_frame"})

    try:

        nose = context.landmarks[1]

        normalized_movement = calculate_nose_movement(
            nose,
            context.session_state.previous_nose_pos,
            context.face_width
        )

        stability_score = 100 - (normalized_movement * 1000)
        stability_score = validate_score_range(stability_score, 0, 100)

        context.session_state.stability_scores.append(stability_score)
        context.stability_score = stability_score

        return PipelineResult.success_result({"stability_score": stability_score})
    except Exception as e:
        return PipelineResult.error_result(f"Failed to analyze stability: {str(e)}")

def update_session_state(context: PipelineContext) -> PipelineResult:
    if context.landmarks is None:
        return PipelineResult.error_result("Landmarks not available")

    context.session_state.previous_landmarks = context.landmarks

    nose = context.landmarks[1]
    context.session_state.previous_nose_pos = (nose.x, nose.y, nose.z)

    return PipelineResult.success_result({"state_updated": True})
