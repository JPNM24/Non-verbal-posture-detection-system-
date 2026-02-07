import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Any, Union

from .models import (
    AnalysisOutput,
    InsufficientDataOutput,
    CancelledSessionOutput,
    SkippedFrameOutput,
    NonVerbalScores
)
from .session_manager import SessionManager, SessionState
from .pipeline import (
    PipelineContext,
    PipelineResult,
    validate_frame,
    convert_to_rgb,
    detect_faces,
    enforce_single_face_rule,
    extract_landmarks,
    normalize_by_face_size,
    analyze_facial_expression,
    analyze_posture,
    analyze_stability,
    update_session_state
)
from .eye_contact_analyzer import analyze_eye_contact, get_eye_contact_score
from .integrity_enforcer import IntegrityEnforcer

class NonVerbalAnalyzer:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.session_manager = SessionManager()

        self.integrity_enforcer = IntegrityEnforcer(multi_face_threshold=15)

    def process_frame(
        self,
        session_id: str,
        frame: np.ndarray
    ) -> Dict[str, Any]:

        state = self.session_manager.get_or_create_session(session_id)

        if not state.can_process_frame():
            return self._generate_cancelled_output(state.cancellation_reason)

        result = validate_frame(frame)
        if not result.success:
            return self._generate_insufficient_data_output(result.error)

        context = PipelineContext(frame, state)

        result = convert_to_rgb(context)
        if not result.success:
            return self._generate_insufficient_data_output(result.error)

        result = detect_faces(context, self.mp_face_mesh)
        if not result.success:
            return self._generate_insufficient_data_output(result.error)

        result = enforce_single_face_rule(context, self.integrity_enforcer.multi_face_threshold)
        if result.should_cancel_session:

            self.session_manager.cancel_session(session_id, result.cancellation_reason)
            return self._generate_cancelled_output(result.cancellation_reason)

        result = extract_landmarks(context)
        if not result.success:
            return self._generate_insufficient_data_output(result.error)

        result = normalize_by_face_size(context)
        if not result.success:
            return self._generate_insufficient_data_output(result.error)

        result = analyze_eye_contact(context)
        if not result.success:
            return self._generate_insufficient_data_output(result.error)

        if result.should_skip_frame:

            return self._generate_skipped_frame_output(result.data.get("skip_reason", "unknown"))

        result = analyze_facial_expression(context)
        if not result.success:
            return self._generate_insufficient_data_output(result.error)

        result = analyze_posture(context, self.mp_pose)
        if not result.success:
            return self._generate_insufficient_data_output(result.error)

        result = analyze_stability(context)
        if not result.success:
            return self._generate_insufficient_data_output(result.error)

        result = update_session_state(context)
        if not result.success:
            return self._generate_insufficient_data_output(result.error)

        return self._generate_analysis_output(state)

    def _generate_analysis_output(self, state: SessionState) -> Dict[str, Any]:

        ROLLING_WINDOW = 30

        eye_contact_score = get_eye_contact_score(state) if state.total_processed_frames > 0 else None

        if len(state.facial_engagement_scores) > 0:
            recent_engagement = state.facial_engagement_scores[-ROLLING_WINDOW:]
            avg_engagement = np.mean(recent_engagement)

            expr_score = min(100, np.log1p(avg_engagement * 1000) / np.log1p(50) * 100)
        else:
            expr_score = None

        if len(state.posture_scores) > 0:
            recent_posture = state.posture_scores[-ROLLING_WINDOW:]
            posture_score = np.mean(recent_posture)
        else:
            posture_score = None

        if len(state.stability_scores) > 0:
            recent_stability = state.stability_scores[-ROLLING_WINDOW:]
            stability_score = np.mean(recent_stability)
        else:
            stability_score = None

        if all(s is not None for s in [eye_contact_score, expr_score, posture_score, stability_score]):
            final_score = (
                0.35 * eye_contact_score +
                0.25 * expr_score +
                0.25 * posture_score +
                0.15 * stability_score
            )
        else:
            final_score = None

        def round_score(s):
            return round(s, 2) if s is not None else None

        scores = NonVerbalScores(
            eye_contact=round_score(eye_contact_score),
            facial_expression=round_score(expr_score),
            posture=round_score(posture_score),
            stability=round_score(stability_score),
            final_non_verbal_score=round_score(final_score)
        )

        insights = []
        if eye_contact_score is not None and eye_contact_score < 50:
            insights.append("Eye contact was inconsistent")
        if posture_score is not None and posture_score < 50:
            insights.append("Posture needs improvement")
        if stability_score is not None and stability_score < 50:
            insights.append("Frequent movement detected; try to remain still")
        if expr_score is not None and expr_score < 30:
            insights.append("Facial engagement appears low")

        output = AnalysisOutput(
            session_status="active",
            non_verbal_scores=scores,
            insights=insights
        )

        return output.dict()

    def _generate_insufficient_data_output(self, reason: str) -> Dict[str, Any]:
        output = InsufficientDataOutput(
            reason=reason,
            insights=["Insufficient data to compute metrics"]
        )
        return output.dict()

    def _generate_cancelled_output(self, reason: str) -> Dict[str, Any]:
        output = CancelledSessionOutput(
            cancellation_reason=reason,
            insights=["Session cancelled due to integrity violation"]
        )
        return output.dict()

    def _generate_skipped_frame_output(self, reason: str) -> Dict[str, Any]:
        output = SkippedFrameOutput(
            skip_reason=reason,
            insights=[]
        )
        return output.dict()

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        if not self.session_manager.validate_session(session_id):
            return {"error": "Session not found or invalid"}

        state = self.session_manager.get_session(session_id)
        duration = self.session_manager.get_session_duration(session_id)

        return {
            "session_id": session_id,
            "duration_seconds": duration,
            "total_frames_processed": state.total_processed_frames,
            "eye_contact_frames": state.eye_contact_frames,
            "is_cancelled": state.is_cancelled,
            "cancellation_reason": state.cancellation_reason
        }

    def delete_session(self, session_id: str) -> None:
        self.session_manager.delete_session(session_id)
