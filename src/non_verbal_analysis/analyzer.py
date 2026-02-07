"""
Non-verbal analysis orchestrator.
Coordinates the strict processing pipeline for frame-by-frame analysis.
"""
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
    """
    Non-verbal analysis orchestrator.
    
    Responsibilities:
    1. Initialize MediaPipe models
    2. Orchestrate pipeline execution
    3. Handle errors and edge cases
    4. Generate structured outputs
    
    This class does NOT contain feature extraction logic.
    All analysis is delegated to pipeline stages.
    """
    
    def __init__(self):
        """Initialize MediaPipe models and pipeline components."""
        # MediaPipe Face Mesh for facial landmarks
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,  # To detect multiple faces for integrity enforcement
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe Pose for posture analysis
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Session manager
        self.session_manager = SessionManager()
        
        # Integrity enforcer
        self.integrity_enforcer = IntegrityEnforcer(multi_face_threshold=15)
    
    def process_frame(
        self,
        session_id: str,
        frame: np.ndarray
    ) -> Dict[str, Any]:
        """
        Process a single frame through the strict pipeline.
        
        Pipeline stages (MUST execute in order):
        1. Validate frame
        2. Convert to RGB
        3. Detect faces
        4. Enforce single-face rule
        5. Extract landmarks
        6. Normalize by face size
        7. Analyze eye contact (with blink exclusion)
        8. Analyze facial expression
        9. Analyze posture
        10. Analyze stability
        11. Update session state
        12. Generate output
        
        Args:
            session_id: Unique session identifier
            frame: Input frame (BGR numpy array)
            
        Returns:
            Dictionary with analysis results (conforms to output schema)
        """
        # Get or create session
        state = self.session_manager.get_or_create_session(session_id)
        
        # Check if session is cancelled
        if not state.can_process_frame():
            return self._generate_cancelled_output(state.cancellation_reason)
        
        # Stage 1: Validate frame
        result = validate_frame(frame)
        if not result.success:
            return self._generate_insufficient_data_output(result.error)
        
        # Create pipeline context
        context = PipelineContext(frame, state)
        
        # Stage 2: Convert to RGB
        result = convert_to_rgb(context)
        if not result.success:
            return self._generate_insufficient_data_output(result.error)
        
        # Stage 3: Detect faces
        result = detect_faces(context, self.mp_face_mesh)
        if not result.success:
            return self._generate_insufficient_data_output(result.error)
        
        # Stage 4: Enforce single-face rule
        result = enforce_single_face_rule(context, self.integrity_enforcer.multi_face_threshold)
        if result.should_cancel_session:
            # Forcibly cancel session
            self.session_manager.cancel_session(session_id, result.cancellation_reason)
            return self._generate_cancelled_output(result.cancellation_reason)
        
        # Stage 5: Extract landmarks
        result = extract_landmarks(context)
        if not result.success:
            return self._generate_insufficient_data_output(result.error)
        
        # Stage 6: Normalize by face size
        result = normalize_by_face_size(context)
        if not result.success:
            return self._generate_insufficient_data_output(result.error)
        
        # Stage 7: Analyze eye contact (with blink exclusion)
        result = analyze_eye_contact(context)
        if not result.success:
            return self._generate_insufficient_data_output(result.error)
        
        # CRITICAL: If blink detected, skip frame entirely
        if result.should_skip_frame:
            # Do NOT update state, do NOT generate output
            # This frame is completely ignored
            return self._generate_skipped_frame_output(result.data.get("skip_reason", "unknown"))
        
        # Stage 8: Analyze facial expression
        result = analyze_facial_expression(context)
        if not result.success:
            return self._generate_insufficient_data_output(result.error)
        
        # Stage 9: Analyze posture
        result = analyze_posture(context, self.mp_pose)
        if not result.success:
            return self._generate_insufficient_data_output(result.error)
        
        # Stage 10: Analyze stability
        result = analyze_stability(context)
        if not result.success:
            return self._generate_insufficient_data_output(result.error)
        
        # Stage 11: Update session state for next frame
        result = update_session_state(context)
        if not result.success:
            return self._generate_insufficient_data_output(result.error)
        
        # Stage 12: Generate output
        return self._generate_analysis_output(state)
    
    def _generate_analysis_output(self, state: SessionState) -> Dict[str, Any]:
        """
        Generate analysis output from session state.
        
        CRITICAL RULES:
        1. Use None for scores that cannot be computed
        2. Never guess or infer missing data
        3. Validate all scores are in range [0, 100]
        4. Use rolling window for responsive score updates
        
        Args:
            state: Session state with accumulated metrics
            
        Returns:
            Dictionary conforming to AnalysisOutput schema
        """
        # Rolling window size (last N frames for responsive updates)
        ROLLING_WINDOW = 30  # ~1 second at 30fps
        
        # Eye contact score
        eye_contact_score = get_eye_contact_score(state) if state.total_processed_frames > 0 else None
        
        # Facial expression score (engagement) - use rolling window
        # Use logarithmic scaling for better sensitivity across engagement range
        if len(state.facial_engagement_scores) > 0:
            recent_engagement = state.facial_engagement_scores[-ROLLING_WINDOW:]
            avg_engagement = np.mean(recent_engagement)
            # Logarithmic scaling: log(1 + x * 1000) / log(1 + 50) * 100
            expr_score = min(100, np.log1p(avg_engagement * 1000) / np.log1p(50) * 100)
        else:
            expr_score = None
        
        # Posture score - use rolling window
        if len(state.posture_scores) > 0:
            recent_posture = state.posture_scores[-ROLLING_WINDOW:]
            posture_score = np.mean(recent_posture)
        else:
            posture_score = None
        
        # Stability score - use rolling window
        if len(state.stability_scores) > 0:
            recent_stability = state.stability_scores[-ROLLING_WINDOW:]
            stability_score = np.mean(recent_stability)
        else:
            stability_score = None
        
        # Final score (weighted average)
        # Only compute if all component scores are available
        if all(s is not None for s in [eye_contact_score, expr_score, posture_score, stability_score]):
            final_score = (
                0.35 * eye_contact_score +
                0.25 * expr_score +
                0.25 * posture_score +
                0.15 * stability_score
            )
        else:
            final_score = None
        
        # Round scores to 2 decimal places
        def round_score(s):
            return round(s, 2) if s is not None else None
        
        scores = NonVerbalScores(
            eye_contact=round_score(eye_contact_score),
            facial_expression=round_score(expr_score),
            posture=round_score(posture_score),
            stability=round_score(stability_score),
            final_non_verbal_score=round_score(final_score)
        )
        
        # Generate insights
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
        """
        Generate output when insufficient data is available.
        
        Args:
            reason: Reason for insufficient data
            
        Returns:
            Dictionary conforming to InsufficientDataOutput schema
        """
        output = InsufficientDataOutput(
            reason=reason,
            insights=["Insufficient data to compute metrics"]
        )
        return output.dict()
    
    def _generate_cancelled_output(self, reason: str) -> Dict[str, Any]:
        """
        Generate output when session is cancelled.
        
        Args:
            reason: Reason for cancellation
            
        Returns:
            Dictionary conforming to CancelledSessionOutput schema
        """
        output = CancelledSessionOutput(
            cancellation_reason=reason,
            insights=["Session cancelled due to integrity violation"]
        )
        return output.dict()
    
    def _generate_skipped_frame_output(self, reason: str) -> Dict[str, Any]:
        """
        Generate output when frame is skipped (e.g., blink).
        
        This is an internal output that may not be shown to users.
        
        Args:
            reason: Reason for skipping frame
            
        Returns:
            Dictionary conforming to SkippedFrameOutput schema
        """
        output = SkippedFrameOutput(
            skip_reason=reason,
            insights=[]
        )
        return output.dict()
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get summary of session metrics.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Dictionary with session summary
        """
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
        """
        Delete session and clean up resources.
        
        Args:
            session_id: Unique session identifier
        """
        self.session_manager.delete_session(session_id)
