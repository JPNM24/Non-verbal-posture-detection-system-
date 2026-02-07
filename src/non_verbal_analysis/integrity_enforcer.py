"""
Interview integrity enforcement.
Enforces rules that maintain interview validity (e.g., single participant).
"""
from typing import Optional
from .session_manager import SessionManager, SessionState


class IntegrityEnforcer:
    """
    Enforces interview integrity rules.
    
    Rules:
    1. Only one face allowed in frame (with tolerance for brief multi-face detections)
    2. Session must be forcibly cancelled if rules are violated
    3. Cancelled sessions cannot process further frames
    """
    
    # Consecutive frames with multiple faces before cancellation
    MULTI_FACE_THRESHOLD = 15
    
    def __init__(self, multi_face_threshold: int = MULTI_FACE_THRESHOLD):
        """
        Initialize integrity enforcer.
        
        Args:
            multi_face_threshold: Number of consecutive multi-face frames before cancellation
        """
        self.multi_face_threshold = multi_face_threshold
    
    def check_multi_face_violation(self, face_count: int, state: SessionState) -> Optional[str]:
        """
        Check if multi-face detection threshold has been exceeded.
        
        Args:
            face_count: Number of faces detected in current frame
            state: Session state
            
        Returns:
            Cancellation reason if violated, None otherwise
        """
        if face_count > 1:
            state.multi_face_counter += 1
            if state.multi_face_counter >= self.multi_face_threshold:
                return "multiple_faces_detected"
        else:
            # Reset counter if single face detected
            state.multi_face_counter = 0
        
        return None
    
    def enforce_cancellation(
        self,
        session_id: str,
        reason: str,
        session_manager: SessionManager
    ) -> None:
        """
        Forcibly cancel session and prevent further processing.
        
        Args:
            session_id: Session identifier
            reason: Reason for cancellation
            session_manager: Session manager instance
        """
        session_manager.cancel_session(session_id, reason)
    
    def can_process_frame(self, state: SessionState) -> bool:
        """
        Check if session can process frames.
        
        Args:
            state: Session state
            
        Returns:
            True if session can process frames, False if cancelled
        """
        return state.can_process_frame()
