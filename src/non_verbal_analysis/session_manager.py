"""
Authoritative session state management.
Manages session lifecycle, validation, and temporal continuity.
"""
import time
from typing import Dict, Any, List, Optional


class SessionState:
    """
    Authoritative session state.
    
    This class maintains all temporal data required for frame-to-frame analysis.
    State MUST be validated before use to ensure temporal continuity.
    """
    
    # Blink buffer size for temporal blink detection
    BLINK_BUFFER_SIZE = 3
    
    def __init__(self):
        # Session metadata
        self.start_time = time.time()
        self.is_cancelled: bool = False
        self.cancellation_reason: Optional[str] = None
        
        # Temporal state (frame-to-frame)
        self.previous_landmarks: Optional[Any] = None
        self.previous_nose_pos: Optional[tuple] = None
        self.face_width_history: List[float] = []
        
        # Counters (CRITICAL: blink frames are excluded)
        self.eye_contact_frames: int = 0
        self.total_processed_frames: int = 0
        self.multi_face_counter: int = 0
        
        # Accumulators for scoring
        self.facial_engagement_scores: List[float] = []
        self.posture_scores: List[float] = []
        self.stability_scores: List[float] = []
        
        # Blink buffer for temporal blink detection
        self.blink_buffer: List[bool] = []
    
    def can_process_frame(self) -> bool:
        """
        Check if session is in valid state for processing.
        
        Returns:
            True if session can process frames, False if cancelled
        """
        return not self.is_cancelled
    
    def validate_temporal_continuity(self) -> bool:
        """
        Ensure previous frame data is compatible with current frame.
        
        This is critical for stability and engagement metrics that depend
        on frame-to-frame comparison.
        
        Returns:
            True if temporal continuity is valid, False otherwise
        """
        # For first frame, temporal continuity is not applicable
        if self.previous_landmarks is None:
            return True
        
        # If we have previous landmarks, we should also have face width history
        if len(self.face_width_history) == 0:
            return False
        
        return True
    
    def has_sufficient_data_for_stability(self) -> bool:
        """
        Check if we have sufficient data to compute stability.
        
        Stability requires previous nose position for comparison.
        
        Returns:
            True if stability can be computed, False otherwise
        """
        return self.previous_nose_pos is not None
    
    def has_sufficient_data_for_engagement(self) -> bool:
        """
        Check if we have sufficient data to compute facial engagement.
        
        Engagement requires previous landmarks for comparison.
        
        Returns:
            True if engagement can be computed, False otherwise
        """
        return self.previous_landmarks is not None
    
    def get_average_face_width(self) -> Optional[float]:
        """
        Get average face width from history.
        
        This can be used for normalization when current face width is unavailable.
        
        Returns:
            Average face width or None if no history
        """
        if len(self.face_width_history) == 0:
            return None
        return sum(self.face_width_history) / len(self.face_width_history)


class SessionManager:
    """
    Manages session lifecycle and state.
    
    Responsibilities:
    1. Create and retrieve sessions
    2. Validate session state
    3. Cancel sessions (forcibly)
    4. Clean up sessions
    """
    
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}
    
    def get_or_create_session(self, session_id: str) -> SessionState:
        """
        Get existing session or create new one.
        
        This is the primary method for retrieving session state.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            SessionState instance
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState()
        return self.sessions[session_id]
    
    def get_session(self, session_id: str) -> SessionState:
        """
        Get existing session (legacy compatibility).
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            SessionState instance
        """
        return self.get_or_create_session(session_id)
    
    def validate_session(self, session_id: str) -> bool:
        """
        Validate that session exists and is in valid state.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            True if session is valid, False otherwise
        """
        if session_id not in self.sessions:
            return False
        
        state = self.sessions[session_id]
        return state.can_process_frame()
    
    def cancel_session(self, session_id: str, reason: str) -> None:
        """
        Forcibly cancel session and prevent further processing.
        
        This is a HARD STOP - no further frames can be processed.
        
        Args:
            session_id: Unique session identifier
            reason: Reason for cancellation
        """
        if session_id in self.sessions:
            state = self.sessions[session_id]
            state.is_cancelled = True
            state.cancellation_reason = reason
    
    def is_cancelled(self, session_id: str) -> bool:
        """
        Check if session is cancelled.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            True if cancelled, False otherwise
        """
        if session_id not in self.sessions:
            return False
        return self.sessions[session_id].is_cancelled
    
    def delete_session(self, session_id: str) -> None:
        """
        Delete session and clean up resources.
        
        Args:
            session_id: Unique session identifier
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_session_duration(self, session_id: str) -> Optional[float]:
        """
        Get session duration in seconds.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Duration in seconds or None if session doesn't exist
        """
        if session_id not in self.sessions:
            return None
        return time.time() - self.sessions[session_id].start_time

