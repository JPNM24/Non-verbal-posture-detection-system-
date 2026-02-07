import time
from typing import Dict, Any, List, Optional

class SessionState:

    BLINK_BUFFER_SIZE = 3

    def __init__(self):

        self.start_time = time.time()
        self.is_cancelled: bool = False
        self.cancellation_reason: Optional[str] = None

        self.previous_landmarks: Optional[Any] = None
        self.previous_nose_pos: Optional[tuple] = None
        self.face_width_history: List[float] = []

        self.eye_contact_frames: int = 0
        self.total_processed_frames: int = 0
        self.multi_face_counter: int = 0

        self.facial_engagement_scores: List[float] = []
        self.posture_scores: List[float] = []
        self.stability_scores: List[float] = []

        self.blink_buffer: List[bool] = []

    def can_process_frame(self) -> bool:
        return not self.is_cancelled

    def validate_temporal_continuity(self) -> bool:

        if self.previous_landmarks is None:
            return True

        if len(self.face_width_history) == 0:
            return False

        return True

    def has_sufficient_data_for_stability(self) -> bool:
        return self.previous_nose_pos is not None

    def has_sufficient_data_for_engagement(self) -> bool:
        return self.previous_landmarks is not None

    def get_average_face_width(self) -> Optional[float]:
        if len(self.face_width_history) == 0:
            return None
        return sum(self.face_width_history) / len(self.face_width_history)

class SessionManager:

    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}

    def get_or_create_session(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState()
        return self.sessions[session_id]

    def get_session(self, session_id: str) -> SessionState:
        return self.get_or_create_session(session_id)

    def validate_session(self, session_id: str) -> bool:
        if session_id not in self.sessions:
            return False

        state = self.sessions[session_id]
        return state.can_process_frame()

    def cancel_session(self, session_id: str, reason: str) -> None:
        if session_id in self.sessions:
            state = self.sessions[session_id]
            state.is_cancelled = True
            state.cancellation_reason = reason

    def is_cancelled(self, session_id: str) -> bool:
        if session_id not in self.sessions:
            return False
        return self.sessions[session_id].is_cancelled

    def delete_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_session_duration(self, session_id: str) -> Optional[float]:
        if session_id not in self.sessions:
            return None
        return time.time() - self.sessions[session_id].start_time
