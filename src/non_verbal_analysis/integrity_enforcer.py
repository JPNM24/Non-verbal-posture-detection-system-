from typing import Optional
from .session_manager import SessionManager, SessionState

class IntegrityEnforcer:

    MULTI_FACE_THRESHOLD = 15

    def __init__(self, multi_face_threshold: int = MULTI_FACE_THRESHOLD):
        self.multi_face_threshold = multi_face_threshold

    def check_multi_face_violation(self, face_count: int, state: SessionState) -> Optional[str]:
        if face_count > 1:
            state.multi_face_counter += 1
            if state.multi_face_counter >= self.multi_face_threshold:
                return "multiple_faces_detected"
        else:

            state.multi_face_counter = 0

        return None

    def enforce_cancellation(
        self,
        session_id: str,
        reason: str,
        session_manager: SessionManager
    ) -> None:
        session_manager.cancel_session(session_id, reason)

    def can_process_frame(self, state: SessionState) -> bool:
        return state.can_process_frame()
