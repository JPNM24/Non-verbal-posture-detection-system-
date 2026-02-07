"""
Strict output schema models.
Enforces type safety and validation for all analysis outputs.
"""
from pydantic import BaseModel, validator
from typing import List, Optional


class NonVerbalScores(BaseModel):
    """
    Non-verbal analysis scores.
    
    All scores are Optional to support null values when data is insufficient.
    Valid scores are in range [0.0, 100.0].
    """
    eye_contact: Optional[float] = None
    facial_expression: Optional[float] = None
    posture: Optional[float] = None
    stability: Optional[float] = None
    final_non_verbal_score: Optional[float] = None
    
    @validator('eye_contact', 'facial_expression', 'posture', 'stability', 'final_non_verbal_score')
    def validate_score_range(cls, v):
        """Validate that scores are in valid range [0, 100]."""
        if v is not None and (v < 0 or v > 100):
            raise ValueError(f"Score must be between 0 and 100, got {v}")
        return v


class AnalysisOutput(BaseModel):
    """
    Standard analysis output for active sessions.
    
    This is returned when session is processing normally.
    """
    session_status: str = "active"
    non_verbal_scores: NonVerbalScores
    insights: List[str] = []


class InsufficientDataOutput(BaseModel):
    """
    Output when insufficient data is available for analysis.
    
    This is returned when:
    - No face detected
    - Face detection failed
    - Landmarks unavailable
    - Any other condition preventing analysis
    """
    session_status: str = "insufficient_data"
    reason: str
    non_verbal_scores: NonVerbalScores = NonVerbalScores()  # All None
    insights: List[str] = []


class CancelledSessionOutput(BaseModel):
    """
    Output when session is cancelled due to integrity violation.
    
    This is returned when:
    - Multiple faces detected for too long
    - Any other integrity rule violation
    """
    session_status: str = "cancelled"
    cancellation_reason: str
    non_verbal_scores: NonVerbalScores = NonVerbalScores()  # All None
    insights: List[str] = []


class SkippedFrameOutput(BaseModel):
    """
    Output when frame is skipped (e.g., blink detected).
    
    This is an internal output that may not be returned to external callers.
    """
    session_status: str = "frame_skipped"
    skip_reason: str
    non_verbal_scores: NonVerbalScores = NonVerbalScores()  # All None
    insights: List[str] = []
