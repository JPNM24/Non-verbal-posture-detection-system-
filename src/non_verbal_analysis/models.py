from pydantic import BaseModel, validator
from typing import List, Optional

class NonVerbalScores(BaseModel):
    eye_contact: Optional[float] = None
    facial_expression: Optional[float] = None
    posture: Optional[float] = None
    stability: Optional[float] = None
    final_non_verbal_score: Optional[float] = None

    @validator('eye_contact', 'facial_expression', 'posture', 'stability', 'final_non_verbal_score')
    def validate_score_range(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError(f"Score must be between 0 and 100, got {v}")
        return v

class AnalysisOutput(BaseModel):
    session_status: str = "active"
    non_verbal_scores: NonVerbalScores
    insights: List[str] = []

class InsufficientDataOutput(BaseModel):
    session_status: str = "insufficient_data"
    reason: str
    non_verbal_scores: NonVerbalScores = NonVerbalScores()
    insights: List[str] = []

class CancelledSessionOutput(BaseModel):
    session_status: str = "cancelled"
    cancellation_reason: str
    non_verbal_scores: NonVerbalScores = NonVerbalScores()
    insights: List[str] = []

class SkippedFrameOutput(BaseModel):
    session_status: str = "frame_skipped"
    skip_reason: str
    non_verbal_scores: NonVerbalScores = NonVerbalScores()
    insights: List[str] = []
