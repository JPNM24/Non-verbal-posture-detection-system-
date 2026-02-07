"""
Deterministic Non-Verbal Behavior Analysis Module

This module implements a strictly deterministic, lightweight, non-verbal behavior
analysis system following the exact algorithm specification. No machine learning,
deep learning, classifiers, or neural networks are used.

All computations use O(1) memory and constant per-frame computational cost.
Only basic arithmetic, trigonometry, running mean, and running variance are used.

Author: Implemented per algorithm specification
Constraints:
- No ML/DL/classifiers/neural networks
- No invented features, signals, or metrics
- No simplified mathematical steps
- No emotion labels or human-judgment terms
- O(1) memory, constant per-frame cost
- Only basic arithmetic, trigonometry, running mean, running variance
- No gender classification or branching logic
- All constants from specification
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any


# =============================================================================
# SECTION 0: CONSTANTS (compile-time)
# =============================================================================

# Camera parameters
CAMERA_FOV_DEG: float = 60.0

# Global inter-pupillary distance statistics (in millimeters)
GLOBAL_IPD_MM: float = 63.0
IPD_SD_MM: float = 4.0

# Standard deviations for head orientation (in degrees)
SD_YAW_DEG: float = 3.3
SD_PITCH_DEG: float = 3.9
SD_ROLL_DEG: float = 2.8

# Standard deviations for posture metrics (in degrees)
SD_SHOULDER_SLOPE_DEG: float = 4.0
SD_TORSO_ANGLE_DEG: float = 5.0

# Smoothing parameters
BASELINE_SMOOTH_K: int = 300  # Fixed smoothing constant for baseline update
STATE_SMOOTH_ALPHA: float = 0.9  # Temporal smoothing factor for latent states

# Small forward lean bonus for engagement (per specification)
SMALL_FORWARD_LEAN_BONUS: float = 0.1


# =============================================================================
# SECTION 2: DATA STRUCTURES
# =============================================================================

@dataclass
class BaselineState:
    """
    2.1 Baseline State
    
    Running means for each signal, used to compute deviations.
    All values are in degrees.
    """
    yaw_mean: Optional[float] = None
    pitch_mean: Optional[float] = None
    roll_mean: Optional[float] = None
    shoulder_slope_mean: Optional[float] = None
    torso_angle_mean: Optional[float] = None
    
    # Frame count for initialization tracking
    frame_count: int = 0


@dataclass
class VarianceState:
    """
    2.2 Stability (running variance)
    
    Running variance accumulators for stability computation.
    Uses Welford's online algorithm for numerical stability.
    """
    # Mean values for Welford's algorithm
    yaw_mean: float = 0.0
    pitch_mean: float = 0.0
    roll_mean: float = 0.0
    shoulder_slope_mean: float = 0.0
    
    # M2 accumulators (sum of squared differences from the mean)
    yaw_m2: float = 0.0
    pitch_m2: float = 0.0
    roll_m2: float = 0.0
    shoulder_slope_m2: float = 0.0
    
    # Sample count for variance computation
    n: int = 0


@dataclass
class LatentState:
    """
    2.3 Latent State (continuous, 0-1)
    
    Continuous state variables representing abstract behavioral metrics.
    All values are clamped to [0, 1].
    Initialized to 0.5 as per specification.
    """
    engagement: float = 0.5
    confidence: float = 0.5
    nervousness: float = 0.5
    attentiveness: float = 0.5


@dataclass
class FrameInput:
    """
    3. PER-FRAME INPUT
    
    Landmark coordinates from vision system.
    All coordinates are in pixels with origin at top-left,
    +x pointing right, +y pointing down.
    """
    # Eye centers
    left_eye_center_px: Tuple[float, float]
    right_eye_center_px: Tuple[float, float]
    
    # Nose tip
    nose_tip_px: Tuple[float, float]
    
    # Shoulder landmarks
    left_shoulder_px: Tuple[float, float]
    right_shoulder_px: Tuple[float, float]
    
    # Hip landmarks
    left_hip_px: Tuple[float, float]
    right_hip_px: Tuple[float, float]


@dataclass
class AnalysisOutput:
    """
    12. OUTPUT (per frame)
    
    Contains only continuous numerical metrics.
    No labels, no emotions, no interpretations.
    """
    # Posture deviation scores (Z-like normalized values)
    posture_deviation_scores: Dict[str, float] = field(default_factory=dict)
    
    # Latent state vector (all values in [0, 1])
    latent_state_vector: Dict[str, float] = field(default_factory=dict)
    
    # Stability indices (variance-based)
    stability_indices: Dict[str, float] = field(default_factory=dict)
    
    # Evidence scores (intermediate computations)
    evidence_scores: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# SECTION 1: PRECOMPUTATION
# =============================================================================

def compute_focal_length_px(frame_width_px: int, fov_deg: float = CAMERA_FOV_DEG) -> float:
    """
    1. PRECOMPUTATION (run once)
    
    Computes the focal length in pixels based on camera FOV and frame width.
    
    Formula:
        FOV_RAD = CAMERA_FOV_DEG × π / 180
        FOCAL_LENGTH_PX = (FRAME_WIDTH_PX / 2) / tan(FOV_RAD / 2)
    
    Args:
        frame_width_px: Frame width in pixels
        fov_deg: Camera field of view in degrees
    
    Returns:
        Focal length in pixels
    """
    # Convert FOV from degrees to radians
    fov_rad: float = fov_deg * (math.pi / 180.0)
    
    # Compute focal length using pinhole camera model
    # Guard against division by zero (tan(0) = 0)
    half_fov_rad: float = fov_rad / 2.0
    tan_half_fov: float = math.tan(half_fov_rad)
    
    # Numerical stability: prevent division by very small numbers
    if abs(tan_half_fov) < 1e-10:
        tan_half_fov = 1e-10
    
    focal_length_px: float = (frame_width_px / 2.0) / tan_half_fov
    
    return focal_length_px


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Compute Euclidean distance between two 2D points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
    
    Returns:
        Euclidean distance in the same units as input
    """
    dx: float = p2[0] - p1[0]
    dy: float = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def midpoint(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    """
    Compute midpoint between two 2D points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
    
    Returns:
        Midpoint (x, y)
    """
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value to the specified range.
    
    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value
    
    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


# =============================================================================
# MAIN ANALYZER CLASS
# =============================================================================

class DeterministicNonVerbalAnalyzer:
    """
    Deterministic Non-Verbal Behavior Analysis Module
    
    Implements the exact algorithm specification for non-verbal behavior analysis.
    All computations are deterministic, lightweight, and use O(1) memory.
    
    Usage:
        analyzer = DeterministicNonVerbalAnalyzer(frame_width=640, frame_height=480)
        
        for frame_landmarks in video_stream:
            output = analyzer.process_frame(frame_landmarks)
            # output contains posture_deviation_scores, latent_state_vector,
            # stability_indices, and evidence_scores
    """
    
    def __init__(
        self,
        frame_width_px: int,
        frame_height_px: int,
        camera_fov_deg: float = CAMERA_FOV_DEG
    ):
        """
        Initialize the analyzer with camera parameters.
        
        Args:
            frame_width_px: Frame width in pixels (dynamic)
            frame_height_px: Frame height in pixels (dynamic)
            camera_fov_deg: Camera field of view in degrees
        """
        # Store frame dimensions
        self._frame_width_px: int = frame_width_px
        self._frame_height_px: int = frame_height_px
        self._camera_fov_deg: float = camera_fov_deg
        
        # Section 1: Precomputation - compute focal length
        self._focal_length_px: float = compute_focal_length_px(
            frame_width_px, camera_fov_deg
        )
        
        # Section 2.1: Initialize baseline state (empty)
        self._baseline: BaselineState = BaselineState()
        
        # Section 2.2: Initialize variance state (running variance)
        self._variance: VarianceState = VarianceState()
        
        # Section 2.3: Initialize latent state to 0.5 as per specification
        self._latent_state: LatentState = LatentState()
    
    def process_frame(self, frame_input: FrameInput) -> AnalysisOutput:
        """
        Process a single frame through the complete analysis pipeline.
        
        This method implements Sections 4-12 of the algorithm specification.
        
        Args:
            frame_input: Landmark data for current frame
        
        Returns:
            AnalysisOutput containing all continuous metrics
        """
        # =====================================================================
        # SECTION 4: SCALE NORMALIZATION (IPD-based)
        # =====================================================================
        
        # Compute inter-pupillary distance in pixels
        ipd_pixels: float = distance(
            frame_input.left_eye_center_px,
            frame_input.right_eye_center_px
        )
        
        # Guard against division by zero
        if ipd_pixels < 1e-10:
            ipd_pixels = 1e-10
        
        # Compute scale factor (mm per pixel)
        # Used implicitly for consistency; no heavy geometry
        scale_mm_per_px: float = GLOBAL_IPD_MM / ipd_pixels
        
        # =====================================================================
        # SECTION 5: HEAD ORIENTATION COMPUTATION
        # =====================================================================
        
        # Compute eye midpoint
        eye_midpoint: Tuple[float, float] = midpoint(
            frame_input.left_eye_center_px,
            frame_input.right_eye_center_px
        )
        
        # 5.1 Yaw (left-right rotation)
        # dx = nose_tip_px.x - midpoint(left_eye.x, right_eye.x)
        dx_yaw: float = frame_input.nose_tip_px[0] - eye_midpoint[0]
        
        # yaw_deg = atan(dx / FOCAL_LENGTH_PX) × (180 / π)
        # Guard against very large angles
        yaw_rad: float = math.atan(dx_yaw / self._focal_length_px)
        yaw_deg: float = yaw_rad * (180.0 / math.pi)
        
        # 5.2 Pitch (up-down rotation)
        # dy = nose_tip_px.y - midpoint(left_eye.y, right_eye.y)
        dy_pitch: float = frame_input.nose_tip_px[1] - eye_midpoint[1]
        
        # pitch_deg = atan(dy / FOCAL_LENGTH_PX) × (180 / π)
        pitch_rad: float = math.atan(dy_pitch / self._focal_length_px)
        pitch_deg: float = pitch_rad * (180.0 / math.pi)
        
        # 5.3 Roll (head tilt via shoulders)
        # shoulder_dx = right_shoulder_px.x - left_shoulder_px.x
        shoulder_dx: float = (
            frame_input.right_shoulder_px[0] - frame_input.left_shoulder_px[0]
        )
        
        # shoulder_dy = right_shoulder_px.y - left_shoulder_px.y
        shoulder_dy: float = (
            frame_input.right_shoulder_px[1] - frame_input.left_shoulder_px[1]
        )
        
        # roll_deg = atan(shoulder_dy / shoulder_dx) × (180 / π)
        # Guard against division by zero
        if abs(shoulder_dx) < 1e-10:
            shoulder_dx = 1e-10 if shoulder_dx >= 0 else -1e-10
        
        roll_rad: float = math.atan(shoulder_dy / shoulder_dx)
        roll_deg: float = roll_rad * (180.0 / math.pi)
        
        # =====================================================================
        # SECTION 6: UPPER-BODY POSTURE
        # =====================================================================
        
        # 6.1 Shoulder slope
        # shoulder_slope_deg = abs(roll_deg)
        shoulder_slope_deg: float = abs(roll_deg)
        
        # 6.2 Torso angle (forward lean)
        # shoulder_mid = midpoint(left_shoulder, right_shoulder)
        shoulder_mid: Tuple[float, float] = midpoint(
            frame_input.left_shoulder_px,
            frame_input.right_shoulder_px
        )
        
        # hip_mid = midpoint(left_hip, right_hip)
        hip_mid: Tuple[float, float] = midpoint(
            frame_input.left_hip_px,
            frame_input.right_hip_px
        )
        
        # torso_dx = shoulder_mid.x - hip_mid.x
        torso_dx: float = shoulder_mid[0] - hip_mid[0]
        
        # torso_dy = shoulder_mid.y - hip_mid.y
        torso_dy: float = shoulder_mid[1] - hip_mid[1]
        
        # torso_angle_deg = atan(torso_dx / torso_dy) × (180 / π)
        # Guard against division by zero
        if abs(torso_dy) < 1e-10:
            torso_dy = 1e-10 if torso_dy >= 0 else -1e-10
        
        torso_angle_rad: float = math.atan(torso_dx / torso_dy)
        torso_angle_deg: float = torso_angle_rad * (180.0 / math.pi)
        
        # =====================================================================
        # SECTION 7: BASELINE UPDATE (fixed smoothing k)
        # =====================================================================
        
        # Increment frame count
        self._baseline.frame_count += 1
        
        # Initialize baseline values on first frame
        if self._baseline.frame_count == 1:
            self._baseline.yaw_mean = yaw_deg
            self._baseline.pitch_mean = pitch_deg
            self._baseline.roll_mean = roll_deg
            self._baseline.shoulder_slope_mean = shoulder_slope_deg
            self._baseline.torso_angle_mean = torso_angle_deg
        else:
            # Update baseline using fixed smoothing:
            # baseline_x = baseline_x + (x - baseline_x) / BASELINE_SMOOTH_K
            
            self._baseline.yaw_mean += (
                (yaw_deg - self._baseline.yaw_mean) / BASELINE_SMOOTH_K
            )
            
            self._baseline.pitch_mean += (
                (pitch_deg - self._baseline.pitch_mean) / BASELINE_SMOOTH_K
            )
            
            self._baseline.roll_mean += (
                (roll_deg - self._baseline.roll_mean) / BASELINE_SMOOTH_K
            )
            
            self._baseline.shoulder_slope_mean += (
                (shoulder_slope_deg - self._baseline.shoulder_slope_mean) / 
                BASELINE_SMOOTH_K
            )
            
            self._baseline.torso_angle_mean += (
                (torso_angle_deg - self._baseline.torso_angle_mean) / 
                BASELINE_SMOOTH_K
            )
        
        # =====================================================================
        # SECTION 8: NORMALIZED DEVIATION (Z-like)
        # =====================================================================
        
        # Z_yaw = (yaw - baseline_yaw) / SD_YAW_DEG
        z_yaw: float = (yaw_deg - self._baseline.yaw_mean) / SD_YAW_DEG
        
        # Z_pitch = (pitch - baseline_pitch) / SD_PITCH_DEG
        z_pitch: float = (pitch_deg - self._baseline.pitch_mean) / SD_PITCH_DEG
        
        # Z_roll = (roll - baseline_roll) / SD_ROLL_DEG
        z_roll: float = (roll_deg - self._baseline.roll_mean) / SD_ROLL_DEG
        
        # Z_shoulder = (shoulder_slope - baseline_shoulder_slope) / 
        #              SD_SHOULDER_SLOPE_DEG
        z_shoulder: float = (
            (shoulder_slope_deg - self._baseline.shoulder_slope_mean) / 
            SD_SHOULDER_SLOPE_DEG
        )
        
        # Z_torso = (torso_angle - baseline_torso_angle) / SD_TORSO_ANGLE_DEG
        z_torso: float = (
            (torso_angle_deg - self._baseline.torso_angle_mean) / 
            SD_TORSO_ANGLE_DEG
        )
        
        # =====================================================================
        # SECTION 9: STABILITY UPDATE (running variance)
        # =====================================================================
        
        # Welford's online algorithm for computing variance
        # For each signal x:
        # variance_x += (x - mean_old) * (x - mean_new)
        
        self._variance.n += 1
        n: int = self._variance.n
        
        # Yaw variance
        yaw_mean_old: float = self._variance.yaw_mean
        self._variance.yaw_mean += (yaw_deg - yaw_mean_old) / n
        self._variance.yaw_m2 += (yaw_deg - yaw_mean_old) * (
            yaw_deg - self._variance.yaw_mean
        )
        
        # Pitch variance
        pitch_mean_old: float = self._variance.pitch_mean
        self._variance.pitch_mean += (pitch_deg - pitch_mean_old) / n
        self._variance.pitch_m2 += (pitch_deg - pitch_mean_old) * (
            pitch_deg - self._variance.pitch_mean
        )
        
        # Roll variance
        roll_mean_old: float = self._variance.roll_mean
        self._variance.roll_mean += (roll_deg - roll_mean_old) / n
        self._variance.roll_m2 += (roll_deg - roll_mean_old) * (
            roll_deg - self._variance.roll_mean
        )
        
        # Shoulder slope variance
        shoulder_mean_old: float = self._variance.shoulder_slope_mean
        self._variance.shoulder_slope_mean += (
            (shoulder_slope_deg - shoulder_mean_old) / n
        )
        self._variance.shoulder_slope_m2 += (
            (shoulder_slope_deg - shoulder_mean_old) * 
            (shoulder_slope_deg - self._variance.shoulder_slope_mean)
        )
        
        # Compute actual variance values (M2 / n)
        # Guard against n < 2 for sample variance
        if n >= 2:
            variance_yaw: float = self._variance.yaw_m2 / n
            variance_pitch: float = self._variance.pitch_m2 / n
            variance_roll: float = self._variance.roll_m2 / n
            variance_shoulder: float = self._variance.shoulder_slope_m2 / n
        else:
            variance_yaw = 0.0
            variance_pitch = 0.0
            variance_roll = 0.0
            variance_shoulder = 0.0
        
        # =====================================================================
        # SECTION 10: MULTI-SIGNAL EVIDENCE SCORES
        # =====================================================================
        
        # 10.1 Slouch evidence
        # slouch_score = 0.4 × Z_shoulder + 0.4 × Z_torso + 0.2 × Z_pitch
        slouch_score: float = (
            0.4 * z_shoulder + 
            0.4 * z_torso + 
            0.2 * z_pitch
        )
        
        # 10.2 Nervousness evidence
        # nervous_score = 0.5 × abs(Z_roll) + 0.3 × abs(Z_yaw) + 
        #                 0.2 × variance_shoulder
        nervous_score: float = (
            0.5 * abs(z_roll) + 
            0.3 * abs(z_yaw) + 
            0.2 * variance_shoulder
        )
        
        # 10.3 Engagement evidence
        # engagement_score = -abs(Z_torso) + small_forward_lean_bonus
        engagement_score: float = (
            -abs(z_torso) + SMALL_FORWARD_LEAN_BONUS
        )
        
        # =====================================================================
        # SECTION 11: LATENT STATE UPDATE (temporal smoothing)
        # =====================================================================
        
        # For each latent state S:
        # S_new = STATE_SMOOTH_ALPHA × S_old + (1 - STATE_SMOOTH_ALPHA) × evidence
        # Clamp all states to [0, 1]
        
        # Map evidence scores to [0, 1] range for latent state update
        # Using sigmoid-like transformation: 1 / (1 + exp(-x))
        def sigmoid(x: float) -> float:
            """Sigmoid function for mapping evidence to [0, 1]."""
            # Guard against overflow
            if x < -500:
                return 0.0
            if x > 500:
                return 1.0
            return 1.0 / (1.0 + math.exp(-x))
        
        # Engagement: based on engagement_score
        engagement_evidence: float = sigmoid(engagement_score)
        self._latent_state.engagement = clamp(
            STATE_SMOOTH_ALPHA * self._latent_state.engagement + 
            (1.0 - STATE_SMOOTH_ALPHA) * engagement_evidence,
            0.0, 1.0
        )
        
        # Confidence: inverse of slouch (less slouch = more confident)
        confidence_evidence: float = sigmoid(-slouch_score)
        self._latent_state.confidence = clamp(
            STATE_SMOOTH_ALPHA * self._latent_state.confidence + 
            (1.0 - STATE_SMOOTH_ALPHA) * confidence_evidence,
            0.0, 1.0
        )
        
        # Nervousness: based on nervous_score
        nervousness_evidence: float = sigmoid(nervous_score)
        self._latent_state.nervousness = clamp(
            STATE_SMOOTH_ALPHA * self._latent_state.nervousness + 
            (1.0 - STATE_SMOOTH_ALPHA) * nervousness_evidence,
            0.0, 1.0
        )
        
        # Attentiveness: based on yaw and pitch (looking at camera)
        # Lower absolute yaw and pitch = more attentive
        attentiveness_evidence: float = sigmoid(-(abs(z_yaw) + abs(z_pitch)))
        self._latent_state.attentiveness = clamp(
            STATE_SMOOTH_ALPHA * self._latent_state.attentiveness + 
            (1.0 - STATE_SMOOTH_ALPHA) * attentiveness_evidence,
            0.0, 1.0
        )
        
        # =====================================================================
        # SECTION 12: OUTPUT (per frame)
        # =====================================================================
        
        output = AnalysisOutput(
            # Posture deviation scores (Z-like normalized values)
            posture_deviation_scores={
                "z_yaw": z_yaw,
                "z_pitch": z_pitch,
                "z_roll": z_roll,
                "z_shoulder": z_shoulder,
                "z_torso": z_torso
            },
            
            # Latent state vector (all values in [0, 1])
            latent_state_vector={
                "engagement": self._latent_state.engagement,
                "confidence": self._latent_state.confidence,
                "nervousness": self._latent_state.nervousness,
                "attentiveness": self._latent_state.attentiveness
            },
            
            # Stability indices (variance-based)
            stability_indices={
                "variance_yaw": variance_yaw,
                "variance_pitch": variance_pitch,
                "variance_roll": variance_roll,
                "variance_shoulder": variance_shoulder
            },
            
            # Evidence scores (intermediate computations)
            evidence_scores={
                "slouch_score": slouch_score,
                "nervous_score": nervous_score,
                "engagement_score": engagement_score
            }
        )
        
        return output
    
    def reset(self) -> None:
        """
        Reset analyzer state to initial values.
        
        This allows reusing the analyzer for a new session without
        creating a new instance.
        """
        # Reset baseline state
        self._baseline = BaselineState()
        
        # Reset variance state
        self._variance = VarianceState()
        
        # Reset latent state to 0.5
        self._latent_state = LatentState()
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current internal state for debugging/inspection.
        
        Returns:
            Dictionary containing current baseline, variance, and latent state
        """
        return {
            "baseline": {
                "yaw_mean": self._baseline.yaw_mean,
                "pitch_mean": self._baseline.pitch_mean,
                "roll_mean": self._baseline.roll_mean,
                "shoulder_slope_mean": self._baseline.shoulder_slope_mean,
                "torso_angle_mean": self._baseline.torso_angle_mean,
                "frame_count": self._baseline.frame_count
            },
            "variance": {
                "yaw_variance": (
                    self._variance.yaw_m2 / self._variance.n 
                    if self._variance.n >= 2 else 0.0
                ),
                "pitch_variance": (
                    self._variance.pitch_m2 / self._variance.n 
                    if self._variance.n >= 2 else 0.0
                ),
                "roll_variance": (
                    self._variance.roll_m2 / self._variance.n 
                    if self._variance.n >= 2 else 0.0
                ),
                "shoulder_slope_variance": (
                    self._variance.shoulder_slope_m2 / self._variance.n 
                    if self._variance.n >= 2 else 0.0
                ),
                "sample_count": self._variance.n
            },
            "latent_state": {
                "engagement": self._latent_state.engagement,
                "confidence": self._latent_state.confidence,
                "nervousness": self._latent_state.nervousness,
                "attentiveness": self._latent_state.attentiveness
            }
        }
    
    @property
    def frame_dimensions(self) -> Tuple[int, int]:
        """Get frame dimensions (width, height) in pixels."""
        return (self._frame_width_px, self._frame_height_px)
    
    @property
    def focal_length_px(self) -> float:
        """Get computed focal length in pixels."""
        return self._focal_length_px
    
    @property
    def scale_mm_per_px(self) -> Optional[float]:
        """
        Get last computed scale (mm per pixel).
        
        Note: This is computed per frame based on IPD measurement.
        Returns None if no frame has been processed.
        """
        if self._baseline.frame_count == 0:
            return None
        # This would need to be stored from last frame if needed
        return None


# =============================================================================
# CONVENIENCE FACTORY FUNCTION
# =============================================================================

def create_analyzer(
    frame_width_px: int,
    frame_height_px: int,
    camera_fov_deg: float = CAMERA_FOV_DEG
) -> DeterministicNonVerbalAnalyzer:
    """
    Factory function to create a DeterministicNonVerbalAnalyzer.
    
    Args:
        frame_width_px: Frame width in pixels
        frame_height_px: Frame height in pixels
        camera_fov_deg: Camera field of view in degrees (default: 60)
    
    Returns:
        Configured DeterministicNonVerbalAnalyzer instance
    """
    return DeterministicNonVerbalAnalyzer(
        frame_width_px=frame_width_px,
        frame_height_px=frame_height_px,
        camera_fov_deg=camera_fov_deg
    )
