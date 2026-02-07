import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

CAMERA_FOV_DEG: float = 60.0

GLOBAL_IPD_MM: float = 63.0
IPD_SD_MM: float = 4.0

SD_YAW_DEG: float = 3.3
SD_PITCH_DEG: float = 3.9
SD_ROLL_DEG: float = 2.8

SD_SHOULDER_SLOPE_DEG: float = 4.0
SD_TORSO_ANGLE_DEG: float = 5.0

BASELINE_SMOOTH_K: int = 300
STATE_SMOOTH_ALPHA: float = 0.9

SMALL_FORWARD_LEAN_BONUS: float = 0.1

@dataclass
class BaselineState:
    yaw_mean: Optional[float] = None
    pitch_mean: Optional[float] = None
    roll_mean: Optional[float] = None
    shoulder_slope_mean: Optional[float] = None
    torso_angle_mean: Optional[float] = None

    frame_count: int = 0

@dataclass
class VarianceState:

    yaw_mean: float = 0.0
    pitch_mean: float = 0.0
    roll_mean: float = 0.0
    shoulder_slope_mean: float = 0.0

    yaw_m2: float = 0.0
    pitch_m2: float = 0.0
    roll_m2: float = 0.0
    shoulder_slope_m2: float = 0.0

    n: int = 0

@dataclass
class LatentState:
    engagement: float = 0.5
    confidence: float = 0.5
    nervousness: float = 0.5
    attentiveness: float = 0.5

@dataclass
class FrameInput:

    left_eye_center_px: Tuple[float, float]
    right_eye_center_px: Tuple[float, float]

    nose_tip_px: Tuple[float, float]

    left_shoulder_px: Tuple[float, float]
    right_shoulder_px: Tuple[float, float]

    left_hip_px: Tuple[float, float]
    right_hip_px: Tuple[float, float]

@dataclass
class AnalysisOutput:

    posture_deviation_scores: Dict[str, float] = field(default_factory=dict)

    latent_state_vector: Dict[str, float] = field(default_factory=dict)

    stability_indices: Dict[str, float] = field(default_factory=dict)

    evidence_scores: Dict[str, float] = field(default_factory=dict)

def compute_focal_length_px(frame_width_px: int, fov_deg: float = CAMERA_FOV_DEG) -> float:

    fov_rad: float = fov_deg * (math.pi / 180.0)

    half_fov_rad: float = fov_rad / 2.0
    tan_half_fov: float = math.tan(half_fov_rad)

    if abs(tan_half_fov) < 1e-10:
        tan_half_fov = 1e-10

    focal_length_px: float = (frame_width_px / 2.0) / tan_half_fov

    return focal_length_px

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    dx: float = p2[0] - p1[0]
    dy: float = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)

def midpoint(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))

class DeterministicNonVerbalAnalyzer:

    def __init__(
        self,
        frame_width_px: int,
        frame_height_px: int,
        camera_fov_deg: float = CAMERA_FOV_DEG
    ):

        self._frame_width_px: int = frame_width_px
        self._frame_height_px: int = frame_height_px
        self._camera_fov_deg: float = camera_fov_deg

        self._focal_length_px: float = compute_focal_length_px(
            frame_width_px, camera_fov_deg
        )

        self._baseline: BaselineState = BaselineState()

        self._variance: VarianceState = VarianceState()

        self._latent_state: LatentState = LatentState()

    def process_frame(self, frame_input: FrameInput) -> AnalysisOutput:

        ipd_pixels: float = distance(
            frame_input.left_eye_center_px,
            frame_input.right_eye_center_px
        )

        if ipd_pixels < 1e-10:
            ipd_pixels = 1e-10

        scale_mm_per_px: float = GLOBAL_IPD_MM / ipd_pixels

        eye_midpoint: Tuple[float, float] = midpoint(
            frame_input.left_eye_center_px,
            frame_input.right_eye_center_px
        )

        dx_yaw: float = frame_input.nose_tip_px[0] - eye_midpoint[0]

        yaw_rad: float = math.atan(dx_yaw / self._focal_length_px)
        yaw_deg: float = yaw_rad * (180.0 / math.pi)

        dy_pitch: float = frame_input.nose_tip_px[1] - eye_midpoint[1]

        pitch_rad: float = math.atan(dy_pitch / self._focal_length_px)
        pitch_deg: float = pitch_rad * (180.0 / math.pi)

        shoulder_dx: float = (
            frame_input.right_shoulder_px[0] - frame_input.left_shoulder_px[0]
        )

        shoulder_dy: float = (
            frame_input.right_shoulder_px[1] - frame_input.left_shoulder_px[1]
        )

        if abs(shoulder_dx) < 1e-10:
            shoulder_dx = 1e-10 if shoulder_dx >= 0 else -1e-10

        roll_rad: float = math.atan(shoulder_dy / shoulder_dx)
        roll_deg: float = roll_rad * (180.0 / math.pi)

        shoulder_slope_deg: float = abs(roll_deg)

        shoulder_mid: Tuple[float, float] = midpoint(
            frame_input.left_shoulder_px,
            frame_input.right_shoulder_px
        )

        hip_mid: Tuple[float, float] = midpoint(
            frame_input.left_hip_px,
            frame_input.right_hip_px
        )

        torso_dx: float = shoulder_mid[0] - hip_mid[0]

        torso_dy: float = shoulder_mid[1] - hip_mid[1]

        if abs(torso_dy) < 1e-10:
            torso_dy = 1e-10 if torso_dy >= 0 else -1e-10

        torso_angle_rad: float = math.atan(torso_dx / torso_dy)
        torso_angle_deg: float = torso_angle_rad * (180.0 / math.pi)

        self._baseline.frame_count += 1

        if self._baseline.frame_count == 1:
            self._baseline.yaw_mean = yaw_deg
            self._baseline.pitch_mean = pitch_deg
            self._baseline.roll_mean = roll_deg
            self._baseline.shoulder_slope_mean = shoulder_slope_deg
            self._baseline.torso_angle_mean = torso_angle_deg
        else:

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

        z_yaw: float = (yaw_deg - self._baseline.yaw_mean) / SD_YAW_DEG

        z_pitch: float = (pitch_deg - self._baseline.pitch_mean) / SD_PITCH_DEG

        z_roll: float = (roll_deg - self._baseline.roll_mean) / SD_ROLL_DEG

        z_shoulder: float = (
            (shoulder_slope_deg - self._baseline.shoulder_slope_mean) /
            SD_SHOULDER_SLOPE_DEG
        )

        z_torso: float = (
            (torso_angle_deg - self._baseline.torso_angle_mean) /
            SD_TORSO_ANGLE_DEG
        )

        self._variance.n += 1
        n: int = self._variance.n

        yaw_mean_old: float = self._variance.yaw_mean
        self._variance.yaw_mean += (yaw_deg - yaw_mean_old) / n
        self._variance.yaw_m2 += (yaw_deg - yaw_mean_old) * (
            yaw_deg - self._variance.yaw_mean
        )

        pitch_mean_old: float = self._variance.pitch_mean
        self._variance.pitch_mean += (pitch_deg - pitch_mean_old) / n
        self._variance.pitch_m2 += (pitch_deg - pitch_mean_old) * (
            pitch_deg - self._variance.pitch_mean
        )

        roll_mean_old: float = self._variance.roll_mean
        self._variance.roll_mean += (roll_deg - roll_mean_old) / n
        self._variance.roll_m2 += (roll_deg - roll_mean_old) * (
            roll_deg - self._variance.roll_mean
        )

        shoulder_mean_old: float = self._variance.shoulder_slope_mean
        self._variance.shoulder_slope_mean += (
            (shoulder_slope_deg - shoulder_mean_old) / n
        )
        self._variance.shoulder_slope_m2 += (
            (shoulder_slope_deg - shoulder_mean_old) *
            (shoulder_slope_deg - self._variance.shoulder_slope_mean)
        )

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

        slouch_score: float = (
            0.4 * z_shoulder +
            0.4 * z_torso +
            0.2 * z_pitch
        )

        nervous_score: float = (
            0.5 * abs(z_roll) +
            0.3 * abs(z_yaw) +
            0.2 * variance_shoulder
        )

        engagement_score: float = (
            -abs(z_torso) + SMALL_FORWARD_LEAN_BONUS
        )

        def sigmoid(x: float) -> float:

            if x < -500:
                return 0.0
            if x > 500:
                return 1.0
            return 1.0 / (1.0 + math.exp(-x))

        engagement_evidence: float = sigmoid(engagement_score)
        self._latent_state.engagement = clamp(
            STATE_SMOOTH_ALPHA * self._latent_state.engagement +
            (1.0 - STATE_SMOOTH_ALPHA) * engagement_evidence,
            0.0, 1.0
        )

        confidence_evidence: float = sigmoid(-slouch_score)
        self._latent_state.confidence = clamp(
            STATE_SMOOTH_ALPHA * self._latent_state.confidence +
            (1.0 - STATE_SMOOTH_ALPHA) * confidence_evidence,
            0.0, 1.0
        )

        nervousness_evidence: float = sigmoid(nervous_score)
        self._latent_state.nervousness = clamp(
            STATE_SMOOTH_ALPHA * self._latent_state.nervousness +
            (1.0 - STATE_SMOOTH_ALPHA) * nervousness_evidence,
            0.0, 1.0
        )

        attentiveness_evidence: float = sigmoid(-(abs(z_yaw) + abs(z_pitch)))
        self._latent_state.attentiveness = clamp(
            STATE_SMOOTH_ALPHA * self._latent_state.attentiveness +
            (1.0 - STATE_SMOOTH_ALPHA) * attentiveness_evidence,
            0.0, 1.0
        )

        output = AnalysisOutput(

            posture_deviation_scores={
                "z_yaw": z_yaw,
                "z_pitch": z_pitch,
                "z_roll": z_roll,
                "z_shoulder": z_shoulder,
                "z_torso": z_torso
            },

            latent_state_vector={
                "engagement": self._latent_state.engagement,
                "confidence": self._latent_state.confidence,
                "nervousness": self._latent_state.nervousness,
                "attentiveness": self._latent_state.attentiveness
            },

            stability_indices={
                "variance_yaw": variance_yaw,
                "variance_pitch": variance_pitch,
                "variance_roll": variance_roll,
                "variance_shoulder": variance_shoulder
            },

            evidence_scores={
                "slouch_score": slouch_score,
                "nervous_score": nervous_score,
                "engagement_score": engagement_score
            }
        )

        return output

    def reset(self) -> None:

        self._baseline = BaselineState()

        self._variance = VarianceState()

        self._latent_state = LatentState()

    def get_current_state(self) -> Dict[str, Any]:
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
        return (self._frame_width_px, self._frame_height_px)

    @property
    def focal_length_px(self) -> float:
        return self._focal_length_px

    @property
    def scale_mm_per_px(self) -> Optional[float]:
        if self._baseline.frame_count == 0:
            return None

        return None

def create_analyzer(
    frame_width_px: int,
    frame_height_px: int,
    camera_fov_deg: float = CAMERA_FOV_DEG
) -> DeterministicNonVerbalAnalyzer:
    return DeterministicNonVerbalAnalyzer(
        frame_width_px=frame_width_px,
        frame_height_px=frame_height_px,
        camera_fov_deg=camera_fov_deg
    )
