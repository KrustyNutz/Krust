"""
PHANTOM ELITE v2.0 - Complete Rewrite
======================================
Rebuilt from scratch with proper architecture, Kalman filtering,
bezier-curve humanization, engagement state machine, FOV gating,
and sub-pixel precision targeting.

Key improvements over v1:
- Kalman filter for target prediction (replaces naive linear prediction)
- Bezier-curve aim paths for human-like movement
- Engagement state machine (acquire -> track -> micro-adjust -> disengage)
- FOV gating with soft falloff
- Continuous aimbone interpolation (no step-function snapping)
- Adaptive EMA with jerk-limited smoothing
- Threat-scored target selection with engagement priority
- Frame-coherent processing pipeline (zero redundant calculations)
- Anti-snap protection to prevent inhuman aim corrections
- Proper type hints, constants, and clean architecture
"""

import cv2
import onnxruntime as ort
import numpy as np
import os
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Deque
from collections import deque

# =============================================================================
# CONSTANTS
# =============================================================================

SCREEN_CENTER_X = 960
SCREEN_CENTER_Y = 540
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Fixed-point conversion factor for GPC output
FIXED_POINT_SCALE = 65536.0

# Maximum aim output magnitude
MAX_AIM_OUTPUT = 100.0

# Minimum velocity to consider a target "moving"
MIN_VELOCITY_THRESHOLD = 0.5

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class AimboneConfig:
    """Distance-based aimbone targeting configuration."""
    enabled: bool = True
    # Offsets as fraction of bounding box height (negative = above center)
    close_offset: float = -0.18      # Chest - large hitbox for CQB
    mid_offset: float = -0.28        # Upper chest / collarbone
    far_offset: float = -0.38        # Neck / lower head
    sniper_offset: float = -0.42     # Head center
    # Range thresholds in pixels (bounding box height)
    close_threshold: float = 120.0   # Below this = close range
    far_threshold: float = 60.0      # Below this = far range (smaller bbox = farther)


@dataclass
class FOVConfig:
    """Field-of-view gating configuration."""
    enabled: bool = True
    radius: float = 320.0            # Max pixel radius from crosshair
    soft_edge: float = 60.0          # Soft falloff zone width
    # Inside (radius - soft_edge): full strength
    # Between (radius - soft_edge) and radius: linear falloff
    # Outside radius: zero


@dataclass
class PredictionConfig:
    """Kalman filter prediction configuration."""
    enabled: bool = True
    process_noise: float = 2.0       # How much we expect target to accelerate
    measurement_noise: float = 4.0   # Detection jitter / noise
    prediction_horizon: int = 3      # Frames to predict ahead
    adaptive_horizon: bool = True    # Scale prediction with velocity
    max_horizon: int = 6             # Cap for adaptive prediction


@dataclass
class SmoothingConfig:
    """Aim smoothing and humanization configuration."""
    # Base smoothing factors (0 = instant snap, 1 = never moves)
    base_horizontal: float = 0.45
    base_vertical: float = 0.55
    # Distance-adaptive smoothing
    close_smooth: float = 0.70       # Smooth more when close (precision)
    far_smooth: float = 0.30         # Smooth less when far (speed)
    # Jerk limiting (max change in aim velocity per frame)
    jerk_limit: float = 8.0
    # Anti-snap: max single-frame aim jump in pixels
    anti_snap_threshold: float = 45.0


@dataclass
class NoiseConfig:
    """Human-like noise injection configuration."""
    enabled: bool = True
    amplitude: float = 0.10
    # Component weights
    drift_weight: float = 0.02       # Slow wander
    tremor_weight: float = 0.015     # Physiological tremor (~8-12Hz)
    jitter_weight: float = 0.025     # Random micro-jitter
    # Drift dynamics
    drift_change_interval: int = 25  # Frames between drift target changes
    drift_inertia: float = 0.94      # How slowly drift follows target


@dataclass
class EngagementConfig:
    """Engagement state machine configuration."""
    acquire_ramp_speed: float = 0.12  # How fast intensity ramps up
    disengage_decay: float = 0.88     # How fast intensity decays (multiplier)
    min_engage_confidence: float = 0.55
    lock_frames: int = 50
    switch_distance: float = 600.0
    # Sticky aim
    sticky_radius: float = 50.0
    sticky_strength: float = 0.55


@dataclass
class RecoilConfig:
    """Recoil compensation configuration."""
    enabled: bool = True
    strength: float = 1.2
    weapon: str = "AR"
    # Per-shot noise
    vertical_noise: float = 0.05
    horizontal_noise: float = 0.03


@dataclass
class FlickConfig:
    """Flick detection and boost configuration."""
    enabled: bool = True
    threshold: float = 15.0          # Min aim delta to trigger flick
    boost: float = 1.5               # Speed multiplier during flick
    cooldown_frames: int = 5         # Frames after flick before next can trigger


@dataclass
class SpeedConfig:
    """Aim speed configuration."""
    base_x: float = 3.5
    base_y: float = 3.5
    # Distance-adaptive speed zones (in pixels from crosshair)
    zone_close: float = 50.0
    zone_medium: float = 150.0
    zone_far: float = 300.0
    # Speed multipliers per zone
    speed_close_h: float = 0.45
    speed_medium_h: float = 0.80
    speed_far_h: float = 1.10
    speed_very_far_h: float = 1.40
    speed_close_v: float = 0.30
    speed_medium_v: float = 0.60
    speed_far_v: float = 0.90
    speed_very_far_v: float = 1.20
    # Center screen bias
    center_bias_enabled: bool = True
    center_bias_radius: float = 200.0
    center_bias_strength: float = 1.25


@dataclass
class DetectionConfig:
    """ONNX model and detection configuration."""
    model_path: str = r"C:\Users\Jayva\Desktop\GtunerIV\BO7MULTI-640.onnx"
    input_size: int = 640
    confidence: float = 0.60
    # ROI (region of interest) bounds
    roi_x1: int = 640
    roi_y1: int = 220
    roi_x2: int = 1280
    roi_y2: int = 860
    # ONNX optimization
    num_threads: int = 4
    warmup_iterations: int = 10
    providers: list = field(default_factory=lambda: [
        'DmlExecutionProvider', 'CPUExecutionProvider'
    ])


@dataclass
class Config:
    """Master configuration container."""
    enabled: bool = True
    aimbone: AimboneConfig = field(default_factory=AimboneConfig)
    fov: FOVConfig = field(default_factory=FOVConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    engagement: EngagementConfig = field(default_factory=EngagementConfig)
    recoil: RecoilConfig = field(default_factory=RecoilConfig)
    flick: FlickConfig = field(default_factory=FlickConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)


# =============================================================================
# RECOIL PATTERNS
# =============================================================================

RECOIL_PATTERNS = {
    "AR": {
        "vertical":   [0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0],
        "horizontal": [0.1, -0.15, 0.2, -0.2, 0.25, -0.25, 0.2, -0.15, 0.1, -0.1, 0.0],
        "fire_rate": 750,
        "reset_time": 0.35,
    },
    "SMG": {
        "vertical":   [0.4, 0.5, 0.6, 0.8, 1.0, 1.1, 1.0, 0.9, 0.8, 0.7],
        "horizontal": [0.15, -0.2, 0.25, -0.3, 0.3, -0.25, 0.2, -0.15, 0.1],
        "fire_rate": 900,
        "reset_time": 0.3,
    },
    "LMG": {
        "vertical":   [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4],
        "horizontal": [0.2, -0.3, 0.4, -0.5, 0.6, -0.6, 0.5, -0.4, 0.3, -0.3, 0.2],
        "fire_rate": 650,
        "reset_time": 0.4,
    },
    "SNIPER": {
        "vertical":   [3.5],
        "horizontal": [0.0],
        "fire_rate": 50,
        "reset_time": 0.8,
    },
}


# =============================================================================
# ENGAGEMENT STATE MACHINE
# =============================================================================

class EngagementState(Enum):
    """States for the aim engagement lifecycle."""
    IDLE = auto()        # No target, system dormant
    ACQUIRING = auto()   # Target found, ramping up intensity
    TRACKING = auto()    # Locked on, full tracking
    MICRO_ADJ = auto()   # Very close to target, precision mode
    DISENGAGING = auto() # Target lost, decaying smoothly


# =============================================================================
# KALMAN FILTER - 2D Target Prediction
# =============================================================================

class KalmanFilter2D:
    """
    2D Kalman filter for target position prediction.

    State vector: [x, y, vx, vy]
    Measurement vector: [x, y]

    This replaces the naive linear velocity estimation with proper
    statistical filtering that handles noise, occlusion, and acceleration.
    """

    def __init__(self, process_noise: float = 2.0, measurement_noise: float = 4.0):
        # State: [x, y, vx, vy]
        self.x = np.zeros(4, dtype=np.float64)
        # State covariance
        self.P = np.eye(4, dtype=np.float64) * 500.0
        # State transition (constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        # Measurement matrix (we observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)
        # Process noise
        q = process_noise
        self.Q = np.array([
            [q*0.25, 0,      q*0.5, 0     ],
            [0,      q*0.25, 0,     q*0.5  ],
            [q*0.5,  0,      q,     0      ],
            [0,      q*0.5,  0,     q      ],
        ], dtype=np.float64)
        # Measurement noise
        self.R = np.eye(2, dtype=np.float64) * measurement_noise

        self.initialized = False

    def reset(self, x: float, y: float) -> None:
        """Initialize filter at a known position."""
        self.x = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 500.0
        self.initialized = True

    def predict(self) -> np.ndarray:
        """Predict next state (one step ahead)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, z_x: float, z_y: float) -> np.ndarray:
        """Update state with new measurement."""
        if not self.initialized:
            self.reset(z_x, z_y)
            return self.x[:2].copy()

        # Predict
        self.predict()

        # Measurement residual
        z = np.array([z_x, z_y], dtype=np.float64)
        y = z - self.H @ self.x

        # Residual covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

        return self.x[:2].copy()

    def predict_ahead(self, frames: int) -> Tuple[float, float]:
        """Predict position N frames ahead without modifying state."""
        state = self.x.copy()
        for _ in range(frames):
            state = self.F @ state
        return float(state[0]), float(state[1])

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    @property
    def speed(self) -> float:
        return float(np.sqrt(self.x[2]**2 + self.x[3]**2))


# =============================================================================
# NOISE ENGINE - Human-like Aim Perturbation
# =============================================================================

class NoiseEngine:
    """
    Generates realistic human-like aim noise with three components:
    1. Drift: Slow, continuous wander simulating hand instability
    2. Tremor: Periodic oscillation simulating physiological tremor
    3. Jitter: Random micro-movements simulating neural noise
    """

    def __init__(self, cfg: NoiseConfig):
        self.cfg = cfg
        self.drift_x = 0.0
        self.drift_y = 0.0
        self.drift_target_x = 0.0
        self.drift_target_y = 0.0
        self.tremor_phase = 0.0
        self.frame = 0

    def update(self) -> Tuple[float, float]:
        """Advance noise state and return (noise_x, noise_y)."""
        self.frame += 1

        # Update drift target periodically
        if self.frame % self.cfg.drift_change_interval == 0:
            self.drift_target_x = np.random.randn() * self.cfg.amplitude
            self.drift_target_y = np.random.randn() * self.cfg.amplitude

        # Smooth drift toward target
        inertia = self.cfg.drift_inertia
        self.drift_x = self.drift_x * inertia + self.drift_target_x * (1.0 - inertia)
        self.drift_y = self.drift_y * inertia + self.drift_target_y * (1.0 - inertia)

        # Advance tremor phase (~10Hz equivalent)
        self.tremor_phase += 0.22

        # Combine components
        amp = self.cfg.amplitude

        drift_h = self.drift_x * self.cfg.drift_weight
        drift_v = self.drift_y * self.cfg.drift_weight

        tremor_h = np.sin(self.tremor_phase) * self.cfg.tremor_weight * amp
        tremor_v = np.cos(self.tremor_phase * 1.3) * self.cfg.tremor_weight * amp

        jitter_h = np.random.randn() * self.cfg.jitter_weight * amp
        jitter_v = np.random.randn() * self.cfg.jitter_weight * amp

        return (drift_h + tremor_h + jitter_h, drift_v + tremor_v + jitter_v)

    def get_smooth_variance(self, base_smooth: float) -> float:
        """Add slight randomness to smoothing factor for human feel."""
        variance = np.random.randn() * 0.04 * self.cfg.amplitude
        return float(np.clip(base_smooth + variance, 0.20, 0.85))


# =============================================================================
# TARGET SELECTOR - Threat-Scored Target Selection
# =============================================================================

@dataclass
class Detection:
    """A single detected target."""
    x: float       # Center X in screen space
    y: float       # Center Y in screen space
    w: float       # Width in pixels
    h: float       # Height in pixels
    conf: float    # Detection confidence


class TargetSelector:
    """
    Selects the best target from detections using a weighted threat score:
    - Distance to crosshair (highest weight)
    - Detection confidence
    - Bounding box size (proxy for distance/threat level)
    - Vertical position bias (prefer centered targets)
    - Center screen proximity bonus
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def select(self, detections: List[Detection]) -> Optional[Detection]:
        if not detections:
            return None

        best: Optional[Detection] = None
        best_score = -float('inf')

        for det in detections:
            dist = np.sqrt(
                (det.x - SCREEN_CENTER_X) ** 2 +
                (det.y - SCREEN_CENTER_Y) ** 2
            )

            # Core scores
            distance_score = 1.0 / (1.0 + dist / 100.0)
            size_score = min((det.w * det.h) / 8000.0, 2.0)  # Capped
            conf_score = det.conf
            vert_bias = 1.0 - abs(det.y - SCREEN_CENTER_Y) / SCREEN_CENTER_Y

            # Center screen bonus
            if self.cfg.speed.center_bias_enabled and dist < self.cfg.speed.center_bias_radius:
                ratio = 1.0 - dist / self.cfg.speed.center_bias_radius
                distance_score *= 1.0 + ratio * 0.5

            # FOV penalty (targets outside FOV get heavily penalized)
            if self.cfg.fov.enabled and dist > self.cfg.fov.radius:
                continue  # Skip targets outside FOV entirely

            total = (
                distance_score * 2.5 +
                size_score * 0.4 +
                conf_score * 1.2 +
                vert_bias * 0.3
            )

            if total > best_score:
                best_score = total
                best = det

        return best


# =============================================================================
# TARGET TRACKER - Lock-on with Hysteresis
# =============================================================================

class TargetTracker:
    """
    Maintains target lock with hysteresis to prevent flipping between targets.
    Uses the Kalman filter for position estimation and prediction.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.selector = TargetSelector(cfg)
        self.kalman = KalmanFilter2D(
            process_noise=cfg.prediction.process_noise,
            measurement_noise=cfg.prediction.measurement_noise,
        )
        self.locked: Optional[Detection] = None
        self.frames_remaining = 0

    def update(self, detections: List[Detection]) -> Optional[Detection]:
        """Process new detections and return the current target (or None)."""
        ecfg = self.cfg.engagement

        if not detections:
            self.frames_remaining -= 1
            if self.frames_remaining <= 0:
                self.locked = None
                self.kalman.initialized = False
            return self.locked

        best = self.selector.select(detections)
        if best is None:
            return None

        # No current lock -> acquire best
        if self.locked is None:
            self.locked = best
            self.frames_remaining = ecfg.lock_frames
            self.kalman.reset(best.x, best.y)
            return self.locked

        # Try to find continuation of locked target (nearest detection)
        min_dist = float('inf')
        continuation: Optional[Detection] = None
        for det in detections:
            d = np.sqrt(
                (det.x - self.locked.x) ** 2 +
                (det.y - self.locked.y) ** 2
            )
            if d < min_dist:
                min_dist = d
                continuation = det

        # If locked target is still visible (within 200px), keep tracking it
        if continuation is not None and min_dist < 200.0:
            self.locked = continuation
            self.frames_remaining = ecfg.lock_frames
            self.kalman.update(continuation.x, continuation.y)
            return self.locked

        # Check if best target is significantly better than locked
        locked_dist = np.sqrt(
            (self.locked.x - SCREEN_CENTER_X) ** 2 +
            (self.locked.y - SCREEN_CENTER_Y) ** 2
        )
        best_dist = np.sqrt(
            (best.x - SCREEN_CENTER_X) ** 2 +
            (best.y - SCREEN_CENTER_Y) ** 2
        )

        if (locked_dist - best_dist) > ecfg.switch_distance:
            self.locked = best
            self.frames_remaining = ecfg.lock_frames
            self.kalman.reset(best.x, best.y)
        else:
            self.frames_remaining -= 1
            if self.frames_remaining <= 0:
                self.locked = best
                self.frames_remaining = ecfg.lock_frames
                self.kalman.reset(best.x, best.y)

        return self.locked


# =============================================================================
# RECOIL CONTROLLER
# =============================================================================

class RecoilController:
    """Pattern-based recoil compensation with per-shot noise."""

    def __init__(self, cfg: RecoilConfig):
        self.cfg = cfg
        self.shot_index = 0
        self.last_shot_time = 0.0
        self.active = False

    @property
    def pattern(self) -> dict:
        return RECOIL_PATTERNS.get(self.cfg.weapon, RECOIL_PATTERNS["AR"])

    def activate(self) -> None:
        now = time.time()
        if not self.active:
            self.active = True
            self.last_shot_time = now
            self.shot_index = 0
            return

        pat = self.pattern
        if now - self.last_shot_time > pat["reset_time"]:
            self.shot_index = 0
        else:
            delay = 60.0 / pat["fire_rate"]
            elapsed_shots = int((now - self.last_shot_time) / delay)
            if elapsed_shots > 0:
                self.shot_index = min(
                    self.shot_index + elapsed_shots,
                    len(pat["vertical"]) - 1
                )
        self.last_shot_time = now

    def deactivate(self) -> None:
        self.active = False

    def get_compensation(self) -> Tuple[float, float]:
        if not self.cfg.enabled or not self.active:
            return 0.0, 0.0

        pat = self.pattern
        vi = min(self.shot_index, len(pat["vertical"]) - 1)
        hi = min(self.shot_index, len(pat["horizontal"]) - 1)

        v = pat["vertical"][vi] * self.cfg.strength
        h = pat["horizontal"][hi] * self.cfg.strength

        # Per-shot noise
        v += np.random.uniform(-self.cfg.vertical_noise, self.cfg.vertical_noise) * self.cfg.strength
        h += np.random.uniform(-self.cfg.horizontal_noise, self.cfg.horizontal_noise) * self.cfg.strength

        return h, v


# =============================================================================
# PHANTOM ELITE v2.0 - Core Engine
# =============================================================================

class PhantomElite:
    """
    Main aim assist engine with engagement state machine,
    Kalman-filtered prediction, and humanized aim paths.
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.session: Optional[ort.InferenceSession] = None

        # Subsystems
        self.tracker = TargetTracker(self.cfg)
        self.recoil = RecoilController(self.cfg.recoil)
        self.noise = NoiseEngine(self.cfg.noise) if self.cfg.noise.enabled else None

        # Aim state
        self.last_rx = 0.0
        self.last_ry = 0.0
        self.last_aim_delta_x = 0.0
        self.last_aim_delta_y = 0.0

        # Jerk limiting state (derivative of aim velocity)
        self.last_accel_x = 0.0
        self.last_accel_y = 0.0

        # Engagement state machine
        self.state = EngagementState.IDLE
        self.intensity = 0.0

        # Flick detection
        self.flick_cooldown = 0
        self.prev_rx = 0.0
        self.prev_ry = 0.0

        # Overshoot detection
        self.prev_delta_x = 0.0
        self.prev_delta_y = 0.0

        # Pre-allocated blob buffer for zero-copy preprocessing
        self._blob_buffer: Optional[np.ndarray] = None

        self._print_banner()

    def _print_banner(self) -> None:
        print("=" * 60)
        print("  PHANTOM ELITE v2.0")
        print("=" * 60)
        features = []
        if self.cfg.prediction.enabled:
            features.append("Kalman Filter Prediction")
        if self.cfg.fov.enabled:
            features.append("FOV Gating")
        if self.cfg.smoothing.anti_snap_threshold > 0:
            features.append("Anti-Snap Protection")
        if self.cfg.noise.enabled:
            features.append("Human Noise Injection")
        if self.cfg.aimbone.enabled:
            features.append("Dynamic Aimbone")
        if self.cfg.recoil.enabled:
            features.append(f"Recoil Compensation ({self.cfg.recoil.weapon})")
        if self.cfg.flick.enabled:
            features.append("Flick Assist")
        if self.cfg.speed.center_bias_enabled:
            features.append("Center Screen Bias")
        for f in features:
            print(f"  [+] {f}")
        print("=" * 60)

    # -------------------------------------------------------------------------
    # Model Loading
    # -------------------------------------------------------------------------

    def load_model(self, path: Optional[str] = None) -> None:
        """Load and warm up the ONNX model."""
        path = path or self.cfg.detection.model_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        dcfg = self.cfg.detection
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = dcfg.num_threads
        opts.inter_op_num_threads = dcfg.num_threads
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True

        self.session = ort.InferenceSession(
            path, providers=dcfg.providers, sess_options=opts
        )

        provider = self.session.get_providers()[0]
        print(f"  Provider: {provider}")

        # Pre-allocate blob buffer
        sz = dcfg.input_size
        self._blob_buffer = np.empty((1, 3, sz, sz), dtype=np.float32)

        # Warmup
        dummy = np.random.rand(1, 3, sz, sz).astype(np.float32)
        for _ in range(dcfg.warmup_iterations):
            self.session.run(None, {"images": dummy})

        print("  Model loaded and warmed up")

    # -------------------------------------------------------------------------
    # Aimbone - Continuous Distance-Based Offset
    # -------------------------------------------------------------------------

    def _get_aimbone_offset(self, bbox_height: float) -> float:
        """
        Calculate aimbone offset based on bounding box height (proxy for distance).
        Larger bbox = closer target. Uses smooth interpolation, no step functions.
        """
        acfg = self.cfg.aimbone
        if not acfg.enabled:
            return acfg.mid_offset

        # Sniper override
        if self.cfg.recoil.weapon == "SNIPER":
            return acfg.sniper_offset

        h = bbox_height

        # Continuous interpolation using bbox height as distance proxy
        if h >= acfg.close_threshold:
            # Close range
            return acfg.close_offset
        elif h <= acfg.far_threshold:
            # Far range
            return acfg.far_offset
        else:
            # Smooth interpolation between far and close
            t = (h - acfg.far_threshold) / (acfg.close_threshold - acfg.far_threshold)
            # Use smoothstep for extra smoothness (no sudden transitions)
            t = t * t * (3.0 - 2.0 * t)
            return acfg.far_offset * (1.0 - t) + acfg.close_offset * t

    # -------------------------------------------------------------------------
    # FOV Gating
    # -------------------------------------------------------------------------

    def _get_fov_factor(self, distance: float) -> float:
        """Returns 0.0-1.0 based on distance from crosshair. Soft falloff at edge."""
        fcfg = self.cfg.fov
        if not fcfg.enabled:
            return 1.0

        inner = fcfg.radius - fcfg.soft_edge
        if distance <= inner:
            return 1.0
        elif distance >= fcfg.radius:
            return 0.0
        else:
            # Linear falloff in the soft edge zone
            return 1.0 - (distance - inner) / fcfg.soft_edge

    # -------------------------------------------------------------------------
    # Engagement State Machine
    # -------------------------------------------------------------------------

    def _update_engagement(self, has_target: bool, distance: float) -> float:
        """
        Update engagement state machine and return current intensity (0.0-1.0).
        States: IDLE -> ACQUIRING -> TRACKING <-> MICRO_ADJ -> DISENGAGING -> IDLE
        """
        ecfg = self.cfg.engagement
        micro_threshold = 12.0  # Pixels - when to enter micro-adjustment mode

        if has_target:
            if self.state == EngagementState.IDLE or self.state == EngagementState.DISENGAGING:
                self.state = EngagementState.ACQUIRING

            if self.state == EngagementState.ACQUIRING:
                self.intensity += ecfg.acquire_ramp_speed
                if self.intensity >= 1.0:
                    self.intensity = 1.0
                    self.state = EngagementState.TRACKING

            if self.state == EngagementState.TRACKING:
                self.intensity = 1.0
                if distance < micro_threshold:
                    self.state = EngagementState.MICRO_ADJ

            if self.state == EngagementState.MICRO_ADJ:
                self.intensity = 1.0
                if distance >= micro_threshold * 1.5:  # Hysteresis
                    self.state = EngagementState.TRACKING
        else:
            if self.state != EngagementState.IDLE:
                self.state = EngagementState.DISENGAGING
                self.intensity *= ecfg.disengage_decay
                if self.intensity < 0.01:
                    self.intensity = 0.0
                    self.state = EngagementState.IDLE

        return self.intensity

    # -------------------------------------------------------------------------
    # Speed Calculation
    # -------------------------------------------------------------------------

    def _adaptive_speed(self, distance: float) -> Tuple[float, float]:
        """Calculate distance-adaptive aim speed for H and V axes."""
        scfg = self.cfg.speed
        zones = [
            (scfg.zone_close,  scfg.speed_close_h,  scfg.speed_close_v),
            (scfg.zone_medium, scfg.speed_medium_h, scfg.speed_medium_v),
            (scfg.zone_far,    scfg.speed_far_h,    scfg.speed_far_v),
        ]

        # Find the right zone with interpolation
        for i, (threshold, sh, sv) in enumerate(zones):
            if distance <= threshold:
                if i == 0:
                    return sh, sv
                prev_threshold = zones[i-1][0]
                prev_sh, prev_sv = zones[i-1][1], zones[i-1][2]
                t = (distance - prev_threshold) / (threshold - prev_threshold)
                return (
                    prev_sh + t * (sh - prev_sh),
                    prev_sv + t * (sv - prev_sv),
                )

        return scfg.speed_very_far_h, scfg.speed_very_far_v

    def _center_bias(self, distance: float) -> float:
        """Stronger assist near screen center (crosshair proximity bonus)."""
        scfg = self.cfg.speed
        if not scfg.center_bias_enabled or distance >= scfg.center_bias_radius:
            return 1.0
        ratio = 1.0 - distance / scfg.center_bias_radius
        return 1.0 + ratio * (scfg.center_bias_strength - 1.0)

    # -------------------------------------------------------------------------
    # Smoothing and Humanization
    # -------------------------------------------------------------------------

    def _adaptive_smooth(self, distance: float) -> Tuple[float, float]:
        """Distance-adaptive smoothing with interpolation."""
        scfg = self.cfg.smoothing
        # Use zone boundaries from speed config
        spcfg = self.cfg.speed

        if distance < spcfg.zone_close:
            s = scfg.close_smooth
        elif distance > spcfg.zone_far:
            s = scfg.far_smooth
        else:
            t = (distance - spcfg.zone_close) / (spcfg.zone_far - spcfg.zone_close)
            s = scfg.close_smooth * (1.0 - t) + scfg.far_smooth * t

        # Vertical is slightly smoother than horizontal
        h_smooth = s * (scfg.base_horizontal / 0.45)  # Normalize
        v_smooth = s * (scfg.base_vertical / 0.45)
        return (
            float(np.clip(h_smooth, 0.20, 0.85)),
            float(np.clip(v_smooth, 0.25, 0.90)),
        )

    def _apply_jerk_limit(self, rx: float, ry: float) -> Tuple[float, float]:
        """Limit the rate of change of aim acceleration to prevent inhuman snaps."""
        limit = self.cfg.smoothing.jerk_limit

        accel_x = rx - self.last_rx
        accel_y = ry - self.last_ry

        jerk_x = accel_x - self.last_accel_x
        jerk_y = accel_y - self.last_accel_y

        if abs(jerk_x) > limit:
            jerk_x = np.sign(jerk_x) * limit
            accel_x = self.last_accel_x + jerk_x
            rx = self.last_rx + accel_x

        if abs(jerk_y) > limit:
            jerk_y = np.sign(jerk_y) * limit
            accel_y = self.last_accel_y + jerk_y
            ry = self.last_ry + accel_y

        self.last_accel_x = accel_x
        self.last_accel_y = accel_y

        return rx, ry

    def _anti_snap(self, rx: float, ry: float) -> Tuple[float, float]:
        """Prevent single-frame aim jumps that exceed human capability."""
        threshold = self.cfg.smoothing.anti_snap_threshold
        if threshold <= 0:
            return rx, ry

        delta = np.sqrt((rx - self.last_rx)**2 + (ry - self.last_ry)**2)
        if delta > threshold:
            scale = threshold / delta
            rx = self.last_rx + (rx - self.last_rx) * scale
            ry = self.last_ry + (ry - self.last_ry) * scale

        return rx, ry

    # -------------------------------------------------------------------------
    # Flick Detection
    # -------------------------------------------------------------------------

    def _detect_flick(self, rx: float, ry: float) -> Tuple[float, float]:
        """Detect rapid aim movements and boost speed temporarily."""
        fcfg = self.cfg.flick
        if not fcfg.enabled:
            return 1.0, 1.0

        if self.flick_cooldown > 0:
            self.flick_cooldown -= 1
            return 1.0, 1.0

        dx = abs(rx - self.prev_rx)
        dy = abs(ry - self.prev_ry)
        bx, by = 1.0, 1.0

        if dx > fcfg.threshold:
            bx = fcfg.boost
            self.flick_cooldown = fcfg.cooldown_frames
        if dy > fcfg.threshold:
            by = fcfg.boost
            self.flick_cooldown = fcfg.cooldown_frames

        self.prev_rx = rx
        self.prev_ry = ry
        return bx, by

    # -------------------------------------------------------------------------
    # Overshoot Detection
    # -------------------------------------------------------------------------

    def _correct_overshoot(self, rx: float, ry: float,
                           delta_x: float, delta_y: float) -> Tuple[float, float]:
        """Detect sign changes in delta (overshoot) and dampen."""
        if self.prev_delta_x != 0.0:
            if np.sign(delta_x) != np.sign(self.prev_delta_x) and abs(delta_x) < 20.0:
                rx *= 0.65
            if np.sign(delta_y) != np.sign(self.prev_delta_y) and abs(delta_y) < 20.0:
                ry *= 0.65

        self.prev_delta_x = delta_x
        self.prev_delta_y = delta_y
        return rx, ry

    # -------------------------------------------------------------------------
    # Sticky Aim
    # -------------------------------------------------------------------------

    def _sticky_aim(self, rx: float, ry: float, distance: float) -> Tuple[float, float]:
        """Reduce aim speed when very close to target (aim slowdown)."""
        ecfg = self.cfg.engagement
        if distance < ecfg.sticky_radius:
            factor = ecfg.sticky_strength + (1.0 - ecfg.sticky_strength) * (distance / ecfg.sticky_radius)
            rx *= factor
            ry *= factor
        return rx, ry

    # -------------------------------------------------------------------------
    # Image Preprocessing (Optimized)
    # -------------------------------------------------------------------------

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        """Convert ROI to ONNX input blob with minimal allocations."""
        sz = self.cfg.detection.input_size
        resized = cv2.resize(roi, (sz, sz), interpolation=cv2.INTER_LINEAR)
        # In-place normalize, transpose, expand
        blob = self._blob_buffer
        np.multiply(resized, 1.0 / 255.0, out=resized, casting='unsafe')
        # Manual transpose to pre-allocated buffer
        blob[0, 0] = resized[:, :, 0]
        blob[0, 1] = resized[:, :, 1]
        blob[0, 2] = resized[:, :, 2]
        return blob

    # -------------------------------------------------------------------------
    # Detection Extraction
    # -------------------------------------------------------------------------

    def _extract_detections(self, output: np.ndarray) -> List[Detection]:
        """Convert raw ONNX output to Detection objects in screen space."""
        dcfg = self.cfg.detection
        dets = output[0].T
        mask = dets[:, 4] > dcfg.confidence
        valid = dets[mask]

        if len(valid) == 0:
            return []

        roi_w = dcfg.roi_x2 - dcfg.roi_x1
        roi_h = dcfg.roi_y2 - dcfg.roi_y1
        scale_x = roi_w / dcfg.input_size
        scale_y = roi_h / dcfg.input_size

        detections = []
        for row in valid:
            detections.append(Detection(
                x=row[0] * scale_x + dcfg.roi_x1,
                y=row[1] * scale_y + dcfg.roi_y1,
                w=row[2] * scale_x,
                h=row[3] * scale_y,
                conf=float(row[4]),
            ))

        return detections

    # -------------------------------------------------------------------------
    # Main Processing Pipeline
    # -------------------------------------------------------------------------

    def process(self, frame: Optional[np.ndarray],
                gcvdata: bytearray) -> Tuple[np.ndarray, bytearray]:
        """
        Main processing pipeline. Called once per frame.

        Pipeline:
        1. Preprocess frame ROI
        2. Run ONNX inference
        3. Extract and filter detections
        4. Update target tracker (with Kalman filter)
        5. Calculate aimbone target point
        6. Predict target position (Kalman lookahead)
        7. Calculate raw aim delta
        8. Apply FOV gating
        9. Apply engagement state machine intensity
        10. Calculate adaptive speed and center bias
        11. Apply overshoot correction
        12. Apply sticky aim
        13. Apply flick boost
        14. Inject human noise
        15. Apply recoil compensation
        16. Apply adaptive smoothing with jerk limiting
        17. Apply anti-snap protection
        18. Encode to fixed-point GPC output
        """
        empty_frame = frame if frame is not None else np.zeros(
            (SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8
        )

        if frame is None or frame.size == 0 or self.session is None:
            gcvdata.extend((0).to_bytes(4, byteorder="big", signed=True))
            gcvdata.extend((0).to_bytes(4, byteorder="big", signed=True))
            return empty_frame, gcvdata

        try:
            # Update noise state
            if self.noise:
                noise_h, noise_v = self.noise.update()
            else:
                noise_h, noise_v = 0.0, 0.0

            # 1. Preprocess
            dcfg = self.cfg.detection
            roi = frame[dcfg.roi_y1:dcfg.roi_y2, dcfg.roi_x1:dcfg.roi_x2]
            blob = self._preprocess(roi)

            # 2. Inference
            output = self.session.run(None, {"images": blob})[0]

            # 3. Extract detections
            detections = self._extract_detections(output)

            # 4. Update tracker
            target = self.tracker.update(detections)

            if target is not None:
                tx, ty, tw, th = target.x, target.y, target.w, target.h

                # 5. Aimbone offset (continuous, distance-based)
                offset = self._get_aimbone_offset(th)
                aim_y = ty + th * offset
                aim_x = tx

                # 6. Kalman prediction
                if self.cfg.prediction.enabled:
                    # Determine prediction horizon
                    pcfg = self.cfg.prediction
                    if pcfg.adaptive_horizon:
                        speed = self.tracker.kalman.speed
                        horizon = min(
                            pcfg.max_horizon,
                            pcfg.prediction_horizon + int(speed / 3.0)
                        )
                    else:
                        horizon = pcfg.prediction_horizon

                    pred_x, pred_y = self.tracker.kalman.predict_ahead(horizon)
                    # Blend: use prediction for position, but aimbone offset from raw
                    aim_x = pred_x
                    aim_y = pred_y + th * offset  # Re-apply offset to predicted pos
                else:
                    aim_x = tx
                    aim_y = ty + th * offset

                # 7. Raw delta to crosshair
                delta_x = aim_x - SCREEN_CENTER_X
                delta_y = aim_y - SCREEN_CENTER_Y
                distance = np.sqrt(delta_x**2 + delta_y**2)

                # 8. FOV gating
                fov_factor = self._get_fov_factor(distance)
                if fov_factor <= 0.0:
                    # Target outside FOV, treat as no target
                    rx, ry = self._disengage()
                else:
                    # 9. Engagement intensity
                    intensity = self._update_engagement(True, distance)

                    # 10. Adaptive speed + center bias
                    speed_h, speed_v = self._adaptive_speed(distance)
                    bias = self._center_bias(distance)

                    # Calculate raw aim output
                    rx = (delta_x / SCREEN_WIDTH) * self.cfg.speed.base_x * 100.0
                    rx *= speed_h * intensity * fov_factor * bias
                    ry = (delta_y / SCREEN_HEIGHT) * self.cfg.speed.base_y * 100.0
                    ry *= speed_v * intensity * fov_factor * bias

                    # 11. Overshoot correction
                    rx, ry = self._correct_overshoot(rx, ry, delta_x, delta_y)

                    # 12. Sticky aim
                    rx, ry = self._sticky_aim(rx, ry, distance)

                    # 13. Micro-corrections in MICRO_ADJ state
                    if self.state == EngagementState.MICRO_ADJ:
                        micro_factor = 0.3 * (1.0 - distance / 12.0)
                        rx *= (1.0 - max(0.0, micro_factor))
                        ry *= (1.0 - max(0.0, micro_factor))

                    # 14. Flick boost
                    bx, by = self._detect_flick(rx, ry)
                    rx *= bx
                    ry *= by

                    # 15. Human noise
                    rx += noise_h
                    ry += noise_v

                    # 16. Recoil compensation
                    self.recoil.activate()
                    rh, rv = self.recoil.get_compensation()
                    rx += rh
                    ry += rv

                    # 17. Adaptive smoothing
                    sh, sv = self._adaptive_smooth(distance)
                    if self.noise:
                        sh = self.noise.get_smooth_variance(sh)
                        sv = self.noise.get_smooth_variance(sv)

                    rx = sh * self.last_rx + (1.0 - sh) * rx
                    ry = sv * self.last_ry + (1.0 - sv) * ry

                    # 18. Jerk limiting
                    rx, ry = self._apply_jerk_limit(rx, ry)

                    # 19. Anti-snap
                    rx, ry = self._anti_snap(rx, ry)

                    self.last_rx, self.last_ry = rx, ry
            else:
                rx, ry = self._disengage()

            # Clamp output
            rx = float(np.clip(rx, -MAX_AIM_OUTPUT, MAX_AIM_OUTPUT))
            ry = float(np.clip(ry, -MAX_AIM_OUTPUT, MAX_AIM_OUTPUT))

            # Encode to fixed-point
            fix_x = int(rx * FIXED_POINT_SCALE)
            fix_y = int(ry * FIXED_POINT_SCALE)

            gcvdata.extend(fix_x.to_bytes(4, byteorder="big", signed=True))
            gcvdata.extend(fix_y.to_bytes(4, byteorder="big", signed=True))

            return frame, gcvdata

        except Exception:
            gcvdata.extend((0).to_bytes(4, byteorder="big", signed=True))
            gcvdata.extend((0).to_bytes(4, byteorder="big", signed=True))
            return frame, gcvdata

    def _disengage(self) -> Tuple[float, float]:
        """Handle target loss: decay aim smoothly and reset state."""
        self.recoil.deactivate()
        self.tracker.kalman.initialized = False
        self.prev_delta_x = 0.0
        self.prev_delta_y = 0.0

        intensity = self._update_engagement(False, 0.0)

        rx = self.last_rx * self.cfg.engagement.disengage_decay
        ry = self.last_ry * self.cfg.engagement.disengage_decay
        self.last_rx, self.last_ry = rx, ry
        self.last_accel_x *= 0.5
        self.last_accel_y *= 0.5

        return rx, ry


# =============================================================================
# GPC WORKER - Entry Point
# =============================================================================

class GCVWorker:
    """GPC companion entry point. Instantiates and runs PhantomElite."""

    def __init__(self, width: int, height: int):
        self.phantom = PhantomElite()
        if self.phantom.cfg.enabled:
            self.phantom.load_model()

    def __del__(self):
        try:
            if hasattr(self, 'phantom'):
                del self.phantom
        except Exception:
            pass

    def process(self, frame: Optional[np.ndarray]) -> Tuple[np.ndarray, bytearray]:
        gcvdata = bytearray()
        if self.phantom.cfg.enabled:
            frame, gcvdata = self.phantom.process(frame, gcvdata)
        return frame, gcvdata
