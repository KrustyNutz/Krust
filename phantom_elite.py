"""
PHANTOM APEX v4 — Imperceptible Aim Engine
===========================================
Every design decision answers one question: "Would a pro player do this?"

10 telltale signatures that v3 leaked, all eliminated:

 1. Fixed spring stiffness → ADAPTIVE ω(distance). Slow micro-adjust close,
    fast flick far. Like a real wrist.
 2. Linear engagement ramp → SIGMOID curve modeling human reaction time
    (~150ms neural delay → explosive acceleration → smooth settle).
 3. No aim deadzone → SUB-PIXEL DEADZONE. Humans can't perceive <1px error.
    Stop correcting when you're close enough. Constant micro-correction = sus.
 4. No temporal smoothing → 3-FRAME OUTPUT BUFFER. Single-frame detection
    noise can't produce single-frame aim spikes — real mice have inertia.
 5. Noisy 2-point EMA velocity → RING BUFFER with weighted regression.
    Clean, stable velocity estimate from last 5 frames.
 6. No NMS → VECTORIZED NMS. Overlapping boxes cause target-switch jitter.
 7. Linear speed curve → POWER-CURVE acceleration matching real mouse feel.
 8. Constant noise → CONTEXT-AWARE noise. Tracking = less drift, more tremor.
    Idle = more drift. Just like a real hand.
 9. Binary flick boost → FLICK DYNAMICS with explosive start, smooth
    follow-through, micro-correction settle. The real flick shape.
10. Asymmetric engage/disengage → SYMMETRIC sigmoid. Real humans don't
    have different acceleration curves for starting vs stopping.

Architecture: still one spring at the core. But now it breathes.
"""

import cv2
import numpy as np
import onnxruntime as ort
import os
import time
import math
from typing import Optional, Tuple

# =============================================================================
# CONFIG
# =============================================================================

class cfg:
    enabled = True
    model_path = r"C:\Users\Jayva\Desktop\GtunerIV\BO7MULTI-640.onnx"

    # --- SCREEN ---
    screen_w     = 1920
    screen_h     = 1080
    center_x     = 960
    center_y     = 540

    # --- DETECTION ---
    input_size   = 640
    confidence   = 0.55
    nms_iou      = 0.45         # NMS IoU threshold
    roi_x1, roi_y1 = 640, 220
    roi_x2, roi_y2 = 1280, 860
    onnx_threads = 4
    warmup_runs  = 10

    # --- FOV ---
    fov_radius   = 340.0
    fov_soft     = 50.0

    # --- ADAPTIVE SPRING ---
    # ω adapts with distance: snappy for flicks, gentle for tracking
    spring_base      = 14.0   # ω at medium distance (the anchor)
    spring_close     = 8.0    # ω when crosshair is nearly on target (micro-adjust)
    spring_far       = 22.0   # ω when target is far (flick acquisition)
    spring_curve     = 0.6    # How quickly ω transitions (0=sharp, 1=gradual)
    spring_dist_near = 15.0   # Distance (px) considered "close" for ω scaling
    spring_dist_far  = 200.0  # Distance (px) considered "far" for ω scaling
    max_velocity     = 95.0

    # --- AIM SPEED ---
    speed_x      = 3.8
    speed_y      = 3.8
    # Power curve: output = input^accel_power. >1 = suppresses small, boosts large.
    accel_power  = 1.15       # Mouse acceleration curve exponent

    # --- AIMBONE ---
    aimbone_close   = -0.18
    aimbone_far     = -0.38
    aimbone_sniper  = -0.42
    bbox_close_px   = 120.0
    bbox_far_px     = 50.0

    # --- VELOCITY LEAD ---
    lead_enabled = True
    lead_frames  = 2.5
    vel_buf_size = 5          # Ring buffer frames for regression

    # --- TARGET LOCK ---
    lock_frames      = 45
    switch_threshold = 500.0

    # --- RECOIL ---
    recoil_enabled   = True
    recoil_weapon    = "AR"
    recoil_strength  = 1.2

    # --- ENGAGEMENT ---
    # Sigmoid ramp: models reaction_time → explosive accel → settle
    engage_speed     = 6.0    # Sigmoid steepness
    engage_midpoint  = 0.25   # Seconds to reach 50% intensity
    disengage_speed  = 5.0    # Symmetric disengage sigmoid
    disengage_mid    = 0.15

    # --- DEADZONE ---
    deadzone_px      = 1.8    # Stop correcting below this (screen px)
    deadzone_smooth  = 3.0    # Smooth fade into deadzone

    # --- TEMPORAL BUFFER ---
    temporal_frames  = 3      # Output smoothing buffer size
    temporal_weights = [0.15, 0.30, 0.55]  # Oldest → newest

    # --- NOISE (context-aware) ---
    noise_enabled    = True
    # Tracking state (focused, less drift)
    noise_drift_track  = 0.006
    noise_tremor_track = 0.014
    noise_micro_track  = 0.005
    # Idle state (relaxed, more drift)
    noise_drift_idle   = 0.022
    noise_tremor_idle  = 0.008
    noise_micro_idle   = 0.010
    noise_blend_speed  = 3.0  # How fast noise profile transitions

    # --- FLICK DYNAMICS ---
    flick_enabled     = True
    flick_threshold   = 20.0  # Delta to trigger flick
    flick_attack      = 1.6   # Peak boost (explosive start)
    flick_sustain     = 0.8   # Sustain fraction
    flick_decay_rate  = 0.12  # Per-frame decay
    flick_correction  = 0.92  # Post-flick micro-correction damping

    # --- OUTPUT ---
    max_output       = 100.0
    fixed_point      = 65536.0


# =============================================================================
# RECOIL PATTERNS
# =============================================================================

RECOIL = {
    "AR": {
        "v": [0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0],
        "h": [0.1, -0.15, 0.2, -0.2, 0.25, -0.25, 0.2, -0.15, 0.1, -0.1, 0.0],
        "rpm": 750, "reset": 0.35,
    },
    "SMG": {
        "v": [0.4, 0.5, 0.6, 0.8, 1.0, 1.1, 1.0, 0.9, 0.8, 0.7],
        "h": [0.15, -0.2, 0.25, -0.3, 0.3, -0.25, 0.2, -0.15, 0.1],
        "rpm": 900, "reset": 0.3,
    },
    "LMG": {
        "v": [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4],
        "h": [0.2, -0.3, 0.4, -0.5, 0.6, -0.6, 0.5, -0.4, 0.3, -0.3, 0.2],
        "rpm": 650, "reset": 0.4,
    },
    "SNIPER": {
        "v": [3.5], "h": [0.0], "rpm": 50, "reset": 0.8,
    },
}


# =============================================================================
# ADAPTIVE SPRING — ω breathes with distance
# =============================================================================

class AdaptiveSpring:
    """
    Critically-damped spring where stiffness (ω) adapts to distance.

    Close to target → low ω → slow, precise micro-adjustments
    Far from target → high ω → fast, snappy acquisition
    Transition follows a smoothstep curve — no discontinuities.

    This is what separates a pro from a bot: a bot uses the same
    sensitivity everywhere. A pro's wrist naturally adjusts.
    """

    __slots__ = ('px', 'py', 'vx', 'vy')

    def __init__(self):
        self.px = 0.0
        self.py = 0.0
        self.vx = 0.0
        self.vy = 0.0

    def update(self, tx: float, ty: float, dt: float, distance: float) -> Tuple[float, float]:
        """Advance spring with distance-adaptive ω."""
        w = self._omega(distance)
        w2 = w * w

        # Critically-damped: ẍ = -ω²(x - target) - 2ωẋ
        ax = -w2 * (self.px - tx) - 2.0 * w * self.vx
        ay = -w2 * (self.py - ty) - 2.0 * w * self.vy

        # Semi-implicit Euler
        self.vx += ax * dt
        self.vy += ay * dt

        # Velocity cap
        spd = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        if spd > cfg.max_velocity:
            s = cfg.max_velocity / spd
            self.vx *= s
            self.vy *= s

        self.px += self.vx * dt
        self.py += self.vy * dt
        return self.px, self.py

    @staticmethod
    def _omega(dist: float) -> float:
        """Distance → spring stiffness. Smoothstep interpolation."""
        near, far = cfg.spring_dist_near, cfg.spring_dist_far

        if dist <= near:
            return cfg.spring_close
        if dist >= far:
            return cfg.spring_far

        # Smoothstep with configurable curve
        t = (dist - near) / (far - near)
        t = t ** cfg.spring_curve  # Adjustable transition shape
        t = t * t * (3.0 - 2.0 * t)  # Hermite smoothstep
        return cfg.spring_close + t * (cfg.spring_far - cfg.spring_close)

    def decay(self, factor: float) -> Tuple[float, float]:
        self.px *= factor
        self.py *= factor
        self.vx *= factor
        self.vy *= factor
        return self.px, self.py

    def reset(self):
        self.px = self.py = self.vx = self.vy = 0.0


# =============================================================================
# SIGMOID ENGAGEMENT — models real human reaction time
# =============================================================================

class Engagement:
    """
    Real human aim engagement is sigmoid-shaped:
    - ~150ms dead zone (neural processing)
    - Explosive acceleration (muscle activation)
    - Smooth asymptotic settle

    σ(t) = 1 / (1 + e^(-speed * (t - midpoint)))

    This replaces the linear ramp (intensity += 0.12) which is a dead
    giveaway — no human accelerates linearly.
    """

    __slots__ = ('timer', 'active', 'value')

    def __init__(self):
        self.timer = 0.0
        self.active = False
        self.value = 0.0

    def engage(self, dt: float) -> float:
        """Ramp up with sigmoid. Returns 0→1 intensity."""
        if not self.active:
            self.active = True
            self.timer = 0.0

        self.timer += dt
        raw = 1.0 / (1.0 + math.exp(-cfg.engage_speed * (self.timer - cfg.engage_midpoint)))
        # Rescale so σ(0)=0 (raw sigmoid starts at ~0.18)
        floor = 1.0 / (1.0 + math.exp(cfg.engage_speed * cfg.engage_midpoint))
        self.value = max(0.0, (raw - floor) / (1.0 - floor))
        return self.value

    def disengage(self, dt: float) -> float:
        """Ramp down with symmetric sigmoid."""
        if self.active:
            self.active = False
            self.timer = 0.0

        self.timer += dt
        raw = 1.0 / (1.0 + math.exp(-cfg.disengage_speed * (self.timer - cfg.disengage_mid)))
        floor = 1.0 / (1.0 + math.exp(cfg.disengage_speed * cfg.disengage_mid))
        fade = 1.0 - max(0.0, (raw - floor) / (1.0 - floor))
        self.value *= fade
        return self.value


# =============================================================================
# CONTEXT-AWARE NOISE — adapts to engagement state
# =============================================================================

class ContextNoise:
    """
    A real hand behaves differently when tracking vs idle:
    - Tracking: grip tightens → less drift, more tremor (focused tension)
    - Idle: grip relaxes → more drift, less tremor (relaxed wander)

    The noise profile smoothly blends between these two states based on
    engagement intensity. This is what makes it imperceptible.
    """

    __slots__ = ('dx', 'dy', 'dtx', 'dty', 'phase', 'last_t',
                 'drift_a', 'tremor_a', 'micro_a')

    def __init__(self):
        self.dx = self.dy = 0.0
        self.dtx = self.dty = 0.0
        self.phase = np.random.uniform(0, 6.28)
        self.last_t = time.perf_counter()
        # Current amplitudes (blend targets)
        self.drift_a = cfg.noise_drift_idle
        self.tremor_a = cfg.noise_tremor_idle
        self.micro_a = cfg.noise_micro_idle

    def sample(self, intensity: float) -> Tuple[float, float]:
        """Returns noise scaled by engagement context."""
        now = time.perf_counter()
        dt = max(now - self.last_t, 0.001)
        self.last_t = now

        # Blend noise profile based on intensity
        blend = min(dt * cfg.noise_blend_speed, 1.0)
        t_drift = cfg.noise_drift_track * intensity + cfg.noise_drift_idle * (1.0 - intensity)
        t_tremor = cfg.noise_tremor_track * intensity + cfg.noise_tremor_idle * (1.0 - intensity)
        t_micro = cfg.noise_micro_track * intensity + cfg.noise_micro_idle * (1.0 - intensity)

        self.drift_a += (t_drift - self.drift_a) * blend
        self.tremor_a += (t_tremor - self.tremor_a) * blend
        self.micro_a += (t_micro - self.micro_a) * blend

        # Drift: slow random walk (~0.5Hz target changes)
        if np.random.random() < dt * 1.5:
            self.dtx = np.random.randn() * self.drift_a
            self.dty = np.random.randn() * self.drift_a
        self.dx += (self.dtx - self.dx) * min(dt * 3.0, 1.0)
        self.dy += (self.dty - self.dy) * min(dt * 3.0, 1.0)

        # Tremor: ~10Hz physiological oscillation
        self.phase += dt * 62.8  # 2π * 10
        tx = math.sin(self.phase) * self.tremor_a
        ty = math.cos(self.phase * 1.37) * self.tremor_a * 0.65

        # Micro: neural white noise
        mx = np.random.randn() * self.micro_a
        my = np.random.randn() * self.micro_a

        return self.dx + tx + mx, self.dy + ty + my


# =============================================================================
# VELOCITY ESTIMATOR — Ring buffer + weighted regression
# =============================================================================

class VelocityEstimator:
    """
    Replaces noisy 2-point EMA with a 5-frame weighted ring buffer.
    Uses weighted least-squares regression for a clean velocity estimate.
    Recent frames matter more (exponential weights).
    """

    __slots__ = ('buf_x', 'buf_y', 'idx', 'count', 'n', 'weights', 'vx', 'vy')

    def __init__(self):
        n = cfg.vel_buf_size
        self.n = n
        self.buf_x = np.zeros(n, dtype=np.float64)
        self.buf_y = np.zeros(n, dtype=np.float64)
        self.idx = 0
        self.count = 0
        # Exponential weights: newest = 1.0, oldest decays
        self.weights = np.array([0.6 ** (n - 1 - i) for i in range(n)], dtype=np.float64)
        self.vx = 0.0
        self.vy = 0.0

    def push(self, x: float, y: float) -> Tuple[float, float]:
        """Add sample and return (vx, vy) via weighted regression."""
        self.buf_x[self.idx] = x
        self.buf_y[self.idx] = y
        self.idx = (self.idx + 1) % self.n
        self.count = min(self.count + 1, self.n)

        if self.count < 3:
            self.vx = self.vy = 0.0
            return 0.0, 0.0

        # Build ordered arrays (oldest → newest)
        k = self.count
        idxs = [(self.idx - k + i) % self.n for i in range(k)]
        xs = self.buf_x[idxs]
        ys = self.buf_y[idxs]
        w = self.weights[-k:]

        # Weighted linear regression: velocity = slope
        t = np.arange(k, dtype=np.float64)
        wt = w * t
        ws = w.sum()
        wts = wt.sum()
        wt2 = (w * t * t).sum()
        denom = ws * wt2 - wts * wts
        if abs(denom) < 1e-12:
            self.vx = self.vy = 0.0
            return 0.0, 0.0

        self.vx = float((ws * (w * t * xs).sum() - wts * (w * xs).sum()) / denom)
        self.vy = float((ws * (w * t * ys).sum() - wts * (w * ys).sum()) / denom)
        return self.vx, self.vy

    def reset(self):
        self.count = 0
        self.idx = 0
        self.vx = self.vy = 0.0


# =============================================================================
# FLICK DYNAMICS — proper acceleration profile
# =============================================================================

class FlickDynamics:
    """
    Real flicks have a shape: explosive start → sustain → micro-correction.
    Not a flat 1.4x multiplier.

    Phase 1 (attack):  Boost spikes to flick_attack (1.6x)
    Phase 2 (sustain): Holds at flick_sustain fraction while moving
    Phase 3 (settle):  Decays below 1.0 for micro-correction (the overshoot fix)

    This produces the natural flick→settle pattern pro players have.
    """

    __slots__ = ('boost', 'phase', 'cooldown')

    def __init__(self):
        self.boost = 1.0
        self.phase = 0  # 0=idle, 1=attack, 2=sustain, 3=settle
        self.cooldown = 0

    def process(self, rx: float, ry: float, prev_rx: float, prev_ry: float) -> Tuple[float, float]:
        if not cfg.flick_enabled:
            return rx, ry

        delta = math.sqrt((rx - prev_rx)**2 + (ry - prev_ry)**2)

        # Trigger detection
        if self.phase == 0 and delta > cfg.flick_threshold and self.cooldown <= 0:
            self.phase = 1
            self.boost = cfg.flick_attack

        # State machine
        if self.phase == 1:
            # Attack: apply peak boost, transition to sustain
            self.phase = 2
        elif self.phase == 2:
            # Sustain: decay toward 1.0
            self.boost -= cfg.flick_decay_rate
            if self.boost <= 1.05:
                self.phase = 3
                self.boost = cfg.flick_correction  # Dip below 1.0
        elif self.phase == 3:
            # Settle: micro-correction (damped, below 1.0)
            self.boost += (1.0 - self.boost) * 0.25
            if abs(self.boost - 1.0) < 0.02:
                self.boost = 1.0
                self.phase = 0
                self.cooldown = 5

        if self.cooldown > 0:
            self.cooldown -= 1

        return rx * self.boost, ry * self.boost


# =============================================================================
# TEMPORAL OUTPUT BUFFER — eliminates single-frame spikes
# =============================================================================

class TemporalBuffer:
    """
    A real mouse has physical inertia — it can't produce a single-frame
    spike and return to normal. But noisy detections can.

    This 3-frame weighted average makes the output physically plausible.
    Weights favor the newest frame (0.55) but blend in recent history
    to eliminate any single-frame anomaly.
    """

    __slots__ = ('buf', 'n', 'weights', 'idx', 'count')

    def __init__(self):
        self.n = cfg.temporal_frames
        self.buf = np.zeros((self.n, 2), dtype=np.float64)
        self.weights = np.array(cfg.temporal_weights, dtype=np.float64)
        self.weights /= self.weights.sum()  # Normalize
        self.idx = 0
        self.count = 0

    def push(self, x: float, y: float) -> Tuple[float, float]:
        self.buf[self.idx, 0] = x
        self.buf[self.idx, 1] = y
        self.idx = (self.idx + 1) % self.n
        self.count = min(self.count + 1, self.n)

        if self.count < self.n:
            return x, y

        # Weighted average (ordered oldest → newest)
        ordered = np.array([self.buf[(self.idx + i) % self.n] for i in range(self.n)])
        result = (ordered * self.weights[:, None]).sum(axis=0)
        return float(result[0]), float(result[1])

    def reset(self):
        self.buf[:] = 0.0
        self.count = 0
        self.idx = 0


# =============================================================================
# RECOIL CONTROLLER
# =============================================================================

class Recoil:
    __slots__ = ('shot_idx', 'last_t', 'active')

    def __init__(self):
        self.shot_idx = 0
        self.last_t = 0.0
        self.active = False

    def tick(self, firing: bool) -> Tuple[float, float]:
        if not cfg.recoil_enabled:
            return 0.0, 0.0
        if not firing:
            self.active = False
            return 0.0, 0.0

        now = time.perf_counter()
        pat = RECOIL.get(cfg.recoil_weapon, RECOIL["AR"])

        if not self.active:
            self.active = True
            self.shot_idx = 0
            self.last_t = now
        elif now - self.last_t > pat["reset"]:
            self.shot_idx = 0
        else:
            delay = 60.0 / pat["rpm"]
            elapsed = int((now - self.last_t) / delay)
            if elapsed > 0:
                self.shot_idx = min(self.shot_idx + elapsed, len(pat["v"]) - 1)

        self.last_t = now
        vi = min(self.shot_idx, len(pat["v"]) - 1)
        hi = min(self.shot_idx, len(pat["h"]) - 1)
        s = cfg.recoil_strength
        return (pat["h"][hi] * s + np.random.uniform(-0.03, 0.03) * s,
                pat["v"][vi] * s + np.random.uniform(-0.05, 0.05) * s)


# =============================================================================
# NMS — Vectorized Non-Maximum Suppression
# =============================================================================

def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """
    Vectorized NMS. boxes: Nx4 (x,y,w,h center format), scores: N.
    Returns indices to keep. Eliminates overlapping detections that
    cause target-switch jitter.
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    # Convert center (x,y,w,h) → corner (x1,y1,x2,y2)
    hw = boxes[:, 2] * 0.5
    hh = boxes[:, 3] * 0.5
    x1 = boxes[:, 0] - hw
    y1 = boxes[:, 1] - hh
    x2 = boxes[:, 0] + hw
    y2 = boxes[:, 1] + hh

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-8)
        order = rest[iou < iou_thresh]

    return np.array(keep, dtype=np.int64)


# =============================================================================
# PHANTOM APEX — The Engine
# =============================================================================

class Phantom:
    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None

        # Core systems
        self.spring = AdaptiveSpring()
        self.engagement = Engagement()
        self.noise = ContextNoise() if cfg.noise_enabled else None
        self.flick = FlickDynamics()
        self.temporal = TemporalBuffer()
        self.velocity = VelocityEstimator()
        self.recoil = Recoil()

        # Target tracking
        self.locked_xy: Optional[Tuple[float, float, float, float]] = None
        self.lock_ttl = 0

        # Timing
        self.last_frame_t = time.perf_counter()

        # Flick previous output
        self.prev_rx = 0.0
        self.prev_ry = 0.0

        # Pre-allocated
        sz = cfg.input_size
        self._blob = np.empty((1, 3, sz, sz), dtype=np.float32)
        self._roi_scale = np.array([
            (cfg.roi_x2 - cfg.roi_x1) / sz,
            (cfg.roi_y2 - cfg.roi_y1) / sz,
            (cfg.roi_x2 - cfg.roi_x1) / sz,
            (cfg.roi_y2 - cfg.roi_y1) / sz,
        ], dtype=np.float32)
        self._roi_offset = np.array([
            cfg.roi_x1, cfg.roi_y1, 0.0, 0.0
        ], dtype=np.float32)

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    def load_model(self) -> None:
        path = cfg.model_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = cfg.onnx_threads
        opts.inter_op_num_threads = cfg.onnx_threads
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True

        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(path, providers=providers, sess_options=opts)
        print(f"APEX v4 | {self.session.get_providers()[0]}")

        dummy = np.random.rand(1, 3, cfg.input_size, cfg.input_size).astype(np.float32)
        for _ in range(cfg.warmup_runs):
            self.session.run(None, {"images": dummy})
        print("Ready")

    # -------------------------------------------------------------------------
    # Detection + NMS
    # -------------------------------------------------------------------------

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        resized = cv2.resize(roi, (cfg.input_size, cfg.input_size), interpolation=cv2.INTER_LINEAR)
        buf = self._blob[0]
        np.divide(resized[:, :, 0], 255.0, out=buf[0])
        np.divide(resized[:, :, 1], 255.0, out=buf[1])
        np.divide(resized[:, :, 2], 255.0, out=buf[2])
        return self._blob

    def _detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect + NMS. Returns Nx5 (x,y,w,h,conf) in screen space or None."""
        roi = frame[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2]
        blob = self._preprocess(roi)
        raw = self.session.run(None, {"images": blob})[0]
        dets = raw[0].T

        mask = dets[:, 4] > cfg.confidence
        valid = dets[mask]
        if len(valid) == 0:
            return None

        # NMS before coordinate transform (faster on model-space coords)
        keep = _nms(valid[:, :4], valid[:, 4], cfg.nms_iou)
        if len(keep) == 0:
            return None
        valid = valid[keep]

        coords = valid[:, :4] * self._roi_scale + self._roi_offset
        confs = valid[:, 4:5]
        return np.hstack([coords, confs])

    # -------------------------------------------------------------------------
    # Target Selection
    # -------------------------------------------------------------------------

    def _select_target(self, dets: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        cx, cy = cfg.center_x, cfg.center_y
        dx = dets[:, 0] - cx
        dy = dets[:, 1] - cy
        dist = np.sqrt(dx * dx + dy * dy)

        in_fov = dist < cfg.fov_radius
        if not np.any(in_fov):
            return None

        dist_score = 1.0 / (1.0 + dist / 80.0)
        conf_score = dets[:, 4]
        size_score = np.minimum(dets[:, 2] * dets[:, 3] / 8000.0, 1.5)
        vert_bias = 1.0 - np.abs(dets[:, 1] - cy) / cy

        scores = dist_score * 3.0 + conf_score * 1.0 + size_score * 0.3 + vert_bias * 0.2
        scores[~in_fov] = -np.inf

        best = np.argmax(scores)
        d = dets[best]
        return (float(d[0]), float(d[1]), float(d[2]), float(d[3]))

    # -------------------------------------------------------------------------
    # Target Lock with Hysteresis
    # -------------------------------------------------------------------------

    def _track(self, dets: Optional[np.ndarray]) -> Optional[Tuple[float, float, float, float]]:
        if dets is None or len(dets) == 0:
            self.lock_ttl -= 1
            if self.lock_ttl <= 0:
                self.locked_xy = None
            return self.locked_xy

        best = self._select_target(dets)
        if best is None:
            self.lock_ttl -= 1
            if self.lock_ttl <= 0:
                self.locked_xy = None
            return self.locked_xy

        if self.locked_xy is None:
            self.locked_xy = best
            self.lock_ttl = cfg.lock_frames
            return best

        lx, ly = self.locked_xy[0], self.locked_xy[1]
        dx = dets[:, 0] - lx
        dy = dets[:, 1] - ly
        dists = np.sqrt(dx * dx + dy * dy)
        nearest_idx = np.argmin(dists)

        if dists[nearest_idx] < 180.0:
            d = dets[nearest_idx]
            self.locked_xy = (float(d[0]), float(d[1]), float(d[2]), float(d[3]))
            self.lock_ttl = cfg.lock_frames
            return self.locked_xy

        cx, cy = cfg.center_x, cfg.center_y
        lock_dist = math.sqrt((lx - cx)**2 + (ly - cy)**2)
        best_dist = math.sqrt((best[0] - cx)**2 + (best[1] - cy)**2)

        if (lock_dist - best_dist) > cfg.switch_threshold:
            self.locked_xy = best
            self.lock_ttl = cfg.lock_frames
        else:
            self.lock_ttl -= 1
            if self.lock_ttl <= 0:
                self.locked_xy = best
                self.lock_ttl = cfg.lock_frames

        return self.locked_xy

    # -------------------------------------------------------------------------
    # Aimbone
    # -------------------------------------------------------------------------

    @staticmethod
    def _aimbone_offset(bbox_h: float) -> float:
        if cfg.recoil_weapon == "SNIPER":
            return cfg.aimbone_sniper
        if bbox_h >= cfg.bbox_close_px:
            return cfg.aimbone_close
        if bbox_h <= cfg.bbox_far_px:
            return cfg.aimbone_far
        t = (bbox_h - cfg.bbox_far_px) / (cfg.bbox_close_px - cfg.bbox_far_px)
        t = t * t * (3.0 - 2.0 * t)
        return cfg.aimbone_far + t * (cfg.aimbone_close - cfg.aimbone_far)

    # -------------------------------------------------------------------------
    # FOV Falloff
    # -------------------------------------------------------------------------

    @staticmethod
    def _fov_factor(distance: float) -> float:
        inner = cfg.fov_radius - cfg.fov_soft
        if distance <= inner:
            return 1.0
        if distance >= cfg.fov_radius:
            return 0.0
        return (cfg.fov_radius - distance) / cfg.fov_soft

    # -------------------------------------------------------------------------
    # Deadzone — stop correcting at sub-pixel distances
    # -------------------------------------------------------------------------

    @staticmethod
    def _deadzone(dx: float, dy: float, dist: float) -> Tuple[float, float]:
        """Smooth deadzone: corrections fade to zero below threshold."""
        if dist < cfg.deadzone_px:
            return 0.0, 0.0
        if dist < cfg.deadzone_px * cfg.deadzone_smooth:
            # Smooth fade
            t = (dist - cfg.deadzone_px) / (cfg.deadzone_px * (cfg.deadzone_smooth - 1.0))
            t = t * t  # Quadratic ease-in
            return dx * t, dy * t
        return dx, dy

    # -------------------------------------------------------------------------
    # Power-curve acceleration
    # -------------------------------------------------------------------------

    @staticmethod
    def _accel_curve(rx: float, ry: float) -> Tuple[float, float]:
        """Mouse acceleration: output = sign(input) * |input|^power."""
        p = cfg.accel_power
        ax = math.copysign(abs(rx) ** p, rx) if rx != 0.0 else 0.0
        ay = math.copysign(abs(ry) ** p, ry) if ry != 0.0 else 0.0
        return ax, ay

    # -------------------------------------------------------------------------
    # Main Pipeline
    # -------------------------------------------------------------------------

    def process(self, frame: Optional[np.ndarray], gcvdata: bytearray) -> Tuple[np.ndarray, bytearray]:
        """
        1. Detect+NMS → 2. Track → 3. Aimbone → 4. Lead (regression) →
        5. Deadzone → 6. FOV gate → 7. Sigmoid engage → 8. Accel curve →
        9. Adaptive spring → 10. Flick dynamics → 11. Context noise →
        12. Recoil → 13. Temporal buffer → 14. Output
        """
        empty = frame if frame is not None else np.zeros((cfg.screen_h, cfg.screen_w, 3), dtype=np.uint8)

        if frame is None or frame.size == 0 or self.session is None:
            gcvdata.extend((0).to_bytes(4, "big", signed=True))
            gcvdata.extend((0).to_bytes(4, "big", signed=True))
            return empty, gcvdata

        now = time.perf_counter()
        dt = min(now - self.last_frame_t, 0.05)
        self.last_frame_t = now

        try:
            # 1. Detect + NMS
            dets = self._detect(frame)

            # 2. Track
            target = self._track(dets)

            if target is not None:
                tx, ty, tw, th = target

                # 3. Aimbone
                aim_y = ty + th * self._aimbone_offset(th)
                aim_x = tx

                # 4. Velocity lead (ring buffer regression)
                vx, vy = self.velocity.push(aim_x, aim_y)
                if cfg.lead_enabled:
                    aim_x += vx * cfg.lead_frames
                    aim_y += vy * cfg.lead_frames

                # 5. Raw delta
                dx = aim_x - cfg.center_x
                dy = aim_y - cfg.center_y
                distance = math.sqrt(dx * dx + dy * dy)

                # 6. Deadzone
                dx, dy = self._deadzone(dx, dy, distance)
                if dx == 0.0 and dy == 0.0 and distance < cfg.deadzone_px:
                    # On target — maintain spring but output zero
                    rx, ry = 0.0, 0.0
                else:
                    # 7. FOV
                    fov = self._fov_factor(distance)
                    if fov <= 0.0:
                        rx, ry = self._do_disengage(dt)
                    else:
                        # 8. Sigmoid engagement
                        intensity = self.engagement.engage(dt)

                        # Raw aim vector
                        raw_x = (dx / cfg.screen_w) * cfg.speed_x * 100.0 * fov * intensity
                        raw_y = (dy / cfg.screen_h) * cfg.speed_y * 100.0 * fov * intensity

                        # 9. Power-curve acceleration
                        raw_x, raw_y = self._accel_curve(raw_x, raw_y)

                        # 10. Adaptive spring (ω scales with distance)
                        rx, ry = self.spring.update(raw_x, raw_y, dt, distance)

                        # 11. Flick dynamics
                        rx, ry = self.flick.process(rx, ry, self.prev_rx, self.prev_ry)

                        # 12. Context-aware noise
                        if self.noise and cfg.noise_enabled:
                            nx, ny = self.noise.sample(intensity)
                            rx += nx
                            ry += ny

                        # 13. Recoil
                        rh, rv = self.recoil.tick(True)
                        rx += rh
                        ry += rv
            else:
                rx, ry = self._do_disengage(dt)
                self.recoil.tick(False)

            # Store for flick delta
            self.prev_rx = rx
            self.prev_ry = ry

            # 14. Temporal buffer (eliminates single-frame spikes)
            rx, ry = self.temporal.push(rx, ry)

            # Clamp + encode
            rx = max(-cfg.max_output, min(cfg.max_output, rx))
            ry = max(-cfg.max_output, min(cfg.max_output, ry))

            gcvdata.extend(int(rx * cfg.fixed_point).to_bytes(4, "big", signed=True))
            gcvdata.extend(int(ry * cfg.fixed_point).to_bytes(4, "big", signed=True))
            return frame, gcvdata

        except Exception:
            gcvdata.extend((0).to_bytes(4, "big", signed=True))
            gcvdata.extend((0).to_bytes(4, "big", signed=True))
            return frame, gcvdata

    def _do_disengage(self, dt: float) -> Tuple[float, float]:
        self.engagement.disengage(dt)
        self.velocity.reset()
        return self.spring.decay(0.85)


# =============================================================================
# GPC ENTRY POINT
# =============================================================================

class GCVWorker:
    def __init__(self, width: int, height: int):
        self.phantom = Phantom()
        if cfg.enabled:
            self.phantom.load_model()

    def __del__(self):
        try:
            del self.phantom
        except Exception:
            pass

    def process(self, frame):
        gcvdata = bytearray()
        if cfg.enabled:
            frame, gcvdata = self.phantom.process(frame, gcvdata)
        return frame, gcvdata
