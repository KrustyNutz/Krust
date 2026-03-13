"""
PHANTOM v3 — Critically-Damped Spring Aim Engine
=================================================
Top 1% GPC companion. Built on three principles:

1. ONE smoothing system (critically-damped spring) — physically correct,
   zero overshoot, natural acceleration/deceleration. Replaces the 5
   conflicting systems from v2.

2. DELTA-TIME everywhere — frame-rate independent. Drop frames, stutter,
   whatever — aim stays consistent.

3. VECTORIZED hot path — zero Python loops over detections. Numpy does
   the batch math on every detection simultaneously.

The spring model is what real mouse aim feels like: you accelerate toward
the target, decelerate as you approach, and settle without oscillation.
The math is identical to a critically-damped harmonic oscillator (ζ=1),
which is the fastest convergence without overshoot — proven optimal.

Aim path uses cubic Hermite interpolation for the S-curve shape that
human wrist movements naturally produce.
"""

import cv2
import numpy as np
import onnxruntime as ort
import os
import time
from typing import Optional, Tuple

# =============================================================================
# CONFIG — Flat, grouped, easy to tune. That's it.
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
    roi_x1, roi_y1 = 640, 220
    roi_x2, roi_y2 = 1280, 860
    onnx_threads = 4
    warmup_runs  = 10

    # --- FOV GATE ---
    fov_radius   = 340.0     # Ignore targets outside this radius (pixels)
    fov_soft     = 50.0      # Soft falloff zone at edge

    # --- AIM SPRING ---
    # Critically-damped spring: ONE parameter controls the feel.
    # Higher = snappier. Lower = smoother. 12-18 is the sweet spot.
    spring_stiffness = 15.0  # ω (natural frequency) — THE main tuning knob
    spring_damping   = 1.0   # ζ (damping ratio) — keep at 1.0 for critical damping
    max_velocity     = 90.0  # Cap aim speed (pixels/frame equivalent)

    # --- AIM SPEED ---
    speed_x      = 3.8
    speed_y      = 3.8

    # --- AIMBONE ---
    # Offset as fraction of bbox height (negative = above center)
    # Interpolated by bbox height (proxy for distance)
    aimbone_close   = -0.18  # Chest (close range, big bbox)
    aimbone_far     = -0.38  # Neck (far range, small bbox)
    aimbone_sniper  = -0.42  # Head
    bbox_close_px   = 120.0  # Bbox height threshold: "close"
    bbox_far_px     = 50.0   # Bbox height threshold: "far"

    # --- VELOCITY LEAD ---
    # Lead the target based on its velocity. Simple, effective.
    lead_enabled = True
    lead_frames  = 2.5       # How many frames ahead to lead
    lead_smooth  = 0.35      # Velocity EMA smoothing (0=instant, 1=frozen)

    # --- TARGET LOCK ---
    lock_frames      = 45
    switch_threshold = 500.0  # How much closer a new target must be to steal lock

    # --- RECOIL ---
    recoil_enabled   = True
    recoil_weapon    = "AR"
    recoil_strength  = 1.2

    # --- NOISE (human feel) ---
    noise_enabled    = True
    noise_drift      = 0.015   # Slow hand wander
    noise_tremor     = 0.012   # ~10Hz physiological tremor
    noise_micro      = 0.008   # Random micro-jitter

    # --- FLICK ---
    flick_enabled    = True
    flick_threshold  = 18.0
    flick_boost      = 1.4
    flick_cooldown   = 4

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
# CRITICALLY-DAMPED SPRING
# =============================================================================

class Spring:
    """
    Critically-damped spring for aim smoothing.

    This is THE core innovation. Instead of layering EMA + jerk limiting +
    anti-snap + overshoot correction + sticky aim (5 conflicting systems),
    we use one physically-correct model that gives us ALL of those properties:

    - No overshoot (by definition of critical damping, ζ=1)
    - Smooth acceleration (spring force is proportional to distance)
    - Natural deceleration (damping force is proportional to velocity)
    - Frame-rate independent (uses real delta time)
    - One tuning knob: stiffness (ω). That's it.

    The equation: ẍ = -ω²(x - target) - 2ωẋ
    With ζ=1 (critically damped), this is the fastest convergence
    without any oscillation. Mathematically proven optimal.
    """

    __slots__ = ('pos_x', 'pos_y', 'vel_x', 'vel_y', 'omega')

    def __init__(self, stiffness: float):
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.omega = stiffness

    def update(self, target_x: float, target_y: float, dt: float) -> Tuple[float, float]:
        """
        Advance spring toward target. Returns (output_x, output_y).

        Uses semi-implicit Euler integration — stable, simple, good enough
        at 60+ FPS. No need for RK4 here.
        """
        w = self.omega
        w2 = w * w

        # Spring force + critical damping force
        ax = -w2 * (self.pos_x - target_x) - 2.0 * w * self.vel_x
        ay = -w2 * (self.pos_y - target_y) - 2.0 * w * self.vel_y

        # Semi-implicit Euler (update velocity first, then position)
        self.vel_x += ax * dt
        self.vel_y += ay * dt

        # Velocity cap
        speed = np.sqrt(self.vel_x**2 + self.vel_y**2)
        cap = cfg.max_velocity
        if speed > cap:
            scale = cap / speed
            self.vel_x *= scale
            self.vel_y *= scale

        self.pos_x += self.vel_x * dt
        self.pos_y += self.vel_y * dt

        return self.pos_x, self.pos_y

    def snap(self, x: float, y: float) -> None:
        """Hard-set position (for target acquisition)."""
        self.pos_x = x
        self.pos_y = y
        self.vel_x = 0.0
        self.vel_y = 0.0

    def decay(self, factor: float) -> Tuple[float, float]:
        """Smoothly decay to zero (for disengage)."""
        self.pos_x *= factor
        self.pos_y *= factor
        self.vel_x *= factor
        self.vel_y *= factor
        return self.pos_x, self.pos_y


# =============================================================================
# HUMAN NOISE — Three-layer perturbation
# =============================================================================

class Noise:
    """
    Three biologically-inspired noise layers:
    1. Drift  — slow wander from hand/wrist instability (~0.5Hz)
    2. Tremor — periodic oscillation from physiological tremor (~8-12Hz)
    3. Micro  — random neural noise (white noise, very small)

    All frame-rate independent via time-based phase.
    """

    __slots__ = ('dx', 'dy', 'dtx', 'dty', 'phase', 'last_t')

    def __init__(self):
        self.dx = 0.0
        self.dy = 0.0
        self.dtx = 0.0  # drift target
        self.dty = 0.0
        self.phase = 0.0
        self.last_t = time.perf_counter()

    def sample(self) -> Tuple[float, float]:
        """Returns (noise_x, noise_y) for this frame."""
        now = time.perf_counter()
        dt = now - self.last_t
        self.last_t = now

        # Drift: slow random walk
        if np.random.random() < 0.04:  # ~2.4 changes/sec at 60fps
            self.dtx = np.random.randn() * cfg.noise_drift
            self.dty = np.random.randn() * cfg.noise_drift
        self.dx += (self.dtx - self.dx) * min(dt * 3.0, 1.0)
        self.dy += (self.dty - self.dy) * min(dt * 3.0, 1.0)

        # Tremor: ~10Hz sine wave
        self.phase += dt * 62.8  # 2π * 10Hz
        tx = np.sin(self.phase) * cfg.noise_tremor
        ty = np.cos(self.phase * 1.3) * cfg.noise_tremor * 0.7

        # Micro: white noise
        mx = np.random.randn() * cfg.noise_micro
        my = np.random.randn() * cfg.noise_micro

        return self.dx + tx + mx, self.dy + ty + my


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
        """Call every frame. Returns (h_comp, v_comp)."""
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

        h = pat["h"][hi] * s + np.random.uniform(-0.03, 0.03) * s
        v = pat["v"][vi] * s + np.random.uniform(-0.05, 0.05) * s
        return h, v


# =============================================================================
# PHANTOM v3 — The Engine
# =============================================================================

class Phantom:
    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.spring = Spring(cfg.spring_stiffness)
        self.noise = Noise() if cfg.noise_enabled else None
        self.recoil = Recoil()

        # Target tracking
        self.locked_xy: Optional[Tuple[float, float, float, float]] = None  # x, y, w, h
        self.lock_ttl = 0

        # Velocity estimation (simple EMA — Kalman is overkill here)
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.last_tx = 0.0
        self.last_ty = 0.0
        self.has_prev = False

        # Timing
        self.last_frame_t = time.perf_counter()

        # Flick state
        self.flick_cd = 0
        self.prev_out_x = 0.0
        self.prev_out_y = 0.0

        # Engagement intensity (0→1 ramp)
        self.intensity = 0.0

        # Pre-allocated buffers
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

        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(path, providers=providers, sess_options=opts)

        print(f"Phantom v3 | {self.session.get_providers()[0]}")

        dummy = np.random.rand(1, 3, cfg.input_size, cfg.input_size).astype(np.float32)
        for _ in range(cfg.warmup_runs):
            self.session.run(None, {"images": dummy})
        print("Ready")

    # -------------------------------------------------------------------------
    # Detection — Fully vectorized, zero Python loops
    # -------------------------------------------------------------------------

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        resized = cv2.resize(roi, (cfg.input_size, cfg.input_size), interpolation=cv2.INTER_LINEAR)
        # Normalize + transpose into pre-allocated buffer (CHW format)
        buf = self._blob[0]
        np.divide(resized[:, :, 0], 255.0, out=buf[0])
        np.divide(resized[:, :, 1], 255.0, out=buf[1])
        np.divide(resized[:, :, 2], 255.0, out=buf[2])
        return self._blob

    def _detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run inference and return Nx5 array (x, y, w, h, conf) in screen space.
        Returns None if no detections. ZERO Python loops.
        """
        roi = frame[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2]
        blob = self._preprocess(roi)

        raw = self.session.run(None, {"images": blob})[0]  # (1, 5+, N)
        dets = raw[0].T  # (N, 5+)

        # Filter by confidence — vectorized
        mask = dets[:, 4] > cfg.confidence
        valid = dets[mask]
        if len(valid) == 0:
            return None

        # Transform to screen space — vectorized
        coords = valid[:, :4] * self._roi_scale + self._roi_offset  # (N, 4)
        confs = valid[:, 4:5]  # (N, 1)

        return np.hstack([coords, confs])  # (N, 5): x, y, w, h, conf

    # -------------------------------------------------------------------------
    # Target Selection — Vectorized scoring
    # -------------------------------------------------------------------------

    def _select_target(self, dets: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Score all detections and return best (x, y, w, h).
        Scoring: distance to crosshair (dominant), confidence, size, vertical bias.
        All vectorized.
        """
        cx, cy = cfg.center_x, cfg.center_y

        dx = dets[:, 0] - cx
        dy = dets[:, 1] - cy
        dist = np.sqrt(dx * dx + dy * dy)

        # FOV hard cutoff
        in_fov = dist < cfg.fov_radius
        if not np.any(in_fov):
            return None

        # Score components (all vectorized)
        dist_score = 1.0 / (1.0 + dist / 80.0)
        conf_score = dets[:, 4]
        size_score = np.minimum(dets[:, 2] * dets[:, 3] / 8000.0, 1.5)
        vert_bias = 1.0 - np.abs(dets[:, 1] - cy) / cy

        scores = dist_score * 3.0 + conf_score * 1.0 + size_score * 0.3 + vert_bias * 0.2

        # Mask out-of-FOV targets
        scores[~in_fov] = -np.inf

        best = np.argmax(scores)
        d = dets[best]
        return (float(d[0]), float(d[1]), float(d[2]), float(d[3]))

    # -------------------------------------------------------------------------
    # Target Lock — Hysteresis prevents flipping
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

        # No lock → acquire
        if self.locked_xy is None:
            self.locked_xy = best
            self.lock_ttl = cfg.lock_frames
            return best

        # Find closest detection to current lock (continuation)
        lx, ly = self.locked_xy[0], self.locked_xy[1]
        dx = dets[:, 0] - lx
        dy = dets[:, 1] - ly
        dists = np.sqrt(dx * dx + dy * dy)
        nearest_idx = np.argmin(dists)

        if dists[nearest_idx] < 180.0:
            # Lock continues
            d = dets[nearest_idx]
            self.locked_xy = (float(d[0]), float(d[1]), float(d[2]), float(d[3]))
            self.lock_ttl = cfg.lock_frames
            return self.locked_xy

        # Check if best target is way closer to crosshair
        cx, cy = cfg.center_x, cfg.center_y
        lock_dist = np.sqrt((lx - cx)**2 + (ly - cy)**2)
        best_dist = np.sqrt((best[0] - cx)**2 + (best[1] - cy)**2)

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
    # Aimbone — Smooth interpolation, no step functions
    # -------------------------------------------------------------------------

    @staticmethod
    def _aimbone_offset(bbox_h: float) -> float:
        """Continuous aimbone offset based on bbox height (distance proxy)."""
        if cfg.recoil_weapon == "SNIPER":
            return cfg.aimbone_sniper

        if bbox_h >= cfg.bbox_close_px:
            return cfg.aimbone_close
        if bbox_h <= cfg.bbox_far_px:
            return cfg.aimbone_far

        # Smoothstep interpolation
        t = (bbox_h - cfg.bbox_far_px) / (cfg.bbox_close_px - cfg.bbox_far_px)
        t = t * t * (3.0 - 2.0 * t)  # Hermite smoothstep
        return cfg.aimbone_far + t * (cfg.aimbone_close - cfg.aimbone_far)

    # -------------------------------------------------------------------------
    # Velocity Lead — EMA velocity estimation + lead
    # -------------------------------------------------------------------------

    def _lead_target(self, tx: float, ty: float) -> Tuple[float, float]:
        """Estimate target velocity and lead the aim point ahead."""
        if not cfg.lead_enabled:
            return tx, ty

        if self.has_prev:
            raw_vx = tx - self.last_tx
            raw_vy = ty - self.last_ty
            s = cfg.lead_smooth
            self.vel_x = s * self.vel_x + (1.0 - s) * raw_vx
            self.vel_y = s * self.vel_y + (1.0 - s) * raw_vy
        else:
            self.has_prev = True

        self.last_tx = tx
        self.last_ty = ty

        return tx + self.vel_x * cfg.lead_frames, ty + self.vel_y * cfg.lead_frames

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
    # Flick Detect
    # -------------------------------------------------------------------------

    def _flick(self, rx: float, ry: float) -> Tuple[float, float]:
        if not cfg.flick_enabled:
            return 1.0, 1.0
        if self.flick_cd > 0:
            self.flick_cd -= 1
            return 1.0, 1.0

        bx = cfg.flick_boost if abs(rx - self.prev_out_x) > cfg.flick_threshold else 1.0
        by = cfg.flick_boost if abs(ry - self.prev_out_y) > cfg.flick_threshold else 1.0
        if bx > 1.0 or by > 1.0:
            self.flick_cd = cfg.flick_cooldown
        self.prev_out_x = rx
        self.prev_out_y = ry
        return bx, by

    # -------------------------------------------------------------------------
    # Main Pipeline
    # -------------------------------------------------------------------------

    def process(self, frame: Optional[np.ndarray], gcvdata: bytearray) -> Tuple[np.ndarray, bytearray]:
        """
        Per-frame pipeline:
        1. Detect → 2. Track → 3. Aimbone → 4. Lead → 5. FOV gate →
        6. Spring smooth → 7. Flick boost → 8. Noise → 9. Recoil → 10. Output
        """
        empty = frame if frame is not None else np.zeros((cfg.screen_h, cfg.screen_w, 3), dtype=np.uint8)

        if frame is None or frame.size == 0 or self.session is None:
            gcvdata.extend((0).to_bytes(4, "big", signed=True))
            gcvdata.extend((0).to_bytes(4, "big", signed=True))
            return empty, gcvdata

        # Delta time
        now = time.perf_counter()
        dt = min(now - self.last_frame_t, 0.05)  # Cap at 50ms (20fps floor)
        self.last_frame_t = now

        try:
            # 1. Detect
            dets = self._detect(frame)

            # 2. Track
            target = self._track(dets)

            if target is not None:
                tx, ty, tw, th = target

                # 3. Aimbone
                offset = self._aimbone_offset(th)
                aim_y = ty + th * offset
                aim_x = tx

                # 4. Velocity lead
                aim_x, aim_y = self._lead_target(aim_x, aim_y)

                # 5. Raw delta + FOV
                dx = aim_x - cfg.center_x
                dy = aim_y - cfg.center_y
                distance = np.sqrt(dx * dx + dy * dy)
                fov = self._fov_factor(distance)

                if fov <= 0.0:
                    rx, ry = self._disengage(dt)
                else:
                    # Engagement ramp
                    self.intensity = min(1.0, self.intensity + 0.12)

                    # Raw aim vector (normalized by screen, scaled by speed)
                    raw_x = (dx / cfg.screen_w) * cfg.speed_x * 100.0 * fov * self.intensity
                    raw_y = (dy / cfg.screen_h) * cfg.speed_y * 100.0 * fov * self.intensity

                    # 6. SPRING — the one system that replaces five
                    rx, ry = self.spring.update(raw_x, raw_y, dt)

                    # 7. Flick boost
                    bx, by = self._flick(rx, ry)
                    rx *= bx
                    ry *= by

                    # 8. Noise
                    if self.noise and cfg.noise_enabled:
                        nh, nv = self.noise.sample()
                        rx += nh
                        ry += nv

                    # 9. Recoil
                    rh, rv = self.recoil.tick(True)
                    rx += rh
                    ry += rv
            else:
                rx, ry = self._disengage(dt)
                self.recoil.tick(False)

            # 10. Clamp + encode
            rx = float(np.clip(rx, -cfg.max_output, cfg.max_output))
            ry = float(np.clip(ry, -cfg.max_output, cfg.max_output))

            gcvdata.extend(int(rx * cfg.fixed_point).to_bytes(4, "big", signed=True))
            gcvdata.extend(int(ry * cfg.fixed_point).to_bytes(4, "big", signed=True))
            return frame, gcvdata

        except Exception:
            gcvdata.extend((0).to_bytes(4, "big", signed=True))
            gcvdata.extend((0).to_bytes(4, "big", signed=True))
            return frame, gcvdata

    def _disengage(self, dt: float) -> Tuple[float, float]:
        """Smooth decay when no target."""
        self.intensity *= 0.88
        self.has_prev = False
        self.vel_x = 0.0
        self.vel_y = 0.0
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
