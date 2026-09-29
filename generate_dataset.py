"""
generate_dataset.py - Modular Synthetic Dataset Generator with True Parameter Causality
FSOC Virtual Camera Tracking System

Every parameter in parameter_matrix.csv and scenarios.csv strictly controls the simulation:
  - Target Speed: controls velocity, angular velocity, and total path length
  - Camera FOV: controls angular projection and target visibility
  - Resolution & FPS: controls frame dimensions, dt, and frame counts
  - Disturbances: noise sigma, fog density, light attenuation, and jitter amplitudes directly modify pixels and motion.
"""

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "FSOC_DATASET"
CONFIG_FILE = DATASET_ROOT / "dataset_config" / "config.json"
PARAM_MATRIX_FILE = DATASET_ROOT / "dataset_config" / "parameter_matrix.csv"
DIST_MATRIX_FILE = DATASET_ROOT / "dataset_config" / "disturbance_matrix.csv"
ACTIVE_CONFIG_FILE = DATASET_ROOT / "dataset_config" / "active_config.json"


def load_config(config_path=CONFIG_FILE):
    """Load dataset configuration from JSON file."""
    default_config = {
        "dataset_name": "FSOC Virtual Camera Tracking Dataset",
        "version": "1.0",
        "resolution": [640, 480],
        "fps": 30,
        "environment": [2000, 2000],
        "default_fov": [4.0, 3.0],
        "default_target_size": [10, 10],
        "max_pan_speed": 5.0,
        "max_tilt_speed": 5.0,
        "duration_per_scenario": 30,
        "random_seed": 26169,
    }
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            default_config.update(cfg)
            return default_config

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(default_config, f, indent=2)
    return default_config


def compute_config_hash(scenario_dict, master_config):
    """Generate deterministic SHA-256 fingerprint from scenario and camera parameters."""
    combined = {
        "scenario": scenario_dict,
        "resolution": master_config["resolution"],
        "fps": master_config["fps"],
        "fov": master_config["default_fov"],
        "duration": master_config["duration_per_scenario"],
        "seed": master_config["random_seed"]
    }
    raw = json.dumps(combined, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def generate_background(env_width=2000, env_height=2000, seed=26169):
    """Generate 2000x2000 grayscale optical space-observation background."""
    rng = np.random.RandomState(seed)
    canvas = np.full((env_height, env_width), 16.0, dtype=np.float32)
    read_noise = rng.normal(0.0, 1.2, (env_height, env_width)).astype(np.float32)
    canvas += read_noise

    num_stars = 1000
    star_x = rng.uniform(20, env_width - 20, num_stars)
    star_y = rng.uniform(20, env_height - 20, num_stars)
    star_brightness = np.clip(rng.exponential(scale=25.0, size=num_stars) + 15.0, 15.0, 140.0)
    star_sigma = rng.uniform(0.65, 1.2, num_stars)

    for sx, sy, sb, sig in zip(star_x, star_y, star_brightness, star_sigma):
        ix, iy = int(round(sx)), int(round(sy))
        rad = int(math.ceil(sig * 3.0))
        y_min, y_max = max(0, iy - rad), min(env_height, iy + rad + 1)
        x_min, x_max = max(0, ix - rad), min(env_width, ix + rad + 1)
        yy, xx = np.ogrid[y_min - sy : y_max - sy, x_min - sx : x_max - sx]
        psf = sb * np.exp(-(xx * xx + yy * yy) / (2.0 * sig * sig))
        canvas[y_min:y_max, x_min:x_max] += psf

    return np.clip(canvas, 0.0, 255.0).astype(np.uint8)


def generate_beacon(size=10, intensity=250):
    """Generate synthetic optical beacon."""
    return np.full((size, size), intensity, dtype=np.uint8)


# -------------------------------------------------------------
# TRAJECTORY MODELS WITH CAUSAL TARGET SPEED
# -------------------------------------------------------------

def straight_motion(start_x, start_y, target_speed, direction_angle=None, end_x=None, end_y=None, fps=30, total_frames=900):
    """
    Generate straight-line motion with speed S = target_speed.
    vx = S * cos(theta), vy = S * sin(theta)
    x(t) = x0 + vx * t,   y(t) = y0 + vy * t
    """
    xs = np.zeros(total_frames, dtype=np.float64)
    ys = np.zeros(total_frames, dtype=np.float64)
    dt = 1.0 / fps

    if end_x is not None and end_y is not None:
        dx = end_x - start_x
        dy = end_y - start_y
        dist = math.hypot(dx, dy)
        if dist > 1e-6:
            vx = target_speed * (dx / dist)
            vy = target_speed * (dy / dist)
        else:
            vx, vy = target_speed, 0.0
    else:
        angle_rad = math.radians(direction_angle if direction_angle is not None else 35.0)
        vx = target_speed * math.cos(angle_rad)
        vy = target_speed * math.sin(angle_rad)

    for f in range(total_frames):
        t = f * dt
        xs[f] = start_x + vx * t
        ys[f] = start_y + vy * t

    return xs, ys


def circular_motion(center_x, center_y, radius, target_speed, initial_angle=0.0, fps=30, total_frames=900):
    """
    Generate circular motion where angular velocity = target_speed / radius.
    v = r * angular_velocity  -->  omega = target_speed / radius
    x = cx + r * cos(theta), y = cy + r * sin(theta)
    """
    xs = np.zeros(total_frames, dtype=np.float64)
    ys = np.zeros(total_frames, dtype=np.float64)
    dt = 1.0 / fps
    angular_velocity = target_speed / max(radius, 1.0)

    for f in range(total_frames):
        t = f * dt
        angle = initial_angle + angular_velocity * t
        xs[f] = center_x + radius * math.cos(angle)
        ys[f] = center_y + radius * math.sin(angle)

    return xs, ys


def figure8_motion(center_x, center_y, amplitude_x, amplitude_y, target_speed, phase=0.0, fps=30, total_frames=900):
    """
    Generate Lissajous figure-8 motion with omega scaled by target_speed.
    Effective path perimeter L_cycle ~ 4 * hypot(amp_x, 2*amp_y) * 0.72.
    omega = (2 * pi * target_speed) / L_cycle
    """
    xs = np.zeros(total_frames, dtype=np.float64)
    ys = np.zeros(total_frames, dtype=np.float64)
    dt = 1.0 / fps

    L_cycle = 4.0 * math.hypot(amplitude_x, 2.0 * amplitude_y) * 0.72
    omega = (2.0 * math.pi * target_speed) / max(L_cycle, 1.0)

    for f in range(total_frames):
        t = f * dt
        xs[f] = center_x + amplitude_x * math.sin(omega * t)
        ys[f] = center_y + amplitude_y * math.sin(2.0 * omega * t + phase)

    return xs, ys


def random_smooth_motion(center_x, center_y, target_speed, fps=30, total_frames=900, seed=26169):
    """
    Generate continuous 2nd-order damped Langevin random walk with max speed = target_speed.
    """
    rng = np.random.RandomState(seed)
    xs = np.zeros(total_frames, dtype=np.float64)
    ys = np.zeros(total_frames, dtype=np.float64)
    x, y = center_x, center_y
    vx = rng.uniform(-1.0, 1.0) * (target_speed * 0.3)
    vy = rng.uniform(-1.0, 1.0) * (target_speed * 0.3)
    dt = 1.0 / fps
    spring = 2.2
    damping = 1.6
    noise_scale = float(target_speed) * 0.40
    max_spd = float(target_speed)

    for f in range(total_frames):
        xs[f] = x
        ys[f] = y
        ax = -spring * (x - center_x) - damping * vx + rng.normal(0.0, noise_scale)
        ay = -spring * (y - center_y) - damping * vy + rng.normal(0.0, noise_scale)
        vx += ax * dt
        vy += ay * dt
        spd = math.hypot(vx, vy)
        if spd > max_spd:
            vx = (vx / spd) * max_spd
            vy = (vy / spd) * max_spd
        x += vx * dt
        y += vy * dt
        x = np.clip(x, center_x - 160.0, center_x + 160.0)
        y = np.clip(y, center_y - 120.0, center_y + 120.0)

    return xs, ys


# -------------------------------------------------------------
# DISTURBANCE FUNCTIONS WITH PHYSICAL PARAMETER SCALING
# -------------------------------------------------------------

def apply_camera_jitter(jitter_type, max_jitter, frequency, fps, total_frames, rng):
    """Apply high-frequency terminal vibration jitter."""
    jx = np.zeros(total_frames, dtype=np.float64)
    jy = np.zeros(total_frames, dtype=np.float64)
    if jitter_type == "high_frequency" and max_jitter > 0:
        for f in range(total_frames):
            t = f / float(fps)
            x_val = (max_jitter * 0.65) * math.sin(2.0 * math.pi * frequency * t) + \
                    (max_jitter * 0.35) * math.cos(2.0 * math.pi * (frequency * 1.58) * t + 0.4) + \
                    rng.normal(0.0, max_jitter * 0.05)
            y_val = (max_jitter * 0.60) * math.cos(2.0 * math.pi * (frequency * 0.84) * t) + \
                    (max_jitter * 0.35) * math.sin(2.0 * math.pi * (frequency * 1.4) * t + 0.2) + \
                    rng.normal(0.0, max_jitter * 0.05)
            jx[f] = np.clip(x_val, -max_jitter, max_jitter)
            jy[f] = np.clip(y_val, -max_jitter, max_jitter)
    return jx, jy


def apply_platform_motion(motion_type, amplitude, frequency, fps, total_frames):
    """Apply low-frequency carrier platform motion sway."""
    px = np.zeros(total_frames, dtype=np.float64)
    py = np.zeros(total_frames, dtype=np.float64)
    if amplitude <= 0 or motion_type == "none":
        return px, py

    for f in range(total_frames):
        t = f / float(fps)
        if motion_type == "circular":
            px[f] = amplitude * math.cos(2.0 * math.pi * frequency * t)
            py[f] = (amplitude * 0.83) * math.sin(2.0 * math.pi * frequency * t)
        elif motion_type == "linear":
            px[f] = amplitude * math.sin(2.0 * math.pi * frequency * t)
            py[f] = (amplitude * 0.76) * math.cos(2.0 * math.pi * frequency * t)
        elif motion_type == "spiral":
            mod = 1.0 + 0.25 * math.sin(2.0 * math.pi * (frequency * 0.5) * t)
            px[f] = amplitude * mod * math.cos(2.0 * math.pi * frequency * t)
            py[f] = (amplitude * 0.78) * mod * math.sin(2.0 * math.pi * frequency * t)

    return px, py


def apply_gaussian_noise(frame, mean=0.0, std=15.0, rng=None):
    """Apply Gaussian noise with exact configured sigma."""
    if std <= 0:
        return frame
    if rng is None:
        rng = np.random.RandomState()
    noise = rng.normal(mean, std, frame.shape)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_salt_pepper_noise(frame, salt_prob=0.003, pepper_prob=0.003, rng=None):
    """Apply Salt-and-Pepper impulsive noise."""
    if salt_prob <= 0 and pepper_prob <= 0:
        return frame
    if rng is None:
        rng = np.random.RandomState()
    img = frame.copy()
    rand_matrix = rng.uniform(0.0, 1.0, frame.shape)
    img[rand_matrix < salt_prob] = 255
    img[rand_matrix > (1.0 - pepper_prob)] = 0
    return img


def apply_poisson_noise(frame, strength=0.8, rng=None):
    """Apply Poisson photon shot noise."""
    if strength <= 0:
        return frame
    if rng is None:
        rng = np.random.RandomState()
    img = frame.astype(np.float32)
    scaled = np.maximum(img, 0.0) * (1.0 / max(strength, 0.1))
    noisy = rng.poisson(scaled).astype(np.float32) * strength
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_atmospheric_effect(frame, condition="Clear", fog_strength=0.55, brightness_factor=1.0, rng=None):
    """Apply atmospheric attenuation and contrast degradation."""
    if condition == "Clear":
        return frame
    if rng is None:
        rng = np.random.RandomState()
    img = frame.astype(np.float32)

    if condition == "Haze":
        img = img * 0.82 + 22.0
    elif condition == "Fog":
        fog_blur = cv2.GaussianBlur(img, (11, 11), 3.0)
        img = img * (1.0 - fog_strength * 0.55) + fog_blur * (fog_strength * 0.45) + (45.0 * fog_strength)
    elif condition == "Rain":
        img = img * 0.88 + 10.0
        h, w = frame.shape[:2]
        num_streaks = 120
        streak_x = rng.randint(0, w, num_streaks)
        streak_y = rng.randint(0, h, num_streaks)
        length = rng.randint(12, 24, num_streaks)
        for sx, sy, l in zip(streak_x, streak_y, length):
            cv2.line(img, (sx, sy), (sx - 3, min(h - 1, sy + l)), float(rng.uniform(35.0, 70.0)), 1)
    elif condition == "Low Light":
        img = img * brightness_factor + 2.0

    return np.clip(img, 0, 255).astype(np.uint8)


def get_scenario_definitions(base_seed=26169):
    """Define the 8 standard baseline scenarios with causal parameters."""
    return [
        {
            "scenario_id": "SCN_001",
            "scenario_name": "Straight Line + Clear",
            "motion_type": "straight_line",
            "target_size": 10,
            "target_speed": 17.77,
            "noise_type": "none",
            "noise_strength": 0.0,
            "atmospheric_condition": "Clear",
            "lighting_condition": "Normal",
            "camera_jitter": "none",
            "camera_jitter_max": 0.0,
            "camera_jitter_freq": 0.0,
            "platform_motion": "none",
            "platform_motion_type": "none",
            "platform_amplitude": 0.0,
            "platform_frequency": 0.0,
            "target_initial_x": 100.0,
            "target_initial_y": 90.0,
            "end_x": 540.0,
            "end_y": 390.0,
            "random_seed": base_seed + 1,
            "loss_periods": [],
        },
        {
            "scenario_id": "SCN_002",
            "scenario_name": "Circular + Gaussian Noise",
            "motion_type": "circular",
            "target_size": 10,
            "target_speed": 67.02,
            "noise_type": "gaussian",
            "noise_strength": 15.0,
            "noise_mean": 0.0,
            "noise_std": 15.0,
            "atmospheric_condition": "Clear",
            "lighting_condition": "Normal",
            "camera_jitter": "none",
            "camera_jitter_max": 0.0,
            "camera_jitter_freq": 0.0,
            "platform_motion": "none",
            "platform_motion_type": "none",
            "platform_amplitude": 0.0,
            "platform_frequency": 0.0,
            "center_x": 320.0,
            "center_y": 240.0,
            "radius": 160.0,
            "initial_angle": 0.0,
            "target_initial_x": 480.0,
            "target_initial_y": 240.0,
            "random_seed": base_seed + 2,
            "loss_periods": [],
        },
        {
            "scenario_id": "SCN_003",
            "scenario_name": "Figure-8 + Camera Jitter",
            "motion_type": "figure_8",
            "target_size": 15,
            "target_speed": 55.40,
            "noise_type": "none",
            "noise_strength": 0.0,
            "atmospheric_condition": "Clear",
            "lighting_condition": "Normal",
            "camera_jitter": "high_frequency",
            "camera_jitter_max": 15.0,
            "camera_jitter_freq": 4.5,
            "platform_motion": "none",
            "platform_motion_type": "none",
            "platform_amplitude": 0.0,
            "platform_frequency": 0.0,
            "center_x": 320.0,
            "center_y": 240.0,
            "amplitude_x": 170.0,
            "amplitude_y": 110.0,
            "phase": 0.0,
            "target_initial_x": 320.0,
            "target_initial_y": 240.0,
            "random_seed": base_seed + 3,
            "loss_periods": [],
        },
        {
            "scenario_id": "SCN_004",
            "scenario_name": "Random Motion + Platform Motion",
            "motion_type": "random",
            "target_size": 10,
            "target_speed": 34.82,
            "noise_type": "none",
            "noise_strength": 0.0,
            "atmospheric_condition": "Clear",
            "lighting_condition": "Normal",
            "camera_jitter": "none",
            "camera_jitter_max": 0.0,
            "camera_jitter_freq": 0.0,
            "platform_motion": "circular",
            "platform_motion_type": "circular",
            "platform_amplitude": 12.0,
            "platform_frequency": 0.15,
            "center_x": 320.0,
            "center_y": 240.0,
            "target_initial_x": 320.0,
            "target_initial_y": 240.0,
            "random_seed": base_seed + 4,
            "loss_periods": [],
        },
        {
            "scenario_id": "SCN_005",
            "scenario_name": "Straight Line + Fog",
            "motion_type": "straight_line",
            "target_size": 12,
            "target_speed": 15.92,
            "noise_type": "none",
            "noise_strength": 0.0,
            "atmospheric_condition": "Fog",
            "fog_strength": 0.55,
            "lighting_condition": "Diffuse",
            "camera_jitter": "none",
            "camera_jitter_max": 0.0,
            "camera_jitter_freq": 0.0,
            "platform_motion": "none",
            "platform_motion_type": "none",
            "platform_amplitude": 0.0,
            "platform_frequency": 0.0,
            "target_initial_x": 520.0,
            "target_initial_y": 110.0,
            "end_x": 120.0,
            "end_y": 370.0,
            "random_seed": base_seed + 5,
            "loss_periods": [],
        },
        {
            "scenario_id": "SCN_006",
            "scenario_name": "Circular + Low Light + Noise",
            "motion_type": "circular",
            "target_size": 8,
            "target_speed": 43.98,
            "noise_type": "gaussian",
            "noise_strength": 16.0,
            "noise_mean": 0.0,
            "noise_std": 16.0,
            "atmospheric_condition": "Low Light",
            "brightness_factor": 0.32,
            "lighting_condition": "Low Light",
            "camera_jitter": "none",
            "camera_jitter_max": 0.0,
            "camera_jitter_freq": 0.0,
            "platform_motion": "none",
            "platform_motion_type": "none",
            "platform_amplitude": 0.0,
            "platform_frequency": 0.0,
            "center_x": 320.0,
            "center_y": 240.0,
            "radius": 140.0,
            "initial_angle": math.pi / 4.0,
            "target_initial_x": 320.0 + 140.0 * math.cos(math.pi / 4.0),
            "target_initial_y": 240.0 + 140.0 * math.sin(math.pi / 4.0),
            "random_seed": base_seed + 6,
            "loss_periods": [],
        },
        {
            "scenario_id": "SCN_007",
            "scenario_name": "Figure-8 + Camera Jitter + Platform Motion",
            "motion_type": "figure_8",
            "target_size": 15,
            "target_speed": 48.65,
            "noise_type": "salt_and_pepper",
            "noise_strength": 0.006,
            "salt_probability": 0.003,
            "pepper_probability": 0.003,
            "atmospheric_condition": "Haze",
            "lighting_condition": "Normal",
            "camera_jitter": "high_frequency",
            "camera_jitter_max": 15.0,
            "camera_jitter_freq": 4.5,
            "platform_motion": "linear",
            "platform_motion_type": "linear",
            "platform_amplitude": 8.5,
            "platform_frequency": 0.12,
            "center_x": 320.0,
            "center_y": 240.0,
            "amplitude_x": 150.0,
            "amplitude_y": 95.0,
            "phase": 0.0,
            "target_initial_x": 320.0,
            "target_initial_y": 240.0,
            "random_seed": base_seed + 7,
            "loss_periods": [],
        },
        {
            "scenario_id": "SCN_008",
            "scenario_name": "Random Motion + Combined Disturbances + Target Loss/Reacquisition",
            "motion_type": "random",
            "target_size": 10,
            "target_speed": 36.14,
            "noise_type": "gaussian",
            "noise_strength": 18.0,
            "noise_mean": 0.0,
            "noise_std": 18.0,
            "atmospheric_condition": "Rain",
            "lighting_condition": "Variable",
            "camera_jitter": "high_frequency",
            "camera_jitter_max": 15.0,
            "camera_jitter_freq": 4.5,
            "platform_motion": "spiral",
            "platform_motion_type": "spiral",
            "platform_amplitude": 9.0,
            "platform_frequency": 0.10,
            "center_x": 320.0,
            "center_y": 240.0,
            "target_initial_x": 320.0,
            "target_initial_y": 240.0,
            "random_seed": base_seed + 8,
            "loss_periods": [(240, 300), (600, 660)],
        },
    ]


def generate_scenario(scn, config, sky_canvas):
    """
    Generate frames, MP4 video, and ground-truth metadata records for one scenario.
    Dynamically respects camera FOV, resolution, FPS, speed, and disturbances.
    """
    scn_id = scn["scenario_id"]
    fps = config["fps"]
    total_frames = int(config["duration_per_scenario"] * fps)
    cam_w, cam_h = config["resolution"]
    env_w, env_h = config["environment"]
    fov_h = float(config.get("default_fov", [4.0, 3.0])[0])
    fov_v = float(config.get("default_fov", [4.0, 3.0])[1])

    # Dynamic center calculation
    center_x = cam_w / 2.0
    center_y = cam_h / 2.0

    # Compute trajectory using actual target_speed
    m_type = scn["motion_type"]
    speed = float(scn.get("target_speed", 30.0))

    if m_type == "straight_line":
        base_xs, base_ys = straight_motion(
            scn["target_initial_x"], scn["target_initial_y"],
            speed, end_x=scn.get("end_x"), end_y=scn.get("end_y"),
            fps=fps, total_frames=total_frames
        )
    elif m_type == "circular":
        base_xs, base_ys = circular_motion(
            center_x, center_y, scn.get("radius", 160.0),
            speed, initial_angle=scn.get("initial_angle", 0.0),
            fps=fps, total_frames=total_frames
        )
    elif m_type == "figure_8":
        base_xs, base_ys = figure8_motion(
            center_x, center_y, scn.get("amplitude_x", 160.0), scn.get("amplitude_y", 100.0),
            speed, phase=scn.get("phase", 0.0),
            fps=fps, total_frames=total_frames
        )
    elif m_type == "random":
        base_xs, base_ys = random_smooth_motion(
            center_x, center_y, speed,
            fps=fps, total_frames=total_frames, seed=scn["random_seed"]
        )

    # Compute disturbances
    dist_rng = np.random.RandomState(scn["random_seed"] + 100)
    jitter_xs, jitter_ys = apply_camera_jitter(
        scn.get("camera_jitter", "none"),
        float(scn.get("camera_jitter_max", 15.0)),
        float(scn.get("camera_jitter_freq", 4.5)),
        fps, total_frames, dist_rng
    )
    plat_xs, plat_ys = apply_platform_motion(
        scn.get("platform_motion_type", "none"),
        float(scn.get("platform_amplitude", 12.0)),
        float(scn.get("platform_frequency", 0.15)),
        fps, total_frames
    )

    # Occlusion loss periods
    occluded_frames = set()
    for start_f, end_f in scn.get("loss_periods", []):
        for f in range(start_f, end_f + 1):
            occluded_frames.add(f)

    target_size = int(scn.get("target_size", 10))
    beacon = generate_beacon(target_size, 250)
    if scn.get("atmospheric_condition") == "Low Light":
        beacon = generate_beacon(target_size, 80)
    elif scn.get("atmospheric_condition") == "Fog":
        beacon = generate_beacon(target_size, 180)
    elif scn.get("atmospheric_condition") == "Haze":
        beacon = generate_beacon(target_size, 220)

    frames_dir = DATASET_ROOT / "frames" / scn_id
    video_path = str(DATASET_ROOT / "videos" / f"{scn_id}.mp4")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_f in frames_dir.glob("*.png"):
        try:
            old_f.unlink()
        except Exception:
            pass

    video_writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (cam_w, cam_h),
        isColor=True,
    )

    gt_records = []
    dist_records = []
    frame_rng = np.random.RandomState(scn["random_seed"] + 500)
    env_cx, env_cy = env_w // 2, env_h // 2

    # Scale factors: baseline 4° x 3° corresponds to cam_w x cam_h
    # Changing FOV changes the pixel scale: scale = (cam_w / fov_h)
    fov_scale_x = (4.0 / fov_h) if fov_h > 0 else 1.0
    fov_scale_y = (3.0 / fov_v) if fov_v > 0 else 1.0

    for i in range(total_frames):
        frame_id = i + 1
        timestamp = round(i / float(fps), 4)
        jx, jy = jitter_xs[i], jitter_ys[i]
        px, py = plat_xs[i], plat_ys[i]

        # Base background frame
        cam_x0 = int(round(env_cx - (cam_w // 2) - jx - px))
        cam_y0 = int(round(env_cy - (cam_h // 2) - jy - py))
        cam_x0 = np.clip(cam_x0, 0, env_w - cam_w)
        cam_y0 = np.clip(cam_y0, 0, env_h - cam_h)
        cam_frame = sky_canvas[cam_y0 : cam_y0 + cam_h, cam_x0 : cam_x0 + cam_w].copy()

        # Target center in camera frame with FOV angular scaling
        target_center_x = center_x + (base_xs[i] - center_x) * fov_scale_x + jx + px
        target_center_y = center_y + (base_ys[i] - center_y) * fov_scale_y + jy + py

        # FOV Visibility Check
        in_sensor_fov = (0.0 <= target_center_x < cam_w) and (0.0 <= target_center_y < cam_h)
        not_occluded = (frame_id not in occluded_frames)
        is_visible = bool(in_sensor_fov and not_occluded)

        gt_w = target_size
        gt_h = target_size
        gt_x = int(round(target_center_x - gt_w / 2.0))
        gt_y = int(round(target_center_y - gt_h / 2.0))

        # Clamp inside frame for ground truth bounding box
        gt_x = max(0, min(cam_w - gt_w, gt_x))
        gt_y = max(0, min(cam_h - gt_h, gt_y))
        gt_cx = round(gt_x + gt_w / 2.0, 2)
        gt_cy = round(gt_y + gt_h / 2.0, 2)

        # Render beacon if visible
        if is_visible:
            cam_frame[gt_y : gt_y + gt_h, gt_x : gt_x + gt_w] = beacon

        # Apply atmospheric and lighting effects
        fog_str = float(scn.get("fog_strength", 0.55))
        bright_fac = float(scn.get("brightness_factor", 0.32))
        cam_frame = apply_atmospheric_effect(
            cam_frame, scn.get("atmospheric_condition", "Clear"),
            fog_strength=fog_str, brightness_factor=bright_fac, rng=frame_rng
        )

        # Apply noise disturbance
        n_type = scn.get("noise_type", "none")
        if n_type == "gaussian":
            cam_frame = apply_gaussian_noise(
                cam_frame, scn.get("noise_mean", 0.0), float(scn.get("noise_strength", 15.0)), frame_rng
            )
        elif n_type == "salt_and_pepper":
            cam_frame = apply_salt_pepper_noise(
                cam_frame, float(scn.get("salt_probability", 0.003)), float(scn.get("pepper_probability", 0.003)), frame_rng
            )
        elif n_type == "poisson":
            cam_frame = apply_poisson_noise(cam_frame, float(scn.get("noise_strength", 0.8)), frame_rng)

        # Write frame to PNG and MP4
        frame_filename = f"frame_{frame_id:06d}.png"
        cv2.imwrite(str(frames_dir / frame_filename), cam_frame)
        bgr_frame = cv2.cvtColor(cam_frame, cv2.COLOR_GRAY2BGR)
        video_writer.write(bgr_frame)

        gt_records.append({
            "scenario_id": scn_id,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "gt_width": gt_w,
            "gt_height": gt_h,
            "gt_center_x": gt_cx,
            "gt_center_y": gt_cy,
            "visible": is_visible,
        })

        dist_records.append({
            "scenario_id": scn_id,
            "frame_id": frame_id,
            "noise_type": scn.get("noise_type", "none"),
            "noise_strength": scn.get("noise_strength", 0.0),
            "camera_jitter_x": round(jx, 4),
            "camera_jitter_y": round(jy, 4),
            "platform_motion_type": scn.get("platform_motion_type", "none"),
            "platform_motion_x": round(px, 4),
            "platform_motion_y": round(py, 4),
            "atmospheric_condition": scn.get("atmospheric_condition", "Clear"),
            "lighting_condition": scn.get("lighting_condition", "Normal"),
            "target_visible": is_visible,
        })

    video_writer.release()

    # Save Ground Truth CSV
    gt_df = pd.DataFrame(gt_records)
    gt_csv_path = DATASET_ROOT / "ground_truth" / f"{scn_id}.csv"
    gt_df.to_csv(gt_csv_path, index=False)

    return dist_records


def main():
    print("=" * 60)
    print("FSOC SYNTHETIC DATASET GENERATION (PARAM-CAUSAL ENGINE)")
    print("=" * 60)

    config = load_config()
    scenarios = get_scenario_definitions(config["random_seed"])

    # Ensure output directories
    for sub in ["videos", "frames", "ground_truth", "scenarios", "disturbances", "benchmarks", "dataset_config"]:
        (DATASET_ROOT / sub).mkdir(parents=True, exist_ok=True)

    print("Generating 2000x2000 celestial background...")
    sky_canvas = generate_background(
        config["environment"][0], config["environment"][1], config["random_seed"]
    )

    all_disturbances = []
    active_configs = {}

    multi_only = "--multi-only" in sys.argv

    if not multi_only:
        for idx, scn in enumerate(scenarios, 1):
            print(f"[{idx}/8] Generating {scn['scenario_id']}: {scn['scenario_name']} (Speed: {scn.get('target_speed')} px/s)...")
            dist_records = generate_scenario(scn, config, sky_canvas)
            all_disturbances.extend(dist_records)

            # Compute hash and fingerprint
            h = compute_config_hash(scn, config)
            active_configs[scn["scenario_id"]] = {
                "config_hash": h,
                "target_speed": scn.get("target_speed"),
                "motion_type": scn.get("motion_type"),
                "target_size": scn.get("target_size"),
                "fov": config.get("default_fov"),
                "resolution": config.get("resolution"),
                "fps": config.get("fps"),
                "disturbances": {
                    "noise": scn.get("noise_type"),
                    "noise_strength": scn.get("noise_strength"),
                    "atmospheric": scn.get("atmospheric_condition"),
                    "jitter": scn.get("camera_jitter"),
                    "platform": scn.get("platform_motion_type")
                }
            }

        # Save all disturbances CSV
        dist_df = pd.DataFrame(all_disturbances)
        dist_df.to_csv(DATASET_ROOT / "disturbances" / "disturbances.csv", index=False)

    # Save active_config.json
    active_payload = {
        "dataset_name": config["dataset_name"],
        "generation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fps": config["fps"],
        "resolution": config["resolution"],
        "fov": config["default_fov"],
        "scenarios": active_configs
    }
    with open(ACTIVE_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(active_payload, f, indent=2)

    # Save scenarios.csv metadata
    scn_rows = []
    for scn in scenarios:
        scn_rows.append({
            "scenario_id": scn["scenario_id"],
            "scenario_name": scn["scenario_name"],
            "motion_type": scn["motion_type"],
            "duration_seconds": config["duration_per_scenario"],
            "fps": config["fps"],
            "total_frames": config["duration_per_scenario"] * config["fps"],
            "environment_width": config["environment"][0],
            "environment_height": config["environment"][1],
            "camera_width": config["resolution"][0],
            "camera_height": config["resolution"][1],
            "fov_horizontal": config["default_fov"][0],
            "fov_vertical": config["default_fov"][1],
            "target_size": scn["target_size"],
            "target_initial_x": scn["target_initial_x"],
            "target_initial_y": scn["target_initial_y"],
            "target_speed": scn["target_speed"],
            "max_pan_speed": config["max_pan_speed"],
            "max_tilt_speed": config["max_tilt_speed"],
            "noise_type": scn["noise_type"],
            "atmospheric_condition": scn["atmospheric_condition"],
            "camera_jitter": scn["camera_jitter"],
            "platform_motion": scn["platform_motion"],
            "random_seed": scn["random_seed"],
        })
    pd.DataFrame(scn_rows).to_csv(DATASET_ROOT / "scenarios" / "scenarios.csv", index=False)

    # -------------------------------------------------------------
    # MULTI-TARGET SCENARIOS (SCN_009 to SCN_012)
    # -------------------------------------------------------------
    multi_scenarios_file = DATASET_ROOT / "scenarios" / "multi_target_scenarios.csv"
    if multi_scenarios_file.exists():
        print("\n" + "=" * 60)
        print("GENERATING MULTI-TARGET SCENARIOS (SCN_009 to SCN_012)")
        print("=" * 60)
        multi_df = pd.read_csv(multi_scenarios_file)

        multi_meta = {
            "SCN_009": {
                "scenario_name": "Two Moving Targets + Clear",
                "atmospheric_condition": "Clear",
                "noise_type": "none",
                "camera_jitter": "none",
                "platform_motion": "none",
                "random_seed": config["random_seed"] + 90,
            },
            "SCN_010": {
                "scenario_name": "Three Moving Targets + Gaussian Noise",
                "atmospheric_condition": "Clear",
                "noise_type": "gaussian",
                "noise_strength": 15.0,
                "camera_jitter": "none",
                "platform_motion": "none",
                "random_seed": config["random_seed"] + 100,
            },
            "SCN_011": {
                "scenario_name": "Multiple Targets + Camera Jitter + Platform Motion",
                "atmospheric_condition": "Clear",
                "noise_type": "gaussian",
                "noise_strength": 6.0,
                "camera_jitter": "high_frequency",
                "camera_jitter_max": 8.0,
                "camera_jitter_freq": 3.0,
                "platform_motion": "circular",
                "platform_amplitude": 7.0,
                "platform_frequency": 0.15,
                "random_seed": config["random_seed"] + 110,
            },
            "SCN_012": {
                "scenario_name": "Multiple Targets + Target Loss + Reacquisition",
                "atmospheric_condition": "Clear",
                "noise_type": "gaussian",
                "noise_strength": 8.0,
                "camera_jitter": "low_frequency",
                "camera_jitter_max": 4.0,
                "camera_jitter_freq": 1.5,
                "platform_motion": "none",
                "loss_periods": [(240, 300), (600, 660)],
                "random_seed": config["random_seed"] + 120,
            },
        }

        for m_scn_id in ["SCN_009", "SCN_010", "SCN_011", "SCN_012"]:
            m_targets = multi_df[multi_df["scenario_id"] == m_scn_id]
            if len(m_targets) == 0:
                continue
            meta = multi_meta.get(m_scn_id, {})
            print(f"Generating {m_scn_id}: {meta.get('scenario_name', m_scn_id)} ({len(m_targets)} targets)...")
            generate_multi_target_scenario(m_scn_id, meta.get("scenario_name", m_scn_id), m_targets, meta, config, sky_canvas)

    print("=" * 60)
    print("DATASET GENERATION COMPLETE & PARAMETER FINGERPRINT RECORDED")
    print(f"Fingerprint File: {ACTIVE_CONFIG_FILE}")
    print("=" * 60)


def generate_multi_target_scenario(scn_id, scn_name, targets_df, scn_disturbances, config, sky_canvas):
    """
    Generate frames, MP4 video, and multi-target ground truth for a multi-target scenario.
    """
    fps = config["fps"]
    total_frames = int(config["duration_per_scenario"] * fps)
    cam_w, cam_h = config["resolution"]
    env_w, env_h = config["environment"]
    fov_h = float(config.get("default_fov", [4.0, 3.0])[0])
    fov_v = float(config.get("default_fov", [4.0, 3.0])[1])

    center_x = cam_w / 2.0
    center_y = cam_h / 2.0
    env_cx, env_cy = env_w // 2, env_h // 2

    fov_scale_x = (4.0 / fov_h) if fov_h > 0 else 1.0
    fov_scale_y = (3.0 / fov_v) if fov_v > 0 else 1.0

    # Disturbances setup
    frame_rng = np.random.RandomState(scn_disturbances.get("random_seed", 500) + 12)
    j_type = scn_disturbances.get("camera_jitter", "none")
    j_max = float(scn_disturbances.get("camera_jitter_max", 0.0))
    j_freq = float(scn_disturbances.get("camera_jitter_freq", 1.0))
    jitter_xs, jitter_ys = apply_camera_jitter(j_type, j_max, j_freq, fps, total_frames, frame_rng)

    p_type = scn_disturbances.get("platform_motion", "none")
    p_amp = float(scn_disturbances.get("platform_amplitude", 0.0))
    p_freq = float(scn_disturbances.get("platform_frequency", 0.0))
    plat_xs, plat_ys = apply_platform_motion(p_type, p_amp, p_freq, fps, total_frames)

    # Pre-generate trajectory for each target
    target_trajectories = {}
    for _, t in targets_df.iterrows():
        tid = t["target_id"]
        m_type = str(t["motion_type"]).lower()
        speed = float(t["target_speed"])
        t_init_x = float(t["initial_x"])
        t_init_y = float(t["initial_y"])
        t_rad = float(t.get("radius", 140.0)) if not pd.isna(t.get("radius")) else 140.0
        t_dir = float(t.get("direction", 0.0)) if not pd.isna(t.get("direction")) else 0.0
        t_phase = float(t.get("initial_phase", 0.0)) if not pd.isna(t.get("initial_phase")) else 0.0

        if m_type == "circular":
            xs, ys = circular_motion(center_x, center_y, t_rad, speed, initial_angle=t_phase, fps=fps, total_frames=total_frames)
        elif m_type == "figure_8":
            xs, ys = figure8_motion(center_x, center_y, t_rad, t_rad * 0.55, speed, phase=t_phase, fps=fps, total_frames=total_frames)
        elif m_type == "random":
            xs, ys = random_smooth_motion(t_init_x, t_init_y, speed, fps=fps, total_frames=total_frames, seed=scn_disturbances.get("random_seed", 999) + int(tid[-1]))
        else:
            xs, ys = straight_motion(t_init_x, t_init_y, speed, direction_angle=t_dir, fps=fps, total_frames=total_frames)
        
        target_trajectories[tid] = (xs, ys)

    # Occlusion loss periods for communication target in SCN_012
    comm_loss_periods = scn_disturbances.get("loss_periods", [])
    comm_occluded_frames = set()
    for start_f, end_f in comm_loss_periods:
        for f in range(start_f, end_f + 1):
            comm_occluded_frames.add(f)

    frames_dir = DATASET_ROOT / "frames" / scn_id
    video_path = str(DATASET_ROOT / "videos" / f"{scn_id}.mp4")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_f in frames_dir.glob("*.png"):
        try: old_f.unlink()
        except Exception: pass

    video_writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (cam_w, cam_h),
        isColor=True,
    )

    gt_records = []
    frame_rng = np.random.RandomState(scn_disturbances.get("random_seed", 500) + 12)

    for i in range(total_frames):
        frame_id = i + 1
        timestamp = round(i / float(fps), 4)
        jx, jy = jitter_xs[i], jitter_ys[i]
        px, py = plat_xs[i], plat_ys[i]

        cam_x0 = int(round(env_cx - (cam_w // 2) - jx - px))
        cam_y0 = int(round(env_cy - (cam_h // 2) - jy - py))
        cam_x0 = np.clip(cam_x0, 0, env_w - cam_w)
        cam_y0 = np.clip(cam_y0, 0, env_h - cam_h)
        cam_frame = sky_canvas[cam_y0 : cam_y0 + cam_h, cam_x0 : cam_x0 + cam_w].copy()

        # Render each target
        for _, t in targets_df.iterrows():
            tid = t["target_id"]
            role = t["target_role"]
            tsize = int(t.get("target_size", 10))
            xs, ys = target_trajectories[tid]

            tx_world = xs[i]
            ty_world = ys[i]

            tcx = center_x + (tx_world - center_x) * fov_scale_x + jx + px
            tcy = center_y + (ty_world - center_y) * fov_scale_y + jy + py

            in_sensor_fov = (0.0 <= tcx < cam_w) and (0.0 <= tcy < cam_h)
            not_occluded = True
            if role == "COMMUNICATION_TARGET" and frame_id in comm_occluded_frames:
                not_occluded = False

            is_visible = bool(in_sensor_fov and not_occluded)

            gw, gh = tsize, tsize
            gx = int(round(tcx - gw / 2.0))
            gy = int(round(tcy - gh / 2.0))
            gx = max(0, min(cam_w - gw, gx))
            gy = max(0, min(cam_h - gh, gy))
            gcx = round(gx + gw / 2.0, 2)
            gcy = round(gy + gh / 2.0, 2)

            if is_visible:
                cam_frame[gy : gy + gh, gx : gx + gw] = generate_beacon(gw, 250)

            gt_records.append({
                "scenario_id": scn_id,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "target_id": tid,
                "target_role": role,
                "gt_x": gx,
                "gt_y": gy,
                "gt_width": gw,
                "gt_height": gh,
                "gt_center_x": gcx,
                "gt_center_y": gcy,
                "visible": is_visible,
                "motion_type": t["motion_type"],
                "target_speed": float(t["target_speed"]),
            })

        # Apply atmospheric and lighting effects
        cam_frame = apply_atmospheric_effect(
            cam_frame, scn_disturbances.get("atmospheric_condition", "Clear"),
            fog_strength=float(scn_disturbances.get("fog_strength", 0.0)),
            brightness_factor=float(scn_disturbances.get("brightness_factor", 1.0)),
            rng=frame_rng
        )

        # Apply noise disturbance
        n_type = scn_disturbances.get("noise_type", "none")
        if n_type == "gaussian":
            cam_frame = apply_gaussian_noise(
                cam_frame, 0.0, float(scn_disturbances.get("noise_strength", 15.0)), frame_rng
            )

        frame_filename = f"frame_{frame_id:06d}.png"
        cv2.imwrite(str(frames_dir / frame_filename), cam_frame)
        bgr_frame = cv2.cvtColor(cam_frame, cv2.COLOR_GRAY2BGR)
        video_writer.write(bgr_frame)

    video_writer.release()
    gt_df = pd.DataFrame(gt_records)
    gt_csv_path = DATASET_ROOT / "ground_truth" / f"{scn_id}.csv"
    gt_df.to_csv(gt_csv_path, index=False)


if __name__ == "__main__":
    main()
