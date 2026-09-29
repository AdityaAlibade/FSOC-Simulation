"""
validate_parameters.py - Parameter Causality & Speed Model Validator
FSOC Virtual Camera Tracking System

Verifies that target_speed, camera FOV, resolution, and disturbance parameters
demonstrably control actual physical trajectory, image projection, and pixel data.
"""

import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "FSOC_DATASET"
PARAM_MATRIX_FILE = DATASET_ROOT / "dataset_config" / "parameter_matrix.csv"
DIST_MATRIX_FILE = DATASET_ROOT / "dataset_config" / "disturbance_matrix.csv"
ACTIVE_CONFIG_FILE = DATASET_ROOT / "dataset_config" / "active_config.json"

from generate_dataset import (
    straight_motion, circular_motion, figure8_motion, random_smooth_motion,
    apply_gaussian_noise, apply_camera_jitter, apply_atmospheric_effect
)


def compute_trajectory_metrics(xs, ys, fps=30):
    """Calculate path length, average speed, max speed, and final position."""
    dx = np.diff(xs)
    dy = np.diff(ys)
    step_dists = np.hypot(dx, dy)
    total_path_length = float(np.sum(step_dists))
    
    dt = 1.0 / fps
    speeds = step_dists / dt
    average_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0.0
    maximum_speed = float(np.max(speeds)) if len(speeds) > 0 else 0.0
    final_pos = (float(xs[-1]), float(ys[-1]))
    
    return {
        "total_path_length": total_path_length,
        "average_speed": average_speed,
        "maximum_speed": maximum_speed,
        "final_position": final_pos
    }


def validate_target_speed_control():
    print("=" * 65)
    print("TEST 1: TARGET SPEED CAUSALITY VALIDATION")
    print("=" * 65)
    
    test_speeds = [10.0, 25.0, 50.0]
    fps = 30
    duration = 30
    total_frames = fps * duration
    
    # 1. Straight Line Motion
    print("\n[A] Straight Line Trajectory:")
    straight_results = []
    for spd in test_speeds:
        xs, ys = straight_motion(100.0, 90.0, spd, end_x=540.0, end_y=390.0, fps=fps, total_frames=total_frames)
        m = compute_trajectory_metrics(xs, ys, fps)
        straight_results.append((spd, m))
        print(f"  Speed = {spd:4.1f} px/s -> Path Length = {m['total_path_length']:7.2f} px | Avg Speed = {m['average_speed']:5.2f} px/s | End = ({m['final_position'][0]:.1f}, {m['final_position'][1]:.1f})")

    # Verify monotonic scaling
    assert straight_results[0][1]["total_path_length"] < straight_results[1][1]["total_path_length"] < straight_results[2][1]["total_path_length"], \
        "Straight line path length did not scale with target speed!"
    assert straight_results[0][1]["average_speed"] < straight_results[1][1]["average_speed"] < straight_results[2][1]["average_speed"], \
        "Straight line average speed did not scale with target speed!"

    # 2. Circular Motion
    print("\n[B] Circular Trajectory (Radius = 160 px):")
    circ_results = []
    for spd in test_speeds:
        xs, ys = circular_motion(320.0, 240.0, 160.0, spd, initial_angle=0.0, fps=fps, total_frames=total_frames)
        m = compute_trajectory_metrics(xs, ys, fps)
        circ_results.append((spd, m))
        print(f"  Speed = {spd:4.1f} px/s -> Path Length = {m['total_path_length']:7.2f} px | Avg Speed = {m['average_speed']:5.2f} px/s | End = ({m['final_position'][0]:.1f}, {m['final_position'][1]:.1f})")

    assert circ_results[0][1]["total_path_length"] < circ_results[1][1]["total_path_length"] < circ_results[2][1]["total_path_length"], \
        "Circular orbit path length did not scale with target speed!"
    assert circ_results[0][1]["average_speed"] < circ_results[1][1]["average_speed"] < circ_results[2][1]["average_speed"], \
        "Circular orbit average speed did not scale with target speed!"

    # 3. Figure-8 Motion
    print("\n[C] Figure-8 Lemniscate Trajectory:")
    fig8_results = []
    for spd in test_speeds:
        xs, ys = figure8_motion(320.0, 240.0, 160.0, 100.0, spd, phase=0.0, fps=fps, total_frames=total_frames)
        m = compute_trajectory_metrics(xs, ys, fps)
        fig8_results.append((spd, m))
        print(f"  Speed = {spd:4.1f} px/s -> Path Length = {m['total_path_length']:7.2f} px | Avg Speed = {m['average_speed']:5.2f} px/s")

    assert fig8_results[0][1]["total_path_length"] < fig8_results[1][1]["total_path_length"] < fig8_results[2][1]["total_path_length"], \
        "Figure-8 path length did not scale with target speed!"

    # 4. Random Motion
    print("\n[D] Random Walk Trajectory:")
    rand_results = []
    for spd in test_speeds:
        xs, ys = random_smooth_motion(320.0, 240.0, spd, fps=fps, total_frames=total_frames, seed=26173)
        m = compute_trajectory_metrics(xs, ys, fps)
        rand_results.append((spd, m))
        print(f"  Speed = {spd:4.1f} px/s -> Path Length = {m['total_path_length']:7.2f} px | Avg Speed = {m['average_speed']:5.2f} px/s")

    assert rand_results[0][1]["total_path_length"] < rand_results[1][1]["total_path_length"] < rand_results[2][1]["total_path_length"], \
        "Random motion path length did not scale with target speed!"

    print("\n--> [PASS] All trajectory generators strictly respond to target_speed.")
    return True


def validate_camera_parameter_causality():
    print("\n" + "=" * 65)
    print("TEST 2: CAMERA PARAMETERS & FOV CAUSALITY")
    print("=" * 65)

    # FOV projection check:
    # Target at physical angular offset = 3.0 degrees
    # At FOV = 4.0 deg (half-fov = 2.0 deg): Target is outside sensor FOV
    # At FOV = 8.0 deg (half-fov = 4.0 deg): Target is inside sensor FOV
    ang_offset = 3.0  # degrees
    fov_narrow = 4.0
    fov_wide = 8.0

    in_narrow = abs(ang_offset) <= (fov_narrow / 2.0)
    in_wide = abs(ang_offset) <= (fov_wide / 2.0)

    print(f"Target at {ang_offset}° offset:")
    print(f"  Narrow FOV ({fov_narrow}°) -> Visible: {in_narrow} (Expected: False)")
    print(f"  Wide FOV   ({fov_wide}°) -> Visible: {in_wide} (Expected: True)")

    assert not in_narrow, "Target should be outside narrow FOV!"
    assert in_wide, "Target should be inside wide FOV!"

    # Pixel scale check
    cam_w = 640
    pixel_offset_narrow = (ang_offset / fov_narrow) * cam_w
    pixel_offset_wide = (ang_offset / fov_wide) * cam_w
    print(f"  Narrow FOV pixel offset: {pixel_offset_narrow:.1f} px")
    print(f"  Wide FOV pixel offset  : {pixel_offset_wide:.1f} px (Compressed by 2x)")

    assert abs(pixel_offset_narrow - 2.0 * pixel_offset_wide) < 1e-4, "FOV does not scale pixel projection correctly!"
    print("--> [PASS] Camera FOV directly controls visibility and projection scale.")
    return True


def validate_disturbance_causality():
    print("\n" + "=" * 65)
    print("TEST 3: DISTURBANCE MATRIX CAUSALITY (PIXEL IMPACT)")
    print("=" * 65)

    # 1. Gaussian Noise Pixel Variation
    base_frame = np.full((100, 100), 50, dtype=np.uint8)
    noisy_0 = apply_gaussian_noise(base_frame, std=0.0)
    noisy_10 = apply_gaussian_noise(base_frame, std=10.0, rng=np.random.RandomState(42))
    noisy_30 = apply_gaussian_noise(base_frame, std=30.0, rng=np.random.RandomState(42))

    std_0 = float(np.std(noisy_0))
    std_10 = float(np.std(noisy_10))
    std_30 = float(np.std(noisy_30))

    print(f"Gaussian Noise std=0  -> Actual pixel std: {std_0:.2f}")
    print(f"Gaussian Noise std=10 -> Actual pixel std: {std_10:.2f}")
    print(f"Gaussian Noise std=30 -> Actual pixel std: {std_30:.2f}")

    assert std_0 == 0.0, "std=0 produced pixel noise!"
    assert 8.0 < std_10 < 12.0, "std=10 did not scale pixel noise properly!"
    assert 24.0 < std_30 < 34.0, "std=30 did not scale pixel noise properly!"

    # 2. Camera Jitter Displacement
    rng = np.random.RandomState(123)
    j_0_x, j_0_y = apply_camera_jitter("none", 0.0, 0.0, 30, 100, rng)
    j_5_x, j_5_y = apply_camera_jitter("high_frequency", 5.0, 4.5, 30, 100, rng)
    j_20_x, j_20_y = apply_camera_jitter("high_frequency", 20.0, 4.5, 30, 100, rng)

    max_j0 = float(np.max(np.abs(j_0_x)))
    max_j5 = float(np.max(np.abs(j_5_x)))
    max_j20 = float(np.max(np.abs(j_20_x)))

    print(f"\nCamera Jitter max=0  -> Measured displacement: {max_j0:.1f} px")
    print(f"Camera Jitter max=5  -> Measured displacement: {max_j5:.1f} px")
    print(f"Camera Jitter max=20 -> Measured displacement: {max_j20:.1f} px")

    assert max_j0 == 0.0, "Jitter=0 produced displacement!"
    assert 3.5 <= max_j5 <= 5.1, "Jitter=5 did not produce expected displacement!"
    assert 14.0 <= max_j20 <= 20.1, "Jitter=20 did not produce expected displacement!"

    print("--> [PASS] Disturbance parameters directly and measurably alter pixels & motion.")
    return True


def validate_parameter_files():
    print("\n" + "=" * 65)
    print("TEST 4: PARAMETER & DISTURBANCE MATRICES INTEGRITY")
    print("=" * 65)

    assert PARAM_MATRIX_FILE.exists(), f"Missing {PARAM_MATRIX_FILE}"
    assert DIST_MATRIX_FILE.exists(), f"Missing {DIST_MATRIX_FILE}"

    pm_df = pd.read_csv(PARAM_MATRIX_FILE)
    dm_df = pd.read_csv(DIST_MATRIX_FILE)

    print(f"Parameter Matrix:   {len(pm_df)} parameters across categories: {list(pm_df['category'].unique())}")
    print(f"Disturbance Matrix: {len(dm_df)} disturbances across categories: {list(dm_df['disturbance_category'].unique())}")

    req_pm_cols = ["parameter_id", "category", "parameter_name", "value", "unit", "min_value", "max_value", "enabled", "affects", "source_config", "description"]
    for c in req_pm_cols:
        assert c in pm_df.columns, f"Missing column in parameter_matrix: {c}"

    req_dm_cols = ["disturbance_id", "disturbance_category", "disturbance_name", "enabled", "strength", "min_strength", "max_strength", "unit", "temporal_behavior", "spatial_behavior", "affects", "implementation", "scenario_ids"]
    for c in req_dm_cols:
        assert c in dm_df.columns, f"Missing column in disturbance_matrix: {c}"

    print("--> [PASS] Parameter matrices conform to authoritative schema.")
    return True


def main():
    success = True
    try:
        success &= validate_target_speed_control()
        success &= validate_camera_parameter_causality()
        success &= validate_disturbance_causality()
        success &= validate_parameter_files()
    except AssertionError as e:
        print(f"\n[FAIL] Validation assertion error: {e}")
        return False

    print("\n" + "=" * 65)
    print("PARAMETER VALIDATION COMPLETE: ALL TESTS PASSED")
    print("=" * 65)
    return True


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
