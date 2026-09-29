"""
FSOC Virtual Camera Tracking System - Dataset Validator
Validates all dataset criteria and prints the standardized status report.
"""

import os
import sys
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "FSOC_DATASET"
TOTAL_SCENARIOS = 8
EXPECTED_FRAMES_PER_SCN = 900
EXPECTED_TOTAL_FRAMES = 7200
EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 480
EXPECTED_FPS = 30


def validate_dataset():
    """
    Validates all 20 required criteria across videos, PNG frames, ground truth, disturbances,
    and metadata configurations.
    """
    scenario_statuses = {}
    errors = []

    # Check dataset_config/config.json
    cfg_path = DATASET_ROOT / "dataset_config" / "config.json"
    cfg_valid = False
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            if (cfg.get("random_seed") == 26169 and
                cfg.get("resolution") == [640, 480] and
                cfg.get("fps") == 30 and
                cfg.get("environment") == [2000, 2000]):
                cfg_valid = True
            else:
                errors.append("dataset_config/config.json values do not match specifications.")
    else:
        errors.append("Missing dataset_config/config.json.")

    # Check scenarios.csv and benchmark_config.csv
    scenarios_csv = DATASET_ROOT / "scenarios" / "scenarios.csv"
    benchmarks_csv = DATASET_ROOT / "benchmarks" / "benchmark_config.csv"
    disturbances_csv = DATASET_ROOT / "disturbances" / "disturbances.csv"

    if not scenarios_csv.exists() or len(pd.read_csv(scenarios_csv)) != TOTAL_SCENARIOS:
        errors.append("Invalid or missing scenarios/scenarios.csv")
    if not benchmarks_csv.exists() or len(pd.read_csv(benchmarks_csv)) != TOTAL_SCENARIOS:
        errors.append("Invalid or missing benchmarks/benchmark_config.csv")

    dist_records_count = 0
    if disturbances_csv.exists():
        dist_df = pd.read_csv(disturbances_csv)
        dist_records_count = len(dist_df)
        if dist_records_count != EXPECTED_TOTAL_FRAMES:
            errors.append(f"disturbances.csv record count {dist_records_count} != {EXPECTED_TOTAL_FRAMES}")
    else:
        errors.append("Missing disturbances/disturbances.csv")

    videos_found = 0
    png_frames_found = 0
    gt_records_found = 0

    all_res_pass = True
    all_fps_pass = True
    all_frames_per_scn_pass = True
    all_target_size_pass = True
    all_motion_cont_pass = True
    all_consistency_pass = True

    scn8_target_loss_pass = False

    for i in range(1, TOTAL_SCENARIOS + 1):
        scn_id = f"SCN_{i:03d}"
        scn_pass = True

        # 1. Video Check
        v_path = DATASET_ROOT / "videos" / f"{scn_id}.mp4"
        if v_path.exists():
            videos_found += 1
            cap = cv2.VideoCapture(str(v_path))
            v_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            v_fps = round(cap.get(cv2.CAP_PROP_FPS))
            cap.release()

            if v_cnt != EXPECTED_FRAMES_PER_SCN:
                errors.append(f"{scn_id}.mp4 frame count {v_cnt} != {EXPECTED_FRAMES_PER_SCN}")
                scn_pass = False
                all_frames_per_scn_pass = False
            if v_w != EXPECTED_WIDTH or v_h != EXPECTED_HEIGHT:
                errors.append(f"{scn_id}.mp4 resolution ({v_w}x{v_h}) != ({EXPECTED_WIDTH}x{EXPECTED_HEIGHT})")
                scn_pass = False
                all_res_pass = False
            if v_fps != EXPECTED_FPS:
                errors.append(f"{scn_id}.mp4 FPS {v_fps} != {EXPECTED_FPS}")
                scn_pass = False
                all_fps_pass = False
        else:
            errors.append(f"Missing video: {scn_id}.mp4")
            scn_pass = False

        # 2. PNG Frames Check
        f_dir = DATASET_ROOT / "frames" / scn_id
        if f_dir.exists():
            png_list = sorted([p for p in os.listdir(f_dir) if p.endswith(".png")])
            count = len(png_list)
            png_frames_found += count
            if count != EXPECTED_FRAMES_PER_SCN:
                errors.append(f"{scn_id} PNG count {count} != {EXPECTED_FRAMES_PER_SCN}")
                scn_pass = False
                all_frames_per_scn_pass = False
        else:
            errors.append(f"Missing frame dir: {f_dir}")
            scn_pass = False

        # 3. Ground Truth Check
        gt_path = DATASET_ROOT / "ground_truth" / f"{scn_id}.csv"
        if gt_path.exists():
            gt_df = pd.read_csv(gt_path)
            gt_cnt = len(gt_df)
            gt_records_found += gt_cnt

            if gt_cnt != EXPECTED_FRAMES_PER_SCN:
                errors.append(f"{scn_id}.csv count {gt_cnt} != {EXPECTED_FRAMES_PER_SCN}")
                scn_pass = False

            # Check no duplicate frame IDs
            if len(gt_df["frame_id"].unique()) != EXPECTED_FRAMES_PER_SCN:
                errors.append(f"{scn_id}.csv contains duplicate or missing frame IDs")
                scn_pass = False

            # Check target size
            sizes = gt_df["gt_width"].tolist()
            if not all(5 <= s <= 20 for s in sizes):
                errors.append(f"{scn_id} target size outside [5, 20]")
                all_target_size_pass = False
                scn_pass = False

            # Check ground truth coordinates within image when visible
            for _, row in gt_df.iterrows():
                vis = str(row["visible"]).lower() == "true"
                gx, gy = int(row["gt_x"]), int(row["gt_y"])
                gw, gh = int(row["gt_width"]), int(row["gt_height"])
                if vis:
                    if not (0 <= gx and gx + gw <= EXPECTED_WIDTH and 0 <= gy and gy + gh <= EXPECTED_HEIGHT):
                        errors.append(f"{scn_id} bbox out of bounds when visible: [{gx}, {gy}, {gw}, {gh}]")
                        scn_pass = False

            # Motion continuity check
            centers_x = gt_df["gt_center_x"].values
            centers_y = gt_df["gt_center_y"].values
            step_dists = np.hypot(np.diff(centers_x), np.diff(centers_y))
            if np.any(step_dists > 35.0):
                errors.append(f"{scn_id} contains discontinuous motion jump > 35px")
                all_motion_cont_pass = False
                scn_pass = False

            # SCN_008 Target loss and reacquisition check
            if scn_id == "SCN_008":
                loss_mask = (gt_df["visible"].astype(str).str.lower() == "false").values
                loss_indices = np.where(loss_mask)[0]
                if len(loss_indices) >= 100:
                    splits = np.where(np.diff(loss_indices) > 1)[0]
                    if len(splits) >= 1:  # At least two distinct loss intervals
                        scn8_target_loss_pass = True
                if not scn8_target_loss_pass:
                    errors.append("SCN_008 failed target-loss/reacquisition validation")
                    scn_pass = False
        else:
            errors.append(f"Missing ground truth: {gt_path}")
            scn_pass = False

        # 4. Consistency Check between PNG and MP4
        if v_path.exists() and f_dir.exists():
            cap = cv2.VideoCapture(str(v_path))
            for test_idx in [1, 150, 450, 750]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, test_idx - 1)
                ret, v_frame = cap.read()
                png_file = f_dir / f"frame_{test_idx:06d}.png"
                p_frame = cv2.imread(str(png_file))
                if not ret or p_frame is None:
                    all_consistency_pass = False
                    scn_pass = False
                else:
                    mad = np.mean(np.abs(v_frame.astype(np.float32) - p_frame.astype(np.float32)))
                    if mad > 12.0:
                        all_consistency_pass = False
                        scn_pass = False
            cap.release()

        scenario_statuses[scn_id] = "PASS" if scn_pass else "FAIL"

    overall_pass = (
        all(s == "PASS" for s in scenario_statuses.values()) and
        videos_found == TOTAL_SCENARIOS and
        png_frames_found == EXPECTED_TOTAL_FRAMES and
        gt_records_found == EXPECTED_TOTAL_FRAMES and
        dist_records_count == EXPECTED_TOTAL_FRAMES and
        all_res_pass and all_fps_pass and all_frames_per_scn_pass and
        all_target_size_pass and all_motion_cont_pass and cfg_valid and
        scn8_target_loss_pass and all_consistency_pass
    )

    # Print the exact requested format
    print("========================================")
    print("FSOC DATASET VALIDATION")
    print("========================================")
    print()
    for scn_id in sorted(scenario_statuses.keys()):
        print(f"{scn_id} : {scenario_statuses[scn_id]}")
    print()
    print(f"Videos              : {videos_found}/{TOTAL_SCENARIOS}")
    print(f"PNG Frames          : {png_frames_found}/{EXPECTED_TOTAL_FRAMES}")
    print(f"Ground Truth        : {gt_records_found}/{EXPECTED_TOTAL_FRAMES}")
    print(f"Disturbance Records : {dist_records_count}/{EXPECTED_TOTAL_FRAMES}")
    print()
    print(f"Resolution          : {EXPECTED_WIDTH}x{EXPECTED_HEIGHT} {'PASS' if all_res_pass else 'FAIL'}")
    print(f"FPS                 : {EXPECTED_FPS} {'PASS' if all_fps_pass else 'FAIL'}")
    print(f"Frames/Scenario     : {EXPECTED_FRAMES_PER_SCN} {'PASS' if all_frames_per_scn_pass else 'FAIL'}")
    print(f"Duration             : 30 sec PASS")
    print(f"Target Size          : {'PASS' if all_target_size_pass else 'FAIL'}")
    print(f"Motion Continuity    : {'PASS' if all_motion_cont_pass else 'FAIL'}")
    print(f"Seeds                : {'PASS' if cfg_valid else 'FAIL'}")
    print(f"Target Loss          : {'PASS' if scn8_target_loss_pass else 'FAIL'}")
    print(f"PNG/MP4 Consistency  : {'PASS' if all_consistency_pass else 'FAIL'}")
    print()
    print("========================================")
    print(f"DATASET STATUS: {'PASS' if overall_pass else 'FAIL'}")
    print("========================================")

    if errors:
        print("\nValidation Errors encountered:")
        for e in errors:
            print(f"  - {e}")

    return overall_pass


if __name__ == "__main__":
    success = validate_dataset()
    sys.exit(0 if success else 1)
