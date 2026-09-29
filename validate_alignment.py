"""
FSOC Virtual Camera Tracking System - Alignment Validation Script
Validates virtual alignment results, pan/tilt velocity limits, angular math,
stability criteria, and event logging according to Section 22.
"""

import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "FSOC_DATASET"
ALIGN_DIR = DATASET_ROOT / "alignment_results"
CONFIG_FILE = DATASET_ROOT / "dataset_config" / "alignment_config.json"

TOTAL_SCENARIOS = 8
EXPECTED_FRAMES = 900
EXPECTED_TOTAL_RECORDS = 7200
TOLERANCE = 1e-3

REQUIRED_COLUMNS = [
    "scenario_id",
    "frame_id",
    "timestamp",
    "tracking_status",
    "tracked_x",
    "tracked_y",
    "image_center_x",
    "image_center_y",
    "pixel_error_x",
    "pixel_error_y",
    "target_angle_x",
    "target_angle_y",
    "remaining_error_x",
    "remaining_error_y",
    "pan_command",
    "tilt_command",
    "virtual_pan",
    "virtual_tilt",
    "alignment_status",
]


def validate_alignment_results():
    print("=" * 80)
    print("VALIDATING VIRTUAL CAMERA ALIGNMENT & PAN-TILT CONTROL SIMULATION")
    print("=" * 80)

    errors = []
    checks = {}

    # 1. 8 alignment CSV files exist
    files_exist = True
    for i in range(1, TOTAL_SCENARIOS + 1):
        f = ALIGN_DIR / f"SCN_{i:03d}_alignment.csv"
        if not f.exists():
            errors.append(f"Missing alignment CSV: {f.name}")
            files_exist = False
    checks["8 Alignment CSV Files Exist"] = files_exist

    events_file = ALIGN_DIR / "alignment_events.csv"
    summary_file = ALIGN_DIR / "alignment_summary.csv"
    checks["Alignment Events CSV Exists"] = events_file.exists()
    checks["Alignment Summary CSV Exists"] = summary_file.exists()

    # Videos exist check
    vids_exist = True
    for i in range(1, TOTAL_SCENARIOS + 1):
        v = ALIGN_DIR / "visualizations" / f"SCN_{i:03d}_alignment.mp4"
        if not v.exists():
            vids_exist = False
            errors.append(f"Missing alignment video: {v.name}")
    checks["8 Alignment Simulation Videos Exist"] = vids_exist

    # 2. Detailed record validation
    total_records = 0
    no_dup_ids = True
    no_missing_ids = True
    columns_valid = True
    pixel_err_valid = True
    ang_conv_valid = True
    limits_valid = True
    lost_state_valid = True
    threshold_valid = True

    max_pan_step = 5.0 / 30.0 + 1e-4  # 0.16667
    max_tilt_step = 5.0 / 30.0 + 1e-4

    th_h = 0.05
    th_v = 0.05

    for i in range(1, TOTAL_SCENARIOS + 1):
        scn_id = f"SCN_{i:03d}"
        a_file = ALIGN_DIR / f"{scn_id}_alignment.csv"
        if not a_file.exists():
            continue

        df = pd.read_csv(a_file)
        total_records += len(df)

        if len(df) != EXPECTED_FRAMES:
            errors.append(f"{scn_id} record count mismatch: {len(df)} != {EXPECTED_FRAMES}")

        # Column check
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            errors.append(f"{scn_id} missing columns: {missing_cols}")
            columns_valid = False

        # Duplicate / missing IDs
        u_ids = df["frame_id"].unique()
        if len(u_ids) != EXPECTED_FRAMES:
            errors.append(f"{scn_id} contains duplicate frame IDs")
            no_dup_ids = False
        if set(u_ids) != set(range(1, EXPECTED_FRAMES + 1)):
            errors.append(f"{scn_id} missing frame IDs")
            no_missing_ids = False

        # Validate math and control constraints
        for _, row in df.iterrows():
            fid = int(row["frame_id"])
            status = row["tracking_status"]
            align_status = row["alignment_status"]
            tx = row["tracked_x"]
            ty = row["tracked_y"]
            cx = row["image_center_x"]
            cy = row["image_center_y"]

            p_cmd = row["pan_command"]
            t_cmd = row["tilt_command"]

            # Limit check: abs(command) <= max_speed / FPS
            if abs(p_cmd) > max_pan_step or abs(t_cmd) > max_tilt_step:
                errors.append(f"{scn_id} frame {fid}: command exceeds max speed step ({p_cmd}, {t_cmd})")
                limits_valid = False

            # Check lost state handling
            if status == "LOST" or pd.isna(tx):
                if p_cmd != 0.0 or t_cmd != 0.0:
                    errors.append(f"{scn_id} frame {fid}: non-zero control during LOST state")
                    lost_state_valid = False
                if align_status not in ["LOST", "NOT_TRACKED"]:
                    errors.append(f"{scn_id} frame {fid}: invalid alignment_status during LOST state")
                    lost_state_valid = False
            else:
                # Check pixel error math
                px_err_x = row["pixel_error_x"]
                px_err_y = row["pixel_error_y"]
                if abs(px_err_x - (tx - cx)) > TOLERANCE or abs(px_err_y - (ty - cy)) > TOLERANCE:
                    errors.append(f"{scn_id} frame {fid}: pixel error math mismatch")
                    pixel_err_valid = False

                # Check angular error conversion
                t_ang_x = row["target_angle_x"]
                t_ang_y = row["target_angle_y"]
                expected_ang_x = (px_err_x / 640.0) * 4.0
                expected_ang_y = (px_err_y / 480.0) * 3.0
                if abs(t_ang_x - expected_ang_x) > TOLERANCE or abs(t_ang_y - expected_ang_y) > TOLERANCE:
                    errors.append(f"{scn_id} frame {fid}: angular conversion mismatch")
                    ang_conv_valid = False

                # Check remaining error & threshold
                rem_x = row["remaining_error_x"]
                rem_y = row["remaining_error_y"]
                is_within = (abs(rem_x) <= th_h and abs(rem_y) <= th_v)
                if is_within and align_status != "ALIGNED":
                    errors.append(f"{scn_id} frame {fid}: status should be ALIGNED")
                    threshold_valid = False
                elif not is_within and align_status == "ALIGNED":
                    errors.append(f"{scn_id} frame {fid}: status should be ALIGNING")
                    threshold_valid = False

    checks["Total Records = 7200"] = (total_records == EXPECTED_TOTAL_RECORDS)
    checks["No Duplicate Frame IDs"] = no_dup_ids
    checks["No Missing Frame IDs"] = no_missing_ids
    checks["Required Columns Exist"] = columns_valid
    checks["Pixel Error Calculation Correct"] = pixel_err_valid
    checks["Angular Error Conversion Correct"] = ang_conv_valid
    checks["Pan/Tilt Step Limits Enforced (<= 0.1667 deg/frame)"] = limits_valid
    checks["No Control During LOST State (Commands = 0)"] = lost_state_valid
    checks["Coarse Alignment Thresholds Correct (0.05 deg)"] = threshold_valid

    # Event logging checks
    if events_file.exists():
        ev_df = pd.read_csv(events_file)
        has_started = (ev_df["event_type"] == "ALIGNMENT_STARTED").any()
        has_aligned = (ev_df["event_type"] == "ALIGNED").any()
        has_lost = (ev_df["event_type"] == "TRACK_LOST").any()
        has_reacq = (ev_df["event_type"] == "TRACK_REACQUIRED").any()
        checks["Event Logging Complete (Started, Aligned, Lost, Reacq)"] = (has_started and has_aligned and has_lost and has_reacq)
    else:
        checks["Event Logging Complete (Started, Aligned, Lost, Reacq)"] = False

    # Original dataset unchanged check
    orig_frames_cnt = sum(len(list((DATASET_ROOT / "frames" / f"SCN_{i:03d}").glob("*.png"))) for i in range(1, 9))
    checks["Original Dataset Frames Intact (7200)"] = (orig_frames_cnt == 7200)

    print("\nVALIDATION CHECKLIST:")
    print("-" * 80)
    all_pass = True
    for item, status in checks.items():
        res_str = "PASS" if status else "FAIL"
        if not status:
            all_pass = False
        print(f"  {item:<55} : {res_str}")
    print("-" * 80)

    if errors:
        print("\nERRORS ENCOUNTERED:")
        for e in errors[:10]:
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors.")

    print("\n" + "=" * 80)
    if all_pass:
        print("VIRTUAL ALIGNMENT VALIDATION STATUS: PASS")
    else:
        print("VIRTUAL ALIGNMENT VALIDATION STATUS: FAIL")
    print("=" * 80)

    return all_pass


if __name__ == "__main__":
    success = validate_alignment_results()
    sys.exit(0 if success else 1)
