"""
FSOC Virtual Camera Tracking System - Tracking Validation Script
Validates tracking outputs, state machine transitions, velocity math, error consistency,
and events logging according to Section 21.
"""

import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "FSOC_DATASET"
TRACK_DIR = DATASET_ROOT / "tracking_results"
CONFIG_FILE = DATASET_ROOT / "dataset_config" / "tracking_config.json"

TOTAL_SCENARIOS = 8
EXPECTED_FRAMES = 900
EXPECTED_TOTAL_RECORDS = 7200
TOLERANCE = 1e-3

VALID_STATES = {"LOST", "ACQUIRING", "TRACKING", "TEMPORARILY_LOST", "REACQUIRED"}

VALID_TRANSITIONS = {
    # From state -> set of allowed next states
    "LOST": {"LOST", "ACQUIRING"},
    "ACQUIRING": {"ACQUIRING", "TRACKING", "TEMPORARILY_LOST", "LOST"},
    "TRACKING": {"TRACKING", "TEMPORARILY_LOST", "LOST"},
    "TEMPORARILY_LOST": {"TEMPORARILY_LOST", "REACQUIRED", "LOST", "TRACKING"},
    "REACQUIRED": {"TRACKING", "REACQUIRED", "TEMPORARILY_LOST", "LOST"},
}

REQUIRED_COLUMNS = [
    "scenario_id",
    "frame_id",
    "timestamp",
    "visible",
    "detected",
    "detected_center_x",
    "detected_center_y",
    "tracked_x",
    "tracked_y",
    "velocity_x",
    "velocity_y",
    "track_age",
    "consecutive_hits",
    "consecutive_misses",
    "tracking_status",
    "gt_center_x",
    "gt_center_y",
]


def validate_tracking_results():
    print("=" * 80)
    print("VALIDATING TEMPORAL BEACON TRACKING RESULTS")
    print("=" * 80)

    errors = []
    checks = {}

    # 1. 8 tracking CSV files exist
    files_exist = True
    for i in range(1, TOTAL_SCENARIOS + 1):
        f = TRACK_DIR / f"SCN_{i:03d}_tracking.csv"
        if not f.exists():
            errors.append(f"Missing tracking CSV: {f.name}")
            files_exist = False
    checks["8 Tracking CSV Files Exist"] = files_exist

    # Events and summary exist
    events_file = TRACK_DIR / "tracking_events.csv"
    summary_file = TRACK_DIR / "tracking_summary.csv"
    checks["Tracking Events CSV Exists"] = events_file.exists()
    checks["Tracking Summary CSV Exists"] = summary_file.exists()

    # 2. Check records, schema, states, velocities, errors
    total_records = 0
    no_dup_ids = True
    no_missing_ids = True
    columns_valid = True
    states_valid = True
    transitions_valid = True
    velocity_math_valid = True
    error_math_valid = True
    loss_handling_valid = True

    dt = 1.0 / 30.0

    for i in range(1, TOTAL_SCENARIOS + 1):
        scn_id = f"SCN_{i:03d}"
        t_file = TRACK_DIR / f"{scn_id}_tracking.csv"
        if not t_file.exists():
            continue

        df = pd.read_csv(t_file)
        total_records += len(df)

        if len(df) != EXPECTED_FRAMES:
            errors.append(f"{scn_id} frame count mismatch: {len(df)} != {EXPECTED_FRAMES}")

        # Check required columns
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

        # Validate state machine and math frame-by-frame
        prev_status = "LOST"
        prev_x = np.nan
        prev_y = np.nan

        for idx, row in df.iterrows():
            fid = int(row["frame_id"])
            status = row["tracking_status"]
            vis = str(row["visible"]).lower() == "true"
            det = str(row["detected"]).lower() == "true"

            # Check valid status enum
            if status not in VALID_STATES:
                errors.append(f"{scn_id} frame {fid}: invalid tracking_status '{status}'")
                states_valid = False

            # Check state transitions
            if idx > 0:
                allowed_next = VALID_TRANSITIONS.get(prev_status, set())
                if status not in allowed_next:
                    errors.append(f"{scn_id} frame {fid}: invalid transition from {prev_status} -> {status}")
                    transitions_valid = False

            # Check velocity math
            vx = row["velocity_x"]
            vy = row["velocity_y"]
            tx = row["tracked_x"]
            ty = row["tracked_y"]

            if not pd.isna(tx) and not pd.isna(prev_x):
                # When position updated
                expected_vx = (tx - prev_x) / dt
                expected_vy = (ty - prev_y) / dt
                if status in ["TRACKING", "ACQUIRING", "REACQUIRED"]:
                    if abs(vx - expected_vx) > 0.05 or abs(vy - expected_vy) > 0.05:
                        errors.append(f"{scn_id} frame {fid}: velocity mismatch ({vx} vs {expected_vx})")
                        velocity_math_valid = False
            elif pd.isna(tx):
                if not pd.isna(vx) or not pd.isna(vy):
                    errors.append(f"{scn_id} frame {fid}: LOST state should have NaN velocity")
                    velocity_math_valid = False

            # Check tracking error math
            if "tracking_error" in df.columns:
                terr = row["tracking_error"]
                gt_cx = row["gt_center_x"]
                gt_cy = row["gt_center_y"]
                if vis and not pd.isna(tx):
                    expected_err = math.hypot(tx - gt_cx, ty - gt_cy)
                    if abs(terr - expected_err) > TOLERANCE:
                        errors.append(f"{scn_id} frame {fid}: tracking_error mismatch ({terr} vs {expected_err})")
                        error_math_valid = False
                elif not vis or pd.isna(tx):
                    if not pd.isna(terr):
                        errors.append(f"{scn_id} frame {fid}: invisible/untracked frame has non-NaN error")
                        error_math_valid = False

            # Check target-loss in SCN_008
            if scn_id == "SCN_008" and not vis:
                if status == "TRACKING":
                    errors.append(f"SCN_008 frame {fid}: target loss frame should not be in TRACKING state")
                    loss_handling_valid = False

            prev_status = status
            prev_x = tx
            prev_y = ty

    checks["Total Records = 7200"] = (total_records == EXPECTED_TOTAL_RECORDS)
    checks["No Duplicate Frame IDs"] = no_dup_ids
    checks["No Missing Frame IDs"] = no_missing_ids
    checks["Required Columns Exist"] = columns_valid
    checks["Tracking States Valid"] = states_valid
    checks["State Transitions Logically Valid"] = transitions_valid
    checks["Velocity Calculations Correct"] = velocity_math_valid
    checks["Tracking Error Math Correct"] = error_math_valid
    checks["Target-Loss Handled Correctly"] = loss_handling_valid

    # Check events CSV
    if events_file.exists():
        ev_df = pd.read_csv(events_file)
        has_acq = (ev_df["event_type"] == "ACQUISITION").any()
        has_trk = (ev_df["event_type"] == "TRACKING_STARTED").any()
        has_reacq = (ev_df["event_type"] == "REACQUISITION").any()
        checks["Reacquisition Events Recorded"] = has_reacq
        checks["Acquisition Events Recorded"] = has_acq and has_trk
    else:
        checks["Reacquisition Events Recorded"] = False
        checks["Acquisition Events Recorded"] = False

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
        print(f"  {item:<50} : {res_str}")
    print("-" * 80)

    if errors:
        print("\nERRORS ENCOUNTERED:")
        for e in errors[:10]:
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors.")

    print("\n" + "=" * 80)
    if all_pass:
        print("TEMPORAL TRACKING VALIDATION STATUS: PASS")
    else:
        print("TEMPORAL TRACKING VALIDATION STATUS: FAIL")
    print("=" * 80)

    return all_pass


if __name__ == "__main__":
    success = validate_tracking_results()
    sys.exit(0 if success else 1)
