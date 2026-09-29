"""
FSOC Virtual Camera Tracking System - Centroid Analysis Validator
Rigorous mathematical and structural validation of all centroid estimation results.
"""

import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "FSOC_DATASET"
CENTROID_DIR = DATASET_ROOT / "centroid_analysis"
DET_DIR = DATASET_ROOT / "detection_results"
GT_DIR = DATASET_ROOT / "ground_truth"

TOTAL_SCENARIOS = 8
EXPECTED_FRAMES = 900
EXPECTED_TOTAL_RECORDS = 7200
TOLERANCE = 1e-4

REQUIRED_COLUMNS = [
    "scenario_id",
    "frame_id",
    "timestamp",
    "visible",
    "detected",
    "false_positive",
    "gt_center_x",
    "gt_center_y",
    "detected_center_x",
    "detected_center_y",
    "error_x",
    "error_y",
    "abs_error_x",
    "abs_error_y",
    "centroid_error",
]


def validate_centroid_results():
    print("=" * 80)
    print("VALIDATING CENTROID ESTIMATION & ERROR ANALYSIS RESULTS")
    print("=" * 80)

    errors = []
    checks = {}

    # 1. Check directory and files exist
    files_exist = True
    for i in range(1, TOTAL_SCENARIOS + 1):
        f = CENTROID_DIR / f"SCN_{i:03d}_centroid.csv"
        if not f.exists():
            errors.append(f"Missing centroid file: {f.name}")
            files_exist = False

    summary_file = CENTROID_DIR / "centroid_summary.csv"
    if not summary_file.exists():
        errors.append("Missing centroid_summary.csv")
        files_exist = False

    checks["8 Centroid CSV Files Exist"] = files_exist

    # 2. Check records, schema, mathematics per scenario
    total_records = 0
    no_dup_ids = True
    no_missing_ids = True
    columns_valid = True
    math_valid = True
    nan_handling_valid = True
    fp_flag_valid = True
    scn8_loss_valid = True

    for i in range(1, TOTAL_SCENARIOS + 1):
        scn_id = f"SCN_{i:03d}"
        c_file = CENTROID_DIR / f"{scn_id}_centroid.csv"
        d_file = DET_DIR / f"{scn_id}_detection.csv"

        if not c_file.exists() or not d_file.exists():
            continue

        c_df = pd.read_csv(c_file)
        d_df = pd.read_csv(d_file)
        total_records += len(c_df)

        if len(c_df) != EXPECTED_FRAMES:
            errors.append(f"{scn_id} record count mismatch: {len(c_df)} != {EXPECTED_FRAMES}")

        # Column existence check
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in c_df.columns]
        if missing_cols:
            errors.append(f"{scn_id} missing columns: {missing_cols}")
            columns_valid = False

        # Duplicate and missing frame IDs
        u_ids = c_df["frame_id"].unique()
        if len(u_ids) != EXPECTED_FRAMES:
            errors.append(f"{scn_id} contains duplicate frame IDs")
            no_dup_ids = False
        if set(u_ids) != set(range(1, EXPECTED_FRAMES + 1)):
            errors.append(f"{scn_id} missing frame IDs in 1..900")
            no_missing_ids = False

        # Merge with detection results to cross-verify bounding box formula
        merged = pd.merge(c_df, d_df, on="frame_id", suffixes=("_c", "_d"))

        for _, row in merged.iterrows():
            fid = int(row["frame_id"])
            vis = str(row["visible"]).lower() == "true"
            det = str(row["detected_c"]).lower() == "true"
            fp = str(row["false_positive"]).lower() == "true"

            # Check false positive flag
            expected_fp = (not vis and det)
            if fp != expected_fp:
                errors.append(f"{scn_id} frame {fid}: false_positive={fp}, expected {expected_fp}")
                fp_flag_valid = False

            # Check SCN_008 loss frames
            if scn_id == "SCN_008" and not vis:
                if not pd.isna(row["error_x"]) or not pd.isna(row["centroid_error"]):
                    errors.append(f"SCN_008 frame {fid}: target loss frame has non-NaN error")
                    scn8_loss_valid = False

            # Case 1: Valid detection
            if vis and det:
                det_x = row["detected_x"]
                det_y = row["detected_y"]
                det_w = row["detected_width"]
                det_h = row["detected_height"]
                cx = row["detected_center_x_c"]
                cy = row["detected_center_y_c"]
                gt_cx = row["gt_center_x_c"]
                gt_cy = row["gt_center_y_c"]

                # Formula 1: detected_center_x = detected_x + detected_width / 2.0
                expected_cx = det_x + det_w / 2.0
                expected_cy = det_y + det_h / 2.0
                if abs(cx - expected_cx) > TOLERANCE or abs(cy - expected_cy) > TOLERANCE:
                    errors.append(f"{scn_id} frame {fid}: centroid formula mismatch ({cx} vs {expected_cx})")
                    math_valid = False

                # Formula 2: centroid_error = sqrt(error_x^2 + error_y^2)
                err_x = row["error_x"]
                err_y = row["error_y"]
                cent_err = row["centroid_error"]

                expected_err_x = cx - gt_cx
                expected_err_y = cy - gt_cy
                expected_cent_err = math.hypot(expected_err_x, expected_err_y)

                if abs(err_x - expected_err_x) > TOLERANCE or abs(err_y - expected_err_y) > TOLERANCE:
                    errors.append(f"{scn_id} frame {fid}: error_x/y mismatch")
                    math_valid = False
                if abs(cent_err - expected_cent_err) > TOLERANCE:
                    errors.append(f"{scn_id} frame {fid}: centroid_error mismatch ({cent_err} vs {expected_cent_err})")
                    math_valid = False

            # Case 2 or 3: Missed or Expected Loss
            elif not det:
                if not pd.isna(row["detected_center_x_c"]) or not pd.isna(row["centroid_error"]):
                    errors.append(f"{scn_id} frame {fid}: missed/loss frame should have NaN centroid/error")
                    nan_handling_valid = False

    checks["Total Records = 7200"] = (total_records == EXPECTED_TOTAL_RECORDS)
    checks["No Duplicate Frame IDs"] = no_dup_ids
    checks["No Missing Frame IDs"] = no_missing_ids
    checks["Required Columns Exist"] = columns_valid
    checks["Centroid & Error Math Valid (within 1e-4)"] = math_valid
    checks["NaN Handling Correct for Missed/Loss"] = nan_handling_valid
    checks["False Positives Correctly Flagged"] = fp_flag_valid
    checks["SCN_008 Loss Frames Handled Correctly"] = scn8_loss_valid

    # Summary table checks
    if summary_file.exists():
        s_df = pd.read_csv(summary_file)
        checks["Summary CSV Has 8 Rows"] = (len(s_df) == TOTAL_SCENARIOS)
    else:
        checks["Summary CSV Has 8 Rows"] = False

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
        print("CENTROID ANALYSIS VALIDATION STATUS: PASS")
    else:
        print("CENTROID ANALYSIS VALIDATION STATUS: FAIL")
    print("=" * 80)

    return all_pass


if __name__ == "__main__":
    success = validate_centroid_results()
    sys.exit(0 if success else 1)
