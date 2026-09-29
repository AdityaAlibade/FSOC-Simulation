"""
FSOC Virtual Camera Tracking System - Centroid Estimation & Detection Error Analysis
Calculates detected beacon centroids, evaluates frame-by-frame positional errors against
ground truth, handles visibility states, and generates analytical plots and visual inspection frames.
"""

import os
import math
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "FSOC_DATASET"
FRAMES_DIR = DATASET_ROOT / "frames"
GT_DIR = DATASET_ROOT / "ground_truth"
DET_DIR = DATASET_ROOT / "detection_results"
OUTPUT_DIR = DATASET_ROOT / "centroid_analysis"
VIS_DIR = OUTPUT_DIR / "visualizations"
PLOTS_DIR = OUTPUT_DIR / "plots"

TOTAL_SCENARIOS = 8


def load_detection_results(scenario_id):
    """
    Load ground truth and detection CSVs for a scenario and merge on frame_id.
    """
    gt_file = GT_DIR / f"{scenario_id}.csv"
    det_file = DET_DIR / f"{scenario_id}_detection.csv"

    if not gt_file.exists():
        raise FileNotFoundError(f"Missing ground truth file: {gt_file}")
    if not det_file.exists():
        raise FileNotFoundError(f"Missing detection file: {det_file}")

    gt_df = pd.read_csv(gt_file)
    det_df = pd.read_csv(det_file)

    merged = pd.merge(gt_df, det_df, on="frame_id", suffixes=("_gt", "_det"))
    return merged


def calculate_centroid(det_x, det_y, det_w, det_h):
    """
    Calculate detected beacon centroid from bounding box coordinates:
      detected_center_x = detected_x + detected_width / 2.0
      detected_center_y = detected_y + detected_height / 2.0
    """
    if pd.isna(det_x) or pd.isna(det_y) or pd.isna(det_w) or pd.isna(det_h):
        return np.nan, np.nan
    cx = float(det_x) + float(det_w) / 2.0
    cy = float(det_y) + float(det_h) / 2.0
    return cx, cy


def calculate_errors(det_cx, det_cy, gt_cx, gt_cy):
    """
    Calculate signed errors, absolute errors, and Euclidean centroid error in pixels.
    """
    if pd.isna(det_cx) or pd.isna(det_cy) or pd.isna(gt_cx) or pd.isna(gt_cy):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    err_x = float(det_cx - gt_cx)
    err_y = float(det_cy - gt_cy)
    abs_err_x = abs(err_x)
    abs_err_y = abs(err_y)
    centroid_err = math.hypot(err_x, err_y)
    return err_x, err_y, abs_err_x, abs_err_y, centroid_err


def handle_visibility(merged_df):
    """
    Applies the strict 4-case visibility rules:
      Case 1: visible=true,  detected=true  -> Valid detection, calculate centroid & errors
      Case 2: visible=true,  detected=false -> Missed detection, errors = NaN
      Case 3: visible=false, detected=false -> Expected target loss, errors = NaN
      Case 4: visible=false, detected=true  -> False positive, flagged, errors = NaN
    """
    records = []

    for _, row in merged_df.iterrows():
        scn_id = row["scenario_id_gt"]
        frame_id = int(row["frame_id"])
        timestamp = float(row["timestamp_gt"])

        vis = str(row["visible"]).lower() == "true"
        det = str(row["detected"]).lower() == "true"

        gt_cx = float(row["gt_center_x_gt"])
        gt_cy = float(row["gt_center_y_gt"])

        det_x = row["detected_x"]
        det_y = row["detected_y"]
        det_w = row["detected_width"]
        det_h = row["detected_height"]

        if vis and det:
            # Case 1: Valid detection
            false_pos = False
            cx, cy = calculate_centroid(det_x, det_y, det_w, det_h)
            err_x, err_y, abs_x, abs_y, cent_err = calculate_errors(cx, cy, gt_cx, gt_cy)
        elif vis and not det:
            # Case 2: Missed detection
            false_pos = False
            cx, cy = np.nan, np.nan
            err_x, err_y, abs_x, abs_y, cent_err = np.nan, np.nan, np.nan, np.nan, np.nan
        elif not vis and not det:
            # Case 3: Expected target loss
            false_pos = False
            cx, cy = np.nan, np.nan
            err_x, err_y, abs_x, abs_y, cent_err = np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            # Case 4: False positive
            false_pos = True
            cx, cy = calculate_centroid(det_x, det_y, det_w, det_h)
            # Not treated as valid centroid measurement
            err_x, err_y, abs_x, abs_y, cent_err = np.nan, np.nan, np.nan, np.nan, np.nan

        records.append({
            "scenario_id": scn_id,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "visible": "true" if vis else "false",
            "detected": "true" if det else "false",
            "false_positive": "true" if false_pos else "false",
            "gt_center_x": gt_cx,
            "gt_center_y": gt_cy,
            "detected_center_x": cx,
            "detected_center_y": cy,
            "error_x": err_x,
            "error_y": err_y,
            "abs_error_x": abs_x,
            "abs_error_y": abs_y,
            "centroid_error": cent_err,
        })

    return pd.DataFrame(records)


def calculate_summary_metrics(centroid_df):
    """
    Computes summary metrics for a scenario according to Section 10 & 11 definitions.
    """
    total_frames = len(centroid_df)
    vis_mask = (centroid_df["visible"] == "true")
    det_mask = (centroid_df["detected"] == "true")
    fp_mask = (centroid_df["false_positive"] == "true")

    visible_frames = int(vis_mask.sum())
    valid_detections = int((vis_mask & det_mask).sum())
    missed_detections = int((vis_mask & (~det_mask)).sum())
    false_positives = int(fp_mask.sum())

    det_rate = (valid_detections / visible_frames) if visible_frames > 0 else 0.0

    valid_errs = centroid_df.loc[vis_mask & det_mask, "centroid_error"].dropna().values
    err_x_vals = centroid_df.loc[vis_mask & det_mask, "error_x"].dropna().values
    err_y_vals = centroid_df.loc[vis_mask & det_mask, "error_y"].dropna().values

    if len(valid_errs) > 0:
        mean_err_x = float(np.mean(err_x_vals))
        mean_err_y = float(np.mean(err_y_vals))
        mean_centroid = float(np.mean(valid_errs))
        rmse = float(np.sqrt(np.mean(valid_errs ** 2)))
        max_err = float(np.max(valid_errs))
        median_err = float(np.median(valid_errs))
    else:
        mean_err_x = 0.0
        mean_err_y = 0.0
        mean_centroid = 0.0
        rmse = 0.0
        max_err = 0.0
        median_err = 0.0

    return {
        "scenario_id": centroid_df["scenario_id"].iloc[0],
        "total_frames": total_frames,
        "visible_frames": visible_frames,
        "valid_detections": valid_detections,
        "missed_detections": missed_detections,
        "false_positives": false_positives,
        "detection_rate": det_rate,
        "mean_error_x": mean_err_x,
        "mean_error_y": mean_err_y,
        "mean_centroid_error": mean_centroid,
        "rmse_centroid_error": rmse,
        "max_centroid_error": max_err,
        "median_centroid_error": median_err,
    }


def save_centroid_results(centroid_df, scenario_id):
    """Save frame-by-frame centroid analysis to CSV."""
    out_file = OUTPUT_DIR / f"{scenario_id}_centroid.csv"
    centroid_df.to_csv(out_file, index=False)


def generate_visualizations(centroid_df, scenario_id):
    """
    Generate representative annotated frames showing GT center, Detected Center,
    and Error Vector.
    """
    frames_folder = FRAMES_DIR / scenario_id
    sample_ids = [1, 150, 450, 750]
    if scenario_id == "SCN_008":
        sample_ids = [1, 260, 301, 620, 661]

    sub_imgs = []
    for fid in sample_ids:
        row = centroid_df.loc[centroid_df["frame_id"] == fid].iloc[0]
        f_path = frames_folder / f"frame_{fid:06d}.png"
        gray = cv2.imread(str(f_path), cv2.IMREAD_GRAYSCALE)
        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        vis = row["visible"] == "true"
        det = row["detected"] == "true"
        gt_cx, gt_cy = float(row["gt_center_x"]), float(row["gt_center_y"])
        det_cx, det_cy = row["detected_center_x"], row["detected_center_y"]
        err = row["centroid_error"]

        # 1. Ground Truth Center (Cyan marker)
        gt_pt = (int(round(gt_cx)), int(round(gt_cy)))
        cv2.circle(annotated, gt_pt, 6, (255, 220, 0), 1)
        cv2.drawMarker(annotated, gt_pt, (255, 220, 0), cv2.MARKER_CROSS, 10, 1)

        if vis and det:
            # 2. Detected Center (Green marker)
            det_pt = (int(round(det_cx)), int(round(det_cy)))
            cv2.circle(annotated, det_pt, 6, (0, 255, 0), 1)
            cv2.drawMarker(annotated, det_pt, (0, 255, 0), cv2.MARKER_TILTED_CROSS, 10, 1)

            # 3. Error Vector (Yellow line from GT -> Detected)
            cv2.line(annotated, gt_pt, det_pt, (0, 255, 255), 2)

            status_str = f"VALID | Err: {err:.2f}px"
            status_color = (0, 255, 0)
        elif vis and not det:
            status_str = "MISSED DETECTION"
            status_color = (0, 0, 255)
        else:
            status_str = "TARGET LOSS PERIOD"
            status_color = (0, 165, 255)

        # Telemetry panel
        cv2.rectangle(annotated, (10, 10), (360, 95), (15, 15, 15), -1)
        cv2.rectangle(annotated, (10, 10), (360, 95), (60, 60, 60), 1)
        cv2.putText(annotated, f"{scenario_id} - Frame {fid:04d}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(annotated, f"Status: {status_str}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        cv2.putText(annotated, f"GT Center: ({gt_cx:.1f}, {gt_cy:.1f})", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 0), 1)
        if det and not pd.isna(det_cx):
            cv2.putText(annotated, f"Det Center: ({det_cx:.1f}, {det_cy:.1f})", (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        sub_imgs.append(annotated)

    if len(sub_imgs) <= 4:
        composite = np.hstack(sub_imgs)
    else:
        top_row = np.hstack(sub_imgs[:3])
        bot_row = np.hstack(sub_imgs[3:] + [np.zeros_like(sub_imgs[0])] * (3 - len(sub_imgs[3:])))
        composite = np.vstack([top_row, bot_row])

    cv2.imwrite(str(VIS_DIR / f"{scenario_id}_centroid_vis.png"), composite)


def generate_plots(all_centroid_dfs):
    """
    Generate required plots:
      Plot 1: Ground Truth vs Detected X
      Plot 2: Ground Truth vs Detected Y
      Plot 3: Centroid Error Over Time
    Generates both per-scenario detailed figures and master 8-scenario panels.
    """
    # 1. Master Trajectory Comparison Plot (X and Y over time across all scenarios)
    fig, axes = plt.subplots(4, 2, figsize=(18, 14))
    fig.suptitle("Ground Truth vs Detected Position Over Time (640x480 Frame)", fontsize=15, fontweight="bold")

    for i in range(1, 9):
        scn_id = f"SCN_{i:03d}"
        df = all_centroid_dfs[scn_id]
        ax = axes[(i - 1) // 2, (i - 1) % 2]

        frames = df["frame_id"].values
        gt_x = df["gt_center_x"].values
        det_x = df["detected_center_x"].values
        gt_y = df["gt_center_y"].values
        det_y = df["detected_center_y"].values

        ax.plot(frames, gt_x, label="GT X", color="#00e5ff", linewidth=1.5, alpha=0.9)
        ax.plot(frames, det_x, label="Det X", color="#00e676", linestyle="--", linewidth=1.2, alpha=0.85)
        ax.plot(frames, gt_y, label="GT Y", color="#ffab00", linewidth=1.5, alpha=0.9)
        ax.plot(frames, det_y, label="Det Y", color="#ff1744", linestyle="--", linewidth=1.2, alpha=0.85)

        ax.set_title(f"{scn_id}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Frame", fontsize=9)
        ax.set_ylabel("Pixel Coordinate", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        if i == 1:
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot_gt_vs_detected_xy_all.png", dpi=180)
    plt.close()

    # 2. Master Centroid Error Over Time Plot
    fig, axes = plt.subplots(4, 2, figsize=(18, 14))
    fig.suptitle("Centroid Error Over Time (Pixels)", fontsize=15, fontweight="bold")

    for i in range(1, 9):
        scn_id = f"SCN_{i:03d}"
        df = all_centroid_dfs[scn_id]
        ax = axes[(i - 1) // 2, (i - 1) % 2]

        frames = df["frame_id"].values
        errs = df["centroid_error"].values

        ax.plot(frames, errs, color="#d500f9", linewidth=1.2, label="Centroid Error")
        ax.axhline(2.0, color="#76ff03", linestyle="--", alpha=0.6, label="2.0 px threshold")

        ax.set_title(f"{scn_id} - Centroid Error", fontsize=11, fontweight="bold")
        ax.set_xlabel("Frame", fontsize=9)
        ax.set_ylabel("Error (pixels)", fontsize=9)
        ax.set_ylim(-0.2, max(5.0, np.nanmax(errs) + 1.0) if not np.all(np.isnan(errs)) else 5.0)
        ax.grid(True, linestyle=":", alpha=0.6)
        if i == 1:
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot_centroid_error_over_time_all.png", dpi=180)
    plt.close()

    # 3. Individual detailed 3-panel plots for each scenario (Plot 1, Plot 2, Plot 3)
    for i in range(1, 9):
        scn_id = f"SCN_{i:03d}"
        df = all_centroid_dfs[scn_id]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f"{scn_id} - Centroid Estimation and Error Analysis", fontsize=14, fontweight="bold")

        frames = df["frame_id"].values

        # Plot 1: GT vs Detected X
        ax1.plot(frames, df["gt_center_x"].values, label="Ground Truth X", color="#00b0ff", linewidth=2.0)
        ax1.plot(frames, df["detected_center_x"].values, label="Detected X", color="#00e676", linestyle="--", linewidth=1.5)
        ax1.set_ylabel("X Position (px)", fontsize=10)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", fontsize=9)

        # Plot 2: GT vs Detected Y
        ax2.plot(frames, df["gt_center_y"].values, label="Ground Truth Y", color="#ff9100", linewidth=2.0)
        ax2.plot(frames, df["detected_center_y"].values, label="Detected Y", color="#ff3d00", linestyle="--", linewidth=1.5)
        ax2.set_ylabel("Y Position (px)", fontsize=10)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", fontsize=9)

        # Plot 3: Centroid Error Over Time
        errs = df["centroid_error"].values
        ax3.plot(frames, errs, color="#d500f9", linewidth=1.5, label="Centroid Error (px)")
        ax3.set_xlabel("Frame ID", fontsize=10)
        ax3.set_ylabel("Centroid Error (px)", fontsize=10)
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{scn_id}_centroid_analysis_plot.png", dpi=160)
        plt.close()


def main():
    print("=" * 75)
    print("FSOC VIRTUAL CAMERA TRACKING - CENTROID ESTIMATION & ERROR ANALYSIS")
    print("=" * 75)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_records = []
    all_dfs = {}

    for i in range(1, TOTAL_SCENARIOS + 1):
        scn_id = f"SCN_{i:03d}"
        print(f"Processing centroid estimation for {scn_id}...")

        merged_df = load_detection_results(scn_id)
        centroid_df = handle_visibility(merged_df)
        save_centroid_results(centroid_df, scn_id)

        metrics = calculate_summary_metrics(centroid_df)
        summary_records.append(metrics)
        all_dfs[scn_id] = centroid_df

        generate_visualizations(centroid_df, scn_id)

    # Save summary CSV
    summary_df = pd.DataFrame(summary_records)
    summary_file = OUTPUT_DIR / "centroid_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print("Generating comprehensive trajectory and error plots...")
    generate_plots(all_dfs)

    print("\n" + "=" * 90)
    print("FSOC CENTROID ESTIMATION SUMMARY")
    print("=" * 90)
    headers = ["Scenario", "Frames", "Visible", "Detected", "Det Rate", "Mean Err", "RMSE", "Max Err", "Median Err"]
    print(f"{headers[0]:<9} {headers[1]:<8} {headers[2]:<9} {headers[3]:<10} {headers[4]:<11} {headers[5]:<11} {headers[6]:<9} {headers[7]:<10} {headers[8]:<11}")
    print("-" * 90)
    for r in summary_records:
        rate_str = f"{r['detection_rate'] * 100:.1f}%"
        print(f"{r['scenario_id']:<9} {r['total_frames']:<8} {r['visible_frames']:<9} {r['valid_detections']:<10} {rate_str:<11} {r['mean_centroid_error']:<11.2f} {r['rmse_centroid_error']:<9.2f} {r['max_centroid_error']:<10.2f} {r['median_centroid_error']:<11.2f}")
    print("=" * 90)
    print(f"\nAll centroid results saved to: {OUTPUT_DIR}")
    print(f"Visualizations saved to: {VIS_DIR}")
    print(f"Analytical plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
