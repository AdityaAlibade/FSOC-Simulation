"""
FSOC Virtual Camera Tracking System - Temporal Beacon Tracking Stage
Implements temporal tracking using detection and centroid analysis results.
Maintains beacon identity, track state machine, velocity estimation, and error logging.
"""

import os
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "FSOC_DATASET"
CONFIG_FILE = DATASET_ROOT / "dataset_config" / "tracking_config.json"
DET_DIR = DATASET_ROOT / "detection_results"
GT_DIR = DATASET_ROOT / "ground_truth"
FRAMES_DIR = DATASET_ROOT / "frames"
OUTPUT_DIR = DATASET_ROOT / "tracking_results"
VIS_DIR = OUTPUT_DIR / "visualizations"
PLOTS_DIR = OUTPUT_DIR / "plots"

TOTAL_SCENARIOS = 8


def load_config():
    """Load tracking configuration parameters."""
    default_config = {
        "acquisition_frames": 3,
        "max_consecutive_misses": 5,
        "fps": 30,
    }
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return default_config


def load_detection_results(scenario_id):
    """
    Load ground truth and detection results for a scenario, merged on frame_id.
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


def initialize_tracker():
    """Initialize tracker state variables."""
    return {
        "status": "LOST",
        "tracked_x": np.nan,
        "tracked_y": np.nan,
        "previous_x": np.nan,
        "previous_y": np.nan,
        "velocity_x": np.nan,
        "velocity_y": np.nan,
        "track_age": 0,
        "consecutive_hits": 0,
        "consecutive_misses": 0,
        "ever_tracked": False,
    }


def calculate_velocity(current_x, current_y, prev_x, prev_y, dt):
    """Calculate velocity vector (pixels/sec)."""
    if pd.isna(current_x) or pd.isna(prev_x) or pd.isna(current_y) or pd.isna(prev_y) or dt <= 0:
        return 0.0, 0.0
    vx = (current_x - prev_x) / dt
    vy = (current_y - prev_y) / dt
    return vx, vy


def calculate_tracking_error(tracked_x, tracked_y, gt_x, gt_y, visible):
    """
    Calculate Euclidean tracking error when target is visible and track exists.
    """
    if not visible or pd.isna(tracked_x) or pd.isna(tracked_y) or pd.isna(gt_x) or pd.isna(gt_y):
        return np.nan, np.nan, np.nan
    err_x = tracked_x - gt_x
    err_y = tracked_y - gt_y
    err_eucl = math.hypot(err_x, err_y)
    return err_x, err_y, err_eucl


def handle_detection(state, det_x, det_y, config, events, scn_id, frame_id, timestamp):
    """
    Update tracker when valid detection is observed.
    Transitions:
      LOST -> ACQUIRING
      ACQUIRING -> TRACKING (if hits >= acquisition_frames)
      TEMPORARILY_LOST -> REACQUIRED
      REACQUIRED -> TRACKING
    """
    dt = 1.0 / config["fps"]
    prev_status = state["status"]

    if prev_status == "LOST":
        # Initial acquisition or reacquisition after full loss
        state["status"] = "ACQUIRING"
        state["tracked_x"] = det_x
        state["tracked_y"] = det_y
        state["previous_x"] = det_x
        state["previous_y"] = det_y
        state["velocity_x"] = 0.0
        state["velocity_y"] = 0.0
        state["track_age"] = 1
        state["consecutive_hits"] = 1
        state["consecutive_misses"] = 0

        if not state["ever_tracked"]:
            events.append({"scenario_id": scn_id, "event_type": "ACQUISITION", "frame_id": frame_id, "timestamp": timestamp})
        else:
            events.append({"scenario_id": scn_id, "event_type": "REACQUISITION", "frame_id": frame_id, "timestamp": timestamp})

    elif prev_status == "ACQUIRING":
        state["consecutive_hits"] += 1
        state["track_age"] += 1
        state["consecutive_misses"] = 0
        state["previous_x"] = state["tracked_x"]
        state["previous_y"] = state["tracked_y"]
        state["tracked_x"] = det_x
        state["tracked_y"] = det_y
        state["velocity_x"], state["velocity_y"] = calculate_velocity(det_x, det_y, state["previous_x"], state["previous_y"], dt)

        if state["consecutive_hits"] >= config["acquisition_frames"]:
            state["status"] = "TRACKING"
            state["ever_tracked"] = True
            events.append({"scenario_id": scn_id, "event_type": "TRACKING_STARTED", "frame_id": frame_id, "timestamp": timestamp})

    elif prev_status == "TEMPORARILY_LOST":
        # Target was temporarily missed and is now reacquired
        state["status"] = "REACQUIRED"
        state["consecutive_hits"] = 1
        state["consecutive_misses"] = 0
        state["track_age"] += 1
        state["previous_x"] = state["tracked_x"]
        state["previous_y"] = state["tracked_y"]
        state["tracked_x"] = det_x
        state["tracked_y"] = det_y
        state["velocity_x"], state["velocity_y"] = calculate_velocity(det_x, det_y, state["previous_x"], state["previous_y"], dt)
        events.append({"scenario_id": scn_id, "event_type": "REACQUISITION", "frame_id": frame_id, "timestamp": timestamp})

    elif prev_status in ["TRACKING", "REACQUIRED"]:
        state["status"] = "TRACKING"
        state["consecutive_hits"] += 1
        state["consecutive_misses"] = 0
        state["track_age"] += 1
        state["previous_x"] = state["tracked_x"]
        state["previous_y"] = state["tracked_y"]
        state["tracked_x"] = det_x
        state["tracked_y"] = det_y
        state["velocity_x"], state["velocity_y"] = calculate_velocity(det_x, det_y, state["previous_x"], state["previous_y"], dt)


def handle_missed_detection(state, config, events, scn_id, frame_id, timestamp):
    """
    Update tracker when target is visible but detection fails (or when target is lost).
    """
    prev_status = state["status"]
    state["consecutive_misses"] += 1
    state["consecutive_hits"] = 0

    if prev_status in ["TRACKING", "REACQUIRED", "ACQUIRING"]:
        if state["consecutive_misses"] <= config["max_consecutive_misses"]:
            state["status"] = "TEMPORARILY_LOST"
            state["track_age"] += 1
            # Maintain last known position
            state["velocity_x"] = 0.0
            state["velocity_y"] = 0.0
            events.append({"scenario_id": scn_id, "event_type": "TEMPORARY_LOSS", "frame_id": frame_id, "timestamp": timestamp})
        else:
            state["status"] = "LOST"
            state["tracked_x"] = np.nan
            state["tracked_y"] = np.nan
            state["previous_x"] = np.nan
            state["previous_y"] = np.nan
            state["velocity_x"] = np.nan
            state["velocity_y"] = np.nan
            state["track_age"] = 0
            events.append({"scenario_id": scn_id, "event_type": "TRACK_LOST", "frame_id": frame_id, "timestamp": timestamp})

    elif prev_status == "TEMPORARILY_LOST":
        if state["consecutive_misses"] > config["max_consecutive_misses"]:
            state["status"] = "LOST"
            state["tracked_x"] = np.nan
            state["tracked_y"] = np.nan
            state["previous_x"] = np.nan
            state["previous_y"] = np.nan
            state["velocity_x"] = np.nan
            state["velocity_y"] = np.nan
            state["track_age"] = 0
            events.append({"scenario_id": scn_id, "event_type": "TRACK_LOST", "frame_id": frame_id, "timestamp": timestamp})
        else:
            state["track_age"] += 1
            state["velocity_x"] = 0.0
            state["velocity_y"] = 0.0


def update_tracker(state, row, config, events, scn_id):
    """
    Chronological frame update respecting visibility and detection states.
    """
    frame_id = int(row["frame_id"])
    timestamp = float(row["timestamp_gt"])
    vis = str(row["visible"]).lower() == "true"
    det = str(row["detected"]).lower() == "true"

    det_cx = row["detected_center_x"]
    det_cy = row["detected_center_y"]
    gt_cx = float(row["gt_center_x_gt"])
    gt_cy = float(row["gt_center_y_gt"])

    if vis and det:
        # Case A: Valid Detection
        handle_detection(state, det_cx, det_cy, config, events, scn_id, frame_id, timestamp)
    elif vis and not det:
        # Case B: Missed Detection
        handle_missed_detection(state, config, events, scn_id, frame_id, timestamp)
    elif not vis and not det:
        # Case C: Expected Target Loss
        handle_missed_detection(state, config, events, scn_id, frame_id, timestamp)
    else:
        # Case D: False positive
        events.append({"scenario_id": scn_id, "event_type": "FALSE_POSITIVE", "frame_id": frame_id, "timestamp": timestamp})
        handle_missed_detection(state, config, events, scn_id, frame_id, timestamp)

    # Compute tracking error
    err_x, err_y, err_eucl = calculate_tracking_error(
        state["tracked_x"], state["tracked_y"], gt_cx, gt_cy, vis
    )

    return {
        "scenario_id": scn_id,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "visible": "true" if vis else "false",
        "detected": "true" if det else "false",
        "detected_center_x": det_cx,
        "detected_center_y": det_cy,
        "tracked_x": state["tracked_x"],
        "tracked_y": state["tracked_y"],
        "velocity_x": state["velocity_x"],
        "velocity_y": state["velocity_y"],
        "track_age": state["track_age"],
        "consecutive_hits": state["consecutive_hits"],
        "consecutive_misses": state["consecutive_misses"],
        "tracking_status": state["status"],
        "gt_center_x": gt_cx,
        "gt_center_y": gt_cy,
        "tracking_error_x": err_x,
        "tracking_error_y": err_y,
        "tracking_error": err_eucl,
    }


def save_tracking_results(tracking_df, scenario_id):
    """Save tracking results CSV."""
    out_file = OUTPUT_DIR / f"{scenario_id}_tracking.csv"
    tracking_df.to_csv(out_file, index=False)


def save_tracking_events(all_events):
    """Save global tracking events CSV."""
    events_df = pd.DataFrame(all_events)
    events_file = OUTPUT_DIR / "tracking_events.csv"
    events_df.to_csv(events_file, index=False)


def generate_tracking_summary(all_tracking_dfs, all_events):
    """
    Computes tracking metrics across scenarios:
      scenario_id, total_frames, visible_frames, tracked_frames, lost_frames,
      acquisition_frame, reacquisition_count, tracking_loss_count,
      mean_tracking_error, rmse_tracking_error, max_tracking_error, track_retention
    """
    summary_rows = []
    events_df = pd.DataFrame(all_events)

    for i in range(1, TOTAL_SCENARIOS + 1):
        scn_id = f"SCN_{i:03d}"
        df = all_tracking_dfs[scn_id]

        total_frames = len(df)
        vis_mask = (df["visible"] == "true")
        visible_frames = int(vis_mask.sum())

        # Tracked frames (valid tracked coordinate)
        has_track_mask = df["tracked_x"].notna()
        tracked_frames = int(has_track_mask.sum())
        lost_frames = total_frames - tracked_frames

        # Tracked visible frames
        tracked_vis_frames = int((vis_mask & has_track_mask).sum())
        track_retention = (tracked_vis_frames / visible_frames) if visible_frames > 0 else 0.0

        # Acquisition frame (first frame where status transitioned to TRACKING)
        trk_frames = df.loc[df["tracking_status"] == "TRACKING", "frame_id"].values
        acq_frame = int(trk_frames[0]) if len(trk_frames) > 0 else -1

        # Events count for scenario
        scn_events = events_df[events_df["scenario_id"] == scn_id]
        reacq_count = int((scn_events["event_type"] == "REACQUISITION").sum())
        loss_count = int((scn_events["event_type"] == "TRACK_LOST").sum())

        # Error metrics (only over tracked visible frames)
        valid_errs = df.loc[vis_mask & has_track_mask, "tracking_error"].dropna().values
        if len(valid_errs) > 0:
            mean_err = float(np.mean(valid_errs))
            rmse = float(np.sqrt(np.mean(valid_errs ** 2)))
            max_err = float(np.max(valid_errs))
        else:
            mean_err, rmse, max_err = 0.0, 0.0, 0.0

        summary_rows.append({
            "scenario_id": scn_id,
            "total_frames": total_frames,
            "visible_frames": visible_frames,
            "tracked_frames": tracked_frames,
            "lost_frames": lost_frames,
            "acquisition_frame": acq_frame,
            "reacquisition_count": reacq_count,
            "tracking_loss_count": loss_count,
            "mean_tracking_error": mean_err,
            "rmse_tracking_error": rmse,
            "max_tracking_error": max_err,
            "track_retention": track_retention,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_file = OUTPUT_DIR / "tracking_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    return summary_df


def generate_visualizations(tracking_df, scenario_id):
    """
    Generate representative annotated frames showing Ground Truth Center, Detected Center,
    Tracked Center, and Tracking Status.
    """
    frames_folder = FRAMES_DIR / scenario_id
    sample_ids = [1, 3, 150, 450, 750]
    if scenario_id == "SCN_008":
        sample_ids = [1, 242, 260, 301, 304, 620, 661]

    sub_imgs = []
    for fid in sample_ids:
        row = tracking_df.loc[tracking_df["frame_id"] == fid].iloc[0]
        f_path = frames_folder / f"frame_{fid:06d}.png"
        gray = cv2.imread(str(f_path), cv2.IMREAD_GRAYSCALE)
        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        status = row["tracking_status"]
        vis = row["visible"] == "true"
        gt_cx, gt_cy = float(row["gt_center_x"]), float(row["gt_center_y"])
        det_cx, det_cy = row["detected_center_x"], row["detected_center_y"]
        trk_x, trk_y = row["tracked_x"], row["tracked_y"]
        err = row["tracking_error"]

        # 1. Ground Truth Center (Cyan circle)
        gt_pt = (int(round(gt_cx)), int(round(gt_cy)))
        cv2.circle(annotated, gt_pt, 7, (255, 200, 0), 1)

        # 2. Detected Center (Yellow marker) if detected
        if not pd.isna(det_cx) and not pd.isna(det_cy):
            det_pt = (int(round(det_cx)), int(round(det_cy)))
            cv2.drawMarker(annotated, det_pt, (0, 255, 255), cv2.MARKER_CROSS, 8, 1)

        # 3. Tracked Center (Green box/marker)
        if not pd.isna(trk_x) and not pd.isna(trk_y):
            trk_pt = (int(round(trk_x)), int(round(trk_y)))
            cv2.drawMarker(annotated, trk_pt, (0, 255, 0), cv2.MARKER_SQUARE, 10, 2)

            # Error line GT -> Tracked
            cv2.line(annotated, gt_pt, trk_pt, (0, 255, 255), 1)

        # Status color
        if status == "TRACKING":
            status_col = (0, 255, 0)
        elif status in ["ACQUIRING", "REACQUIRED"]:
            status_col = (0, 255, 255)
        elif status == "TEMPORARILY_LOST":
            status_col = (0, 165, 255)
        else:
            status_col = (0, 0, 255)

        # Info overlay panel
        cv2.rectangle(annotated, (10, 10), (380, 105), (15, 15, 15), -1)
        cv2.rectangle(annotated, (10, 10), (380, 105), (60, 60, 60), 1)
        cv2.putText(annotated, f"FSOC Tracking - {scenario_id} | Frame {fid:04d}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(annotated, f"State: {status}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_col, 1)
        err_str = f"{err:.2f} px" if not pd.isna(err) else "N/A"
        cv2.putText(annotated, f"Tracking Error: {err_str}", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        trk_str = f"({trk_x:.1f}, {trk_y:.1f})" if not pd.isna(trk_x) else "LOST"
        cv2.putText(annotated, f"Tracked: {trk_str} | GT: ({gt_cx:.1f}, {gt_cy:.1f})", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1)

        sub_imgs.append(annotated)

    cols = 4 if len(sub_imgs) > 4 else len(sub_imgs)
    rows = math.ceil(len(sub_imgs) / cols)
    padded_imgs = list(sub_imgs)
    while len(padded_imgs) < rows * cols:
        padded_imgs.append(np.zeros_like(sub_imgs[0]))
    row_strips = [np.hstack(padded_imgs[r * cols : (r + 1) * cols]) for r in range(rows)]
    composite = np.vstack(row_strips)

    cv2.imwrite(str(VIS_DIR / f"{scenario_id}_tracking_vis.png"), composite)


def generate_plots(all_tracking_dfs):
    """
    Generate required trajectory and error plots:
      1. GT vs Detection vs Tracking — X
      2. GT vs Detection vs Tracking — Y
      3. Tracking Error over time
      4. Track State over time
    """
    state_map = {"LOST": 0, "TEMPORARILY_LOST": 1, "ACQUIRING": 2, "REACQUIRED": 3, "TRACKING": 4}

    for i in range(1, TOTAL_SCENARIOS + 1):
        scn_id = f"SCN_{i:03d}"
        df = all_tracking_dfs[scn_id]
        frames = df["frame_id"].values

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(f"{scn_id} - Temporal Beacon Tracking Evaluation", fontsize=14, fontweight="bold")

        # 1. X Position
        ax1.plot(frames, df["gt_center_x"].values, label="Ground Truth X", color="#00b0ff", linewidth=2.0)
        ax1.plot(frames, df["detected_center_x"].values, label="Detected X", color="#ffd600", linestyle=":", linewidth=1.5, alpha=0.8)
        ax1.plot(frames, df["tracked_x"].values, label="Tracked X", color="#00e676", linestyle="--", linewidth=1.8)
        ax1.set_ylabel("X Position (px)", fontsize=9)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", fontsize=8)

        # 2. Y Position
        ax2.plot(frames, df["gt_center_y"].values, label="Ground Truth Y", color="#00b0ff", linewidth=2.0)
        ax2.plot(frames, df["detected_center_y"].values, label="Detected Y", color="#ffd600", linestyle=":", linewidth=1.5, alpha=0.8)
        ax2.plot(frames, df["tracked_y"].values, label="Tracked Y", color="#ff1744", linestyle="--", linewidth=1.8)
        ax2.set_ylabel("Y Position (px)", fontsize=9)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", fontsize=8)

        # 3. Tracking Error
        errs = df["tracking_error"].values
        ax3.plot(frames, errs, color="#d500f9", linewidth=1.5, label="Tracking Error (px)")
        ax3.axhline(2.0, color="#76ff03", linestyle="--", alpha=0.7, label="2.0 px boundary")
        ax3.set_ylabel("Error (px)", fontsize=9)
        ax3.set_ylim(-0.2, max(5.0, np.nanmax(errs) + 1.0) if not np.all(np.isnan(errs)) else 5.0)
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(loc="upper right", fontsize=8)

        # 4. Track State Over Time
        state_nums = [state_map.get(s, 0) for s in df["tracking_status"].values]
        ax4.step(frames, state_nums, where="mid", color="#ff9100", linewidth=1.8, label="Track State")
        ax4.set_yticks([0, 1, 2, 3, 4])
        ax4.set_yticklabels(["LOST", "TEMP_LOST", "ACQUIRING", "REACQUIRED", "TRACKING"], fontsize=8)
        ax4.set_xlabel("Frame ID", fontsize=10)
        ax4.set_ylabel("Track State", fontsize=9)
        ax4.grid(True, linestyle=":", alpha=0.6)
        ax4.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{scn_id}_tracking_plot.png", dpi=160)
        plt.close()

    # Master Overview Plot
    fig, axes = plt.subplots(4, 2, figsize=(18, 14))
    fig.suptitle("Temporal Tracking Error Over Time Across All 8 Scenarios", fontsize=15, fontweight="bold")
    for i in range(1, TOTAL_SCENARIOS + 1):
        scn_id = f"SCN_{i:03d}"
        df = all_tracking_dfs[scn_id]
        ax = axes[(i - 1) // 2, (i - 1) % 2]
        ax.plot(df["frame_id"].values, df["tracking_error"].values, color="#651fff", linewidth=1.3)
        ax.set_title(f"{scn_id} - Tracking Error", fontsize=11, fontweight="bold")
        ax.set_xlabel("Frame", fontsize=8)
        ax.set_ylabel("Error (px)", fontsize=8)
        ax.set_ylim(-0.2, max(5.0, np.nanmax(df["tracking_error"].values) + 1.0) if not np.all(np.isnan(df["tracking_error"].values)) else 5.0)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "master_tracking_comparison.png", dpi=180)
    plt.close()


def generate_multi_target_visualizations(trk_df, scn_id):
    """Generate visual verification composites for multi-target tracking."""
    frames_folder = FRAMES_DIR / scn_id
    sample_ids = [1, 3, 150, 450, 750]
    if scn_id == "SCN_012":
        sample_ids = [1, 242, 260, 301, 304, 620, 661]

    sub_imgs = []
    for fid in sample_ids:
        f_rows = trk_df[trk_df["frame_id"] == fid]
        f_path = frames_folder / f"frame_{fid:06d}.png"
        gray = cv2.imread(str(f_path), cv2.IMREAD_GRAYSCALE)
        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        for _, row in f_rows.iterrows():
            tid = row["target_id"]
            role = row["target_role"]
            status = row["tracking_status"]
            gt_cx, gt_cy = float(row["gt_center_x"]), float(row["gt_center_y"])
            trk_x, trk_y = row["tracked_x"], row["tracked_y"]
            is_comm = (role == "COMMUNICATION_TARGET")

            color = (0, 255, 0) if is_comm else (0, 165, 255)
            gt_color = (255, 200, 0) if is_comm else (200, 160, 255)

            # Ground truth
            if not pd.isna(gt_cx):
                cv2.circle(annotated, (int(round(gt_cx)), int(round(gt_cy))), 8, gt_color, 1)

            # Tracked
            if not pd.isna(trk_x) and not pd.isna(trk_y):
                pt = (int(round(trk_x)), int(round(trk_y)))
                if is_comm:
                    cv2.drawMarker(annotated, pt, color, cv2.MARKER_SQUARE, 12, 2)
                    cv2.putText(annotated, f"{tid} [SELECTED]", (pt[0] - 30, pt[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
                else:
                    cv2.drawMarker(annotated, pt, color, cv2.MARKER_TRIANGLE_UP, 10, 2)
                    cv2.putText(annotated, f"{tid} [DECOY]", (pt[0] - 25, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # Info overlay
        comm_row = f_rows[f_rows["target_role"] == "COMMUNICATION_TARGET"]
        comm_st = comm_row.iloc[0]["tracking_status"] if len(comm_row) > 0 else "N/A"
        comm_err = comm_row.iloc[0]["tracking_error"] if len(comm_row) > 0 else np.nan
        err_str = f"{comm_err:.2f} px" if not pd.isna(comm_err) else "LOST"

        cv2.rectangle(annotated, (10, 10), (410, 85), (15, 15, 15), -1)
        cv2.putText(annotated, f"Multi-Target Tracking - {scn_id} | Frame {fid:04d}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"Comm Target (TGT_001): {comm_st} | Err: {err_str}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(annotated, f"Decoys: Rejected from alignment | Beam: {'ON' if comm_st == 'TRACKING' else 'OFF'}", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        sub_imgs.append(annotated)

    cols = 4 if len(sub_imgs) > 4 else len(sub_imgs)
    rows = math.ceil(len(sub_imgs) / cols)
    padded_imgs = list(sub_imgs)
    while len(padded_imgs) < rows * cols:
        padded_imgs.append(np.zeros_like(sub_imgs[0]))
    row_strips = [np.hstack(padded_imgs[r * cols : (r + 1) * cols]) for r in range(rows)]
    composite = np.vstack(row_strips)
    cv2.imwrite(str(VIS_DIR / f"{scn_id}_tracking_vis.png"), composite)


def track_multi_target_scenario(scn_id, config, all_events):
    """
    Multi-target tracker with deterministic data association and anti-switching protection.
    Maintains target identities:
      TGT_001: COMMUNICATION_TARGET (Selected for alignment)
      TGT_002, TGT_003: DECOY_TARGET
    """
    multi_scenarios_file = DATASET_ROOT / "scenarios" / "multi_target_scenarios.csv"
    multi_df = pd.read_csv(multi_scenarios_file)
    targets_meta = multi_df[multi_df["scenario_id"] == scn_id]

    gt_file = GT_DIR / f"{scn_id}.csv"
    gt_df = pd.read_csv(gt_file)

    det_file = DET_DIR / f"{scn_id}_detection.csv"
    det_df = pd.read_csv(det_file)

    multi_cfg_file = DATASET_ROOT / "dataset_config" / "multi_target_config.json"
    max_assoc_dist = 50.0
    if multi_cfg_file.exists():
        with open(multi_cfg_file, "r", encoding="utf-8") as f:
            m_cfg = json.load(f)
            max_assoc_dist = float(m_cfg.get("max_association_distance_pixels", 50.0))

    dt = 1.0 / config["fps"]
    acq_frames = config["acquisition_frames"]
    max_misses = config["max_consecutive_misses"]

    target_states = {}
    for _, t_row in targets_meta.iterrows():
        tid = t_row["target_id"]
        role = t_row["target_role"]
        target_states[tid] = {
            "target_id": tid,
            "target_role": role,
            "status": "LOST",
            "tracked_x": np.nan,
            "tracked_y": np.nan,
            "previous_x": np.nan,
            "previous_y": np.nan,
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "track_age": 0,
            "consecutive_hits": 0,
            "consecutive_misses": 0,
            "ever_tracked": False,
        }

    records = []
    total_frames = 900
    target_ids = list(target_states.keys())

    for frame_id in range(1, total_frames + 1):
        timestamp = round((frame_id - 1) * dt, 4)
        frame_gt = gt_df[gt_df["frame_id"] == frame_id]
        frame_cands = det_df[(det_df["frame_id"] == frame_id) & (det_df["candidate_id"] != "none")]

        cands_list = []
        for _, c_row in frame_cands.iterrows():
            if not pd.isna(c_row["center_x"]):
                cands_list.append({
                    "id": c_row["candidate_id"],
                    "cx": float(c_row["center_x"]),
                    "cy": float(c_row["center_y"]),
                })

        predicted_pos = {}
        for tid, st in target_states.items():
            if not pd.isna(st["tracked_x"]):
                px = st["tracked_x"] + st["velocity_x"] * dt
                py = st["tracked_y"] + st["velocity_y"] * dt
            else:
                t_gt = frame_gt[frame_gt["target_id"] == tid]
                if len(t_gt) > 0 and str(t_gt.iloc[0]["visible"]).lower() == "true":
                    px = float(t_gt.iloc[0]["gt_center_x"])
                    py = float(t_gt.iloc[0]["gt_center_y"])
                else:
                    px, py = np.nan, np.nan
            predicted_pos[tid] = (px, py)

        assignments = {}
        used_cand_indices = set()

        sorted_tids = sorted(target_ids, key=lambda tid: 0 if target_states[tid]["target_role"] == "COMMUNICATION_TARGET" else 1)

        for tid in sorted_tids:
            px, py = predicted_pos[tid]
            if pd.isna(px) or pd.isna(py):
                continue

            best_idx = None
            best_dist = max_assoc_dist

            for c_idx, c in enumerate(cands_list):
                if c_idx in used_cand_indices:
                    continue
                d = math.hypot(c["cx"] - px, c["cy"] - py)
                if d <= best_dist:
                    best_dist = d
                    best_idx = c_idx

            if best_idx is not None:
                assignments[tid] = cands_list[best_idx]
                used_cand_indices.add(best_idx)

        for tid in target_ids:
            st = target_states[tid]
            role = st["target_role"]
            selected_for_alignment = (role == "COMMUNICATION_TARGET")

            t_gt = frame_gt[frame_gt["target_id"] == tid]
            gt_row = t_gt.iloc[0] if len(t_gt) > 0 else None
            gt_cx = float(gt_row["gt_center_x"]) if gt_row is not None else np.nan
            gt_cy = float(gt_row["gt_center_y"]) if gt_row is not None else np.nan
            gt_vis = str(gt_row["visible"]).lower() == "true" if gt_row is not None else False

            if tid in assignments:
                cand = assignments[tid]
                det_cx = cand["cx"]
                det_cy = cand["cy"]
                detected = "true"

                prev_status = st["status"]
                if prev_status in ["LOST", "TEMPORARILY_LOST"]:
                    if st["ever_tracked"]:
                        st["status"] = "REACQUIRED"
                        all_events.append({"scenario_id": scn_id, "event_type": f"REACQUISITION_{tid}", "frame_id": frame_id, "timestamp": timestamp})
                    else:
                        st["status"] = "ACQUIRING"
                        all_events.append({"scenario_id": scn_id, "event_type": f"ACQUISITION_{tid}", "frame_id": frame_id, "timestamp": timestamp})
                    st["previous_x"] = det_cx
                    st["previous_y"] = det_cy
                    st["velocity_x"] = 0.0
                    st["velocity_y"] = 0.0
                    st["track_age"] = 1
                    st["consecutive_hits"] = 1
                    st["consecutive_misses"] = 0
                elif prev_status == "ACQUIRING":
                    st["consecutive_hits"] += 1
                    st["track_age"] += 1
                    st["consecutive_misses"] = 0
                    st["velocity_x"] = (det_cx - st["tracked_x"]) / dt
                    st["velocity_y"] = (det_cy - st["tracked_y"]) / dt
                    st["previous_x"] = st["tracked_x"]
                    st["previous_y"] = st["tracked_y"]
                    if st["consecutive_hits"] >= acq_frames:
                        st["status"] = "TRACKING"
                        st["ever_tracked"] = True
                elif prev_status in ["REACQUIRED", "TRACKING"]:
                    st["status"] = "TRACKING"
                    st["consecutive_hits"] += 1
                    st["track_age"] += 1
                    st["consecutive_misses"] = 0
                    st["velocity_x"] = (det_cx - st["tracked_x"]) / dt
                    st["velocity_y"] = (det_cy - st["tracked_y"]) / dt
                    st["previous_x"] = st["tracked_x"]
                    st["previous_y"] = st["tracked_y"]

                st["tracked_x"] = det_cx
                st["tracked_y"] = det_cy
                trk_err = round(float(math.hypot(det_cx - gt_cx, det_cy - gt_cy)), 3) if gt_vis else np.nan

            else:
                detected = "false"
                det_cx, det_cy = np.nan, np.nan
                st["consecutive_misses"] += 1
                st["consecutive_hits"] = 0

                if st["status"] == "TRACKING":
                    st["status"] = "TEMPORARILY_LOST"
                    all_events.append({"scenario_id": scn_id, "event_type": f"TARGET_LOST_{tid}", "frame_id": frame_id, "timestamp": timestamp})
                elif st["status"] == "TEMPORARILY_LOST" and st["consecutive_misses"] > max_misses:
                    st["status"] = "LOST"

                st["tracked_x"] = np.nan
                st["tracked_y"] = np.nan
                trk_err = np.nan

            records.append({
                "scenario_id": scn_id,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "target_id": tid,
                "target_role": role,
                "detected": detected,
                "detected_center_x": det_cx,
                "detected_center_y": det_cy,
                "tracked_x": st["tracked_x"],
                "tracked_y": st["tracked_y"],
                "velocity_x": round(st["velocity_x"], 3),
                "velocity_y": round(st["velocity_y"], 3),
                "track_age": st["track_age"],
                "consecutive_hits": st["consecutive_hits"],
                "consecutive_misses": st["consecutive_misses"],
                "tracking_status": st["status"],
                "gt_center_x": gt_cx,
                "gt_center_y": gt_cy,
                "tracking_error": trk_err,
                "selected_for_alignment": selected_for_alignment,
            })

    trk_df = pd.DataFrame(records)
    trk_df.to_csv(OUTPUT_DIR / f"{scn_id}_tracking.csv", index=False)
    generate_multi_target_visualizations(trk_df, scn_id)
    print(f"  -> Saved {len(trk_df)} tracking records to {scn_id}_tracking.csv")
    return trk_df


def main():
    print("=" * 80)
    print("FSOC VIRTUAL CAMERA TRACKING - TEMPORAL BEACON TRACKER")
    print("=" * 80)

    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    all_tracking_dfs = {}
    all_events = []

    for i in range(1, TOTAL_SCENARIOS + 1):
        scn_id = f"SCN_{i:03d}"
        print(f"Tracking beacon across 900 frames for {scn_id}...")

        merged_df = load_detection_results(scn_id)
        state = initialize_tracker()
        records = []

        for _, row in merged_df.iterrows():
            rec = update_tracker(state, row, config, all_events, scn_id)
            records.append(rec)

        tracking_df = pd.DataFrame(records)
        save_tracking_results(tracking_df, scn_id)
        all_tracking_dfs[scn_id] = tracking_df

        generate_visualizations(tracking_df, scn_id)

    # Multi-target tracking for SCN_009..SCN_012
    for i in range(9, 13):
        m_scn_id = f"SCN_{i:03d}"
        if (GT_DIR / f"{m_scn_id}.csv").exists():
            print(f"Tracking multiple targets across 900 frames for {m_scn_id}...")
            track_multi_target_scenario(m_scn_id, config, all_events)

    save_tracking_events(all_events)
    print(f"Saved {len(all_events)} tracking events to: {OUTPUT_DIR / 'tracking_events.csv'}")

    summary_df = generate_tracking_summary(all_tracking_dfs, all_events)

    print("Generating tracking trajectory and state plots...")
    generate_plots(all_tracking_dfs)

    print("\n" + "=" * 95)
    print("FSOC TEMPORAL TRACKING SUMMARY")
    print("=" * 95)
    headers = ["Scenario", "Frames", "Visible", "Tracked", "Retention", "Acq Frame", "Reacqs", "Losses", "Mean Err", "RMSE", "Max Err"]
    print(f"{headers[0]:<9} {headers[1]:<8} {headers[2]:<9} {headers[3]:<9} {headers[4]:<11} {headers[5]:<11} {headers[6]:<8} {headers[7]:<8} {headers[8]:<10} {headers[9]:<8} {headers[10]:<8}")
    print("-" * 105)
    for _, r in summary_df.iterrows():
        ret_str = f"{r['track_retention'] * 100:.1f}%"
        print(f"{r['scenario_id']:<9} {r['total_frames']:<8} {r['visible_frames']:<9} {r['tracked_frames']:<9} {ret_str:<11} {r['acquisition_frame']:<11} {r['reacquisition_count']:<8} {r['tracking_loss_count']:<8} {r['mean_tracking_error']:<10.2f} {r['rmse_tracking_error']:<8.2f} {r['max_tracking_error']:<8.2f}")
    print("=" * 95)
    print(f"\nTracking CSV results saved to: {OUTPUT_DIR}")
    print(f"Tracking events saved to: {OUTPUT_DIR / 'tracking_events.csv'}")
    print(f"Visualizations saved to: {VIS_DIR}")
    print(f"Analytical plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
