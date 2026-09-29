"""
FSOC Virtual Camera Tracking System - Alignment Error & Virtual Pan-Tilt Controller
Simulates virtual camera pointing response based on tracked beacon angular errors.
Maintains two-axis virtual pan/tilt orientation, closed-loop proportional control,
event logging, analytical plots, and visual simulation videos.
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
TRACKING_DIR = DATASET_ROOT / "tracking_results"
FRAMES_DIR = DATASET_ROOT / "frames"
CONFIG_FILE = DATASET_ROOT / "dataset_config" / "config.json"
ALIGN_CONFIG_FILE = DATASET_ROOT / "dataset_config" / "alignment_config.json"
OUTPUT_DIR = DATASET_ROOT / "alignment_results"
VIS_DIR = OUTPUT_DIR / "visualizations"
PLOTS_DIR = OUTPUT_DIR / "plots"

TOTAL_SCENARIOS = 8


def load_camera_config():
    """Load and merge camera and alignment configuration parameters."""
    cfg = {
        "resolution": [640, 480],
        "default_fov": [4, 3],
        "fps": 30,
        "kp_pan": 0.8,
        "kp_tilt": 0.8,
        "max_pan_speed": 5.0,
        "max_tilt_speed": 5.0,
        "alignment_threshold_horizontal": 0.05,
        "alignment_threshold_vertical": 0.05,
        "alignment_stability_frames": 5,
    }
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            base_cfg = json.load(f)
            cfg.update(base_cfg)

    if ALIGN_CONFIG_FILE.exists():
        with open(ALIGN_CONFIG_FILE, "r", encoding="utf-8") as f:
            align_cfg = json.load(f)
            cfg.update(align_cfg)

    cfg["image_width"] = cfg["resolution"][0]
    cfg["image_height"] = cfg["resolution"][1]
    cfg["image_center_x"] = cfg["image_width"] / 2.0
    cfg["image_center_y"] = cfg["image_height"] / 2.0
    cfg["horizontal_fov"] = float(cfg["default_fov"][0])
    cfg["vertical_fov"] = float(cfg["default_fov"][1])
    return cfg


def load_tracking_data(scenario_id):
    """Load tracking CSV for a given scenario."""
    track_file = TRACKING_DIR / f"{scenario_id}_tracking.csv"
    if not track_file.exists():
        raise FileNotFoundError(f"Missing tracking file: {track_file}")
    return pd.read_csv(track_file)


def calculate_pixel_error(tracked_x, tracked_y, center_x, center_y):
    """
    Calculate image-space error relative to camera optical boresight:
      error_pixel_x = tracked_x - image_center_x
      error_pixel_y = tracked_y - image_center_y
    """
    if pd.isna(tracked_x) or pd.isna(tracked_y):
        return np.nan, np.nan
    return float(tracked_x - center_x), float(tracked_y - center_y)


def calculate_angular_error(pixel_err_x, pixel_err_y, img_w, img_h, fov_h, fov_v):
    """
    Convert image-space pixel error to angular displacement (degrees):
      target_angle_x = (error_pixel_x / image_width) * horizontal_fov
      target_angle_y = (error_pixel_y / image_height) * vertical_fov
    """
    if pd.isna(pixel_err_x) or pd.isna(pixel_err_y):
        return np.nan, np.nan
    ang_x = (pixel_err_x / float(img_w)) * fov_h
    ang_y = (pixel_err_y / float(img_h)) * fov_v
    return float(ang_x), float(ang_y)


def calculate_remaining_error(target_ang_x, target_ang_y, current_pan, current_tilt):
    """
    Calculate remaining angular error relative to current virtual camera pointing:
      remaining_error_x = target_angle_x - current_pan
      remaining_error_y = target_angle_y - current_tilt
    """
    if pd.isna(target_ang_x) or pd.isna(target_ang_y):
        return np.nan, np.nan
    return float(target_ang_x - current_pan), float(target_ang_y - current_tilt)


def check_alignment(rem_err_x, rem_err_y, th_h, th_v):
    """
    Check if remaining angular error is within coarse alignment thresholds:
      |remaining_error_x| <= threshold_h AND |remaining_error_y| <= threshold_v
    """
    if pd.isna(rem_err_x) or pd.isna(rem_err_y):
        return False
    return abs(rem_err_x) <= th_h and abs(rem_err_y) <= th_v


def update_pan_tilt(rem_err_x, rem_err_y, current_pan, current_tilt, cfg):
    """
    Compute closed-loop proportional pan/tilt commands and update camera orientation:
      pan_cmd = clip(kp_pan * rem_err_x, -pan_step, pan_step)
      tilt_cmd = clip(kp_tilt * rem_err_y, -tilt_step, tilt_step)
    """
    if pd.isna(rem_err_x) or pd.isna(rem_err_y):
        return 0.0, 0.0, current_pan, current_tilt

    fps = cfg["fps"]
    pan_step = cfg["max_pan_speed"] / float(fps)
    tilt_step = cfg["max_tilt_speed"] / float(fps)

    raw_pan_cmd = cfg["kp_pan"] * rem_err_x
    raw_tilt_cmd = cfg["kp_tilt"] * rem_err_y

    pan_cmd = float(np.clip(raw_pan_cmd, -pan_step, pan_step))
    tilt_cmd = float(np.clip(raw_tilt_cmd, -tilt_step, tilt_step))

    new_pan = float(current_pan + pan_cmd)
    new_tilt = float(current_tilt + tilt_cmd)
    return pan_cmd, tilt_cmd, new_pan, new_tilt


def handle_tracking_loss(current_pan, current_tilt):
    """Zero control commands during tracking loss."""
    return 0.0, 0.0, current_pan, current_tilt


def handle_reacquisition(events, scn_id, frame_id, timestamp):
    """Log tracking reacquisition and resume alignment events."""
    events.append({"scenario_id": scn_id, "event_type": "TRACK_REACQUIRED", "frame_id": frame_id, "timestamp": timestamp})
    events.append({"scenario_id": scn_id, "event_type": "ALIGNMENT_STARTED", "frame_id": frame_id, "timestamp": timestamp})


def save_alignment_results(alignment_df, scenario_id):
    """Save alignment results CSV."""
    out_file = OUTPUT_DIR / f"{scenario_id}_alignment.csv"
    alignment_df.to_csv(out_file, index=False)


def generate_alignment_events(all_events):
    """Save all alignment events to CSV."""
    events_df = pd.DataFrame(all_events)
    events_file = OUTPUT_DIR / "alignment_events.csv"
    events_df.to_csv(events_file, index=False)


def generate_summary(all_alignment_dfs, cfg):
    """
    Compute summary metrics:
      scenario_id, total_frames, tracked_frames, alignment_frames, aligned_frames,
      alignment_rate, initial_angular_error, final_angular_error, maximum_angular_error,
      mean_angular_error, time_to_alignment
    """
    rows = []
    fps = cfg["fps"]
    stability_thresh = cfg["alignment_stability_frames"]

    for i in range(1, TOTAL_SCENARIOS + 1):
        scn_id = f"SCN_{i:03d}"
        df = all_alignment_dfs[scn_id]

        total_frames = len(df)
        tracked_mask = df["tracked_x"].notna()
        tracked_frames = int(tracked_mask.sum())

        aligned_mask = (df["alignment_status"] == "ALIGNED")
        aligning_mask = (df["alignment_status"] == "ALIGNING")
        aligned_frames = int(aligned_mask.sum())
        alignment_frames = int((aligned_mask | aligning_mask).sum())

        alignment_rate = (aligned_frames / tracked_frames) if tracked_frames > 0 else 0.0

        # Angular errors over tracked frames
        valid_errs = np.hypot(df.loc[tracked_mask, "remaining_error_x"], df.loc[tracked_mask, "remaining_error_y"]).values

        if len(valid_errs) > 0:
            init_err = float(valid_errs[0])
            final_err = float(valid_errs[-1])
            max_err = float(np.max(valid_errs))
            mean_err = float(np.mean(valid_errs))
        else:
            init_err, final_err, max_err, mean_err = 0.0, 0.0, 0.0, 0.0

        # Time to stable alignment
        time_to_align = np.nan
        consec_aligned = 0
        first_align_frame = None

        for _, row in df.iterrows():
            if row["alignment_status"] == "ALIGNED":
                consec_aligned += 1
                if consec_aligned >= stability_thresh and first_align_frame is None:
                    first_align_frame = int(row["frame_id"]) - stability_thresh + 1
                    time_to_align = float((first_align_frame - 1) / float(fps))
                    break
            else:
                consec_aligned = 0

        rows.append({
            "scenario_id": scn_id,
            "total_frames": total_frames,
            "tracked_frames": tracked_frames,
            "alignment_frames": alignment_frames,
            "aligned_frames": aligned_frames,
            "alignment_rate": round(alignment_rate, 4),
            "initial_angular_error": round(init_err, 4),
            "final_angular_error": round(final_err, 4),
            "maximum_angular_error": round(max_err, 4),
            "mean_angular_error": round(mean_err, 4),
            "time_to_alignment": round(time_to_align, 3) if not np.isnan(time_to_align) else np.nan,
        })

    summary_df = pd.DataFrame(rows)
    summary_file = OUTPUT_DIR / "alignment_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    return summary_df


def generate_visualization(alignment_df, scenario_id, cfg):
    """
    Generate an annotated simulation video (640x480, 30 FPS, 900 frames)
    showing virtual camera pointing, target, boresight, and telemetry.
    """
    frames_folder = FRAMES_DIR / scenario_id
    out_video_path = VIS_DIR / f"{scenario_id}_alignment.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(out_video_path),
        fourcc,
        cfg["fps"],
        (cfg["image_width"], cfg["image_height"]),
        isColor=True,
    )

    cx = int(round(cfg["image_center_x"]))
    cy = int(round(cfg["image_center_y"]))
    fov_h = cfg["horizontal_fov"]
    fov_v = cfg["vertical_fov"]
    px_per_deg_x = cfg["image_width"] / fov_h
    px_per_deg_y = cfg["image_height"] / fov_v

    for _, row in alignment_df.iterrows():
        fid = int(row["frame_id"])
        timestamp = float(row["timestamp"])
        status = row["alignment_status"]

        f_path = frames_folder / f"frame_{fid:06d}.png"
        gray = cv2.imread(str(f_path), cv2.IMREAD_GRAYSCALE)
        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 1. Fixed Optical Sensor Boresight Center Crosshair (+)
        cv2.drawMarker(annotated, (cx, cy), (100, 100, 100), cv2.MARKER_CROSS, 20, 1)
        cv2.circle(annotated, (cx, cy), 4, (100, 100, 100), 1)

        # 2. Virtual Alignment Reticle (Shows where virtual camera is currently pointed)
        v_pan = float(row["virtual_pan"])
        v_tilt = float(row["virtual_tilt"])
        reticle_x = int(round(cx + v_pan * px_per_deg_x))
        reticle_y = int(round(cy + v_tilt * px_per_deg_y))

        # 3. Target Position
        tx = row["tracked_x"]
        ty = row["tracked_y"]

        if status == "ALIGNED":
            stat_col = (0, 255, 0)
        elif status == "ALIGNING":
            stat_col = (0, 220, 255)
        else:
            stat_col = (0, 0, 255)

        if not pd.isna(tx) and not pd.isna(ty):
            t_pt = (int(round(tx)), int(round(ty)))
            # Target ring (Cyan)
            cv2.circle(annotated, t_pt, 8, (255, 200, 0), 2)
            cv2.putText(annotated, "COMM_TARGET", (t_pt[0] + 12, t_pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

            # Error line from virtual reticle -> target
            cv2.line(annotated, (reticle_x, reticle_y), t_pt, (0, 255, 255), 1)

            # Virtual FSOC Optical Laser Beam: Active ONLY when coarse aligned with communication target
            if status == "ALIGNED":
                cv2.line(annotated, (cx, cy), t_pt, (0, 255, 0), 2)
                cv2.putText(annotated, "LASER BEAM: ACTIVE [LOCKED]", (cx - 70, cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Virtual Boresight Center Marker (Green/Yellow circle with crosshair)
        if 0 <= reticle_x < cfg["image_width"] and 0 <= reticle_y < cfg["image_height"]:
            cv2.circle(annotated, (reticle_x, reticle_y), 10, stat_col, 2)
            cv2.drawMarker(annotated, (reticle_x, reticle_y), stat_col, cv2.MARKER_CROSS, 12, 1)

        # Telemetry Dashboard HUD
        cv2.rectangle(annotated, (10, 10), (390, 125), (15, 15, 15), -1)
        cv2.rectangle(annotated, (10, 10), (390, 125), (60, 60, 60), 1)

        cv2.putText(annotated, f"FSOC Virtual Alignment - {scenario_id}", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(annotated, f"Frame: {fid:04d} | Time: {timestamp:.2f}s | Status: {status}", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, stat_col, 1)

        rx = row["remaining_error_x"]
        ry = row["remaining_error_y"]
        rx_str = f"{rx:.3f} deg" if not pd.isna(rx) else "N/A"
        ry_str = f"{ry:.3f} deg" if not pd.isna(ry) else "N/A"
        cv2.putText(annotated, f"Rem Error: Pan={rx_str} | Tilt={ry_str}", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        cv2.putText(annotated, f"Virtual PT : Pan={v_pan:+.3f} deg | Tilt={v_tilt:+.3f} deg", (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)

        p_cmd = row["pan_command"]
        t_cmd = row["tilt_command"]
        cv2.putText(annotated, f"Command    : Pan={p_cmd:+.3f} deg/f | Tilt={t_cmd:+.3f} deg/f", (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 220, 255), 1)

        writer.write(annotated)

    writer.release()


def generate_plots(all_alignment_dfs, cfg):
    """
    Generate all 6 required plots for each scenario:
      Plot 1: Horizontal angular error vs frame
      Plot 2: Vertical angular error vs frame
      Plot 3: Virtual pan angle vs frame
      Plot 4: Virtual tilt angle vs frame
      Plot 5: Remaining angular error vs frame
      Plot 6: Alignment status vs frame
    """
    status_map = {"LOST": 0, "NOT_TRACKED": 0, "ALIGNING": 1, "ALIGNED": 2}
    th_h = cfg["alignment_threshold_horizontal"]
    th_v = cfg["alignment_threshold_vertical"]

    for i in range(1, TOTAL_SCENARIOS + 1):
        scn_id = f"SCN_{i:03d}"
        df = all_alignment_dfs[scn_id]
        frames = df["frame_id"].values

        fig, axes = plt.subplots(6, 1, figsize=(14, 15), sharex=True)
        fig.suptitle(f"{scn_id} - Virtual Pan-Tilt Alignment Response", fontsize=14, fontweight="bold")

        # Plot 1: Horizontal Angular Error
        axes[0].plot(frames, df["target_angle_x"].values, label="Target Angle X", color="#00e5ff", linewidth=1.5)
        axes[0].plot(frames, df["remaining_error_x"].values, label="Remaining Error X", color="#ff1744", linestyle="--", linewidth=1.5)
        axes[0].axhline(th_h, color="#76ff03", linestyle=":", alpha=0.7, label=f"+{th_h} deg Threshold")
        axes[0].axhline(-th_h, color="#76ff03", linestyle=":", alpha=0.7)
        axes[0].set_ylabel("Pan Err (deg)", fontsize=9)
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend(loc="upper right", fontsize=8)

        # Plot 2: Vertical Angular Error
        axes[1].plot(frames, df["target_angle_y"].values, label="Target Angle Y", color="#00e5ff", linewidth=1.5)
        axes[1].plot(frames, df["remaining_error_y"].values, label="Remaining Error Y", color="#ff1744", linestyle="--", linewidth=1.5)
        axes[1].axhline(th_v, color="#76ff03", linestyle=":", alpha=0.7, label=f"+{th_v} deg Threshold")
        axes[1].axhline(-th_v, color="#76ff03", linestyle=":", alpha=0.7)
        axes[1].set_ylabel("Tilt Err (deg)", fontsize=9)
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend(loc="upper right", fontsize=8)

        # Plot 3: Virtual Pan Angle
        axes[2].plot(frames, df["virtual_pan"].values, label="Virtual Pan Angle", color="#651fff", linewidth=1.8)
        axes[2].set_ylabel("Pan (deg)", fontsize=9)
        axes[2].grid(True, linestyle=":", alpha=0.6)
        axes[2].legend(loc="upper right", fontsize=8)

        # Plot 4: Virtual Tilt Angle
        axes[3].plot(frames, df["virtual_tilt"].values, label="Virtual Tilt Angle", color="#d500f9", linewidth=1.8)
        axes[3].set_ylabel("Tilt (deg)", fontsize=9)
        axes[3].grid(True, linestyle=":", alpha=0.6)
        axes[3].legend(loc="upper right", fontsize=8)

        # Plot 5: Remaining Total Angular Error
        tot_rem = np.hypot(df["remaining_error_x"].values, df["remaining_error_y"].values)
        axes[4].plot(frames, tot_rem, label="Total Remaining Error", color="#ff9100", linewidth=1.8)
        axes[4].axhline(math.hypot(th_h, th_v), color="#76ff03", linestyle="--", alpha=0.8, label="Alignment Threshold")
        axes[4].set_ylabel("Total Err (deg)", fontsize=9)
        axes[4].grid(True, linestyle=":", alpha=0.6)
        axes[4].legend(loc="upper right", fontsize=8)

        # Plot 6: Alignment Status Over Time
        stat_vals = [status_map.get(s, 0) for s in df["alignment_status"].values]
        axes[5].step(frames, stat_vals, where="mid", color="#00e676", linewidth=1.8, label="Alignment Status")
        axes[5].set_yticks([0, 1, 2])
        axes[5].set_yticklabels(["LOST", "ALIGNING", "ALIGNED"], fontsize=8)
        axes[5].set_xlabel("Frame ID", fontsize=10)
        axes[5].set_ylabel("Status", fontsize=9)
        axes[5].grid(True, linestyle=":", alpha=0.6)
        axes[5].legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{scn_id}_alignment_plots.png", dpi=160)
        plt.close()


def process_scenario(scn_id, cfg, all_events):
    """
    Process full alignment control simulation for a scenario.
    In multi-target scenarios, alignment operates ONLY on the designated COMMUNICATION_TARGET.
    """
    raw_track_df = load_tracking_data(scn_id)
    if "selected_for_alignment" in raw_track_df.columns:
        track_df = raw_track_df[raw_track_df["selected_for_alignment"].astype(str).str.lower() == "true"].copy()
    elif "target_role" in raw_track_df.columns:
        track_df = raw_track_df[raw_track_df["target_role"] == "COMMUNICATION_TARGET"].copy()
    else:
        track_df = raw_track_df
    records = []

    current_pan = 0.0
    current_tilt = 0.0

    prev_status = "LOST"
    prev_align_status = "NOT_TRACKED"

    th_h = cfg["alignment_threshold_horizontal"]
    th_v = cfg["alignment_threshold_vertical"]
    img_w = cfg["image_width"]
    img_h = cfg["image_height"]
    cx = cfg["image_center_x"]
    cy = cfg["image_center_y"]
    fov_h = cfg["horizontal_fov"]
    fov_v = cfg["vertical_fov"]

    # Start event
    all_events.append({"scenario_id": scn_id, "event_type": "ALIGNMENT_STARTED", "frame_id": 1, "timestamp": 0.0})

    consecutive_aligned = 0

    for _, row in track_df.iterrows():
        fid = int(row["frame_id"])
        timestamp = float(row["timestamp"])
        trk_status = row["tracking_status"]
        tx = row["tracked_x"]
        ty = row["tracked_y"]

        is_lost = (trk_status == "LOST" or pd.isna(tx))

        # Check reacquisition from lost state
        if prev_status == "LOST" and not is_lost:
            handle_reacquisition(all_events, scn_id, fid, timestamp)

        if is_lost:
            # Case: Lost tracking
            px_err_x, px_err_y = np.nan, np.nan
            t_ang_x, t_ang_y = np.nan, np.nan
            rem_err_x, rem_err_y = np.nan, np.nan
            pan_cmd, tilt_cmd, current_pan, current_tilt = handle_tracking_loss(current_pan, current_tilt)
            align_status = "LOST"
            consecutive_aligned = 0

            if prev_align_status in ["ALIGNED", "ALIGNING"]:
                all_events.append({"scenario_id": scn_id, "event_type": "TRACK_LOST", "frame_id": fid, "timestamp": timestamp})
        else:
            # Step 1: Pixel Error
            px_err_x, px_err_y = calculate_pixel_error(tx, ty, cx, cy)

            # Step 2: Target Angular Offset
            t_ang_x, t_ang_y = calculate_angular_error(px_err_x, px_err_y, img_w, img_h, fov_h, fov_v)

            # Step 3: Remaining Angular Error relative to current camera pan/tilt
            rem_err_x, rem_err_y = calculate_remaining_error(t_ang_x, t_ang_y, current_pan, current_tilt)

            # Step 4: Check coarse alignment threshold
            is_aligned = check_alignment(rem_err_x, rem_err_y, th_h, th_v)

            if is_aligned:
                align_status = "ALIGNED"
                consecutive_aligned += 1
                if consecutive_aligned == cfg["alignment_stability_frames"]:
                    all_events.append({"scenario_id": scn_id, "event_type": "ALIGNED", "frame_id": fid, "timestamp": timestamp})
            else:
                align_status = "ALIGNING"
                if prev_align_status == "ALIGNED":
                    all_events.append({"scenario_id": scn_id, "event_type": "ALIGNMENT_LOST", "frame_id": fid, "timestamp": timestamp})
                consecutive_aligned = 0

            # Step 5: Proportional Closed-Loop Command and Orientation Update
            pan_cmd, tilt_cmd, current_pan, current_tilt = update_pan_tilt(
                rem_err_x, rem_err_y, current_pan, current_tilt, cfg
            )

        records.append({
            "scenario_id": scn_id,
            "frame_id": fid,
            "timestamp": timestamp,
            "tracking_status": trk_status,
            "tracked_x": tx,
            "tracked_y": ty,
            "image_center_x": cx,
            "image_center_y": cy,
            "pixel_error_x": px_err_x,
            "pixel_error_y": px_err_y,
            "target_angle_x": t_ang_x,
            "target_angle_y": t_ang_y,
            "remaining_error_x": rem_err_x,
            "remaining_error_y": rem_err_y,
            "pan_command": pan_cmd,
            "tilt_command": tilt_cmd,
            "virtual_pan": current_pan,
            "virtual_tilt": current_tilt,
            "alignment_status": align_status,
        })

        prev_status = trk_status
        prev_align_status = align_status

    alignment_df = pd.DataFrame(records)
    save_alignment_results(alignment_df, scn_id)
    return alignment_df


def main():
    print("=" * 80)
    print("FSOC VIRTUAL CAMERA ALIGNMENT & PAN-TILT CONTROL SIMULATION")
    print("=" * 80)

    cfg = load_camera_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    all_alignment_dfs = {}
    all_events = []

    for i in range(1, TOTAL_SCENARIOS + 1):
        scn_id = f"SCN_{i:03d}"
        print(f"Simulating pan-tilt alignment for {scn_id}...")
        df = process_scenario(scn_id, cfg, all_events)
        all_alignment_dfs[scn_id] = df

        print(f"  Generating visual simulation video for {scn_id}...")
    # Process multi-target scenarios SCN_009..SCN_012 for designated communication target
    for i in range(9, 13):
        m_scn_id = f"SCN_{i:03d}"
        if (TRACKING_DIR / f"{m_scn_id}_tracking.csv").exists():
            print(f"Simulating pan-tilt alignment for {m_scn_id} (Communication Target)...")
            df = process_scenario(m_scn_id, cfg, all_events)
            print(f"  Generating visual simulation video for {m_scn_id}...")
            generate_visualization(df, m_scn_id, cfg)

    generate_alignment_events(all_events)
    print(f"Saved {len(all_events)} alignment events to {OUTPUT_DIR / 'alignment_events.csv'}")

    print("Generating comprehensive alignment plots...")
    generate_plots(all_alignment_dfs, cfg)

    summary_df = generate_summary(all_alignment_dfs, cfg)

    print("\n" + "=" * 105)
    print("FSOC VIRTUAL ALIGNMENT SUMMARY")
    print("=" * 105)
    headers = ["Scenario", "Frames", "Tracked", "Aligned", "Align Rate", "Init Err", "Final Err", "Max Err", "Mean Err", "Time to Align"]
    print(f"{headers[0]:<9} {headers[1]:<8} {headers[2]:<9} {headers[3]:<9} {headers[4]:<12} {headers[5]:<10} {headers[6]:<11} {headers[7]:<10} {headers[8]:<10} {headers[9]:<14}")
    print("-" * 105)
    for _, r in summary_df.iterrows():
        t_align_str = f"{r['time_to_alignment']:.2f}s" if not pd.isna(r['time_to_alignment']) else "N/A"
        rate_str = f"{r['alignment_rate'] * 100:.1f}%"
        print(f"{r['scenario_id']:<9} {r['total_frames']:<8} {r['tracked_frames']:<9} {r['aligned_frames']:<9} {rate_str:<12} {r['initial_angular_error']:<10.3f} {r['final_angular_error']:<11.3f} {r['maximum_angular_error']:<10.3f} {r['mean_angular_error']:<10.3f} {t_align_str:<14}")
    print("=" * 105)

    print(f"\nAlignment results CSV files saved to: {OUTPUT_DIR}")
    print(f"Alignment simulation videos saved to: {VIS_DIR}")
    print(f"Analytical plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
