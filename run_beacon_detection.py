"""
FSOC Virtual Camera Tracking System - Classical Beacon Detection Pipeline
Reads the existing synthetic dataset and performs automated classical optical beacon detection
frame-by-frame across all 8 operational scenarios.

Outputs detection results CSVs and visual verification artifacts.
"""

import os
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "FSOC_DATASET"
FRAMES_DIR = DATASET_ROOT / "frames"
GT_DIR = DATASET_ROOT / "ground_truth"
CONFIG_FILE = DATASET_ROOT / "dataset_config" / "config.json"
RESULTS_DIR = DATASET_ROOT / "detection_results"
VIS_DIR = DATASET_ROOT / "detection_visualizations"


def load_config():
    """Load camera and dataset configuration."""
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_beacon_in_frame(gray_frame, is_low_light=False):
    """
    Classical Computer Vision Optical Beacon Detector.
    
    Pipeline:
      1. Gaussian Pre-filter (noise suppression)
      2. Morphological Top-Hat Transform (isolates localized bright structures from non-uniform background/fog/haze)
      3. Dynamic Adaptive Thresholding (distinguishes beacon from space background while rejecting noise/rain)
      4. Morphological Opening (removes single-pixel impulsive salt noise and fine rain streaks)
      5. Connected Components / Contour Extraction
      6. Geometric & Radiometric Candidate Filtering (area, aspect ratio, fill factor, local contrast)
      7. Optimal Candidate Selection
      
    Returns:
      detected (bool): True if beacon is detected, False otherwise.
      result (dict or None): Bounding box (x, y, w, h) and center (cx, cy) if detected.
    """
    # Step 1: Gaussian Pre-filter
    blurred = cv2.GaussianBlur(gray_frame, (3, 3), 0.7)

    # Step 2: Morphological Top-Hat Transform
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)

    top_max = float(np.max(tophat))
    top_mean = float(np.mean(tophat))
    top_std = float(np.std(tophat))

    # Step 3: Dynamic Adaptive Thresholding
    if is_low_light:
        # Low illumination scenario (dim beacon ~30-70 DN, dark sky ~10 DN)
        th = max(14.0, top_mean + 2.2 * top_std)
        min_p_mean = 18.0
        min_p_max = 24.0
    elif top_max >= 150.0:
        # High contrast scenario (SCN_001..004, 007, 008 when beacon visible)
        th = 130.0
        min_p_mean = 50.0
        min_p_max = 120.0
    elif top_max >= 90.0:
        # Medium contrast scenario (SCN_005 Fog)
        th = 55.0
        min_p_mean = 40.0
        min_p_max = 80.0
    else:
        # No strong bright target candidate in frame (e.g. SCN_008 Target Loss periods)
        return False, None

    _, binary = cv2.threshold(tophat, th, 255, cv2.THRESH_BINARY)

    # Step 4: Morphological Opening (2x2 kernel cleans single-pixel noise and thin rain streaks)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

    # Step 5: Contours Extraction
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        # Step 6: Candidate Filtering based on physical beacon geometry (5-25 px square)
        if area < 8 or area > 750:
            continue
        if w < 3 or w > 35 or h < 3 or h > 35:
            continue
        aspect = float(w) / max(h, 1)
        if aspect < 0.45 or aspect > 2.2:
            continue

        patch = gray_frame[y : y + h, x : x + w]
        p_mean = float(np.mean(patch))
        p_max = float(np.max(patch))

        # Check intensity criteria
        if p_mean < min_p_mean or p_max < min_p_max:
            continue

        # Local background contrast (excluding the target beacon itself)
        border = max(6, int(max(w, h) * 0.4))
        y1, y2 = max(0, y - border), min(gray_frame.shape[0], y + h + border)
        x1, x2 = max(0, x - border), min(gray_frame.shape[1], x + w + border)
        surround = gray_frame[y1:y2, x1:x2]

        mask = np.ones((y2 - y1, x2 - x1), dtype=bool)
        mask[(y - y1) : (y - y1 + h), (x - x1) : (x - x1 + w)] = False
        bg_pixels = surround[mask]
        bg_level = float(np.median(bg_pixels)) if len(bg_pixels) > 0 else float(np.median(gray_frame))
        contrast = p_max - bg_level

        if contrast >= 10.0:
            fill_factor = area / float(w * h)
            score = contrast * 0.6 + p_mean * 0.3 + fill_factor * 20.0
            candidates.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "w": int(w),
                "h": int(h),
                "cx": round(float(x + w / 2.0), 2),
                "cy": round(float(y + h / 2.0), 2),
                "center_x": round(float(x + w / 2.0), 2),
                "center_y": round(float(y + h / 2.0), 2),
                "area": float(area),
                "brightness": round(p_max, 2),
                "confidence": round(min(1.0, max(0.1, score / 150.0)), 3),
                "score": score,
            })

    if not candidates:
        return False, []

    # Step 7: Sort candidates by detection score and assign candidate IDs
    candidates.sort(key=lambda c: c["score"], reverse=True)
    for idx, c in enumerate(candidates, 1):
        c["candidate_id"] = f"cand_{idx:03d}"

    return True, candidates


def detect_beacon_in_frame(gray_frame, is_low_light=False):
    """
    Backwards-compatible single-beacon detector: returns best candidate.
    """
    found, cands = detect_all_candidates_in_frame(gray_frame, is_low_light=is_low_light)
    if found and len(cands) > 0:
        return True, cands[0]
    return False, None


def detect_all_candidates_in_frame(gray_frame, is_low_light=False):
    """Alias for candidate detection."""
    return _detect_candidates_core(gray_frame, is_low_light=is_low_light)


def _detect_candidates_core(gray_frame, is_low_light=False):
    # Gaussian Pre-filter
    blurred = cv2.GaussianBlur(gray_frame, (3, 3), 0.7)

    # Morphological Top-Hat Transform
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)

    top_max = float(np.max(tophat))
    top_mean = float(np.mean(tophat))
    top_std = float(np.std(tophat))

    if is_low_light:
        th = max(14.0, top_mean + 2.2 * top_std)
        min_p_mean = 18.0
        min_p_max = 24.0
    elif top_max >= 150.0:
        th = 130.0
        min_p_mean = 50.0
        min_p_max = 120.0
    elif top_max >= 90.0:
        th = 55.0
        min_p_mean = 40.0
        min_p_max = 80.0
    else:
        return False, []

    _, binary = cv2.threshold(tophat, th, 255, cv2.THRESH_BINARY)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        if area < 8 or area > 750:
            continue
        if w < 3 or w > 35 or h < 3 or h > 35:
            continue
        aspect = float(w) / max(h, 1)
        if aspect < 0.45 or aspect > 2.2:
            continue

        patch = gray_frame[y : y + h, x : x + w]
        p_mean = float(np.mean(patch))
        p_max = float(np.max(patch))

        if p_mean < min_p_mean or p_max < min_p_max:
            continue

        border = max(6, int(max(w, h) * 0.4))
        y1, y2 = max(0, y - border), min(gray_frame.shape[0], y + h + border)
        x1, x2 = max(0, x - border), min(gray_frame.shape[1], x + w + border)
        surround = gray_frame[y1:y2, x1:x2]

        mask = np.ones((y2 - y1, x2 - x1), dtype=bool)
        mask[(y - y1) : (y - y1 + h), (x - x1) : (x - x1 + w)] = False
        bg_pixels = surround[mask]
        bg_level = float(np.median(bg_pixels)) if len(bg_pixels) > 0 else float(np.median(gray_frame))
        contrast = p_max - bg_level

        if contrast >= 10.0:
            fill_factor = area / float(w * h)
            score = contrast * 0.6 + p_mean * 0.3 + fill_factor * 20.0
            candidates.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "w": int(w),
                "h": int(h),
                "cx": round(float(x + w / 2.0), 2),
                "cy": round(float(y + h / 2.0), 2),
                "center_x": round(float(x + w / 2.0), 2),
                "center_y": round(float(y + h / 2.0), 2),
                "area": float(area),
                "brightness": round(p_max, 2),
                "confidence": round(min(1.0, max(0.1, score / 150.0)), 3),
                "score": score,
            })

    if not candidates:
        return False, []

    candidates.sort(key=lambda c: c["score"], reverse=True)
    for idx, c in enumerate(candidates, 1):
        c["candidate_id"] = f"cand_{idx:03d}"

    return True, candidates


def process_scenario(scn_id, config):
    """Process all 900 frames of a scenario and write detection CSV."""
    gt_file = GT_DIR / f"{scn_id}.csv"
    gt_df = pd.read_csv(gt_file)

    frames_folder = FRAMES_DIR / scn_id
    total_frames = len(gt_df)

    is_low_light = (scn_id == "SCN_006")

    records = []
    vis_frames_to_save = [1, 150, 450, 750]
    if scn_id == "SCN_008":
        vis_frames_to_save = [1, 260, 301, 620, 661]  # Include loss and reacquisition frames

    saved_samples = {}

    for i in range(total_frames):
        row = gt_df.iloc[i]
        frame_id = int(row["frame_id"])
        timestamp = float(row["timestamp"])
        gt_cx = float(row["gt_center_x"])
        gt_cy = float(row["gt_center_y"])
        is_visible = str(row["visible"]).lower() == "true"

        frame_file = frames_folder / f"frame_{frame_id:06d}.png"
        gray = cv2.imread(str(frame_file), cv2.IMREAD_GRAYSCALE)

        detected, cand = detect_beacon_in_frame(gray, is_low_light=is_low_light)

        if detected:
            det_x = cand["x"]
            det_y = cand["y"]
            det_w = cand["w"]
            det_h = cand["h"]
            det_cx = cand["cx"]
            det_cy = cand["cy"]
            det_flag = "true"
            error = round(float(math.hypot(det_cx - gt_cx, det_cy - gt_cy)), 3)
        else:
            det_x = np.nan
            det_y = np.nan
            det_w = np.nan
            det_h = np.nan
            det_cx = np.nan
            det_cy = np.nan
            det_flag = "false"
            error = np.nan

        records.append({
            "scenario_id": scn_id,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "gt_center_x": gt_cx,
            "gt_center_y": gt_cy,
            "detected_x": det_x,
            "detected_y": det_y,
            "detected_width": det_w,
            "detected_height": det_h,
            "detected_center_x": det_cx,
            "detected_center_y": det_cy,
            "detected": det_flag,
            "detection_error": error,
        })

        # Save annotated visualization for selected representative frames
        if frame_id in vis_frames_to_save:
            annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Ground truth marker (Cyan circle)
            cv2.circle(annotated, (int(round(gt_cx)), int(round(gt_cy))), 12, (255, 200, 0), 1)
            cv2.drawMarker(annotated, (int(round(gt_cx)), int(round(gt_cy))), (255, 200, 0), cv2.MARKER_CROSS, 8, 1)

            if detected:
                # Detected bounding box (Green)
                cv2.rectangle(annotated, (det_x, det_y), (det_x + det_w, det_y + det_h), (0, 255, 0), 2)
                # Detected center marker (Green plus)
                cv2.drawMarker(annotated, (int(round(det_cx)), int(round(det_cy))), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 8, 1)
                status_text = f"DETECTED | Error: {error:.2f}px"
                status_color = (0, 255, 0)
            else:
                status_text = "TARGET NOT DETECTED" if is_visible else "TARGET LOSS PERIOD (INTENTIONAL)"
                status_color = (0, 0, 255) if is_visible else (0, 165, 255)

            # Informational Overlay
            cv2.rectangle(annotated, (10, 10), (380, 110), (20, 20, 20), -1)
            cv2.rectangle(annotated, (10, 10), (380, 110), (60, 60, 60), 1)

            cv2.putText(annotated, f"FSOC Virtual Tracking - {scn_id}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(annotated, f"Frame: {frame_id:04d} | Time: {timestamp:.2f}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(annotated, f"Status: {status_text}", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            if detected:
                cv2.putText(annotated, f"Det: ({det_cx:.1f}, {det_cy:.1f}) | GT: ({gt_cx:.1f}, {gt_cy:.1f})", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)
            else:
                cv2.putText(annotated, f"GT Position: ({gt_cx:.1f}, {gt_cy:.1f}) [Hidden]", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 255), 1)

            saved_samples[frame_id] = annotated

    # Save scenario detection CSV
    out_df = pd.DataFrame(records)
    out_csv = RESULTS_DIR / f"{scn_id}_detection.csv"
    out_df.to_csv(out_csv, index=False)

    # Save visual verification composites
    vis_out_file = VIS_DIR / f"{scn_id}_detection_vis.png"
    sample_imgs = [saved_samples[fid] for fid in sorted(saved_samples.keys())]
    if len(sample_imgs) <= 4:
        composite = np.hstack(sample_imgs)
    else:
        top_row = np.hstack(sample_imgs[:3])
        bot_row = np.hstack(sample_imgs[3:] + [np.zeros_like(sample_imgs[0])] * (3 - len(sample_imgs[3:])))
        composite = np.vstack([top_row, bot_row])
    cv2.imwrite(str(vis_out_file), composite)

    # Compute Summary Stats
    vis_count = (gt_df["visible"].astype(str).str.lower() == "true").sum()
    det_mask = (out_df["detected"] == "true")
    det_count = det_mask.sum()
    valid_errors = out_df.loc[det_mask, "detection_error"].dropna().values

    mean_err = float(np.mean(valid_errors)) if len(valid_errors) > 0 else 0.0
    rmse = float(np.sqrt(np.mean(valid_errors ** 2))) if len(valid_errors) > 0 else 0.0
    max_err = float(np.max(valid_errors)) if len(valid_errors) > 0 else 0.0

    return {
        "scenario_id": scn_id,
        "total_frames": total_frames,
        "gt_visible": vis_count,
        "detected": det_count,
        "detection_rate": round(float(det_count / vis_count * 100.0), 2),
        "mean_error": round(mean_err, 3),
        "rmse": round(rmse, 3),
        "max_error": round(max_err, 3),
    }


def process_multi_target_scenario(scn_id, config):
    """
    Process all frames of a multi-target scenario:
    - Extracts all candidates per frame.
    - Saves candidate records into detection_results/{scn_id}_candidates.csv and {scn_id}_detection.csv.
    - Visualizes all detected candidates and ground-truth positions.
    """
    gt_file = GT_DIR / f"{scn_id}.csv"
    gt_df = pd.read_csv(gt_file)
    frames_folder = FRAMES_DIR / scn_id

    total_frames = 900
    candidate_records = []
    vis_frames_to_save = [1, 150, 450, 750]
    if scn_id == "SCN_012":
        vis_frames_to_save = [1, 260, 301, 620, 661]
    saved_samples = {}

    for frame_id in range(1, total_frames + 1):
        timestamp = round((frame_id - 1) / float(config["fps"]), 4)
        frame_file = frames_folder / f"frame_{frame_id:06d}.png"
        gray = cv2.imread(str(frame_file), cv2.IMREAD_GRAYSCALE)

        found, cands = detect_all_candidates_in_frame(gray, is_low_light=False)
        frame_gt = gt_df[gt_df["frame_id"] == frame_id]

        if found and len(cands) > 0:
            for c in cands:
                candidate_records.append({
                    "scenario_id": scn_id,
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "candidate_id": c["candidate_id"],
                    "x": c["x"],
                    "y": c["y"],
                    "width": c["width"],
                    "height": c["height"],
                    "center_x": c["center_x"],
                    "center_y": c["center_y"],
                    "area": c["area"],
                    "brightness": c["brightness"],
                    "confidence": c["confidence"],
                })
        else:
            candidate_records.append({
                "scenario_id": scn_id,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "candidate_id": "none",
                "x": np.nan, "y": np.nan, "width": np.nan, "height": np.nan,
                "center_x": np.nan, "center_y": np.nan, "area": np.nan,
                "brightness": np.nan, "confidence": 0.0,
            })

        # Visualization
        if frame_id in vis_frames_to_save:
            annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for _, gt_row in frame_gt.iterrows():
                gt_cx = float(gt_row["gt_center_x"])
                gt_cy = float(gt_row["gt_center_y"])
                gt_vis = str(gt_row["visible"]).lower() == "true"
                role = gt_row["target_role"]
                color = (255, 200, 0) if role == "COMMUNICATION_TARGET" else (200, 160, 255)
                if gt_vis:
                    cv2.circle(annotated, (int(round(gt_cx)), int(round(gt_cy))), 12, color, 1)
                    cv2.putText(annotated, f"GT:{gt_row['target_id']}", (int(gt_cx)+14, int(gt_cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            if found:
                for c in cands:
                    cx, cy = int(round(c["center_x"])), int(round(c["center_y"]))
                    x, y, w, h = c["x"], c["y"], c["width"], c["height"]
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv2.drawMarker(annotated, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 8, 1)
                    cv2.putText(annotated, c["candidate_id"], (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

            cv2.rectangle(annotated, (10, 10), (380, 75), (20, 20, 20), -1)
            cv2.putText(annotated, f"Multi-Target Detection - {scn_id}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(annotated, f"Frame: {frame_id:04d} | Cands: {len(cands) if found else 0}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            saved_samples[frame_id] = annotated

    cands_df = pd.DataFrame(candidate_records)
    cands_df.to_csv(RESULTS_DIR / f"{scn_id}_candidates.csv", index=False)
    cands_df.to_csv(RESULTS_DIR / f"{scn_id}_detection.csv", index=False)

    vis_out_file = VIS_DIR / f"{scn_id}_detection_vis.png"
    sample_imgs = [saved_samples[fid] for fid in sorted(saved_samples.keys())]
    if len(sample_imgs) <= 4:
        composite = np.hstack(sample_imgs)
    else:
        top_row = np.hstack(sample_imgs[:3])
        bot_row = np.hstack(sample_imgs[3:] + [np.zeros_like(sample_imgs[0])] * (3 - len(sample_imgs[3:])))
        composite = np.vstack([top_row, bot_row])
    cv2.imwrite(str(vis_out_file), composite)
    print(f"  -> Saved {len(cands_df)} candidate records to {scn_id}_detection.csv")


def main():
    print("=" * 70)
    print("FSOC VIRTUAL CAMERA TRACKING - BEACON DETECTION STAGE")
    print("=" * 70)

    config = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    summary_stats = []

    for i in range(1, 9):
        scn_id = f"SCN_{i:03d}"
        print(f"Detecting beacon across 900 frames for {scn_id}...")
        stats = process_scenario(scn_id, config)
        summary_stats.append(stats)

    # Process multi-target scenarios if present
    for i in range(9, 13):
        m_scn_id = f"SCN_{i:03d}"
        if (GT_DIR / f"{m_scn_id}.csv").exists():
            print(f"Detecting multiple candidate beacons across 900 frames for {m_scn_id}...")
            process_multi_target_scenario(m_scn_id, config)

    print("\n" + "=" * 75)
    print("BEACON DETECTION RESULTS SUMMARY (SCN_001 to SCN_008)")
    print("=" * 75)
    headers = ["Scenario", "Frames", "GT Visible", "Detected", "Det Rate (%)", "Mean Err (px)", "RMSE (px)", "Max Err (px)"]
    print(f"{headers[0]:<10} {headers[1]:<8} {headers[2]:<11} {headers[3]:<10} {headers[4]:<13} {headers[5]:<14} {headers[6]:<11} {headers[7]:<11}")
    print("-" * 90)

    for s in summary_stats:
        print(f"{s['scenario_id']:<10} {s['total_frames']:<8} {s['gt_visible']:<11} {s['detected']:<10} {s['detection_rate']:<13.1f} {s['mean_error']:<14.2f} {s['rmse']:<11.2f} {s['max_error']:<11.2f}")
    print("=" * 75)
    print(f"\nDetection results CSV files saved to: {RESULTS_DIR}")
    print(f"Visual verification images saved to: {VIS_DIR}")


if __name__ == "__main__":
    main()
