"""
prepare_ml_dataset.py
Extracts and partitions FSOC synthetic camera frames into an authoritative
object-detection dataset in standard YOLO format:
  class 0: satellite
Split by scenario to prevent temporal leakage:
  Train: SCN_001, SCN_002, SCN_003, SCN_005 (3,600 frames)
  Val:   SCN_004, SCN_006 (1,800 frames)
  Test:  SCN_007, SCN_008 (1,800 frames, includes 122 true negative frames)
"""

import os
import json
import shutil
from pathlib import Path
import pandas as pd
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "FSOC_DATASET"
FRAMES_DIR = DATASET_ROOT / "frames"
GT_DIR = DATASET_ROOT / "ground_truth"

OUTPUT_DIR = BASE_DIR / "ml_dataset"

SPLITS = {
    "train": ["SCN_001", "SCN_002", "SCN_003", "SCN_005"],
    "val": ["SCN_004", "SCN_006"],
    "test": ["SCN_007", "SCN_008"]
}

IMAGE_WIDTH = 640.0
IMAGE_HEIGHT = 480.0

def build_dataset():
    print("=" * 70)
    print("CREATING REAL OBJECT-DETECTION DATASET FOR SATELLITE TRACKING")
    print("=" * 70)

    # Clean / prepare output dirs
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {
        "dataset_name": "FSOC Optical Satellite Tracking Dataset",
        "format": "YOLO (class x_center y_center width height normalized)",
        "classes": ["satellite"],
        "splits": {}
    }

    total_images_all = 0
    total_pos_all = 0
    total_neg_all = 0

    for split, scenarios in SPLITS.items():
        print(f"\nProcessing {split.upper()} split from scenarios: {scenarios}...")
        img_out = OUTPUT_DIR / "images" / split
        lbl_out = OUTPUT_DIR / "labels" / split

        split_images = 0
        split_pos = 0
        split_neg = 0

        for scn in scenarios:
            gt_path = GT_DIR / f"{scn}.csv"
            scn_frames_dir = FRAMES_DIR / scn
            if not gt_path.exists() or not scn_frames_dir.exists():
                print(f"  WARNING: Missing data for {scn}, skipping.")
                continue

            gt_df = pd.read_csv(gt_path)

            for _, row in gt_df.iterrows():
                frame_id = int(row["frame_id"])
                is_visible = str(row["visible"]).strip().lower() == "true"
                src_filename = f"frame_{frame_id:06d}.png"
                src_path = scn_frames_dir / src_filename

                if not src_path.exists():
                    continue

                dest_base = f"{scn}_{frame_id:06d}"
                dest_img = img_out / f"{dest_base}.png"
                dest_lbl = lbl_out / f"{dest_base}.txt"

                # Copy image
                shutil.copy2(src_path, dest_img)

                # Write label
                if is_visible:
                    # Bounding box in YOLO format (class x_center y_center width height, normalized 0..1)
                    gx = float(row["gt_x"])
                    gy = float(row["gt_y"])
                    gw = float(row["gt_width"])
                    gh = float(row["gt_height"])

                    # Center
                    cx = gx + gw / 2.0
                    cy = gy + gh / 2.0

                    norm_cx = max(0.0, min(1.0, cx / IMAGE_WIDTH))
                    norm_cy = max(0.0, min(1.0, cy / IMAGE_HEIGHT))
                    norm_w = max(0.0, min(1.0, gw / IMAGE_WIDTH))
                    norm_h = max(0.0, min(1.0, gh / IMAGE_HEIGHT))

                    label_line = f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n"
                    dest_lbl.write_text(label_line, encoding="utf-8")
                    split_pos += 1
                else:
                    # Negative sample: empty text file
                    dest_lbl.write_text("", encoding="utf-8")
                    split_neg += 1

                split_images += 1

        stats["splits"][split] = {
            "scenarios": scenarios,
            "total_images": split_images,
            "positive_images": split_pos,
            "negative_images": split_neg,
            "percentage": 0.0 # Will compute
        }
        total_images_all += split_images
        total_pos_all += split_pos
        total_neg_all += split_neg
        print(f"  {split.upper()}: {split_images} images (Pos: {split_pos}, Neg: {split_neg})")

    for split in stats["splits"]:
        if total_images_all > 0:
            stats["splits"][split]["percentage"] = round((stats["splits"][split]["total_images"] / total_images_all) * 100.0, 1)

    stats["summary"] = {
        "total_images": total_images_all,
        "total_positive": total_pos_all,
        "total_negative": total_neg_all,
        "image_resolution": [int(IMAGE_WIDTH), int(IMAGE_HEIGHT)],
        "channels": 1
    }

    # Save stats.json
    stats_path = OUTPUT_DIR / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Save data.yaml (standard YOLO config)
    yaml_content = f"""path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: satellite
"""
    (OUTPUT_DIR / "data.yaml").write_text(yaml_content, encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"DATASET GENERATION COMPLETE: {total_images_all} images")
    print(f"Train: {stats['splits']['train']['total_images']} ({stats['splits']['train']['percentage']}%)")
    print(f"Val:   {stats['splits']['val']['total_images']} ({stats['splits']['val']['percentage']}%)")
    print(f"Test:  {stats['splits']['test']['total_images']} ({stats['splits']['test']['percentage']}%)")
    print(f"Dataset stats saved to: {stats_path}")
    print("=" * 70)
    return stats

if __name__ == "__main__":
    build_dataset()
