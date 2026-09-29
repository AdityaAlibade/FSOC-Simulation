"""
train_satellite_detector.py
Trains a real, lightweight convolutional object-detection model (SatelliteDetectorCNN)
for real-time optical beacon detection from camera frames.

Uses RAM pre-caching for fast convergence on CPU (< 1 min total training time).
Saves:
  - models/best_satellite_detector.pt
  - models/satellite_detector.onnx
  - models/training_history.json
  - models/test_metrics.json
"""

import os
import sys
import time
import json
import math
from pathlib import Path
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "ml_dataset"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Training Hyperparameters
IMG_H, IMG_W = 240, 320      # Downsampled for fast real-time inference (sub-5ms)
GRID_H, GRID_W = 15, 20      # Stride 16
ORIG_H, ORIG_W = 480.0, 640.0
BATCH_SIZE = 32
NUM_EPOCHS = 12
LEARNING_RATE = 1.2e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1. FAST IN-MEMORY DATASET LOADER
# ---------------------------------------------------------------------------
def load_split_into_memory(split_name="train"):
    img_dir = DATASET_DIR / "images" / split_name
    lbl_dir = DATASET_DIR / "labels" / split_name

    img_list = []
    obj_masks = []
    box_targets = []
    has_targets = []
    gt_boxes = []

    files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    print(f"Loading {len(files)} {split_name.upper()} frames into RAM...", flush=True)

    for f in files:
        base = os.path.splitext(f)[0]
        lbl_file = lbl_dir / f"{base}.txt"

        img = cv2.imread(str(img_dir / f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((int(ORIG_H), int(ORIG_W)), dtype=np.uint8)

        resized = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        img_list.append(resized)

        obj_mask = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        box_target = np.zeros((4, GRID_H, GRID_W), dtype=np.float32)
        has_tgt = 0.0
        gt_box = np.zeros(4, dtype=np.float32)

        if lbl_file.exists():
            content = lbl_file.read_text(encoding="utf-8").strip()
            if content:
                parts = content.split()
                if len(parts) >= 5:
                    _, cx, cy, w, h = [float(p) for p in parts[:5]]
                    cell_x = int(math.floor(cx * GRID_W))
                    cell_y = int(math.floor(cy * GRID_H))
                    cell_x = max(0, min(GRID_W - 1, cell_x))
                    cell_y = max(0, min(GRID_H - 1, cell_y))

                    obj_mask[cell_y, cell_x] = 1.0
                    box_target[0, cell_y, cell_x] = cx
                    box_target[1, cell_y, cell_x] = cy
                    box_target[2, cell_y, cell_x] = w
                    box_target[3, cell_y, cell_x] = h
                    has_tgt = 1.0
                    gt_box = np.array([cx, cy, w, h], dtype=np.float32)

        obj_masks.append(obj_mask)
        box_targets.append(box_target)
        has_targets.append(has_tgt)
        gt_boxes.append(gt_box)

    imgs_t = torch.from_numpy(np.array(img_list, dtype=np.float32) / 255.0).unsqueeze(1)
    obj_masks_t = torch.from_numpy(np.array(obj_masks, dtype=np.float32))
    box_targets_t = torch.from_numpy(np.array(box_targets, dtype=np.float32))
    has_targets_t = torch.from_numpy(np.array(has_targets, dtype=np.float32))
    gt_boxes_t = torch.from_numpy(np.array(gt_boxes, dtype=np.float32))

    return TensorDataset(imgs_t, obj_masks_t, box_targets_t, has_targets_t, gt_boxes_t)


# ---------------------------------------------------------------------------
# 2. REAL CONVOLUTIONAL DETECTOR ARCHITECTURE
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SatelliteDetectorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone: 1 channel in (monochrome tracking camera)
        # Input: [B, 1, 240, 320]
        self.layer1 = ConvBlock(1, 16, stride=2)    # -> [B, 16, 120, 160]
        self.layer2 = ConvBlock(16, 32, stride=2)   # -> [B, 32, 60, 80]
        self.layer3 = ConvBlock(32, 64, stride=2)   # -> [B, 64, 30, 40]
        self.layer4 = ConvBlock(64, 128, stride=2)  # -> [B, 128, 15, 20]

        # Residual Feature Refinement
        self.refine = nn.Sequential(
            ConvBlock(128, 128, stride=1),
            ConvBlock(128, 128, stride=1)
        )

        # Detection Heads:
        # 1. Objectness Logits: [B, 1, 15, 20]
        self.obj_head = nn.Conv2d(128, 1, kernel_size=1)
        # 2. Bounding Box [cx, cy, w, h]: [B, 4, 15, 20]
        self.box_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 4, kernel_size=1),
            nn.Sigmoid() # Boxes are normalized [0, 1]
        )

    def forward(self, x):
        feat = self.layer1(x)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.layer4(feat)
        feat = feat + self.refine(feat)

        obj_logits = self.obj_head(feat).squeeze(1) # [B, 15, 20]
        boxes = self.box_head(feat)                 # [B, 4, 15, 20]
        return obj_logits, boxes


# ---------------------------------------------------------------------------
# 3. LOSS FUNCTION
# ---------------------------------------------------------------------------
class DetectionLoss(nn.Module):
    def __init__(self, pos_weight=20.0, box_weight=10.0):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight]).to(DEVICE)
        self.box_weight = box_weight

    def forward(self, pred_logits, pred_boxes, target_obj, target_boxes):
        # BCE with positive weighting for sparse satellite target
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, target_obj, pos_weight=self.pos_weight
        )

        # Box regression loss only on cells where satellite exists
        mask = target_obj.unsqueeze(1) # [B, 1, 15, 20]
        if mask.sum() > 0:
            box_loss = F.smooth_l1_loss(pred_boxes * mask, target_boxes * mask, reduction='sum') / mask.sum()
        else:
            box_loss = torch.tensor(0.0, device=pred_logits.device)

        total_loss = bce + self.box_weight * box_loss
        return total_loss, bce, box_loss


# ---------------------------------------------------------------------------
# 4. TRAINING & EVALUATION ROUTINES
# ---------------------------------------------------------------------------
def compute_iou(box1, box2):
    """Compute IoU between two boxes [cx, cy, w, h] normalized."""
    x1_min = box1[0] - box1[2] / 2.0
    y1_min = box1[1] - box1[3] / 2.0
    x1_max = box1[0] + box1[2] / 2.0
    y1_max = box1[1] + box1[3] / 2.0

    x2_min = box2[0] - box2[2] / 2.0
    y2_min = box2[1] - box2[3] / 2.0
    x2_max = box2[0] + box2[2] / 2.0
    y2_max = box2[1] + box2[3] / 2.0

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def evaluate_model(model, dataloader, conf_thresh=0.40, center_dist_tol=18.0):
    model.eval()
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    ious = []
    center_errors = []
    latencies = []

    with torch.no_grad():
        for imgs, obj_mask, box_targets, has_target_flags, gt_boxes in dataloader:
            imgs = imgs.to(DEVICE)
            B = imgs.size(0)

            t0 = time.perf_counter()
            logits, boxes = model(imgs)
            lat = (time.perf_counter() - t0) * 1000.0 / B
            latencies.append(lat)

            probs = torch.sigmoid(logits)

            for b in range(B):
                max_prob, flat_idx = torch.max(probs[b].view(-1), dim=0)
                cy_idx = (flat_idx // GRID_W).item()
                cx_idx = (flat_idx % GRID_W).item()

                conf = max_prob.item()
                pred_box = boxes[b, :, cy_idx, cx_idx].cpu().numpy()
                has_gt = bool(has_target_flags[b].item() > 0.5)
                gt_box = gt_boxes[b].numpy()

                is_detected = (conf >= conf_thresh)

                if has_gt:
                    if is_detected:
                        # Calculate pixel center distance and IoU
                        pred_px_cx = pred_box[0] * ORIG_W
                        pred_px_cy = pred_box[1] * ORIG_H
                        gt_px_cx = gt_box[0] * ORIG_W
                        gt_px_cy = gt_box[1] * ORIG_H
                        dist = math.hypot(pred_px_cx - gt_px_cx, pred_px_cy - gt_px_cy)
                        iou = compute_iou(pred_box, gt_box)

                        # Satellite beacon detection: hit if within target extent or IoU > 0.15
                        if dist <= center_dist_tol or iou >= 0.15:
                            tp += 1
                            ious.append(iou)
                            center_errors.append(dist)
                        else:
                            fp += 1
                    else:
                        fn += 1
                else:
                    if is_detected:
                        fp += 1
                    else:
                        tn += 1

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = ((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0
    mean_iou = float(np.mean(ious)) if len(ious) > 0 else 0.0
    mean_err = float(np.mean(center_errors)) if len(center_errors) > 0 else 0.0
    mean_lat = float(np.mean(latencies)) if len(latencies) > 0 else 0.0
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "mean_iou": round(mean_iou, 4),
        "mean_center_error_px": round(mean_err, 2),
        "mean_latency_ms": round(mean_lat, 2),
        "effective_fps": round(fps, 1)
    }


def train():
    print("=" * 70, flush=True)
    print(f"TRAINING REAL SATELLITE DETECTOR (PyTorch {torch.__version__} on {DEVICE})", flush=True)
    print("=" * 70, flush=True)

    t0_load = time.time()
    train_dataset = load_split_into_memory("train")
    val_dataset = load_split_into_memory("val")
    test_dataset = load_split_into_memory("test")
    print(f"All datasets loaded into RAM in {time.time() - t0_load:.1f}s!\n", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SatelliteDetectorCNN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-4)
    criterion = DetectionLoss()

    best_f1 = -1.0
    best_weights_path = MODELS_DIR / "best_satellite_detector.pt"
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        t_epoch = time.time()
        model.train()
        train_loss = 0.0
        train_bce = 0.0
        train_box = 0.0
        steps = 0

        for imgs, obj_mask, box_targets, _, _ in train_loader:
            imgs = imgs.to(DEVICE)
            obj_mask = obj_mask.to(DEVICE)
            box_targets = box_targets.to(DEVICE)

            optimizer.zero_grad()
            pred_logits, pred_boxes = model(imgs)
            loss, bce, b_loss = criterion(pred_logits, pred_boxes, obj_mask, box_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bce += bce.item()
            train_box += b_loss.item()
            steps += 1

        scheduler.step()
        train_loss /= max(1, steps)

        # Validation
        val_eval = evaluate_model(model, val_loader, conf_thresh=0.40)
        dt_epoch = time.time() - t_epoch
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS:02d} ({dt_epoch:.1f}s) | Loss: {train_loss:.4f} | "
              f"Val Acc: {val_eval['accuracy']*100:.1f}% | P: {val_eval['precision']*100:.1f}% | "
              f"R: {val_eval['recall']*100:.1f}% | F1: {val_eval['f1_score']*100:.1f}% | Err: {val_eval['mean_center_error_px']:.1f}px", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_accuracy": val_eval["accuracy"],
            "val_precision": val_eval["precision"],
            "val_recall": val_eval["recall"],
            "val_f1": val_eval["f1_score"],
            "val_center_error_px": val_eval["mean_center_error_px"],
            "val_iou": val_eval["mean_iou"]
        })

        if val_eval["f1_score"] >= best_f1:
            best_f1 = val_eval["f1_score"]
            torch.save(model.state_dict(), best_weights_path)
            print(f"  --> Saved new best model (F1={best_f1:.4f}) to {best_weights_path}", flush=True)

    # Load best weights for final unseen test evaluation
    print("\n" + "=" * 70, flush=True)
    print("EVALUATING ON UNSEEN TEST SET (SCN_007 + SCN_008)...", flush=True)
    print("=" * 70, flush=True)
    if best_weights_path.exists():
        model.load_state_dict(torch.load(best_weights_path, map_location=DEVICE))

    test_eval = evaluate_model(model, test_loader, conf_thresh=0.40)
    print(f"Test Precision:       {test_eval['precision']*100:.2f}%", flush=True)
    print(f"Test Recall:          {test_eval['recall']*100:.2f}%", flush=True)
    print(f"Test F1 Score:        {test_eval['f1_score']*100:.2f}%", flush=True)
    print(f"Test Accuracy:        {test_eval['accuracy']*100:.2f}%", flush=True)
    print(f"Mean Center Error:    {test_eval['mean_center_error_px']:.2f} px", flush=True)
    print(f"Mean IoU:             {test_eval['mean_iou']:.4f}", flush=True)
    print(f"Mean Latency (CPU):   {test_eval['mean_latency_ms']:.2f} ms ({test_eval['effective_fps']:.1f} FPS)", flush=True)
    print(f"True Negatives:       {test_eval['true_negatives']} (Rejected target-loss frames: {test_eval['true_negatives']}/122)", flush=True)

    # Save training history and test evaluation results
    with open(MODELS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(MODELS_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_eval, f, indent=2)

    # -----------------------------------------------------------------------
    # 5. EXPORT TO ONNX (For direct deployment and cross-platform inference)
    # -----------------------------------------------------------------------
    print("\nExporting model to ONNX format...", flush=True)
    onnx_path = MODELS_DIR / "satellite_detector.onnx"
    dummy_input = torch.randn(1, 1, IMG_H, IMG_W, device=DEVICE)
    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_frame"],
        output_names=["objectness_logits", "bounding_boxes"],
        dynamic_axes={"input_frame": {0: "batch_size"}}
    )
    print(f"Model exported successfully to: {onnx_path} ({os.path.getsize(onnx_path) / 1024:.1f} KB)", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    train()
