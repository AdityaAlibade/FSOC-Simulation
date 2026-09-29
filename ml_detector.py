"""
ml_detector.py
Real-time Machine Learning Satellite Detector and Feature Extractor.
Uses the trained PyTorch / ONNX SatelliteDetectorCNN model to process
raw camera frames and output real detections with true model confidence.
"""

import os
import sys
import time
import math
from pathlib import Path
import numpy as np
import cv2

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PT_PATH = MODELS_DIR / "best_satellite_detector.pt"
MODEL_ONNX_PATH = MODELS_DIR / "satellite_detector.onnx"

# Model specs
IMG_H, IMG_W = 240, 320
GRID_H, GRID_W = 15, 20
ORIG_H, ORIG_W = 480.0, 640.0

class MLSatelliteDetector:
    def __init__(self, model_path=None, use_onnx=True, conf_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.use_onnx = use_onnx
        self.model = None
        self.ort_session = None
        self.is_loaded = False
        self.device = "cpu"

        # Determine path
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = MODEL_ONNX_PATH if use_onnx and MODEL_ONNX_PATH.exists() else MODEL_PT_PATH

        self.load_model()

    def load_model(self):
        """Loads trained weights from ONNX or PyTorch checkpoint."""
        try:
            if self.model_path.suffix == ".onnx" and self.model_path.exists():
                import onnxruntime as ort
                # Use CPU execution provider for deterministic portable real-time inference
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                self.ort_session = ort.InferenceSession(str(self.model_path), opts, providers=["CPUExecutionProvider"])
                self.use_onnx = True
                self.is_loaded = True
                print(f"[MLDetector] Loaded ONNX model from: {self.model_path}")
            elif MODEL_PT_PATH.exists():
                import torch
                from train_satellite_detector import SatelliteDetectorCNN
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model = SatelliteDetectorCNN().to(self.device)
                self.model.load_state_dict(torch.load(MODEL_PT_PATH, map_location=self.device))
                self.model.eval()
                self.use_onnx = False
                self.is_loaded = True
                print(f"[MLDetector] Loaded PyTorch model on {self.device} from: {MODEL_PT_PATH}")
            else:
                print(f"[MLDetector] WARNING: Model weights not found at {self.model_path}")
                self.is_loaded = False
        except Exception as e:
            print(f"[MLDetector] ERROR loading model: {e}")
            self.is_loaded = False

    def preprocess(self, frame_bgr_or_gray):
        """Preprocesses camera frame into model input tensor [1, 1, 240, 320]."""
        if len(frame_bgr_or_gray.shape) == 3:
            gray = cv2.cvtColor(frame_bgr_or_gray, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_bgr_or_gray

        # Resize to model input resolution
        resized = cv2.resize(gray, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        # Normalize to [0, 1]
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=(0, 1)) # [1, 1, 240, 320]
        return tensor, gray

    def detect(self, frame, target_hint=None, conf_threshold=None):
        """
        Runs real ML inference on a camera frame.

        Parameters:
          frame: numpy array (640x480 grayscale or BGR)
          target_hint: optional dict with expected coordinates (for multi-target association)
          conf_threshold: optional override for detection confidence threshold

        Returns dict:
          detected (bool): True if satellite is detected
          confidence (float): Real model confidence in [0.0, 1.0]
          bbox (list or None): [x, y, width, height] in 640x480 pixels
          center (tuple or None): (center_x, center_y) in 640x480 pixels
          all_detections (list): list of all candidates above threshold
          inference_ms (float): elapsed inference latency in ms
        """
        if not self.is_loaded:
            # Try reloading in case training just completed
            self.load_model()
            if not self.is_loaded:
                return {
                    "detected": False,
                    "confidence": 0.0,
                    "bbox": None,
                    "center": None,
                    "all_detections": [],
                    "inference_ms": 0.0,
                    "error": "MODEL NOT LOADED"
                }

        thresh = conf_threshold if conf_threshold is not None else self.conf_threshold

        t0 = time.perf_counter()
        inp_tensor, orig_gray = self.preprocess(frame)

        if self.use_onnx:
            ort_inputs = {self.ort_session.get_inputs()[0].name: inp_tensor}
            logits, boxes = self.ort_session.run(None, ort_inputs)
            # logits: [1, 15, 20], boxes: [1, 4, 15, 20]
            logits = logits[0]
            boxes = boxes[0]
            # Sigmoid activation
            probs = 1.0 / (1.0 + np.exp(-logits))
        else:
            import torch
            with torch.no_grad():
                t_inp = torch.from_numpy(inp_tensor).to(self.device)
                logits_t, boxes_t = self.model(t_inp)
                probs = torch.sigmoid(logits_t[0]).cpu().numpy()
                boxes = boxes_t[0].cpu().numpy()

        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Extract all grid cells passing the confidence threshold
        candidates = []
        for cy in range(GRID_H):
            for cx in range(GRID_W):
                conf = float(probs[cy, cx])
                if conf >= thresh:
                    norm_cx = float(boxes[0, cy, cx])
                    norm_cy = float(boxes[1, cy, cx])
                    norm_w = float(boxes[2, cy, cx])
                    norm_h = float(boxes[3, cy, cx])

                    # Convert to camera image coordinate space (640x480)
                    px_cx = norm_cx * ORIG_W
                    px_cy = norm_cy * ORIG_H
                    px_w = max(4.0, norm_w * ORIG_W)
                    px_h = max(4.0, norm_h * ORIG_H)
                    px_x = px_cx - px_w / 2.0
                    px_y = px_cy - px_h / 2.0

                    candidates.append({
                        "confidence": conf,
                        "bbox": [round(px_x, 1), round(px_y, 1), round(px_w, 1), round(px_h, 1)],
                        "center": (round(px_cx, 2), round(px_cy, 2)),
                        "grid_pos": (cx, cy)
                    })

        # Max confidence observed in the entire grid (even if below threshold)
        max_prob = float(np.max(probs))

        if len(candidates) == 0:
            return {
                "detected": False,
                "confidence": round(max_prob, 4),
                "bbox": None,
                "center": None,
                "all_detections": [],
                "inference_ms": round(dt_ms, 2)
            }

        # Multi-satellite target association:
        # If target_hint is provided, select candidate closest to expected position
        if target_hint and len(candidates) > 1:
            if isinstance(target_hint, (list, tuple)):
                hint_x, hint_y = float(target_hint[0]), float(target_hint[1])
            elif isinstance(target_hint, dict):
                hint_x, hint_y = float(target_hint.get("x", 320.0)), float(target_hint.get("y", 240.0))
            else:
                hint_x, hint_y = 320.0, 240.0
            candidates.sort(key=lambda c: math.hypot(c["center"][0] - hint_x, c["center"][1] - hint_y))
            selected = candidates[0]
        else:
            # Sort by highest confidence
            candidates.sort(key=lambda c: c["confidence"], reverse=True)
            selected = candidates[0]

        cx, cy = selected["center"]
        err_x = round(cx - (ORIG_W / 2.0), 2)
        err_y = round(cy - (ORIG_H / 2.0), 2)
        rmse_err = round(math.hypot(err_x, err_y), 2)

        return {
            "detected": True,
            "confidence": round(selected["confidence"], 4),
            "bbox": selected["bbox"],
            "center": selected["center"],
            "tracking_error": {
                "error_x": err_x,
                "error_y": err_y,
                "rmse_px": rmse_err
            },
            "all_detections": candidates,
            "inference_ms": round(dt_ms, 2)
        }
