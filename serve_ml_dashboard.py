"""
serve_ml_dashboard.py
Integrated Dashboard and Real-Time ML Inference Server for FSOC.

Serves:
  1. Interactive Web Dashboard (index.html) on HTTP port 8080
  2. REST API endpoints:
     - POST /api/ml/detect       -> Runs real trained ML model on camera frame
     - GET  /api/ml/metrics      -> Returns actual test set evaluation metrics
     - GET  /api/ml/dataset      -> Returns actual dataset split statistics
     - GET  /api/ml/status       -> Returns model operational status
"""

import http.server
import socketserver
import json
import base64
import time
import os
import sys
import webbrowser
from pathlib import Path
import numpy as np
import cv2

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "ml_dataset"
PORT = 8080

# Lazy-loaded ML Detector
_ml_detector = None

def get_detector():
    global _ml_detector
    if _ml_detector is None:
        from ml_detector import MLSatelliteDetector
        _ml_detector = MLSatelliteDetector(conf_threshold=0.45)
    return _ml_detector

class MLDashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def log_message(self, format, *args):
        # Suppress routine GET spam
        if "api/ml" in str(args[0]):
            pass # Keep terminal clean
        else:
            super().log_message(format, *args)

    def do_GET(self):
        if self.path == "/api/ml/status":
            self.send_json_response(self.handle_get_status())
        elif self.path == "/api/ml/metrics":
            self.send_json_response(self.handle_get_metrics())
        elif self.path == "/api/ml/dataset":
            self.send_json_response(self.handle_get_dataset())
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/ml/detect":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                result = self.handle_post_detect(data)
                self.send_json_response(result)
            except Exception as e:
                self.send_json_response({"error": str(e), "detected": False, "confidence": 0.0}, status_code=500)
        else:
            self.send_error(404, "Endpoint not found")

    def send_json_response(self, data, status_code=200):
        resp_bytes = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(resp_bytes)

    def handle_get_status(self):
        detector = get_detector()
        pt_exists = (MODELS_DIR / "best_satellite_detector.pt").exists()
        onnx_exists = (MODELS_DIR / "satellite_detector.onnx").exists()
        return {
            "status": "OPERATIONAL" if detector.is_loaded else "AWAITING_WEIGHTS",
            "is_loaded": detector.is_loaded,
            "engine": "ONNX Runtime" if detector.use_onnx else "PyTorch",
            "weights_pt_available": pt_exists,
            "weights_onnx_available": onnx_exists,
            "confidence_threshold": detector.conf_threshold,
            "model_architecture": "SatelliteDetectorCNN (Convolutional Object Detector)"
        }

    def handle_get_metrics(self):
        metrics_file = MODELS_DIR / "test_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["status"] = "VALIDATED_TEST_SET"
            return data
        return {
            "status": "MODEL NOT TRAINED",
            "precision": None,
            "recall": None,
            "f1_score": None,
            "accuracy": None,
            "mean_latency_ms": None,
            "effective_fps": None
        }

    def handle_get_dataset(self):
        stats_file = DATASET_DIR / "dataset_stats.json"
        if stats_file.exists():
            with open(stats_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "DATASET NOT PREPARED"}

    def handle_post_detect(self, data):
        detector = get_detector()
        target_hint = data.get("target_hint")
        conf_thresh = data.get("conf_threshold")

        # Scenario frame lookup (fastest & 100% pixel-faithful)
        if "scenario_id" in data and "frame_id" in data:
            scn = data["scenario_id"]
            fid = int(data["frame_id"])
            frame_path = BASE_DIR / "FSOC_DATASET" / "frames" / scn / f"frame_{fid:06d}.png"
            if frame_path.exists():
                frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
                return detector.detect(frame, target_hint=target_hint, conf_threshold=conf_thresh)

        # Base64 camera frame image sent from canvas
        if "image_base64" in data:
            raw_b64 = data["image_base64"]
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",")[1]
            img_bytes = base64.b64decode(raw_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if frame is not None:
                return detector.detect(frame, target_hint=target_hint, conf_threshold=conf_thresh)

        # Raw grayscale pixel buffer
        if "pixel_array" in data and "width" in data and "height" in data:
            w, h = data["width"], data["height"]
            arr = np.array(data["pixel_array"], dtype=np.uint8).reshape((h, w))
            return detector.detect(arr, target_hint=target_hint, conf_threshold=conf_thresh)

        return {"detected": False, "confidence": 0.0, "error": "No valid frame data provided"}

def run_server(port=PORT):
    print("=" * 70)
    print(f"FSOC ML TRACKING SERVER STARTING ON http://localhost:{port}")
    print("=" * 70)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), MLDashboardHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[Server stopped by user]")

if __name__ == "__main__":
    run_server()
