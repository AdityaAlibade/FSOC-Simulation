#!/usr/bin/env python3
"""
run_fsoc_pipeline.py - Master Pipeline & Application Runner
FSOC Virtual Camera Tracking System
"""

import argparse
import http.server
import os
import socketserver
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PORT = 8080

def print_banner():
    banner = """
======================================================================
  FSOC VIRTUAL CAMERA TRACKING SYSTEM
  AI-Based Coarse Alignment for Mobile Optical Terminals
======================================================================
"""
    print(banner)

def run_stage(script_name, description):
    print(f"\n[RUNNING] {description} ({script_name})...")
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_name}")
        return False
        
    t0 = time.time()
    res = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR))
    dt = time.time() - t0
    if res.returncode == 0:
        print(f"[SUCCESS] {description} completed in {dt:.2f}s.")
        return True
    else:
        print(f"[FAILED] {description} exited with code {res.returncode}.")
        return False

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)
        
    def log_message(self, format, *args):
        # Suppress routine GET request spam in terminal
        pass

def start_dashboard(port=PORT):
    print(f"\n======================================================================")
    print(f"  LAUNCHING FSOC INTERACTIVE APPLICATION DASHBOARD")
    print(f"  URL: http://localhost:{port}/index.html")
    print(f"  Press Ctrl+C to stop the dashboard server.")
    print(f"======================================================================\n")

    # Try opening browser
    def open_browser():
        time.sleep(1.0)
        webbrowser.open(f"http://localhost:{port}/index.html")
        
    threading.Thread(target=open_browser, daemon=True).start()

    # Reuse address to prevent 'address already in use' errors on quick restarts
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", port), DashboardHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Dashboard server stopped by user.")
    except OSError as e:
        # If port 8080 is taken, try port 8081
        alt_port = port + 1
        print(f"[WARN] Port {port} in use. Attempting fallback port {alt_port}...")
        webbrowser.open(f"http://localhost:{alt_port}/index.html")
        with socketserver.TCPServer(("", alt_port), DashboardHandler) as httpd:
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n[INFO] Dashboard server stopped by user.")

def main():
    print_banner()
    parser = argparse.ArgumentParser(description="FSOC Virtual Camera Tracking System Runner")
    parser.add_argument("--mode", choices=["dashboard", "benchmark", "validate", "full"], default="dashboard",
                        help="Execution mode: 'dashboard' (default), 'benchmark', 'validate', or 'full'")
    parser.add_argument("--port", type=int, default=PORT, help="Port for dashboard web server")
    args = parser.parse_args()

    if args.mode == "benchmark":
        run_stage("run_benchmark.py", "Benchmark Evaluation")
        run_stage("validate_benchmark.py", "Benchmark Validation")
    elif args.mode == "validate":
        run_stage("validate_dataset.py", "Dataset Validation")
        run_stage("validate_detection.py", "Detection Validation")
        run_stage("validate_centroid_analysis.py", "Centroid Validation")
        run_stage("validate_tracking.py", "Tracking Validation")
        run_stage("validate_alignment.py", "Alignment Validation")
        run_stage("validate_benchmark.py", "Benchmark Validation")
    elif args.mode == "full":
        stages = [
            ("generate_dataset.py", "Synthetic Dataset Generation"),
            ("validate_dataset.py", "Dataset Validation"),
            ("run_beacon_detection.py", "Beacon Detection"),
            ("validate_detection.py", "Detection Validation"),
            ("centroid_analysis.py", "Centroid Estimation & Analysis"),
            ("validate_centroid_analysis.py", "Centroid Validation"),
            ("track_beacon.py", "Temporal Beacon Tracking"),
            ("validate_tracking.py", "Tracking Validation"),
            ("alignment_controller.py", "Virtual Pan-Tilt Alignment"),
            ("validate_alignment.py", "Alignment Validation"),
            ("run_benchmark.py", "Final Benchmark Evaluation"),
            ("validate_benchmark.py", "Benchmark Validation"),
        ]
        for script, desc in stages:
            if not run_stage(script, desc):
                print(f"[ABORT] Pipeline stopped due to failure in {script}")
                sys.exit(1)
        print("\n[COMPLETE] Full pipeline executed successfully!")
        start_dashboard(args.port)
    else:
        # Default: Start Dashboard
        start_dashboard(args.port)

if __name__ == "__main__":
    main()
