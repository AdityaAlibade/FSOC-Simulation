"""
serve_dashboard.py - Local Web Server for FSOC Virtual Camera Tracking System Dashboard
Run this script to launch the dashboard in your web browser.
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

PORT = 8080
BASE_DIR = Path(__file__).resolve().parent

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

def main():
    print(f"Starting FSOC Dashboard Server at http://localhost:{PORT}")
    print("Press Ctrl+C to stop the server.")
    webbrowser.open(f"http://localhost:{PORT}/index.html")
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main()
