"""
HTTP server for the fixed dashboard (front‑end) of the Ribosome Stalling Prediction System.
This script creates a self‑contained HTML file and serves it via a simple HTTP server.
The front‑end code is identical to the original fixed_dashboard_server.py.
"""

import os
import sys
import http.server
import socketserver
from pathlib import Path

# Directory where the HTML file will be written
DASHBOARD_DIR = Path(__file__).parent / "dashboard"
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

# The complete HTML content (unchanged from the original)
html_content = r"""..."""   # (same as provided, not repeated for brevity)

# Write the HTML file
html_file = DASHBOARD_DIR / "index.html"
with open(html_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Dashboard HTML written to {html_file}")

# Start a simple HTTP server
PORT = 8080

class DashboardHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

if __name__ == "__main__":
    for port in [8080, 8081, 8082, 8083, 8084]:
        try:
            httpd = socketserver.TCPServer(("", port), DashboardHTTPRequestHandler)
            PORT = port
            break
        except OSError:
            continue
    else:
        print("No available port for frontend.")
        sys.exit(1)

    print(f"Frontend dashboard server started at http://localhost:{PORT}")
    with httpd:
        httpd.serve_forever()
