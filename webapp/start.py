"""
One‑click launcher for the Ribosome Stalling Prediction System.
Starts both the Flask backend and the frontend dashboard server.
"""

import subprocess
import time
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent
    backend_script = project_root / "webapp" / "backend" / "app.py"
    frontend_script = project_root / "webapp" / "frontend" / "dashboard_server.py"

    if not backend_script.exists():
        print("Error: backend script not found.")
        sys.exit(1)
    if not frontend_script.exists():
        print("Error: frontend script not found.")
        sys.exit(1)

    print("Starting backend API server...")
    backend_proc = subprocess.Popen([sys.executable, str(backend_script)],
                                    cwd=str(backend_script.parent))
    time.sleep(3)

    print("Starting frontend dashboard...")
    frontend_proc = subprocess.Popen([sys.executable, str(frontend_script)],
                                     cwd=str(frontend_script.parent))
    time.sleep(2)

    import webbrowser
    webbrowser.open('http://localhost:8080')

    print("\nSystem is running.")
    print("  Backend API: http://localhost:5000")
    print("  Frontend Dashboard: http://localhost:8080")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Services stopped.")

if __name__ == '__main__':
    main()
