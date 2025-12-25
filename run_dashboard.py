#!/usr/bin/env python3
"""
HAEM Dashboard Runner.

Launch the Streamlit dashboard for weather analysis visualization.

Usage:
    python run_dashboard.py
    # Or
    streamlit run src/haem/dashboard/app.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit dashboard."""

    # Get the app path
    app_path = Path(__file__).parent / "src" / "haem" / "dashboard" / "app.py"

    if not app_path.exists():
        print(f"Error: Dashboard app not found at {app_path}")
        sys.exit(1)

    print("🌍 Starting HAEM Dashboard...")
    print(f"   App: {app_path}")
    print()
    print("   Dashboard will open in your browser automatically.")
    print("   Press Ctrl+C to stop the server.")
    print()

    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
        "--theme.primaryColor", "#1f77b4",
        "--theme.backgroundColor", "#ffffff",
        "--theme.secondaryBackgroundColor", "#f0f2f6",
    ])


if __name__ == "__main__":
    main()
