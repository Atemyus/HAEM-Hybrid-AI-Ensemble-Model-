#!/usr/bin/env python3
"""Start script for Railway deployment."""
import os
import subprocess
import sys

# Get PORT from environment, default to 8501
port = os.environ.get("PORT", "8501")

print(f"Starting Streamlit on port {port}")

# Run streamlit
cmd = [
    sys.executable, "-m", "streamlit", "run",
    "src/haem/dashboard/app.py",
    f"--server.port={port}",
    "--server.address=0.0.0.0",
    "--server.headless=true",
    "--browser.gatherUsageStats=false"
]

subprocess.run(cmd)
