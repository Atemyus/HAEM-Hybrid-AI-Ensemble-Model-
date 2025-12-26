#!/usr/bin/env python3
"""Start script for Railway deployment."""
import os
import sys

# Get PORT from environment, default to 8501
port = os.environ.get("PORT", "8501")

print(f"=== HAEM Starting ===")
print(f"Python: {sys.executable}")
print(f"PORT: {port}")
print(f"Working directory: {os.getcwd()}")
print(f"Files: {os.listdir('.')}")
sys.stdout.flush()

# Import and run streamlit
from streamlit.web import cli as stcli

sys.argv = [
    "streamlit", "run",
    "src/haem/dashboard/app.py",
    f"--server.port={port}",
    "--server.address=0.0.0.0",
    "--server.headless=true",
    "--browser.gatherUsageStats=false",
    "--server.enableCORS=false",
    "--server.enableXsrfProtection=false"
]

print(f"Running: {' '.join(sys.argv)}")
sys.stdout.flush()

stcli.main()
