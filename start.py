#!/usr/bin/env python3
"""Start script for Railway deployment."""
import os
import sys

# Get PORT from environment, default to 8501
port = os.environ.get("PORT", "8501")

print(f"=== HAEM Starting ===")
print(f"PORT from env: {port}")

# IMPORTANT: Remove any STREAMLIT_SERVER_PORT that might be set incorrectly
# This fixes Railway deployment issues
if "STREAMLIT_SERVER_PORT" in os.environ:
    print(f"Removing STREAMLIT_SERVER_PORT: {os.environ['STREAMLIT_SERVER_PORT']}")
    del os.environ["STREAMLIT_SERVER_PORT"]

# Set the correct port via environment variable
os.environ["STREAMLIT_SERVER_PORT"] = port
print(f"Set STREAMLIT_SERVER_PORT to: {port}")

sys.stdout.flush()

# Import and run streamlit
from streamlit.web import cli as stcli

sys.argv = [
    "streamlit", "run",
    "src/haem/dashboard/app.py",
    "--server.address=0.0.0.0",
    "--server.headless=true",
    "--browser.gatherUsageStats=false",
    "--server.enableCORS=false",
    "--server.enableXsrfProtection=false"
]

print(f"Running: {' '.join(sys.argv)}")
sys.stdout.flush()

stcli.main()
