#!/bin/bash
# Start script for Railway deployment

# Use PORT from Railway or default to 8501
PORT="${PORT:-8501}"

echo "Starting Streamlit on port $PORT"

exec streamlit run src/haem/dashboard/app.py \
    --server.port="$PORT" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
