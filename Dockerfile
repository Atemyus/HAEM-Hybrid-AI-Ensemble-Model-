# HAEM - Hybrid AI-NWP Ensemble Model
# Railway Deployment Dockerfile

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Railway uses PORT env variable - default to 8501 for local testing
ENV PORT=8501

# Run Streamlit with dynamic port from Railway
CMD streamlit run src/haem/dashboard/app.py --server.port=$PORT --server.address=0.0.0.0
