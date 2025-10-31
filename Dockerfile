FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PDF/image processing, OpenCV, Postgres, Poppler
RUN apt-get update \
    && apt-get install -y \
        build-essential \
        libpq-dev \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create persistent directories for uploads, vector index, and logs if needed
RUN mkdir -p uploads faiss_indexes logs

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI application via Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
