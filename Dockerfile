FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies first (cached layer for fast rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and saved model artifact
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]