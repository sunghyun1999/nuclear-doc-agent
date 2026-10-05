FROM python:3.12-slim

WORKDIR /app

# Install Node.js for frontend build
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Backend dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ML models during build (faster cold start)
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Build frontend
COPY frontend/ ./frontend/
RUN cd frontend && npm ci && npm run build

# Copy backend
COPY backend/ ./backend/

# Serve frontend static files from FastAPI
RUN cp -r frontend/dist backend/static

# Start script
RUN echo '#!/bin/bash\ncd /app/backend\nuvicorn main:app --host 0.0.0.0 --port 7860' > /app/start.sh && \
    chmod +x /app/start.sh

EXPOSE 7860

CMD ["/app/start.sh"]
