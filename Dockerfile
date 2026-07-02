FROM python:3.11-slim

RUN apt-get update && apt-get install -y libmagic1 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir ultralytics sahi && \
    python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')" && \
    rm -rf /root/.cache/pip

COPY backend/ backend/
COPY frontend/ frontend/
RUN mkdir -p uploads processed

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
