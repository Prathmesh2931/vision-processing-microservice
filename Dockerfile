FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV and AI libraries
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt

# Pre-download YOLO model to avoid download during runtime
RUN python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('YOLO model downloaded successfully!')" || echo "YOLO download failed, will use fallback"

# Copy application files
COPY . .

# Set environment variables for headless operation
ENV DISPLAY=:99
ENV QT_X11_NO_MITSHM=1
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "app.py"]
