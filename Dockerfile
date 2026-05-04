# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose Gradio port
EXPOSE 7860

# Set environment variable for Gradio to listen on all interfaces
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Command to run the application
CMD ["python", "scripts/vision_assistant.py"]
