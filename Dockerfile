FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    accelerate==0.34.2 \
    safetensors==0.4.5 \
    huggingface_hub==0.25.2 \
    runpod==1.7.5 \
    boto3==1.35.65 \
    toml==0.10.2 \
    pillow==10.4.0 \
    transformers==4.46.3 \
    diffusers==0.31.0 \
    fastapi==0.115.5 \
    uvicorn==0.32.1 \
    pydantic==2.10.3 \
    httpx==0.28.1

# Clone and install kohya-ss sd-scripts
RUN git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git /kohya && \
    cd /kohya && \
    sed -i 's|file:///kohya/sd-scripts|.|g' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Create model directory and download SDXL base model
RUN mkdir -p /workspace/models && \
    cd /workspace/models && \
    wget --progress=bar:force:noscroll \
    https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

# Copy handler
COPY handler.py /workspace/handler.py

# Create output directories
RUN mkdir -p /tmp/training_data /tmp/output

# Set Python unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

# Run handler
CMD ["python", "-u", "handler.py"]
