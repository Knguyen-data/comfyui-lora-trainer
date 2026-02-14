FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# HuggingFace token for gated model downloads (Flux Dev)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

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
    pip install --no-cache-dir -r requirements.txt

# Create model directories
RUN mkdir -p /workspace/models

# Download Flux Dev model files:
# 1. flux1-dev.safetensors (23.8 GB) - gated, requires HF_TOKEN
# 2. ae.safetensors (335 MB) - VAE/AutoEncoder
# 3. clip_l.safetensors (246 MB) - CLIP-L text encoder
# 4. t5xxl_fp16.safetensors (9.79 GB) - T5-XXL text encoder

# Flux Dev base model (gated - needs HF_TOKEN)
RUN wget --progress=bar:force:noscroll \
    --header="Authorization: Bearer ${HF_TOKEN}" \
    -O /workspace/models/flux1-dev.safetensors \
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors"

# Flux VAE (ae.safetensors from same repo)
RUN wget --progress=bar:force:noscroll \
    --header="Authorization: Bearer ${HF_TOKEN}" \
    -O /workspace/models/ae.safetensors \
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors"

# CLIP-L and T5-XXL text encoders (public, no auth needed)
RUN wget --progress=bar:force:noscroll \
    -O /workspace/models/clip_l.safetensors \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors" && \
    wget --progress=bar:force:noscroll \
    -O /workspace/models/t5xxl_fp16.safetensors \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"

# Copy handler
COPY handler.py /workspace/handler.py

# Create output directories
RUN mkdir -p /tmp/training_data /tmp/output

# Set Python unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

# Run handler
CMD ["python", "-u", "handler.py"]
