# LoRA Trainer Worker - Deployment to RunPod

## Quick Deploy

### Step 1: Build & Push Docker Image
```bash
cd workers/lora-trainer
docker build -t kie1/lora-trainer:v1.0 .
docker push kie1/lora-trainer:v1.0
```

Or use the deployment script:
```bash
chmod +x deploy-dockerhub.sh
./deploy-dockerhub.sh
```

### Step 2: Create RunPod Endpoint
1. Go to https://runpod.io/serverless
2. Create new endpoint:
   - **Name**: LoRA-Trainer-SDXL
   - **Container**: `kie1/lora-trainer:v1.0`
   - **GPU**: RTX 4090 or A100 40GB
   - **Timeout**: 1800s (30min for training)
   - **Active Workers**: 0 (scale to zero)
   - **Max Workers**: 3
3. Save endpoint ID

### Step 3: Update Frontend
Update your frontend configuration with the endpoint ID for LoRA training API calls.

## Models Included
- SDXL Base 1.0 (stabilityai)
- Kohya-ss sd-scripts (latest)

## API Request Format

```json
{
  "input": {
    "training_images": [
      {"filename": "face1.jpg", "data": "base64..."},
      {"filename": "face2.jpg", "data": "base64..."}
    ],
    "config": {
      "steps": 1500,
      "lr": 1e-4,
      "network_dim": 32,
      "network_alpha": 16,
      "resolution": 1024,
      "trigger_word": "ohwx"
    },
    "storage": {
      "r2_endpoint": "https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com",
      "r2_access_key": "YOUR_ACCESS_KEY",
      "r2_secret_key": "YOUR_SECRET_KEY",
      "r2_bucket": "lora-models",
      "upload_path": "user123/lora_20240101.safetensors"
    }
  }
}
```

## API Response Format

Success:
```json
{
  "status": "completed",
  "lora_url": "https://...r2.cloudflarestorage.com/lora-models/user123/lora_20240101.safetensors",
  "file_size": 143654912
}
```

Failed:
```json
{
  "status": "failed",
  "error": "Training process failed with exit code 1",
  "log": "... last 2000 chars of training log ..."
}
```

## Local Testing

1. Build image:
```bash
docker build -t lora-trainer-test .
```

2. Run locally (requires NVIDIA GPU):
```bash
docker run -it --gpus all -p 8000:8000 lora-trainer-test
```

3. Test endpoint:
```bash
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d @test_payload.json
```

## Training Parameters

**Defaults:**
- `steps`: 1500
- `lr`: 1e-4 (learning rate)
- `network_dim`: 32 (LoRA rank)
- `network_alpha`: 16
- `resolution`: 1024
- `trigger_word`: "ohwx"

**Auto-captioning:** All images are captioned as "{trigger_word} person"

## System Requirements

- GPU: 24GB+ VRAM (RTX 4090, A100 40GB recommended)
- Training time: ~15-20 minutes for 1500 steps with 10-20 images
- Output file size: ~140MB (.safetensors)

## Troubleshooting

**Training fails with OOM:**
- Reduce resolution to 512 or 768
- Reduce network_dim to 16

**Training hangs:**
- Check RunPod logs for subprocess output
- Verify training images are valid base64

**R2 upload fails:**
- Verify R2 credentials
- Check bucket permissions
- Ensure upload_path doesn't conflict
