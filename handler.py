import asyncio
import base64
import os
import glob
import toml
import runpod
import boto3
import httpx
from botocore.client import Config
from pathlib import Path
from urllib.parse import urlparse

# Flux Dev model paths (baked into Docker image)
FLUX_MODEL = "/workspace/models/flux1-dev.safetensors"
FLUX_AE = "/workspace/models/ae.safetensors"
FLUX_CLIP_L = "/workspace/models/clip_l.safetensors"
FLUX_T5XXL = "/workspace/models/t5xxl_fp16.safetensors"

# HuggingFace gated model URLs
_GATED_MODELS = [
    (FLUX_MODEL, "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors"),
    (FLUX_AE, "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors"),
]


def ensure_gated_models():
    """Download gated Flux Dev models at startup using HF_TOKEN env var."""
    import subprocess
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("WARNING: HF_TOKEN not set, gated model downloads will fail")
        return

    for dest, url in _GATED_MODELS:
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"Model already exists: {dest}")
            continue
        print(f"Downloading gated model: {url}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        result = subprocess.run([
            "wget", "--progress=bar:force:noscroll",
            "--header", f"Authorization: Bearer {hf_token}",
            "-O", dest, url
        ], timeout=3600)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download {url} (exit {result.returncode})")
        print(f"Downloaded: {dest} ({os.path.getsize(dest)} bytes)")


def download_image(url: str, output_path: str):
    """Download and save image from URL."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    parsed = urlparse(url)
    filename = os.path.basename(parsed.path) or f"image_{hash(url) % 10000}.jpg"

    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        filename += ".jpg"

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, follow_redirects=True)
        response.raise_for_status()

        file_path = os.path.join(output_path, filename)
        with open(file_path, "wb") as f:
            f.write(response.content)

    return filename


def generate_dataset_toml(training_data_dir: str, resolution: int) -> str:
    """Generate kohya dataset TOML config for Flux training."""
    toml_config = {
        "general": {
            "shuffle_caption": True,
            "caption_extension": ".txt",
            "keep_tokens": 1,
        },
        "datasets": [{
            "resolution": resolution,
            "batch_size": 1,
            "keep_tokens": 1,
            "subsets": [{
                "image_dir": training_data_dir,
                "num_repeats": 10,
            }],
        }],
    }

    config_path = "/tmp/dataset_config.toml"
    with open(config_path, "w") as f:
        toml.dump(toml_config, f)

    return config_path


def upload_to_r2(file_path: str, storage: dict) -> str:
    """Upload file to Cloudflare R2 storage."""

    s3 = boto3.client(
        "s3",
        endpoint_url=storage["r2_endpoint"],
        aws_access_key_id=storage["r2_access_key"],
        aws_secret_access_key=storage["r2_secret_key"],
        config=Config(signature_version="s3v4"),
    )

    upload_path = storage["upload_path"]
    bucket = storage["r2_bucket"]

    s3.upload_file(file_path, bucket, upload_path)

    # Return public URL (R2 public access domain)
    r2_public = storage.get("r2_public_url", "")
    if r2_public:
        return f"{r2_public.rstrip('/')}/{upload_path}"
    return f"{storage['r2_endpoint']}/{bucket}/{upload_path}"


def find_output_lora() -> str:
    """Find the trained LoRA file in output directory."""
    lora_files = glob.glob("/tmp/output/*.safetensors")

    if not lora_files:
        raise FileNotFoundError("No .safetensors file found in /tmp/output/")

    return lora_files[0]


async def handler(event: dict) -> dict:
    """
    RunPod handler for Flux Dev LoRA training via kohya-ss/sd-scripts.

    Expected input format:
    {
        "mode": "train_lora",
        "trigger_word": "ohwx",
        "training_images": ["url1", "url2"],
        "steps": 1000,
        "learning_rate": 1e-4,
        "output_name": "lora_userid_123abc"
    }
    """

    try:
        input_data = event.get("input", {})

        mode = input_data.get("mode")
        if mode != "train_lora":
            return {"status": "failed", "error": f"Unknown mode: {mode}"}

        trigger_word = input_data.get("trigger_word", "ohwx")
        image_urls = input_data.get("training_images", [])
        steps = input_data.get("steps", 1000)
        learning_rate = input_data.get("learning_rate", 1e-4)
        network_dim = input_data.get("network_dim", 16)
        network_alpha = input_data.get("network_alpha", 8)
        resolution = input_data.get("resolution", 1024)
        output_name = input_data.get("output_name", "lora")

        storage = {
            "r2_endpoint": os.environ.get("R2_ENDPOINT", ""),
            "r2_access_key": os.environ.get("R2_ACCESS_KEY", ""),
            "r2_secret_key": os.environ.get("R2_SECRET_KEY", ""),
            "r2_bucket": os.environ.get("R2_BUCKET", "lora-models"),
            "r2_public_url": os.environ.get("R2_PUBLIC_URL", "https://pub-b96a8c9acdb44455af16952661f6d90a.r2.dev"),
            "upload_path": f"{output_name}.safetensors",
        }

        if not image_urls:
            return {"status": "failed", "error": "No training images provided"}

        training_data_dir = f"/tmp/training_data/{trigger_word}"
        os.makedirs(training_data_dir, exist_ok=True)

        print(f"Downloading {len(image_urls)} training images...")
        for i, url in enumerate(image_urls):
            try:
                filename = download_image(url, training_data_dir)
                print(f"  [{i + 1}/{len(image_urls)}] Downloaded: {filename}")
            except Exception as e:
                print(f"  [{i + 1}/{len(image_urls)}] Failed to download: {url} - {e}")

        print("Generating auto-captions...")
        for img_file in os.listdir(training_data_dir):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                base_name = os.path.splitext(img_file)[0]
                caption_path = os.path.join(training_data_dir, f"{base_name}.txt")

                with open(caption_path, "w") as f:
                    f.write(f"{trigger_word} person")

        print("Generating dataset config...")
        dataset_config_path = generate_dataset_toml(training_data_dir, resolution)

        print("Starting Flux Dev LoRA training...")
        log_file = open("/tmp/training.log", "w")

        # Flux Dev LoRA training via kohya flux_train_network.py
        # Flags verified from kohya docs: docs/flux_train_network.md
        process = await asyncio.create_subprocess_exec(
            "accelerate",
            "launch",
            "--num_cpu_threads_per_process=1",
            "/kohya/flux_train_network.py",
            # Model paths
            "--pretrained_model_name_or_path", FLUX_MODEL,
            "--clip_l", FLUX_CLIP_L,
            "--t5xxl", FLUX_T5XXL,
            "--ae", FLUX_AE,
            # Dataset
            "--dataset_config", dataset_config_path,
            # Output
            "--output_dir", "/tmp/output",
            "--output_name", output_name,
            "--save_model_as", "safetensors",
            # Training params
            "--max_train_steps", str(steps),
            "--learning_rate", str(learning_rate),
            "--lr_scheduler", "cosine",
            "--lr_warmup_steps", "0",
            "--train_batch_size", "1",
            "--mixed_precision", "bf16",
            "--save_precision", "bf16",
            "--gradient_checkpointing",
            "--gradient_accumulation_steps", "1",
            "--optimizer_type", "AdamW8bit",
            # LoRA network config (Flux uses lora_flux, not lora)
            "--network_module", "networks.lora_flux",
            "--network_dim", str(network_dim),
            "--network_alpha", str(network_alpha),
            # Flux-specific flags (from kohya docs)
            "--timestep_sampling", "flux_shift",
            "--model_prediction_type", "raw",
            "--guidance_scale", "1.0",
            "--fp8_base",
            "--cache_text_encoder_outputs",
            "--cache_latents",
            # Memory optimization: swap 18 blocks to CPU (fits 24GB VRAM)
            "--blocks_to_swap", "18",
            stdout=log_file,
            stderr=asyncio.subprocess.STDOUT,
        )

        returncode = await process.wait()
        log_file.close()

        if returncode != 0:
            with open("/tmp/training.log", "r") as f:
                log_content = f.read()

            return {
                "status": "failed",
                "error": f"Training process failed with exit code {returncode}",
                "log": log_content[-2000:],
            }

        print("Training completed. Finding output LoRA...")
        lora_path = find_output_lora()

        file_size = os.path.getsize(lora_path)
        print(f"LoRA file size: {file_size} bytes")

        print("Uploading to R2...")
        r2_url = upload_to_r2(lora_path, storage)

        print(f"Upload complete: {r2_url}")

        return {"status": "completed", "lora_url": r2_url, "file_size": file_size}

    except Exception as e:
        print(f"Error during training: {str(e)}")

        log_content = ""
        if os.path.exists("/tmp/training.log"):
            with open("/tmp/training.log", "r") as f:
                log_content = f.read()[-2000:]

        return {"status": "failed", "error": str(e), "log": log_content}


if __name__ == "__main__":
    from fastapi import FastAPI, Request
    import uvicorn

    app = FastAPI()

    @app.post("/runsync")
    async def runsync(request: Request):
        body = await request.json()
        result = await handler({"input": body})
        return result

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    print("Starting local testing server on port 8000...")
    print("POST to http://localhost:8000/runsync with training payload")
    uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    ensure_gated_models()
    runpod.serverless.start({"handler": handler})
