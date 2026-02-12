import asyncio
import base64
import os
import glob
import toml
import runpod
import boto3
from botocore.client import Config
from pathlib import Path


def save_base64_image(base64_data: str, output_path: str):
    """Decode and save base64 image to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if ',' in base64_data:
        base64_data = base64_data.split(',')[1]

    image_bytes = base64.b64decode(base64_data)

    with open(output_path, 'wb') as f:
        f.write(image_bytes)


def generate_toml_config(config: dict, training_data_dir: str) -> str:
    """Generate kohya TOML configuration file."""

    steps = config.get('steps', 1500)
    lr = config.get('lr', 1e-4)
    network_dim = config.get('network_dim', 32)
    network_alpha = config.get('network_alpha', 16)
    resolution = config.get('resolution', 1024)

    toml_config = {
        'general': {
            'enable_bucket': True,
            'bucket_reso_steps': 64,
            'bucket_no_upscale': False
        },
        'model_arguments': {
            'pretrained_model_name_or_path': '/workspace/models/sd_xl_base_1.0.safetensors'
        },
        'dataset_arguments': {
            'resolution': resolution,
            'train_data_dir': training_data_dir,
            'enable_bucket': True,
            'min_bucket_reso': 256,
            'max_bucket_reso': 2048
        },
        'training_arguments': {
            'output_dir': '/tmp/output',
            'output_name': 'lora',
            'save_model_as': 'safetensors',
            'max_train_steps': steps,
            'learning_rate': lr,
            'lr_scheduler': 'cosine',
            'lr_warmup_steps': 0,
            'train_batch_size': 1,
            'mixed_precision': 'fp16',
            'save_precision': 'fp16',
            'gradient_checkpointing': True,
            'gradient_accumulation_steps': 1,
            'optimizer_type': 'AdamW8bit',
            'network_module': 'networks.lora',
            'network_dim': network_dim,
            'network_alpha': network_alpha,
            'xformers': True,
            'sdpa': False
        }
    }

    config_path = '/tmp/training_config.toml'
    with open(config_path, 'w') as f:
        toml.dump(toml_config, f)

    return config_path


def upload_to_r2(file_path: str, storage: dict) -> str:
    """Upload file to Cloudflare R2 storage."""

    s3 = boto3.client(
        's3',
        endpoint_url=storage['r2_endpoint'],
        aws_access_key_id=storage['r2_access_key'],
        aws_secret_access_key=storage['r2_secret_key'],
        config=Config(signature_version='s3v4')
    )

    upload_path = storage['upload_path']
    bucket = storage['r2_bucket']

    s3.upload_file(file_path, bucket, upload_path)

    return f"{storage['r2_endpoint']}/{bucket}/{upload_path}"


def find_output_lora() -> str:
    """Find the trained LoRA file in output directory."""
    lora_files = glob.glob('/tmp/output/*.safetensors')

    if not lora_files:
        raise FileNotFoundError("No .safetensors file found in /tmp/output/")

    return lora_files[0]


async def handler(event: dict) -> dict:
    """
    RunPod handler for LoRA training.

    Expected input format:
    {
        "training_images": [
            {"filename": "image1.jpg", "data": "base64..."},
            {"filename": "image2.jpg", "data": "base64..."}
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
            "r2_endpoint": "https://...",
            "r2_access_key": "...",
            "r2_secret_key": "...",
            "r2_bucket": "lora-models",
            "upload_path": "user123/lora_20240101.safetensors"
        }
    }
    """

    try:
        input_data = event.get("input", {})

        training_images = input_data.get("training_images", [])
        config = input_data.get("config", {})
        storage = input_data.get("storage", {})

        if not training_images:
            return {
                "status": "failed",
                "error": "No training images provided"
            }

        trigger_word = config.get("trigger_word", "ohwx")

        training_data_dir = f"/tmp/training_data/{trigger_word}"
        os.makedirs(training_data_dir, exist_ok=True)

        print(f"Saving {len(training_images)} training images...")
        for img in training_images:
            filename = img.get("filename", "image.jpg")
            image_data = img.get("data", "")

            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filename += '.jpg'

            image_path = os.path.join(training_data_dir, filename)
            save_base64_image(image_data, image_path)

        print("Generating auto-captions...")
        for img_file in os.listdir(training_data_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                base_name = os.path.splitext(img_file)[0]
                caption_path = os.path.join(training_data_dir, f"{base_name}.txt")

                with open(caption_path, 'w') as f:
                    f.write(f"{trigger_word} person")

        print("Generating training config...")
        config_path = generate_toml_config(config, '/tmp/training_data')

        print("Starting LoRA training...")
        log_file = open("/tmp/training.log", "w")

        process = await asyncio.create_subprocess_exec(
            "accelerate", "launch",
            "--num_cpu_threads_per_process=1",
            "/workspace/kohya/sdxl_train_network.py",
            "--config_file", config_path,
            stdout=log_file,
            stderr=asyncio.subprocess.STDOUT
        )

        returncode = await process.wait()
        log_file.close()

        if returncode != 0:
            with open("/tmp/training.log", "r") as f:
                log_content = f.read()

            return {
                "status": "failed",
                "error": f"Training process failed with exit code {returncode}",
                "log": log_content[-2000:]
            }

        print("Training completed. Finding output LoRA...")
        lora_path = find_output_lora()

        file_size = os.path.getsize(lora_path)
        print(f"LoRA file size: {file_size} bytes")

        print("Uploading to R2...")
        r2_url = upload_to_r2(lora_path, storage)

        print(f"Upload complete: {r2_url}")

        return {
            "status": "completed",
            "lora_url": r2_url,
            "file_size": file_size
        }

    except Exception as e:
        print(f"Error during training: {str(e)}")

        log_content = ""
        if os.path.exists("/tmp/training.log"):
            with open("/tmp/training.log", "r") as f:
                log_content = f.read()[-2000:]

        return {
            "status": "failed",
            "error": str(e),
            "log": log_content
        }


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
    runpod.serverless.start({"handler": handler})
