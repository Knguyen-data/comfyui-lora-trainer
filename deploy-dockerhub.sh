#!/bin/bash
set -e

DOCKERHUB_USER="kie1"
IMAGE_NAME="lora-trainer"
TAG="v1.0"

echo "========================================"
echo "Building LoRA Trainer Docker Image"
echo "========================================"
echo ""
echo "Image: ${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"
echo ""

echo "Building Docker image..."
docker build -t ${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG} .

echo ""
echo "Tagging as latest..."
docker tag ${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG} ${DOCKERHUB_USER}/${IMAGE_NAME}:latest

echo ""
echo "Pushing to Docker Hub..."
docker push ${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}
docker push ${DOCKERHUB_USER}/${IMAGE_NAME}:latest

echo ""
echo "========================================"
echo "Deployment Complete!"
echo "========================================"
echo ""
echo "Image: ${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"
echo "Latest: ${DOCKERHUB_USER}/${IMAGE_NAME}:latest"
echo ""
echo "Next steps:"
echo "1. Go to https://runpod.io/serverless"
echo "2. Create new endpoint:"
echo "   - Name: LoRA-Trainer-SDXL"
echo "   - Container: ${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"
echo "   - GPU: RTX 4090 or A100 40GB"
echo "   - Timeout: 1800s"
echo "3. Save endpoint ID and update frontend config"
echo ""
