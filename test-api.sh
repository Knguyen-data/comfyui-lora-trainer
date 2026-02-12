#!/bin/bash
# LoRA Training API Test Script
# Tests RunPod endpoint 3gtffjptcviz6c

ENDPOINT_ID="3gtffjptcviz6c"
API_KEY="${RUNPOD_API_KEY:-your_api_key_here}"
BASE_URL="https://api.runpod.ai/v2/${ENDPOINT_ID}"

echo "=========================================="
echo "LoRA Training API Test Suite"
echo "=========================================="

# Mock training images (these would be real Supabase URLs in production)
MOCK_IMAGES='[
  "https://example.com/training/image1.jpg",
  "https://example.com/training/image2.jpg",
  "https://example.com/training/image3.jpg"
]'

# Mock payload matching lora-model-service.ts format
MOCK_PAYLOAD='{
  "mode": "train_lora",
  "trigger_word": "ohwx",
  "training_images": '"$MOCK_IMAGES"',
  "steps": 100,
  "learning_rate": 0.001,
  "output_name": "test_lora_'"$(date +%s)"'"
}'

echo ""
echo "1. Testing /health endpoint..."
echo "------------------------------------------"
curl -s "${BASE_URL}/health" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "accept: application/json"
echo ""
echo ""

echo "2. Testing /runsync (synchronous training)..."
echo "------------------------------------------"
echo "Payload: $MOCK_PAYLOAD"
echo ""

RESPONSE=$(curl -s -X POST "${BASE_URL}/runsync" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d "$MOCK_PAYLOAD")

echo "Response:"
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
echo ""

# Extract job_id if present
JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id', d.get('jobId', '')))" 2>/dev/null)

if [ -n "$JOB_ID" ]; then
  echo "Job ID: $JOB_ID"
  echo ""

  echo "3. Testing /status/${JOB_ID}..."
  echo "------------------------------------------"
  curl -s "${BASE_URL}/status/${JOB_ID}" \
    -H "Authorization: Bearer ${API_KEY}"
  echo ""
  echo ""

  echo "4. Testing /stream/${JOB_ID}..."
  echo "------------------------------------------"
  curl -s "${BASE_URL}/stream/${JOB_ID}" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}"
  echo ""
  echo ""
else
  echo "No job ID returned - testing /run (async) instead..."
  echo ""

  echo "3. Testing /run (asynchronous training)..."
  echo "------------------------------------------"
  ASYNC_RESPONSE=$(curl -s -X POST "${BASE_URL}/run" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d "$MOCK_PAYLOAD")

  echo "Response:"
  echo "$ASYNC_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$ASYNC_RESPONSE"
  echo ""

  ASYNC_JOB_ID=$(echo "$ASYNC_RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id', ''))" 2>/dev/null)
  if [ -n "$ASYNC_JOB_ID" ]; then
    echo "Async Job ID: $ASYNC_JOB_ID"
    echo ""

    echo "4. Polling /status/${ASYNC_JOB_ID}..."
    echo "------------------------------------------"
    for i in {1..5}; do
      STATUS=$(curl -s "${BASE_URL}/status/${ASYNC_JOB_ID}" \
        -H "Authorization: Bearer ${API_KEY}")
      echo "Attempt $i: $STATUS"
      echo ""

      STATUS_CHECK=$(echo "$STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status', ''))" 2>/dev/null)
      if [ "$STATUS_CHECK" = "COMPLETED" ] || [ "$STATUS_CHECK" = "FAILED" ]; then
        break
      fi
      sleep 2
    done
  fi
fi

echo "=========================================="
echo "Test suite completed!"
echo "=========================================="
