#!/usr/bin/env bash

# Wait for server to be ready
echo "Waiting for server..."
for i in {1..30}; do
    if curl -s http://localhost:8009/healthz > /dev/null; then
        echo "Server is up!"
        break
    fi
    sleep 2
    echo -n "."
done

echo ""
echo "=== Benchmarking 'Hello world' (2 words) ==="
start_time=$(date +%s.%N)
curl -s -X POST http://localhost:8009/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "temperature": 0.8
  }' --output /dev/null
end_time=$(date +%s.%N)
duration=$(echo "$end_time - $start_time" | bc)
echo "Duration: $duration seconds"

echo ""
echo "=== Benchmarking 'This is a test of the optimized inference server.' ==="
start_time=$(date +%s.%N)
curl -s -X POST http://localhost:8009/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a test of the optimized inference server.",
    "temperature": 0.8
  }' --output /dev/null
end_time=$(date +%s.%N)
duration=$(echo "$end_time - $start_time" | bc)
echo "Duration: $duration seconds"
