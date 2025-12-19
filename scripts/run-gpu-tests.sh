#!/bin/bash
set -e

# Environment variables expected:
# - GITHUB_REPOSITORY: e.g., "ChipFlow/jax-spice"
# - GITHUB_SHA: commit to checkout
# - GITHUB_TOKEN: for private repo access (optional for public repos)

echo "=== GPU Test Runner ==="
echo "Repository: ${GITHUB_REPOSITORY}"
echo "Commit: ${GITHUB_SHA}"

cd /app

# Clone the repository at the specific commit
echo "Cloning repository..."
if [ -n "$GITHUB_TOKEN" ]; then
    git clone --depth 1 --recurse-submodules \
        "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" \
        --branch main source
    cd source
    git fetch --depth 1 origin "$GITHUB_SHA"
    git checkout "$GITHUB_SHA"
else
    git clone --depth 1 --recurse-submodules \
        "https://github.com/${GITHUB_REPOSITORY}.git" \
        --branch main source
    cd source
    git fetch --depth 1 origin "$GITHUB_SHA"
    git checkout "$GITHUB_SHA"
fi

# Update submodules if needed
git submodule update --init --recursive

echo "=== sccache diagnostics (before build) ==="
sccache --show-stats || echo "sccache stats not available"

echo "Installing workspace packages..."
# Install the workspace in the pre-existing venv
uv sync --locked --extra cuda12 --extra dev

echo "=== sccache diagnostics (after build) ==="
sccache --show-stats || echo "sccache stats not available"

echo "=== CUDA Environment Diagnostics ==="
# Check NVIDIA driver version
nvidia-smi || echo "nvidia-smi not available"

# Check installed CUDA packages
uv pip list | grep -i nvidia || echo "No nvidia packages found"
uv pip list | grep -i cuda || echo "No cuda packages found"

# Check JAX CUDA detection
uv run python -c "
import jax
print('JAX version:', jax.__version__)
print('JAX backend:', jax.default_backend())
print('JAX devices:', jax.devices())

# Check for GPU devices
gpu_devices = [d for d in jax.devices() if d.platform != 'cpu']
if gpu_devices:
    print('GPU devices found:', gpu_devices)
    print('GPU device kind:', gpu_devices[0].device_kind)
else:
    print('No GPU devices found')
"

echo "Running JAX-SPICE vs VACASK benchmark comparison..."
# Enable JAX profiling to capture GPU traces
uv run python scripts/compare_vacask.py \
  --benchmark rc,graetz,ring,c6288 \
  --max-steps 200 \
  --use-scan \
  --profile \
  --profile-dir /tmp/jax-spice-traces

# Upload traces to GCS for artifact download
echo "=== Uploading profiling traces to GCS ==="
GCS_BUCKET="jax-spice-cuda-test-traces"
TRACE_PATH="${GITHUB_SHA:-$(date +%s)}"

# Get access token from metadata server (workload identity)
TOKEN=$(curl -s -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])" 2>/dev/null || echo "")

if [ -n "$TOKEN" ] && [ -d "/tmp/jax-spice-traces" ]; then
  echo "Uploading traces to gs://${GCS_BUCKET}/${TRACE_PATH}/"

  # JAX profiler creates subdirectories for each trace (e.g., benchmark_rc/plugins/...)
  # Find all actual files recursively
  file_count=0
  find /tmp/jax-spice-traces -type f | while read -r f; do
    # Get relative path from trace dir
    relpath="${f#/tmp/jax-spice-traces/}"
    echo "  Uploading ${relpath}..."
    curl -s -X PUT -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/octet-stream" \
      --data-binary @"$f" \
      "https://storage.googleapis.com/upload/storage/v1/b/${GCS_BUCKET}/o?uploadType=media&name=${TRACE_PATH}/${relpath}" || true
    file_count=$((file_count + 1))
  done

  echo "Traces uploaded to: gs://${GCS_BUCKET}/${TRACE_PATH}/"
  echo "TRACE_GCS_PATH=${TRACE_PATH}" >> /tmp/trace_info.env
else
  echo "Skipping trace upload (no token or no traces)"
fi

echo "Running tests..."
uv run pytest tests/ -v --tb=short -x
