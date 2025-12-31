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

# Set up LD_LIBRARY_PATH for NVIDIA pip packages
# JAX's pip packages install CUDA libraries to site-packages/nvidia/*/lib
NVIDIA_BASE=".venv/lib/python3.13/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_BASE}/cuda_runtime/lib:${NVIDIA_BASE}/cublas/lib:${NVIDIA_BASE}/cusparse/lib:${NVIDIA_BASE}/cudnn/lib:${NVIDIA_BASE}/cufft/lib:${NVIDIA_BASE}/cusolver/lib:${NVIDIA_BASE}/nvjitlink/lib:${NVIDIA_BASE}/nccl/lib:${NVIDIA_BASE}/cu12/lib:${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH set for NVIDIA packages"

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
# Enable JAX profiling to capture GPU traces (Perfetto format)
# Note: nsys-jax profiling can be enabled with --profile-mode=nsys but
# generates larger output files - use for detailed GPU kernel analysis
# Running single benchmark (ring) for faster iteration and cleaner Perfetto traces
# Use --force-gpu to ensure GPU is used even for small circuits (ring has only 47 nodes)
uv run python scripts/compare_vacask.py \
  --benchmark ring \
  --max-steps 50 \
  --use-scan \
  --force-gpu \
  --profile-mode jax \
  --profile-dir /tmp/jax-spice-traces

# Upload traces to GCS for artifact download
echo "=== Uploading profiling traces to GCS ==="
GCS_BUCKET="jax-spice-cuda-test-traces"
TRACE_PATH="${GITHUB_SHA:-$(date +%s)}"

if [ -d "/tmp/jax-spice-traces" ]; then
  echo "Uploading traces to gs://${GCS_BUCKET}/${TRACE_PATH}/"

  # Use gsutil with workload identity credentials (auto-detected from metadata server)
  gsutil -m cp -r /tmp/jax-spice-traces/* "gs://${GCS_BUCKET}/${TRACE_PATH}/" || {
    echo "Warning: Failed to upload traces (gsutil error)"
  }

  echo "Traces uploaded to: gs://${GCS_BUCKET}/${TRACE_PATH}/"
  echo "TRACE_GCS_PATH=${TRACE_PATH}" >> /tmp/trace_info.env
else
  echo "Skipping trace upload (no traces found)"
fi

echo "Running tests..."
uv run pytest tests/ -v --tb=short -x
