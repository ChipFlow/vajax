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
uv sync --locked --extra cuda12

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

echo "Running GPU profiler..."
uv run python scripts/profile_gpu.py

echo "Running tests..."
uv run pytest tests/ -v --tb=short -x
