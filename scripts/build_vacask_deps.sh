#!/bin/bash
# Build OpenVAF and VACASK dependencies for JAX-SPICE benchmarks
#
# This script:
# 1. Checks out the macos-patches-v3 branch of OpenVAF (fixes macOS linking)
# 2. Builds openvaf with the correct LLVM path
# 3. Reconfigures VACASK to use our openvaf binary
# 4. Builds the VACASK simulator and OSDI devices
#
# Usage:
#   ./scripts/build_vacask_deps.sh
#
# Requirements:
#   - LLVM 18 installed (brew install llvm@18)
#   - Rust toolchain
#   - CMake and Ninja

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OPENVAF_DIR="$PROJECT_ROOT/openvaf-py/vendor/OpenVAF"
VACASK_DIR="$PROJECT_ROOT/vendor/VACASK"

# LLVM 18 path (adjust if needed)
LLVM_PREFIX="${LLVM_PREFIX:-/opt/homebrew/opt/llvm@18}"

echo "=== Building OpenVAF and VACASK Dependencies ==="
echo "Project root: $PROJECT_ROOT"
echo "LLVM prefix: $LLVM_PREFIX"
echo

# Check LLVM exists
if [ ! -d "$LLVM_PREFIX" ]; then
    echo "ERROR: LLVM 18 not found at $LLVM_PREFIX"
    echo "Install with: brew install llvm@18"
    echo "Or set LLVM_PREFIX environment variable"
    exit 1
fi

# Step 1: Checkout macos-patches-v3 branch of OpenVAF
echo "=== Step 1: Checking out OpenVAF macos-patches-v3 branch ==="
cd "$OPENVAF_DIR"
git fetch origin
git checkout origin/branches/macos-patches-v3
echo "OpenVAF branch: $(git describe --always)"
echo

# Step 2: Build OpenVAF
echo "=== Step 2: Building OpenVAF ==="
cd "$OPENVAF_DIR"

# Always rebuild to ensure binary matches checked out branch
OPENVAF_BIN="$OPENVAF_DIR/target/release/openvaf-r"
LLVM_SYS_181_PREFIX="$LLVM_PREFIX" cargo build --release --bin openvaf-r
echo "OpenVAF binary: $OPENVAF_BIN"
echo

# Verify openvaf works
if [ ! -x "$OPENVAF_BIN" ]; then
    echo "ERROR: openvaf binary not found at $OPENVAF_BIN"
    exit 1
fi
echo "OpenVAF version: $($OPENVAF_BIN --version)"
echo

# Step 3: Reconfigure VACASK to use our openvaf
echo "=== Step 3: Configuring VACASK ==="
cd "$VACASK_DIR"

# Remove old build directory to force reconfigure
if [ -d "build" ]; then
    echo "Removing old build directory..."
    rm -rf build
fi

mkdir -p build
cd build

cmake .. \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DOPENVAF_COMPILER="$OPENVAF_BIN"

echo

# Step 4: Build VACASK simulator
echo "=== Step 4: Building VACASK simulator ==="
ninja vacask
echo

# Step 5: Build OSDI devices for benchmarks
echo "=== Step 5: Building OSDI devices ==="
# Build basic devices for simpler benchmarks
ninja devices/spice/resistor.osdi \
      devices/spice/capacitor.osdi \
      devices/spice/diode.osdi \
      devices/spice/inductor.osdi \
      devices/resistor.osdi \
      devices/capacitor.osdi \
      devices/diode.osdi

# Build additional devices needed for graetz (sn diode) and ring (psp103)
echo "=== Building additional devices for graetz and ring ==="
ninja devices/psp103v4.osdi || true

# Build spice/sn/diode for graetz benchmark (different from standard diode)
# Check if it exists as a target
ninja -t targets | grep -q "spice/sn/diode" && ninja devices/spice/sn/diode.osdi || echo "Note: spice/sn/diode not available as separate target"

echo

# Step 6: Stage devices for benchmarks
echo "=== Step 6: Staging devices for benchmarks ==="
# Each benchmark directory needs appropriate OSDI files
BENCHMARKS="rc graetz ring c6288"
for bench in $BENCHMARKS; do
    BENCH_DIR="$VACASK_DIR/benchmark/$bench/vacask"
    if [ -d "$BENCH_DIR" ]; then
        mkdir -p "$BENCH_DIR/spice"
        # Copy spice-style devices
        cp -f "$VACASK_DIR/build/devices/spice/resistor.osdi" "$BENCH_DIR/spice/" 2>/dev/null || true
        cp -f "$VACASK_DIR/build/devices/spice/capacitor.osdi" "$BENCH_DIR/spice/" 2>/dev/null || true
        cp -f "$VACASK_DIR/build/devices/spice/diode.osdi" "$BENCH_DIR/spice/" 2>/dev/null || true
        cp -f "$VACASK_DIR/build/devices/spice/inductor.osdi" "$BENCH_DIR/spice/" 2>/dev/null || true

        # Copy additional devices for specific benchmarks
        # ring needs psp103v4
        cp -f "$VACASK_DIR/build/devices/psp103v4.osdi" "$BENCH_DIR/" 2>/dev/null || true

        # graetz needs sn/diode (spice-like diode for sn model)
        if [ -d "$VACASK_DIR/build/devices/spice/sn" ]; then
            mkdir -p "$BENCH_DIR/spice/sn"
            cp -f "$VACASK_DIR/build/devices/spice/sn/"*.osdi "$BENCH_DIR/spice/sn/" 2>/dev/null || true
        fi

        echo "Staged devices for $bench"
    fi
done

echo
echo "=== Build Complete ==="
echo "VACASK binary: $VACASK_DIR/build/simulator/vacask"
echo
echo "To run benchmarks:"
echo "  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run scripts/compare_vacask.py --use-scan"
