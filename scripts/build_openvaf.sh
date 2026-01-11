#!/bin/bash
# Build OpenVAF-reloaded compiler from vendored source
#
# Prerequisites (macOS with Homebrew):
#   brew install llvm@18 rust
#
# Prerequisites (Linux):
#   Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
#   Build LLVM 18.1.8 from source (see vendor/OpenVAF/README.md)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENVAF_SRC="$PROJECT_ROOT/vendor/OpenVAF"

# Check if OpenVAF source exists
if [ ! -d "$OPENVAF_SRC" ]; then
    echo "Error: OpenVAF source not found at $OPENVAF_SRC"
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

# Check for Rust
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust/Cargo not found"
    echo "Install with: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Find LLVM 18
case "$(uname -s)" in
    Darwin*)
        BREW_PREFIX="$(brew --prefix 2>/dev/null || echo /opt/homebrew)"
        LLVM_PREFIX="$BREW_PREFIX/opt/llvm@18"
        if [ ! -d "$LLVM_PREFIX" ]; then
            echo "Error: LLVM 18 not found at $LLVM_PREFIX"
            echo "Install with: brew install llvm@18"
            exit 1
        fi
        ;;
    Linux*)
        # Try common locations
        for path in /usr/lib/llvm-18 /usr/local/llvm-18 "$HOME/llvm-18"; do
            if [ -d "$path" ]; then
                LLVM_PREFIX="$path"
                break
            fi
        done
        if [ -z "$LLVM_PREFIX" ]; then
            echo "Error: LLVM 18 not found"
            echo "Build LLVM 18.1.8 from source (see vendor/OpenVAF/README.md)"
            exit 1
        fi
        ;;
    *)
        echo "Unsupported platform: $(uname -s)"
        exit 1
        ;;
esac

echo "Using LLVM at: $LLVM_PREFIX"

# Set environment for llvm-sys crate
export LLVM_SYS_181_PREFIX="$LLVM_PREFIX"

# Build OpenVAF
cd "$OPENVAF_SRC"

echo ""
echo "Building OpenVAF-reloaded..."
echo "  Source: $OPENVAF_SRC"
echo "  LLVM: $LLVM_PREFIX"
echo ""

# Build using xtask (handles macOS linking correctly)
cargo xtask cargo-build --release

# Check result
OPENVAF_BIN="$OPENVAF_SRC/target/release/openvaf-r"
if [ -x "$OPENVAF_BIN" ]; then
    echo ""
    echo "Build successful!"
    echo "OpenVAF binary: $OPENVAF_BIN"
    echo ""

    # Show version
    "$OPENVAF_BIN" --version || true

    echo ""
    echo "To use with VACASK build:"
    echo "  export OPENVAF_DIR=\"$OPENVAF_SRC/target/release\""
else
    echo ""
    echo "Build completed but openvaf-r binary not found."
    exit 1
fi
