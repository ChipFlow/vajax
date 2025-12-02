#!/bin/bash
# Setup local PDK environment for testing
# Adds PDK paths to shell profile
#
# Usage:
#   ./scripts/setup-pdk-local.sh

set -e

echo "=== Local PDK Setup ==="
echo ""
echo "This script configures environment variables for local PDK testing."
echo ""

# Detect shell
SHELL_RC=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_RC="$HOME/.bashrc"
elif [ -f "$HOME/.bash_profile" ]; then
    SHELL_RC="$HOME/.bash_profile"
else
    echo "Could not detect shell configuration file."
    echo "Please manually add the environment variables shown below."
    SHELL_RC=""
fi

# Prompt for PDK paths
echo "Enter the paths to your PDK installations."
echo "Leave blank to skip a PDK."
echo ""

read -p "GF130 PDK path: " GF130_PATH
read -p "IHP PDK path: " IHP_PATH
read -p "Skywater PDK path: " SKYWATER_PATH

# Validate paths
echo ""
for path_var in GF130_PATH IHP_PATH SKYWATER_PATH; do
    path="${!path_var}"
    if [ -n "$path" ]; then
        if [ -d "$path" ]; then
            echo "OK: $path_var -> $path"
        else
            echo "Warning: $path does not exist"
        fi
    fi
done

echo ""
echo "=== Environment Variables ==="
echo ""
echo "Add these to your shell configuration:"
echo ""
echo "# JAX-SPICE PDK paths"
[ -n "$GF130_PATH" ] && echo "export PDK_GF130_PATH=\"$GF130_PATH\""
[ -n "$IHP_PATH" ] && echo "export PDK_IHP_PATH=\"$IHP_PATH\""
[ -n "$SKYWATER_PATH" ] && echo "export PDK_SKYWATER_PATH=\"$SKYWATER_PATH\""
echo ""

if [ -n "$SHELL_RC" ]; then
    read -p "Add to $SHELL_RC automatically? [y/N]: " CONFIRM
    if [ "$CONFIRM" = "y" ] || [ "$CONFIRM" = "Y" ]; then
        echo "" >> "$SHELL_RC"
        echo "# JAX-SPICE PDK paths" >> "$SHELL_RC"
        [ -n "$GF130_PATH" ] && echo "export PDK_GF130_PATH=\"$GF130_PATH\"" >> "$SHELL_RC"
        [ -n "$IHP_PATH" ] && echo "export PDK_IHP_PATH=\"$IHP_PATH\"" >> "$SHELL_RC"
        [ -n "$SKYWATER_PATH" ] && echo "export PDK_SKYWATER_PATH=\"$SKYWATER_PATH\"" >> "$SHELL_RC"
        echo ""
        echo "Added to $SHELL_RC"
        echo "Run 'source $SHELL_RC' or open a new terminal to apply."
    fi
fi

echo ""
echo "=== Running PDK Tests ==="
echo ""
echo "To run PDK tests locally:"
echo ""
echo "  cd openvaf-py"
echo "  uv run pytest tests_pdk/ -v"
echo ""
echo "Tests will skip automatically if PDK paths are not set."
echo ""
