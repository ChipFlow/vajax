#!/bin/bash
# Setup script for PDK CI access
# Run this on a machine with GitHub CLI authenticated
#
# This script:
# 1. Generates an SSH deploy key for CI access to private PDK repos
# 2. Adds the public key to PDK repos as a deploy key
# 3. Adds the private key as a secret to the jax-spice repo

set -e

REPO="ChipFlow/jax-spice"
PDK_REPOS=("ChipFlow/pdk-gf130")

echo "=== PDK CI Setup Script ==="
echo ""

# Check prerequisites
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is required"
    echo "Install with: brew install gh"
    exit 1
fi

if ! gh auth status &> /dev/null; then
    echo "Error: Please authenticate with 'gh auth login'"
    exit 1
fi

# Check we have admin access to the repos
echo "Checking repository access..."
for repo in "${PDK_REPOS[@]}"; do
    if ! gh repo view "$repo" &> /dev/null; then
        echo "Error: Cannot access $repo"
        echo "Make sure you have admin access to add deploy keys."
        exit 1
    fi
done

if ! gh repo view "$REPO" &> /dev/null; then
    echo "Error: Cannot access $REPO"
    exit 1
fi
echo "Access verified."
echo ""

# Generate deploy key
KEY_DIR="$(mktemp -d)"
KEY_FILE="${KEY_DIR}/pdk_deploy_key"
echo "Generating deploy key..."
ssh-keygen -t ed25519 -C "jax-spice-ci-$(date +%Y%m%d)" -f "$KEY_FILE" -N ""
echo ""

# Add deploy key to PDK repos
echo "=== Adding Deploy Keys to PDK Repos ==="
for repo in "${PDK_REPOS[@]}"; do
    echo "Adding deploy key to $repo..."
    if gh repo deploy-key add "${KEY_FILE}.pub" --repo "$repo" --title "jax-spice-ci" 2>/dev/null; then
        echo "  Done."
    else
        echo "  Warning: Could not add key (may already exist or insufficient permissions)"
        echo "  Manual: https://github.com/${repo}/settings/keys"
    fi
done
echo ""

# Add private key as secret
echo "=== Adding Secret to $REPO ==="
echo "Setting PDK_DEPLOY_KEY secret..."
if gh secret set PDK_DEPLOY_KEY --repo "$REPO" < "$KEY_FILE"; then
    echo "  Done."
else
    echo "  Error: Could not set secret"
    echo "  Manual: gh secret set PDK_DEPLOY_KEY --repo $REPO < \"$KEY_FILE\""
    exit 1
fi
echo ""

# Cleanup
echo "=== Cleanup ==="
echo "Removing temporary key files..."
rm -rf "$KEY_DIR"
echo "  Done."
echo ""

echo "=== Setup Complete ==="
echo ""
echo "The PDK CI workflow is now configured."
echo "Push a commit or open a PR to trigger the PDK tests:"
echo "  https://github.com/${REPO}/actions/workflows/test-pdk.yml"
echo ""
echo "Security notes:"
echo "- Deploy keys have read-only access to PDK repos"
echo "- PDK paths are masked in CI logs"
echo ""
