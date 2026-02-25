#!/bin/bash
# Setup script for PDK CI access
# Run this on a machine with GitHub CLI authenticated
#
# This script is idempotent - safe to run multiple times.
#
# This script:
# 1. Generates an SSH deploy key for CI access to private PDK repos
# 2. Adds the public key to PDK repos as a deploy key (skips if exists)
# 3. Adds the private key as a secret to the vajax repo (overwrites if exists)

set -e

REPO="ChipFlow/vajax"
PDK_REPOS=("ChipFlow/pdk-gf130")
KEY_TITLE="vajax-ci"

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

# Check we have access to the repos
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

# Check if deploy keys already exist
KEYS_EXIST=true
for repo in "${PDK_REPOS[@]}"; do
    if ! gh repo deploy-key list --repo "$repo" 2>/dev/null | grep -q "$KEY_TITLE"; then
        KEYS_EXIST=false
        break
    fi
done

if $KEYS_EXIST; then
    echo "Deploy keys already exist for all PDK repos."
    echo "To regenerate, first delete existing keys:"
    for repo in "${PDK_REPOS[@]}"; do
        echo "  gh repo deploy-key delete <key-id> --repo $repo"
    done
    echo ""
    read -p "Continue anyway and update the secret? [y/N]: " CONFIRM
    if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
        echo "Aborted."
        exit 0
    fi
    echo ""
fi

# Generate deploy key
KEY_DIR="$(mktemp -d)"
KEY_FILE="${KEY_DIR}/pdk_deploy_key"
echo "Generating deploy key..."
ssh-keygen -t ed25519 -C "vajax-ci-$(date +%Y%m%d)" -f "$KEY_FILE" -N ""
echo ""

# Add deploy key to PDK repos
echo "=== Adding Deploy Keys to PDK Repos ==="
for repo in "${PDK_REPOS[@]}"; do
    # Check if key already exists
    if gh repo deploy-key list --repo "$repo" 2>/dev/null | grep -q "$KEY_TITLE"; then
        echo "Deploy key '$KEY_TITLE' already exists in $repo, skipping."
    else
        echo "Adding deploy key to $repo..."
        if gh repo deploy-key add "${KEY_FILE}.pub" --repo "$repo" --title "$KEY_TITLE" 2>/dev/null; then
            echo "  Done."
        else
            echo "  Warning: Could not add key (insufficient permissions?)"
            echo "  Manual: https://github.com/${repo}/settings/keys"
        fi
    fi
done
echo ""

# Add private key as secret (always overwrites)
echo "=== Adding Secret to $REPO ==="
echo "Setting PDK_DEPLOY_KEY secret..."
if gh secret set PDK_DEPLOY_KEY --repo "$REPO" < "$KEY_FILE"; then
    echo "  Done."
else
    echo "  Error: Could not set secret"
    echo "  Manual: gh secret set PDK_DEPLOY_KEY --repo $REPO < \"$KEY_FILE\""
    rm -rf "$KEY_DIR"
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
