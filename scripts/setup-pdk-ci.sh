#!/bin/bash
# Setup script for PDK CI access
# Run this on a machine with GitHub CLI authenticated
#
# This script:
# 1. Generates an SSH deploy key for CI access to private PDK repos
# 2. Provides instructions to add the public key to PDK repos
# 3. Provides instructions to add the private key as a secret

set -e

REPO="ChipFlow/jax-spice"
PDK_REPOS=("ChipFlow/pdk-gf130" "ChipFlow/pdk-ihp")

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

# Generate deploy key
KEY_DIR="$(mktemp -d)"
KEY_FILE="${KEY_DIR}/pdk_deploy_key"
echo "Generating deploy key..."
ssh-keygen -t ed25519 -C "jax-spice-ci-$(date +%Y%m%d)" -f "$KEY_FILE" -N ""

echo ""
echo "=== Deploy Key Generated ==="
echo ""
echo "Private key: $KEY_FILE"
echo "Public key:  ${KEY_FILE}.pub"
echo ""

# Show public key
echo "=== Public Key (add to PDK repos) ==="
echo ""
cat "${KEY_FILE}.pub"
echo ""

# Provide instructions
echo "=== Setup Instructions ==="
echo ""
echo "STEP 1: Add the PUBLIC key to each PDK repo as a deploy key (read-only access):"
echo ""
for repo in "${PDK_REPOS[@]}"; do
    echo "  - https://github.com/${repo}/settings/keys"
done
echo ""
echo "  Click 'Add deploy key', paste the public key above, and check 'Allow read access'."
echo ""

echo "STEP 2: Add the PRIVATE key as a secret to the jax-spice repo:"
echo ""
echo "  Option A - Use gh CLI (recommended):"
echo "    gh secret set PDK_DEPLOY_KEY --repo $REPO < \"$KEY_FILE\""
echo ""
echo "  Option B - Manual:"
echo "    1. Go to: https://github.com/${REPO}/settings/secrets/actions"
echo "    2. Click 'New repository secret'"
echo "    3. Name: PDK_DEPLOY_KEY"
echo "    4. Value: Paste the contents of $KEY_FILE"
echo ""

echo "STEP 3: Verify the workflow runs:"
echo ""
echo "  After adding the keys, push a commit or open a PR to trigger the PDK tests."
echo "  Check the Actions tab: https://github.com/${REPO}/actions/workflows/test-pdk.yml"
echo ""

echo "=== Security Notes ==="
echo ""
echo "- Keep the private key file secure until you've added it as a secret"
echo "- After setup is complete, delete the key files:"
echo "    rm -rf $KEY_DIR"
echo "- The deploy key only has read access to the PDK repos"
echo "- PDK paths are masked in CI logs to prevent leakage"
echo ""
