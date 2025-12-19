#!/bin/bash
# Setup script for GCP GPU CI infrastructure
# This script is fully idempotent - safe to run multiple times
#
# Usage: ./setup_gcp_gpu_ci.sh [--with-vm]
#
# Features:
# - Creates service account with minimal permissions
# - Stores service account key in GCP Secret Manager
# - Syncs secret to GitHub Actions
# - Creates Artifact Registry remote repo for ghcr.io caching
# - Creates sccache GCS bucket for Rust compilation cache
# - Provisions GPU VM for testing (optional, use --with-vm)
#
# Prerequisites:
# - gcloud CLI installed and authenticated
# - gh CLI installed and authenticated (for GitHub secret sync)
# - Billing enabled on the GCP project
# - GPU quota available in the region

set -euo pipefail

# Parse arguments
SETUP_VM=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --with-vm)
            SETUP_VM=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--with-vm]"
            exit 1
            ;;
    esac
done

# Configuration
PROJECT_ID="${GCP_PROJECT:-jax-spice-cuda-test}"
REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-us-central1-f}"
VM_NAME="${GCP_VM_NAME:-jax-spice-cuda}"
SA_NAME="github-gpu-ci"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
SECRET_NAME="github-gpu-ci-key"
GITHUB_REPO="${GITHUB_REPO:-ChipFlow/jax-spice}"

# Cloud Run specific configuration
AR_REMOTE_REPO="ghcr-remote"
SCCACHE_BUCKET="jax-spice-sccache"
TRACES_BUCKET="jax-spice-cuda-test-traces"

# Machine configuration
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
BOOT_DISK_SIZE="100GB"
IMAGE_FAMILY="common-cu128-ubuntu-2204-nvidia-570"
IMAGE_PROJECT="deeplearning-platform-release"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "=========================================="
echo "  JAX-SPICE GPU CI Setup (Idempotent)"
echo "=========================================="
echo ""
echo "Project:     ${PROJECT_ID}"
echo "Zone:        ${ZONE}"
echo "VM Name:     ${VM_NAME}"
echo "GitHub Repo: ${GITHUB_REPO}"
echo ""

# Set project
gcloud config set project "${PROJECT_ID}" --quiet

# =============================================================================
# Step 1: Enable required APIs (idempotent)
# =============================================================================
log_info "Enabling required APIs..."
gcloud services enable compute.googleapis.com --quiet
gcloud services enable secretmanager.googleapis.com --quiet
gcloud services enable iam.googleapis.com --quiet
gcloud services enable run.googleapis.com --quiet
gcloud services enable artifactregistry.googleapis.com --quiet
log_info "APIs enabled"

# =============================================================================
# Step 2: Create service account (idempotent)
# =============================================================================
log_info "Setting up service account..."
if gcloud iam service-accounts describe "${SA_EMAIL}" &>/dev/null; then
    log_info "Service account already exists: ${SA_EMAIL}"
else
    gcloud iam service-accounts create "${SA_NAME}" \
        --display-name="GitHub GPU CI Runner" \
        --quiet
    log_info "Created service account: ${SA_EMAIL}"
fi

# =============================================================================
# Step 3: Grant permissions (idempotent - add-iam-policy-binding is idempotent)
# =============================================================================
log_info "Configuring IAM permissions..."

# Define required roles
ROLES=(
    "roles/compute.instanceAdmin.v1"   # Start/stop VMs
    "roles/iam.serviceAccountUser"     # Use VM's service account
    "roles/compute.osLogin"            # SSH access
    "roles/run.admin"                  # Cloud Run Jobs
    "roles/artifactregistry.admin"     # Push container images
    "roles/logging.viewer"             # View Cloud Run logs
)

for ROLE in "${ROLES[@]}"; do
    gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="${ROLE}" \
        --quiet \
        --condition=None 2>/dev/null || true
done
log_info "IAM permissions configured"

# =============================================================================
# Step 3.5: Create Cloud Run resources (idempotent)
# =============================================================================
log_info "Setting up Cloud Run resources..."

# Create Artifact Registry remote repository for ghcr.io (caches container images)
if gcloud artifacts repositories describe "${AR_REMOTE_REPO}" --location="${REGION}" &>/dev/null; then
    log_info "Artifact Registry remote repo already exists: ${AR_REMOTE_REPO}"
else
    log_info "Creating Artifact Registry remote repo: ${AR_REMOTE_REPO}"
    gcloud artifacts repositories create "${AR_REMOTE_REPO}" \
        --repository-format=docker \
        --location="${REGION}" \
        --description="Remote repository caching ghcr.io images" \
        --mode=remote-repository \
        --remote-repo-config-desc="GitHub Container Registry" \
        --remote-docker-repo=https://ghcr.io \
        --quiet
fi

# Create sccache GCS bucket (caches Rust compilation for openvaf-py)
if gcloud storage buckets describe "gs://${SCCACHE_BUCKET}" &>/dev/null; then
    log_info "sccache bucket already exists: ${SCCACHE_BUCKET}"
else
    log_info "Creating sccache bucket: ${SCCACHE_BUCKET}"
    gcloud storage buckets create "gs://${SCCACHE_BUCKET}" \
        --location="${REGION}" \
        --uniform-bucket-level-access \
        --quiet
fi

# Grant service account access to sccache bucket
log_info "Granting service account access to sccache bucket..."
gcloud storage buckets add-iam-policy-binding "gs://${SCCACHE_BUCKET}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectAdmin" \
    --quiet 2>/dev/null || true

# Create traces GCS bucket (stores profiling traces from GPU CI)
if gcloud storage buckets describe "gs://${TRACES_BUCKET}" &>/dev/null; then
    log_info "Traces bucket already exists: ${TRACES_BUCKET}"
else
    log_info "Creating traces bucket: ${TRACES_BUCKET}"
    gcloud storage buckets create "gs://${TRACES_BUCKET}" \
        --location="${REGION}" \
        --uniform-bucket-level-access \
        --quiet
fi

# Grant service account access to traces bucket
log_info "Granting service account access to traces bucket..."
gcloud storage buckets add-iam-policy-binding "gs://${TRACES_BUCKET}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectAdmin" \
    --quiet 2>/dev/null || true

# Grant Cloud Run default service account access to traces bucket
# (Cloud Run jobs use the default Compute Engine service account for workload identity)
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
log_info "Granting Cloud Run default SA access to traces bucket..."
gcloud storage buckets add-iam-policy-binding "gs://${TRACES_BUCKET}" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/storage.objectAdmin" \
    --quiet 2>/dev/null || true

log_info "Cloud Run resources configured"

# =============================================================================
# Step 4: Create/update secret in Secret Manager (idempotent)
# =============================================================================
log_info "Setting up Secret Manager..."

# Check if secret exists
if gcloud secrets describe "${SECRET_NAME}" &>/dev/null; then
    log_info "Secret already exists: ${SECRET_NAME}"

    # Check if we need to create a new key version
    # Get the latest version's create time
    LATEST_VERSION=$(gcloud secrets versions list "${SECRET_NAME}" \
        --filter="state=ENABLED" \
        --sort-by="~createTime" \
        --limit=1 \
        --format="value(name)" 2>/dev/null || echo "")

    if [ -n "${LATEST_VERSION}" ]; then
        log_info "Using existing secret version"
        NEED_NEW_KEY=false
    else
        log_warn "No enabled secret versions found, creating new key"
        NEED_NEW_KEY=true
    fi
else
    log_info "Creating new secret: ${SECRET_NAME}"
    gcloud secrets create "${SECRET_NAME}" \
        --replication-policy="automatic" \
        --quiet
    NEED_NEW_KEY=true
fi

# Create new key and add to secret if needed
if [ "${NEED_NEW_KEY:-true}" = true ]; then
    log_info "Generating new service account key..."

    # Create temporary key file
    KEY_FILE=$(mktemp)
    trap "rm -f ${KEY_FILE}" EXIT

    gcloud iam service-accounts keys create "${KEY_FILE}" \
        --iam-account="${SA_EMAIL}" \
        --quiet

    # Add new version to secret
    gcloud secrets versions add "${SECRET_NAME}" \
        --data-file="${KEY_FILE}" \
        --quiet

    log_info "Service account key stored in Secret Manager"

    # Clean up old keys (keep only the 2 most recent)
    log_info "Cleaning up old service account keys..."
    OLD_KEYS=$(gcloud iam service-accounts keys list \
        --iam-account="${SA_EMAIL}" \
        --format="value(name)" \
        --filter="keyType=USER_MANAGED" \
        --sort-by="~validAfterTime" 2>/dev/null | tail -n +3)

    for KEY_ID in ${OLD_KEYS}; do
        gcloud iam service-accounts keys delete "${KEY_ID}" \
            --iam-account="${SA_EMAIL}" \
            --quiet 2>/dev/null || true
    done
fi

# =============================================================================
# Step 5: Sync secret to GitHub (idempotent)
# =============================================================================
log_info "Syncing secret to GitHub..."

if command -v gh &>/dev/null; then
    # Check if gh is authenticated
    if gh auth status &>/dev/null; then
        # Get the secret from Secret Manager
        SECRET_VALUE=$(gcloud secrets versions access latest --secret="${SECRET_NAME}" 2>/dev/null)

        if [ -n "${SECRET_VALUE}" ]; then
            # Set GitHub secret (idempotent - overwrites if exists)
            echo "${SECRET_VALUE}" | gh secret set GCP_SERVICE_ACCOUNT_KEY \
                --repo="${GITHUB_REPO}" 2>/dev/null && \
                log_info "GitHub secret 'GCP_SERVICE_ACCOUNT_KEY' updated" || \
                log_warn "Failed to update GitHub secret (check gh permissions)"
        else
            log_error "Could not retrieve secret from Secret Manager"
        fi
    else
        log_warn "gh CLI not authenticated. Run 'gh auth login' to sync secrets"
    fi
else
    log_warn "gh CLI not installed. Install it to auto-sync GitHub secrets"
    echo ""
    echo "To manually set the GitHub secret:"
    echo "  1. Get the secret: gcloud secrets versions access latest --secret=${SECRET_NAME}"
    echo "  2. Add to GitHub: Settings -> Secrets -> Actions -> New repository secret"
    echo "     Name: GCP_SERVICE_ACCOUNT_KEY"
fi

# =============================================================================
# Step 6: Create GPU VM (idempotent) - OPTIONAL
# =============================================================================
if [ "${SETUP_VM}" = true ]; then
log_info "Setting up GPU VM..."

if gcloud compute instances describe "${VM_NAME}" --zone="${ZONE}" &>/dev/null; then
    log_info "VM already exists: ${VM_NAME}"

    # Check if VM is running
    VM_STATUS=$(gcloud compute instances describe "${VM_NAME}" \
        --zone="${ZONE}" \
        --format="value(status)")

    if [ "${VM_STATUS}" = "RUNNING" ]; then
        log_info "VM is currently running"
    else
        log_info "VM is ${VM_STATUS}"
    fi
else
    log_info "Creating GPU VM: ${VM_NAME}"
    gcloud compute instances create "${VM_NAME}" \
        --zone="${ZONE}" \
        --machine-type="${MACHINE_TYPE}" \
        --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
        --image-family="${IMAGE_FAMILY}" \
        --image-project="${IMAGE_PROJECT}" \
        --boot-disk-size="${BOOT_DISK_SIZE}" \
        --boot-disk-type="pd-ssd" \
        --maintenance-policy="TERMINATE" \
        --no-restart-on-failure \
        --metadata="install-nvidia-driver=True" \
        --scopes="cloud-platform" \
        --quiet

    log_info "Waiting for VM to initialize (60s)..."
    sleep 60
fi

# =============================================================================
# Step 7: Setup Python environment on VM (idempotent)
# =============================================================================
log_info "Configuring VM environment..."

# Check if VM is running, start if not
VM_STATUS=$(gcloud compute instances describe "${VM_NAME}" \
    --zone="${ZONE}" \
    --format="value(status)")

if [ "${VM_STATUS}" != "RUNNING" ]; then
    log_info "Starting VM..."
    gcloud compute instances start "${VM_NAME}" --zone="${ZONE}" --quiet
    sleep 30
fi

# Wait for SSH to be ready
log_info "Waiting for SSH..."
for i in {1..30}; do
    if gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --command="true" 2>/dev/null; then
        break
    fi
    if [ $i -eq 30 ]; then
        log_error "SSH not available after 5 minutes"
        exit 1
    fi
    sleep 10
done

# Setup environment (idempotent commands)
gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --quiet --command="
    set -e

    # Install Python 3.10 if not present
    if ! command -v python3.10 &>/dev/null; then
        echo 'Installing Python 3.10...'
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3.10 python3.10-venv python3.10-dev
    fi

    # Create working directory
    mkdir -p ~/jax-spice-ci

    # Verify GPU is accessible
    echo ''
    echo '=== GPU Status ==='
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

    echo ''
    echo 'VM environment ready!'
"

# =============================================================================
# Step 8: Stop VM to save costs
# =============================================================================
log_info "Stopping VM to save costs..."
gcloud compute instances stop "${VM_NAME}" --zone="${ZONE}" --quiet

else
    log_info "Skipping GPU VM setup (use --with-vm to enable)"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Resources created/verified:"
echo "  - Service Account: ${SA_EMAIL}"
echo "  - Secret (GCP):    ${SECRET_NAME}"
echo "  - GitHub Secret:   GCP_SERVICE_ACCOUNT_KEY"
echo "  - Artifact Repo:   ${AR_REMOTE_REPO} (${REGION})"
echo "  - sccache Bucket:  ${SCCACHE_BUCKET}"
echo "  - Traces Bucket:   ${TRACES_BUCKET}"
if [ "${SETUP_VM}" = true ]; then
echo "  - GPU VM:          ${VM_NAME} (${ZONE})"
fi
echo ""
echo "To trigger GPU tests:"
echo "  1. Push to main or create a PR modifying GPU-related code"
echo "  2. Or manually: Actions -> GPU Tests -> Run workflow"
echo ""
echo "To access the secret:"
echo "  gcloud secrets versions access latest --secret=${SECRET_NAME}"
if [ "${SETUP_VM}" = true ]; then
echo ""
echo "To SSH into the VM:"
echo "  gcloud compute instances start ${VM_NAME} --zone=${ZONE}"
echo "  gcloud compute ssh ${VM_NAME} --zone=${ZONE}"
fi
