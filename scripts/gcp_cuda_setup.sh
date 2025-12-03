#!/bin/bash
# GCP CUDA VM Setup Script for JAX-SPICE Testing
#
# This script creates a GCP VM with NVIDIA GPU, installs CUDA, and runs JAX-SPICE benchmarks.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Sufficient GPU quota in your GCP project
#
# Usage:
#   ./scripts/gcp_cuda_setup.sh [create|ssh|run|delete|status]

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${GCP_INSTANCE_NAME:-jax-spice-cuda}"
MACHINE_TYPE="${GCP_MACHINE_TYPE:-n1-standard-4}"
GPU_TYPE="${GCP_GPU_TYPE:-nvidia-tesla-t4}"
GPU_COUNT="${GCP_GPU_COUNT:-1}"
BOOT_DISK_SIZE="100GB"

# Use Ubuntu with NVIDIA driver pre-installed
IMAGE_FAMILY="pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
IMAGE_PROJECT="deeplearning-platform-release"

print_usage() {
    echo "Usage: $0 [create|ssh|run|delete|status|logs]"
    echo ""
    echo "Commands:"
    echo "  create  - Create the GCP VM with GPU"
    echo "  ssh     - SSH into the VM"
    echo "  run     - Run the JAX-SPICE benchmark on the VM"
    echo "  delete  - Delete the VM"
    echo "  status  - Check VM status"
    echo "  logs    - View startup script logs"
    echo ""
    echo "Environment variables:"
    echo "  GCP_PROJECT_ID    - GCP project ID (default: current gcloud project)"
    echo "  GCP_ZONE          - GCP zone (default: us-central1-a)"
    echo "  GCP_INSTANCE_NAME - VM instance name (default: jax-spice-cuda)"
    echo "  GCP_MACHINE_TYPE  - Machine type (default: n1-standard-4)"
    echo "  GCP_GPU_TYPE      - GPU type (default: nvidia-tesla-t4)"
    echo "  GCP_GPU_COUNT     - Number of GPUs (default: 1)"
}

check_prerequisites() {
    if ! command -v gcloud &> /dev/null; then
        echo "Error: gcloud CLI not found. Please install Google Cloud SDK."
        exit 1
    fi

    if [ -z "$PROJECT_ID" ]; then
        echo "Error: No GCP project configured. Set GCP_PROJECT_ID or run 'gcloud config set project PROJECT_ID'"
        exit 1
    fi

    echo "Using project: $PROJECT_ID"
    echo "Using zone: $ZONE"
}

create_vm() {
    echo "Creating VM: $INSTANCE_NAME"
    echo "  Machine type: $MACHINE_TYPE"
    echo "  GPU: $GPU_COUNT x $GPU_TYPE"
    echo ""

    # Check if instance already exists
    if gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" &>/dev/null; then
        echo "Instance $INSTANCE_NAME already exists. Use 'delete' first or 'ssh' to connect."
        exit 1
    fi

    # Create the instance
    gcloud compute instances create "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --boot-disk-size="$BOOT_DISK_SIZE" \
        --boot-disk-type="pd-ssd" \
        --maintenance-policy="TERMINATE" \
        --metadata="install-nvidia-driver=True" \
        --scopes="https://www.googleapis.com/auth/cloud-platform"

    echo ""
    echo "VM created successfully!"
    echo "Waiting for VM to be ready (this may take a few minutes for driver installation)..."

    # Wait for the instance to be ready
    sleep 30

    # Wait for SSH to be available
    for i in {1..20}; do
        if gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" --command="echo 'SSH ready'" &>/dev/null; then
            echo "VM is ready!"
            break
        fi
        echo "Waiting for SSH... (attempt $i/20)"
        sleep 15
    done
}

ssh_to_vm() {
    echo "Connecting to $INSTANCE_NAME..."
    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID"
}

run_benchmark() {
    echo "Running JAX-SPICE benchmark on $INSTANCE_NAME..."

    # Create a temporary script to run on the VM
    cat > /tmp/run_jax_spice_benchmark.sh << 'REMOTE_SCRIPT'
#!/bin/bash
set -e

echo "=== JAX-SPICE CUDA Benchmark ==="
echo ""

# Check NVIDIA driver
echo "Checking NVIDIA driver..."
nvidia-smi

echo ""
echo "=== Setting up environment ==="

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Create working directory
WORKDIR="$HOME/jax-spice-test"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Clone the repository (now public)
echo "Cloning jax-spice repository..."
git clone --depth 1 https://github.com/ChipFlow/jax-spice.git .

# Create virtual environment and install dependencies
echo "Creating virtual environment..."
uv venv .venv --python 3.10
source .venv/bin/activate

# Install JAX with CUDA support
echo "Installing JAX with CUDA support..."
uv pip install "jax[cuda12]"

# Install other dependencies (skip jax-metal since we're on CUDA)
echo "Installing jax-spice dependencies..."
uv pip install numpy scipy matplotlib pytest lark

# Check JAX can see the GPU
echo ""
echo "=== Checking JAX GPU access ==="
python3 -c "
import jax
print(f'JAX version: {jax.__version__}')
print(f'Available devices: {jax.devices()}')
print(f'Default backend: {jax.default_backend()}')
"

# Run the benchmark
echo ""
echo "=== Running Transient Analysis Benchmark ==="
python3 << 'PYTHON_BENCHMARK'
import os
import time
import jax
import jax.numpy as jnp
from jax import Array

# Enable float64 for numerical precision
jax.config.update('jax_enable_x64', True)

print(f"JAX Backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")
print()

# Import after setting up JAX
import sys
sys.path.insert(0, '.')

from jax_spice.devices.base import DeviceStamps
from jax_spice.analysis.context import AnalysisContext
from jax_spice.analysis.mna import MNASystem, DeviceInfo
# Import transient module directly to avoid openvaf_py dependency
from jax_spice.analysis import transient as transient_module
transient_analysis = transient_module.transient_analysis

def resistor_eval(voltages, params, context):
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)
    R = float(params.get('r', 1000.0))
    G = 1.0 / R
    I = G * (Vp - Vn)
    return DeviceStamps(
        currents={'p': jnp.array(I), 'n': jnp.array(-I)},
        conductances={
            ('p', 'p'): jnp.array(G), ('p', 'n'): jnp.array(-G),
            ('n', 'p'): jnp.array(-G), ('n', 'n'): jnp.array(G)
        }
    )

def capacitor_eval(voltages, params, context):
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)
    C = float(params.get('c', 1e-6))
    V = Vp - Vn
    Q = C * V
    G_small = 1e-15
    return DeviceStamps(
        currents={'p': jnp.array(0.0), 'n': jnp.array(0.0)},
        conductances={
            ('p', 'p'): jnp.array(G_small), ('p', 'n'): jnp.array(-G_small),
            ('n', 'p'): jnp.array(-G_small), ('n', 'n'): jnp.array(G_small)
        },
        charges={'p': jnp.array(Q), 'n': jnp.array(-Q)},
        capacitances={
            ('p', 'p'): jnp.array(C), ('p', 'n'): jnp.array(-C),
            ('n', 'p'): jnp.array(-C), ('n', 'n'): jnp.array(C)
        }
    )

def vsource_eval(voltages, params, context):
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)
    V_target = float(params.get('v', 5.0))
    V_actual = Vp - Vn
    G_big = 1e12
    I = G_big * (V_actual - V_target)
    return DeviceStamps(
        currents={'p': jnp.array(I), 'n': jnp.array(-I)},
        conductances={
            ('p', 'p'): jnp.array(G_big), ('p', 'n'): jnp.array(-G_big),
            ('n', 'p'): jnp.array(-G_big), ('n', 'n'): jnp.array(G_big)
        }
    )

def create_rc_ladder(num_stages, R=1000.0, C=1e-6, V_s=5.0):
    num_nodes = num_stages + 2
    node_names = {'0': 0, 'vs': 1}
    for i in range(num_stages):
        node_names[f'n{i+1}'] = i + 2

    system = MNASystem(num_nodes=num_nodes, node_names=node_names)

    system.devices.append(DeviceInfo(
        name='Vs', model_name='vsource', terminals=['p', 'n'],
        node_indices=[1, 0], params={'v': V_s}, eval_fn=vsource_eval
    ))

    for i in range(num_stages):
        prev_node = 1 if i == 0 else i + 1
        curr_node = i + 2

        system.devices.append(DeviceInfo(
            name=f'R{i+1}', model_name='resistor', terminals=['p', 'n'],
            node_indices=[prev_node, curr_node], params={'r': R}, eval_fn=resistor_eval
        ))

        system.devices.append(DeviceInfo(
            name=f'C{i+1}', model_name='capacitor', terminals=['p', 'n'],
            node_indices=[curr_node, 0], params={'c': C}, eval_fn=capacitor_eval
        ))

    return system

def benchmark(num_stages, num_timesteps, use_jit=True):
    R, C = 1000.0, 1e-6
    tau = R * C

    system = create_rc_ladder(num_stages, R, C)
    t_step = tau / 10
    t_stop = t_step * num_timesteps

    initial_conditions = {'vs': 5.0}
    for i in range(num_stages):
        initial_conditions[f'n{i+1}'] = 0.0

    start = time.perf_counter()
    times, solutions, info = transient_analysis(
        system, t_stop=t_stop, t_step=t_step,
        initial_conditions=initial_conditions,
        use_jit=use_jit
    )
    solutions.block_until_ready()
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed

# Warm up
print("Warming up JIT compilation...")
_ = benchmark(2, 10)
print()

# Circuit size scaling
print("-" * 70)
print("Circuit Size Scaling (100 timesteps)")
print("-" * 70)
print(f"{'Stages':>8} | {'Nodes':>8} | {'Devices':>8} | {'Sim (ms)':>10}")
print("-" * 70)

for num_stages in [1, 2, 5, 10, 20, 50, 100]:
    sim_time = benchmark(num_stages, 100)
    num_nodes = num_stages + 2
    num_devices = 1 + 2 * num_stages
    print(f"{num_stages:>8} | {num_nodes:>8} | {num_devices:>8} | {sim_time:>10.2f}")

print()

# Simulation length scaling
print("-" * 70)
print("Simulation Length Scaling (10-stage RC ladder)")
print("-" * 70)
print(f"{'Timesteps':>10} | {'Sim (ms)':>10} | {'ms/step':>10}")
print("-" * 70)

for num_timesteps in [10, 50, 100, 500, 1000, 5000]:
    sim_time = benchmark(10, num_timesteps)
    per_step = sim_time / num_timesteps
    print(f"{num_timesteps:>10} | {sim_time:>10.2f} | {per_step:>10.4f}")

print()
print("=" * 70)
print("Benchmark complete!")
PYTHON_BENCHMARK

echo ""
echo "=== Benchmark Complete ==="
REMOTE_SCRIPT

    # Copy and run the script on the VM
    gcloud compute scp /tmp/run_jax_spice_benchmark.sh "$INSTANCE_NAME":~/run_benchmark.sh \
        --zone="$ZONE" --project="$PROJECT_ID"

    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" \
        --command="chmod +x ~/run_benchmark.sh && ~/run_benchmark.sh"
}

delete_vm() {
    echo "Deleting VM: $INSTANCE_NAME"
    gcloud compute instances delete "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT_ID" \
        --quiet
    echo "VM deleted."
}

check_status() {
    echo "Checking status of $INSTANCE_NAME..."
    gcloud compute instances describe "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT_ID" \
        --format="table(name,status,networkInterfaces[0].accessConfigs[0].natIP,machineType.basename())"
}

view_logs() {
    echo "Viewing startup script logs for $INSTANCE_NAME..."
    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" \
        --command="sudo journalctl -u google-startup-scripts.service -f"
}

# Main
check_prerequisites

case "${1:-}" in
    create)
        create_vm
        ;;
    ssh)
        ssh_to_vm
        ;;
    run)
        run_benchmark
        ;;
    delete)
        delete_vm
        ;;
    status)
        check_status
        ;;
    logs)
        view_logs
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
