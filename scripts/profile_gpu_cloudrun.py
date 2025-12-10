#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Run GPU profiling on Cloud Run and download Perfetto traces locally.

This script:
1. Triggers a Cloud Run job with JAX profiler enabled
2. Waits for completion
3. Downloads the trace files from Cloud Storage
4. Opens Perfetto UI with the traces

Usage:
    uv run scripts/profile_gpu_cloudrun.py [--circuit inv_test|nor_test|and_test|c6288_test]

Prerequisites:
    - gcloud CLI authenticated with access to jax-spice-cuda-test project
    - gsutil for downloading traces
"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path


GCP_PROJECT = "jax-spice-cuda-test"
GCP_REGION = "us-central1"
JOB_NAME = "jax-spice-gpu-profile"
GCS_BUCKET = f"gs://{GCP_PROJECT}-traces"
GCS_BUCKET_NAME = f"{GCP_PROJECT}-traces"  # Without gs:// prefix for REST API


def run_cmd(
    cmd: list[str], check: bool = True, capture: bool = False
) -> subprocess.CompletedProcess:
    """Run a command and optionally capture output."""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
    )
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    if capture:
        stdout = None
        stderr = None
        if result.stdout:
            stdout =  result.stdout.strip()
        if result.stderr:
            stderr = result.stderr.strip()
        return (result, stdout, stderr)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run GPU profiling on Cloud Run with Perfetto"
    )
    parser.add_argument(
        "--circuit",
        default="and_test",
        choices=["inv_test", "nor_test", "and_test", "c6288_test"],
        help="Circuit to profile (default: and_test)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50,
        help="Number of timesteps to simulate (default: 50)",
    )
    parser.add_argument(
        "--open-perfetto",
        action="store_true",
        help="Open Perfetto UI after downloading traces",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip warmup run (only for debugging)",
    )
    args = parser.parse_args()

    timestamp = int(time.time())
    trace_gcs_path = f"{GCS_BUCKET}/{args.circuit}-{timestamp}"
    local_trace_dir = Path(tempfile.gettempdir()) / f"jax-trace-{args.circuit}-{timestamp}"

    print("=" * 60)
    print("JAX-SPICE GPU Profiling on Cloud Run")
    print("=" * 60)
    print(f"Circuit: {args.circuit}")
    print(f"Timesteps: {args.timesteps}")
    print(f"GCS path: {trace_gcs_path}")
    print(f"Local dir: {local_trace_dir}")
    print()

    # Check gcloud auth
    print("[1/5] Checking gcloud authentication...")
    (result, stdout, stderr)  = run_cmd(
        ["gcloud", "auth", "print-identity-token"], capture=True, check=False
    )
    print(stdout)
    print(stdout, file=sys.stderr)
    if result.returncode != 0:
        print("Error: Not authenticated with gcloud. Run: gcloud auth login")
        sys.exit(1)
    print("  Authenticated")
    print()

    # Create GCS bucket if needed
    print("[2/5] Ensuring GCS bucket exists...")
    run_cmd(["gsutil", "ls", GCS_BUCKET], check=False)
    run_cmd(
        ["gsutil", "mb", "-l", GCP_REGION, "-p", GCP_PROJECT, GCS_BUCKET], check=False
    )
    print()

    # Generate the Python profiling script
    # This script will be written to a file and executed
    warmup_code = "" if args.skip_warmup else """
# Warmup (JIT compile)
print("Warmup run...")
_, _, _ = transient_analysis_gpu(
    system,
    t_stop=1e-12,
    t_step=1e-12,
    vdd=1.2,
    icmode="uic",
    verbose=False,
)
print("Warmup complete")
"""

    profile_python = f'''import os
import sys
sys.stdout.reconfigure(line_buffering=True)  # Force line buffering
sys.path.insert(0, ".")

print("=== Python profiling script started ===", flush=True)

print("Importing JAX...", flush=True)
import jax
jax.config.update("jax_enable_x64", True)

print(f"JAX backend: {{jax.default_backend()}}", flush=True)
print(f"JAX devices: {{jax.devices()}}", flush=True)

print("Importing jax_spice modules...", flush=True)
from jax_spice.benchmarks.c6288 import C6288Benchmark
from jax_spice.analysis.transient_gpu import transient_analysis_gpu

# Setup circuit
print("Setting up circuit...", flush=True)
bench = C6288Benchmark(verbose=False)
print("  Parsing...", flush=True)
bench.parse()
print("  Flattening...", flush=True)
bench.flatten("{args.circuit}")
print("  Building system...", flush=True)
bench.build_system("{args.circuit}")

system = bench.system
print("  Building device groups...", flush=True)
system.build_device_groups()

print(f"Circuit: {{system.num_nodes}} nodes, {{len(system.devices)}} devices", flush=True)
{warmup_code}
# Profiled run with trace
print("Starting traced run...", flush=True)
trace_dir = "/tmp/jax-trace"
os.makedirs(trace_dir, exist_ok=True)

print(f"Trace directory: {{trace_dir}}", flush=True)
print("Entering jax.profiler.trace context...", flush=True)

with jax.profiler.trace(trace_dir, create_perfetto_link=False):
    print("Running transient_analysis_gpu...", flush=True)
    times, solutions, info = transient_analysis_gpu(
        system,
        t_stop={args.timesteps}e-12,
        t_step=1e-12,
        vdd=1.2,
        icmode="uic",
        verbose=True,
    )

print(f"Completed: {{len(times)}} timesteps, {{info['total_iterations']}} iterations", flush=True)
print(f"Trace saved to: {{trace_dir}}", flush=True)
'''

    # Upload the Python script to GCS
    import base64
    python_script_gcs = f"{GCS_BUCKET}/scripts/profile_run_{timestamp}.py"

    # Write Python script to temp file and upload
    python_script_path = Path(tempfile.gettempdir()) / f"profile_run_{timestamp}.py"
    python_script_path.write_text(profile_python)

    print(f"[3/6] Uploading profiling script to GCS...")
    run_cmd(["gsutil", "cp", str(python_script_path), python_script_gcs])
    print(f"  Uploaded to: {python_script_gcs}")
    print()

    # Create the bash script that runs the profiling
    profile_script = f'''#!/bin/bash
set -e

cd /app

# Clone repo at main
git clone --depth 1 --recurse-submodules https://github.com/ChipFlow/jax-spice.git source
cd source

# Install deps
uv sync --locked --extra cuda12

echo "=== Starting GPU Profiling ==="
echo "Circuit: {args.circuit}"
echo "Trace output: {trace_gcs_path}"

# Get access token from metadata server (workload identity)
echo "Fetching access token from metadata server..."
TOKEN=$(curl -s -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

if [ -z "$TOKEN" ]; then
  echo "ERROR: Failed to get access token"
  exit 1
fi
echo "Got access token (length: ${{#TOKEN}})"

# Download the profiling script from GCS
echo "Downloading profiling script from GCS..."
curl -v -H "Authorization: Bearer $TOKEN" \
  "https://storage.googleapis.com/{GCS_BUCKET_NAME}/scripts/profile_run_{timestamp}.py" \
  -o /tmp/profile_run.py 2>&1

echo "Script downloaded, size: $(wc -c < /tmp/profile_run.py) bytes"
echo "First 5 lines:"
head -5 /tmp/profile_run.py

echo "Running profiling script..."
uv run python /tmp/profile_run.py

# Upload traces to GCS
echo "=== Uploading traces to GCS ==="
for f in /tmp/jax-trace/*; do
  fname=$(basename "$f")
  curl -s -X PUT -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/octet-stream" \
    --data-binary @"$f" \
    "https://storage.googleapis.com/upload/storage/v1/b/{GCS_BUCKET_NAME}/o?uploadType=media&name={args.circuit}-{timestamp}/$fname"
done

echo "=== Profiling Complete ==="
echo "Traces uploaded to: {trace_gcs_path}"
'''

    # Write bash script to temp file for debugging
    script_path = Path(tempfile.gettempdir()) / "profile_script.sh"
    script_path.write_text(profile_script)
    print(f"  Bash script saved to: {script_path}")

    # Create or update the Cloud Run job
    print("[4/6] Creating/updating Cloud Run job...")

    # Use a base64-encoded script to avoid shell escaping issues
    script_b64 = base64.b64encode(profile_script.encode()).decode()

    # The command decodes and runs the script
    job_args = f"echo {script_b64} | base64 -d | bash"

    job_cmd = [
        "gcloud",
        "run",
        "jobs",
        "create",
        JOB_NAME,
        f"--region={GCP_REGION}",
        f"--project={GCP_PROJECT}",
        "--image=us-central1-docker.pkg.dev/jax-spice-cuda-test/ghcr-remote/chipflow/jax-spice/gpu-base:latest",
        "--execution-environment=gen2",
        "--gpu=1",
        "--gpu-type=nvidia-l4",
        "--no-gpu-zonal-redundancy",
        "--cpu=4",
        "--memory=16Gi",
        "--task-timeout=30m",
        "--max-retries=0",
        "--command=bash",
        f"--args=-c,{job_args}",
    ]

    # Try to create, or update if exists
    result = run_cmd(job_cmd, check=False)
    if result.returncode != 0:
        # Job exists, update it
        job_cmd[3] = "update"
        run_cmd(job_cmd)
    print()

    # Execute the job
    print("[5/6] Executing Cloud Run job...")
    (result, exec_id, _)  = run_cmd(
        [
            "gcloud",
            "run",
            "jobs",
            "execute",
            JOB_NAME,
            f"--region={GCP_REGION}",
            f"--project={GCP_PROJECT}",
            "--async",
            "--format=value(metadata.name)",
        ],
        capture=True,
    )

    print(f"Job Execution ID: {exec_id}")

    # Start log tailing in background
    log_proc = subprocess.Popen(
        [
            "gcloud",
            "beta",
            "run",
            "jobs",
            "executions",
            "logs",
            "tail",
            exec_id,
            f"--region={GCP_REGION}",
            f"--project={GCP_PROJECT}",
        ],
        stdout=None,  # Inherit stdout
        stderr=None,  # Inherit stderr
    )

    # Poll job status until completion
    print("Waiting for job to complete...")
    job_succeeded = False
    try:
        while True:
            time.sleep(5)
            result = subprocess.run(
                [
                    "gcloud",
                    "run",
                    "jobs",
                    "executions",
                    "describe",
                    exec_id,
                    f"--region={GCP_REGION}",
                    f"--project={GCP_PROJECT}",
                    "--format=value(status.conditions[0].type,status.conditions[0].status)",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                continue

            output = result.stdout.strip()
            if not output:
                continue

            # Parse "Completed;True" or "Completed;False" format
            parts = output.split(";")
            if len(parts) >= 2:
                condition_type, status = parts[0], parts[1]
                if condition_type == "Completed":
                    job_succeeded = status == "True"
                    break
    finally:
        # Stop log tailing
        log_proc.terminate()
        try:
            log_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log_proc.kill()

    print()
    if not job_succeeded:
        print("Job failed! Check logs with:")
        print(
            f"  gcloud beta run jobs executions logs read {exec_id} --region={GCP_REGION}"
        )
        sys.exit(1)
    print()

    # Download traces
    print("[6/6] Downloading traces...")
    local_trace_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "gsutil",
            "-m",
            "cp",
            "-r",
            f"{trace_gcs_path}/*",
            str(local_trace_dir),
        ]
    )
    print()

    print("=" * 60)
    print("Profiling complete!")
    print("=" * 60)
    print(f"Traces downloaded to: {local_trace_dir}")
    print()
    print("To view in Perfetto:")
    print("  1. Open https://ui.perfetto.dev/")
    print(f"  2. Load trace file from: {local_trace_dir}")
    print()

    # List trace files
    trace_files = list(local_trace_dir.rglob("*"))
    if trace_files:
        print("Trace files:")
        for f in sorted(trace_files)[:10]:
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                print(f"  {f.name} ({size_kb:.1f} KB)")

    if args.open_perfetto:
        import webbrowser

        webbrowser.open("https://ui.perfetto.dev/")


if __name__ == "__main__":
    main()
