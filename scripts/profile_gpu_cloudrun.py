#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Run JAX-SPICE vs VACASK benchmark comparison on Cloud Run GPU with profiling.

This script:
1. Triggers a Cloud Run job that runs scripts/compare_vacask.py with --profile
2. Streams logs and waits for completion
3. Downloads profiling traces from Cloud Storage
4. Optionally opens Perfetto UI

Usage:
    # Run benchmarks with profiling
    uv run scripts/profile_gpu_cloudrun.py

    # Specific benchmarks
    uv run scripts/profile_gpu_cloudrun.py --benchmark ring,c6288

    # Open Perfetto after download
    uv run scripts/profile_gpu_cloudrun.py --open-perfetto

Prerequisites:
    - gcloud CLI authenticated with access to jax-spice-cuda-test project
    - gsutil for downloading traces
"""

import argparse
import base64
import subprocess
import sys
import tempfile
import time
from pathlib import Path


GCP_PROJECT = "jax-spice-cuda-test"
GCP_REGION = "us-central1"
JOB_NAME = "jax-spice-gpu-benchmark"
GCS_BUCKET = f"gs://{GCP_PROJECT}-traces"
GCS_BUCKET_NAME = f"{GCP_PROJECT}-traces"


def run_cmd(
    cmd: list[str], check: bool = True, capture: bool = False
) -> subprocess.CompletedProcess:
    """Run a command and optionally capture output."""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=capture, text=True)
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    if capture:
        stdout = result.stdout.strip() if result.stdout else None
        stderr = result.stderr.strip() if result.stderr else None
        return (result, stdout, stderr)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run JAX-SPICE vs VACASK benchmark on Cloud Run GPU with profiling"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="rc,graetz,ring,c6288",
        help="Comma-separated benchmarks to run (default: rc,graetz,ring,c6288)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum timesteps per benchmark (default: 200)",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Disable profiling (just run benchmarks)",
    )
    parser.add_argument(
        "--open-perfetto",
        action="store_true",
        help="Open Perfetto UI after downloading traces",
    )
    args = parser.parse_args()

    timestamp = int(time.time())
    trace_gcs_path = f"{GCS_BUCKET}/{args.benchmark.replace(',', '-')}-{timestamp}"
    local_trace_dir = Path(tempfile.gettempdir()) / f"jax-trace-{args.benchmark.replace(',', '-')}-{timestamp}"
    enable_profiling = not args.no_profile

    print("=" * 60)
    print("JAX-SPICE vs VACASK Benchmark on Cloud Run GPU")
    print("=" * 60)
    print(f"Benchmarks: {args.benchmark}")
    print(f"Max steps: {args.max_steps}")
    print(f"Profiling: {'ENABLED' if enable_profiling else 'DISABLED'}")
    if enable_profiling:
        print(f"GCS path: {trace_gcs_path}")
        print(f"Local dir: {local_trace_dir}")
    print()

    # Check gcloud auth
    print("[1/5] Checking gcloud authentication...")
    (result, stdout, stderr) = run_cmd(
        ["gcloud", "auth", "print-identity-token"], capture=True, check=False
    )
    if result.returncode != 0:
        print("Error: Not authenticated with gcloud. Run: gcloud auth login")
        sys.exit(1)
    print("  Authenticated")
    print()

    # Create GCS bucket if needed (for profiling)
    if enable_profiling:
        print("[2/5] Ensuring GCS bucket exists...")
        run_cmd(["gsutil", "ls", GCS_BUCKET], check=False)
        run_cmd(
            ["gsutil", "mb", "-l", GCP_REGION, "-p", GCP_PROJECT, GCS_BUCKET], check=False
        )
        print()
    else:
        print("[2/5] Skipping GCS bucket (profiling disabled)")
        print()

    # Build the compare_vacask.py command
    compare_cmd = [
        "uv", "run", "python", "scripts/compare_vacask.py",
        "--benchmark", args.benchmark,
        "--max-steps", str(args.max_steps),
        "--use-scan",
    ]
    if enable_profiling:
        compare_cmd.extend(["--profile", "--profile-dir", "/tmp/jax-trace"])
    compare_cmd_str = " ".join(compare_cmd)

    # Build the trace upload script (only if profiling enabled)
    if enable_profiling:
        upload_script = f'''
# Get access token from metadata server (workload identity)
echo "Fetching access token from metadata server..."
TOKEN=$(curl -s -H "Metadata-Flavor: Google" \\
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | \\
  python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

if [ -z "$TOKEN" ]; then
  echo "WARNING: Failed to get access token, skipping trace upload"
else
  echo "Got access token (length: ${{#TOKEN}})"

  # Upload traces to GCS (recursively - JAX creates subdirectories)
  echo "=== Uploading traces to GCS ==="
  find /tmp/jax-trace -type f | while read -r f; do
    relpath="${{f#/tmp/jax-trace/}}"
    echo "Uploading $relpath..."
    curl -s -X PUT -H "Authorization: Bearer $TOKEN" \\
      -H "Content-Type: application/octet-stream" \\
      --data-binary @"$f" \\
      "https://storage.googleapis.com/upload/storage/v1/b/{GCS_BUCKET_NAME}/o?uploadType=media&name={args.benchmark.replace(',', '-')}-{timestamp}/$relpath"
  done
  echo "Traces uploaded to: {trace_gcs_path}"
fi
'''
    else:
        upload_script = ""

    # Create the bash script that runs on Cloud Run
    benchmark_script = f'''#!/bin/bash
set -e

cd /app

# Clone repo at main
git clone --depth 1 --recurse-submodules https://github.com/ChipFlow/jax-spice.git source
cd source

# Install deps (with CUDA support)
uv sync --locked --extra cuda12

# Check GPU detection
echo "=== Checking JAX GPU Detection ==="
uv run python -c "import jax; print('Backend:', jax.default_backend()); print('Devices:', jax.devices())"

echo ""
echo "=== Starting Benchmark Comparison ==="
echo "Benchmarks: {args.benchmark}"
echo "Max steps: {args.max_steps}"
echo "Profiling: {'ENABLED' if enable_profiling else 'DISABLED'}"
echo ""

# Run the benchmark comparison
{compare_cmd_str}

{upload_script}

echo ""
echo "=== Benchmark Complete ==="
'''

    # Create or update the Cloud Run job
    print("[3/5] Creating/updating Cloud Run job...")

    # Use base64-encoded script to avoid shell escaping issues
    script_b64 = base64.b64encode(benchmark_script.encode()).decode()
    job_args = f"echo {script_b64} | base64 -d | bash"

    job_cmd = [
        "gcloud", "run", "jobs", "create", JOB_NAME,
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

    result = run_cmd(job_cmd, check=False)
    if result.returncode != 0:
        job_cmd[3] = "update"
        run_cmd(job_cmd)
    print()

    # Execute the job
    print("[4/5] Executing Cloud Run job...")
    (result, exec_id, _) = run_cmd(
        [
            "gcloud", "run", "jobs", "execute", JOB_NAME,
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
            "gcloud", "beta", "run", "jobs", "executions", "logs", "tail",
            exec_id,
            f"--region={GCP_REGION}",
            f"--project={GCP_PROJECT}",
        ],
        stdout=None,
        stderr=None,
    )

    # Poll job status until completion
    print("Waiting for job to complete (streaming logs)...")
    print()
    job_succeeded = False
    try:
        while True:
            time.sleep(5)
            result = subprocess.run(
                [
                    "gcloud", "run", "jobs", "executions", "describe", exec_id,
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

            parts = output.split(";")
            if len(parts) >= 2:
                condition_type, status = parts[0], parts[1]
                if condition_type == "Completed":
                    job_succeeded = status == "True"
                    break
    finally:
        log_proc.terminate()
        try:
            log_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log_proc.kill()

    print()
    if not job_succeeded:
        print("Job failed! Check logs with:")
        print(f"  gcloud beta run jobs executions logs read {exec_id} --region={GCP_REGION}")
        sys.exit(1)

    # Download traces if profiling was enabled
    if enable_profiling:
        print("[5/5] Downloading traces from GCS...")
        local_trace_dir.mkdir(parents=True, exist_ok=True)
        dl_result = run_cmd(
            ["gsutil", "-m", "cp", "-r", f"{trace_gcs_path}/*", str(local_trace_dir)],
            check=False
        )
        print()

        if dl_result.returncode == 0:
            # List downloaded files
            trace_files = list(local_trace_dir.rglob("*"))
            if trace_files:
                print("Downloaded trace files:")
                for f in sorted(trace_files)[:10]:
                    if f.is_file():
                        size_kb = f.stat().st_size / 1024
                        print(f"  {f.name} ({size_kb:.1f} KB)")
                if len(trace_files) > 10:
                    print(f"  ... and {len(trace_files) - 10} more")
                print()
        else:
            print("Warning: Failed to download traces (they may not have been uploaded)")
            print()
    else:
        print("[5/5] Skipping trace download (profiling disabled)")
        print()

    print("=" * 60)
    print("Benchmark complete!")
    print("=" * 60)
    print()

    if enable_profiling and local_trace_dir.exists():
        print(f"Traces downloaded to: {local_trace_dir}")
        print()
        print("To view in Perfetto:")
        print("  1. Open https://ui.perfetto.dev/")
        print(f"  2. Load trace files from: {local_trace_dir}")
        print()

        if args.open_perfetto:
            import webbrowser
            webbrowser.open("https://ui.perfetto.dev/")

    print("To view full logs:")
    print(f"  gcloud beta run jobs executions logs read {exec_id} --region={GCP_REGION}")


if __name__ == "__main__":
    main()
