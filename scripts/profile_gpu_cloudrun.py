#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Run GPU profiling on Cloud Run and download Perfetto traces locally.

This script:
1. Triggers a Cloud Run job that runs scripts/profile_gpu.py with --trace
2. Waits for completion
3. Downloads the trace files from Cloud Storage
4. Opens Perfetto UI with the traces

Usage:
    uv run scripts/profile_gpu_cloudrun.py [--benchmark rc|graetz|mul|ring|c6288]

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
JOB_NAME = "jax-spice-gpu-profile"
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
        description="Run GPU profiling on Cloud Run with Perfetto"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="ring",
        help="Comma-separated benchmarks to profile (default: ring)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50,
        help="Number of timesteps to simulate (default: 50)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of warmup steps (default: 5)",
    )
    parser.add_argument(
        "--sparse-only",
        action="store_true",
        help="Only run sparse solver",
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

    print("=" * 60)
    print("JAX-SPICE GPU Profiling on Cloud Run")
    print("=" * 60)
    print(f"Benchmark: {args.benchmark}")
    print(f"Timesteps: {args.timesteps}")
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

    # Create GCS bucket if needed
    print("[2/5] Ensuring GCS bucket exists...")
    run_cmd(["gsutil", "ls", GCS_BUCKET], check=False)
    run_cmd(
        ["gsutil", "mb", "-l", GCP_REGION, "-p", GCP_PROJECT, GCS_BUCKET], check=False
    )
    print()

    # Build the profile_gpu.py command
    profile_cmd = [
        "uv", "run", "python", "scripts/profile_gpu.py",
        "--benchmark", args.benchmark,
        "--timesteps", str(args.timesteps),
        "--warmup-steps", str(args.warmup_steps),
        "--trace",
        "--trace-dir", "/tmp/jax-trace",
    ]
    if args.sparse_only:
        profile_cmd.append("--sparse-only")

    profile_cmd_str = " ".join(profile_cmd)

    # Create the bash script that runs on Cloud Run
    profile_script = f'''#!/bin/bash
set -e

cd /app

# Clone repo at main
git clone --depth 1 --recurse-submodules https://github.com/ChipFlow/jax-spice.git source
cd source

# Install deps
uv sync --locked --extra cuda12

echo "=== Starting GPU Profiling ==="
echo "Benchmark: {args.benchmark}"
echo "Trace output: {trace_gcs_path}"

# Run the profiling script
{profile_cmd_str}

# Get access token from metadata server (workload identity)
echo "Fetching access token from metadata server..."
TOKEN=$(curl -s -H "Metadata-Flavor: Google" \\
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | \\
  python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

if [ -z "$TOKEN" ]; then
  echo "ERROR: Failed to get access token"
  exit 1
fi
echo "Got access token (length: ${{#TOKEN}})"

# Upload traces to GCS
echo "=== Uploading traces to GCS ==="
for f in /tmp/jax-trace/*; do
  if [ -f "$f" ]; then
    fname=$(basename "$f")
    echo "Uploading $fname..."
    curl -s -X PUT -H "Authorization: Bearer $TOKEN" \\
      -H "Content-Type: application/octet-stream" \\
      --data-binary @"$f" \\
      "https://storage.googleapis.com/upload/storage/v1/b/{GCS_BUCKET_NAME}/o?uploadType=media&name={args.benchmark.replace(',', '-')}-{timestamp}/$fname"
  fi
done

# Also upload the reports
for f in profile_report.md profile_report.json; do
  if [ -f "$f" ]; then
    echo "Uploading $f..."
    curl -s -X PUT -H "Authorization: Bearer $TOKEN" \\
      -H "Content-Type: application/octet-stream" \\
      --data-binary @"$f" \\
      "https://storage.googleapis.com/upload/storage/v1/b/{GCS_BUCKET_NAME}/o?uploadType=media&name={args.benchmark.replace(',', '-')}-{timestamp}/$f"
  fi
done

echo "=== Profiling Complete ==="
echo "Traces uploaded to: {trace_gcs_path}"
'''

    # Write bash script to temp file for debugging
    script_path = Path(tempfile.gettempdir()) / "profile_script.sh"
    script_path.write_text(profile_script)
    print(f"[3/5] Bash script saved to: {script_path}")

    # Create or update the Cloud Run job
    print("[4/5] Creating/updating Cloud Run job...")

    # Use base64-encoded script to avoid shell escaping issues
    script_b64 = base64.b64encode(profile_script.encode()).decode()
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
    print("[5/5] Executing Cloud Run job...")
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
    print("Waiting for job to complete...")
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
    print()

    # Download traces
    print("Downloading traces...")
    local_trace_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(["gsutil", "-m", "cp", "-r", f"{trace_gcs_path}/*", str(local_trace_dir)])
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

    # List downloaded files
    trace_files = list(local_trace_dir.rglob("*"))
    if trace_files:
        print("Downloaded files:")
        for f in sorted(trace_files)[:10]:
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                print(f"  {f.name} ({size_kb:.1f} KB)")

    if args.open_perfetto:
        import webbrowser
        webbrowser.open("https://ui.perfetto.dev/")


if __name__ == "__main__":
    main()
