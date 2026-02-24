#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Run nsys-jax GPU profiling on Cloud Run and download profile archives locally.

This script:
1. Triggers a Cloud Run job with nsys-jax profiler enabled
2. Waits for completion
3. Downloads the profile archive from Cloud Storage
4. Provides instructions for analysis

Usage:
    uv run scripts/profile_nsys_cloudrun.py [--circuit inv_test|nor_test|and_test|c6288_test]

Prerequisites:
    - gcloud CLI authenticated with access to va-jax-cuda-test project
    - gsutil for downloading traces

Output:
    - .zip archive containing:
      - .nsys-rep (Nsight Systems profile)
      - .parquet files (tabular profile data)
      - Analysis.ipynb (Jupyter analysis template)
      - install.sh (environment setup script)
"""

import argparse
import base64
import subprocess
import sys
import tempfile
import time
from pathlib import Path

GCP_PROJECT = "va-jax-cuda-test"
GCP_REGION = "us-central1"
JOB_NAME = "va-jax-nsys-profile"
GCS_BUCKET = f"gs://{GCP_PROJECT}-traces"


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
    if capture and result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run nsys-jax GPU profiling on Cloud Run")
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
        "--skip-warmup",
        action="store_true",
        help="Skip warmup run (includes JIT compilation in profile)",
    )
    args = parser.parse_args()

    timestamp = int(time.time())
    profile_name = f"nsys-{args.circuit}-{timestamp}"
    profile_gcs_path = f"{GCS_BUCKET}/{profile_name}"
    local_profile_dir = Path(tempfile.gettempdir()) / f"nsys-profile-{args.circuit}-{timestamp}"

    print("=" * 60)
    print("VA-JAX nsys-jax GPU Profiling on Cloud Run")
    print("=" * 60)
    print(f"Circuit: {args.circuit}")
    print(f"Timesteps: {args.timesteps}")
    print(f"GCS path: {profile_gcs_path}")
    print(f"Local dir: {local_profile_dir}")
    print()

    # Check gcloud auth
    print("[1/5] Checking gcloud authentication...")
    result = run_cmd(["gcloud", "auth", "print-identity-token"], capture=True, check=False)
    if result.returncode != 0:
        print("Error: Not authenticated with gcloud. Run: gcloud auth login")
        sys.exit(1)
    print("  Authenticated")
    print()

    # Create GCS bucket if needed
    print("[2/5] Ensuring GCS bucket exists...")
    run_cmd(["gsutil", "ls", GCS_BUCKET], check=False)
    run_cmd(["gsutil", "mb", "-l", GCP_REGION, "-p", GCP_PROJECT, GCS_BUCKET], check=False)
    print()

    # Build the bash script that runs nsys-jax profiling
    skip_warmup_flag = "--skip-warmup" if args.skip_warmup else ""

    profile_script = f"""#!/bin/bash
set -e

cd /app

# Clone repo at main
git clone --depth 1 --recurse-submodules https://github.com/ChipFlow/va-jax.git source
cd source

# Install deps
uv sync --locked --extra cuda12

echo "=== Starting nsys-jax GPU Profiling ==="
echo "Circuit: {args.circuit}"
echo "Timesteps: {args.timesteps}"
echo "Output: {profile_gcs_path}"

# Run profiling with nsys-jax
# nsys-jax automatically configures XLA_FLAGS and JAX environment variables
nsys-jax -o /tmp/{profile_name}.zip \\
    uv run python scripts/nsys_profile_target.py {args.circuit} {args.timesteps} {skip_warmup_flag}

# Upload profile archive to GCS
echo "=== Uploading profile to GCS ==="
gsutil cp /tmp/{profile_name}.zip {profile_gcs_path}.zip

echo "=== Profiling Complete ==="
echo "Profile uploaded to: {profile_gcs_path}.zip"
"""

    # Write script to temp file for debugging
    script_path = Path(tempfile.gettempdir()) / "nsys_profile_script.sh"
    script_path.write_text(profile_script)
    print(f"  Script saved to: {script_path}")

    # Create or update the Cloud Run job
    print("[3/5] Creating/updating Cloud Run job...")

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
        "--image=us-central1-docker.pkg.dev/va-jax-cuda-test/ghcr-remote/chipflow/va-jax/gpu-base:latest",
        "--execution-environment=gen2",
        "--gpu=1",
        "--gpu-type=nvidia-l4",
        "--no-gpu-zonal-redundancy",
        "--cpu=4",
        "--memory=16Gi",
        "--task-timeout=30m",
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
    print("[4/5] Executing Cloud Run job...")
    result = run_cmd(
        [
            "gcloud",
            "run",
            "jobs",
            "execute",
            JOB_NAME,
            f"--region={GCP_REGION}",
            f"--project={GCP_PROJECT}",
            "--wait",
        ],
        check=False,
    )

    if result.returncode != 0:
        print("Job failed! Check logs with:")
        print(f"  gcloud run jobs executions list --job={JOB_NAME} --region={GCP_REGION}")
        print()
        print("View logs:")
        print(
            f"  gcloud beta run jobs executions logs read $(gcloud run jobs executions list --job={JOB_NAME} --region={GCP_REGION} --limit=1 --format='value(name)')"
        )
        sys.exit(1)
    print()

    # Download profile archive
    print("[5/5] Downloading profile archive...")
    local_profile_dir.mkdir(parents=True, exist_ok=True)
    local_archive = local_profile_dir / f"{profile_name}.zip"
    run_cmd(
        [
            "gsutil",
            "cp",
            f"{profile_gcs_path}.zip",
            str(local_archive),
        ]
    )
    print()

    print("=" * 60)
    print("Profiling complete!")
    print("=" * 60)
    print(f"Profile archive: {local_archive}")
    print()
    print("To analyze the profile:")
    print()
    print("  1. Extract the archive:")
    print(f"     unzip {local_archive} -d {local_profile_dir}/extracted")
    print()
    print("  2. Set up analysis environment:")
    print(f"     cd {local_profile_dir}/extracted && bash install.sh")
    print()
    print("  3. Open the analysis notebook:")
    print("     jupyter lab Analysis.ipynb")
    print()
    print("  4. Or view in Nsight Systems GUI:")
    print(f"     nsys-ui {local_profile_dir}/extracted/*.nsys-rep")
    print()

    # Show archive size
    if local_archive.exists():
        size_mb = local_archive.stat().st_size / (1024 * 1024)
        print(f"Archive size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
