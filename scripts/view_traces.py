#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
View JAX profiling traces in Perfetto UI.

This script:
1. Lists available trace files in a directory
2. Opens Perfetto UI in the browser
3. Provides instructions for loading traces

Usage:
    # View traces from a directory
    uv run scripts/view_traces.py /tmp/jax-spice-traces

    # View traces from CI artifact (after downloading)
    uv run scripts/view_traces.py ~/Downloads/profiling-traces-abc123

    # Download and view traces from GCS
    uv run scripts/view_traces.py --gcs gs://jax-spice-cuda-test-traces/abc123
"""

import argparse
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path


def list_trace_files(trace_dir: Path) -> list[Path]:
    """List all trace files in a directory."""
    trace_files = []
    for pattern in ["*.pb", "*.pb.gz", "*.json", "*.perfetto-trace"]:
        trace_files.extend(trace_dir.glob(pattern))
        trace_files.extend(trace_dir.glob(f"**/{pattern}"))
    return sorted(set(trace_files))


def download_from_gcs(gcs_path: str) -> Path:
    """Download traces from GCS to a temporary directory."""
    # Extract bucket and path for unique local dir name
    gcs_path = gcs_path.rstrip('/')
    path_parts = gcs_path.replace('gs://', '').split('/')
    dir_name = path_parts[-1] if len(path_parts) > 1 else 'traces'

    local_dir = Path(tempfile.gettempdir()) / f"jax-traces-{dir_name}"
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading traces from {gcs_path}...")

    # Use gsutil rsync for recursive download (handles nested directories)
    result = subprocess.run(
        ["gsutil", "-m", "rsync", "-r", gcs_path, str(local_dir)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Try cp -r as fallback
        result = subprocess.run(
            ["gsutil", "-m", "cp", "-r", f"{gcs_path}/**", str(local_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error downloading traces: {result.stderr}")
            sys.exit(1)

    return local_dir


def main():
    parser = argparse.ArgumentParser(
        description="View JAX profiling traces in Perfetto UI"
    )
    parser.add_argument(
        "trace_dir",
        type=str,
        nargs="?",
        default="/tmp/jax-spice-traces",
        help="Directory containing trace files (default: /tmp/jax-spice-traces)",
    )
    parser.add_argument(
        "--gcs",
        type=str,
        help="Download traces from GCS path (e.g., gs://bucket/path)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser, just list trace files",
    )
    args = parser.parse_args()

    # Handle GCS download
    if args.gcs:
        trace_dir = download_from_gcs(args.gcs)
    else:
        trace_dir = Path(args.trace_dir)

    if not trace_dir.exists():
        print(f"Error: Trace directory does not exist: {trace_dir}")
        print()
        print("To generate traces, run benchmarks with profiling enabled:")
        print("  JAX_SPICE_PROFILE_JAX=1 uv run python scripts/compare_vacask.py --profile")
        print()
        print("Or run on Cloud Run GPU:")
        print("  uv run scripts/profile_gpu_cloudrun.py")
        sys.exit(1)

    # List trace files
    trace_files = list_trace_files(trace_dir)

    print("=" * 60)
    print("JAX-SPICE Profiling Trace Viewer")
    print("=" * 60)
    print()
    print(f"Trace directory: {trace_dir}")
    print()

    if not trace_files:
        print("No trace files found.")
        print()
        print("Expected file types: .pb, .pb.gz, .json, .perfetto-trace")
        sys.exit(1)

    print(f"Found {len(trace_files)} trace file(s):")
    for f in trace_files[:10]:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")
    if len(trace_files) > 10:
        print(f"  ... and {len(trace_files) - 10} more")
    print()

    print("To view traces in Perfetto:")
    print("  1. Open https://ui.perfetto.dev/")
    print("  2. Click 'Open trace file' or drag & drop")
    print(f"  3. Select file(s) from: {trace_dir}")
    print()

    if not args.no_browser:
        print("Opening Perfetto UI in browser...")
        webbrowser.open("https://ui.perfetto.dev/")
    else:
        print("(Browser opening skipped with --no-browser)")


if __name__ == "__main__":
    main()
