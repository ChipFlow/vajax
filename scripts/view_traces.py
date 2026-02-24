#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
View JAX profiling traces in Perfetto UI.

This script:
1. Downloads traces from CI or GCS (with local caching)
2. Generates a signed URL for direct Perfetto access
3. Opens Perfetto UI with the trace pre-loaded

Usage:
    # View traces from latest CI run (default)
    uv run scripts/view_traces.py

    # View traces from a specific workflow run
    uv run scripts/view_traces.py --run 20378600298

    # View traces from a local directory
    uv run scripts/view_traces.py /tmp/va-jax-traces

    # Download and view traces from GCS
    uv run scripts/view_traces.py --gcs gs://va-jax-cuda-test-traces/abc123
"""

import argparse
import json
import subprocess
import sys
import urllib.parse
import webbrowser
from pathlib import Path

# Cache directory for downloaded traces
CACHE_DIR = Path.home() / ".cache" / "va-jax-traces"
GCS_BUCKET = "va-jax-cuda-test-traces"


def list_trace_files(trace_dir: Path) -> list[Path]:
    """List all trace files in a directory."""
    trace_files = []
    for pattern in ["*.pb", "*.pb.gz", "*.json", "*.perfetto-trace"]:
        trace_files.extend(trace_dir.glob(pattern))
        trace_files.extend(trace_dir.glob(f"**/{pattern}"))
    return sorted(set(trace_files))


def get_gcs_path_for_trace(trace_file: Path, cache_dir: Path) -> str | None:
    """Get the GCS path for a cached trace file.

    Returns the GCS object path if the trace was downloaded from GCS/CI.
    """
    # Check if this is from a CI run (has commit SHA in path)
    rel_path = trace_file.relative_to(cache_dir)
    parts = rel_path.parts

    # Pattern: run-{run_id}/profiling-traces-{sha}/benchmark_*/...
    # or: {sha}/benchmark_*/...
    for i, part in enumerate(parts):
        if part.startswith("profiling-traces-"):
            sha = part.replace("profiling-traces-", "")
            remaining = "/".join(parts[i + 1 :])
            return f"{sha}/{remaining}"
        if len(part) == 40 and all(c in "0123456789abcdef" for c in part):
            # Looks like a commit SHA
            remaining = "/".join(parts[i + 1 :])
            return f"{part}/{remaining}"

    return None


def generate_signed_url(gcs_object_path: str, duration: str = "1h") -> str | None:
    """Generate a signed URL for a GCS object."""
    gcs_url = f"gs://{GCS_BUCKET}/{gcs_object_path}"

    result = subprocess.run(
        ["gsutil", "signurl", "-d", duration, "-u", gcs_url],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # signurl requires a service account key, try public URL if bucket is public
        return None

    # Parse the signed URL from output (last line, last column)
    for line in result.stdout.strip().split("\n"):
        if "storage.googleapis.com" in line:
            return line.split()[-1]

    return None


def open_perfetto_with_url(trace_url: str) -> None:
    """Open Perfetto UI with a trace URL."""
    encoded_url = urllib.parse.quote(trace_url, safe="")
    perfetto_url = f"https://ui.perfetto.dev/#!/?url={encoded_url}"

    print("Opening Perfetto UI with trace...")
    print(f"  {perfetto_url[:80]}...")
    webbrowser.open(perfetto_url)


def download_from_github(run_id: str | None = None) -> tuple[Path, str | None]:
    """Download traces from GitHub workflow artifact.

    Returns (local_dir, commit_sha) tuple.
    If run_id is None, downloads from the latest successful GPU Tests run.
    """
    # Find the run ID and commit SHA if not specified
    print("Finding latest GPU Tests workflow run...")
    result = subprocess.run(
        [
            "gh",
            "run",
            "list",
            "--workflow=GPU Tests (Cloud Run)",
            "--status=success",
            "--limit=10",
            "--json=databaseId,headSha,createdAt",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error listing workflow runs: {result.stderr}")
        sys.exit(1)

    runs = json.loads(result.stdout)
    if not runs:
        print("No successful GPU Tests runs found.")
        sys.exit(1)

    # Find the matching run
    if run_id is None:
        selected_run = runs[0]
    else:
        selected_run = next((r for r in runs if str(r["databaseId"]) == run_id), None)
        if not selected_run:
            print(f"Run {run_id} not found in recent successful runs.")
            sys.exit(1)

    run_id = str(selected_run["databaseId"])
    commit_sha = selected_run["headSha"]
    created = selected_run["createdAt"]
    print(f"  Found run {run_id} (commit {commit_sha[:8]}, {created})")

    # Check cache
    cache_dir = CACHE_DIR / f"run-{run_id}"
    if cache_dir.exists() and list_trace_files(cache_dir):
        print(f"  Using cached traces from {cache_dir}")
        return cache_dir, commit_sha

    # Download the artifact
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading profiling traces from run {run_id}...")

    result = subprocess.run(
        [
            "gh",
            "run",
            "download",
            run_id,
            "--name",
            "profiling-traces-*",
            "--dir",
            str(cache_dir),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Try with pattern matching (older gh versions)
        result = subprocess.run(
            [
                "gh",
                "run",
                "download",
                run_id,
                "--pattern",
                "profiling-traces-*",
                "--dir",
                str(cache_dir),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error downloading artifact: {result.stderr}")
            print()
            print("Make sure you have gh CLI installed and authenticated:")
            print("  gh auth login")
            sys.exit(1)

    return cache_dir, commit_sha


def download_from_gcs(gcs_path: str) -> tuple[Path, str | None]:
    """Download traces from GCS to cache directory.

    Returns (local_dir, commit_sha) tuple.
    """
    # Extract path components for cache key
    gcs_path = gcs_path.rstrip("/")
    path_parts = gcs_path.replace("gs://", "").replace(f"{GCS_BUCKET}/", "").split("/")
    cache_key = path_parts[0] if path_parts else "traces"

    # Commit SHA is typically the first path component
    commit_sha = cache_key if len(cache_key) == 40 else None

    # Check cache
    cache_dir = CACHE_DIR / cache_key
    if cache_dir.exists() and list_trace_files(cache_dir):
        print(f"Using cached traces from {cache_dir}")
        return cache_dir, commit_sha

    # Download
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading traces from {gcs_path}...")

    result = subprocess.run(
        ["gsutil", "-m", "rsync", "-r", gcs_path, str(cache_dir)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Try cp -r as fallback
        result = subprocess.run(
            ["gsutil", "-m", "cp", "-r", f"{gcs_path}/*", str(cache_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error downloading traces: {result.stderr}")
            sys.exit(1)

    return cache_dir, commit_sha


def main():
    parser = argparse.ArgumentParser(description="View JAX profiling traces in Perfetto UI")
    parser.add_argument(
        "trace_dir",
        type=str,
        nargs="?",
        help="Directory containing trace files (default: download from latest CI run)",
    )
    parser.add_argument(
        "--run",
        type=str,
        nargs="?",
        const="latest",
        help="Download from GitHub workflow run (default: latest successful run)",
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
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the trace cache before downloading",
    )
    args = parser.parse_args()

    # Clear cache if requested
    if args.clear_cache and CACHE_DIR.exists():
        import shutil

        print(f"Clearing cache: {CACHE_DIR}")
        shutil.rmtree(CACHE_DIR)

    # Determine trace source
    commit_sha = None
    if args.gcs:
        trace_dir, commit_sha = download_from_gcs(args.gcs)
    elif args.run is not None:
        run_id = None if args.run == "latest" else args.run
        trace_dir, commit_sha = download_from_github(run_id)
    elif args.trace_dir:
        trace_dir = Path(args.trace_dir)
    else:
        # Default: download from latest CI run
        trace_dir, commit_sha = download_from_github(None)

    if not trace_dir.exists():
        print(f"Error: Trace directory does not exist: {trace_dir}")
        print()
        print("To generate traces, run benchmarks with profiling enabled:")
        print("  VA_JAX_PROFILE_JAX=1 uv run python scripts/compare_vacask.py --profile")
        print()
        print("Or run on Cloud Run GPU:")
        print("  uv run scripts/profile_gpu_cloudrun.py")
        sys.exit(1)

    # List trace files
    trace_files = list_trace_files(trace_dir)

    print("=" * 60)
    print("VA-JAX Profiling Trace Viewer")
    print("=" * 60)
    print()
    print(f"Trace directory: {trace_dir}")
    print()

    if not trace_files:
        print("No trace files found.")
        print()
        print("Expected file types: .pb, .pb.gz, .json, .perfetto-trace")
        sys.exit(1)

    # Sort by size (largest first - usually most interesting)
    trace_files_by_size = sorted(trace_files, key=lambda f: f.stat().st_size, reverse=True)

    print(f"Found {len(trace_files)} trace file(s):")
    for i, f in enumerate(trace_files_by_size[:10]):
        size_kb = f.stat().st_size / 1024
        # Show relative path from trace_dir for clarity
        try:
            rel_path = f.relative_to(trace_dir)
        except ValueError:
            rel_path = f.name
        marker = " <- largest" if i == 0 else ""
        print(f"  {rel_path} ({size_kb:.1f} KB){marker}")
    if len(trace_files) > 10:
        print(f"  ... and {len(trace_files) - 10} more")
    print()

    if args.no_browser:
        print("To view traces in Perfetto:")
        print("  1. Open https://ui.perfetto.dev/")
        print("  2. Click 'Open trace file' or drag & drop")
        print(f"  3. Select file(s) from: {trace_dir}")
        return

    # Select the largest .xplane.pb file (most detailed trace)
    xplane_files = [f for f in trace_files_by_size if f.name.endswith(".xplane.pb")]
    trace_to_open = xplane_files[0] if xplane_files else trace_files_by_size[0]

    try:
        rel_path = trace_to_open.relative_to(trace_dir)
    except ValueError:
        rel_path = trace_to_open.name
    print(f"Opening: {rel_path}")
    print()

    # Try to get a signed GCS URL for direct Perfetto access
    if commit_sha:
        # Build GCS object path
        gcs_object_path = get_gcs_path_for_trace(trace_to_open, CACHE_DIR)
        if gcs_object_path:
            print(f"Generating signed URL for gs://{GCS_BUCKET}/{gcs_object_path}...")
            signed_url = generate_signed_url(gcs_object_path)
            if signed_url:
                open_perfetto_with_url(signed_url)
                return

    # Fallback: just open Perfetto and tell user to drag & drop
    print("Could not generate signed URL. Opening Perfetto UI...")
    print(f"Drag & drop this file: {trace_to_open}")
    webbrowser.open("https://ui.perfetto.dev/")


if __name__ == "__main__":
    main()
