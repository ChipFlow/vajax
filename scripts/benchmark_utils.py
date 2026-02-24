"""Shared utilities for benchmark and profiling scripts.

This module consolidates common functionality used across profile_cpu.py,
profile_gpu.py, compare_vacask.py, and test files.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Re-export find_vacask_binary from vajax.utils to avoid duplication


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run.

    Metrics are designed to be directly comparable with VACASK benchmark.py output:
    - total_time_s: Total wall-clock time for the simulation (comparable to VACASK "Runtime")
    - time_per_step_ms: Derived metric for per-step performance analysis
    """

    name: str
    nodes: int
    devices: int
    openvaf_devices: int
    timesteps: int
    total_time_s: float
    time_per_step_ms: float
    solver: str  # 'dense' or 'sparse'
    backend: str = "cpu"  # 'cpu' or 'gpu'
    # Analysis parameters (for VACASK comparison)
    t_stop: float = 0.0  # Simulation stop time (seconds)
    dt: float = 0.0  # Time step (seconds)
    converged: bool = True
    error: Optional[str] = None


def get_vacask_benchmarks(names: Optional[List[str]] = None) -> List[Tuple[str, Path]]:
    """Get list of VACASK benchmark .sim files.

    Args:
        names: Optional list of benchmark names to filter. If None, returns all.

    Returns:
        List of (name, sim_path) tuples for available benchmarks.
    """
    # Find the benchmark directory relative to this file or the project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    base = project_root / "vendor" / "VACASK" / "benchmark"

    all_benchmarks = ["rc", "graetz", "mul", "ring", "c6288"]

    if names:
        all_benchmarks = [n for n in names if n in all_benchmarks]

    benchmarks = []
    for name in all_benchmarks:
        sim_path = base / name / "vacask" / "runme.sim"
        if sim_path.exists():
            benchmarks.append((name, sim_path))

    return benchmarks


def log(msg: str = "", end: str = "\n") -> None:
    """Print with flush for real-time output.

    Args:
        msg: Message to print
        end: Line ending (default newline)
    """
    print(msg, end=end, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
