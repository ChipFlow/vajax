"""Waveform comparison utilities for VACASK vs JAX-SPICE validation.

This module provides tools to compare simulation results between VACASK
and JAX-SPICE, including:
- Running VACASK and parsing raw file output
- Comparing waveforms with tolerance-based matching
- Generating comparison reports
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np

from jax_spice.utils.rawfile import rawread, RawFile

# Type alias for array-like inputs (numpy or JAX arrays)
ArrayLike = Any  # Could be np.ndarray or jax.Array


@dataclass
class WaveformComparison:
    """Result of comparing two waveforms."""
    name: str
    max_abs_error: float
    max_rel_error: float
    rms_error: float
    points_compared: int
    within_tolerance: bool
    abs_tol: float
    rel_tol: float

    def __str__(self) -> str:
        status = "✓" if self.within_tolerance else "✗"
        return (
            f"{status} {self.name}: max_abs={self.max_abs_error:.2e}, "
            f"max_rel={self.max_rel_error:.2%}, rms={self.rms_error:.2e}"
        )


@dataclass
class ComparisonResult:
    """Result of comparing VACASK and JAX-SPICE simulations."""
    benchmark: str
    vacask_points: int
    jaxspice_points: int
    waveform_comparisons: List[WaveformComparison]
    all_passed: bool
    vacask_raw_path: Optional[Path] = None
    error: Optional[str] = None

    def summary(self) -> str:
        """Generate a summary string."""
        lines = [
            f"Benchmark: {self.benchmark}",
            f"VACASK points: {self.vacask_points}",
            f"JAX-SPICE points: {self.jaxspice_points}",
            "",
            "Waveform comparisons:"
        ]
        for wc in self.waveform_comparisons:
            lines.append(f"  {wc}")
        lines.append("")
        lines.append(f"Overall: {'PASS' if self.all_passed else 'FAIL'}")
        return "\n".join(lines)


def find_vacask_binary() -> Optional[Path]:
    """Find the VACASK binary."""
    import os

    # Check environment variable first
    if env_path := os.environ.get('VACASK_BIN'):
        path = Path(env_path)
        if path.exists() and path.is_file():
            return path

    # Check common locations
    # __file__ is jax_spice/utils/waveform_compare.py
    # .parent.parent.parent gets to jax-spice root
    project_root = Path(__file__).parent.parent.parent
    search_paths = [
        project_root / "vendor/VACASK/build/simulator/vacask",
        project_root / "vendor/VACASK/build.VACASK/Release/simulator/vacask",
        project_root / "vendor/VACASK/build.VACASK/Debug/simulator/vacask",
        Path.home() / "bin/vacask",
        Path("/usr/local/bin/vacask"),
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None


def run_vacask(
    sim_file: Path,
    output_dir: Optional[Path] = None,
    vacask_bin: Optional[Path] = None,
    timeout: int = 60,
) -> Tuple[Optional[Path], Optional[str]]:
    """Run VACASK on a simulation file.

    Args:
        sim_file: Path to .sim file
        output_dir: Directory for output files (default: temp dir)
        vacask_bin: Path to vacask binary (default: auto-detect)
        timeout: Timeout in seconds

    Returns:
        (raw_file_path, error_message) - raw_file_path is None if failed
    """
    if vacask_bin is None:
        vacask_bin = find_vacask_binary()
        if vacask_bin is None:
            return None, "VACASK binary not found"

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="vacask_"))

    # VACASK outputs to current directory, so we need to run from output_dir
    # Set PYTHONPATH to include VACASK's python directory for postprocess scripts
    # vacask_bin is typically at vendor/VACASK/build/simulator/vacask
    # so we need parent.parent.parent to get to vendor/VACASK
    vacask_python_dir = vacask_bin.parent.parent.parent / "python"
    env = os.environ.copy()
    if vacask_python_dir.exists():
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            env["PYTHONPATH"] = f"{vacask_python_dir}:{existing_pythonpath}"
        else:
            env["PYTHONPATH"] = str(vacask_python_dir)

    try:
        result = subprocess.run(
            [str(vacask_bin), str(sim_file.absolute())],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        # Find the .raw file - check even if returncode != 0 because
        # postprocess scripts may fail but simulation still produces output
        raw_files = list(output_dir.glob("*.raw"))
        if raw_files:
            return raw_files[0], None

        # No raw file found - report the actual error
        if result.returncode != 0:
            return None, f"VACASK failed: {result.stderr}"

        return None, f"No .raw file found in {output_dir}"

    except subprocess.TimeoutExpired:
        return None, f"VACASK timed out after {timeout}s"
    except Exception as e:
        return None, f"Error running VACASK: {e}"


def compare_waveforms(
    vacask_data: ArrayLike,
    jaxspice_data: ArrayLike,
    name: str,
    abs_tol: float = 1e-9,
    rel_tol: float = 1e-3,
) -> WaveformComparison:
    """Compare two waveforms.

    Args:
        vacask_data: VACASK waveform (numpy array)
        jaxspice_data: JAX-SPICE waveform (jax or numpy array)
        name: Name of the waveform for reporting
        abs_tol: Absolute tolerance
        rel_tol: Relative tolerance

    Returns:
        WaveformComparison with comparison metrics
    """
    # Convert to numpy for comparison
    if hasattr(jaxspice_data, 'block_until_ready'):
        jaxspice_data = np.asarray(jaxspice_data)

    # Ensure same length (interpolate if needed)
    n_vacask = len(vacask_data)
    n_jaxspice = len(jaxspice_data)

    if n_vacask != n_jaxspice:
        # Simple linear interpolation to match lengths
        # Use smaller length as reference
        if n_vacask < n_jaxspice:
            # Downsample JAX-SPICE
            indices = np.linspace(0, n_jaxspice - 1, n_vacask).astype(int)
            jaxspice_data = jaxspice_data[indices]
        else:
            # Downsample VACASK
            indices = np.linspace(0, n_vacask - 1, n_jaxspice).astype(int)
            vacask_data = vacask_data[indices]

    # Compute errors
    abs_error = np.abs(vacask_data - jaxspice_data)
    max_abs_error = float(np.max(abs_error))

    # Relative error (avoid division by zero)
    denom = np.maximum(np.abs(vacask_data), abs_tol)
    rel_error = abs_error / denom
    max_rel_error = float(np.max(rel_error))

    rms_error = float(np.sqrt(np.mean(abs_error ** 2)))

    # Check tolerance
    within_tol = max_abs_error <= abs_tol or max_rel_error <= rel_tol

    return WaveformComparison(
        name=name,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        rms_error=rms_error,
        points_compared=len(vacask_data),
        within_tolerance=within_tol,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )


def compare_transient(
    vacask_raw: RawFile,
    jaxspice_result: Dict[str, ArrayLike],
    node_mapping: Optional[Dict[str, str]] = None,
    abs_tol: float = 1e-9,
    rel_tol: float = 1e-3,
) -> List[WaveformComparison]:
    """Compare transient simulation results.

    Args:
        vacask_raw: Parsed VACASK raw file
        jaxspice_result: JAX-SPICE result dict with 'time' and node voltages
        node_mapping: Optional mapping from JAX-SPICE names to VACASK names
        abs_tol: Absolute tolerance
        rel_tol: Relative tolerance

    Returns:
        List of WaveformComparison for each compared signal
    """
    if node_mapping is None:
        node_mapping = {}

    comparisons = []

    # Compare time vectors first
    if 'time' in vacask_raw.names and 'time' in jaxspice_result:
        comparisons.append(compare_waveforms(
            vacask_raw['time'].real,
            jaxspice_result['time'],
            'time',
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        ))

    # Compare voltage signals
    for jax_name, jax_data in jaxspice_result.items():
        if jax_name == 'time':
            continue

        # Try to find matching VACASK signal
        vacask_name = node_mapping.get(jax_name, jax_name)

        if vacask_name in vacask_raw.names:
            comparisons.append(compare_waveforms(
                vacask_raw[vacask_name].real,
                jax_data,
                jax_name,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            ))

    return comparisons


def run_comparison(
    sim_file: Path,
    jaxspice_result: Dict[str, ArrayLike],
    benchmark_name: str,
    node_mapping: Optional[Dict[str, str]] = None,
    abs_tol: float = 1e-9,
    rel_tol: float = 1e-3,
    keep_raw: bool = False,
) -> ComparisonResult:
    """Run full comparison between VACASK and JAX-SPICE.

    Args:
        sim_file: Path to .sim file
        jaxspice_result: JAX-SPICE simulation result
        benchmark_name: Name of the benchmark for reporting
        node_mapping: Optional mapping from JAX-SPICE names to VACASK names
        abs_tol: Absolute tolerance
        rel_tol: Relative tolerance
        keep_raw: Keep the VACASK raw file after comparison

    Returns:
        ComparisonResult with all comparison data
    """
    # Run VACASK
    raw_path, error = run_vacask(sim_file)
    if error:
        return ComparisonResult(
            benchmark=benchmark_name,
            vacask_points=0,
            jaxspice_points=len(next(iter(jaxspice_result.values()))),
            waveform_comparisons=[],
            all_passed=False,
            error=error,
        )

    # Parse VACASK output
    try:
        vacask_raw = rawread(str(raw_path)).get()
    except Exception as e:
        return ComparisonResult(
            benchmark=benchmark_name,
            vacask_points=0,
            jaxspice_points=len(next(iter(jaxspice_result.values()))),
            waveform_comparisons=[],
            all_passed=False,
            error=f"Failed to parse VACASK output: {e}",
        )

    # Compare waveforms
    comparisons = compare_transient(
        vacask_raw,
        jaxspice_result,
        node_mapping=node_mapping,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )

    all_passed = all(c.within_tolerance for c in comparisons)

    result = ComparisonResult(
        benchmark=benchmark_name,
        vacask_points=vacask_raw.data.shape[0],
        jaxspice_points=len(next(iter(jaxspice_result.values()))),
        waveform_comparisons=comparisons,
        all_passed=all_passed,
        vacask_raw_path=raw_path if keep_raw else None,
    )

    # Cleanup if not keeping
    if not keep_raw and raw_path and raw_path.exists():
        raw_path.unlink()

    return result
