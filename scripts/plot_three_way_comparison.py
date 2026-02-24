#!/usr/bin/env python3
"""Three-way comparison: VACASK vs VA-JAX Full MNA vs ngspice.

Runs all three simulators and plots voltage/current comparisons.

Usage:
    # Ring oscillator (default)
    uv run scripts/plot_three_way_comparison.py

    # C6288 multiplier
    uv run scripts/plot_three_way_comparison.py --benchmark c6288

    # Skip running simulators (use existing data)
    uv run scripts/plot_three_way_comparison.py --skip-build

    # Skip ngspice (useful when it produces non-physical results)
    uv run scripts/plot_three_way_comparison.py --benchmark c6288 --skip-ngspice

    # Skip VACASK
    uv run scripts/plot_three_way_comparison.py --skip-vacask
"""

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

os.environ["JAX_PLATFORMS"] = "cpu"

import matplotlib.pyplot as plt
import numpy as np

from vajax._logging import enable_performance_logging, logger
from vajax.analysis.engine import CircuitEngine
from vajax.analysis.transient import FullMNAStrategy, extract_results
from vajax.utils import find_ngspice_binary
from vajax.utils import run_ngspice as run_ngspice_util
from vajax.utils import run_vacask as run_vacask_util

enable_performance_logging(with_memory=True, with_perf_counter=True)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark.

    t_stop, dt: If None, uses values from netlist analysis line.
    plot_window: If None, defaults to (0, t_stop).
    """

    name: str
    vacask_sim: Path
    ngspice_sim: Path
    voltage_nodes: list  # Nodes to plot
    current_source: str  # Name of vsource for current
    t_stop: Optional[float] = None  # From netlist if None
    dt: Optional[float] = None  # From netlist 'step' if None
    plot_window: Optional[Tuple[float, float]] = None  # (0, t_stop) if None
    use_sparse: bool = False
    icmode: Optional[str] = None
    input_nodes_a: Optional[list] = None  # Input bus A nodes (e.g., a0-a15)
    input_nodes_b: Optional[list] = None  # Input bus B nodes (e.g., b0-b15)


BENCHMARKS = {
    "rc": BenchmarkConfig(
        name="RC Low-Pass Filter",
        vacask_sim=Path("vendor/VACASK/benchmark/rc/vacask/runme.sim"),
        ngspice_sim=Path("vendor/VACASK/benchmark/rc/ngspice/runme.sim"),
        voltage_nodes=["1", "2"],
        current_source="vs",
        t_stop=10e-3,  # 10ms (netlist is 1s - too long)
        plot_window=(0, 10e-3),
    ),
    "graetz": BenchmarkConfig(
        name="Graetz Bridge Rectifier",
        vacask_sim=Path("vendor/VACASK/benchmark/graetz/vacask/runme.sim"),
        ngspice_sim=Path("vendor/VACASK/benchmark/graetz/ngspice/runme.sim"),
        voltage_nodes=["inp", "outp"],
        current_source="vs",
        t_stop=100e-3,  # 100ms (netlist is 1s - show 5 cycles at 50Hz)
        plot_window=(0, 100e-3),
    ),
    "mul": BenchmarkConfig(
        name="Diode Voltage Multiplier",
        vacask_sim=Path("vendor/VACASK/benchmark/mul/vacask/runme.sim"),
        ngspice_sim=Path("vendor/VACASK/benchmark/mul/ngspice/runme.sim"),
        voltage_nodes=["1", "2", "10", "20"],
        current_source="vs",
        t_stop=100e-6,  # 100us (netlist is 5ms - show 10 cycles at 100kHz)
        plot_window=(0, 100e-6),
    ),
    "ring": BenchmarkConfig(
        name="PSP103 Ring Oscillator",
        vacask_sim=Path("vendor/VACASK/benchmark/ring/vacask/runme.sim"),
        ngspice_sim=Path("vendor/VACASK/benchmark/ring/ngspice/runme.sim"),
        voltage_nodes=["1", "2"],
        current_source="vdd",
        t_stop=20e-9,  # 20ns (netlist is 1us - show several oscillation periods)
        plot_window=(2e-9, 18e-9),
    ),
    "c6288": BenchmarkConfig(
        name="C6288 16-bit Multiplier (PSP103)",
        vacask_sim=Path("vendor/VACASK/benchmark/c6288/vacask/runme.sim"),
        ngspice_sim=Path("vendor/VACASK/benchmark/c6288/ngspice/runme.sim"),
        voltage_nodes=[f"top.p{n}" for n in range(32)],
        current_source="vdd",
        # Uses netlist: stop=2n, step=2p
        use_sparse=True,
        input_nodes_a=[f"a{i}" for i in range(16)],
        input_nodes_b=[f"b{i}" for i in range(16)],
    ),
}


def read_spice_raw(filename: Path) -> Dict[str, np.ndarray]:
    """Read a SPICE raw file (binary format)."""
    with open(filename, "rb") as f:
        content = f.read()

    binary_marker = b"Binary:\n"
    binary_pos = content.find(binary_marker)
    if binary_pos < 0:
        raise ValueError(f"No binary marker found in {filename}")

    header = content[:binary_pos].decode("utf-8")
    lines = header.strip().split("\n")

    n_vars = n_points = None
    variables = []
    in_variables = False

    for line in lines:
        if line.startswith("No. Variables:"):
            n_vars = int(line.split(":")[1].strip())
        elif line.startswith("No. Points:"):
            n_points = int(line.split(":")[1].strip())
        elif line.startswith("Variables:"):
            in_variables = True
        elif in_variables and line.strip():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                variables.append(parts[1])

    binary_data = content[binary_pos + len(binary_marker) :]
    point_size = n_vars * 8
    n_points = min(n_points, len(binary_data) // point_size)

    data = np.zeros((n_points, n_vars), dtype=np.float64)
    for i in range(n_points):
        offset = i * point_size
        for j in range(n_vars):
            val_bytes = binary_data[offset + j * 8 : offset + (j + 1) * 8]
            if len(val_bytes) == 8:
                data[i, j] = struct.unpack("d", val_bytes)[0]

    return {name: data[:, i] for i, name in enumerate(variables)}


def get_config_hash(config: BenchmarkConfig, simulator: str) -> str:
    """Compute hash of netlist file for caching.

    Since we run netlists as-is, the hash is based on file content/mtime.
    """
    sim_file = config.vacask_sim if simulator == "vacask" else config.ngspice_sim
    params = {
        "simulator": simulator,
        "sim_file_mtime": sim_file.stat().st_mtime,
    }
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


def check_cache(raw_file: Path, stamp_file: Path, expected_hash: str) -> bool:
    """Check if cached data is valid. Returns True if cache hit."""
    if not raw_file.exists():
        return False
    if not stamp_file.exists():
        return False
    try:
        stored_hash = stamp_file.read_text().strip()
        return stored_hash == expected_hash
    except Exception:
        return False


def write_stamp(stamp_file: Path, config_hash: str) -> None:
    """Write stamp file with config hash."""
    stamp_file.write_text(config_hash)


def run_vacask(config: BenchmarkConfig, output_dir: Path, benchmark_key: str) -> Optional[Path]:
    """Run VACASK simulator and return path to raw file.

    Runs the original netlist as-is. VA-JAX uses the same netlist settings
    via the API, so all simulators run with identical parameters.
    """
    import shutil
    import tempfile

    raw_file = output_dir / f"{benchmark_key}_vacask.raw"
    stamp_file = output_dir / f"{benchmark_key}_vacask.stamp"

    # Check cache (hash based on netlist content)
    config_hash = get_config_hash(config, "vacask")
    if check_cache(raw_file, stamp_file, config_hash):
        logger.info(f"Using cached VACASK data: {raw_file}")
        return raw_file

    logger.info(f"Running VACASK ({config.name})...")
    start = time.perf_counter()

    # Run original netlist - no modifications needed
    temp_output = Path(tempfile.mkdtemp(prefix="vacask_"))
    result_raw, error = run_vacask_util(config.vacask_sim, output_dir=temp_output, timeout=600)
    elapsed = time.perf_counter() - start

    if error:
        logger.error(f"VACASK failed: {error}")
        return None

    if result_raw and result_raw.exists():
        shutil.copy(result_raw, raw_file)
        write_stamp(stamp_file, config_hash)
        logger.info(f"VACASK completed in {elapsed:.1f}s -> {raw_file}")
        return raw_file
    else:
        logger.error("VACASK did not produce output")
        return None


def run_ngspice(config: BenchmarkConfig, output_dir: Path, benchmark_key: str) -> Optional[Path]:
    """Run ngspice and return path to raw file."""
    import shutil
    import tempfile

    # Check if ngspice is available
    if find_ngspice_binary() is None:
        logger.warning("ngspice not found - skipping")
        return None

    sim_dir = config.ngspice_sim.parent
    raw_file = output_dir / f"{benchmark_key}_ngspice.raw"
    stamp_file = output_dir / f"{benchmark_key}_ngspice.stamp"

    # Check cache
    config_hash = get_config_hash(config, "ngspice")
    if check_cache(raw_file, stamp_file, config_hash):
        logger.info(f"Using cached ngspice data: {raw_file}")
        return raw_file

    # Copy OSDI files from vacask directory if needed
    with open(config.ngspice_sim) as f:
        sim_content = f.read()
    osdi_files = re.findall(r"pre_osdi\s+(\S+)", sim_content)
    vacask_dir = config.vacask_sim.parent
    for osdi_file in osdi_files:
        osdi_src = vacask_dir / osdi_file
        osdi_dst = sim_dir / osdi_file
        if osdi_src.exists() and not osdi_dst.exists():
            shutil.copy(osdi_src, osdi_dst)
            logger.info(f"Copied {osdi_file} from vacask to ngspice directory")

    logger.info(f"Running ngspice ({config.name})...")
    start = time.perf_counter()

    # Run original netlist - no modifications needed
    temp_output = Path(tempfile.mkdtemp(prefix="ngspice_"))
    result_raw, error = run_ngspice_util(config.ngspice_sim, output_dir=temp_output, timeout=600)
    elapsed = time.perf_counter() - start

    if error:
        logger.error(f"ngspice failed: {error}")
        return None

    if result_raw and result_raw.exists():
        shutil.copy(result_raw, raw_file)
        write_stamp(stamp_file, config_hash)
        logger.info(f"ngspice completed in {elapsed:.1f}s -> {raw_file}")
        return raw_file
    else:
        logger.warning("ngspice did not produce raw file")
        return None


def run_vajax(config: BenchmarkConfig) -> Tuple[np.ndarray, Dict, Dict]:
    """Run VA-JAX Full MNA and return (times, voltages, currents)."""
    logger.info(f"Running VA-JAX Full MNA ({config.name})...")

    runner = CircuitEngine(config.vacask_sim)
    runner.parse()

    # Strategy uses netlist options (tran_lteratio, nr_convtol, tran_method, etc.)
    full_mna = FullMNAStrategy(runner, use_sparse=config.use_sparse)

    # Warmup uses netlist 'step' as default dt
    logger.info("Warmup...")
    _ = full_mna.warmup()

    # Run uses netlist 'stop' and 'step' as defaults, config overrides if set
    logger.info("Running simulation...")
    times_mna, V_out, stats_mna = full_mna.run(t_stop=config.t_stop, dt=config.dt)

    t_mna, voltages_mna, currents_mna = extract_results(times_mna, V_out, stats_mna)
    logger.info(f"VA-JAX completed: {len(t_mna)} points")

    return t_mna, voltages_mna, currents_mna


def plot_comparison(
    config: BenchmarkConfig,
    vacask_data: Optional[Dict],
    ngspice_data: Optional[Dict],
    jax_data: Tuple,
    output_file: Path,
):
    """Create comparison plot."""
    t_mna, voltages_mna, currents_mna = jax_data

    # Determine number of panels
    has_inputs = config.input_nodes_a or config.input_nodes_b
    n_panels = 2  # Output voltage + current
    if has_inputs:
        n_panels += 2  # Add input A and input B panels

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    # Default plot window to full simulation range
    if config.plot_window:
        t_start, t_end = config.plot_window
    else:
        t_start = 0.0
        t_end = float(t_mna[-1]) if len(t_mna) > 0 else 1e-9
    # Use picoseconds for very short simulations
    time_scale = 1e12 if t_end < 1e-9 else 1e9
    time_unit = "ps" if time_scale == 1e12 else "ns"

    c_vac = "blue"
    panel_idx = 0

    # Get time masks
    mask_mna = (t_mna >= t_start) & (t_mna <= t_end)
    if vacask_data:
        t_vac = vacask_data["time"]
        mask_vac = (t_vac >= t_start) & (t_vac <= t_end)
    if ngspice_data:
        t_ng = ngspice_data["time"]
        mask_ng = (t_ng >= t_start) & (t_ng <= t_end)

    # Helper to get ngspice voltage (handles v(node) naming convention)
    def get_ngspice_voltage(data, node):
        """Get voltage from ngspice data, trying both 'node' and 'v(node)' formats."""
        if data is None:
            return None
        if node in data:
            return data[node]
        if f"v({node})" in data:
            return data[f"v({node})"]
        return None

    # Panel: Input A (if configured)
    if config.input_nodes_a:
        ax = axes[panel_idx]
        # Plot each input bit with offset for visibility
        for i, node in enumerate(config.input_nodes_a):
            offset = i * 1.5  # Offset each bit for visibility
            # Plot VACASK
            if vacask_data and node in vacask_data:
                ax.plot(
                    t_vac[mask_vac] * time_scale,
                    vacask_data[node][mask_vac] + offset,
                    "b-",
                    lw=0.8,
                    alpha=0.8,
                    label=f"VAC {node}" if i < 2 else None,
                )
            # Plot ngspice
            ng_voltage = get_ngspice_voltage(ngspice_data, node)
            if ng_voltage is not None:
                ax.plot(
                    t_ng[mask_ng] * time_scale,
                    ng_voltage[mask_ng] + offset,
                    "g--",
                    lw=0.8,
                    alpha=0.8,
                    label=f"NG {node}" if i < 2 else None,
                )
            # Plot VA-JAX - try with and without prefix
            jax_input = voltages_mna.get(node)
            if jax_input is None:
                jax_input = voltages_mna.get(f"top.{node}")
            if jax_input is not None:
                ax.plot(
                    t_mna[mask_mna] * time_scale,
                    jax_input[mask_mna] + offset,
                    "r:",
                    lw=0.8,
                    alpha=0.8,
                    label=f"JAX {node}" if i < 2 else None,
                )
        ax.set_ylabel("Input A [V + offset]", fontsize=11)
        ax.set_title(f"{config.name}: Input Bus A (a0-a15)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=4, fontsize=8)
        panel_idx += 1

    # Panel: Input B (if configured)
    if config.input_nodes_b:
        ax = axes[panel_idx]
        for i, node in enumerate(config.input_nodes_b):
            offset = i * 1.5
            # Plot VACASK
            if vacask_data and node in vacask_data:
                ax.plot(
                    t_vac[mask_vac] * time_scale,
                    vacask_data[node][mask_vac] + offset,
                    "b-",
                    lw=0.8,
                    alpha=0.8,
                    label=f"VAC {node}" if i < 2 else None,
                )
            # Plot ngspice
            ng_voltage = get_ngspice_voltage(ngspice_data, node)
            if ng_voltage is not None:
                ax.plot(
                    t_ng[mask_ng] * time_scale,
                    ng_voltage[mask_ng] + offset,
                    "g--",
                    lw=0.8,
                    alpha=0.8,
                    label=f"NG {node}" if i < 2 else None,
                )
            # Plot VA-JAX
            jax_input = voltages_mna.get(node)
            if jax_input is None:
                jax_input = voltages_mna.get(f"top.{node}")
            if jax_input is not None:
                ax.plot(
                    t_mna[mask_mna] * time_scale,
                    jax_input[mask_mna] + offset,
                    "r:",
                    lw=0.8,
                    alpha=0.8,
                    label=f"JAX {node}" if i < 2 else None,
                )
        ax.set_ylabel("Input B [V + offset]", fontsize=11)
        ax.set_title("Input Bus B (b0-b15)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=4, fontsize=8)
        panel_idx += 1

    # Panel: Output Voltages
    ax = axes[panel_idx]
    for i, v_node in enumerate(config.voltage_nodes):
        # VACASK uses short names (p0), JAX uses full names (top.p0)
        vac_node = v_node.split(".")[-1]  # Get short name for VACASK
        offset = i * 1.5  # Offset each bit for visibility

        # Plot VACASK data
        if vacask_data and vac_node in vacask_data:
            ax.plot(
                t_vac[mask_vac] * time_scale,
                vacask_data[vac_node][mask_vac] + offset,
                c_vac,
                lw=0.8,
                alpha=0.8,
                label=f"VAC {vac_node}" if i < 2 else None,
            )

        # Plot ngspice data
        ng_voltage = get_ngspice_voltage(ngspice_data, vac_node)
        if ng_voltage is not None:
            ax.plot(
                t_ng[mask_ng] * time_scale,
                ng_voltage[mask_ng] + offset,
                "g--",
                lw=0.8,
                alpha=0.8,
                label=f"NG {vac_node}" if i < 2 else None,
            )

        # Plot VA-JAX data - try both full name and short name
        jax_voltage = voltages_mna.get(v_node)
        if jax_voltage is None:
            jax_voltage = voltages_mna.get(vac_node)
        if jax_voltage is not None:
            ax.plot(
                t_mna[mask_mna] * time_scale,
                jax_voltage[mask_mna] + offset,
                "r:",
                lw=0.8,
                alpha=0.8,
                label=f"JAX {vac_node}" if i < 2 else None,
            )

    ax.set_ylabel("Output [V + offset]", fontsize=11)
    ax.set_title(
        f"Output Bits ({', '.join([n.split('.')[-1] for n in config.voltage_nodes[:3]])}...)",
        fontsize=12,
    )
    ax.legend(loc="upper right", fontsize=9, ncol=4)
    ax.grid(True, alpha=0.3)
    panel_idx += 1

    # Panel: Current
    ax = axes[panel_idx]
    I_vac = None
    if vacask_data:
        I_vac = vacask_data.get(f"{config.current_source}:flow(br)")
        if I_vac is not None:
            ax.plot(
                t_vac[mask_vac] * time_scale,
                I_vac[mask_vac] * 1e6,
                c_vac,
                lw=1.5,
                label="VACASK",
                alpha=0.9,
            )

    # ngspice current - try different naming conventions
    if ngspice_data:
        # ngspice names current through voltage source vdd as i(vdd)
        I_ng = ngspice_data.get(f"i({config.current_source})")
        if I_ng is None:
            I_ng = ngspice_data.get(f"{config.current_source}#branch")
        if I_ng is not None:
            ax.plot(
                t_ng[mask_ng] * time_scale,
                I_ng[mask_ng] * 1e6,
                "green",
                lw=1.5,
                label="ngspice",
                alpha=0.9,
                linestyle="--",
            )

    I_mna = currents_mna.get(config.current_source)
    if I_mna is not None:
        ax.plot(
            t_mna[mask_mna] * time_scale,
            I_mna[mask_mna] * 1e6,
            "red",
            lw=1.5,
            label="VA-JAX",
            alpha=0.9,
            linestyle=":",
        )

    ax.set_ylabel(f"I({config.current_source}) [µA]", fontsize=11)
    ax.set_xlabel(f"Time [{time_unit}]", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Three-way simulator comparison")
    parser.add_argument(
        "--benchmark",
        choices=list(BENCHMARKS.keys()),
        default="ring",
        help="Benchmark to run (default: ring)",
    )
    parser.add_argument(
        "--skip-build", action="store_true", help="Skip running VACASK/ngspice, use existing data"
    )
    parser.add_argument(
        "--skip-ngspice",
        action="store_true",
        help="Skip running ngspice (useful when ngspice produces bad results)",
    )
    parser.add_argument("--skip-vacask", action="store_true", help="Skip running VACASK")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."), help="Output directory for plots and data"
    )
    args = parser.parse_args()

    config = BENCHMARKS[args.benchmark]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    vacask_data = None
    ngspice_data = None

    if not args.skip_build:
        # Run VACASK
        if not args.skip_vacask:
            vacask_raw = run_vacask(config, output_dir, args.benchmark)
            if vacask_raw and vacask_raw.exists():
                vacask_data = read_spice_raw(vacask_raw)
                logger.info(f"VACASK data: {list(vacask_data.keys())[:10]}...")
        else:
            logger.info("Skipping VACASK (--skip-vacask)")

        # Run ngspice
        if not args.skip_ngspice:
            ngspice_raw = run_ngspice(config, output_dir, args.benchmark)
            if ngspice_raw and ngspice_raw.exists():
                ngspice_data = read_spice_raw(ngspice_raw)
                logger.info(f"ngspice data: {list(ngspice_data.keys())[:10]}...")
        else:
            logger.info("Skipping ngspice (--skip-ngspice)")
    else:
        # Try to load existing data
        vacask_raw = output_dir / f"{args.benchmark}_vacask.raw"
        ngspice_raw = output_dir / f"{args.benchmark}_ngspice.raw"
        if vacask_raw.exists() and not args.skip_vacask:
            vacask_data = read_spice_raw(vacask_raw)
        if ngspice_raw.exists() and not args.skip_ngspice:
            ngspice_data = read_spice_raw(ngspice_raw)

    # Always run VA-JAX
    jax_data = run_vajax(config)

    # Plot comparison
    output_file = output_dir / f"{args.benchmark}_three_way_comparison.png"
    plot_comparison(config, vacask_data, ngspice_data, jax_data, output_file)

    # Print metrics
    logger.info("\n" + "=" * 70)
    logger.info(f"Comparison complete: {config.name}")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
