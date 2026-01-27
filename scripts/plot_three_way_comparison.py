#!/usr/bin/env python3
"""Three-way comparison: VACASK vs JAX-SPICE Full MNA vs ngspice.

Runs all three simulators and plots voltage/current comparisons.

Usage:
    # Ring oscillator (default)
    uv run scripts/plot_three_way_comparison.py

    # C6288 multiplier
    uv run scripts/plot_three_way_comparison.py --benchmark c6288

    # Skip running simulators (use existing data)
    uv run scripts/plot_three_way_comparison.py --skip-build
"""

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

os.environ['JAX_PLATFORMS'] = 'cpu'

import matplotlib.pyplot as plt
import numpy as np

from jax_spice._logging import enable_performance_logging, logger
from jax_spice.analysis.engine import CircuitEngine
from jax_spice.analysis.transient import AdaptiveConfig, FullMNAStrategy, extract_results
from jax_spice.utils import run_vacask as run_vacask_util

enable_performance_logging(with_memory=True, with_perf_counter=True)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark."""
    name: str
    vacask_sim: Path
    ngspice_sim: Path
    t_stop: float
    dt: float
    max_dt: float
    plot_window: Tuple[float, float]
    voltage_nodes: list  # Nodes to plot
    current_source: str  # Name of vsource for current
    use_sparse: bool = False
    icmode: Optional[str] = None
    input_nodes_a: Optional[list] = None  # Input bus A nodes (e.g., a0-a15)
    input_nodes_b: Optional[list] = None  # Input bus B nodes (e.g., b0-b15)


BENCHMARKS = {
    'ring': BenchmarkConfig(
        name='Ring Oscillator',
        vacask_sim=Path('vendor/VACASK/benchmark/ring/vacask/runme.sim'),
        ngspice_sim=Path('vendor/VACASK/benchmark/ring/ngspice/runme.sim'),
        t_stop=20e-9,
        dt=1e-12,
        max_dt=50e-12,
        plot_window=(2e-9, 15e-9),
        voltage_nodes=['1', '2'],
        current_source='vdd',
    ),
    'c6288': BenchmarkConfig(
        name='C6288 16x16 Multiplier',
        vacask_sim=Path('vendor/VACASK/benchmark/c6288/vacask/runme.sim'),
        ngspice_sim=Path('vendor/VACASK/benchmark/c6288/ngspice/runme.sim'),
        t_stop=5e-14,
        dt=1e-15,
        max_dt=1e-10,
        plot_window=(0.0, 5e-14),
        voltage_nodes=[ f'top.p{n}' for n in range(32) ],
        current_source='vdd',
        use_sparse=True,
        icmode='uic',
        input_nodes_a=[f'a{i}' for i in range(16)],  # a0-a15
        input_nodes_b=[f'b{i}' for i in range(16)],  # b0-b15
    ),
}


def read_spice_raw(filename: Path) -> Dict[str, np.ndarray]:
    """Read a SPICE raw file (binary format)."""
    with open(filename, 'rb') as f:
        content = f.read()

    binary_marker = b'Binary:\n'
    binary_pos = content.find(binary_marker)
    if binary_pos < 0:
        raise ValueError(f"No binary marker found in {filename}")

    header = content[:binary_pos].decode('utf-8')
    lines = header.strip().split('\n')

    n_vars = n_points = None
    variables = []
    in_variables = False

    for line in lines:
        if line.startswith('No. Variables:'):
            n_vars = int(line.split(':')[1].strip())
        elif line.startswith('No. Points:'):
            n_points = int(line.split(':')[1].strip())
        elif line.startswith('Variables:'):
            in_variables = True
        elif in_variables and line.strip():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                variables.append(parts[1])

    binary_data = content[binary_pos + len(binary_marker):]
    point_size = n_vars * 8
    n_points = min(n_points, len(binary_data) // point_size)

    data = np.zeros((n_points, n_vars), dtype=np.float64)
    for i in range(n_points):
        offset = i * point_size
        for j in range(n_vars):
            val_bytes = binary_data[offset + j*8 : offset + (j+1)*8]
            if len(val_bytes) == 8:
                data[i, j] = struct.unpack('d', val_bytes)[0]

    return {name: data[:, i] for i, name in enumerate(variables)}


def get_config_hash(config: BenchmarkConfig, simulator: str) -> str:
    """Compute hash of config parameters relevant for caching."""
    # Include key parameters that affect simulation output
    params = {
        'simulator': simulator,
        't_stop': config.t_stop,
        'dt': config.dt,
        'current_source': config.current_source,
        'sim_file_mtime': config.vacask_sim.stat().st_mtime if simulator == 'vacask'
        else config.ngspice_sim.stat().st_mtime,
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
    """Run VACASK simulator and return path to raw file."""
    import shutil
    import tempfile

    sim_dir = config.vacask_sim.parent
    raw_file = output_dir / f'{benchmark_key}_vacask.raw'
    stamp_file = output_dir / f'{benchmark_key}_vacask.stamp'

    # Check cache
    config_hash = get_config_hash(config, 'vacask')
    if check_cache(raw_file, stamp_file, config_hash):
        logger.info(f"Using cached VACASK data: {raw_file}")
        return raw_file

    # Create modified sim file with our t_stop
    with open(config.vacask_sim) as f:
        sim_content = f.read()

    # Modify the analysis line
    modified = re.sub(
        r'(analysis\s+\w+\s+tran\s+.*?stop=)[^\s]+',
        f'\\g<1>{config.t_stop:.2e}',
        sim_content
    )

    # Add current save directive if current_source is specified
    if config.current_source:
        # Insert save i(source) before the analysis line
        modified = re.sub(
            r'(analysis\s+)',
            f'  save i({config.current_source})\n\\1',
            modified
        )

    temp_sim = sim_dir / 'plot_temp.sim'
    with open(temp_sim, 'w') as f:
        f.write(modified)

    try:
        logger.info(f"Running VACASK ({config.name})...")
        start = time.perf_counter()

        # Use utility function which handles PYTHONPATH and venv PATH
        temp_output = Path(tempfile.mkdtemp(prefix="vacask_"))
        result_raw, error = run_vacask_util(temp_sim, output_dir=temp_output, timeout=600)
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

    finally:
        if temp_sim.exists():
            temp_sim.unlink()


def run_ngspice(config: BenchmarkConfig, output_dir: Path, benchmark_key: str) -> Optional[Path]:
    """Run ngspice and return path to raw file."""
    # Check if ngspice is available
    try:
        subprocess.run(['ngspice', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("ngspice not found - skipping")
        return None

    sim_dir = config.ngspice_sim.parent
    raw_file = output_dir / f'{benchmark_key}_ngspice.raw'
    stamp_file = output_dir / f'{benchmark_key}_ngspice.stamp'

    # Check cache
    config_hash = get_config_hash(config, 'ngspice')
    if check_cache(raw_file, stamp_file, config_hash):
        logger.info(f"Using cached ngspice data: {raw_file}")
        return raw_file

    # Read original sim file
    with open(config.ngspice_sim) as f:
        sim_content = f.read()

    # Copy OSDI files from vacask directory if needed
    # Look for pre_osdi directives and copy the referenced files
    import shutil
    osdi_files = re.findall(r'pre_osdi\s+(\S+)', sim_content)
    vacask_dir = config.vacask_sim.parent
    for osdi_file in osdi_files:
        osdi_src = vacask_dir / osdi_file
        osdi_dst = sim_dir / osdi_file
        if osdi_src.exists() and not osdi_dst.exists():
            shutil.copy(osdi_src, osdi_dst)
            logger.info(f"Copied {osdi_file} from vacask to ngspice directory")

    # Modify tran command to use our t_stop and write rawfile
    # ngspice tran format: tran tstep tstop [tstart [tmax]] [uic]
    modified = re.sub(
        r'tran\s+[\d.]+[a-z]*\s+[\d.]+[a-z]*',
        f'tran {config.dt:.2e} {config.t_stop:.2e}',
        sim_content
    )

    # Add wrdata command before quit
    if 'wrdata' not in modified:
        modified = modified.replace(
            'quit',
            f'write {raw_file.name}\n  quit'
        )

    temp_sim = sim_dir / 'plot_temp.sp'
    with open(temp_sim, 'w') as f:
        f.write(modified)

    try:
        logger.info(f"Running ngspice ({config.name})...")
        start = time.perf_counter()
        result = subprocess.run(
            ['ngspice', '-b', 'plot_temp.sp'],
            cwd=sim_dir,
            capture_output=True,
            text=True,
            timeout=600
        )
        elapsed = time.perf_counter() - start

        # ngspice writes to cwd
        ng_raw = sim_dir / raw_file.name
        if ng_raw.exists():
            import shutil
            shutil.move(ng_raw, raw_file)
            write_stamp(stamp_file, config_hash)
            logger.info(f"ngspice completed in {elapsed:.1f}s -> {raw_file}")
            return raw_file
        else:
            # Check stderr for errors
            if result.returncode != 0:
                logger.error(f"ngspice failed: {result.stderr[:500]}")
            else:
                logger.warning("ngspice did not produce raw file")
            return None

    except subprocess.TimeoutExpired:
        logger.error("ngspice timed out")
        return None
    finally:
        if temp_sim.exists():
            temp_sim.unlink()


def run_jax_spice(config: BenchmarkConfig) -> Tuple[np.ndarray, Dict, Dict]:
    """Run JAX-SPICE Full MNA and return (times, voltages, currents)."""
    logger.info(f"Running JAX-SPICE Full MNA ({config.name})...")

    runner = CircuitEngine(config.vacask_sim)
    runner.parse()

    adaptive_config = AdaptiveConfig(max_dt=config.max_dt, min_dt=1e-15)
    full_mna = FullMNAStrategy(runner, use_sparse=config.use_sparse, config=adaptive_config)

    logger.info("Warmup...")
    _ = full_mna.warmup(dt=config.dt)

    logger.info(f"Running simulation (t_stop={config.t_stop:.2e}s)...")
    times_mna, V_out, stats_mna = full_mna.run(t_stop=config.t_stop, dt=config.dt)

    t_mna, voltages_mna, currents_mna = extract_results(times_mna, V_out, stats_mna)
    logger.info(f"JAX-SPICE completed: {len(t_mna)} points")

    return t_mna, voltages_mna, currents_mna


def compute_didt(t: np.ndarray, current: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute dI/dt from time and current arrays."""
    dt = np.diff(t)
    dI = np.diff(current)
    # Avoid division by zero
    dt = np.where(dt == 0, 1e-20, dt)
    dIdt = dI / dt
    t_mid = t[:-1] + dt / 2
    return t_mid, dIdt


def plot_comparison(config: BenchmarkConfig, vacask_data: Optional[Dict],
                    ngspice_data: Optional[Dict], jax_data: Tuple,
                    output_file: Path):
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

    t_start, t_end = config.plot_window
    # Use picoseconds for very short simulations
    time_scale = 1e12 if t_end < 1e-9 else 1e9
    time_unit = 'ps' if time_scale == 1e12 else 'ns'

    c_vac = 'blue'
    panel_idx = 0

    # Get time masks
    mask_mna = (t_mna >= t_start) & (t_mna <= t_end)
    if vacask_data:
        t_vac = vacask_data['time']
        mask_vac = (t_vac >= t_start) & (t_vac <= t_end)

    # Panel: Input A (if configured)
    if config.input_nodes_a:
        ax = axes[panel_idx]
        # Plot each input bit with offset for visibility
        for i, node in enumerate(config.input_nodes_a):
            offset = i * 1.5  # Offset each bit for visibility
            if vacask_data and node in vacask_data:
                ax.plot(t_vac[mask_vac] * time_scale, vacask_data[node][mask_vac] + offset,
                        lw=0.8, alpha=0.8, label=node if i < 4 else None)
        ax.set_ylabel('Input A [V + offset]', fontsize=11)
        ax.set_title(f'{config.name}: Input Bus A (a0-a15)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if config.input_nodes_a[:4]:
            ax.legend(loc='upper right', ncol=4, fontsize=8)
        panel_idx += 1

    # Panel: Input B (if configured)
    if config.input_nodes_b:
        ax = axes[panel_idx]
        for i, node in enumerate(config.input_nodes_b):
            offset = i * 1.5
            if vacask_data and node in vacask_data:
                ax.plot(t_vac[mask_vac] * time_scale, vacask_data[node][mask_vac] + offset,
                        lw=0.8, alpha=0.8, label=node if i < 4 else None)
        ax.set_ylabel('Input B [V + offset]', fontsize=11)
        ax.set_title('Input Bus B (b0-b15)', fontsize=12)
        ax.grid(True, alpha=0.3)
        if config.input_nodes_b[:4]:
            ax.legend(loc='upper right', ncol=4, fontsize=8)
        panel_idx += 1

    # Panel: Output Voltages
    ax = axes[panel_idx]
    for v_node in config.voltage_nodes:
        # VACASK uses short names (p0), JAX uses full names (top.p0)
        vac_node = v_node.split('.')[-1]  # Get short name for VACASK

        # Plot VACASK data
        if vacask_data and vac_node in vacask_data:
            ax.plot(t_vac[mask_vac] * time_scale, vacask_data[vac_node][mask_vac],
                    c_vac, lw=1.5, label=f'VACASK V({vac_node})', alpha=0.9)

        # Plot JAX-SPICE data - try both full name and short name
        jax_voltage = voltages_mna.get(v_node) or voltages_mna.get(vac_node)
        if jax_voltage is not None:
            ax.plot(t_mna[mask_mna] * time_scale, jax_voltage[mask_mna],
                    'red', lw=1.5, label=f'JAX V({vac_node})', alpha=0.9, linestyle=':')

    ax.set_ylabel('Output Voltage [V]', fontsize=11)
    ax.set_title(f'Output Bits ({", ".join(config.voltage_nodes[:3])}...)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    panel_idx += 1

    # Panel: Current
    ax = axes[panel_idx]
    I_vac = None
    if vacask_data:
        I_vac = vacask_data.get(f'{config.current_source}:flow(br)')
        if I_vac is not None:
            ax.plot(t_vac[mask_vac] * time_scale, I_vac[mask_vac] * 1e6, c_vac, lw=1.5,
                    label='VACASK', alpha=0.9)

    I_mna = currents_mna.get(config.current_source)
    if I_mna is not None:
        ax.plot(t_mna[mask_mna] * time_scale, I_mna[mask_mna] * 1e6, 'red', lw=1.5,
                label='JAX-SPICE', alpha=0.9, linestyle=':')

    ax.set_ylabel(f'I({config.current_source}) [µA]', fontsize=11)
    ax.set_xlabel(f'Time [{time_unit}]', fontsize=11)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Three-way simulator comparison')
    parser.add_argument('--benchmark', choices=list(BENCHMARKS.keys()), default='ring',
                        help='Benchmark to run (default: ring)')
    parser.add_argument('--skip-build', action='store_true',
                        help='Skip running VACASK/ngspice, use existing data')
    parser.add_argument('--output-dir', type=Path, default=Path('.'),
                        help='Output directory for plots and data')
    args = parser.parse_args()

    config = BENCHMARKS[args.benchmark]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    vacask_data = None
    ngspice_data = None

    if not args.skip_build:
        # Run VACASK
        vacask_raw = run_vacask(config, output_dir, args.benchmark)
        if vacask_raw and vacask_raw.exists():
            vacask_data = read_spice_raw(vacask_raw)
            logger.info(f"VACASK data: {list(vacask_data.keys())[:10]}...")

        # Run ngspice
        ngspice_raw = run_ngspice(config, output_dir, args.benchmark)
        if ngspice_raw and ngspice_raw.exists():
            ngspice_data = read_spice_raw(ngspice_raw)
            logger.info(f"ngspice data: {list(ngspice_data.keys())[:10]}...")
    else:
        # Try to load existing data
        vacask_raw = output_dir / f'{args.benchmark}_vacask.raw'
        ngspice_raw = output_dir / f'{args.benchmark}_ngspice.raw'
        if vacask_raw.exists():
            vacask_data = read_spice_raw(vacask_raw)
        if ngspice_raw.exists():
            ngspice_data = read_spice_raw(ngspice_raw)

    # Always run JAX-SPICE
    jax_data = run_jax_spice(config)

    # Plot comparison
    output_file = output_dir / f'{args.benchmark}_three_way_comparison.png'
    plot_comparison(config, vacask_data, ngspice_data, jax_data, output_file)

    # Print metrics
    logger.info("\n" + "=" * 70)
    logger.info(f"Comparison complete: {config.name}")
    logger.info("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
