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
        t_stop=2e-11,
        dt=2e-15,
        max_dt=10e-14,
        plot_window=(0.0, 2e-9),
        voltage_nodes=['top.p0', 'top.p1'],  # Output bits
        current_source='vdd',
        use_sparse=True,
        icmode='uic',
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

    # Setup figure
    n_panels = 3 if vacask_data or ngspice_data else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=True)

    t_start, t_end = config.plot_window
    c_vac, c_ng, c_mna = 'blue', 'green', 'red'

    # Get first voltage node for plotting
    v_node = config.voltage_nodes[0]

    # Panel 1: Voltages
    ax1 = axes[0]
    mask_mna = (t_mna >= t_start) & (t_mna <= t_end)

    # Find voltage in JAX data (may have different naming)
    V_mna = None
    for key in voltages_mna:
        if v_node in key or key == v_node:
            V_mna = voltages_mna[key]
            break
    if V_mna is None and voltages_mna:
        # Use first available
        V_mna = list(voltages_mna.values())[0]
        v_node = list(voltages_mna.keys())[0]

    if V_mna is not None:
        ax1.plot(t_mna[mask_mna] * 1e9, V_mna[mask_mna], c_mna, lw=1.5,
                 label=f'JAX-SPICE V({v_node})', alpha=0.9, linestyle=':')

    if vacask_data:
        t_vac = vacask_data['time']
        V_vac = vacask_data.get(v_node) or vacask_data.get(v_node.split('.')[-1])
        if V_vac is not None:
            mask_vac = (t_vac >= t_start) & (t_vac <= t_end)
            ax1.plot(t_vac[mask_vac] * 1e9, V_vac[mask_vac], c_vac, lw=1.5,
                     label=f'VACASK V({v_node})', alpha=0.9)

    if ngspice_data:
        t_ng = ngspice_data['time']
        V_ng = ngspice_data.get(v_node) or ngspice_data.get(f'v({v_node})')
        if V_ng is not None:
            mask_ng = (t_ng >= t_start) & (t_ng <= t_end)
            ax1.plot(t_ng[mask_ng] * 1e9, V_ng[mask_ng], c_ng, lw=1.5,
                     label=f'ngspice V({v_node})', alpha=0.9, linestyle='--')

    ax1.set_ylabel('Voltage [V]', fontsize=11)
    ax1.legend(loc='upper right', ncol=3, fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{config.name}: VACASK vs ngspice vs JAX-SPICE', fontsize=12, fontweight='bold')

    # Panel 2: Current
    ax2 = axes[1]
    I_mna = currents_mna.get(config.current_source)

    if I_mna is not None:
        ax2.plot(t_mna[mask_mna] * 1e9, I_mna[mask_mna] * 1e6, c_mna, lw=1.5,
                 label='JAX-SPICE', alpha=0.9, linestyle=':')

    if vacask_data:
        I_vac = vacask_data.get(f'{config.current_source}:flow(br)')
        if I_vac is not None:
            ax2.plot(t_vac[mask_vac] * 1e9, I_vac[mask_vac] * 1e6, c_vac, lw=1.5,
                     label='VACASK', alpha=0.9)

    if ngspice_data:
        I_ng = ngspice_data.get(f'i({config.current_source})')
        if I_ng is not None:
            ax2.plot(t_ng[mask_ng] * 1e9, I_ng[mask_ng] * 1e6, c_ng, lw=1.5,
                     label='ngspice', alpha=0.9, linestyle='--')

    ax2.set_ylabel(f'I({config.current_source}) [µA]', fontsize=11)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: dI/dt (if we have current data)
    if n_panels > 2 and I_mna is not None:
        ax3 = axes[2]
        t_didt_mna, dIdt_mna = compute_didt(t_mna, I_mna)
        mask_didt_mna = (t_didt_mna >= t_start) & (t_didt_mna <= t_end)
        ax3.plot(t_didt_mna[mask_didt_mna] * 1e9, dIdt_mna[mask_didt_mna] * 1e-6, c_mna,
                 lw=1, label='JAX-SPICE', alpha=0.8, linestyle=':')

        if vacask_data and I_vac is not None:
            t_didt_vac, dIdt_vac = compute_didt(t_vac, I_vac)
            mask_didt_vac = (t_didt_vac >= t_start) & (t_didt_vac <= t_end)
            ax3.plot(t_didt_vac[mask_didt_vac] * 1e9, dIdt_vac[mask_didt_vac] * 1e-6, c_vac,
                     lw=1, label='VACASK', alpha=0.8)

        if ngspice_data and I_ng is not None:
            t_didt_ng, dIdt_ng = compute_didt(t_ng, I_ng)
            mask_didt_ng = (t_didt_ng >= t_start) & (t_didt_ng <= t_end)
            ax3.plot(t_didt_ng[mask_didt_ng] * 1e9, dIdt_ng[mask_didt_ng] * 1e-6, c_ng,
                     lw=1, label='ngspice', alpha=0.8, linestyle='--')

        ax3.set_ylabel('dI/dt [mA/ns]', fontsize=11)
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [ns]', fontsize=11)

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
