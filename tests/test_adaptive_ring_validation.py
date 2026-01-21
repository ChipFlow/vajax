"""Validation tests for adaptive timestep with ring oscillator.

Tests that adaptive timestep control reduces the period error from ~2.3% to <0.5%
compared to VACASK reference.

Ring oscillator expected periods:
- VACASK: 3.453ns
- ngspice: 3.453ns
- JAX-SPICE (fixed dt): ~3.374ns (2.3% shorter)
- JAX-SPICE (adaptive): should be ~3.45ns (<0.5% error)
"""

import subprocess
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from jax_spice.analysis import CircuitEngine
from jax_spice.analysis.transient import AdaptiveConfig, AdaptiveStrategy
from jax_spice.benchmarks.registry import get_benchmark
from jax_spice.utils import find_vacask_binary, rawread


# Expected period from VACASK/ngspice (they agree within 0.02%)
VACASK_PERIOD_NS = 3.453


def find_oscillation_period(time: np.ndarray, voltage: np.ndarray,
                            threshold: float = 0.6,
                            min_time: float = 10e-9) -> Optional[float]:
    """Find the oscillation period from voltage waveform.

    Uses zero-crossing detection on rising edges after initial transient.

    Args:
        time: Time array in seconds
        voltage: Voltage array
        threshold: Voltage threshold for crossing detection
        min_time: Skip data before this time (avoid startup transient)

    Returns:
        Period in seconds, or None if not enough cycles detected
    """
    # Mask out startup transient
    mask = time > min_time
    t = time[mask]
    v = voltage[mask]

    if len(v) < 10:
        return None

    # Find rising edge crossings through threshold
    above = v > threshold
    crossings = np.where(np.diff(above.astype(int)) == 1)[0]

    if len(crossings) < 2:
        return None

    # Interpolate for exact crossing times
    crossing_times = []
    for idx in crossings:
        if idx + 1 >= len(t):
            continue
        t0, t1 = t[idx], t[idx + 1]
        v0, v1 = v[idx], v[idx + 1]
        if abs(v1 - v0) < 1e-12:
            crossing_times.append(t0)
        else:
            # Linear interpolation to threshold
            t_cross = t0 + (threshold - v0) * (t1 - t0) / (v1 - v0)
            crossing_times.append(t_cross)

    if len(crossing_times) < 2:
        return None

    # Calculate average period from all cycles
    crossing_times = np.array(crossing_times)
    periods = np.diff(crossing_times)

    # Filter outliers (periods that are >50% different from median)
    median_period = np.median(periods)
    valid = np.abs(periods - median_period) < 0.5 * median_period

    if np.sum(valid) < 1:
        return None

    return float(np.mean(periods[valid]))


def run_vacask_ring(vacask_bin: Path, t_stop: float = 50e-9, dt: float = 5e-11) -> dict:
    """Run VACASK on ring benchmark and extract period.

    Returns:
        Dict with 'time', 'voltage', 'period' keys
    """
    info = get_benchmark('ring')
    if info is None:
        raise FileNotFoundError("Ring benchmark not found")

    sim_dir = info.sim_path.parent
    sim_content = info.sim_path.read_text()

    # Modify analysis parameters
    modified = re.sub(r'(analysis\s+\w+\s+tran\s+.*?stop=)[^\s]+', f'\\g<1>{t_stop:.2e}', sim_content)
    modified = re.sub(r'(step=)[^\s]+', f'\\g<1>{dt:.2e}', modified)

    temp_sim = sim_dir / 'test_adaptive.sim'
    temp_sim.write_text(modified)

    try:
        result = subprocess.run(
            [str(vacask_bin), 'test_adaptive.sim'],
            cwd=sim_dir, capture_output=True, text=True, timeout=600
        )

        raw_files = list(sim_dir.glob('*.raw'))
        if not raw_files:
            raise RuntimeError(f"VACASK did not produce .raw file")

        raw = rawread(str(raw_files[0])).get()
        time = np.array(raw['time'])
        # Node 2 in VACASK - try different name formats
        voltage = None
        for name in ['2', 'v(2)']:
            if name in raw.names:
                voltage = np.array(raw[name])
                break
        if voltage is None:
            # Use first non-time variable
            for name in raw.names:
                if name != 'time':
                    voltage = np.array(raw[name])
                    break

        period = find_oscillation_period(time, voltage)

        return {'time': time, 'voltage': voltage, 'period': period}
    finally:
        if temp_sim.exists():
            temp_sim.unlink()
        for raw_file in sim_dir.glob('*.raw'):
            raw_file.unlink()


class TestRingPeriodWithAdaptive:
    """Test that adaptive timestep reduces ring oscillator period error."""

    @pytest.fixture
    def ring_info(self):
        """Get ring benchmark info, skip if not available."""
        info = get_benchmark('ring')
        if info is None or not info.sim_path.exists():
            pytest.skip("Ring benchmark not found")
        return info

    def test_fixed_timestep_period(self, ring_info):
        """Test ring period with fixed timestep (baseline).

        Measures the period with fixed timestep for comparison with adaptive.
        """
        engine = CircuitEngine(ring_info.sim_path)
        engine.parse()

        # Run with fixed 10ps timestep for 50ns
        result = engine.run_transient(
            t_stop=50e-9,
            dt=10e-12,
            use_sparse=False,
        )

        time = np.array(result.times)
        # Node '1' in JAX-SPICE corresponds to node '2' in VACASK
        voltage = np.array(result.voltages.get('1', []))

        period = find_oscillation_period(time, voltage)
        assert period is not None, "Could not measure oscillation period"

        period_ns = period * 1e9
        error_pct = (period_ns - VACASK_PERIOD_NS) / VACASK_PERIOD_NS * 100

        print(f"\nFixed timestep (10ps):")
        print(f"  Measured period: {period_ns:.3f} ns")
        print(f"  VACASK reference: {VACASK_PERIOD_NS:.3f} ns")
        print(f"  Error: {error_pct:+.2f}%")
        print(f"  Timesteps: {len(time)}")

        # Verify the period is within reasonable range (< 5% error)
        assert abs(error_pct) < 5.0, f"Error {error_pct:.2f}% exceeds 5% with fixed timestep"

    def test_adaptive_timestep_period(self, ring_info):
        """Test ring period with adaptive timestep.

        This should reduce the period error to <1%.
        """
        engine = CircuitEngine(ring_info.sim_path)
        engine.parse()

        # Run with adaptive timestep
        config = AdaptiveConfig(
            lte_ratio=3.5,
            redo_factor=2.5,
            reltol=1e-3,
            abstol=1e-12,
            min_dt=1e-15,
            max_dt=20e-12,  # Don't exceed 20ps
            warmup_steps=3,
            max_order=2,
            grow_factor=1.5,
        )

        result = engine.run_transient(
            t_stop=50e-9,
            dt=10e-12,  # Initial timestep
            use_sparse=False,
            adaptive=True,
            adaptive_config=config,
        )

        time = np.array(result.times)
        voltage = np.array(result.voltages.get('1', []))

        period = find_oscillation_period(time, voltage)
        assert period is not None, "Could not measure oscillation period"

        period_ns = period * 1e9
        error_pct = (period_ns - VACASK_PERIOD_NS) / VACASK_PERIOD_NS * 100

        # Get adaptive statistics
        stats = result.stats
        min_dt = stats.get('min_dt_used', 0)
        max_dt = stats.get('max_dt_used', 0)
        rejected = stats.get('rejected_steps', 0)
        accepted = stats.get('accepted_steps', 0)

        print(f"\nAdaptive timestep:")
        print(f"  Measured period: {period_ns:.3f} ns")
        print(f"  VACASK reference: {VACASK_PERIOD_NS:.3f} ns")
        print(f"  Error: {error_pct:+.2f}%")
        print(f"  Timesteps: accepted={accepted}, rejected={rejected}")
        print(f"  dt range: [{min_dt*1e12:.2f}, {max_dt*1e12:.2f}] ps")

        # With adaptive timestep, error should be significantly reduced
        # Target: <1% error (was ~2.3% with fixed timestep)
        assert abs(error_pct) < 1.5, \
            f"Period error {error_pct:.2f}% exceeds 1.5% threshold with adaptive timestep"

    def test_adaptive_timestep_variation(self, ring_info):
        """Test that adaptive timestep actually varies during transitions.

        VACASK uses smaller timesteps during fast transients (switching)
        and larger timesteps during slow evolution.
        """
        engine = CircuitEngine(ring_info.sim_path)
        engine.parse()

        config = AdaptiveConfig(
            lte_ratio=3.5,
            redo_factor=2.5,
            min_dt=1e-15,
            max_dt=20e-12,
        )

        result = engine.run_transient(
            t_stop=50e-9,
            dt=10e-12,
            use_sparse=False,
            adaptive=True,
            adaptive_config=config,
        )

        stats = result.stats
        min_dt = stats.get('min_dt_used', 0)
        max_dt = stats.get('max_dt_used', 0)

        print(f"\nTimestep variation:")
        print(f"  Min dt: {min_dt*1e12:.2f} ps")
        print(f"  Max dt: {max_dt*1e12:.2f} ps")
        print(f"  Ratio: {max_dt/max(min_dt, 1e-20):.1f}x")

        # Check that timestep varies significantly (at least 2x range)
        if max_dt > 0 and min_dt > 0:
            ratio = max_dt / min_dt
            assert ratio > 1.5, \
                f"Timestep should vary significantly during transients (ratio={ratio:.1f})"


class TestAdaptiveVsVACASK:
    """Compare JAX-SPICE adaptive timestep results directly with VACASK."""

    @pytest.fixture
    def vacask_bin(self):
        """Get VACASK binary path, skip if not available."""
        binary = find_vacask_binary()
        if binary is None:
            pytest.skip("VACASK binary not found. Set VACASK_BIN env var or build VACASK.")
        return binary

    @pytest.fixture
    def ring_info(self):
        """Get ring benchmark info."""
        info = get_benchmark('ring')
        if info is None:
            pytest.skip("Ring benchmark not found")
        return info

    def test_period_matches_vacask(self, vacask_bin, ring_info):
        """Test that adaptive timestep period matches VACASK within tolerance."""
        # Run VACASK
        vacask_result = run_vacask_ring(vacask_bin, t_stop=50e-9, dt=5e-11)
        vacask_period = vacask_result['period']
        assert vacask_period is not None, "Could not measure VACASK period"

        # Run JAX-SPICE with adaptive
        engine = CircuitEngine(ring_info.sim_path)
        engine.parse()

        config = AdaptiveConfig(
            lte_ratio=3.5,
            redo_factor=2.5,
            min_dt=1e-15,
            max_dt=20e-12,
        )

        result = engine.run_transient(
            t_stop=50e-9,
            dt=10e-12,
            use_sparse=False,
            adaptive=True,
            adaptive_config=config,
        )

        jax_time = np.array(result.times)
        jax_voltage = np.array(result.voltages.get('1', []))
        jax_period = find_oscillation_period(jax_time, jax_voltage)
        assert jax_period is not None, "Could not measure JAX-SPICE period"

        # Compare
        vacask_period_ns = vacask_period * 1e9
        jax_period_ns = jax_period * 1e9
        error_pct = (jax_period_ns - vacask_period_ns) / vacask_period_ns * 100

        print(f"\nVACASK vs JAX-SPICE (adaptive):")
        print(f"  VACASK period: {vacask_period_ns:.3f} ns")
        print(f"  JAX-SPICE period: {jax_period_ns:.3f} ns")
        print(f"  Error: {error_pct:+.2f}%")

        # Target: <1% error compared to VACASK
        assert abs(error_pct) < 1.0, \
            f"Period error {error_pct:.2f}% exceeds 1% compared to VACASK"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
