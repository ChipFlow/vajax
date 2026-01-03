"""ngspice regression test runner.

Runs selected tests from vendor/ngspice/tests/ against JAX-SPICE.
Compatible tests must:
- Use devices supported by JAX-SPICE (R, C, L, D, V, I, M via OpenVAF)
- Use transient analysis (.tran)
- Not require ngspice-specific features (xspice, etc.)

The test workflow:
1. Run ngspice to generate reference raw file
2. Convert ngspice netlist to VACASK format
3. Run JAX-SPICE simulator
4. Compare waveforms with tolerance
"""

import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array
import numpy as np
import pytest

from jax_spice.analysis.engine import CircuitEngine
from jax_spice.utils import rawread
from jax_spice.utils.ngspice import (
    find_ngspice_binary,
    run_ngspice,
    parse_control_section,
)
from jax_spice.utils.waveform_compare import compare_waveforms, WaveformComparison
from jax_spice.netlist_converter import Converter
from tests.ngspice_test_registry import (
    NgspiceTestCase,
    CURATED_TESTS,
    get_compatible_tests,
    PROJECT_ROOT,
    NGSPICE_TESTS,
)


def parse_si_value(s: str) -> float:
    """Parse SPICE value with SI suffix.

    Args:
        s: String like "1m", "100n", "10k", etc.

    Returns:
        Float value with SI scaling applied
    """
    s = s.strip().lower()

    # Order matters - check longer suffixes first
    suffixes = [
        ('meg', 1e6),
        ('mil', 25.4e-6),
        ('ms', 1e-3),
        ('us', 1e-6),
        ('ns', 1e-9),
        ('ps', 1e-12),
        ('fs', 1e-15),
        ('f', 1e-15),
        ('p', 1e-12),
        ('n', 1e-9),
        ('u', 1e-6),
        ('m', 1e-3),
        ('k', 1e3),
        ('g', 1e9),
        ('t', 1e12),
    ]

    for suffix, mult in suffixes:
        if s.endswith(suffix):
            return float(s[:-len(suffix)]) * mult
    return float(s)


def convert_ngspice_to_vacask(netlist_path: Path, output_path: Path) -> None:
    """Convert ngspice netlist to VACASK .sim format.

    Args:
        netlist_path: Input ngspice .cir file
        output_path: Output VACASK .sim file
    """
    from jax_spice.netlist_converter.ng2vclib.converter import Converter
    from jax_spice.netlist_converter.ng2vclib.dfl import default_config

    cfg = default_config()
    cfg["sourcepath"] = [str(netlist_path.parent)]

    converter = Converter(cfg, dialect="ngspice")
    converter.convert(str(netlist_path), str(output_path))


def run_ngspice_reference(
    test_case: NgspiceTestCase,
    ngspice_bin: Path,
) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[str]]:
    """Run ngspice and return parsed results.

    Args:
        test_case: Test case specification
        ngspice_bin: Path to ngspice binary

    Returns:
        (results_dict, error) where results_dict maps signal name to array
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path, error = run_ngspice(
            test_case.netlist_path,
            output_dir=Path(tmpdir),
            ngspice_bin=ngspice_bin,
        )

        if error:
            return None, error

        try:
            raw = rawread(str(raw_path)).get()
            results = {name: np.array(raw[name]) for name in raw.names}
            return results, None
        except Exception as e:
            return None, f"Failed to parse raw file: {e}"


def run_jaxspice(
    netlist_path: Path,
    control_params: Dict,
) -> Tuple[Optional[Dict[str, Array]], Optional[str]]:
    """Run JAX-SPICE on converted netlist.

    Args:
        netlist_path: Path to ngspice netlist
        control_params: Parsed .control section parameters

    Returns:
        (results_dict, error) where results_dict maps signal name to array
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        sim_path = Path(tmpdir) / "test.sim"

        try:
            convert_ngspice_to_vacask(netlist_path, sim_path)
        except Exception as e:
            return None, f"Conversion failed: {e}"

        try:
            engine = CircuitEngine(sim_path)
            engine.parse()

            # Parse analysis parameters
            t_stop = parse_si_value(control_params.get('stop', '1m'))
            dt = parse_si_value(control_params.get('step', '1u'))

            # Ensure we have reasonable defaults
            if dt <= 0 or dt > t_stop:
                dt = t_stop / 1000

            max_steps = int(t_stop / dt) + 100

            result = engine.run_transient(t_stop=t_stop, dt=dt, max_steps=max_steps)

            results: Dict[str, Array] = {'time': result.times}
            for node_name, voltage in result.voltages.items():
                # Store as both v(node) and node for flexibility
                results[f'v({node_name})'] = voltage
                results[node_name] = voltage

            return results, None

        except Exception as e:
            return None, f"Simulation failed: {e}"


def compare_results(
    ngspice_results: Dict[str, np.ndarray],
    jaxspice_results: Dict[str, Array],
    nodes: List[str],
    rtol: float = 0.05,
    atol: float = 1e-9,
) -> List[WaveformComparison]:
    """Compare ngspice and JAX-SPICE results.

    Args:
        ngspice_results: ngspice simulation results
        jaxspice_results: JAX-SPICE simulation results
        nodes: Node names to compare
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        List of WaveformComparison results
    """
    comparisons = []

    # Get time vectors
    ng_time = None
    for key in ['time', 'TIME', 'Time']:
        if key in ngspice_results:
            ng_time = ngspice_results[key]
            break

    jax_time = np.array(jaxspice_results.get('time', []))

    if ng_time is None or len(jax_time) == 0:
        return comparisons

    for node in nodes:
        # Find matching signals in both results
        ng_signal = None
        jax_signal = None

        # Try various naming conventions for ngspice
        for key in ngspice_results:
            key_lower = key.lower()
            node_lower = node.lower()
            if (node_lower in key_lower or
                f'v({node_lower})' == key_lower or
                f'{node_lower}' == key_lower):
                ng_signal = ngspice_results[key]
                break

        # Try various naming conventions for JAX-SPICE
        for key in jaxspice_results:
            key_lower = key.lower()
            node_lower = node.lower()
            if (node_lower in key_lower or
                f'v({node_lower})' == key_lower or
                node_lower == key_lower):
                jax_signal = np.array(jaxspice_results[key])
                break

        if ng_signal is None or jax_signal is None:
            continue

        # Take real part (ngspice may return complex for some analyses)
        ng_signal = np.real(ng_signal)

        # Interpolate JAX-SPICE to ngspice timepoints
        ng_time_real = np.real(ng_time)
        if len(ng_time_real) != len(jax_time):
            jax_interp = np.interp(ng_time_real, jax_time, jax_signal)
        else:
            jax_interp = jax_signal

        comparison = compare_waveforms(
            ng_signal,
            jax_interp,
            name=f"v({node})",
            abs_tol=atol,
            rel_tol=rtol,
        )
        comparisons.append(comparison)

    return comparisons


class TestNgspiceRegression:
    """ngspice regression tests against JAX-SPICE."""

    @pytest.fixture
    def ngspice_bin(self):
        """Get ngspice binary, skip if not available."""
        binary = find_ngspice_binary()
        if binary is None:
            pytest.skip(
                "ngspice binary not found. Install ngspice or set NGSPICE_BIN."
            )
        return binary

    @pytest.mark.parametrize(
        "test_name",
        list(CURATED_TESTS.keys()),
        ids=list(CURATED_TESTS.keys()),
    )
    def test_curated(self, ngspice_bin, test_name):
        """Run curated ngspice test cases known to work with JAX-SPICE."""
        test_case = CURATED_TESTS[test_name]

        if test_case.skip:
            pytest.skip(test_case.skip_reason)

        if test_case.xfail:
            pytest.xfail(test_case.xfail_reason)

        if not test_case.netlist_path.exists():
            pytest.skip(f"Netlist not found: {test_case.netlist_path}")

        # Only run transient tests for now
        if test_case.analysis_type != 'tran':
            pytest.skip(f"Analysis type {test_case.analysis_type} not supported yet")

        # Parse control section
        control_params = parse_control_section(test_case.netlist_path)

        # Run ngspice
        ng_results, ng_error = run_ngspice_reference(test_case, ngspice_bin)
        if ng_error:
            pytest.skip(f"ngspice failed: {ng_error}")

        # Run JAX-SPICE
        jax_results, jax_error = run_jaxspice(
            test_case.netlist_path,
            control_params,
        )
        if jax_error:
            pytest.fail(f"JAX-SPICE failed: {jax_error}")

        # Compare results
        comparisons = compare_results(
            ng_results,
            jax_results,
            test_case.expected_nodes,
            rtol=test_case.rtol,
            atol=test_case.atol,
        )

        # Report results
        failed = [c for c in comparisons if not c.within_tolerance]
        if failed:
            msg = "Waveform mismatch:\n"
            for c in failed:
                msg += f"  {c}\n"
            pytest.fail(msg)


class TestVacaskBenchmarksWithNgspice:
    """Compare VACASK benchmarks against ngspice reference.

    Uses the existing VACASK benchmark circuits but runs ngspice
    on the ngspice version for reference comparison.
    """

    @pytest.fixture
    def ngspice_bin(self):
        """Get ngspice binary, skip if not available."""
        binary = find_ngspice_binary()
        if binary is None:
            pytest.skip(
                "ngspice binary not found. Install ngspice or set NGSPICE_BIN."
            )
        return binary

    def test_rc_benchmark(self, ngspice_bin):
        """Compare VACASK RC benchmark against ngspice reference."""
        ngspice_rc = (
            PROJECT_ROOT / "vendor" / "VACASK" / "benchmark" / "rc" /
            "ngspice" / "runme.sim"
        )

        if not ngspice_rc.exists():
            pytest.skip(f"ngspice RC benchmark not found: {ngspice_rc}")

        # Parse control params
        control_params = parse_control_section(ngspice_rc)

        # Run ngspice
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path, error = run_ngspice(ngspice_rc, Path(tmpdir), ngspice_bin)
            if error:
                pytest.skip(f"ngspice failed: {error}")

            ng_raw = rawread(str(raw_path)).get()
            ng_time = np.array(ng_raw['time'])

            # Find the voltage node
            ng_v2 = None
            for name in ng_raw.names:
                if '2' in name.lower() or 'out' in name.lower():
                    ng_v2 = np.array(ng_raw[name])
                    break

            if ng_v2 is None:
                pytest.skip("Could not find output node in ngspice results")

        # Convert ngspice netlist to VACASK format and run JAX-SPICE
        with tempfile.TemporaryDirectory() as convert_tmpdir:
            converted_path = Path(convert_tmpdir) / "converted.sim"

            # Convert ngspice netlist to VACASK format
            converter = Converter()
            converter.convert(str(ngspice_rc), str(converted_path))

            if not converted_path.exists():
                pytest.skip("Netlist conversion failed")

            engine = CircuitEngine(converted_path)
            engine.parse()

            t_stop = parse_si_value(control_params.get('stop', '1'))
            dt = parse_si_value(control_params.get('step', '1u'))
            if dt <= 0:
                dt = t_stop / 1000

            # Limit simulation time for CI performance while still testing accuracy
            # RC circuit has τ=1ms, so 20ms captures key dynamics (rise + fall)
            max_sim_time = 20e-3
            if t_stop > max_sim_time:
                t_stop = max_sim_time

            result = engine.run_transient(t_stop=t_stop, dt=dt)

            # Get output voltage - try various node names
            jax_v2 = None
            for node in ['2', 'out', 'OUT']:
                if node in result.voltages:
                    jax_v2 = np.array(result.voltages[node])
                    break

            if jax_v2 is None:
                # Just use the first non-ground node
                for node, voltage in result.voltages.items():
                    if node != '0':
                        jax_v2 = np.array(voltage)
                        break

            if jax_v2 is None:
                pytest.fail("No voltage output found in JAX-SPICE results")

            jax_time = np.array(result.times)

        # Interpolate to ngspice timepoints (limited to JAX simulation range)
        ng_time_real = np.real(ng_time)
        # Only compare timepoints within JAX simulation range
        mask = ng_time_real <= jax_time[-1]
        ng_time_trimmed = ng_time_real[mask]
        ng_v2_trimmed = np.real(ng_v2)[mask]
        jax_interp = np.interp(ng_time_trimmed, jax_time, jax_v2)

        comparison = compare_waveforms(
            ng_v2_trimmed,
            jax_interp,
            name="v(2)",
            rel_tol=0.05,  # 5% relative tolerance
            abs_tol=1e-3,  # 1mV absolute tolerance (for small signal comparison)
        )

        print(f"\nRC comparison: {comparison}")
        assert comparison.within_tolerance, f"RC mismatch: {comparison}"
