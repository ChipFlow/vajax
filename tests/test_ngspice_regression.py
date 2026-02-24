"""ngspice regression test runner.

Runs selected tests from vendor/ngspice/tests/ against VA-JAX.
Compatible tests must:
- Use devices supported by VA-JAX (R, C, L, D, V, I, M via OpenVAF)
- Use transient analysis (.tran)
- Not require ngspice-specific features (xspice, etc.)

The test workflow:
1. Run ngspice to generate reference raw file
2. Convert ngspice netlist to VACASK format
3. Run VA-JAX simulator
4. Compare waveforms with tolerance
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
from jax import Array

from vajax.analysis.engine import CircuitEngine
from vajax.io.ngspice_out_reader import read_reference_file
from vajax.netlist_converter import Converter
from vajax.utils import rawread
from vajax.utils.ngspice import (
    find_ngspice_binary,
    parse_control_section,
    run_ngspice,
)
from vajax.utils.waveform_compare import WaveformComparison, compare_waveforms
from tests.ngspice_test_registry import (
    PROJECT_ROOT,
    NgspiceTestCase,
    discover_ngspice_tests,
)

# Get ALL discovered tests at module load time for parametrization
# No device filtering - we want to see how we do against the full suite
ALL_TESTS = {t.name: t for t in discover_ngspice_tests()}


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
        ("meg", 1e6),
        ("mil", 25.4e-6),
        ("ms", 1e-3),
        ("us", 1e-6),
        ("ns", 1e-9),
        ("ps", 1e-12),
        ("fs", 1e-15),
        ("f", 1e-15),
        ("p", 1e-12),
        ("n", 1e-9),
        ("u", 1e-6),
        ("m", 1e-3),
        ("k", 1e3),
        ("g", 1e9),
        ("t", 1e12),
    ]

    for suffix, mult in suffixes:
        if s.endswith(suffix):
            return float(s[: -len(suffix)]) * mult
    return float(s)


def convert_ngspice_to_vacask(netlist_path: Path, output_path: Path) -> None:
    """Convert ngspice netlist to VACASK .sim format.

    Args:
        netlist_path: Input ngspice .cir file
        output_path: Output VACASK .sim file
    """
    from vajax.netlist_converter.ng2vclib.converter import Converter
    from vajax.netlist_converter.ng2vclib.dfl import default_config

    cfg = default_config()
    cfg["sourcepath"] = [str(netlist_path.parent)]

    converter = Converter(cfg, dialect="ngspice")
    converter.convert(str(netlist_path), str(output_path))


def load_reference_data(
    test_case: NgspiceTestCase,
) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[str]]:
    """Load reference data from .out or .standard file.

    Args:
        test_case: Test case specification

    Returns:
        (results_dict, error) where results_dict maps signal name to array
    """
    if test_case.reference_path is None:
        return None, "No reference file available"

    if not test_case.reference_path.exists():
        return None, f"Reference file not found: {test_case.reference_path}"

    try:
        columns, data = read_reference_file(test_case.reference_path)
        if not data:
            return None, "Empty reference file"
        return data, None
    except Exception as e:
        return None, f"Failed to parse reference file: {e}"


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
    """Run VA-JAX on converted netlist.

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
            t_stop = parse_si_value(control_params.get("stop", "1m"))
            dt = parse_si_value(control_params.get("step", "1u"))

            # Ensure we have reasonable defaults
            if dt <= 0 or dt > t_stop:
                dt = t_stop / 1000

            engine.prepare(t_stop=t_stop, dt=dt)
            result = engine.run_transient()

            results: Dict[str, Array] = {"time": result.times}
            for node_name, voltage in result.voltages.items():
                # Store as both v(node) and node for flexibility
                results[f"v({node_name})"] = voltage
                results[node_name] = voltage

            return results, None

        except Exception as e:
            return None, f"Simulation failed: {e}"


def compare_results(
    ref_results: Dict[str, np.ndarray],
    jaxspice_results: Dict[str, Array],
    signals: List[str],
    rtol: float = 0.05,
    atol: float = 1e-9,
) -> List[WaveformComparison]:
    """Compare reference and VA-JAX results.

    Args:
        ref_results: Reference simulation results (from .out file or ngspice)
        jaxspice_results: VA-JAX simulation results
        signals: Signal names to compare (e.g., 'v(1)', 'v1#branch')
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        List of WaveformComparison results
    """
    comparisons = []

    # Get time vectors
    ref_time = None
    for key in ["time", "TIME", "Time"]:
        if key in ref_results:
            ref_time = ref_results[key]
            break

    jax_time = np.array(jaxspice_results.get("time", []))

    if ref_time is None or len(jax_time) == 0:
        return comparisons

    for sig in signals:
        sig_lower = sig.lower()

        # Find matching signal in reference results
        ref_signal = None
        for key in ref_results:
            if key.lower() == sig_lower:
                ref_signal = ref_results[key]
                break

        if ref_signal is None:
            continue

        # Find matching signal in VA-JAX results
        jax_signal = None

        # Handle voltage signals: v(node) -> look up node in voltages
        if sig_lower.startswith("v(") and sig_lower.endswith(")"):
            node = sig_lower[2:-1]  # Extract node name
            for key in jaxspice_results:
                if key.lower() == node:
                    jax_signal = np.array(jaxspice_results[key])
                    break

        # Handle branch currents: source#branch
        # VA-JAX doesn't currently support branch currents, skip these
        elif "#branch" in sig_lower:
            # Can't compare branch currents - VA-JAX doesn't compute them
            continue

        if jax_signal is None:
            continue

        # Take real part (ngspice may return complex for some analyses)
        ref_signal = np.real(ref_signal)

        # Interpolate VA-JAX to reference timepoints
        ref_time_real = np.real(ref_time)
        if len(ref_time_real) != len(jax_time):
            jax_interp = np.interp(ref_time_real, jax_time, jax_signal)
        else:
            jax_interp = jax_signal

        comparison = compare_waveforms(
            ref_signal,
            jax_interp,
            name=sig,
            abs_tol=atol,
            rel_tol=rtol,
        )
        comparisons.append(comparison)

    return comparisons


class TestNgspiceRegression:
    """ngspice regression tests against VA-JAX.

    Auto-discovers all tests from vendor/ngspice/tests/.
    Compares against .out reference files when available,
    falls back to running ngspice if no reference file exists.
    """

    @pytest.fixture
    def ngspice_bin(self):
        """Get ngspice binary, returns None if not available."""
        return find_ngspice_binary()

    # Device types currently supported by VA-JAX
    SUPPORTED_DEVICES = {"resistor", "capacitor", "diode", "vsource", "isource"}

    # Device types that are WIP or planned
    WIP_DEVICES = {"mosfet", "inductor", "bjt", "jfet", "subckt"}

    # Device types that require specific models we don't have
    UNSUPPORTED_DEVICES = {"vcvs", "cccs", "vccs", "ccvs", "bsource", "tline"}

    @pytest.mark.parametrize(
        "test_name",
        list(ALL_TESTS.keys()),
        ids=list(ALL_TESTS.keys()),
    )
    def test_ngspice(self, ngspice_bin, test_name):
        """Run auto-discovered ngspice test against VA-JAX."""
        test_case = ALL_TESTS[test_name]

        if not test_case.netlist_path.exists():
            pytest.skip(f"Netlist not found: {test_case.netlist_path}")

        # Check for unsupported device types
        unsupported = test_case.device_types & self.UNSUPPORTED_DEVICES
        if unsupported:
            pytest.xfail(f"Unsupported device types: {unsupported}")

        # Check for WIP device types (mosfet needs specific model support)
        wip = test_case.device_types & self.WIP_DEVICES
        if wip:
            # mosfet is only supported via PSP103 OpenVAF model, not ngspice builtin models
            if "mosfet" in wip:
                pytest.xfail("ngspice builtin MOSFET models not supported (need OpenVAF model)")
            else:
                pytest.xfail(f"WIP device types: {wip}")

        # Skip non-transient tests for now
        if test_case.analysis_type != "tran":
            pytest.skip(f"Analysis type {test_case.analysis_type} not yet supported")

        # Try to load reference data from .out file first
        ref_results, ref_error = load_reference_data(test_case)

        # If no reference file, try running ngspice
        if ref_error and ngspice_bin:
            ref_results, ref_error = run_ngspice_reference(test_case, ngspice_bin)

        if ref_error:
            pytest.skip(f"No reference data: {ref_error}")

        # Parse control section for simulation params
        control_params = parse_control_section(test_case.netlist_path)

        # Run VA-JAX
        jax_results, jax_error = run_jaxspice(
            test_case.netlist_path,
            control_params,
        )
        if jax_error:
            # Conversion failures are expected for unsupported netlist features
            if "Conversion failed" in jax_error:
                pytest.xfail(f"Netlist conversion not supported: {jax_error}")
            pytest.fail(f"VA-JAX failed: {jax_error}")

        # Compare results
        comparisons = compare_results(
            ref_results,
            jax_results,
            test_case.expected_nodes,
            rtol=test_case.rtol,
            atol=test_case.atol,
        )

        # If no comparisons could be made (e.g., only branch currents), skip
        if not comparisons:
            pytest.skip("No comparable signals (branch currents not supported)")

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
            pytest.skip("ngspice binary not found. Install ngspice or set NGSPICE_BIN.")
        return binary

    @pytest.mark.xfail(
        reason="Pulse delay parameter not handled correctly in ngspice->VACASK conversion"
    )
    def test_rc_benchmark(self, ngspice_bin):
        """Compare VACASK RC benchmark against ngspice reference."""
        ngspice_rc = (
            PROJECT_ROOT / "vendor" / "VACASK" / "benchmark" / "rc" / "ngspice" / "runme.sim"
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
            ng_time = np.array(ng_raw["time"])

            # Find the voltage node
            ng_v2 = None
            for name in ng_raw.names:
                if "2" in name.lower() or "out" in name.lower():
                    ng_v2 = np.array(ng_raw[name])
                    break

            if ng_v2 is None:
                pytest.skip("Could not find output node in ngspice results")

        # Convert ngspice netlist to VACASK format and run VA-JAX
        with tempfile.TemporaryDirectory() as convert_tmpdir:
            converted_path = Path(convert_tmpdir) / "converted.sim"

            # Convert ngspice netlist to VACASK format
            converter = Converter()
            converter.convert(str(ngspice_rc), str(converted_path))

            if not converted_path.exists():
                pytest.skip("Netlist conversion failed")

            engine = CircuitEngine(converted_path)
            engine.parse()

            t_stop = parse_si_value(control_params.get("stop", "1"))
            dt = parse_si_value(control_params.get("step", "1u"))
            if dt <= 0:
                dt = t_stop / 1000

            # Limit simulation time for CI performance while still testing accuracy
            # RC circuit has τ=1ms, so 20ms captures key dynamics (rise + fall)
            max_sim_time = 20e-3
            if t_stop > max_sim_time:
                t_stop = max_sim_time

            engine.prepare(t_stop=t_stop, dt=dt)
            result = engine.run_transient()

            # Get output voltage - try various node names
            jax_v2 = None
            for node in ["2", "out", "OUT"]:
                if node in result.voltages:
                    jax_v2 = np.array(result.voltages[node])
                    break

            if jax_v2 is None:
                # Just use the first non-ground node
                for node, voltage in result.voltages.items():
                    if node != "0":
                        jax_v2 = np.array(voltage)
                        break

            if jax_v2 is None:
                pytest.fail("No voltage output found in VA-JAX results")

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
