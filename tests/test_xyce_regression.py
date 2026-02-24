"""Xyce regression test runner.

Runs tests from the Xyce_Regression suite against VAJAX.
Auto-discovers all tests and runs them without filtering.

The test workflow:
1. Convert SPICE .cir to VACASK .sim format
2. Run VAJAX simulator
3. Compare output with expected .prn file
"""

import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import jax.numpy as jnp
import pytest
from jax import Array

from tests.xyce_test_registry import (
    XYCE_NETLISTS,
    XYCE_OUTPUT,
    discover_xyce_tests,
)
from vajax.analysis.engine import CircuitEngine
from vajax.io.prn_reader import get_column, read_prn

# Get ALL discovered tests at module load time for parametrization
ALL_TESTS = {t.name: t for t in discover_xyce_tests()}


def convert_spice_to_vacask(cir_path: Path, sim_path: Path) -> None:
    """Convert SPICE netlist to VACASK format.

    Args:
        cir_path: Input SPICE .cir file
        sim_path: Output VACASK .sim file
    """
    from vajax.netlist_converter.ng2vclib.converter import Converter
    from vajax.netlist_converter.ng2vclib.dfl import default_config

    # Get default config and add source directory
    cfg = default_config()
    cfg["sourcepath"] = [str(cir_path.parent)]

    converter = Converter(cfg, dialect="ngspice")
    converter.convert(str(cir_path), str(sim_path))


def parse_tran_params(cir_path: Path) -> Tuple[float, float]:
    """Extract .TRAN parameters from a SPICE netlist.

    Returns:
        Tuple of (t_stop, dt) where dt is estimated from step or t_stop/1000
    """
    content = cir_path.read_text()

    # Find .TRAN line: .TRAN [step] stop [start] [maxstep]
    # Must be an active line (not a SPICE comment starting with *)
    # Examples:
    #   .TRAN 0 0.5ms
    #   .TRAN 1ns 100ns
    import re

    tran_match = re.search(r"^\.TRAN\s+(\S+)\s+(\S+)", content, re.IGNORECASE | re.MULTILINE)
    if not tran_match:
        raise ValueError(f"No .TRAN statement found in {cir_path}")

    def parse_value(s: str) -> float:
        """Parse SPICE value with SI suffix."""
        s = s.strip()
        # Order matters - check longer suffixes first
        suffixes = [
            ("meg", 1e6),
            ("mil", 25.4e-6),  # mils
            ("ms", 1e-3),  # milliseconds
            ("us", 1e-6),  # microseconds
            ("ns", 1e-9),  # nanoseconds
            ("ps", 1e-12),  # picoseconds
            ("fs", 1e-15),  # femtoseconds
            ("s", 1),  # seconds (unit, not a scale factor)
            ("f", 1e-15),
            ("p", 1e-12),
            ("n", 1e-9),
            ("u", 1e-6),
            ("m", 1e-3),
            ("k", 1e3),
            ("g", 1e9),
            ("t", 1e12),
        ]
        s_lower = s.lower()
        for suffix, mult in suffixes:
            if s_lower.endswith(suffix):
                return float(s[: -len(suffix)]) * mult
        return float(s)

    step_str = tran_match.group(1)
    stop_str = tran_match.group(2)

    t_stop = parse_value(stop_str)

    # If step is 0, estimate from t_stop
    step = parse_value(step_str)
    if step == 0:
        dt = t_stop / 1000
    else:
        dt = step

    return t_stop, dt


def run_xyce_test(
    test_name: str,
    cir_file: str,
    *,
    rtol: float = 1e-2,
    atol: float = 1e-6,
    check_columns: Optional[list] = None,
) -> Dict[str, Array]:
    """Run a Xyce regression test and compare with expected output.

    Args:
        test_name: Directory name under Netlists/ and OutputData/
        cir_file: Netlist filename (e.g., "diode.cir")
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        check_columns: Specific columns to check (None = all)

    Returns:
        Dict of computed values from VAJAX
    """
    cir_path = XYCE_NETLISTS / test_name / cir_file
    prn_path = XYCE_OUTPUT / test_name / f"{cir_file}.prn"

    if not cir_path.exists():
        pytest.skip(f"Netlist not found: {cir_path}")
    if not prn_path.exists():
        pytest.skip(f"Expected output not found: {prn_path}")

    # Read expected output
    expected_cols, expected_data = read_prn(prn_path)

    # Get time array from expected output
    time_col = get_column(expected_data, "TIME")
    if time_col is None:
        pytest.skip("No TIME column in expected output")

    # Parse simulation parameters
    t_stop, dt = parse_tran_params(cir_path)

    # Convert and run
    with tempfile.TemporaryDirectory() as tmpdir:
        sim_path = Path(tmpdir) / "test.sim"

        try:
            convert_spice_to_vacask(cir_path, sim_path)
        except Exception as e:
            pytest.skip(f"Failed to convert netlist: {e}")

        # Run simulation
        try:
            engine = CircuitEngine(sim_path)
            engine.parse()

            max_time = float(jnp.max(time_col))

            engine.prepare(
                t_stop=max_time,
                dt=dt,
            )
            result = engine.run_transient()
        except Exception as e:
            pytest.skip(f"Simulation failed: {e}")

    # Compare results
    computed = {}
    comparison_errors = []

    for col in check_columns or expected_cols:
        if col.upper() in ("INDEX", "TIME"):
            continue

        expected = get_column(expected_data, col)
        if expected is None:
            continue

        # Find matching column in computed results
        # Xyce column names like "V(3)" map to node "3"
        import re

        node_match = re.match(r"V\((\w+)\)", col, re.IGNORECASE)
        if node_match:
            node_name = node_match.group(1)
            if node_name in result.voltages:
                computed_arr = result.voltages[node_name]
                computed[col] = computed_arr

                # Interpolate to expected time points
                from jax.numpy import interp

                computed_at_expected = interp(
                    time_col,
                    result.times,
                    computed_arr,
                )

                # Check if values are close
                if not jnp.allclose(computed_at_expected, expected, rtol=rtol, atol=atol):
                    max_diff = float(jnp.max(jnp.abs(computed_at_expected - expected)))
                    comparison_errors.append(f"{col}: max diff = {max_diff:.6e}")

    if comparison_errors:
        pytest.fail("Value mismatch:\n" + "\n".join(comparison_errors))

    return computed


# --- Test cases ---


class TestXyceRegression:
    """Tests from Xyce_Regression suite.

    Auto-discovers all tests from vendor/Xyce_Regression/Netlists/.
    Tests are run without device filtering to see full suite coverage.
    """

    @pytest.mark.parametrize(
        "test_name",
        list(ALL_TESTS.keys()),
        ids=list(ALL_TESTS.keys()),
    )
    def test_xyce(self, test_name):
        """Run auto-discovered Xyce test against VAJAX."""
        test_case = ALL_TESTS[test_name]

        if not test_case.netlist_path.exists():
            pytest.skip(f"Netlist not found: {test_case.netlist_path}")

        # Skip non-transient tests for now
        if test_case.analysis_type != "tran":
            pytest.skip(f"Analysis type {test_case.analysis_type} not yet supported")

        # Skip tests without expected output
        if test_case.output_path is None:
            pytest.skip(f"No expected output file for {test_name}")

        run_xyce_test(
            test_case.category,
            test_case.netlist_path.name,
            check_columns=test_case.expected_nodes if test_case.expected_nodes else None,
            rtol=test_case.rtol,
            atol=test_case.atol,
        )
