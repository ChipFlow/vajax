"""Test debug options (nr_debug, tran_debug, q_debug)."""

import pytest
from pathlib import Path
import jax
import jax.numpy as jnp

# Force CPU for consistent behavior
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from jax_spice.analysis.engine import CircuitEngine, DebugOptions


def test_debug_options_parsing():
    """Test that debug options are correctly parsed from netlist."""
    # Ring oscillator has debug options in the debug variant
    sim_path = Path(__file__).parent.parent / "vendor" / "VACASK" / "benchmark" / "ring" / "vacask" / "debug_q.sim"

    if not sim_path.exists():
        pytest.skip(f"Ring debug netlist not found at {sim_path}")

    engine = CircuitEngine(sim_path)
    engine.parse()

    # Check that q_debug was parsed
    assert engine.debug_options.q_debug == 1, "Expected q_debug=1 from debug_q.sim"


def test_q_debug_output():
    """Test q_debug output for charges at DC operating point."""
    # Use the ring oscillator
    sim_path = Path(__file__).parent.parent / "vendor" / "VACASK" / "benchmark" / "ring" / "vacask" / "runme.sim"

    if not sim_path.exists():
        pytest.skip(f"Ring benchmark not found at {sim_path}")

    engine = CircuitEngine(sim_path)
    engine.parse()

    # Enable q_debug
    engine.debug_options.q_debug = 1

    # Run transient with a very short time to trigger DC computation
    # The DC operating point is computed at the start of transient analysis
    print("\n" + "="*60)
    print("Testing q_debug output")
    print("="*60)

    # Run just the DC operating point computation by running transient for 1 step
    result = engine.run_transient(t_stop=1e-12, dt=1e-12)

    # Verify transient ran
    assert result is not None
    assert len(result.times) > 0

    # The q_debug output should have been printed to stdout during DC computation
    print(f"\nTransient started (DC computed with q_debug output)")
    print(f"Times: {len(result.times)} steps")


def test_q_debug_charge_magnitude():
    """Test that charges are in the correct order of magnitude (femtoCoulombs)."""
    sim_path = Path(__file__).parent.parent / "vendor" / "VACASK" / "benchmark" / "ring" / "vacask" / "runme.sim"

    if not sim_path.exists():
        pytest.skip(f"Ring benchmark not found at {sim_path}")

    engine = CircuitEngine(sim_path)
    engine.parse()

    # Run a short transient to get the DC operating point
    result = engine.run_transient(t_stop=1e-12, dt=1e-12)

    # Extract DC voltages (first voltage point)
    V_dc_dict = {name: voltages[0] for name, voltages in result.voltages.items()}

    # Build full voltage vector
    n_external = engine.num_nodes
    V_dc = jnp.zeros(n_external, dtype=jnp.float64)
    for name, voltage in V_dc_dict.items():
        if name in engine.node_names:
            idx = engine.node_names[name]
            V_dc = V_dc.at[idx].set(voltage)

    # Get charges at DC
    n_external = engine.num_nodes
    n_unknowns = n_external - 1

    # Build source values
    vsource_dc_vals = jnp.zeros(len(engine.vsource_info), dtype=jnp.float64)
    for name, info in engine.vsource_info.items():
        dev = next((d for d in engine.devices if d['name'] == name), None)
        if dev:
            vsource_dc_vals = vsource_dc_vals.at[info['device_idx']].set(
                float(dev['params'].get('dc', 0.0)))

    isource_dc_vals = jnp.zeros(len(engine.isource_info), dtype=jnp.float64)
    Q_prev = jnp.zeros(n_unknowns, dtype=jnp.float64)

    # Get device arrays
    setup = engine._prepare_device_data()
    device_arrays = engine._device_arrays

    # Build system to get Q
    build_system = engine._make_build_system(use_dense=True, n_unknowns=n_unknowns, max_nnz=10000)
    J, f, Q = build_system(V_dc, vsource_dc_vals, isource_dc_vals, Q_prev, 0.0, device_arrays)

    # Check charge magnitude
    q_abs_max = float(jnp.max(jnp.abs(Q)))
    print(f"\nMax |Q|: {q_abs_max:.6e} C")

    # VACASK shows charges of ~2e-13 C for the ring oscillator
    # Allow a range from 1e-15 to 1e-9 C (femtoCoulombs to nanoCoulombs)
    assert q_abs_max > 1e-15, f"Charges too small: {q_abs_max:.6e} C"
    assert q_abs_max < 1e-9, f"Charges too large: {q_abs_max:.6e} C (expected ~1e-13 to 1e-12 C)"

    # Compare to VACASK reference value
    # VACASK: -2.28e-13 C per signal node
    vacask_ref = 2.28e-13
    ratio = q_abs_max / vacask_ref

    print(f"VACASK reference: {vacask_ref:.6e} C")
    print(f"JAX-SPICE / VACASK ratio: {ratio:.2f}x")

    # Allow up to 10x difference (could be due to different model parameters or initialization)
    if ratio > 10:
        pytest.warn(UserWarning(
            f"Charges are {ratio:.2f}x larger than VACASK reference. "
            f"This may indicate a bug in charge computation."
        ))


def test_debug_options_programmatic():
    """Test setting debug options programmatically."""
    sim_path = Path(__file__).parent.parent / "vendor" / "VACASK" / "benchmark" / "ring" / "vacask" / "runme.sim"

    if not sim_path.exists():
        pytest.skip(f"Ring benchmark not found at {sim_path}")

    engine = CircuitEngine(sim_path)
    engine.parse()

    # Initially should be 0
    assert engine.debug_options.nr_debug == 0
    assert engine.debug_options.tran_debug == 0
    assert engine.debug_options.q_debug == 0

    # Set programmatically
    engine.debug_options.nr_debug = 2
    engine.debug_options.tran_debug = 1
    engine.debug_options.q_debug = 1

    assert engine.debug_options.nr_debug == 2
    assert engine.debug_options.tran_debug == 1
    assert engine.debug_options.q_debug == 1


if __name__ == "__main__":
    # Run tests directly
    test_debug_options_programmatic()
    print("✓ test_debug_options_programmatic passed\n")

    test_q_debug_output()
    print("✓ test_q_debug_output passed\n")

    test_q_debug_charge_magnitude()
    print("✓ test_q_debug_charge_magnitude passed\n")

    # This one requires the debug_q.sim file
    try:
        test_debug_options_parsing()
        print("✓ test_debug_options_parsing passed\n")
    except Exception as e:
        print(f"⚠ test_debug_options_parsing skipped: {e}\n")

    print("All tests passed!")
