#!/usr/bin/env python3
"""Debug script to compare charges between single MOSFET and ring oscillator.

The goal is to understand why we see:
- Ring oscillator: Q[0] = -5.29e-13 C (correct order of magnitude, femtoCoulombs)
- Single MOSFET test: -5.16 C (WRONG - 13 orders of magnitude too large)

Same PSP103 model should give consistent results.
"""

import sys
from pathlib import Path

# Add openvaf-py to path
openvaf_path = Path(__file__).parent.parent / "openvaf-py"
if str(openvaf_path) not in sys.path:
    sys.path.insert(0, str(openvaf_path))

import jax
import jax.numpy as jnp
import numpy as np

# Force CPU for consistent behavior
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from jax_spice.analysis.engine import CircuitEngine


def test_ring_charges():
    """Get charges from ring oscillator at DC operating point."""
    print("=" * 60)
    print("Ring Oscillator Charges")
    print("=" * 60)

    sim_path = Path(__file__).parent.parent / "vendor" / "VACASK" / "benchmark" / "ring" / "ring.sim"

    if not sim_path.exists():
        print(f"Ring benchmark not found at {sim_path}")
        return None

    engine = CircuitEngine(sim_path)
    engine.parse()

    print(f"Nodes: {engine.n_total} total, {engine.n_external} external")
    print(f"Node names: {engine.node_names}")

    # Get device info
    for model_type, instances in engine.instances_by_model.items():
        print(f"\nModel: {model_type}")
        for inst in instances[:3]:  # First 3 instances
            print(f"  Instance: {inst.name}")
            print(f"    Terminals: {inst.terminals}")
            print(f"    Params: W={inst.params.get('w', 'N/A')}, L={inst.params.get('l', 'N/A')}")

    # Run DC operating point
    V_dc = engine.run_dc()
    print(f"\nDC voltages (first 10): {V_dc[:10]}")

    # Get charges at DC operating point
    # Need to evaluate the system at the DC point and extract Q
    setup = engine._prepare_device_data()

    # Build system to get Q
    device_arrays = engine._device_arrays
    n_total = engine.n_total
    n_external = engine.n_external
    n_unknowns = n_external - 1  # Exclude ground

    # Call build_system directly
    build_system = engine._make_build_system(use_dense=True, n_unknowns=n_unknowns, max_nnz=10000)

    # Evaluate at DC (inv_dt=0, Q_prev=zeros)
    V_dc_full = jnp.zeros(n_total, dtype=jnp.float64)
    V_dc_full = V_dc_full.at[:len(V_dc)].set(V_dc)

    # Get source values at t=0
    source_fn = setup.source_fn
    source_values = source_fn(0.0)

    vsource_vals = jnp.zeros(len(engine.vsource_info), dtype=jnp.float64)
    isource_vals = jnp.zeros(len(engine.isource_info), dtype=jnp.float64)

    for name, val in source_values.items():
        if name in engine.vsource_info:
            idx = engine.vsource_info[name]['device_idx']
            vsource_vals = vsource_vals.at[idx].set(val)
        elif name in engine.isource_info:
            idx = engine.isource_info[name]['device_idx']
            isource_vals = isource_vals.at[idx].set(val)

    Q_prev = jnp.zeros(n_unknowns, dtype=jnp.float64)

    # Build system at DC
    J, f, Q = build_system(
        V_dc_full, vsource_vals, isource_vals, Q_prev,
        0.0,  # inv_dt = 0 for DC
        device_arrays
    )

    print(f"\n=== Charges at DC operating point ===")
    print(f"Q shape: {Q.shape}")
    print(f"Q[0:10]: {Q[:10]}")
    print(f"Q max: {float(jnp.max(Q)):.6e}")
    print(f"Q min: {float(jnp.min(Q)):.6e}")
    print(f"Q sum: {float(jnp.sum(Q)):.6e}")

    # VACASK reference: -2.28e-13 C per signal node
    print(f"\nVACASK reference: -2.28e-13 C per signal node")

    return Q


def test_single_mosfet_charges():
    """Get charges from a single MOSFET at same operating point."""
    print("\n" + "=" * 60)
    print("Single MOSFET Charges")
    print("=" * 60)

    # Create a simple single MOSFET netlist
    # Same W/L as ring oscillator (w=10u, l=1u)
    # Same Vds, Vgs as ring DC operating point (~0.6V per node)

    # We need to create a temporary .sim file
    import tempfile

    netlist = """
// Single MOSFET test - same operating point as ring oscillator
load "psp103v4.osdi"

include "models.inc"

model vsource vsource

mn (d g 0 0) nmos w=10u l=1u

// Same voltages as ring DC operating point
vd (d 0) vsource dc=0.6
vg (g 0) vsource dc=0.6

control
  options gmin=1e-15
  analysis op1 op
endc
"""

    # Need the models.inc from VACASK benchmark
    models_inc = Path(__file__).parent.parent / "vendor" / "VACASK" / "benchmark" / "ring" / "models.inc"

    if not models_inc.exists():
        print(f"models.inc not found at {models_inc}")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Copy models.inc
        import shutil
        shutil.copy(models_inc, tmpdir / "models.inc")

        # Write netlist
        sim_file = tmpdir / "single_mos.sim"
        sim_file.write_text(netlist)

        engine = CircuitEngine(sim_file)
        engine.parse()

        print(f"Nodes: {engine.n_total} total, {engine.n_external} external")
        print(f"Node names: {engine.node_names}")

        # Get device info
        for model_type, instances in engine.instances_by_model.items():
            print(f"\nModel: {model_type}")
            for inst in instances:
                print(f"  Instance: {inst.name}")
                print(f"    Terminals: {inst.terminals}")
                print(f"    Params: W={inst.params.get('w', 'N/A')}, L={inst.params.get('l', 'N/A')}")

        # Run DC operating point
        try:
            V_dc = engine.run_dc()
            print(f"\nDC voltages: {V_dc}")
        except Exception as e:
            print(f"DC failed: {e}")
            V_dc = jnp.zeros(engine.n_total, dtype=jnp.float64)

        # Get charges at DC operating point
        setup = engine._prepare_device_data()
        device_arrays = engine._device_arrays
        n_total = engine.n_total
        n_external = engine.n_external
        n_unknowns = n_external - 1

        build_system = engine._make_build_system(use_dense=True, n_unknowns=n_unknowns, max_nnz=10000)

        # Set voltages to same as ring (~0.6V for all signal nodes, 1.2V for VDD)
        V_test = jnp.zeros(n_total, dtype=jnp.float64)
        # Set d=0.6V, g=0.6V
        if 'd' in engine.node_names:
            V_test = V_test.at[engine.node_names['d']].set(0.6)
        if 'g' in engine.node_names:
            V_test = V_test.at[engine.node_names['g']].set(0.6)

        # Get source values at t=0
        source_fn = setup.source_fn
        source_values = source_fn(0.0)

        vsource_vals = jnp.zeros(len(engine.vsource_info), dtype=jnp.float64)
        isource_vals = jnp.zeros(len(engine.isource_info), dtype=jnp.float64)

        for name, val in source_values.items():
            if name in engine.vsource_info:
                idx = engine.vsource_info[name]['device_idx']
                vsource_vals = vsource_vals.at[idx].set(val)
            elif name in engine.isource_info:
                idx = engine.isource_info[name]['device_idx']
                isource_vals = isource_vals.at[idx].set(val)

        Q_prev = jnp.zeros(n_unknowns, dtype=jnp.float64)

        # Build system
        J, f, Q = build_system(
            V_test, vsource_vals, isource_vals, Q_prev,
            0.0,  # inv_dt = 0 for DC
            device_arrays
        )

        print(f"\n=== Charges at V(d)=0.6V, V(g)=0.6V ===")
        print(f"Q shape: {Q.shape}")
        print(f"Q: {Q}")
        print(f"Q max: {float(jnp.max(Q)):.6e}")
        print(f"Q min: {float(jnp.min(Q)):.6e}")

        # Check for extreme values
        if jnp.any(jnp.abs(Q) > 1e-6):
            print("\n⚠️  WARNING: Charges are too large (>1µC)!")
            print("   Expected: ~1e-13 C (femtoCoulombs)")
            print("   Got: >1e-6 C")

        return Q


def test_raw_device_eval():
    """Directly evaluate the PSP103 device to see raw charge outputs."""
    print("\n" + "=" * 60)
    print("Raw PSP103 Device Evaluation")
    print("=" * 60)

    import openvaf_py
    import openvaf_jax

    # Find PSP103 model
    psp_va = Path(__file__).parent.parent / "openvaf-py" / "vendor" / "OpenVAF" / "integration_tests" / "PSP103" / "psp103.va"

    if not psp_va.exists():
        print(f"PSP103 VA file not found at {psp_va}")
        return

    print(f"Loading PSP103 from: {psp_va}")

    # Compile with OpenVAF
    try:
        modules = openvaf_py.compile_va(str(psp_va))
        print(f"Compiled: {modules}")
        if not modules:
            print("No modules found")
            return
        module = modules[0]
    except Exception as e:
        print(f"Compilation failed: {e}")
        return

    print(f"Model type: {module.module_name}")
    print(f"Parameters: {len(module.params)} params")
    print(f"Terminals: {module.nodes}")

    # Create JAX function using translator
    translator = openvaf_jax.OpenVAFToJAX(module)
    jax_fns = translator.translate()

    # Get parameter info
    param_info = translator.param_info
    print(f"\nParam kinds: {[p['kind'] for p in param_info[:10]]}...")

    # Count by kind
    kind_counts = {}
    for p in param_info:
        kind = p['kind']
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    print(f"Kind counts: {kind_counts}")

    # Build input array
    n_params = len(param_info)
    params = np.zeros(n_params, dtype=np.float64)

    # Set W/L (same as ring)
    for i, p in enumerate(param_info):
        if p['name'] == 'W':
            params[i] = 10e-6  # 10um
            print(f"Set W={params[i]} at index {i}")
        elif p['name'] == 'L':
            params[i] = 1e-6   # 1um
            print(f"Set L={params[i]} at index {i}")
        elif p['kind'] == 'temperature':
            params[i] = 300.15  # 27C
        elif p['name'] == 'mfactor':
            params[i] = 1.0

    # Set voltages: Vd=0.6, Vg=0.6, Vs=0, Vb=0
    # Need to find voltage indices
    for i, p in enumerate(param_info):
        if p['kind'] == 'voltage':
            name = p['name']
            if 'd' in name.lower():
                params[i] = 0.6
                print(f"Set V({name})={params[i]} at index {i}")
            elif 'g' in name.lower():
                params[i] = 0.6
                print(f"Set V({name})={params[i]} at index {i}")
            elif 's' in name.lower():
                params[i] = 0.0
                print(f"Set V({name})={params[i]} at index {i}")
            elif 'b' in name.lower():
                params[i] = 0.0
                print(f"Set V({name})={params[i]} at index {i}")

    # Get init and eval functions
    init_fn = jax_fns['init_fn']
    eval_fn = jax_fns['eval_fn']

    # Run init to get cache
    params_jax = jnp.array(params)
    cache = init_fn(params_jax)
    print(f"\nCache shape: {cache.shape}")
    print(f"Cache[0:10]: {cache[:10]}")
    print(f"Cache has inf: {jnp.any(jnp.isinf(cache))}")
    print(f"Cache has nan: {jnp.any(jnp.isnan(cache))}")

    # Find extreme cache values
    cache_abs = jnp.abs(cache)
    extreme_idx = jnp.where(cache_abs > 1e10)[0]
    if len(extreme_idx) > 0:
        print(f"\nExtreme cache values (>1e10):")
        for idx in extreme_idx[:10]:
            print(f"  cache[{idx}] = {float(cache[idx]):.6e}")

    # Run eval
    res_resist, res_react, jac_resist, jac_react, lim_rhs_resist, lim_rhs_react = eval_fn(params_jax, cache)

    print(f"\n=== Device outputs ===")
    print(f"res_resist: {res_resist}")
    print(f"res_react (charges): {res_react}")
    print(f"jac_resist (conductances): {jac_resist[:10] if len(jac_resist) > 10 else jac_resist}")
    print(f"jac_react (capacitances): {jac_react[:10] if len(jac_react) > 10 else jac_react}")

    # Check for extreme charges
    if jnp.any(jnp.abs(res_react) > 1e-6):
        print("\n⚠️  WARNING: res_react (charges) are too large (>1µC)!")
        print("   Expected: ~1e-13 C (femtoCoulombs)")
        max_charge = float(jnp.max(jnp.abs(res_react)))
        print(f"   Got max: {max_charge:.6e} C")
    else:
        print(f"\n✓ Charges look reasonable: max={float(jnp.max(jnp.abs(res_react))):.6e} C")


if __name__ == "__main__":
    # Run all tests
    test_raw_device_eval()
    Q_ring = test_ring_charges()
    Q_single = test_single_mosfet_charges()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if Q_ring is not None:
        print(f"Ring Q[0]: {float(Q_ring[0]):.6e} C")
    if Q_single is not None:
        print(f"Single MOSFET Q[0]: {float(Q_single[0]):.6e} C")
