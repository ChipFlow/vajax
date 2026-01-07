#!/usr/bin/env python3
"""Compare JAX-SPICE vs VACASK debug output for ring oscillator.

This script:
1. Runs JAX-SPICE with debug options enabled
2. Compares with VACASK reference data from debug_traced_data.json
3. Identifies where behavior diverges
"""

import sys
from pathlib import Path
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp

# Force CPU for consistent behavior
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from jax_spice.analysis.engine import CircuitEngine


def load_vacask_reference():
    """Load VACASK reference data."""
    # Try within jax-spice-analysis/vendor first
    ref_path = Path(__file__).parent.parent / "vendor" / "VACASK" / "benchmark" / "ring" / "vacask" / "debug_traced_data.json"

    # If not found, try parent reference directory
    if not ref_path.exists():
        ref_path = Path(__file__).parent.parent.parent / "VACASK" / "benchmark" / "ring" / "vacask" / "debug_traced_data.json"

    if not ref_path.exists():
        print(f"VACASK reference not found at {ref_path}")
        return None

    with open(ref_path) as f:
        return json.load(f)


def run_jax_spice_with_debug():
    """Run JAX-SPICE ring oscillator with debug options."""
    # Try within jax-spice-analysis/vendor first
    sim_path = Path(__file__).parent.parent / "vendor" / "VACASK" / "benchmark" / "ring" / "vacask" / "runme.sim"

    # If not found, try parent reference directory
    if not sim_path.exists():
        sim_path = Path(__file__).parent.parent.parent / "VACASK" / "benchmark" / "ring" / "vacask" / "runme.sim"

    if not sim_path.exists():
        print(f"Ring benchmark not found at {sim_path}")
        return None

    print("="*80)
    print("Running JAX-SPICE with debug options")
    print("="*80)

    engine = CircuitEngine(sim_path)
    engine.parse()

    # Enable debug options
    engine.debug_options.q_debug = 1
    engine.debug_options.nr_debug = 1
    engine.debug_options.tran_debug = 1

    print(f"\nCircuit info:")
    print(f"  Nodes: {engine.num_nodes} external")
    print(f"  Devices: {len(engine.devices)}")
    print(f"  Node names: {list(engine.node_names.keys())[:10]}...")

    # Run a short transient to get DC and first few timesteps
    print("\n" + "="*80)
    print("Running transient analysis (first 5ns)")
    print("="*80)

    result = engine.run_transient(t_stop=5e-9, dt=0.1e-9)

    print(f"\n✓ Transient completed: {len(result.times)} timesteps")

    return {
        'engine': engine,
        'result': result,
    }


def compare_dc_operating_point(jax_result, vacask_ref):
    """Compare DC operating point."""
    print("\n" + "="*80)
    print("Comparing DC Operating Point")
    print("="*80)

    # Get JAX-SPICE DC voltages (first timestep)
    jax_dc = {}
    for name, voltages in jax_result['result'].voltages.items():
        jax_dc[name] = float(voltages[0])

    # VACASK reference from charges_dc in debug_traced_data.json
    # All signal nodes at ~0.66V according to the note
    vacask_dc_ref = 0.6606  # From transient_v1.voltages[0]

    print("\nDC voltages (signal nodes):")
    print(f"{'Node':<10} {'JAX-SPICE':>12} {'VACASK Ref':>12} {'Diff':>12} {'% Error':>10}")
    print("-"*60)

    for i in range(1, 10):  # Nodes 1-9
        node_name = str(i)
        jax_v = jax_dc.get(node_name, 0.0)
        diff = jax_v - vacask_dc_ref
        pct_error = (diff / vacask_dc_ref) * 100 if vacask_dc_ref != 0 else 0

        print(f"{node_name:<10} {jax_v:>12.6f} {vacask_dc_ref:>12.6f} {diff:>12.6f} {pct_error:>9.2f}%")

    # VDD node
    if 'vdd' in jax_dc:
        jax_vdd = jax_dc['vdd']
        vacask_vdd = 1.2
        diff = jax_vdd - vacask_vdd
        pct_error = (diff / vacask_vdd) * 100
        print(f"{'vdd':<10} {jax_vdd:>12.6f} {vacask_vdd:>12.6f} {diff:>12.6f} {pct_error:>9.2f}%")

    # Compute RMS error for signal nodes
    errors = []
    for i in range(1, 10):
        node_name = str(i)
        if node_name in jax_dc:
            diff = jax_dc[node_name] - vacask_dc_ref
            errors.append(diff)

    if errors:
        rms_error = (sum(e**2 for e in errors) / len(errors)) ** 0.5
        print(f"\nRMS error (signal nodes): {rms_error:.6f}V ({rms_error/vacask_dc_ref*100:.2f}%)")


def compare_charges(jax_result, vacask_ref):
    """Compare charge values."""
    print("\n" + "="*80)
    print("Comparing Charges at DC")
    print("="*80)

    # VACASK reference charges from debug_traced_data.json
    vacask_charges = vacask_ref['charges_dc']['values']

    print(f"\nVACASK reference charges:")
    print(f"  Signal nodes: ~{vacask_charges['node_1']:.6e} C")
    print(f"  VDD: {vacask_charges['vdd']:.6e} C")
    print(f"  Note: {vacask_ref['charges_dc']['note']}")

    # Note: JAX-SPICE charges are printed during DC computation via q_debug
    # We'd need to capture that output or re-compute charges here
    print("\n(JAX-SPICE charges were printed during DC computation above)")


def compare_transient_response(jax_result, vacask_ref):
    """Compare transient voltage response."""
    print("\n" + "="*80)
    print("Comparing Transient Response")
    print("="*80)

    # Get V(1) waveform
    if '1' not in jax_result['result'].voltages:
        print("Node 1 not found in JAX-SPICE results")
        return

    jax_v1 = jax_result['result'].voltages['1']
    jax_times = jax_result['result'].times

    # VACASK reference for V(1)
    vacask_times_ns = vacask_ref['transient_v1']['times_ns']
    vacask_v1 = vacask_ref['transient_v1']['voltages_v']
    vacask_summary = vacask_ref['transient_v1']['summary']

    print(f"\nV(1) summary:")
    print(f"{'':20} {'JAX-SPICE':>15} {'VACASK':>15} {'Match?':>10}")
    print("-"*62)

    jax_initial = float(jax_v1[0])
    vacask_initial = vacask_summary['initial_dc']
    initial_match = abs(jax_initial - vacask_initial) < 0.01
    print(f"{'Initial DC (V)':20} {jax_initial:>15.6f} {vacask_initial:>15.6f} {'✓' if initial_match else '✗':>10}")

    jax_max = float(jnp.max(jax_v1))
    vacask_max = vacask_summary['peak']
    max_match = abs(jax_max - vacask_max) < 0.1
    print(f"{'Peak (V)':20} {jax_max:>15.6f} {vacask_max:>15.6f} {'✓' if max_match else '✗':>10}")

    jax_min = float(jnp.min(jax_v1))
    vacask_min = vacask_summary['min']
    min_match = abs(jax_min - vacask_min) < 0.1
    print(f"{'Min (V)':20} {jax_min:>15.6f} {vacask_min:>15.6f} {'✓' if min_match else '✗':>10}")

    jax_final = float(jax_v1[-1])
    vacask_final = vacask_summary['final']
    final_match = abs(jax_final - vacask_final) < 0.1
    print(f"{'Final (V)':20} {jax_final:>15.6f} {vacask_final:>15.6f} {'✓' if final_match else '✗':>10}")

    # Check if oscillation is happening
    jax_range = jax_max - jax_min
    vacask_range = vacask_max - vacask_min

    print(f"\nVoltage swing:")
    print(f"  JAX-SPICE: {jax_range:.6f}V")
    print(f"  VACASK: {vacask_range:.6f}V")

    if jax_range < 0.1:
        print("\n⚠️  WARNING: JAX-SPICE shows minimal voltage swing (<0.1V)")
        print("   Circuit does not appear to be oscillating!")
        print("   VACASK shows swing of {:.2f}V - this is the main discrepancy".format(vacask_range))
    else:
        print("\n✓ Circuit is oscillating")

    # Sample a few time points for detailed comparison
    print(f"\nVoltage at key times:")
    print(f"{'Time (ns)':>10} {'JAX-SPICE (V)':>15} {'VACASK (V)':>15} {'Diff (V)':>12}")
    print("-"*54)

    # Compare at t=0, 1ns, 2ns, 3ns, 4ns, 5ns
    for t_ns in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
        # Find closest JAX-SPICE time
        jax_idx = jnp.argmin(jnp.abs(jax_times - t_ns * 1e-9))
        jax_v = float(jax_v1[jax_idx])
        jax_t = float(jax_times[jax_idx]) * 1e9

        # Find closest VACASK time
        vacask_idx = min(range(len(vacask_times_ns)),
                        key=lambda i: abs(vacask_times_ns[i] - t_ns))
        vacask_v = vacask_v1[vacask_idx]

        diff = jax_v - vacask_v
        print(f"{jax_t:>10.3f} {jax_v:>15.6f} {vacask_v:>15.6f} {diff:>12.6f}")


def main():
    """Main comparison routine."""
    print("JAX-SPICE vs VACASK Ring Oscillator Debug Comparison")
    print("="*80)

    # Load VACASK reference
    vacask_ref = load_vacask_reference()
    if vacask_ref is None:
        return 1

    print(f"\n✓ Loaded VACASK reference data")
    print(f"  Circuit: {vacask_ref['circuit']}")
    print(f"  Simulator: {vacask_ref['simulator']} {vacask_ref['version']}")

    # Run JAX-SPICE
    jax_result = run_jax_spice_with_debug()
    if jax_result is None:
        return 1

    # Compare results
    compare_dc_operating_point(jax_result, vacask_ref)
    compare_charges(jax_result, vacask_ref)
    compare_transient_response(jax_result, vacask_ref)

    print("\n" + "="*80)
    print("Comparison Complete")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
