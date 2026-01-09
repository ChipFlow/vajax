#!/usr/bin/env python3
"""Debug what charges the device models are returning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from jax_spice.analysis.engine import CircuitEngine


def main():
    sim_path = Path(__file__).parent.parent.parent / "VACASK" / "benchmark" / "ring" / "vacask" / "runme.sim"

    if not sim_path.exists():
        print(f"Ring benchmark not found at {sim_path}")
        return 1

    print("="*80)
    print("Debugging Device Charge Returns")
    print("="*80)

    engine = CircuitEngine(sim_path)
    engine.parse()

    # Count PSP103 devices
    psp103_count = sum(1 for d in engine.devices if d['model'] == 'psp103')
    print(f"\nPSP103 devices: {psp103_count}")

    # Run minimal transient to get setup
    result = engine.run_transient(t_stop=0.1e-9, dt=0.1e-9)

    setup = engine._transient_setup_cache
    device_arrays = engine._device_arrays
    n_external = engine.num_nodes
    n_unknowns = n_external - 1

    # Get DC voltage
    V_dc = jnp.zeros(n_external, dtype=jnp.float64)
    for name, voltages in result.voltages.items():
        if name in engine.node_names:
            idx = engine.node_names[name]
            V_dc = V_dc.at[idx].set(float(voltages[0]))

    print(f"\nDC operating point:")
    print(f"  V(1) = {float(V_dc[1]):.6f}V")

    # Get the vmapped PSP103 eval function
    if 'psp103' not in setup.get('vmapped_fns', {}):
        print("\nERROR: No PSP103 vmapped function found!")
        return 1

    psp103_info = setup['vmapped_fns']['psp103']
    vmapped_split_eval = psp103_info['vmapped_split_eval']
    shared_params = psp103_info['shared_params']
    device_params = psp103_info['device_params']
    voltage_positions = psp103_info['voltage_positions']
    use_cache_split = psp103_info['use_cache_split']
    shared_cache = psp103_info.get('shared_cache')

    # Get voltage indices for PSP103 devices
    static_metadata = psp103_info['static_metadata']
    voltage_indices, stamp_indices, voltage_node1, voltage_node2 = static_metadata

    # Extract voltages for PSP103 devices
    voltage_updates = V_dc[voltage_node1] - V_dc[voltage_node2]

    # Update device_params with voltages
    device_params_updated = device_params.at[:, voltage_positions].set(voltage_updates)

    # Set analysis_type for DC (0)
    uses_analysis = engine._compiled_models.get('psp103', {}).get('uses_analysis', False)
    if uses_analysis:
        device_params_updated = device_params_updated.at[:, -2].set(0.0)  # DC analysis
        device_params_updated = device_params_updated.at[:, -1].set(1e-12)  # gmin

    # Get PSP103 cache
    cache = device_arrays['psp103']

    print(f"\nEvaluating {psp103_count} PSP103 devices at DC...")

    # Call vmapped eval
    if use_cache_split:
        batch_res_resist, batch_res_react, batch_jac_resist, batch_jac_react, \
            batch_lim_rhs_resist, batch_lim_rhs_react = \
            vmapped_split_eval(shared_params, device_params_updated, shared_cache, cache)
    else:
        batch_res_resist, batch_res_react, batch_jac_resist, batch_jac_react, \
            batch_lim_rhs_resist, batch_lim_rhs_react = \
            vmapped_split_eval(shared_params, device_params_updated, cache)

    print(f"\nPSP103 charge statistics (DC):")
    print(f"  batch_res_react shape: {batch_res_react.shape}")
    print(f"  Min charge: {float(jnp.min(batch_res_react)):.6e} C")
    print(f"  Max charge: {float(jnp.max(batch_res_react)):.6e} C")
    print(f"  Mean |charge|: {float(jnp.mean(jnp.abs(batch_res_react))):.6e} C")
    print(f"  Non-zero charges: {int(jnp.sum(batch_res_react != 0.0))}/{batch_res_react.size}")

    # Check for huge values
    huge_mask = jnp.abs(batch_res_react) > 1e-10
    if jnp.any(huge_mask):
        print(f"\n⚠️  WARNING: Found {int(jnp.sum(huge_mask))} charge values > 1e-10 C!")
        huge_indices = jnp.where(huge_mask)
        for i in range(min(10, int(jnp.sum(huge_mask)))):
            idx0, idx1 = int(huge_indices[0][i]), int(huge_indices[1][i])
            val = float(batch_res_react[idx0, idx1])
            print(f"    Device {idx0}, terminal {idx1}: {val:.6e} C")

    # Check capacitance values
    print(f"\nPSP103 capacitance statistics (DC):")
    print(f"  batch_jac_react shape: {batch_jac_react.shape}")
    print(f"  Min capacitance: {float(jnp.min(batch_jac_react)):.6e} F")
    print(f"  Max capacitance: {float(jnp.max(batch_jac_react)):.6e} F")
    print(f"  Mean |capacitance|: {float(jnp.mean(jnp.abs(batch_jac_react))):.6e} F")

    # Now check what happens after one transient timestep
    print("\n" + "="*80)
    print("After one transient timestep (dt=0.1ns)")
    print("="*80)

    # Set analysis_type for transient (2)
    if uses_analysis:
        device_params_updated = device_params_updated.at[:, -2].set(2.0)  # Transient analysis

    # Evaluate again
    if use_cache_split:
        batch_res_resist_t, batch_res_react_t, batch_jac_resist_t, batch_jac_react_t, \
            batch_lim_rhs_resist_t, batch_lim_rhs_react_t = \
            vmapped_split_eval(shared_params, device_params_updated, shared_cache, cache)
    else:
        batch_res_resist_t, batch_res_react_t, batch_jac_resist_t, batch_jac_react_t, \
            batch_lim_rhs_resist_t, batch_lim_rhs_react_t = \
            vmapped_split_eval(shared_params, device_params_updated, cache)

    print(f"\nPSP103 charge statistics (transient):")
    print(f"  Min charge: {float(jnp.min(batch_res_react_t)):.6e} C")
    print(f"  Max charge: {float(jnp.max(batch_res_react_t)):.6e} C")
    print(f"  Mean |charge|: {float(jnp.mean(jnp.abs(batch_res_react_t))):.6e} C")

    # Check difference
    delta_Q = batch_res_react_t - batch_res_react
    print(f"\nCharge change (transient - DC):")
    print(f"  Min ΔQ: {float(jnp.min(delta_Q)):.6e} C")
    print(f"  Max ΔQ: {float(jnp.max(delta_Q)):.6e} C")
    print(f"  Mean |ΔQ|: {float(jnp.mean(jnp.abs(delta_Q))):.6e} C")

    if jnp.max(jnp.abs(delta_Q)) > 1e-10:
        print(f"\n⚠️  WARNING: Charges changed significantly just from analysis_type change!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
