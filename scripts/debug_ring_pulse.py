#!/usr/bin/env python3
"""Debug ring oscillator pulse current source.

Check if the pulse is being applied correctly during transient.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from jax_spice.analysis.engine import CircuitEngine


def main():
    sim_path = Path(__file__).parent.parent.parent / "VACASK" / "benchmark" / "ring" / "vacask" / "runme.sim"

    if not sim_path.exists():
        print(f"Ring benchmark not found at {sim_path}")
        return 1

    print("="*80)
    print("Ring Oscillator Pulse Debug")
    print("="*80)

    engine = CircuitEngine(sim_path)
    engine.parse()

    print(f"\nCircuit devices:")
    for dev in engine.devices:
        if dev['model'] == 'isource':
            print(f"\nCurrent source: {dev['name']}")
            print(f"  Nodes: {dev['nodes']}")
            print(f"  Params: {dev['params']}")

    # Run transient
    print("\n" + "="*80)
    print("Running transient (10ns)")
    print("="*80)

    result = engine.run_transient(t_stop=10e-9, dt=0.1e-9)

    # Get V(1) and currents
    times_ns = result.times * 1e9
    v1 = result.voltages['1']

    # Check if there are any currents in the result
    print(f"\nAvailable currents: {list(result.currents.keys())}")

    # Plot V(1) and pulse timing
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # V(1) voltage
    ax1.plot(times_ns, v1, 'b-', linewidth=1.5, label='V(1)')
    ax1.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, label='DC level')
    ax1.set_ylabel('Voltage (V)', fontsize=12)
    ax1.set_title('Ring Oscillator Node 1 Voltage', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Pulse timing markers
    pulse_delay = 1.0  # 1ns
    pulse_width = 1.0  # 1ns
    pulse_end = pulse_delay + pulse_width

    ax2.axvline(x=pulse_delay, color='red', linestyle='--', alpha=0.7, label=f'Pulse start ({pulse_delay}ns)')
    ax2.axvline(x=pulse_end, color='orange', linestyle='--', alpha=0.7, label=f'Pulse end ({pulse_end}ns)')
    ax2.axhspan(0, 10, alpha=0.2, color='red', label='Pulse active')

    # Mark pulse region
    ax2.fill_betweenx([0, 1], pulse_delay, pulse_end, alpha=0.3, color='red')
    ax2.set_xlabel('Time (ns)', fontsize=12)
    ax2.set_ylabel('Expected\nPulse', fontsize=12)
    ax2.set_ylim([0, 1.5])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['0µA', '10µA'])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ring_pulse_debug.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to ring_pulse_debug.png")

    # Print voltage statistics
    print(f"\nV(1) statistics:")
    print(f"  Min: {float(jnp.min(v1)):.6f}V")
    print(f"  Max: {float(jnp.max(v1)):.6f}V")
    print(f"  Swing: {float(jnp.max(v1) - jnp.min(v1)):.6f}V")
    print(f"  Initial: {float(v1[0]):.6f}V")
    print(f"  Final: {float(v1[-1]):.6f}V")

    # Check voltage change during pulse
    pulse_start_idx = jnp.argmin(jnp.abs(result.times - pulse_delay * 1e-9))
    pulse_end_idx = jnp.argmin(jnp.abs(result.times - pulse_end * 1e-9))

    v_before_pulse = v1[pulse_start_idx - 1] if pulse_start_idx > 0 else v1[0]
    v_during_pulse = v1[pulse_start_idx:pulse_end_idx]
    v_after_pulse = v1[pulse_end_idx] if pulse_end_idx < len(v1) else v1[-1]

    print(f"\nVoltage response to pulse:")
    print(f"  Before pulse ({pulse_delay-0.1:.1f}ns): {float(v_before_pulse):.6f}V")
    print(f"  During pulse ({pulse_delay:.1f}-{pulse_end:.1f}ns): {float(jnp.mean(v_during_pulse)):.6f}V avg")
    print(f"  After pulse ({pulse_end+0.1:.1f}ns): {float(v_after_pulse):.6f}V")

    delta_during = float(jnp.max(jnp.abs(v_during_pulse - v_before_pulse)))
    print(f"  Max voltage change during pulse: {delta_during:.6f}V")

    if delta_during < 0.001:
        print("\n⚠️  WARNING: Voltage barely changes during pulse (<1mV)")
        print("   The circuit is not responding to the current injection!")
    else:
        print(f"\n✓ Circuit responds to pulse with {delta_during*1000:.2f}mV change")

    return 0


if __name__ == "__main__":
    sys.exit(main())
