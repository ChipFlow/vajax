"""CMOS Inverter DC Sweep - Comparison with ngspice

This example replicates the ngspice test case: tests/bsim3soidd/inv2.cir
Circuit topology:
- NMOS: W=10um, L=0.25um
- PMOS: W=20um, L=0.25um
- VDD = 2.5V
- Input swept 0 → 2.5V in 251 points
"""

import sys
sys.path.insert(0, '/Users/roberttaylor/Code/ChipFlow/reference/jax-spice')

import numpy as np
import matplotlib.pyplot as plt
from jax_spice import Circuit, MOSFETSimple
from jax_spice.analysis import dc_sweep, get_node_voltage


def create_inverter_circuit():
    """Create CMOS inverter matching ngspice test case"""
    ckt = Circuit()

    # NMOS transistor (pulls output low)
    nmos = MOSFETSimple(
        W=10e-6,           # 10 um
        L=0.25e-6,         # 0.25 um
        Vth0=0.4,          # Typical NMOS threshold
        pmos=False,
        u0=400e-4,         # 400 cm^2/V/s
        lambda_=0.05,      # Channel length modulation
        gamma=0.5,         # Body effect
        tox=5e-9,          # 5nm oxide
    )

    # PMOS transistor (pulls output high)
    pmos = MOSFETSimple(
        W=20e-6,           # 20 um (2x NMOS for symmetric switching)
        L=0.25e-6,         # 0.25 um
        Vth0=-0.4,         # Typical PMOS threshold (negative)
        pmos=True,
        u0=150e-4,         # 150 cm^2/V/s (holes slower than electrons)
        lambda_=0.05,
        gamma=0.5,
        tox=5e-9,
    )

    # Add devices to circuit
    ckt.add_device("M1", pmos, connections={
        'd': 'out',   # Drain
        'g': 'in',    # Gate
        's': 'vdd',   # Source (connected to VDD for PMOS)
        'b': 'vdd',   # Bulk (connected to VDD for PMOS)
    })

    ckt.add_device("M2", nmos, connections={
        'd': 'out',   # Drain
        'g': 'in',    # Gate
        's': 'gnd',   # Source (connected to GND for NMOS)
        'b': 'gnd',   # Bulk (connected to GND for NMOS)
    })

    # Add voltage sources
    ckt.add_vsource("VDD", 'vdd', 'gnd', 2.5)   # Power supply
    ckt.add_vsource("VIN", 'in', 'gnd', 1.25)   # Input (initial value, will be swept)

    # Set ground reference
    ckt.set_ground('gnd')

    return ckt


def run_dc_sweep_analysis():
    """Run DC sweep and plot transfer curve"""
    print("=" * 70)
    print("JAX-SPICE: CMOS Inverter DC Sweep Analysis")
    print("=" * 70)

    # Create circuit
    print("\nCreating circuit...")
    ckt = create_inverter_circuit()
    print(f"  {ckt}")
    print(f"  Nodes: {list(ckt.nodes.keys())}")
    print(f"  Devices: {list(ckt.devices.keys())}")

    # Run DC sweep
    print("\nRunning DC sweep: Vin = 0 → 2.5V (251 points)...")
    Vin_sweep, solutions, converged = dc_sweep(
        ckt,
        sweep_source='VIN',
        start=0.0,
        stop=2.5,
        points=251,
        verbose=True
    )

    # Extract output voltage
    Vout = np.array([get_node_voltage(ckt, sol, 'out') for sol in solutions])

    # Convergence summary
    num_converged = sum(converged)
    print(f"\nConvergence: {num_converged}/251 points converged")

    if num_converged < 251:
        print(f"  Warning: {251 - num_converged} points failed to converge")
        # Show which points failed
        failed_indices = [i for i, c in enumerate(converged) if not c]
        print(f"  Failed at Vin = {Vin_sweep[failed_indices]} V")

    # Print some key points
    print("\nTransfer Curve (selected points):")
    print("  Vin (V) | Vout (V)")
    print("  --------|----------")
    for i in [0, 50, 100, 125, 150, 200, 250]:
        if converged[i]:
            print(f"  {Vin_sweep[i]:6.3f}  | {Vout[i]:8.6f}")
        else:
            print(f"  {Vin_sweep[i]:6.3f}  | (failed)")

    # Find switching point (Vin = Vout)
    idx_switch = np.argmin(np.abs(Vin_sweep - Vout))
    print(f"\nSwitching point: Vin = {Vin_sweep[idx_switch]:.3f}V, Vout = {Vout[idx_switch]:.3f}V")

    # Compute gain at switching point
    dVout = Vout[idx_switch + 1] - Vout[idx_switch - 1]
    dVin = Vin_sweep[idx_switch + 1] - Vin_sweep[idx_switch - 1]
    gain = -dVout / dVin
    print(f"Gain at switching point: {gain:.1f}")

    return Vin_sweep, Vout, converged


def plot_results(Vin, Vout, converged):
    """Plot transfer curve"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Transfer curve
    converged_mask = np.array(converged)
    ax1.plot(Vin[converged_mask], Vout[converged_mask], 'b-', linewidth=2, label='JAX-SPICE')
    ax1.plot(Vin[~converged_mask], Vout[~converged_mask], 'rx', label='Failed points')
    ax1.plot([0, 2.5], [0, 2.5], 'k--', alpha=0.3, label='Vin=Vout')
    ax1.set_xlabel('Input Voltage Vin (V)', fontsize=12)
    ax1.set_ylabel('Output Voltage Vout (V)', fontsize=12)
    ax1.set_title('CMOS Inverter Transfer Curve', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim([0, 2.5])
    ax1.set_ylim([0, 2.5])

    # Plot 2: Gain (derivative)
    dVout_dVin = np.gradient(Vout[converged_mask], Vin[converged_mask])
    ax2.plot(Vin[converged_mask], -dVout_dVin, 'r-', linewidth=2)
    ax2.set_xlabel('Input Voltage Vin (V)', fontsize=12)
    ax2.set_ylabel('Voltage Gain |dVout/dVin|', fontsize=12)
    ax2.set_title('Inverter Gain', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 2.5])
    ax2.set_ylim([0, max(20, np.max(-dVout_dVin) * 1.1)])

    plt.tight_layout()
    plt.savefig('/Users/roberttaylor/Code/ChipFlow/reference/jax-spice/examples/inverter_transfer_curve.png', dpi=150)
    print("\nPlot saved to: examples/inverter_transfer_curve.png")
    plt.show()


if __name__ == "__main__":
    Vin, Vout, converged = run_dc_sweep_analysis()
    plot_results(Vin, Vout, converged)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
