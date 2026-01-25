#!/usr/bin/env python3
"""Three-way comparison: VACASK vs Full MNA vs ngspice."""

import os
import sys
import struct

os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import matplotlib.pyplot as plt

from jax_spice.analysis.engine import CircuitEngine
from jax_spice.analysis.transient import AdaptiveConfig, FullMNAStrategy


def read_spice_raw(filename):
    """Read a SPICE raw file (binary format)."""
    with open(filename, 'rb') as f:
        content = f.read()

    binary_marker = b'Binary:\n'
    binary_pos = content.find(binary_marker)
    header = content[:binary_pos].decode('utf-8')
    lines = header.strip().split('\n')

    n_vars = n_points = None
    variables = []
    in_variables = False

    for line in lines:
        if line.startswith('No. Variables:'):
            n_vars = int(line.split(':')[1].strip())
        elif line.startswith('No. Points:'):
            n_points = int(line.split(':')[1].strip())
        elif line.startswith('Variables:'):
            in_variables = True
        elif in_variables and line.strip():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                variables.append(parts[1])

    binary_data = content[binary_pos + len(binary_marker):]
    point_size = n_vars * 8
    n_points = min(n_points, len(binary_data) // point_size)

    data = np.zeros((n_points, n_vars), dtype=np.float64)
    for i in range(n_points):
        offset = i * point_size
        for j in range(n_vars):
            val_bytes = binary_data[offset + j*8 : offset + (j+1)*8]
            if len(val_bytes) == 8:
                data[i, j] = struct.unpack('d', val_bytes)[0]

    return {name: data[:, i] for i, name in enumerate(variables)}


def read_ngspice_wrdata(filename):
    """Read ngspice wrdata format (columns: time v1 time v2 time i)."""
    data = np.loadtxt(filename)
    # Format: time, v(1), time, v(2), time, i(vdd)
    t = data[:, 0]
    v1 = data[:, 1]
    v2 = data[:, 3]
    i_vdd = data[:, 5]
    return {'time': t, '1': v1, '2': v2, 'i_vdd': i_vdd}


def main():
    ring_sim = 'vendor/VACASK/benchmark/ring/vacask/runme.sim'
    vacask_raw = 'vendor/VACASK/tran1.raw'
    ngspice_data_file = 'vendor/VACASK/benchmark/ring/ngspice/ring_with_current.txt'

    print("Loading VACASK data...")
    vacask = read_spice_raw(vacask_raw)
    t_vac = vacask['time']
    V1_vac = vacask['1']
    V2_vac = vacask['2']
    I_vac = vacask.get('vdd:flow(br)')

    print("Loading ngspice data...")
    ng = read_ngspice_wrdata(ngspice_data_file)
    t_ng = ng['time']
    V1_ng = ng['1']
    V2_ng = ng['2']
    I_ng = ng['i_vdd']

    print("Running Full MNA (20ns)...")
    runner = CircuitEngine(ring_sim)
    runner.parse()
    # max_dt must be less than oscillation period (~3.5ns) to capture dynamics
    config = AdaptiveConfig(max_dt=50e-12, min_dt=1e-15)
    full_mna = FullMNAStrategy(runner, use_sparse=False, config=config)
    times_mna, voltages_mna, stats_mna = full_mna.run(t_stop=20e-9, dt=1e-12)

    t_mna = np.asarray(times_mna)
    V1_mna = np.asarray(voltages_mna['1'])
    V2_mna = np.asarray(voltages_mna['2'])
    I_mna = np.asarray(stats_mna['currents']['vdd'])

    # Compute dI/dt for all three
    def compute_didt(t, I):
        dt = np.diff(t)
        dI = np.diff(I)
        dIdt = dI / dt
        t_mid = t[:-1] + dt / 2
        return t_mid, dIdt

    t_didt_vac, dIdt_vac = compute_didt(t_vac, I_vac)
    t_didt_ng, dIdt_ng = compute_didt(t_ng, I_ng)
    t_didt_mna, dIdt_mna = compute_didt(t_mna, I_mna)

    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Time window
    t_start, t_end = 2e-9, 15e-9

    # Colors
    c_vac = 'blue'
    c_ng = 'green'
    c_mna = 'red'

    # Panel 1: Voltages
    ax1 = axes[0]
    mask_vac = (t_vac >= t_start) & (t_vac <= t_end)
    mask_ng = (t_ng >= t_start) & (t_ng <= t_end)
    mask_mna = (t_mna >= t_start) & (t_mna <= t_end)

    ax1.plot(t_vac[mask_vac] * 1e9, V1_vac[mask_vac], c_vac, lw=1.5, label='VACASK V(1)', alpha=0.9)
    ax1.plot(t_ng[mask_ng] * 1e9, V1_ng[mask_ng], c_ng, lw=1.5, label='ngspice V(1)', alpha=0.9, linestyle='--')
    ax1.plot(t_mna[mask_mna] * 1e9, V1_mna[mask_mna], c_mna, lw=1.5, label='Full MNA V(1)', alpha=0.9, linestyle=':')
    ax1.set_ylabel('Voltage [V]', fontsize=11)
    ax1.set_ylim(-0.2, 1.4)
    ax1.legend(loc='upper right', ncol=3, fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Ring Oscillator: VACASK vs ngspice vs Full MNA', fontsize=12, fontweight='bold')

    # Panel 2: Current
    ax2 = axes[1]
    ax2.plot(t_vac[mask_vac] * 1e9, I_vac[mask_vac] * 1e6, c_vac, lw=1.5, label='VACASK', alpha=0.9)
    ax2.plot(t_ng[mask_ng] * 1e9, I_ng[mask_ng] * 1e6, c_ng, lw=1.5, label='ngspice', alpha=0.9, linestyle='--')
    ax2.plot(t_mna[mask_mna] * 1e9, I_mna[mask_mna] * 1e6, c_mna, lw=1.5, label='Full MNA', alpha=0.9, linestyle=':')
    ax2.set_ylabel('I(VDD) [µA]', fontsize=11)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: dI/dt
    ax3 = axes[2]
    mask_didt_vac = (t_didt_vac >= t_start) & (t_didt_vac <= t_end)
    mask_didt_ng = (t_didt_ng >= t_start) & (t_didt_ng <= t_end)
    mask_didt_mna = (t_didt_mna >= t_start) & (t_didt_mna <= t_end)

    ax3.plot(t_didt_vac[mask_didt_vac] * 1e9, dIdt_vac[mask_didt_vac] * 1e-6, c_vac, lw=1, label='VACASK', alpha=0.8)
    ax3.plot(t_didt_ng[mask_didt_ng] * 1e9, dIdt_ng[mask_didt_ng] * 1e-6, c_ng, lw=1, label='ngspice', alpha=0.8, linestyle='--')
    ax3.plot(t_didt_mna[mask_didt_mna] * 1e9, dIdt_mna[mask_didt_mna] * 1e-6, c_mna, lw=1, label='Full MNA', alpha=0.8, linestyle=':')
    ax3.set_ylabel('dI/dt [mA/ns]', fontsize=11)
    ax3.set_xlabel('Time [ns]', fontsize=11)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = 'ring_three_way_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")

    # Metrics
    print("\n" + "=" * 70)
    print("Comparison Metrics (2-15ns window):")
    print("=" * 70)

    t_common = np.linspace(t_start, t_end, 2000)
    I_vac_i = np.interp(t_common, t_vac, I_vac)
    I_ng_i = np.interp(t_common, t_ng, I_ng)
    I_mna_i = np.interp(t_common, t_mna, I_mna)

    print(f"\nMean current:")
    print(f"  VACASK:   {I_vac_i.mean()*1e6:.1f} µA")
    print(f"  ngspice:  {I_ng_i.mean()*1e6:.1f} µA")
    print(f"  Full MNA: {I_mna_i.mean()*1e6:.1f} µA")

    print(f"\nMax |dI/dt|:")
    print(f"  VACASK:   {np.max(np.abs(dIdt_vac[mask_didt_vac]))*1e-6:.2f} mA/ns")
    print(f"  ngspice:  {np.max(np.abs(dIdt_ng[mask_didt_ng]))*1e-6:.2f} mA/ns")
    print(f"  Full MNA: {np.max(np.abs(dIdt_mna[mask_didt_mna]))*1e-6:.2f} mA/ns")

    return 0


if __name__ == '__main__':
    sys.exit(main())
