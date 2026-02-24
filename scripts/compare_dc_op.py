#!/usr/bin/env python3
"""Compare DC operating points between simulators."""

import os
import struct

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np


def read_spice_raw(filename):
    """Read a SPICE raw file."""
    with open(filename, "rb") as f:
        content = f.read()
    binary_marker = b"Binary:\n"
    binary_pos = content.find(binary_marker)
    header = content[:binary_pos].decode("utf-8")
    lines = header.strip().split("\n")
    n_vars = n_points = None
    variables = []
    in_variables = False
    for line in lines:
        if line.startswith("No. Variables:"):
            n_vars = int(line.split(":")[1].strip())
        elif line.startswith("No. Points:"):
            n_points = int(line.split(":")[1].strip())
        elif line.startswith("Variables:"):
            in_variables = True
        elif in_variables and line.strip():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                variables.append(parts[1])
    binary_data = content[binary_pos + len(binary_marker) :]
    point_size = n_vars * 8
    n_points = min(n_points, len(binary_data) // point_size)
    data = np.zeros((n_points, n_vars), dtype=np.float64)
    for i in range(n_points):
        offset = i * point_size
        for j in range(n_vars):
            val_bytes = binary_data[offset + j * 8 : offset + (j + 1) * 8]
            if len(val_bytes) == 8:
                data[i, j] = struct.unpack("d", val_bytes)[0]
    return {name: data[:, i] for i, name in enumerate(variables)}


# Load VACASK data
vacask = read_spice_raw("vendor/VACASK/tran1.raw")
print("VACASK DC operating point (t=0):")
print(f"  V(1) = {vacask['1'][0]:.6f} V")
print(f"  V(2) = {vacask['2'][0]:.6f} V")
print(f"  V(3) = {vacask['3'][0]:.6f} V")
print(f"  I(VDD) = {vacask['vdd:flow(br)'][0] * 1e6:.2f} µA")

# Load ngspice data
ng = np.loadtxt("vendor/VACASK/benchmark/ring/ngspice/ring_with_current.txt")
print("\nngspice DC operating point (t=0):")
print(f"  V(1) = {ng[0, 1]:.6f} V")
print(f"  V(2) = {ng[0, 3]:.6f} V")
print(f"  I(VDD) = {ng[0, 5] * 1e6:.2f} µA")

# Run VA-JAX Full MNA
from vajax.analysis.engine import CircuitEngine
from vajax.analysis.transient import FullMNAStrategy

runner = CircuitEngine("vendor/VACASK/benchmark/ring/vacask/runme.sim")
runner.parse()
full_mna = FullMNAStrategy(runner, use_sparse=False)
times, voltages, stats = full_mna.run(t_stop=1e-12, dt=1e-12)

print("\nFull MNA DC operating point (t=0):")
print(f"  V(1) = {float(voltages['1'][0]):.6f} V")
print(f"  V(2) = {float(voltages['2'][0]):.6f} V")
print(f"  V(3) = {float(voltages['3'][0]):.6f} V")
print(f"  I(VDD) = {float(stats['currents']['vdd'][0]) * 1e6:.2f} µA")

# Compare
print("\n" + "=" * 50)
print("Comparison:")
print("=" * 50)
v1_vac = vacask["1"][0]
v1_ng = ng[0, 1]
v1_mna = float(voltages["1"][0])

print("\nV(1) differences:")
print(f"  VACASK - ngspice: {(v1_vac - v1_ng) * 1e3:.3f} mV")
print(f"  VACASK - Full MNA: {(v1_vac - v1_mna) * 1e3:.3f} mV")
print(f"  ngspice - Full MNA: {(v1_ng - v1_mna) * 1e3:.3f} mV")

# Check if all nodes have same voltage (ring oscillator at DC should be symmetric)
print("\nRing symmetry check (all nodes should be equal at DC):")
for i in range(1, 10):
    node = str(i)
    if node in voltages:
        print(f"  V({i}) = {float(voltages[node][0]):.6f} V")
