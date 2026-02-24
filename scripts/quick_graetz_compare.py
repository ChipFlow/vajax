#!/usr/bin/env python3
"""Quick graetz accuracy check vs VACASK with short t_stop."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

from pathlib import Path

import numpy as np

from vajax.analysis.engine import CircuitEngine

sim_path = Path("vendor/VACASK/benchmark/graetz/vacask/runme.sim")
engine = CircuitEngine(sim_path)
engine.parse()

# Run short simulation: 600µs
t_stop = 600e-6
dt = 1e-6

print(f"Running graetz: t_stop={t_stop}, dt={dt}")
engine.prepare(t_stop=t_stop, dt=dt)
result = engine.run_transient()

print(f"Result: {result.num_steps} steps")
print(f"Node names: {result.node_names}")

times = np.array(result.times)

# Check for NaN/Inf in any node voltage
has_nan = False
has_inf = False
for name in result.node_names:
    v = np.array(result.voltage(name))
    if np.any(np.isnan(v)):
        has_nan = True
    if np.any(np.isinf(v)):
        has_inf = True
print(f"NaN: {has_nan}, Inf: {has_inf}")

if not has_nan and not has_inf:
    for name in result.node_names[:5]:
        v = np.array(result.voltage(name))
        print(f"  Node {name}: min={v.min():.4f} max={v.max():.4f}")
    print(f"\nSimulation completed successfully to t={times[-1]:.6e}s")

    # Check voltage at specific times to find divergence
    node0 = result.node_names[0]
    v0 = np.array(result.voltage(node0))
    print(f"\nVoltage at node '{node0}' over time:")
    for frac in [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.85,
        0.9,
        0.92,
        0.93,
        0.94,
        0.95,
        0.96,
        0.97,
        0.98,
        0.99,
        1.0,
    ]:
        idx = min(int(frac * len(times)) - 1, len(times) - 1)
        if idx >= 0:
            print(f"  t={times[idx]:.3e} ({frac * 100:.0f}%): V={v0[idx]:.6f}")

    # Find first "unreasonable" voltage (> 100V)
    for i in range(len(v0)):
        if abs(v0[i]) > 100:
            print(f"\n  First unreasonable voltage at step {i}, t={times[i]:.3e}: V={v0[i]:.2e}")
            if i > 0:
                print(f"  Previous step t={times[i - 1]:.3e}: V={v0[i - 1]:.6f}")
            break
else:
    print("Simulation has NaN/Inf values")
