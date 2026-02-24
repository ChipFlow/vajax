#!/usr/bin/env python3
"""Debug graetz early transient steps to find where voltages go wrong."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

from pathlib import Path

import numpy as np

from vajax.analysis.engine import CircuitEngine

sim_path = Path("vendor/VACASK/benchmark/graetz/vacask/runme.sim")
engine = CircuitEngine(sim_path)
engine.parse()

# Run very short simulation: 50µs
t_stop = 50e-6
dt = 1e-6

print(f"Running graetz: t_stop={t_stop}, dt={dt}")
engine.prepare(t_stop=t_stop, dt=dt)
result = engine.run_transient()

print(f"Result: {result.num_steps} steps")
times = np.array(result.times)

# Print ALL voltages at every step
node0 = result.node_names[0]
v0 = np.array(result.voltage(node0))

print(f"\nStep-by-step voltage at node '{node0}':")
print("Source: ampl=20, freq=50Hz, V_source(t) = 20*sin(2*pi*50*t)")
for i in range(min(len(times), 50)):
    t = times[i]
    v_source = 20.0 * np.sin(2 * np.pi * 50 * t)
    flag = " <<<" if abs(v0[i]) > 10 else ""
    print(f"  Step {i:3d}: t={t:.3e}  V_source={v_source:.6f}  V_node={v0[i]:.6f}{flag}")
