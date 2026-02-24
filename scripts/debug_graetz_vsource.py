#!/usr/bin/env python3
"""Debug graetz voltage source constraint and node voltages."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

from pathlib import Path

import numpy as np

from vajax.analysis.engine import CircuitEngine

sim_path = Path("vendor/VACASK/benchmark/graetz/vacask/runme.sim")
engine = CircuitEngine(sim_path)
engine.parse()

# Run 300µs to capture the divergence region
t_stop = 300e-6
dt = 1e-6

print(f"Running graetz: t_stop={t_stop}, dt={dt}")
engine.prepare(t_stop=t_stop, dt=dt)
result = engine.run_transient()

print(f"Result: {result.num_steps} steps, nodes={result.node_names}")
times = np.array(result.times)

v_inn = np.array(result.voltage("inn"))
v_inp = np.array(result.voltage("inp"))
v_outn = np.array(result.voltage("outn"))
v_outp = np.array(result.voltage("outp"))

print("\nStep-by-step analysis (every 10 steps):")
print(
    f"{'Step':>5} {'t(µs)':>8} {'V_src':>8} {'V_inp':>10} {'V_inn':>10} {'Vdiff':>10} {'err':>10} {'V_outp':>10} {'V_outn':>10}"
)
for i in range(0, min(len(times), 300), 10):
    t = times[i]
    v_src = 20.0 * np.sin(2 * np.pi * 50 * t)
    v_diff = v_inp[i] - v_inn[i]
    err = v_diff - v_src
    print(
        f"{i:5d} {t * 1e6:8.2f} {v_src:8.4f} {v_inp[i]:10.4f} {v_inn[i]:10.4f} {v_diff:10.4f} {err:10.2e} {v_outp[i]:10.4f} {v_outn[i]:10.4f}"
    )

# Check common mode drift
print("\nCommon mode (V_inp + V_inn)/2:")
for i in range(0, min(len(times), 300), 20):
    t = times[i]
    cm = (v_inp[i] + v_inn[i]) / 2
    print(f"  Step {i:3d}: t={t * 1e6:.1f}µs  CM={cm:.4f}V")
