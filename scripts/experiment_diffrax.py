#!/usr/bin/env python3
"""Minimal diffrax experiment: RC circuit validation.

Tests if diffrax can solve a simple circuit ODE using Kvaerno5 (implicit solver).
Compares against analytical solution.

Circuit:
    Vs(5V) --[R=1k]--+--[C=1µF]--GND
                      |
                      V_cap

ODE: dV_cap/dt = (Vs - V_cap) / (R*C)
Analytical: V_cap(t) = Vs * (1 - exp(-t/τ)) where τ = RC

Usage:
    JAX_PLATFORMS=cpu uv run python scripts/experiment_diffrax.py
"""

import jax
import jax.numpy as jnp
import diffrax

jax.config.update('jax_enable_x64', True)


def rc_circuit_ode(t, V, args):
    """dV/dt for RC charging circuit."""
    Vs, R, C = args
    return (Vs - V) / (R * C)


def main():
    # Circuit parameters
    Vs = 5.0        # Voltage source: 5V
    R = 1000.0      # Resistance: 1kΩ
    C = 1e-6        # Capacitance: 1µF
    tau = R * C     # Time constant: 1ms

    print(f"RC Circuit: Vs={Vs}V, R={R}Ω, C={C*1e6}µF")
    print(f"Time constant τ = {tau*1000}ms")
    print()

    # Solve with diffrax using Kvaerno5 (implicit, stiff-capable)
    term = diffrax.ODETerm(rc_circuit_ode)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)

    t_end = 5 * tau  # 5 time constants (99.3% charged)
    save_times = jnp.linspace(0, t_end, 100)

    print("Solving with diffrax.Kvaerno5...")
    solution = diffrax.diffeqsolve(
        term, solver,
        t0=0.0, t1=t_end, dt0=tau/100,
        y0=jnp.array(0.0),
        args=(Vs, R, C),
        stepsize_controller=stepsize_controller,
        saveat=diffrax.SaveAt(ts=save_times)
    )

    # Compare to analytical solution
    V_analytical = Vs * (1 - jnp.exp(-solution.ts / tau))
    errors = jnp.abs(solution.ys - V_analytical)
    max_error = float(jnp.max(errors))

    # Results
    print(f"Time points: {len(solution.ts)}")
    print(f"Final V: {float(solution.ys[-1]):.6f}V (expected: {float(Vs * (1 - jnp.exp(-5))):.6f}V)")
    print(f"Max error vs analytical: {max_error:.2e}")
    print()

    # Check at key time constants
    for n in [1, 2, 3, 5]:
        t_check = n * tau
        # Find closest index in save_times
        idx = int(jnp.argmin(jnp.abs(solution.ts - t_check)))
        t_actual = float(solution.ts[idx])
        V_sim = float(solution.ys[idx])
        V_expected = float(Vs * (1 - jnp.exp(-t_actual / tau)))
        err = abs(V_sim - V_expected)
        print(f"  t={n}τ ({t_actual*1000:.3f}ms): V={V_sim:.4f}V (expected {V_expected:.4f}V, err={err:.2e})")

    print()
    if max_error < 1e-5:
        print("✓ SUCCESS: diffrax + Kvaerno5 solves RC circuit correctly")
        return 0
    else:
        print(f"✗ FAILED: Error {max_error:.2e} > 1e-5 threshold")
        return 1


if __name__ == "__main__":
    exit(main())
