"""Test fori_loop NR solver correctness against while_loop baseline.

Verifies that the fori_loop NR mode (use_fori_loop=True) produces
identical results to the while_loop mode on a simple resistive divider
circuit that converges in 2-3 NR iterations.
"""

import jax
import jax.numpy as jnp
import pytest

from vajax.analysis.solver_factories import _make_nr_solver_common
from vajax.analysis.options import SimulationOptions


def _make_resistive_divider_system(n_nodes: int = 3, n_vsources: int = 1):
    """Create a simple 2-resistor voltage divider for NR solver testing.

    Circuit: V1 (1V) -- R1 (1k) -- node1 -- R2 (1k) -- GND
    Expected: V(node1) = 0.5V, I(V1) = -0.5mA

    The MNA system is:
        [G11  1] [V1]   [0    ]
        [1    0] [Iv1] = [V1_val]

    where G11 = 1/R1 + 1/R2 = 0.002

    Args:
        n_nodes: Total nodes including ground (3: gnd=0, node1=1, vsrc_node=2)
        n_vsources: Number of voltage sources (1)

    Returns:
        build_system_fn that matches the NR solver's expected interface
    """
    R1 = 1000.0  # 1kΩ
    R2 = 1000.0  # 1kΩ
    n_unknowns = n_nodes - 1  # Exclude ground
    n_aug = n_unknowns + n_vsources

    def build_system(X, vsource_vals, isource_vals, Q_prev, integ_c0,
                     device_arrays, gmin, gshunt, integ_c1, integ_d1,
                     dQdt_prev, integ_c2, Q_prev2, limit_state, iteration):
        """Build MNA Jacobian and residual for resistive divider."""
        # X layout: [V_gnd=0 (implicit), V_node1, V_vsrc_node, I_v1]
        V1 = X[1]       # node1 voltage
        V2 = X[2]       # vsrc node voltage (should be 1V)
        I_v1 = X[3]     # V1 branch current

        V1_val = vsource_vals[0]  # Source voltage (1.0V)

        # Jacobian (augmented MNA)
        J = jnp.zeros((n_aug, n_aug), dtype=X.dtype)
        # Node 1 (internal node): (V1-0)/R2 + (V1-V2)/R1 = 0
        J = J.at[0, 0].set(1.0/R1 + 1.0/R2 + gmin)  # dI1/dV1
        J = J.at[0, 1].set(-1.0/R1)                    # dI1/dV2
        # Node 2 (vsrc node): (V2-V1)/R1 + I_v1 = 0
        J = J.at[1, 0].set(-1.0/R1)                    # dI2/dV1
        J = J.at[1, 1].set(1.0/R1 + gmin)              # dI2/dV2
        J = J.at[1, 2].set(1.0)                         # dI2/dI_v1
        # V1 equation: V2 = V1_val
        J = J.at[2, 1].set(1.0)                         # dVeq/dV2

        # Residual f = J*X - b (KCL errors)
        f = jnp.zeros(n_aug, dtype=X.dtype)
        f = f.at[0].set((V1 - 0)/R2 + (V1 - V2)/R1 + gmin * V1)
        f = f.at[1].set((V2 - V1)/R1 + I_v1 + gmin * V2)
        f = f.at[2].set(V2 - V1_val)

        # Q (charge), I_vsource, limit_state, max_res_contrib
        Q = jnp.zeros(n_unknowns, dtype=X.dtype)
        I_vsource = jnp.array([I_v1], dtype=X.dtype)
        limit_state_out = limit_state
        max_res_contrib = jnp.abs(jnp.array([
            jnp.maximum(jnp.abs(V1/R2), jnp.abs((V1-V2)/R1)),
            jnp.maximum(jnp.abs((V2-V1)/R1), jnp.abs(I_v1)),
        ], dtype=X.dtype))

        return J, f, Q, I_vsource, limit_state_out, max_res_contrib

    return build_system


def _run_nr_solver(use_fori_loop: bool, max_nr_iters: int = 8):
    """Run NR solver on the resistive divider with given mode.

    Returns:
        Tuple of (X_final, iterations, converged, max_f)
    """
    n_nodes = 3
    n_vsources = 1

    build_system = _make_resistive_divider_system(n_nodes, n_vsources)
    options = SimulationOptions()

    def linear_solve(J, f):
        reg = 1e-14 * jnp.eye(J.shape[0], dtype=J.dtype)
        return jax.scipy.linalg.solve(J + reg, -f)

    def enforce_noi(J, f):
        return J, f

    nr_solve = _make_nr_solver_common(
        build_system_jit=build_system,
        n_nodes=n_nodes,
        n_vsources=n_vsources,
        linear_solve_fn=linear_solve,
        enforce_noi_fn=enforce_noi,
        max_iterations=100,
        abstol=1e-12,
        options=options,
        use_fori_loop=use_fori_loop,
        max_nr_iters=max_nr_iters if use_fori_loop else None,
    )

    nr_solve_jit = jax.jit(nr_solve)

    fdtype = jnp.float64
    X_init = jnp.zeros(n_nodes + n_vsources, dtype=fdtype)
    X_init = X_init.at[1:n_nodes].set(0.5)  # Mid-rail init
    vsource_vals = jnp.array([1.0], dtype=fdtype)
    isource_vals = jnp.zeros(0, dtype=fdtype)
    Q_prev = jnp.zeros(n_nodes - 1, dtype=fdtype)
    device_arrays = {}

    result = nr_solve_jit(
        X_init, vsource_vals, isource_vals, Q_prev,
        jnp.array(0.0, dtype=fdtype),  # integ_c0
        device_arrays,
    )

    X_final, iterations, converged, max_f = result[0], result[1], result[2], result[3]
    return X_final, iterations, converged, max_f


def test_fori_loop_converges():
    """fori_loop NR solver converges on a simple linear circuit."""
    X, iters, converged, max_f = _run_nr_solver(use_fori_loop=True, max_nr_iters=8)
    assert converged, f"fori_loop NR did not converge after {iters} iterations, max_f={max_f}"

    # Check expected solution: V(node1) = 0.5V, V(vsrc) = 1.0V
    assert jnp.allclose(X[1], 0.5, atol=1e-10), f"V(node1) = {X[1]}, expected 0.5"
    assert jnp.allclose(X[2], 1.0, atol=1e-10), f"V(vsrc) = {X[2]}, expected 1.0"


def test_while_loop_converges():
    """while_loop NR solver converges on a simple linear circuit."""
    X, iters, converged, max_f = _run_nr_solver(use_fori_loop=False)
    assert converged, f"while_loop NR did not converge after {iters} iterations, max_f={max_f}"

    assert jnp.allclose(X[1], 0.5, atol=1e-10), f"V(node1) = {X[1]}, expected 0.5"
    assert jnp.allclose(X[2], 1.0, atol=1e-10), f"V(vsrc) = {X[2]}, expected 1.0"


def test_fori_matches_while_loop():
    """fori_loop and while_loop produce the same solution."""
    X_fori, _, converged_fori, _ = _run_nr_solver(use_fori_loop=True, max_nr_iters=8)
    X_while, _, converged_while, _ = _run_nr_solver(use_fori_loop=False)

    assert converged_fori and converged_while, "Both modes must converge"
    assert jnp.allclose(X_fori, X_while, atol=1e-10), (
        f"Solutions differ:\n  fori:  {X_fori}\n  while: {X_while}\n"
        f"  diff:  {jnp.abs(X_fori - X_while)}"
    )


def test_fori_with_min_iters():
    """fori_loop with just 2 iterations converges on a linear circuit."""
    # A linear circuit (resistive divider) should converge in ~2 NR iterations
    X, iters, converged, max_f = _run_nr_solver(use_fori_loop=True, max_nr_iters=2)
    # Even with 2 iters, a linear system should converge or be very close
    assert jnp.allclose(X[1], 0.5, atol=1e-6), f"V(node1) = {X[1]}, expected ~0.5"
    assert jnp.allclose(X[2], 1.0, atol=1e-6), f"V(vsrc) = {X[2]}, expected ~1.0"


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    test_while_loop_converges()
    print("PASS: while_loop converges")
    test_fori_loop_converges()
    print("PASS: fori_loop converges")
    test_fori_matches_while_loop()
    print("PASS: fori matches while_loop")
    test_fori_with_min_iters()
    print("PASS: fori with min iters")
    print("\nAll tests passed!")
