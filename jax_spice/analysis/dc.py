"""DC operating point analysis for JAX-SPICE

Uses Newton-Raphson iteration to find the DC operating point
where all capacitor currents are zero and the circuit is in equilibrium.

This module provides GPU-accelerated DC analysis using JAX's lax.while_loop
for fully JIT-compiled Newton iteration.
"""

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from jax_spice.analysis.mna import MNASystem
from jax_spice.analysis.solver import (
    newton_solve, NRConfig, NRResult,
)
from jax_spice.analysis.gpu_backend import select_backend, get_device, get_default_dtype


def dc_operating_point(
    system: MNASystem,
    initial_guess: Optional[Array] = None,
    max_iterations: int = 50,
    abstol: float = 1e-12,
    reltol: float = 1e-3,
    damping: float = 1.0,
    vdd: float = 1.2,
    init_supplies: bool = True,
    backend: Optional[str] = None,
) -> Tuple[Array, Dict]:
    """Find DC operating point using Newton-Raphson iteration.

    This is the main entry point for DC analysis. It uses a JIT-compiled
    Newton-Raphson solver that runs entirely on the selected device
    (CPU or GPU) without host-device transfers during iteration.

    The solver finds V such that f(V) = 0, where f is the sum of currents
    at each node (KCL).

    Args:
        system: MNA system with devices
        initial_guess: Initial voltage estimate (shape: [num_nodes])
                      If None, starts from zero with supply nodes initialized
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance for convergence
        reltol: Relative tolerance for convergence
        damping: Damping factor (0 < damping <= 1)
        vdd: Supply voltage for initialization
        init_supplies: If True, initialize nodes with 'vdd' in name to vdd
        backend: 'gpu', 'cpu', or None (auto-select based on circuit size)

    Returns:
        Tuple of (solution, info) where:
            solution: Node voltages (shape: [num_nodes])
            info: Dict with convergence information
    """
    n = system.num_nodes

    # Select backend
    if backend is None or backend == "auto":
        backend = select_backend(n)

    device = get_device(backend)
    dtype = get_default_dtype(backend)

    # Check if system has device_groups for GPU path
    # If not, fall back to legacy MNA stamping approach
    has_device_groups = hasattr(system, 'device_groups') and len(system.device_groups) > 0

    if has_device_groups:
        # GPU path: Use vectorized device groups with autodiff
        residual_fn = system.build_gpu_residual_fn(vdd=vdd, gmin=1e-12)
        jacobian_fn = jax.jacfwd(residual_fn)
    else:
        # Legacy path: Use MNA stamping with individual device eval_fn
        # This path supports manually constructed MNASystem objects
        from jax_spice.analysis.context import AnalysisContext

        context = AnalysisContext(
            time=None,
            dt=None,
            analysis_type='dc'
        )

        def build_system_from_mna(V):
            """Build system using MNA stamping."""
            ctx = AnalysisContext(time=None, dt=None, analysis_type='dc')
            return system.build_jacobian_and_residual(V, ctx)

        # For legacy path, use system builder approach
        return _dc_operating_point_legacy(
            system, initial_guess, max_iterations, abstol, reltol,
            damping, vdd, init_supplies, backend, device, dtype
        )

    # GPU path continues here
    with jax.default_device(device):
        if initial_guess is not None:
            V_init = jnp.array(initial_guess, dtype=dtype)
        else:
            V_init = jnp.zeros(n, dtype=dtype)

            # Initialize supply nodes if requested
            if init_supplies:
                for name, idx in system.node_names.items():
                    name_lower = name.lower()
                    if "vdd" in name_lower:
                        V_init = V_init.at[idx].set(vdd)

        # Configure solver
        config = NRConfig(
            max_iterations=max_iterations,
            abstol=abstol,
            reltol=reltol,
            damping=damping,
            max_step=2.0,
        )

        # Run unified NR solver
        result = newton_solve(residual_fn, jacobian_fn, V_init, config)

    info = {
        "converged": result.converged,
        "iterations": result.iterations,
        "residual_norm": result.residual_norm,
        "backend": backend,
        "device": str(device),
    }

    return result.V, info


def _dc_operating_point_legacy(
    system: MNASystem,
    initial_guess: Optional[Array],
    max_iterations: int,
    abstol: float,
    reltol: float,
    damping: float,
    vdd: float,
    init_supplies: bool,
    backend: str,
    device,
    dtype,
) -> Tuple[Array, Dict]:
    """Legacy DC solver using MNA stamping with individual device eval_fn.

    This is used when device_groups are not available (e.g., manually
    constructed MNASystem for testing).
    """
    from jax.scipy.linalg import solve
    from jax_spice.analysis.context import AnalysisContext

    n = system.num_nodes

    with jax.default_device(device):
        if initial_guess is not None:
            V = jnp.array(initial_guess, dtype=dtype)
        else:
            V = jnp.zeros(n, dtype=dtype)

        context = AnalysisContext(
            time=None,
            dt=None,
            analysis_type='dc'
        )

        converged = False
        iterations = 0
        residual_norm = float('inf')
        delta_norm = 0.0

        for iteration in range(max_iterations):
            context.iteration = iteration

            # Build Jacobian and residual using MNA stamping
            J, f = system.build_jacobian_and_residual(V, context)

            # Check residual convergence
            residual_norm = float(jnp.max(jnp.abs(f)))

            if residual_norm < abstol:
                converged = True
                iterations = iteration + 1
                break

            # Solve for Newton update
            try:
                delta_V = solve(J, -f)
            except Exception:
                reg = 1e-12 * jnp.eye(J.shape[0])
                delta_V = solve(J + reg, -f)

            # Apply update with damping
            V = V.at[1:].add(damping * delta_V)
            iterations = iteration + 1

            # Check delta convergence
            delta_norm = float(jnp.max(jnp.abs(delta_V)))
            v_norm = float(jnp.max(jnp.abs(V[1:])))

            if delta_norm < abstol + reltol * max(v_norm, 1.0):
                converged = True
                break

    info = {
        "converged": converged,
        "iterations": iterations,
        "residual_norm": residual_norm,
        "delta_norm": delta_norm,
        "backend": backend,
        "device": str(device),
        "mode": "legacy",
    }

    return V, info


# Backwards compatibility aliases
dc_operating_point_gpu = dc_operating_point
dc_operating_point_sparse = dc_operating_point
