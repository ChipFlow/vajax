"""Unified Newton-Raphson solver for JAX-SPICE.

This module provides the core NR iteration loop used by DC, transient,
and benchmark runner modules. All implementations use JAX for GPU
compatibility and JIT compilation.

The solver uses jax.lax.while_loop for fully traceable iteration that
keeps computation on-device without Python callbacks.
"""

from typing import Any, Callable, Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax


class NRConfig(NamedTuple):
    """Configuration for Newton-Raphson solver.

    Attributes:
        max_iterations: Maximum number of NR iterations
        abstol: Absolute tolerance for residual convergence
        reltol: Relative tolerance for voltage update convergence
        damping: Damping factor for voltage updates (1.0 = no damping)
        max_step: Maximum allowed voltage change per iteration
    """
    max_iterations: int = 50
    abstol: float = 1e-12
    reltol: float = 1e-3
    damping: float = 1.0
    max_step: float = 2.0


class NRResult(NamedTuple):
    """Result from Newton-Raphson solver.

    Attributes:
        V: Final voltage solution
        iterations: Number of iterations performed
        converged: Whether the solver converged
        residual_norm: Final residual norm
    """
    V: jax.Array
    iterations: int
    converged: bool
    residual_norm: float


def newton_solve(
    residual_fn: Callable[[jax.Array], jax.Array],
    jacobian_fn: Callable[[jax.Array], jax.Array],
    V_init: jax.Array,
    config: NRConfig | None = None,
) -> NRResult:
    """Solve nonlinear system using Newton-Raphson iteration.

    This is the core NR solver used by DC and transient analysis.
    It uses jax.lax.while_loop for JIT-compiled iteration.

    The system to solve is: f(V) = 0
    where f is the residual function (sum of currents at each node).

    Newton-Raphson update:
        J(V_k) * delta_V = -f(V_k)
        V_{k+1} = V_k + step_scale * delta_V

    Args:
        residual_fn: Function V -> residual vector (shape: n-1, excludes ground)
        jacobian_fn: Function V -> Jacobian matrix (shape: n-1 x n)
                    The Jacobian is computed w.r.t. full V but only non-ground
                    columns (1:) are used for the solve.
        V_init: Initial voltage guess (shape: n, includes ground at index 0)
        config: Solver configuration (uses defaults if None)

    Returns:
        NRResult with final voltage, iteration count, convergence flag, and residual
    """
    if config is None:
        config = NRConfig()

    V_final, iterations, converged, residual_norm = _newton_loop(
        residual_fn,
        jacobian_fn,
        V_init,
        config.max_iterations,
        config.abstol,
        config.reltol,
        config.damping,
        config.max_step,
    )

    return NRResult(
        V=V_final,
        iterations=int(iterations),
        converged=bool(converged),
        residual_norm=float(residual_norm),
    )


def _newton_loop(
    residual_fn: Callable,
    jacobian_fn: Callable,
    V_init: jax.Array,
    max_iterations: int,
    abstol: float,
    reltol: float,
    damping: float,
    max_step: float,
) -> Tuple[jax.Array, int, bool, float]:
    """JIT-compiled Newton-Raphson iteration using lax.while_loop.

    This function runs entirely on device with no host-device transfers
    during iteration.
    """
    # State: (V, iteration, converged, residual_norm)
    init_state = (V_init, 0, False, jnp.array(jnp.inf))

    def cond_fn(state):
        V, iteration, converged, residual_norm = state
        return jnp.logical_and(~converged, iteration < max_iterations)

    def body_fn(state):
        """Single Newton-Raphson step."""
        V, iteration, _, _ = state

        # Compute residual
        f = residual_fn(V)
        residual_norm = jnp.max(jnp.abs(f))

        # Check residual convergence
        converged = residual_norm < abstol

        # Compute Jacobian
        # J_full has shape (n-1, n) - residual w.r.t. all voltages
        J_full = jacobian_fn(V)

        # Extract columns for non-ground nodes (1:) to get square matrix
        J = J_full[:, 1:]

        # Add regularization for numerical stability
        reg = 1e-14 * jnp.eye(J.shape[0], dtype=J.dtype)
        J_reg = J + reg

        # Solve: J * delta_V = -f
        delta_V = jax.scipy.linalg.solve(J_reg, -f)

        # Apply damping and step limiting
        delta_norm = jnp.max(jnp.abs(delta_V))
        step_scale = jnp.minimum(damping, max_step / (delta_norm + 1e-15))

        # Update V (ground at index 0 stays fixed)
        V_new = V.at[1:].add(step_scale * delta_V)

        # Check delta-based convergence
        actual_delta_norm = jnp.max(jnp.abs(step_scale * delta_V))
        v_norm = jnp.max(jnp.abs(V_new[1:]))
        delta_converged = actual_delta_norm < (abstol + reltol * jnp.maximum(v_norm, 1.0))

        converged = jnp.logical_or(converged, delta_converged)

        return (V_new, iteration + 1, converged, residual_norm)

    # Run the Newton iteration loop
    V_final, iterations, converged, residual_norm = lax.while_loop(
        cond_fn, body_fn, init_state
    )

    return V_final, iterations, converged, residual_norm


def newton_solve_with_system(
    build_system_fn: Callable[[jax.Array], Tuple[jax.Array, jax.Array]],
    V_init: jax.Array,
    config: NRConfig | None = None,
) -> NRResult:
    """Solve using a combined system builder function.

    This variant takes a function that returns both Jacobian and residual,
    which can be more efficient when they share computation.

    Args:
        build_system_fn: Function V -> (Jacobian, residual)
                        Jacobian shape: (n-1, n-1) - already reduced to non-ground
                        Residual shape: (n-1,)
        V_init: Initial voltage guess (shape: n, includes ground at index 0)
        config: Solver configuration (uses defaults if None)

    Returns:
        NRResult with final voltage, iteration count, convergence flag, and residual
    """
    if config is None:
        config = NRConfig()

    V_final, iterations, converged, residual_norm = _newton_loop_system(
        build_system_fn,
        V_init,
        config.max_iterations,
        config.abstol,
        config.reltol,
        config.damping,
        config.max_step,
    )

    return NRResult(
        V=V_final,
        iterations=int(iterations),
        converged=bool(converged),
        residual_norm=float(residual_norm),
    )


def _newton_loop_system(
    build_system_fn: Callable,
    V_init: jax.Array,
    max_iterations: int,
    abstol: float,
    reltol: float,
    damping: float,
    max_step: float,
) -> Tuple[jax.Array, int, bool, float]:
    """JIT-compiled Newton loop using combined system builder."""
    init_state = (V_init, 0, False, jnp.array(jnp.inf))

    def cond_fn(state):
        V, iteration, converged, residual_norm = state
        return jnp.logical_and(~converged, iteration < max_iterations)

    def body_fn(state):
        V, iteration, _, _ = state

        # Build Jacobian and residual together
        J, f = build_system_fn(V)
        residual_norm = jnp.max(jnp.abs(f))

        # Check residual convergence
        converged = residual_norm < abstol

        # Add regularization
        reg = 1e-14 * jnp.eye(J.shape[0], dtype=J.dtype)
        J_reg = J + reg

        # Solve
        delta_V = jax.scipy.linalg.solve(J_reg, -f)

        # Apply damping and step limiting
        delta_norm = jnp.max(jnp.abs(delta_V))
        step_scale = jnp.minimum(damping, max_step / (delta_norm + 1e-15))

        # Update V (ground at index 0 stays fixed)
        V_new = V.at[1:].add(step_scale * delta_V)

        # Check delta convergence
        actual_delta_norm = jnp.max(jnp.abs(step_scale * delta_V))
        v_norm = jnp.max(jnp.abs(V_new[1:]))
        delta_converged = actual_delta_norm < (abstol + reltol * jnp.maximum(v_norm, 1.0))

        converged = jnp.logical_or(converged, delta_converged)

        return (V_new, iteration + 1, converged, residual_norm)

    return lax.while_loop(cond_fn, body_fn, init_state)
