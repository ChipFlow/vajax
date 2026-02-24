"""Unified Newton-Raphson solver for VA-JAX.

This module provides the core NR iteration loop used by DC, transient,
and benchmark runner modules. All implementations use JAX for GPU
compatibility and JIT compilation.

The solver uses jax.lax.while_loop for fully traceable iteration that
keeps computation on-device without Python callbacks.
"""

from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import Array, lax
from jaxtyping import Bool, Num

# Scalar types for JIT-traced values
Scalar = Num[Array, ""]
IntScalar = Num[Array, ""]
BoolScalar = Bool[Array, ""]


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
) -> Tuple[jax.Array, IntScalar, BoolScalar, Scalar]:
    """JIT-compiled Newton-Raphson iteration using lax.while_loop.

    This function runs entirely on device with no host-device transfers
    during iteration.
    """

    # Wrap separate residual/jacobian functions into combined system builder
    def build_system_fn(V: jax.Array) -> Tuple[jax.Array, jax.Array]:
        f = residual_fn(V)
        J_full = jacobian_fn(V)
        J = J_full[:, 1:]  # Extract non-ground columns
        return J, f

    return _newton_loop_core(
        build_system_fn, V_init, max_iterations, abstol, reltol, damping, max_step
    )


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

    V_final, iterations, converged, residual_norm = _newton_loop_core(
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


def _newton_loop_core(
    build_system_fn: Callable[[jax.Array], Tuple[jax.Array, jax.Array]],
    V_init: jax.Array,
    max_iterations: int,
    abstol: float,
    reltol: float,
    damping: float,
    max_step: float,
    use_jax_init: bool = False,
) -> Tuple[jax.Array, IntScalar, BoolScalar, Scalar]:
    """Core JIT-compiled Newton-Raphson loop.

    All NR loop variants delegate to this function with different build_system_fn.

    Args:
        build_system_fn: Function V -> (J, f) returning reduced Jacobian and residual
        V_init: Initial voltage guess (shape: n, includes ground at index 0)
        max_iterations: Maximum number of iterations
        abstol: Absolute tolerance for residual convergence
        reltol: Relative tolerance for voltage update convergence
        damping: Damping factor for voltage updates
        max_step: Maximum allowed voltage change per iteration
        use_jax_init: If True, use JAX arrays for init_state (needed for lax.scan)

    Returns:
        Tuple of (V_final, iterations, converged, residual_norm)
    """
    if use_jax_init:
        # Use JAX arrays for lax.scan compatibility
        init_state = (V_init, jnp.array(0), jnp.array(False), jnp.array(jnp.inf))
    else:
        # Python types work fine for standalone while_loop
        init_state = (V_init, 0, False, jnp.array(jnp.inf))

    def cond_fn(state):
        V, iteration, converged, residual_norm = state
        return jnp.logical_and(~converged, iteration < max_iterations)

    def body_fn(state):
        V, iteration, _, _ = state

        # Build Jacobian and residual
        J, f = build_system_fn(V)
        residual_norm = jnp.max(jnp.abs(f))

        # Check residual convergence
        converged = residual_norm < abstol

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

    return lax.while_loop(cond_fn, body_fn, init_state)


def _newton_loop_parameterized(
    residual_fn: Callable[[jax.Array, float], jax.Array],
    V_init: jax.Array,
    param: float,
    max_iterations: int,
    abstol: float,
    reltol: float,
    damping: float,
    max_step: float,
) -> Tuple[jax.Array, IntScalar, BoolScalar, Scalar]:
    """JIT-compatible Newton loop for parameterized residuals.

    Returns raw JAX arrays, no Python type conversion, safe for lax.scan.
    """

    # Wrap parameterized residual into combined system builder with autodiff Jacobian
    def build_system_fn(V: jax.Array) -> Tuple[jax.Array, jax.Array]:
        f = residual_fn(V, param)
        J_full = jax.jacfwd(lambda v: residual_fn(v, param))(V)
        J = J_full[:, 1:]  # Extract non-ground columns
        return J, f

    return _newton_loop_core(
        build_system_fn,
        V_init,
        max_iterations,
        abstol,
        reltol,
        damping,
        max_step,
        use_jax_init=True,  # Needed for lax.scan compatibility
    )


def newton_solve_parameterized(
    residual_fn: Callable[[jax.Array, float], jax.Array],
    V_init: jax.Array,
    param: float,
    config: NRConfig | None = None,
) -> NRResult:
    """Newton-Raphson solver for parameterized residual functions.

    This variant takes a residual function f(V, param) where param is a
    scalar parameter (e.g., VDD scale for source stepping).

    Args:
        residual_fn: Function f(V, param) -> residual
        V_init: Initial voltage guess
        param: Parameter value for this solve
        config: Solver configuration

    Returns:
        NRResult with solution voltage
    """
    if config is None:
        config = NRConfig()

    V_final, iterations, converged, residual_norm = _newton_loop_parameterized(
        residual_fn,
        V_init,
        param,
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


def source_stepping_solve(
    residual_fn: Callable[[jax.Array, float], jax.Array],
    V_init: jax.Array,
    vdd_steps: jax.Array,
    config: NRConfig | None = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """JIT-compiled source stepping using lax.scan.

    Performs source stepping by solving at each VDD scale value in sequence,
    using the previous solution as the initial guess for the next step.

    Args:
        residual_fn: Parameterized residual f(V, vdd_scale) -> residual
        V_init: Initial voltage guess (typically zeros)
        vdd_steps: Array of VDD scale values (e.g., linspace(0.1, 1.0, 10))
        config: Solver configuration

    Returns:
        Tuple of (final_V, all_V, converged) where:
            final_V: Solution at final VDD step
            all_V: Solutions at all steps, shape (n_steps, n_nodes)
            converged: Boolean array indicating convergence at each step
    """
    if config is None:
        config = NRConfig()

    def step_fn(V_prev, vdd_scale):
        """Solve at one VDD step, using previous solution as initial guess."""
        # Use the raw loop function (no Python type conversion) for lax.scan compatibility
        V_final, iterations, converged, residual_norm = _newton_loop_parameterized(
            residual_fn,
            V_prev,
            vdd_scale,
            config.max_iterations,
            config.abstol,
            config.reltol,
            config.damping,
            config.max_step,
        )
        return V_final, (V_final, converged)

    final_V, (all_V, converged) = lax.scan(step_fn, V_init, vdd_steps)

    return final_V, all_V, converged
