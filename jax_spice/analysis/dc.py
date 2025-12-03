"""DC operating point analysis for JAX-SPICE

Uses Newton-Raphson iteration to find the DC operating point
where all capacitor currents are zero and the circuit is in equilibrium.
"""

from typing import Dict, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve

from jax_spice.analysis.mna import MNASystem
from jax_spice.analysis.context import AnalysisContext


def dc_operating_point(
    system: MNASystem,
    initial_guess: Optional[Array] = None,
    max_iterations: int = 50,
    abstol: float = 1e-12,
    reltol: float = 1e-3,
    damping: float = 1.0,
) -> Tuple[Array, Dict]:
    """Find DC operating point using Newton-Raphson iteration
    
    Solves the nonlinear system: f(V) = 0
    where f is the sum of currents at each node (KCL).
    
    Newton-Raphson update:
        J(V_k) * delta_V = -f(V_k)
        V_{k+1} = V_k + damping * delta_V
    
    Args:
        system: MNA system with devices
        initial_guess: Initial voltage estimate (shape: [num_nodes])
                      If None, starts from zero
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance for convergence
        reltol: Relative tolerance for convergence
        damping: Damping factor (0 < damping <= 1)
        
    Returns:
        Tuple of (solution, info) where:
            solution: Node voltages (shape: [num_nodes])
            info: Dict with convergence information
    """
    n = system.num_nodes

    # Use float32 on Metal (no float64 support), float64 elsewhere
    dtype = jnp.float32 if jax.default_backend() == 'METAL' else jnp.float64

    # Initialize solution
    if initial_guess is not None:
        V = jnp.array(initial_guess, dtype=dtype)
    else:
        V = jnp.zeros(n, dtype=dtype)
    
    # Create DC context
    context = AnalysisContext(
        time=None,  # DC analysis
        dt=None,
        analysis_type='dc'
    )
    
    converged = False
    iterations = 0
    residual_history = []
    
    for iteration in range(max_iterations):
        context.iteration = iteration
        
        # Build Jacobian and residual
        J, f = system.build_jacobian_and_residual(V, context)
        
        # Check residual norm for convergence
        residual_norm = jnp.max(jnp.abs(f))
        residual_history.append(float(residual_norm))
        
        if residual_norm < abstol:
            converged = True
            iterations = iteration + 1
            break
        
        # Solve for Newton update: J * delta_V = -f
        try:
            delta_V = solve(J, -f)
        except Exception as e:
            # Matrix is singular - try with regularization
            reg = 1e-12 * jnp.eye(J.shape[0])
            delta_V = solve(J + reg, -f)
        
        # Update solution with damping FIRST
        # Note: V[0] is ground, stays at 0
        V = V.at[1:].add(damping * delta_V)
        iterations = iteration + 1

        # THEN check delta for convergence
        delta_norm = jnp.max(jnp.abs(delta_V))
        v_norm = jnp.max(jnp.abs(V[1:]))  # Exclude ground

        if delta_norm < abstol + reltol * max(v_norm, 1.0):
            converged = True
            break
    
    info = {
        'converged': converged,
        'iterations': iterations,
        'residual_norm': float(residual_norm),
        'delta_norm': float(delta_norm) if 'delta_norm' in dir() else 0.0,
        'residual_history': residual_history,
    }
    
    return V, info
