"""DC analysis: operating point and DC sweep

Uses Newton-Raphson iteration implemented in JAX for automatic differentiation and GPU acceleration.
"""

from typing import Dict, List, Tuple, Optional
import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from jax_spice.circuit import Circuit
from jax_spice.analysis.context import AnalysisContext


def newton_raphson_step(circuit_build_fn, V: Array) -> Tuple[Array, Array, Array]:
    """Single Newton-Raphson iteration

    Solves: J * δV = -F(V)
    Updates: V_new = V + δV

    Args:
        circuit_build_fn: Function that builds F(V) and J
        V: Current voltage solution

    Returns:
        (V_new, residual_norm, delta_norm)
    """
    # Build system at current operating point
    residual, jacobian = circuit_build_fn(V)

    # Solve linear system: J * δV = -residual
    # Use JAX's linear solver (LU decomposition)
    delta_V = jnp.linalg.solve(jacobian, -residual)

    # Update voltages
    V_new = V + delta_V

    # Compute norms for convergence check
    res_norm = jnp.linalg.norm(residual)
    delta_norm = jnp.linalg.norm(delta_V)

    return V_new, res_norm, delta_norm


def dc_operating_point(
    circuit: Circuit,
    V_initial: Optional[Array] = None,
    max_iter: int = 50,
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-6,
    verbose: bool = False,
    temperature: float = 300.0,
) -> Tuple[Array, bool, int]:
    """Find DC operating point using Newton-Raphson

    Args:
        circuit: Circuit to analyze
        V_initial: Initial voltage guess (if None, uses zeros)
        max_iter: Maximum Newton iterations
        abs_tol: Absolute tolerance on residual norm
        rel_tol: Relative tolerance on voltage update
        verbose: Print convergence information
        temperature: Circuit temperature in Kelvin (default 300K = 27C)

    Returns:
        (solution_voltages, converged, num_iterations)
    """
    n = circuit.num_unknowns()

    # Create DC analysis context
    context = AnalysisContext.dc(temperature=temperature)

    # Initial guess
    if V_initial is None:
        V = jnp.zeros(n)
    else:
        V = jnp.array(V_initial)

    # Create circuit evaluation function (no JIT yet - python loop is fine for now)
    def build_fn(v):
        return circuit.build_system(v, context=context)

    converged = False
    for iteration in range(max_iter):
        V_new, res_norm, delta_norm = newton_raphson_step(build_fn, V)

        if verbose:
            print(f"Iteration {iteration:3d}: ||F||={res_norm:.3e}, ||δV||={delta_norm:.3e}")

        # Check convergence
        if res_norm < abs_tol and delta_norm < rel_tol * (jnp.linalg.norm(V_new) + 1e-10):
            converged = True
            V = V_new
            if verbose:
                print(f"Converged in {iteration + 1} iterations")
            break

        V = V_new

        # Check for divergence
        if jnp.any(jnp.isnan(V)) or jnp.any(jnp.isinf(V)):
            if verbose:
                print("Newton-Raphson diverged (NaN/Inf detected)")
            break

    if not converged and verbose:
        print(f"Failed to converge in {max_iter} iterations")

    return V, converged, iteration + 1


def dc_sweep(
    circuit: Circuit,
    sweep_source: str,
    start: float,
    stop: float,
    points: int,
    verbose: bool = False,
    temperature: float = 300.0,
) -> Tuple[np.ndarray, np.ndarray, List[bool]]:
    """DC sweep analysis

    Sweeps a voltage source and computes operating point at each value.
    Uses previous solution as initial guess (continuation).

    Args:
        circuit: Circuit to analyze
        sweep_source: Name of voltage source to sweep
        start: Start voltage
        stop: Stop voltage
        points: Number of sweep points
        verbose: Print progress
        temperature: Circuit temperature in Kelvin (default 300K = 27C)

    Returns:
        (sweep_values, solution_matrix, converged_list)
        where solution_matrix[i, j] is voltage of node j at sweep point i
    """
    if sweep_source not in circuit.vsources:
        raise ValueError(f"Voltage source '{sweep_source}' not found in circuit")

    sweep_values = np.linspace(start, stop, points)
    n = circuit.num_unknowns()
    solutions = np.zeros((points, n))
    converged = []

    # Get original source spec
    pos_node, neg_node, original_voltage = circuit.vsources[sweep_source]

    # Initial guess
    V_guess = None

    for i, V_sweep in enumerate(sweep_values):
        if verbose and i % max(1, points // 10) == 0:
            print(f"DC sweep: {i}/{points} ({V_sweep:.3f} V)")

        # Update voltage source
        circuit.vsources[sweep_source] = (pos_node, neg_node, V_sweep)

        # Solve operating point (use previous solution as guess)
        V_solution, conv, num_iter = dc_operating_point(
            circuit,
            V_initial=V_guess,
            verbose=False,
            temperature=temperature,
        )

        solutions[i, :] = np.array(V_solution)
        converged.append(conv)

        # Use this solution as initial guess for next point
        V_guess = V_solution

        if not conv and verbose:
            print(f"  Warning: Failed to converge at V={V_sweep:.3f}V")

    # Restore original voltage
    circuit.vsources[sweep_source] = (pos_node, neg_node, original_voltage)

    return sweep_values, solutions, converged


def get_node_voltage(
    circuit: Circuit,
    solution: Array,
    node_name: str
) -> float:
    """Extract voltage of a specific node from solution vector

    Args:
        circuit: Circuit
        solution: Solution vector from dc_operating_point or dc_sweep
        node_name: Node to query

    Returns:
        Node voltage (V)
    """
    if node_name == circuit.ground_node:
        return 0.0

    node_idx = circuit.get_node_index(node_name)
    ground_idx = circuit.get_node_index(circuit.ground_node) if circuit.ground_node else -1

    # Map to reduced system index
    if ground_idx >= 0 and node_idx > ground_idx:
        reduced_idx = node_idx - 1
    else:
        reduced_idx = node_idx

    return float(solution[reduced_idx])
