"""Transient analysis for JAX-SPICE

Implements time-domain simulation using backward Euler integration
with Newton-Raphson iteration at each timestep.

Backward Euler formulation:
    For capacitor: I = C * dV/dt ≈ C * (V_n - V_{n-1}) / h
    
    This is equivalent to a companion resistor model:
    I = G_eq * V_n + I_eq
    where:
        G_eq = C / h
        I_eq = -C * V_{n-1} / h
"""

from typing import Dict, List, Optional, Tuple, Callable
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve

from jax_spice.analysis.mna import MNASystem
from jax_spice.analysis.context import AnalysisContext
from jax_spice.analysis.dc import dc_operating_point


def transient_analysis(
    system: MNASystem,
    t_stop: float,
    t_step: float,
    t_start: float = 0.0,
    initial_conditions: Optional[Dict[str, float]] = None,
    max_iterations: int = 20,
    abstol: float = 1e-12,
    reltol: float = 1e-3,
    save_all: bool = True,
) -> Tuple[Array, Array, Dict]:
    """Run transient analysis
    
    Simulates the circuit from t_start to t_stop using backward Euler
    integration with fixed timestep.
    
    Args:
        system: MNA system with devices
        t_stop: End time
        t_step: Timestep (fixed)
        t_start: Start time (default 0)
        initial_conditions: Optional dict of node_name -> initial voltage
        max_iterations: Max NR iterations per timepoint
        abstol: Absolute convergence tolerance
        reltol: Relative convergence tolerance
        save_all: If True, save solution at every timestep
        
    Returns:
        Tuple of (times, solutions, info) where:
            times: Array of time points
            solutions: Array of solutions, shape [num_times, num_nodes]
            info: Dict with simulation statistics
    """
    n = system.num_nodes

    # Use float32 on Metal (no float64 support), float64 elsewhere
    dtype = jnp.float32 if jax.default_backend() == 'METAL' else jnp.float64

    # Find DC operating point for initial condition
    if initial_conditions is not None:
        # Build initial guess from ICs
        V0 = jnp.zeros(n, dtype=dtype)
        for name, voltage in initial_conditions.items():
            idx = system.node_names.get(name)
            if idx is not None and idx > 0:
                V0 = V0.at[idx].set(voltage)
    else:
        # Use DC operating point
        V0, dc_info = dc_operating_point(system)
        if not dc_info['converged']:
            raise RuntimeError(f"DC operating point did not converge: {dc_info}")
    
    # Generate time points
    num_points = int((t_stop - t_start) / t_step) + 1
    times = jnp.linspace(t_start, t_stop, num_points)
    
    # Storage for results
    solutions = [V0]
    convergence_info = []
    
    # Current solution
    V_prev = V0
    V_curr = V0.copy()
    
    # Time stepping loop
    for i, t in enumerate(times[1:], 1):
        dt = float(times[i] - times[i-1])
        
        # Create transient context with backward Euler coefficients
        # For backward Euler: q_dot = (q - q_prev) / dt
        # So: c0 = 1/dt, c1 = -1/dt
        context = AnalysisContext(
            time=float(t),
            dt=dt,
            analysis_type='tran',
            c0=1.0 / dt,
            c1=-1.0 / dt,
            rhs_correction=0.0,
        )
        
        # Newton-Raphson iteration for this timepoint
        converged = False
        V_iter = V_curr.copy()
        
        for iteration in range(max_iterations):
            context.iteration = iteration
            context.prev_solution = {'voltages': V_prev}
            
            # Build Jacobian and residual with transient contributions
            J, f = _build_transient_system(
                system, V_iter, V_prev, context
            )
            
            # Check convergence
            residual_norm = jnp.max(jnp.abs(f))
            if residual_norm < abstol:
                converged = True
                break
            
            # Solve for update
            try:
                delta_V = solve(J, -f)
            except Exception:
                reg = 1e-12 * jnp.eye(J.shape[0])
                delta_V = solve(J + reg, -f)
            
            # Apply update FIRST
            V_iter = V_iter.at[1:].add(delta_V)

            # THEN check delta convergence
            delta_norm = jnp.max(jnp.abs(delta_V))
            v_norm = jnp.max(jnp.abs(V_iter[1:]))

            if delta_norm < abstol + reltol * max(v_norm, 1.0):
                converged = True
                break
        
        convergence_info.append({
            'time': float(t),
            'converged': converged,
            'iterations': iteration + 1,
            'residual_norm': float(residual_norm),
        })
        
        if not converged:
            print(f"Warning: timepoint t={t:.3e} did not converge after {max_iterations} iterations")
        
        # Accept timestep
        V_prev = V_iter
        V_curr = V_iter
        
        if save_all:
            solutions.append(V_iter)
    
    # Stack solutions
    solutions_array = jnp.stack(solutions)
    
    info = {
        'num_timepoints': num_points,
        'converged_all': all(c['converged'] for c in convergence_info),
        'convergence_info': convergence_info,
    }
    
    return times, solutions_array, info


def _build_transient_system(
    system: MNASystem,
    V_curr: Array,
    V_prev: Array,
    context: AnalysisContext
) -> Tuple[Array, Array]:
    """Build Jacobian and residual for transient analysis

    Includes both resistive and reactive contributions.
    For backward Euler, reactive elements contribute:
        J_reactive = C / dt
        f_reactive = C * (V_curr - V_prev) / dt

    Args:
        system: MNA system
        V_curr: Current voltage estimate
        V_prev: Previous timepoint voltage
        context: Analysis context with integration coefficients

    Returns:
        Tuple of (jacobian, residual)
    """
    n = system.num_nodes - 1

    # Use float32 on Metal (no float64 support), float64 elsewhere
    dtype = jnp.float32 if jax.default_backend() == 'METAL' else jnp.float64

    # Initialize
    jacobian = jnp.zeros((n, n), dtype=dtype)
    residual = jnp.zeros(n, dtype=dtype)

    dt = context.dt
    c0 = context.c0  # 1/dt for backward Euler

    # Evaluate each device
    for device in system.devices:
        if device.eval_fn is None:
            continue

        # Build voltage dictionaries
        dev_voltages = {}
        dev_voltages_prev = {}
        for term, node_idx in zip(device.terminals, device.node_indices):
            dev_voltages[term] = V_curr[node_idx]
            dev_voltages_prev[term] = V_prev[node_idx]

        # Evaluate device (returns stamps with currents and conductances)
        stamps = device.eval_fn(dev_voltages, device.params, context)

        # Stamp resistive currents into residual
        for term, current in stamps.currents.items():
            term_idx = device.terminals.index(term)
            node_idx = device.node_indices[term_idx]
            if node_idx != system.ground_node:
                residual = residual.at[node_idx - 1].add(current)

        # Stamp resistive conductances into Jacobian
        # Handle stamp format where entries can be diagonal or off-diagonal
        for (term_i, term_j), conductance in stamps.conductances.items():
            idx_i = device.terminals.index(term_i)
            idx_j = device.terminals.index(term_j)
            node_i = device.node_indices[idx_i]
            node_j = device.node_indices[idx_j]

            # Stamp into Jacobian (skip ground rows/cols)
            if node_i != system.ground_node:
                if node_j != system.ground_node:
                    jacobian = jacobian.at[node_i - 1, node_j - 1].add(conductance)
                # If node_j is ground, we still add to diagonal (absorbed into RHS)
                elif term_i == term_j:  # Self-conductance to ground
                    jacobian = jacobian.at[node_i - 1, node_i - 1].add(conductance)

        # Handle reactive contributions (capacitances)
        # Capacitor between nodes i and j:
        # I = C * d(Vi - Vj)/dt
        # For BE: I = C/dt * (Vi - Vj) - C/dt * (Vi_prev - Vj_prev)
        if hasattr(stamps, 'capacitances') and stamps.capacitances:
            # Get the two terminal node indices
            node_p = device.node_indices[0]
            node_n = device.node_indices[1]

            # Get capacitance value (from the stamp, e.g., ('p', 'p') entry)
            cap_value = None
            for key, val in stamps.capacitances.items():
                if key[0] == key[1]:  # Diagonal entry has the capacitance
                    cap_value = abs(float(val))
                    break

            if cap_value is not None and cap_value > 0:
                G_eq = cap_value / dt  # Equivalent conductance

                # Current voltage across capacitor
                V_cap = V_curr[node_p] - V_curr[node_n]
                V_cap_prev = V_prev[node_p] - V_prev[node_n]

                # Backward Euler companion model:
                # Capacitor current: I = C * dV/dt ≈ C/dt * (V - V_prev) = G_eq * (V - V_prev)
                #
                # This current flows through the capacitor from p to n when charging.
                # In KCL terms: current LEAVING node_p = I_cap (flows into capacitor)
                #               current LEAVING node_n = -I_cap (flows out of capacitor)
                #
                # Residual contribution (current leaving node):
                #   node_p: +I_cap = G_eq * (V_cap - V_cap_prev)
                #   node_n: -I_cap
                #
                # Jacobian = d(residual)/dV:
                #   d(I_cap)/dVp = G_eq, d(I_cap)/dVn = -G_eq

                I_cap = G_eq * (V_cap - V_cap_prev)

                # Stamp residual (full capacitor current)
                if node_p != system.ground_node:
                    residual = residual.at[node_p - 1].add(I_cap)
                    jacobian = jacobian.at[node_p - 1, node_p - 1].add(G_eq)
                    if node_n != system.ground_node:
                        jacobian = jacobian.at[node_p - 1, node_n - 1].add(-G_eq)

                if node_n != system.ground_node:
                    residual = residual.at[node_n - 1].add(-I_cap)
                    jacobian = jacobian.at[node_n - 1, node_n - 1].add(G_eq)
                    if node_p != system.ground_node:
                        jacobian = jacobian.at[node_n - 1, node_p - 1].add(-G_eq)

    return jacobian, residual
