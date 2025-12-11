"""Transient analysis for JAX-SPICE

Implements time-domain simulation using backward Euler integration
with Newton-Raphson iteration at each timestep.

This module provides two implementations:
1. transient_analysis(): Python-loop based, flexible but slower
2. transient_analysis_jit(): JIT-compiled using jax.lax.scan, much faster

Backward Euler formulation:
    For capacitor: I = C * dV/dt ≈ C * (V_n - V_{n-1}) / h

    This is equivalent to a companion resistor model:
    I = G_eq * V_n + I_eq
    where:
        G_eq = C / h
        I_eq = -C * V_{n-1} / h
"""

from typing import Dict, List, Optional, Tuple, Callable, NamedTuple
from functools import partial
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve
import jax.lax as lax

from jax_spice.analysis.mna import MNASystem
from jax_spice.analysis.context import AnalysisContext
from jax_spice.analysis.dc import dc_operating_point


class CircuitData(NamedTuple):
    """Pre-compiled circuit data for JIT-compatible simulation

    All arrays have consistent dtypes for JAX tracing.
    """
    num_nodes: int
    ground_node: int

    # Device connectivity: [num_devices, 2] for two-terminal devices
    # node_indices[i] = [node_p, node_n] for device i
    device_node_indices: Array

    # Device types: 0=resistor, 1=capacitor, 2=vsource
    device_types: Array

    # Device parameters: [num_devices, max_params]
    # For resistor: [R, 0, 0], capacitor: [C, 0, 0], vsource: [V, 0, 0]
    device_params: Array

    # Number of devices
    num_devices: int


def _compile_circuit(system: MNASystem, dtype) -> CircuitData:
    """Convert MNASystem to JIT-compatible arrays

    Args:
        system: MNA system with devices
        dtype: JAX dtype to use

    Returns:
        CircuitData with pre-compiled arrays
    """
    num_devices = len(system.devices)

    # Pre-allocate arrays
    node_indices = []
    device_types = []
    device_params = []

    for device in system.devices:
        # Get node indices (assume 2-terminal devices)
        if len(device.node_indices) >= 2:
            node_indices.append([device.node_indices[0], device.node_indices[1]])
        else:
            node_indices.append([device.node_indices[0], 0])

        # Determine device type and extract parameter
        # Types: 0=resistor, 1=capacitor, 2=vsource, 3=isource, 4=diode
        model = device.model_name.lower()
        if 'resistor' in model or model == 'r':
            device_types.append(0)
            R = device.params.get('r', device.params.get('R', 1000.0))
            device_params.append([float(R), 0.0, 0.0])
        elif 'capacitor' in model or model == 'c':
            device_types.append(1)
            C = device.params.get('c', device.params.get('C', 1e-12))
            device_params.append([float(C), 0.0, 0.0])
        elif 'vsource' in model or model == 'v':
            device_types.append(2)
            V = device.params.get('v', device.params.get('dc', 0.0))
            device_params.append([float(V), 0.0, 0.0])
        elif 'isource' in model or model == 'i':
            device_types.append(3)
            I = device.params.get('i', device.params.get('dc', 0.0))
            device_params.append([float(I), 0.0, 0.0])
        elif 'diode' in model or model == 'd':
            device_types.append(4)
            Is = device.params.get('is', 1e-14)
            n = device.params.get('n', 1.0)
            # params: [Is, n, Vt] where Vt = kT/q ≈ 0.0258V at 300K
            device_params.append([float(Is), float(n), 0.0258])
        else:
            # Default to resistor-like behavior
            device_types.append(0)
            device_params.append([1000.0, 0.0, 0.0])

    return CircuitData(
        num_nodes=system.num_nodes,
        ground_node=system.ground_node,
        device_node_indices=jnp.array(node_indices, dtype=jnp.int32),
        device_types=jnp.array(device_types, dtype=jnp.int32),
        device_params=jnp.array(device_params, dtype=dtype),
        num_devices=num_devices,
    )


def _stamp_device(
    jacobian: Array,
    residual: Array,
    V_curr: Array,
    V_prev: Array,
    node_p: int,
    node_n: int,
    device_type: int,
    params: Array,
    dt: float,
    ground_node: int,
) -> Tuple[Array, Array]:
    """Stamp a single device into Jacobian and residual

    This function is designed to be vmap'd over devices.

    Args:
        jacobian: Current Jacobian matrix
        residual: Current residual vector
        V_curr: Current voltage vector
        V_prev: Previous timestep voltage vector
        node_p: Positive node index
        node_n: Negative node index
        device_type: 0=resistor, 1=capacitor, 2=vsource, 3=isource, 4=diode
        params: Device parameters [p0, p1, p2]
        dt: Timestep
        ground_node: Ground node index

    Returns:
        Updated (jacobian, residual)
    """
    Vp = V_curr[node_p]
    Vn = V_curr[node_n]
    V = Vp - Vn

    Vp_prev = V_prev[node_p]
    Vn_prev = V_prev[node_n]
    V_prev_diff = Vp_prev - Vn_prev

    # Compute stamps based on device type
    # Resistor: I = G * V, G = 1/R
    R = params[0]
    G_resistor = 1.0 / jnp.maximum(R, 1e-12)
    I_resistor = G_resistor * V

    # Capacitor: I = G_eq * (V - V_prev), G_eq = C/dt
    C = params[0]
    G_cap = C / jnp.maximum(dt, 1e-15)
    I_cap = G_cap * (V - V_prev_diff)

    # Voltage source: I = G_big * (V - V_target)
    V_target = params[0]
    G_vsource = 1e12
    I_vsource = G_vsource * (V - V_target)

    # Current source: I = I_dc (constant current, no conductance)
    I_dc = params[0]
    G_isource = 0.0
    I_isource = I_dc

    # Diode: Shockley equation with limiting
    # params: [Is, n, Vt]
    Is = params[0]
    n_diode = params[1]
    Vt = params[2]
    nVt = n_diode * Vt

    # Limit exponential argument to avoid overflow
    Vd_norm = V / jnp.maximum(nVt, 1e-12)
    Vd_limited = jnp.clip(Vd_norm, -40.0, 40.0)

    # Compute diode current and conductance
    exp_term = jnp.exp(Vd_limited)
    I_diode_base = Is * (exp_term - 1.0)
    G_diode_base = Is * exp_term / jnp.maximum(nVt, 1e-12)

    # Handle extreme forward bias with linear extrapolation
    exp_40 = jnp.exp(40.0)
    I_diode_high = Is * (exp_40 + exp_40 * (Vd_norm - 40.0) - 1.0)
    G_diode_high = Is * exp_40 / jnp.maximum(nVt, 1e-12)

    # Select appropriate diode model region
    I_diode = jnp.where(Vd_norm > 40.0, I_diode_high, I_diode_base)
    G_diode = jnp.where(Vd_norm > 40.0, G_diode_high, G_diode_base)

    # Minimum conductance for numerical stability
    G_diode = jnp.maximum(G_diode, 1e-12)

    # Select based on device type
    is_resistor = device_type == 0
    is_capacitor = device_type == 1
    is_vsource = device_type == 2
    is_isource = device_type == 3
    is_diode = device_type == 4

    # Combined conductance and current
    G = jnp.where(is_resistor, G_resistor,
                  jnp.where(is_capacitor, G_cap,
                            jnp.where(is_vsource, G_vsource,
                                      jnp.where(is_isource, G_isource,
                                                jnp.where(is_diode, G_diode, 0.0)))))

    I = jnp.where(is_resistor, I_resistor,
                  jnp.where(is_capacitor, I_cap,
                            jnp.where(is_vsource, I_vsource,
                                      jnp.where(is_isource, I_isource,
                                                jnp.where(is_diode, I_diode, 0.0)))))

    # Stamp into Jacobian and residual
    # Node p (row node_p - 1 if not ground)
    p_row = node_p - 1
    n_row = node_n - 1

    # Create masks for ground handling
    p_not_ground = node_p != ground_node
    n_not_ground = node_n != ground_node

    # Stamp residual
    residual = jnp.where(
        p_not_ground,
        residual.at[p_row].add(I),
        residual
    )
    residual = jnp.where(
        n_not_ground,
        residual.at[n_row].add(-I),
        residual
    )

    # Stamp Jacobian diagonal
    jacobian = jnp.where(
        p_not_ground,
        jacobian.at[p_row, p_row].add(G),
        jacobian
    )
    jacobian = jnp.where(
        n_not_ground,
        jacobian.at[n_row, n_row].add(G),
        jacobian
    )

    # Stamp Jacobian off-diagonal
    jacobian = jnp.where(
        p_not_ground & n_not_ground,
        jacobian.at[p_row, n_row].add(-G),
        jacobian
    )
    jacobian = jnp.where(
        p_not_ground & n_not_ground,
        jacobian.at[n_row, p_row].add(-G),
        jacobian
    )

    return jacobian, residual


def _build_system_jit(
    circuit: CircuitData,
    V_curr: Array,
    V_prev: Array,
    dt: float,
    dtype,
) -> Tuple[Array, Array]:
    """Build Jacobian and residual using JIT-compatible operations

    Args:
        circuit: Pre-compiled circuit data
        V_curr: Current voltage estimate
        V_prev: Previous timestep voltage
        dt: Timestep
        dtype: JAX dtype

    Returns:
        Tuple of (jacobian, residual)
    """
    # Derive n from voltage vector shape (V has ground at index 0)
    n = V_curr.shape[0] - 1

    # Initialize
    jacobian = jnp.zeros((n, n), dtype=dtype)
    residual = jnp.zeros(n, dtype=dtype)

    # Process each device using lax.fori_loop
    def stamp_one_device(i, carry):
        J, r = carry
        node_p = circuit.device_node_indices[i, 0]
        node_n = circuit.device_node_indices[i, 1]
        dev_type = circuit.device_types[i]
        params = circuit.device_params[i]

        J_new, r_new = _stamp_device(
            J, r, V_curr, V_prev,
            node_p, node_n, dev_type, params,
            dt, circuit.ground_node
        )
        return (J_new, r_new)

    jacobian, residual = lax.fori_loop(
        0, circuit.num_devices,
        stamp_one_device,
        (jacobian, residual)
    )

    return jacobian, residual


def _newton_step(
    circuit: CircuitData,
    V_iter: Array,
    V_prev: Array,
    dt: float,
    abstol: float,
    reltol: float,
) -> Tuple[Array, bool, float]:
    """Perform one Newton-Raphson iteration

    Args:
        circuit: Pre-compiled circuit data
        V_iter: Current voltage estimate
        V_prev: Previous timestep voltage
        dt: Timestep
        abstol: Absolute tolerance
        reltol: Relative tolerance

    Returns:
        Tuple of (updated_V, converged, residual_norm)
    """
    # Derive dtype from input arrays
    dtype = V_iter.dtype

    # Build system
    J, f = _build_system_jit(circuit, V_iter, V_prev, dt, dtype)

    # Check residual convergence
    residual_norm = jnp.max(jnp.abs(f))

    # Solve for update with regularization for numerical stability
    reg = 1e-12 * jnp.eye(J.shape[0], dtype=dtype)
    delta_V = solve(J + reg, -f)

    # Apply update (skip ground node at index 0)
    V_new = V_iter.at[1:].add(delta_V)

    # Check convergence
    delta_norm = jnp.max(jnp.abs(delta_V))
    v_norm = jnp.max(jnp.abs(V_new[1:]))
    converged = (residual_norm < abstol) | (delta_norm < abstol + reltol * jnp.maximum(v_norm, 1.0))

    return V_new, converged, residual_norm


def _newton_solve(
    circuit: CircuitData,
    V_init: Array,
    V_prev: Array,
    dt: float,
    max_iterations: int,
    abstol: float,
    reltol: float,
) -> Tuple[Array, int, float]:
    """Solve one timestep using Newton-Raphson with jax.lax.while_loop

    Args:
        circuit: Pre-compiled circuit data
        V_init: Initial voltage guess
        V_prev: Previous timestep voltage
        dt: Timestep
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance
        reltol: Relative tolerance

    Returns:
        Tuple of (solution, iterations, final_residual)
    """
    # Derive dtype from the input arrays
    dtype = V_init.dtype

    def cond_fn(state):
        V, iteration, converged, _ = state
        return (~converged) & (iteration < max_iterations)

    def body_fn(state):
        V, iteration, _, residual = state
        V_new, converged, new_residual = _newton_step(
            circuit, V, V_prev, dt, abstol, reltol
        )
        return (V_new, iteration + 1, converged, new_residual)

    # Initial state
    init_state = (V_init, 0, False, jnp.array(1e10, dtype=dtype))

    # Run Newton iteration
    final_V, iterations, converged, final_residual = lax.while_loop(
        cond_fn, body_fn, init_state
    )

    return final_V, iterations, final_residual


def _timestep_fn(carry, t_idx):
    """Process one timestep - designed for jax.lax.scan

    Args:
        carry: Tuple of (V_prev, circuit, dt, max_iter, abstol, reltol)
        t_idx: Time index (not used directly, just for iteration)

    Returns:
        (new_carry, output) where output is the solution at this timestep
    """
    V_prev, circuit, dt, max_iter, abstol, reltol = carry

    # Solve for this timestep - derive dtype from V_prev
    V_new, iterations, residual = _newton_solve(
        circuit, V_prev, V_prev, dt, max_iter, abstol, reltol
    )

    # Return new carry and output
    new_carry = (V_new, circuit, dt, max_iter, abstol, reltol)
    return new_carry, V_new


@partial(jax.jit, static_argnames=['num_timesteps', 'max_iterations'])
def _run_simulation_jit(
    circuit: CircuitData,
    V0: Array,
    dt: float,
    num_timesteps: int,
    max_iterations: int,
    abstol: float,
    reltol: float,
) -> Array:
    """JIT-compiled simulation loop using jax.lax.scan

    Args:
        circuit: Pre-compiled circuit data
        V0: Initial voltage vector
        dt: Timestep
        num_timesteps: Number of timesteps to simulate
        max_iterations: Max NR iterations per timestep
        abstol: Absolute tolerance
        reltol: Relative tolerance

    Returns:
        Array of solutions, shape [num_timesteps + 1, num_nodes]
    """
    # Initial carry - dtype is derived from V0.dtype in subfunctions
    init_carry = (V0, circuit, dt, max_iterations, abstol, reltol)

    # Run scan over timesteps
    _, solutions = lax.scan(
        _timestep_fn,
        init_carry,
        jnp.arange(num_timesteps),
    )

    # Prepend initial condition
    all_solutions = jnp.concatenate([V0[None, :], solutions], axis=0)

    return all_solutions


def transient_analysis_jit(
    system: MNASystem,
    t_stop: float,
    t_step: float,
    t_start: float = 0.0,
    initial_conditions: Optional[Dict[str, float]] = None,
    max_iterations: int = 20,
    abstol: float = 1e-12,
    reltol: float = 1e-3,
    backend: Optional[str] = None,
) -> Tuple[Array, Array, Dict]:
    """Run JIT-compiled transient analysis

    This version uses jax.lax.scan to JIT-compile the entire time loop,
    providing significant speedup over the Python-loop version.

    Args:
        system: MNA system with devices
        t_stop: End time
        t_step: Timestep (fixed)
        t_start: Start time (default 0)
        initial_conditions: Optional dict of node_name -> initial voltage
        max_iterations: Max NR iterations per timepoint
        abstol: Absolute convergence tolerance
        reltol: Relative convergence tolerance
        backend: 'gpu', 'cpu', or None (auto-select based on circuit size)

    Returns:
        Tuple of (times, solutions, info) where:
            times: Array of time points
            solutions: Array of solutions, shape [num_times, num_nodes]
            info: Dict with simulation statistics
    """
    from jax_spice.analysis.gpu_backend import select_backend, get_device, get_default_dtype

    n = system.num_nodes

    # Select backend
    if backend is None or backend == "auto":
        backend = select_backend(n)

    device = get_device(backend)
    dtype = get_default_dtype(backend)

    # Run on selected device
    with jax.default_device(device):
        # Build initial conditions
        if initial_conditions is not None:
            V0 = jnp.zeros(n, dtype=dtype)
            for name, voltage in initial_conditions.items():
                idx = system.node_names.get(name)
                if idx is not None and idx > 0:
                    V0 = V0.at[idx].set(voltage)
        else:
            V0, dc_info = dc_operating_point(system)
            V0 = V0.astype(dtype)
            if not dc_info['converged']:
                raise RuntimeError(f"DC operating point did not converge: {dc_info}")

        # Generate time points
        num_timesteps = int((t_stop - t_start) / t_step)
        times = jnp.linspace(t_start, t_stop, num_timesteps + 1)

        # Compile circuit to JIT-compatible format
        circuit = _compile_circuit(system, dtype)

        # Run JIT-compiled simulation
        solutions = _run_simulation_jit(
            circuit, V0, t_step, num_timesteps, max_iterations,
            abstol, reltol
        )

    info = {
        'num_timepoints': num_timesteps + 1,
        'jit_compiled': True,
        'backend': backend,
        'device': str(device),
    }

    return times, solutions, info


# Keep original implementation for compatibility and debugging
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
    use_jit: bool = True,
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
        use_jit: If True, use JIT-compiled version (default)

    Returns:
        Tuple of (times, solutions, info) where:
            times: Array of time points
            solutions: Array of solutions, shape [num_times, num_nodes]
            info: Dict with simulation statistics
    """
    if use_jit:
        return transient_analysis_jit(
            system, t_stop, t_step, t_start, initial_conditions,
            max_iterations, abstol, reltol
        )

    # Original Python-loop implementation
    return _transient_analysis_python(
        system, t_stop, t_step, t_start, initial_conditions,
        max_iterations, abstol, reltol, save_all
    )


def _transient_analysis_python(
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
    """Original Python-loop based transient analysis (for debugging)"""
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
    """Build Jacobian and residual for transient analysis (Python version)

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
        for (term_i, term_j), conductance in stamps.conductances.items():
            idx_i = device.terminals.index(term_i)
            idx_j = device.terminals.index(term_j)
            node_i = device.node_indices[idx_i]
            node_j = device.node_indices[idx_j]

            if node_i != system.ground_node:
                if node_j != system.ground_node:
                    jacobian = jacobian.at[node_i - 1, node_j - 1].add(conductance)
                elif term_i == term_j:
                    jacobian = jacobian.at[node_i - 1, node_i - 1].add(conductance)

        # Handle reactive contributions (capacitances)
        if hasattr(stamps, 'capacitances') and stamps.capacitances:
            node_p = device.node_indices[0]
            node_n = device.node_indices[1]

            cap_value = None
            for key, val in stamps.capacitances.items():
                if key[0] == key[1]:
                    cap_value = abs(float(val))
                    break

            if cap_value is not None and cap_value > 0:
                G_eq = cap_value / dt
                V_cap = V_curr[node_p] - V_curr[node_n]
                V_cap_prev = V_prev[node_p] - V_prev[node_n]
                I_cap = G_eq * (V_cap - V_cap_prev)

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
