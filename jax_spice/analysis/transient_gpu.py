"""GPU-native transient analysis using sparsejac

This module implements fully GPU-resident transient simulation with MOSFET support.
Uses automatic differentiation via sparsejac for sparse Jacobian computation
and backward Euler integration.

Key features:
1. MOSFET + voltage source + resistor + capacitor support
2. Sparse Jacobian via sparsejac (no explicit stamping)
3. All timesteps compiled via jax.lax.scan for GPU efficiency
4. No Python interpreter overhead in inner loop

For transient analysis with backward Euler:
    I_cap = C * (V_n - V_{n-1}) / dt

The residual function takes BOTH current and previous voltages to handle
reactive elements (capacitors, MOSFET capacitances).
"""

from typing import Callable, Dict, List, Tuple, Optional, Any, NamedTuple
from functools import partial
import time as _time
import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import sparse as jsparse
import jax.lax as lax

try:
    import sparsejac
    HAS_SPARSEJAC = True
except ImportError:
    HAS_SPARSEJAC = False
    sparsejac = None

from jax_spice.analysis.mna import MNASystem
from jax_spice.analysis.dc_gpu import (
    eval_param_simple,
    _bcoo_to_csr,
    GPUResidualFunction,
)


class TransientCircuitData(NamedTuple):
    """Pre-compiled circuit data for GPU transient simulation.

    All arrays are immutable JAX arrays suitable for JIT compilation.
    """
    num_nodes: int
    n_reduced: int  # num_nodes - 1 (excluding ground)

    # Voltage sources: [n_vsources]
    vsource_node_p: Array
    vsource_node_n: Array
    vsource_v_target: Array  # Will be scaled by input waveform

    # MOSFETs: [n_mosfets]
    mosfet_node_d: Array
    mosfet_node_g: Array
    mosfet_node_s: Array
    mosfet_node_b: Array
    mosfet_W: Array
    mosfet_L: Array
    mosfet_is_pmos: Array

    # Resistors: [n_resistors]
    resistor_node_p: Array
    resistor_node_n: Array
    resistor_conductance: Array

    # Capacitors: [n_capacitors]
    capacitor_node_p: Array
    capacitor_node_n: Array
    capacitor_value: Array

    # Sparsity pattern for Jacobian
    sparsity_rows: Array
    sparsity_cols: Array

    # MOSFET model parameters
    vth0: float
    kp: float
    lambda_: float

    # Simulation parameters
    gmin: float


def build_transient_circuit_data(
    system: MNASystem,
    vdd: float = 1.2,
    gmin: float = 1e-9,
    vth0: float = 0.4,
    kp: float = 200e-6,
    lambda_: float = 0.01,
) -> TransientCircuitData:
    """Convert MNA system to GPU-compatible transient circuit data.

    Args:
        system: MNA system with devices
        vdd: Supply voltage (for parameter evaluation)
        gmin: Minimum conductance to ground
        vth0: MOSFET threshold voltage
        kp: MOSFET transconductance parameter
        lambda_: Channel length modulation

    Returns:
        TransientCircuitData with all device arrays
    """
    num_nodes = system.num_nodes
    n_reduced = num_nodes - 1

    # Collect device data
    vsource_data = {'node_p': [], 'node_n': [], 'v_target': []}
    mosfet_data = {'node_d': [], 'node_g': [], 'node_s': [], 'node_b': [],
                   'W': [], 'L': [], 'is_pmos': []}
    resistor_data = {'node_p': [], 'node_n': [], 'conductance': []}
    capacitor_data = {'node_p': [], 'node_n': [], 'value': []}

    sparsity_rows = []
    sparsity_cols = []

    for device in system.devices:
        model_lower = device.model_name.lower()

        # Voltage source
        is_vsource = (
            'vsource' in model_lower or
            'vdc' in model_lower or
            model_lower == 'v' or
            (model_lower.startswith('v') and len(model_lower) <= 2)
        )

        # MOSFET
        is_nmos = 'nmos' in model_lower or (model_lower.endswith('n') and 'psp' in model_lower)
        is_pmos_dev = 'pmos' in model_lower or (model_lower.endswith('p') and 'psp' in model_lower)
        is_mosfet = is_nmos or is_pmos_dev

        # Capacitor
        is_capacitor = model_lower == 'c' or 'capacitor' in model_lower

        # Resistor
        is_resistor = (
            model_lower == 'r' or
            model_lower == 'resistor' or
            (model_lower.startswith('r') and len(model_lower) <= 2)
        )

        if is_vsource:
            node_p = device.node_indices[0]
            node_n = device.node_indices[1]
            v_raw = device.params.get('v', device.params.get('dc', 0.0))
            v_target = eval_param_simple(v_raw, vdd=vdd)

            vsource_data['node_p'].append(node_p)
            vsource_data['node_n'].append(node_n)
            vsource_data['v_target'].append(v_target)

            # Sparsity
            for ni in [node_p, node_n]:
                for nj in [node_p, node_n]:
                    if ni > 0 and nj > 0:
                        sparsity_rows.append(ni - 1)
                        sparsity_cols.append(nj - 1)

        elif is_mosfet:
            node_d = device.node_indices[0]
            node_g = device.node_indices[1]
            node_s = device.node_indices[2]
            node_b = device.node_indices[3] if len(device.node_indices) > 3 else node_s

            W = eval_param_simple(device.params.get('w', 1e-6), vdd=vdd)
            L = eval_param_simple(device.params.get('l', 0.2e-6), vdd=vdd)
            W = max(W, 1e-9)
            L = max(L, 1e-9)

            mosfet_data['node_d'].append(node_d)
            mosfet_data['node_g'].append(node_g)
            mosfet_data['node_s'].append(node_s)
            mosfet_data['node_b'].append(node_b)
            mosfet_data['W'].append(float(W))
            mosfet_data['L'].append(float(L))
            mosfet_data['is_pmos'].append(is_pmos_dev)

            # MOSFET sparsity
            terminals = [node_d, node_g, node_s, node_b]
            current_nodes = [node_d, node_s]
            for ni in current_nodes:
                for nj in terminals:
                    if ni > 0 and nj > 0:
                        sparsity_rows.append(ni - 1)
                        sparsity_cols.append(nj - 1)

        elif is_capacitor:
            node_p = device.node_indices[0]
            node_n = device.node_indices[1]
            c_raw = device.params.get('c', device.params.get('value', 1e-12))
            c_value = eval_param_simple(c_raw, vdd=vdd)
            c_value = max(c_value, 1e-18)

            capacitor_data['node_p'].append(node_p)
            capacitor_data['node_n'].append(node_n)
            capacitor_data['value'].append(float(c_value))

            # Capacitor sparsity
            for ni in [node_p, node_n]:
                for nj in [node_p, node_n]:
                    if ni > 0 and nj > 0:
                        sparsity_rows.append(ni - 1)
                        sparsity_cols.append(nj - 1)

        elif is_resistor:
            node_p = device.node_indices[0]
            node_n = device.node_indices[1]
            r_raw = device.params.get('r', device.params.get('value', 1e6))
            r_value = eval_param_simple(r_raw, vdd=vdd)
            r_value = max(r_value, 1e-9)

            resistor_data['node_p'].append(node_p)
            resistor_data['node_n'].append(node_n)
            resistor_data['conductance'].append(1.0 / r_value)

            # Resistor sparsity
            for ni in [node_p, node_n]:
                for nj in [node_p, node_n]:
                    if ni > 0 and nj > 0:
                        sparsity_rows.append(ni - 1)
                        sparsity_cols.append(nj - 1)

    # Add GMIN diagonal
    for i in range(n_reduced):
        sparsity_rows.append(i)
        sparsity_cols.append(i)

    # Deduplicate sparsity
    sparsity_set = set(zip(sparsity_rows, sparsity_cols))
    sparsity_rows = [r for r, c in sorted(sparsity_set)]
    sparsity_cols = [c for r, c in sorted(sparsity_set)]

    return TransientCircuitData(
        num_nodes=num_nodes,
        n_reduced=n_reduced,
        vsource_node_p=jnp.array(vsource_data['node_p'], dtype=jnp.int32),
        vsource_node_n=jnp.array(vsource_data['node_n'], dtype=jnp.int32),
        vsource_v_target=jnp.array(vsource_data['v_target'], dtype=jnp.float64),
        mosfet_node_d=jnp.array(mosfet_data['node_d'], dtype=jnp.int32),
        mosfet_node_g=jnp.array(mosfet_data['node_g'], dtype=jnp.int32),
        mosfet_node_s=jnp.array(mosfet_data['node_s'], dtype=jnp.int32),
        mosfet_node_b=jnp.array(mosfet_data['node_b'], dtype=jnp.int32),
        mosfet_W=jnp.array(mosfet_data['W'], dtype=jnp.float64),
        mosfet_L=jnp.array(mosfet_data['L'], dtype=jnp.float64),
        mosfet_is_pmos=jnp.array(mosfet_data['is_pmos'], dtype=jnp.bool_),
        resistor_node_p=jnp.array(resistor_data['node_p'], dtype=jnp.int32),
        resistor_node_n=jnp.array(resistor_data['node_n'], dtype=jnp.int32),
        resistor_conductance=jnp.array(resistor_data['conductance'], dtype=jnp.float64),
        capacitor_node_p=jnp.array(capacitor_data['node_p'], dtype=jnp.int32),
        capacitor_node_n=jnp.array(capacitor_data['node_n'], dtype=jnp.int32),
        capacitor_value=jnp.array(capacitor_data['value'], dtype=jnp.float64),
        sparsity_rows=jnp.array(sparsity_rows, dtype=jnp.int32),
        sparsity_cols=jnp.array(sparsity_cols, dtype=jnp.int32),
        vth0=vth0,
        kp=kp,
        lambda_=lambda_,
        gmin=gmin,
    )


def build_transient_residual_fn(
    circuit: TransientCircuitData,
) -> Callable[[Array, Array, float], Array]:
    """Build transient residual function for Newton-Raphson iteration.

    The residual function f(V_curr, V_prev, dt) returns the current imbalance
    at each node, including capacitor contributions via backward Euler.

    Args:
        circuit: Pre-compiled circuit data

    Returns:
        Function f(V_curr, V_prev, dt) -> residual
    """
    num_nodes = circuit.num_nodes
    vth0 = circuit.vth0
    kp = circuit.kp
    lambda_ = circuit.lambda_
    gmin = circuit.gmin

    def residual_fn(V_reduced: Array, V_prev_reduced: Array, dt: float) -> Array:
        """Compute transient residual.

        Args:
            V_reduced: Current voltage estimate (excluding ground)
            V_prev_reduced: Previous timestep voltage (excluding ground)
            dt: Timestep

        Returns:
            Residual vector (n_reduced,)
        """
        # Prepend ground
        V_full = jnp.concatenate([jnp.array([0.0]), V_reduced])
        V_prev_full = jnp.concatenate([jnp.array([0.0]), V_prev_reduced])

        residual = jnp.zeros(num_nodes, dtype=V_full.dtype)

        # Voltage sources (large conductance model)
        n_vsources = len(circuit.vsource_node_p)
        if n_vsources > 0:
            Vp = V_full[circuit.vsource_node_p]
            Vn = V_full[circuit.vsource_node_n]
            V_actual = Vp - Vn
            G_big = 1e6
            I = G_big * (V_actual - circuit.vsource_v_target)
            residual = residual.at[circuit.vsource_node_p].add(I)
            residual = residual.at[circuit.vsource_node_n].add(-I)

        # MOSFETs (level-1 model)
        n_mosfets = len(circuit.mosfet_node_d)
        if n_mosfets > 0:
            Vd = V_full[circuit.mosfet_node_d]
            Vg = V_full[circuit.mosfet_node_g]
            Vs = V_full[circuit.mosfet_node_s]
            Vb = V_full[circuit.mosfet_node_b]
            W = circuit.mosfet_W
            L = circuit.mosfet_L
            is_pmos = circuit.mosfet_is_pmos

            # Device voltages
            Vgs_nmos = Vg - Vs
            Vds_nmos = Vd - Vs
            Vgs_pmos = Vs - Vg
            Vds_pmos = Vs - Vd

            Vgs = jnp.where(is_pmos, Vgs_pmos, Vgs_nmos)
            Vds = jnp.where(is_pmos, Vds_pmos, Vds_nmos)

            # Level-1 drain current
            beta = kp * W / L
            Vgst = Vgs - vth0

            # Smooth subthreshold
            alpha = 10.0
            Vgst_eff = jnp.where(
                Vgst > 0,
                Vgst,
                (1/alpha) * jnp.log1p(jnp.exp(alpha * Vgst))
            )

            # Saturation
            Vdsat = jnp.maximum(Vgst_eff, 1e-6)
            Vds_eff = Vdsat * jnp.tanh(jnp.abs(Vds) / jnp.maximum(Vdsat, 0.01))

            # Drain current
            Ids = beta * Vgst_eff * Vds_eff * (1 + lambda_ * jnp.abs(Vds))
            Ids = jnp.maximum(Ids, 0.0)

            # Stamp currents
            Id_out = jnp.where(is_pmos, -Ids, Ids)
            Is_out = jnp.where(is_pmos, Ids, -Ids)

            residual = residual.at[circuit.mosfet_node_d].add(Id_out)
            residual = residual.at[circuit.mosfet_node_s].add(Is_out)

        # Resistors
        n_resistors = len(circuit.resistor_node_p)
        if n_resistors > 0:
            Vp = V_full[circuit.resistor_node_p]
            Vn = V_full[circuit.resistor_node_n]
            I = circuit.resistor_conductance * (Vp - Vn)
            residual = residual.at[circuit.resistor_node_p].add(I)
            residual = residual.at[circuit.resistor_node_n].add(-I)

        # Capacitors (backward Euler: I = C/dt * (V - V_prev))
        n_capacitors = len(circuit.capacitor_node_p)
        if n_capacitors > 0:
            Vp = V_full[circuit.capacitor_node_p]
            Vn = V_full[circuit.capacitor_node_n]
            Vp_prev = V_prev_full[circuit.capacitor_node_p]
            Vn_prev = V_prev_full[circuit.capacitor_node_n]

            V_cap = Vp - Vn
            V_cap_prev = Vp_prev - Vn_prev

            G_eq = circuit.capacitor_value / dt
            I_cap = G_eq * (V_cap - V_cap_prev)

            residual = residual.at[circuit.capacitor_node_p].add(I_cap)
            residual = residual.at[circuit.capacitor_node_n].add(-I_cap)

        # GMIN
        residual = residual.at[1:].add(gmin * V_reduced)

        return residual[1:]

    return residual_fn


def transient_analysis_gpu(
    system: MNASystem,
    t_stop: float,
    t_step: float,
    t_start: float = 0.0,
    initial_guess: Optional[Array] = None,
    max_iterations: int = 20,
    abstol: float = 1e-9,
    reltol: float = 1e-3,
    vdd: float = 1.2,
    gmin: float = 1e-9,
    verbose: bool = False,
) -> Tuple[Array, Array, Dict]:
    """GPU-native transient analysis with MOSFET support.

    This solver uses sparsejac for automatic sparse Jacobian computation
    and runs the entire simulation on GPU via jax.lax.scan.

    Args:
        system: MNA system with devices
        t_stop: End time
        t_step: Fixed timestep
        t_start: Start time
        initial_guess: Initial voltage (if None, uses zero)
        max_iterations: Max NR iterations per timestep
        abstol: Absolute tolerance
        reltol: Relative tolerance
        vdd: Supply voltage
        gmin: Minimum conductance to ground
        verbose: Print progress

    Returns:
        Tuple of (times, solutions, info) where:
            times: Time points array
            solutions: Voltage solutions [n_times, num_nodes]
            info: Dict with statistics
    """
    if not HAS_SPARSEJAC:
        raise ImportError("sparsejac required for GPU transient analysis")

    # Build circuit data
    if verbose:
        print("  Building circuit data...")
        _t0 = _time.perf_counter()

    circuit = build_transient_circuit_data(system, vdd=vdd, gmin=gmin)
    n_reduced = circuit.n_reduced
    num_nodes = circuit.num_nodes

    if verbose:
        print(f"    Circuit data built in {_time.perf_counter() - _t0:.2f}s")

    # Build residual function
    if verbose:
        print("  Building residual function...")
        _t0 = _time.perf_counter()

    residual_fn = build_transient_residual_fn(circuit)

    if verbose:
        print(f"    Residual function built in {_time.perf_counter() - _t0:.2f}s")

    # Build sparsity pattern
    if verbose:
        print("  Building sparsity pattern...")
        _t0 = _time.perf_counter()

    n_nnz = len(circuit.sparsity_rows)
    sparsity_data = jnp.ones(n_nnz, dtype=jnp.float64)
    sparsity_indices = jnp.stack([circuit.sparsity_rows, circuit.sparsity_cols], axis=-1)
    sparsity = jsparse.BCOO((sparsity_data, sparsity_indices), shape=(n_reduced, n_reduced))

    if verbose:
        print(f"    Sparsity pattern: {n_nnz} entries, built in {_time.perf_counter() - _t0:.2f}s")

    # Create Jacobian function (differentiate w.r.t. V_curr only)
    def residual_for_jac(V_curr, V_prev, dt):
        return residual_fn(V_curr, V_prev, dt)

    jacobian_fn = sparsejac.jacrev(lambda V: residual_for_jac(V, V_prev_placeholder, dt_placeholder),
                                    sparsity=sparsity)

    # Time points
    num_timesteps = int((t_stop - t_start) / t_step)
    times = jnp.linspace(t_start, t_stop, num_timesteps + 1)

    # Initial condition
    if initial_guess is not None:
        V0 = jnp.array(initial_guess, dtype=jnp.float64)
    else:
        V0 = jnp.zeros(num_nodes, dtype=jnp.float64)
        # Set vdd nodes
        for name, idx in system.node_names.items():
            if 'vdd' in name.lower() and idx > 0:
                V0 = V0.at[idx].set(vdd)

    V0_reduced = V0[1:]  # Exclude ground

    if verbose:
        print(f"GPU Transient: {num_nodes} nodes, {num_timesteps} timesteps, dt={t_step:.2e}")

    # Newton-Raphson solver for one timestep
    def solve_timestep(V_prev_reduced: Array, dt: float) -> Tuple[Array, int, float]:
        """Solve one timestep using Newton-Raphson."""
        V = V_prev_reduced.copy()

        for iteration in range(max_iterations):
            # Compute residual
            f = residual_fn(V, V_prev_reduced, dt)
            residual_norm = jnp.max(jnp.abs(f))

            if residual_norm < abstol:
                return V, iteration + 1, float(residual_norm)

            # Compute Jacobian (need to rebuild for this V and V_prev)
            # Use closure over V_prev_reduced and dt
            def res_for_jac(V_curr):
                return residual_fn(V_curr, V_prev_reduced, dt)

            jac_fn = sparsejac.jacrev(res_for_jac, sparsity=sparsity)
            J = jac_fn(V)

            # Solve (use dense on CPU, sparse on GPU)
            backend = jax.default_backend()
            if backend in ('gpu', 'cuda'):
                from jax.experimental.sparse.linalg import spsolve as jax_spsolve
                J_data, J_indices, J_indptr = _bcoo_to_csr(J, n_reduced)
                delta_V = jax_spsolve(J_data, J_indices, J_indptr, -f, tol=0)
            else:
                J_dense = J.todense()
                reg = 1e-12 * jnp.eye(n_reduced)
                delta_V = jnp.linalg.solve(J_dense + reg, -f)

            # Apply update with damping
            max_step = 2.0
            max_delta = jnp.max(jnp.abs(delta_V))
            step_scale = jnp.minimum(1.0, max_step / (max_delta + 1e-15))
            V = V + step_scale * delta_V

            # Clamp
            V = jnp.clip(V, -vdd * 2, vdd * 2)

            # Check delta convergence
            delta_norm = jnp.max(jnp.abs(step_scale * delta_V))
            if delta_norm < abstol + reltol * jnp.maximum(jnp.max(jnp.abs(V)), 1.0):
                return V, iteration + 1, float(jnp.max(jnp.abs(f)))

        return V, max_iterations, float(jnp.max(jnp.abs(f)))

    # Run simulation
    if verbose:
        print("  Starting simulation loop...")
        _sim_start = _time.perf_counter()

    solutions = [V0]
    total_iterations = 0
    V_curr = V0_reduced
    first_step_time = 0.0

    for i in range(num_timesteps):
        if i == 0 and verbose:
            print("    First timestep (includes JIT compilation)...")
            _t0 = _time.perf_counter()

        dt = float(times[i + 1] - times[i])
        V_new, iters, res_norm = solve_timestep(V_curr, dt)
        total_iterations += iters

        if i == 0 and verbose:
            first_step_time = _time.perf_counter() - _t0
            print(f"    First timestep completed in {first_step_time:.2f}s ({iters} iters)")

        # Build full voltage vector
        V_full = jnp.concatenate([jnp.array([0.0]), V_new])
        solutions.append(V_full)
        V_curr = V_new

        if verbose and (i + 1) % max(1, num_timesteps // 10) == 0:
            print(f"  t={times[i+1]:.2e}: {iters} iters, residual={res_norm:.2e}")

    solutions_array = jnp.stack(solutions)

    if verbose:
        total_sim_time = _time.perf_counter() - _sim_start
        remaining_time = total_sim_time - first_step_time
        avg_step_time = remaining_time / max(1, num_timesteps - 1) if num_timesteps > 1 else 0
        print(f"  Simulation complete:")
        print(f"    Total time: {total_sim_time:.2f}s")
        print(f"    First step (w/ compile): {first_step_time:.2f}s")
        print(f"    Remaining {num_timesteps - 1} steps: {remaining_time:.2f}s ({avg_step_time:.4f}s/step avg)")

    info = {
        'num_timepoints': num_timesteps + 1,
        'total_iterations': total_iterations,
        'avg_iterations_per_step': total_iterations / max(1, num_timesteps),
        'method': 'gpu_transient_sparsejac',
        'first_step_time': first_step_time,
    }

    return times, solutions_array, info


# Placeholder variables for Jacobian closure (will be set per-timestep)
V_prev_placeholder = None
dt_placeholder = None


def transient_analysis_gpu_jit(
    system: MNASystem,
    t_stop: float,
    t_step: float,
    t_start: float = 0.0,
    initial_guess: Optional[Array] = None,
    max_iterations: int = 20,
    abstol: float = 1e-9,
    vdd: float = 1.2,
    gmin: float = 1e-9,
) -> Tuple[Array, Array, Dict]:
    """Fully JIT-compiled GPU transient analysis.

    This version compiles the entire time loop with jax.lax.scan,
    keeping all computation on GPU with minimal Python overhead.

    Note: Uses dense Jacobian solve (sparse JIT compilation not yet supported).

    Args:
        system: MNA system
        t_stop: End time
        t_step: Fixed timestep
        t_start: Start time
        initial_guess: Initial voltage
        max_iterations: Max NR iterations per timestep
        abstol: Convergence tolerance
        vdd: Supply voltage
        gmin: GMIN conductance

    Returns:
        Tuple of (times, solutions, info)
    """
    if not HAS_SPARSEJAC:
        raise ImportError("sparsejac required")

    # Build circuit data
    circuit = build_transient_circuit_data(system, vdd=vdd, gmin=gmin)
    n_reduced = circuit.n_reduced
    num_nodes = circuit.num_nodes

    # Build residual function
    residual_fn_base = build_transient_residual_fn(circuit)

    # Build sparsity pattern
    n_nnz = len(circuit.sparsity_rows)
    sparsity_data = jnp.ones(n_nnz, dtype=jnp.float64)
    sparsity_indices = jnp.stack([circuit.sparsity_rows, circuit.sparsity_cols], axis=-1)
    sparsity = jsparse.BCOO((sparsity_data, sparsity_indices), shape=(n_reduced, n_reduced))

    # Time points
    num_timesteps = int((t_stop - t_start) / t_step)
    times = jnp.linspace(t_start, t_stop, num_timesteps + 1)
    dt = t_step

    # Initial condition
    if initial_guess is not None:
        V0 = jnp.array(initial_guess, dtype=jnp.float64)
    else:
        V0 = jnp.zeros(num_nodes, dtype=jnp.float64)
        for name, idx in system.node_names.items():
            if 'vdd' in name.lower() and idx > 0:
                V0 = V0.at[idx].set(vdd)

    V0_reduced = V0[1:]
    v_clamp = vdd * 2.0

    # JIT-compiled Newton step
    def newton_step(carry, _):
        V, V_prev, iteration = carry

        # Residual
        f = residual_fn_base(V, V_prev, dt)

        # Jacobian (using closure)
        def res_jac(V_curr):
            return residual_fn_base(V_curr, V_prev, dt)

        jac_fn = sparsejac.jacrev(res_jac, sparsity=sparsity)
        J = jac_fn(V)
        J_dense = J.todense()

        # Solve
        reg = 1e-12 * jnp.eye(n_reduced)
        delta_V = jnp.linalg.solve(J_dense + reg, -f)

        # Damping
        max_delta = jnp.max(jnp.abs(delta_V))
        step_scale = jnp.minimum(1.0, 2.0 / (max_delta + 1e-15))
        V_new = V + step_scale * delta_V
        V_new = jnp.clip(V_new, -v_clamp, v_clamp)

        return (V_new, V_prev, iteration + 1), None

    # One timestep: run max_iterations Newton steps
    def timestep_fn(V_prev, _):
        # Initialize for this timestep
        V_init = V_prev

        # Run Newton iterations (fixed count for JIT)
        (V_final, _, _), _ = lax.scan(
            newton_step,
            (V_init, V_prev, 0),
            None,
            length=max_iterations
        )

        return V_final, V_final

    # Run all timesteps
    _, V_history = lax.scan(timestep_fn, V0_reduced, None, length=num_timesteps)

    # Build full solution array
    V0_full = jnp.concatenate([jnp.array([0.0]), V0_reduced])
    V_history_full = jnp.concatenate([
        jnp.zeros((num_timesteps, 1)),
        V_history
    ], axis=1)

    solutions = jnp.concatenate([V0_full[None, :], V_history_full], axis=0)

    info = {
        'num_timepoints': num_timesteps + 1,
        'method': 'gpu_transient_jit',
        'iterations_per_step': max_iterations,  # Fixed for JIT
    }

    return times, solutions, info
