"""DC operating point analysis for JAX-SPICE

Uses Newton-Raphson iteration to find the DC operating point
where all capacitor currents are zero and the circuit is in equilibrium.
"""

from typing import Dict, Optional, Tuple, Callable
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve
import numpy as np

from jax_spice.analysis.mna import MNASystem
from jax_spice.analysis.context import AnalysisContext
from jax_spice.analysis.sparse import sparse_solve_csr


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


def dc_operating_point_sparse(
    system: MNASystem,
    initial_guess: Optional[Array] = None,
    max_iterations: int = 100,
    abstol: float = 1e-9,
    reltol: float = 1e-3,
    damping: float = 1.0,
    vdd: float = 1.2,
    init_supplies: bool = True,
    verbose: bool = False,
    source_stepping: bool = False,
    source_steps: int = 5,
) -> Tuple[Array, Dict]:
    """Find DC operating point using sparse Newton-Raphson iteration

    This version uses sparse matrix assembly and sparse linear solvers,
    which is much more memory-efficient for large circuits (>1000 nodes).

    Memory comparison for 5000-node circuit:
    - Dense: ~200MB per Jacobian matrix
    - Sparse: ~5MB (assuming 0.1% fill)

    Args:
        system: MNA system with devices
        initial_guess: Initial voltage estimate (shape: [num_nodes])
                      If None, starts from zero with supply nodes initialized
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance for convergence
        reltol: Relative tolerance for convergence
        damping: Damping factor (0 < damping <= 1), or 'auto' for adaptive
        vdd: Supply voltage for initialization and clamping
        init_supplies: If True, initialize nodes with 'vdd' in name to vdd
        verbose: Print iteration details

    Returns:
        Tuple of (solution, info) where:
            solution: Node voltages (shape: [num_nodes])
            info: Dict with convergence information
    """
    n = system.num_nodes

    # Initialize solution
    if initial_guess is not None:
        V = np.array(initial_guess, dtype=np.float64)
    else:
        V = np.zeros(n, dtype=np.float64)

        # Initialize supply nodes if requested
        if init_supplies:
            for name, idx in system.node_names.items():
                name_lower = name.lower()
                if 'vdd' in name_lower:
                    V[idx] = vdd

    # Find supply node indices (these have fixed voltage via voltage sources)
    # Their residual represents supply current, not an error
    supply_node_indices = set()
    for name, idx in system.node_names.items():
        name_lower = name.lower()
        if name_lower in ('vdd', 'vss', 'gnd', '0') or 'vdd' in name_lower:
            if idx > 0:  # Skip ground (index 0)
                supply_node_indices.add(idx - 1)  # Convert to reduced index

    # Create DC context
    context = AnalysisContext(
        time=0.0,
        dt=1e-9,
        analysis_type='dc',
        c0=0.0,
        c1=0.0,
        rhs_correction=0.0,
    )

    converged = False
    iterations = 0
    residual_norm = 1e20
    delta_norm = 0.0
    residual_history = []

    for iteration in range(max_iterations):
        context.iteration = iteration

        # Build sparse Jacobian and residual
        (data, indices, indptr, shape), f = system.build_sparse_jacobian_and_residual(
            jnp.array(V), context
        )

        # Check residual norm for convergence, excluding supply nodes
        # Supply nodes have residual = supply current, not an error
        f_check = f.copy()
        for idx in supply_node_indices:
            f_check[idx] = 0.0
        residual_norm = float(np.max(np.abs(f_check)))
        residual_norm_full = float(np.max(np.abs(f)))
        residual_history.append(residual_norm)

        if verbose and iteration < 20:
            print(f"    Iter {iteration}: residual={residual_norm:.2e} (full={residual_norm_full:.2e}), "
                  f"V_max={np.max(V[1:]):.4f}, V_min={np.min(V[1:]):.4f}")

        if residual_norm < abstol:
            converged = True
            iterations = iteration + 1
            break

        # Solve for Newton update using sparse solver
        # J * delta_V = -f
        delta_V = sparse_solve_csr(
            jnp.array(data),
            jnp.array(indices),
            jnp.array(indptr),
            jnp.array(-f),
            shape
        )
        delta_V = np.array(delta_V)

        # Apply damping with voltage step limiting
        max_step = 2.0  # Maximum voltage change per iteration
        max_delta = np.max(np.abs(delta_V))
        step_scale = min(damping, max_step / (max_delta + 1e-15))

        # Update solution (skip ground node at index 0)
        V[1:] += step_scale * delta_V

        # Clamp voltages to reasonable range for CMOS circuits
        # Use +/- 2*Vdd to allow for some overshoot while preventing runaway
        v_clamp = vdd * 2.0
        V = np.clip(V, -v_clamp, v_clamp)

        iterations = iteration + 1

        # Check delta for convergence
        delta_norm = float(np.max(np.abs(step_scale * delta_V)))
        v_norm = float(np.max(np.abs(V[1:])))

        if delta_norm < abstol + reltol * max(v_norm, 1.0):
            converged = True
            break

    # Convert back to JAX array
    V_jax = jnp.array(V)

    info = {
        'converged': converged,
        'iterations': iterations,
        'residual_norm': residual_norm,
        'delta_norm': delta_norm,
        'residual_history': residual_history,
    }

    return V_jax, info


def dc_operating_point_source_stepping(
    system: MNASystem,
    initial_guess: Optional[Array] = None,
    vdd_target: float = 1.2,
    vdd_steps: int = 12,
    max_iterations_per_step: int = 50,
    abstol: float = 1e-9,
    reltol: float = 1e-3,
    damping: float = 1.0,
    init_supplies: bool = True,
    verbose: bool = False,
) -> Tuple[Array, Dict]:
    """Find DC operating point using source stepping for difficult circuits

    Source stepping is a homotopy method that gradually ramps the supply voltage
    from 0 to the target value. At Vdd=0, all transistors are OFF and the circuit
    is trivially solved. As Vdd increases, the solution evolves continuously.

    This is particularly effective for:
    - Large digital circuits with many cascaded stages
    - Circuits where all inputs are held at fixed values (e.g., logic low)
    - Circuits that fail to converge because PMOS are ON but NMOS are OFF

    The method works by:
    1. Starting with Vdd=0 (or small value), where all devices are off
    2. Solving at each Vdd step using previous solution as initial guess
    3. Gradually increasing Vdd until target is reached

    Args:
        system: MNA system with devices
        initial_guess: Initial voltage estimate (shape: [num_nodes])
        vdd_target: Target supply voltage (default 1.2V)
        vdd_steps: Number of voltage steps (default 12, so 0.1V increments)
        max_iterations_per_step: Max NR iterations per source step
        abstol: Absolute tolerance for convergence
        reltol: Relative tolerance for convergence
        damping: Damping factor (0 < damping <= 1)
        init_supplies: If True, initialize vdd nodes to current step voltage
        verbose: Print progress information

    Returns:
        Tuple of (solution, info) where:
            solution: Node voltages (shape: [num_nodes])
            info: Dict with convergence information including source_steps
    """
    n = system.num_nodes

    # Initialize solution to zeros (all transistors off at Vdd=0)
    if initial_guess is not None:
        V = np.array(initial_guess, dtype=np.float64)
    else:
        V = np.zeros(n, dtype=np.float64)

    # Find vdd node indices - we'll scale these during stepping
    vdd_node_indices = []
    for name, idx in system.node_names.items():
        name_lower = name.lower()
        if 'vdd' in name_lower and name_lower not in ('vss', 'gnd', '0'):
            vdd_node_indices.append(idx)

    total_iterations = 0
    source_steps = 0
    all_residual_history = []

    # Generate voltage steps: start from small non-zero value to target
    # Using linear steps from vdd_target/vdd_steps to vdd_target
    vdd_values = np.linspace(vdd_target / vdd_steps, vdd_target, vdd_steps)

    if verbose:
        print(f"Source stepping: 0 -> {vdd_target:.2f}V in {vdd_steps} steps", flush=True)

    converged_at_target = False
    last_info = {'converged': False, 'iterations': 0, 'residual_norm': 1e20}

    step_idx = 0
    while step_idx < len(vdd_values):
        vdd_step = vdd_values[step_idx]
        source_steps += 1
        is_final_step = (step_idx == len(vdd_values) - 1)

        # Set vdd nodes to current step voltage
        if init_supplies:
            for idx in vdd_node_indices:
                V[idx] = vdd_step

        if verbose:
            print(f"  Source step {source_steps}: Vdd={vdd_step:.3f}V", flush=True)

        # Use relaxed tolerance for intermediate steps, tight tolerance only for final
        step_abstol = abstol if is_final_step else max(abstol, 1e-4)

        # First try with moderate GMIN
        # Use 1e-6 for better matrix conditioning on large digital circuits
        # (1e-9 produces near-singular Jacobians with high-impedance MOSFET gates)
        V_jax, info = _dc_solve_with_source_scaling(
            system,
            initial_guess=V,
            vdd_scale=vdd_step / vdd_target,
            vdd_target=vdd_target,
            gmin=1e-6,  # GMIN for good matrix conditioning
            max_iterations=max_iterations_per_step,
            abstol=step_abstol,
            reltol=reltol,
            damping=damping,
            verbose=False,
        )

        V = np.array(V_jax)
        total_iterations += info['iterations']
        all_residual_history.extend(info['residual_history'])
        last_info = info

        if verbose:
            print(f"    -> iter={info['iterations']}, residual={info['residual_norm']:.2e}, "
                  f"converged={info['converged']}", flush=True)

        # For steps that don't converge, try fallback with higher GMIN
        if not info['converged']:
            if not is_final_step and info['residual_norm'] < 1e-3:
                # Accept partial convergence for intermediate steps
                if verbose:
                    print(f"    Accepting partial convergence for intermediate step", flush=True)
            else:
                # Try much higher GMIN for this difficult step
                if verbose:
                    print(f"    Trying with higher GMIN (1e-3)...", flush=True)

                V_jax_h, info_h = _dc_solve_with_source_scaling(
                    system,
                    initial_guess=V,
                    vdd_scale=vdd_step / vdd_target,
                    vdd_target=vdd_target,
                    gmin=1e-3,  # Much higher GMIN for difficult regions
                    max_iterations=max_iterations_per_step * 2,
                    abstol=1e-3 if not is_final_step else abstol,  # Tighter tol for final
                    reltol=reltol,
                    damping=damping,
                    verbose=False,
                )

                if info_h['residual_norm'] < info['residual_norm']:
                    V = np.array(V_jax_h)
                    total_iterations += info_h['iterations']
                    all_residual_history.extend(info_h['residual_history'])
                    last_info = info_h

                if verbose:
                    print(f"      -> iter={info_h['iterations']}, residual={info_h['residual_norm']:.2e}", flush=True)

        # Only check for true convergence at final step
        if is_final_step and last_info['converged']:
            converged_at_target = True

        step_idx += 1

    result_info = {
        'converged': converged_at_target,
        'iterations': total_iterations,
        'source_steps': source_steps,
        'final_vdd': vdd_step,
        'residual_norm': last_info['residual_norm'],
        'delta_norm': last_info.get('delta_norm', 0.0),
        'residual_history': all_residual_history,
    }

    if verbose:
        print(f"  Source stepping complete: steps={source_steps}, "
              f"total_iter={total_iterations}, converged={converged_at_target}", flush=True)

    return jnp.array(V), result_info


def _dc_solve_with_source_scaling(
    system: MNASystem,
    initial_guess: Optional[Array] = None,
    vdd_scale: float = 1.0,
    vdd_target: float = 1.2,
    gmin: float = 1e-12,
    max_iterations: int = 100,
    abstol: float = 1e-9,
    reltol: float = 1e-3,
    damping: float = 1.0,
    verbose: bool = False,
) -> Tuple[Array, Dict]:
    """Internal DC solver with voltage source scaling for source stepping

    This solver scales all voltage source targets by vdd_scale factor,
    allowing gradual ramping of supply voltage during source stepping.

    Args:
        system: MNA system with devices
        initial_guess: Initial voltage estimate
        vdd_scale: Scale factor for voltage sources (0 to 1)
        vdd_target: Target Vdd value (used for clamping)
        gmin: GMIN value for diagonal stabilization
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance
        reltol: Relative tolerance
        damping: Damping factor
        verbose: Print iteration details

    Returns:
        Tuple of (solution, info)
    """
    n = system.num_nodes

    # Initialize solution
    if initial_guess is not None:
        V = np.array(initial_guess, dtype=np.float64)
    else:
        V = np.zeros(n, dtype=np.float64)

    # Find supply node indices for residual exclusion
    supply_node_indices = set()
    for name, idx in system.node_names.items():
        name_lower = name.lower()
        if name_lower in ('vdd', 'vss', 'gnd', '0') or 'vdd' in name_lower:
            if idx > 0:
                supply_node_indices.add(idx - 1)

    # Create context with source scaling
    context = AnalysisContext(
        time=0.0,
        dt=1e-9,
        analysis_type='dc',
        c0=0.0,
        c1=0.0,
        rhs_correction=0.0,
        gmin=gmin,
    )
    # Store vdd_scale in context for voltage sources to use
    context.vdd_scale = vdd_scale

    converged = False
    iterations = 0
    residual_norm = 1e20
    delta_norm = 0.0
    residual_history = []

    # Current scaled Vdd for clamping
    vdd_current = vdd_target * vdd_scale

    for iteration in range(max_iterations):
        context.iteration = iteration

        # Build sparse Jacobian and residual
        (data, indices, indptr, shape), f = system.build_sparse_jacobian_and_residual(
            jnp.array(V), context
        )

        # Check residual norm, excluding supply nodes
        f_check = f.copy()
        for idx in supply_node_indices:
            f_check[idx] = 0.0
        residual_norm = float(np.max(np.abs(f_check)))
        residual_history.append(residual_norm)

        if verbose and iteration < 20:
            print(f"      Iter {iteration}: residual={residual_norm:.2e}")

        if residual_norm < abstol:
            converged = True
            iterations = iteration + 1
            break

        # Solve for Newton update
        delta_V = sparse_solve_csr(
            jnp.array(data),
            jnp.array(indices),
            jnp.array(indptr),
            jnp.array(-f),
            shape
        )
        delta_V = np.array(delta_V)

        # Apply damping with voltage step limiting
        max_step = 2.0
        max_delta = np.max(np.abs(delta_V))
        step_scale = min(damping, max_step / (max_delta + 1e-15))

        # Update solution
        V[1:] += step_scale * delta_V

        # Clamp to reasonable range based on current Vdd
        v_clamp = max(vdd_current * 2.0, 0.5)  # At least 0.5V headroom
        V = np.clip(V, -v_clamp, v_clamp)

        iterations = iteration + 1

        # Check delta for convergence
        delta_norm = float(np.max(np.abs(step_scale * delta_V)))
        v_norm = float(np.max(np.abs(V[1:])))

        if delta_norm < abstol + reltol * max(v_norm, 1.0):
            converged = True
            break

    V_jax = jnp.array(V)

    info = {
        'converged': converged,
        'iterations': iterations,
        'residual_norm': residual_norm,
        'delta_norm': delta_norm,
        'residual_history': residual_history,
    }

    return V_jax, info


def dc_operating_point_gmin_stepping(
    system: MNASystem,
    initial_guess: Optional[Array] = None,
    start_gmin: float = 1e-2,
    target_gmin: float = 1e-12,
    gmin_factor: float = 10.0,
    max_gmin_steps: int = 20,
    max_iterations_per_step: int = 50,
    abstol: float = 1e-9,
    reltol: float = 1e-3,
    damping: float = 1.0,
    vdd: float = 1.2,
    init_supplies: bool = True,
    verbose: bool = False,
) -> Tuple[Array, Dict]:
    """Find DC operating point using GMIN stepping for difficult circuits

    GMIN stepping is a homotopy method that starts with a large minimum
    conductance (GMIN) from each node to ground, making the matrix well-
    conditioned. The GMIN is then gradually reduced to the target value,
    using the previous solution as a warm start.

    This is particularly effective for:
    - Large digital circuits with many cascaded stages
    - Circuits with floating nodes
    - Circuits that fail to converge with standard Newton-Raphson

    Args:
        system: MNA system with devices
        initial_guess: Initial voltage estimate (shape: [num_nodes])
        start_gmin: Initial large GMIN value (default 1e-2)
        target_gmin: Final small GMIN value (default 1e-12)
        gmin_factor: Factor to reduce GMIN by each step (default 10.0)
        max_gmin_steps: Maximum number of GMIN stepping iterations
        max_iterations_per_step: Max NR iterations per GMIN step
        abstol: Absolute tolerance for convergence
        reltol: Relative tolerance for convergence
        damping: Damping factor (0 < damping <= 1)
        vdd: Supply voltage for initialization and clamping
        init_supplies: If True, initialize nodes with 'vdd' in name to vdd
        verbose: Print progress information

    Returns:
        Tuple of (solution, info) where:
            solution: Node voltages (shape: [num_nodes])
            info: Dict with convergence information including gmin_steps
    """
    n = system.num_nodes

    # Initialize solution
    if initial_guess is not None:
        V = np.array(initial_guess, dtype=np.float64)
    else:
        V = np.zeros(n, dtype=np.float64)

        # Initialize supply nodes if requested
        if init_supplies:
            for name, idx in system.node_names.items():
                name_lower = name.lower()
                if 'vdd' in name_lower:
                    V[idx] = vdd

    gmin = start_gmin
    total_iterations = 0
    gmin_steps = 0
    all_residual_history = []

    if verbose:
        print(f"GMIN stepping: {start_gmin:.0e} -> {target_gmin:.0e} (factor {gmin_factor})", flush=True)

    while gmin >= target_gmin:
        gmin_steps += 1

        if verbose:
            print(f"  GMIN step {gmin_steps}: gmin={gmin:.0e}", flush=True)

        # Run sparse Newton-Raphson with current GMIN
        # We need to pass gmin via the context, which is created inside dc_operating_point_sparse
        # But dc_operating_point_sparse doesn't accept gmin, so we need to call a lower-level
        # function or modify dc_operating_point_sparse. Let's create an internal version.
        V_jax, info = _dc_solve_with_gmin(
            system,
            initial_guess=V,
            gmin=gmin,
            max_iterations=max_iterations_per_step,
            abstol=abstol,
            reltol=reltol,
            damping=damping,
            vdd=vdd,
            verbose=False,
        )

        V = np.array(V_jax)
        total_iterations += info['iterations']
        all_residual_history.extend(info['residual_history'])

        if verbose:
            print(f"    -> iter={info['iterations']}, residual={info['residual_norm']:.2e}, "
                  f"converged={info['converged']}", flush=True)

        if not info['converged']:
            # Back off: increase GMIN slightly and retry
            gmin = min(gmin * 2.0, start_gmin)
            if verbose:
                print(f"    Not converged, backing off to gmin={gmin:.0e}", flush=True)

            if gmin_steps > max_gmin_steps:
                if verbose:
                    print(f"  GMIN stepping failed after {max_gmin_steps} steps", flush=True)
                break
            continue

        # Converged at this GMIN level
        if gmin <= target_gmin:
            # Reached target GMIN
            break

        # Reduce GMIN for next step
        gmin = max(gmin / gmin_factor, target_gmin)

    final_converged = info['converged'] and gmin <= target_gmin

    result_info = {
        'converged': final_converged,
        'iterations': total_iterations,
        'gmin_steps': gmin_steps,
        'final_gmin': gmin,
        'residual_norm': info['residual_norm'],
        'delta_norm': info.get('delta_norm', 0.0),
        'residual_history': all_residual_history,
    }

    if verbose:
        print(f"  GMIN stepping complete: gmin_steps={gmin_steps}, "
              f"total_iter={total_iterations}, converged={final_converged}", flush=True)

    return jnp.array(V), result_info


def _dc_solve_with_gmin(
    system: MNASystem,
    initial_guess: Optional[Array] = None,
    gmin: float = 1e-12,
    max_iterations: int = 100,
    abstol: float = 1e-9,
    reltol: float = 1e-3,
    damping: float = 1.0,
    vdd: float = 1.2,
    verbose: bool = False,
) -> Tuple[Array, Dict]:
    """Internal DC solver with explicit GMIN parameter

    This is a variant of dc_operating_point_sparse that accepts an explicit
    GMIN value for use with GMIN stepping.
    """
    n = system.num_nodes

    # Initialize solution
    if initial_guess is not None:
        V = np.array(initial_guess, dtype=np.float64)
    else:
        V = np.zeros(n, dtype=np.float64)

    # Find supply node indices (these have fixed voltage via voltage sources)
    supply_node_indices = set()
    for name, idx in system.node_names.items():
        name_lower = name.lower()
        if name_lower in ('vdd', 'vss', 'gnd', '0') or 'vdd' in name_lower:
            if idx > 0:
                supply_node_indices.add(idx - 1)

    # Create DC context with specified GMIN
    context = AnalysisContext(
        time=0.0,
        dt=1e-9,
        analysis_type='dc',
        c0=0.0,
        c1=0.0,
        rhs_correction=0.0,
        gmin=gmin,  # Pass explicit GMIN
    )

    converged = False
    iterations = 0
    residual_norm = 1e20
    delta_norm = 0.0
    residual_history = []

    for iteration in range(max_iterations):
        context.iteration = iteration

        # Build sparse Jacobian and residual
        (data, indices, indptr, shape), f = system.build_sparse_jacobian_and_residual(
            jnp.array(V), context
        )

        # Check residual norm for convergence, excluding supply nodes
        f_check = f.copy()
        for idx in supply_node_indices:
            f_check[idx] = 0.0
        residual_norm = float(np.max(np.abs(f_check)))
        residual_history.append(residual_norm)

        if verbose and iteration < 20:
            print(f"      Iter {iteration}: residual={residual_norm:.2e}")

        if residual_norm < abstol:
            converged = True
            iterations = iteration + 1
            break

        # Solve for Newton update using sparse solver
        delta_V = sparse_solve_csr(
            jnp.array(data),
            jnp.array(indices),
            jnp.array(indptr),
            jnp.array(-f),
            shape
        )
        delta_V = np.array(delta_V)

        # Apply damping with voltage step limiting
        max_step = 2.0
        max_delta = np.max(np.abs(delta_V))
        step_scale = min(damping, max_step / (max_delta + 1e-15))

        # Update solution (skip ground node at index 0)
        V[1:] += step_scale * delta_V

        # Clamp voltages to reasonable range
        v_clamp = vdd * 2.0
        V = np.clip(V, -v_clamp, v_clamp)

        iterations = iteration + 1

        # Check delta for convergence
        delta_norm = float(np.max(np.abs(step_scale * delta_V)))
        v_norm = float(np.max(np.abs(V[1:])))

        if delta_norm < abstol + reltol * max(v_norm, 1.0):
            converged = True
            break

    V_jax = jnp.array(V)

    info = {
        'converged': converged,
        'iterations': iterations,
        'residual_norm': residual_norm,
        'delta_norm': delta_norm,
        'residual_history': residual_history,
    }

    return V_jax, info


# =============================================================================
# GPU-Native DC Solver
# =============================================================================


def dc_operating_point_gpu(
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
    """Find DC operating point using GPU-native JIT-compiled Newton-Raphson.

    This version uses JAX's lax.while_loop for a fully JIT-compiled Newton
    iteration that keeps all data GPU-resident, avoiding CPU-GPU transfers.

    For small circuits (<500 nodes), CPU may be faster due to transfer overhead.
    Use backend='auto' or None to let the system choose automatically.

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
    from jax import lax
    from jax_spice.analysis.gpu_backend import (
        select_backend,
        get_device,
        get_default_dtype,
    )

    n = system.num_nodes

    # Select backend
    if backend is None or backend == "auto":
        backend = select_backend(n)

    device = get_device(backend)
    dtype = get_default_dtype(backend)

    # Build GPU residual function (pure JAX, JIT-compatible)
    # This is the key function that computes f(V) = 0
    residual_fn = system.build_gpu_residual_fn(vdd=vdd, gmin=1e-12)

    # Build Jacobian function using autodiff
    jacobian_fn = jax.jacfwd(residual_fn)

    # Initialize solution on the target device
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

        # Run JIT-compiled Newton solver
        V_final, iterations, converged, residual_norm = _dc_newton_gpu_jit(
            residual_fn,
            jacobian_fn,
            V_init,
            max_iterations,
            abstol,
            reltol,
            damping,
        )

    info = {
        "converged": bool(converged),
        "iterations": int(iterations),
        "residual_norm": float(residual_norm),
        "backend": backend,
        "device": str(device),
    }

    return V_final, info


def _dc_newton_gpu_jit(
    residual_fn: Callable,
    jacobian_fn: Callable,
    V_init: Array,
    max_iterations: int,
    abstol: float,
    reltol: float,
    damping: float,
) -> Tuple[Array, int, bool, float]:
    """JIT-compiled Newton-Raphson iteration using lax.while_loop.

    This function runs entirely on GPU with no host-device transfers
    during iteration.

    Args:
        residual_fn: Function V -> residual vector
        jacobian_fn: Function V -> Jacobian matrix
        V_init: Initial voltage guess
        max_iterations: Max iterations
        abstol: Absolute tolerance
        reltol: Relative tolerance
        damping: Damping factor

    Returns:
        Tuple of (V_final, iterations, converged, final_residual_norm)
    """
    from jax import lax

    # State: (V, iteration, converged, residual_norm)
    init_state = (V_init, 0, False, jnp.array(jnp.inf))

    def cond_fn(state):
        V, iteration, converged, residual_norm = state
        return jnp.logical_and(~converged, iteration < max_iterations)

    def body_fn(state):
        """Single Newton-Raphson step."""
        V, iteration, _, _ = state

        # Compute residual (captures residual_fn via closure)
        f = residual_fn(V)
        residual_norm = jnp.max(jnp.abs(f))

        # Check convergence
        converged = residual_norm < abstol

        # Compute Jacobian and solve (captures jacobian_fn via closure)
        # J has shape (num_nodes-1, num_nodes) because:
        # - residual has num_nodes-1 elements (excluding ground)
        # - V has num_nodes elements (including ground at index 0)
        J_full = jacobian_fn(V)

        # Extract the Jacobian w.r.t. non-ground nodes (columns 1:)
        # This gives a square matrix of shape (num_nodes-1, num_nodes-1)
        J = J_full[:, 1:]

        # Add small regularization for numerical stability
        reg = 1e-14 * jnp.eye(J.shape[0], dtype=J.dtype)
        J_reg = J + reg

        # Solve: J * delta_V = -f (delta_V has shape (num_nodes-1,))
        delta_V = jax.scipy.linalg.solve(J_reg, -f)

        # Apply damping and voltage limiting
        max_step = 2.0
        max_delta = jnp.max(jnp.abs(delta_V))
        step_scale = jnp.minimum(damping, max_step / (max_delta + 1e-15))

        # Update V (ground at index 0 stays fixed)
        V_new = V.at[1:].add(step_scale * delta_V)

        # Check delta-based convergence
        delta_norm = jnp.max(jnp.abs(step_scale * delta_V))
        v_norm = jnp.max(jnp.abs(V_new[1:]))
        delta_converged = delta_norm < (abstol + reltol * jnp.maximum(v_norm, 1.0))

        converged = jnp.logical_or(converged, delta_converged)

        return (V_new, iteration + 1, converged, residual_norm)

    # Run the Newton iteration loop
    V_final, iterations, converged, residual_norm = lax.while_loop(
        cond_fn, body_fn, init_state
    )

    return V_final, iterations, converged, residual_norm
