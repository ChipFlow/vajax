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
    source_stepping_solve,
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
    backend: Optional[str] = None,
) -> Tuple[Array, Dict]:
    """Find DC operating point using source stepping for difficult circuits.

    Source stepping is a homotopy method that gradually ramps the supply voltage
    from 0 to the target value. At Vdd=0, all transistors are OFF and the circuit
    is trivially solved. As Vdd increases, the solution evolves continuously.

    This is particularly effective for:
    - Large digital circuits with many cascaded stages
    - Circuits where all inputs are held at fixed values (e.g., logic low)
    - Circuits that fail to converge because PMOS are ON but NMOS are OFF

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
        backend: 'gpu', 'cpu', or None (auto-select)

    Returns:
        Tuple of (solution, info) where:
            solution: Node voltages (shape: [num_nodes])
            info: Dict with convergence information including source_steps
    """
    import numpy as np

    n = system.num_nodes

    # Select backend
    if backend is None or backend == "auto":
        backend = select_backend(n)

    device = get_device(backend)
    dtype = get_default_dtype(backend)

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
    vdd_values = np.linspace(vdd_target / vdd_steps, vdd_target, vdd_steps)

    if verbose:
        print(f"Source stepping: 0 -> {vdd_target:.2f}V in {vdd_steps} steps", flush=True)

    converged_at_target = False
    last_info = {'converged': False, 'iterations': 0, 'residual_norm': 1e20}

    with jax.default_device(device):
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

            # Use relaxed tolerance for intermediate steps
            step_abstol = abstol if is_final_step else max(abstol, 1e-4)

            # Build residual function for this Vdd level
            residual_fn = system.build_gpu_residual_fn(vdd=vdd_step, gmin=1e-6)
            jacobian_fn = jax.jacfwd(residual_fn)

            V_init = jnp.array(V, dtype=dtype)
            config = NRConfig(
                max_iterations=max_iterations_per_step,
                abstol=step_abstol,
                reltol=reltol,
                damping=damping,
                max_step=2.0,
            )

            result = newton_solve(residual_fn, jacobian_fn, V_init, config)

            V = np.array(result.V)
            total_iterations += result.iterations
            last_info = {
                'converged': result.converged,
                'iterations': result.iterations,
                'residual_norm': result.residual_norm,
            }

            if verbose:
                print(f"    -> iter={result.iterations}, residual={result.residual_norm:.2e}, "
                      f"converged={result.converged}", flush=True)

            # Handle non-convergence
            if not result.converged:
                if not is_final_step and result.residual_norm < 1e-3:
                    if verbose:
                        print(f"    Accepting partial convergence for intermediate step", flush=True)
                else:
                    # Try with higher GMIN
                    if verbose:
                        print(f"    Trying with higher GMIN (1e-3)...", flush=True)

                    residual_fn_h = system.build_gpu_residual_fn(vdd=vdd_step, gmin=1e-3)
                    jacobian_fn_h = jax.jacfwd(residual_fn_h)

                    config_h = NRConfig(
                        max_iterations=max_iterations_per_step * 2,
                        abstol=1e-3 if not is_final_step else abstol,
                        reltol=reltol,
                        damping=damping,
                        max_step=2.0,
                    )

                    result_h = newton_solve(residual_fn_h, jacobian_fn_h, V_init, config_h)

                    if result_h.residual_norm < result.residual_norm:
                        V = np.array(result_h.V)
                        total_iterations += result_h.iterations
                        last_info = {
                            'converged': result_h.converged,
                            'iterations': result_h.iterations,
                            'residual_norm': result_h.residual_norm,
                        }

                    if verbose:
                        print(f"      -> iter={result_h.iterations}, residual={result_h.residual_norm:.2e}",
                              flush=True)

            # Only check for true convergence at final step
            if is_final_step and last_info['converged']:
                converged_at_target = True

            step_idx += 1

    result_info = {
        'converged': converged_at_target,
        'iterations': total_iterations,
        'source_steps': source_steps,
        'final_vdd': float(vdd_values[-1]),
        'residual_norm': last_info['residual_norm'],
        'backend': backend,
    }

    if verbose:
        print(f"  Source stepping complete: steps={source_steps}, "
              f"total_iter={total_iterations}, converged={converged_at_target}", flush=True)

    return jnp.array(V), result_info


def dc_operating_point_source_stepping_jit(
    system: MNASystem,
    initial_guess: Optional[Array] = None,
    vdd_target: float = 1.2,
    vdd_steps: int = 12,
    max_iterations_per_step: int = 50,
    abstol: float = 1e-9,
    reltol: float = 1e-3,
    damping: float = 1.0,
    gmin: float = 1e-6,
    backend: Optional[str] = None,
) -> Tuple[Array, Dict]:
    """JIT-compiled source stepping for DC operating point.

    This is a fully JIT-compiled version of source stepping using lax.scan.
    The entire stepping sequence runs on GPU without host-device transfers.

    Args:
        system: MNA system with devices (must have device_groups built)
        initial_guess: Initial voltage estimate
        vdd_target: Target supply voltage
        vdd_steps: Number of voltage steps
        max_iterations_per_step: Max NR iterations per step
        abstol: Absolute tolerance
        reltol: Relative tolerance
        damping: Damping factor
        gmin: GMIN conductance
        backend: 'gpu', 'cpu', or None (auto-select)

    Returns:
        Tuple of (solution, info)
    """
    n = system.num_nodes

    # Select backend
    if backend is None or backend == "auto":
        backend = select_backend(n)

    device = get_device(backend)
    dtype = get_default_dtype(backend)

    # Ensure device groups are built
    if not system.device_groups:
        system.build_device_groups(vdd=vdd_target)

    # Build parameterized residual function
    residual_fn = system.build_parameterized_residual_fn(gmin=gmin)

    # Generate VDD scale steps (0.1 to 1.0)
    vdd_scales = jnp.linspace(1.0 / vdd_steps, 1.0, vdd_steps)

    with jax.default_device(device):
        if initial_guess is not None:
            V_init = jnp.array(initial_guess, dtype=dtype)
        else:
            V_init = jnp.zeros(n, dtype=dtype)

        config = NRConfig(
            max_iterations=max_iterations_per_step,
            abstol=abstol,
            reltol=reltol,
            damping=damping,
            max_step=2.0,
        )

        # Run JIT-compiled source stepping
        final_V, all_V, converged = source_stepping_solve(
            residual_fn, V_init, vdd_scales, config
        )

    info = {
        'converged': bool(converged[-1]),
        'all_converged': bool(jnp.all(converged)),
        'source_steps': vdd_steps,
        'final_vdd': float(vdd_target),
        'backend': backend,
        'device': str(device),
        'mode': 'jit_source_stepping',
    }

    return final_V, info


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
    backend: Optional[str] = None,
) -> Tuple[Array, Dict]:
    """Find DC operating point using GMIN stepping for difficult circuits.

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
        backend: 'gpu', 'cpu', or None (auto-select)

    Returns:
        Tuple of (solution, info) where:
            solution: Node voltages (shape: [num_nodes])
            info: Dict with convergence information including gmin_steps
    """
    import numpy as np

    n = system.num_nodes

    # Select backend
    if backend is None or backend == "auto":
        backend = select_backend(n)

    device = get_device(backend)
    dtype = get_default_dtype(backend)

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

    if verbose:
        print(f"GMIN stepping: {start_gmin:.0e} -> {target_gmin:.0e} (factor {gmin_factor})", flush=True)

    last_info = {'converged': False, 'iterations': 0, 'residual_norm': 1e20}

    with jax.default_device(device):
        while gmin >= target_gmin:
            gmin_steps += 1

            if verbose:
                print(f"  GMIN step {gmin_steps}: gmin={gmin:.0e}", flush=True)

            # Build residual function with current GMIN
            residual_fn = system.build_gpu_residual_fn(vdd=vdd, gmin=gmin)
            jacobian_fn = jax.jacfwd(residual_fn)

            V_init = jnp.array(V, dtype=dtype)
            config = NRConfig(
                max_iterations=max_iterations_per_step,
                abstol=abstol,
                reltol=reltol,
                damping=damping,
                max_step=2.0,
            )

            result = newton_solve(residual_fn, jacobian_fn, V_init, config)

            V = np.array(result.V)
            total_iterations += result.iterations
            last_info = {
                'converged': result.converged,
                'iterations': result.iterations,
                'residual_norm': result.residual_norm,
            }

            if verbose:
                print(f"    -> iter={result.iterations}, residual={result.residual_norm:.2e}, "
                      f"converged={result.converged}", flush=True)

            if not result.converged:
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
                break

            # Reduce GMIN for next step
            gmin = max(gmin / gmin_factor, target_gmin)

    final_converged = last_info['converged'] and gmin <= target_gmin

    result_info = {
        'converged': final_converged,
        'iterations': total_iterations,
        'gmin_steps': gmin_steps,
        'final_gmin': gmin,
        'residual_norm': last_info['residual_norm'],
        'backend': backend,
    }

    if verbose:
        print(f"  GMIN stepping complete: gmin_steps={gmin_steps}, "
              f"total_iter={total_iterations}, converged={final_converged}", flush=True)

    return jnp.array(V), result_info


# Backwards compatibility aliases
dc_operating_point_gpu = dc_operating_point
dc_operating_point_sparse = dc_operating_point
