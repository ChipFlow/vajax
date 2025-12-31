"""Homotopy algorithms for DC operating point convergence.

This module implements VACASK-style homotopy continuation methods to help
Newton-Raphson converge for difficult circuits (e.g., ring oscillators,
analog circuits with feedback).

The default homotopy chain is: gdev -> gshunt -> src
- gdev: Extra GMIN added to device Jacobian diagonals (stepped down)
- gshunt: Shunt conductance from all nodes to ground (stepped down)
- src: Source stepping from 0->100% (with GMIN fallback at factor=0)

Reference: VACASK lib/hmtpgmin.cpp and lib/hmtpsrc.cpp

Note on interface design:
The homotopy functions take the cached nr_solve function directly, which
uses analytic jacobians from OpenVAF. This avoids jax.jacfwd() autodiff
and benefits from JIT compilation and warmup of the transient solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
from jax import Array


@dataclass
class HomotopyConfig:
    """Configuration for homotopy algorithms.

    Based on VACASK's default options from lib/options.cpp.
    """

    # Base GMIN (applied to nonlinear device diagonals)
    gmin: float = 1e-12

    # GMIN stepping parameters (gdev/gshunt modes)
    gdev_start: float = 1e-3  # homotopy_startgmin
    gdev_target: float = 1e-13  # Target: gmin/10
    gmin_factor: float = 10.0  # homotopy_gminfactor
    gmin_factor_min: float = 1.1  # homotopy_mingminfactor
    gmin_factor_max: float = 100.0  # homotopy_maxgminfactor
    gmin_max: float = 1.0  # homotopy_maxgmin
    gmin_max_steps: int = 100  # homotopy_gminsteps

    # Source stepping parameters
    source_step: float = 0.1  # homotopy_srcstep
    source_step_min: float = 0.001  # homotopy_minsrcstep
    source_scale: float = 2.0  # homotopy_srcscale
    source_max_steps: int = 100  # homotopy_srcsteps

    # Homotopy chain (default: gdev -> gshunt -> src)
    chain: Tuple[str, ...] = ("gdev", "gshunt", "src")

    # NR config
    max_iterations: int = 100
    abstol: float = 1e-9

    # Debug level (0=silent, 1=progress, 2=verbose)
    debug: int = 0


@dataclass
class HomotopyResult:
    """Result from a homotopy algorithm."""

    converged: bool
    V: Array
    method: str = ""
    iterations: int = 0
    homotopy_steps: int = 0
    final_gmin: float = 0.0
    final_source_scale: float = 1.0


def _debug_print(config: HomotopyConfig, level: int, msg: str) -> None:
    """Print debug message if debug level is high enough."""
    if config.debug >= level:
        print(msg, flush=True)


def gmin_stepping(
    nr_solve: Callable,
    V_init: Array,
    vsource_vals: Array,
    isource_vals: Array,
    Q_prev: Array,
    device_arrays: Dict[str, Any],
    source_scale: float,
    config: HomotopyConfig,
    mode: str = "gdev",
) -> HomotopyResult:
    """VACASK-style adaptive GMIN stepping using cached NR solver.

    This algorithm gradually reduces GMIN from a high starting value
    down to the target, with adaptive step adjustment based on
    convergence behavior.

    Args:
        nr_solve: Cached NR solver function with analytic jacobians
        V_init: Initial voltage guess
        vsource_vals: DC voltage source values (unscaled)
        isource_vals: DC current source values (unscaled)
        Q_prev: Previous charges (zeros for DC)
        device_arrays: Dict of device arrays to pass to nr_solve (avoids XLA constant folding)
        source_scale: Fixed source scaling factor for this stepping
        config: Homotopy configuration
        mode: "gdev" (device GMIN) or "gshunt" (shunt to ground)

    Returns:
        HomotopyResult with final voltage and convergence info
    """
    at_gmin = config.gdev_start
    target_gmin = config.gdev_target
    factor = config.gmin_factor

    V = V_init
    V_good = V_init
    good_gmin = at_gmin
    continuation = False
    total_iterations = 0
    homotopy_steps = 0

    _debug_print(config, 1, f"Homotopy: Starting {mode} stepping from {at_gmin:.2e}")

    # Scale sources by source_scale
    scaled_vsource = vsource_vals * source_scale
    scaled_isource = isource_vals * source_scale

    for step in range(config.gmin_max_steps):
        homotopy_steps += 1

        # Set gmin/gshunt based on mode
        if mode == "gdev":
            effective_gmin = config.gmin + at_gmin
            gshunt = 0.0
        else:  # gshunt mode
            effective_gmin = config.gmin
            gshunt = at_gmin

        _debug_print(config, 2, f"  [step {step}] gmin={effective_gmin:.2e}, gshunt={gshunt:.2e}")

        # Run NR solver with analytic jacobians
        V_new, iters, converged, max_f, _ = nr_solve(
            V, scaled_vsource, scaled_isource, Q_prev, 0.0, device_arrays, effective_gmin, gshunt
        )
        total_iterations += int(iters)

        if converged:
            continuation = True
            V_good = V_new
            good_gmin = at_gmin

            _debug_print(
                config,
                1,
                f"Homotopy: {mode}={at_gmin:.2e}, step {homotopy_steps} "
                f"converged in {iters} iterations",
            )

            if at_gmin <= target_gmin:
                # Success - reached target
                _debug_print(config, 1, f"Homotopy: {mode} stepping succeeded")
                break

            # Adaptive factor adjustment
            if iters <= config.max_iterations // 4:
                pass  # Keep factor unchanged
            elif iters > config.max_iterations * 3 // 4:
                factor = max(jnp.sqrt(factor), config.gmin_factor_min)

            # Update gmin
            if at_gmin / factor < target_gmin:
                factor = at_gmin / target_gmin
                at_gmin = target_gmin
            else:
                at_gmin = at_gmin / factor

            V = V_new  # Use as initial guess for next step
        else:
            _debug_print(
                config,
                1,
                f"Homotopy: {mode}={at_gmin:.2e}, step {homotopy_steps} "
                f"failed to converge in {iters} iterations",
            )

            if not continuation:
                # No good solution yet, increase gmin
                at_gmin = at_gmin * factor
                if at_gmin > config.gmin_max:
                    _debug_print(config, 1, f"Homotopy: {mode} stepping failed (gmin too large)")
                    return HomotopyResult(
                        converged=False,
                        V=V_init,
                        method=f"{mode}_stepping",
                        iterations=total_iterations,
                        homotopy_steps=homotopy_steps,
                        final_gmin=at_gmin,
                    )
            else:
                # Have a good solution, decrease factor and backtrack
                factor = factor**0.25
                if factor < config.gmin_factor_min:
                    _debug_print(config, 1, f"Homotopy: {mode} stepping failed (factor exhausted)")
                    break
                V = V_good
                at_gmin = good_gmin

    # Final solve at original gmin
    if continuation:
        V_final, iters, converged, _, _ = nr_solve(
            V_good, scaled_vsource, scaled_isource, Q_prev, 0.0, device_arrays, config.gmin, 0.0
        )
        total_iterations += int(iters)
        homotopy_steps += 1

        _debug_print(
            config,
            1,
            f"Homotopy: {mode} final step "
            f"{'converged' if converged else 'failed'} in {iters} iterations",
        )

        return HomotopyResult(
            converged=bool(converged),
            V=V_final if converged else V_good,
            method=f"{mode}_stepping",
            iterations=total_iterations,
            homotopy_steps=homotopy_steps,
            final_gmin=config.gmin,
        )

    return HomotopyResult(
        converged=False,
        V=V_good,
        method=f"{mode}_stepping",
        iterations=total_iterations,
        homotopy_steps=homotopy_steps,
        final_gmin=at_gmin,
    )


def source_stepping(
    nr_solve: Callable,
    V_init: Array,
    vsource_vals: Array,
    isource_vals: Array,
    Q_prev: Array,
    device_arrays: Dict[str, Any],
    config: HomotopyConfig,
) -> HomotopyResult:
    """VACASK-style adaptive source stepping with GMIN fallback.

    This algorithm gradually ramps voltage/current sources from 0 to 100%.
    If the initial solve at source_factor=0 fails, it falls back to
    GMIN stepping first.

    Args:
        nr_solve: Cached NR solver function with analytic jacobians
        V_init: Initial voltage guess
        vsource_vals: DC voltage source values (unscaled)
        isource_vals: DC current source values (unscaled)
        Q_prev: Previous charges (zeros for DC)
        device_arrays: Dict of device arrays to pass to nr_solve (avoids XLA constant folding)
        config: Homotopy configuration

    Returns:
        HomotopyResult with final voltage and convergence info
    """
    raise_step = config.source_step
    V = V_init
    V_good = V_init
    good_factor = 0.0
    total_iterations = 0
    homotopy_steps = 0

    _debug_print(config, 1, "Homotopy: Starting source stepping")

    # Initial solve at source_factor=0
    zero_vsource = vsource_vals * 0.0
    zero_isource = isource_vals * 0.0
    V_new, iters, converged, _, _ = nr_solve(
        V, zero_vsource, zero_isource, Q_prev, 0.0, device_arrays, config.gmin, 0.0
    )
    total_iterations += int(iters)
    homotopy_steps += 1

    _debug_print(
        config,
        1,
        f"Homotopy: srcfact=0.00, initial solve "
        f"{'converged' if converged else 'failed'} in {iters} iterations",
    )

    if not converged:
        # Fallback to GMIN stepping at source_factor=0
        _debug_print(config, 1, "Homotopy: Trying gdev stepping at source_factor=0")

        gmin_result = gmin_stepping(
            nr_solve,
            V,
            vsource_vals,
            isource_vals,
            Q_prev,
            device_arrays,
            source_scale=0.0,
            config=config,
            mode="gdev",
        )
        total_iterations += gmin_result.iterations
        homotopy_steps += gmin_result.homotopy_steps

        if not gmin_result.converged:
            _debug_print(config, 1, "Homotopy: Trying gshunt stepping at source_factor=0")
            gmin_result = gmin_stepping(
                nr_solve,
                V,
                vsource_vals,
                isource_vals,
                Q_prev,
                device_arrays,
                source_scale=0.0,
                config=config,
                mode="gshunt",
            )
            total_iterations += gmin_result.iterations
            homotopy_steps += gmin_result.homotopy_steps

        if not gmin_result.converged:
            _debug_print(config, 1, "Homotopy: Source stepping failed (could not solve at source=0)")
            return HomotopyResult(
                converged=False,
                V=V_init,
                method="source_stepping",
                iterations=total_iterations,
                homotopy_steps=homotopy_steps,
                final_source_scale=0.0,
            )

        V_good = gmin_result.V
    else:
        V_good = V_new

    # Source stepping loop
    for step in range(config.source_max_steps):
        new_factor = min(good_factor + raise_step, 1.0)

        scaled_vsource = vsource_vals * new_factor
        scaled_isource = isource_vals * new_factor
        V_new, iters, converged, _, _ = nr_solve(
            V_good, scaled_vsource, scaled_isource, Q_prev, 0.0, device_arrays, config.gmin, 0.0
        )
        total_iterations += int(iters)
        homotopy_steps += 1

        if converged:
            V_good = V_new
            good_factor = new_factor

            _debug_print(
                config,
                1,
                f"Homotopy: srcfact={new_factor:.2f}, step {homotopy_steps} "
                f"converged in {iters} iterations",
            )

            if good_factor >= 1.0:
                # Success!
                _debug_print(config, 1, "Homotopy: Source stepping succeeded")
                return HomotopyResult(
                    converged=True,
                    V=V_good,
                    method="source_stepping",
                    iterations=total_iterations,
                    homotopy_steps=homotopy_steps,
                    final_source_scale=1.0,
                )

            # Adaptive step adjustment
            if iters <= config.max_iterations // 4:
                raise_step *= config.source_scale
            elif iters > config.max_iterations * 3 // 4:
                raise_step = max(raise_step / config.source_scale, config.source_step_min)
        else:
            _debug_print(
                config,
                1,
                f"Homotopy: srcfact={new_factor:.2f}, step {homotopy_steps} "
                f"failed to converge in {iters} iterations",
            )

            # Not converged, reduce step and retry
            raise_step *= 0.5
            if raise_step < config.source_step_min:
                _debug_print(config, 1, "Homotopy: Source stepping failed (step too small)")
                break

    return HomotopyResult(
        converged=good_factor >= 1.0,
        V=V_good,
        method="source_stepping",
        iterations=total_iterations,
        homotopy_steps=homotopy_steps,
        final_source_scale=good_factor,
    )


def run_homotopy_chain(
    nr_solve: Callable,
    V_init: Array,
    vsource_vals: Array,
    isource_vals: Array,
    Q_prev: Array,
    device_arrays: Dict[str, Any],
    config: HomotopyConfig,
) -> HomotopyResult:
    """Run VACASK-style homotopy chain: gdev -> gshunt -> src.

    Tries each algorithm in sequence until one succeeds.

    Args:
        nr_solve: Cached NR solver function with analytic jacobians
        V_init: Initial voltage guess
        vsource_vals: DC voltage source values (unscaled)
        isource_vals: DC current source values (unscaled)
        Q_prev: Previous charges (zeros for DC)
        device_arrays: Dict of device arrays to pass to nr_solve (avoids XLA constant folding)
        config: Homotopy configuration

    Returns:
        HomotopyResult with final voltage and convergence info
    """
    V = V_init
    total_iterations = 0
    total_steps = 0

    _debug_print(config, 1, f"Homotopy: Running chain {config.chain}")

    for algorithm in config.chain:
        _debug_print(config, 1, f"Homotopy: Trying {algorithm}")

        if algorithm == "gdev":
            result = gmin_stepping(
                nr_solve,
                V,
                vsource_vals,
                isource_vals,
                Q_prev,
                device_arrays,
                source_scale=1.0,  # Full sources for GMIN-only stepping
                config=config,
                mode="gdev",
            )
        elif algorithm == "gshunt":
            result = gmin_stepping(
                nr_solve,
                V,
                vsource_vals,
                isource_vals,
                Q_prev,
                device_arrays,
                source_scale=1.0,  # Full sources for GSHUNT-only stepping
                config=config,
                mode="gshunt",
            )
        elif algorithm == "src":
            result = source_stepping(
                nr_solve,
                V,
                vsource_vals,
                isource_vals,
                Q_prev,
                device_arrays,
                config,
            )
        else:
            _debug_print(config, 1, f"Homotopy: Unknown algorithm '{algorithm}', skipping")
            continue

        total_iterations += result.iterations
        total_steps += result.homotopy_steps

        if result.converged:
            _debug_print(config, 1, f"Homotopy: Chain succeeded with {algorithm}")
            return HomotopyResult(
                converged=True,
                V=result.V,
                method=f"chain_{result.method}",
                iterations=total_iterations,
                homotopy_steps=total_steps,
                final_gmin=result.final_gmin,
                final_source_scale=result.final_source_scale,
            )

        # Use best V from failed attempt as next starting point
        V = result.V

    _debug_print(config, 1, "Homotopy: Chain exhausted, all algorithms failed")
    return HomotopyResult(
        converged=False,
        V=V,
        method="chain_failed",
        iterations=total_iterations,
        homotopy_steps=total_steps,
    )
