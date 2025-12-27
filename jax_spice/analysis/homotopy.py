"""Homotopy algorithms for DC operating point convergence.

This module implements VACASK-style homotopy continuation methods to help
Newton-Raphson converge for difficult circuits (e.g., ring oscillators,
analog circuits with feedback).

The default homotopy chain is: gdev -> gshunt -> src
- gdev: Extra GMIN added to device Jacobian diagonals (stepped down)
- gshunt: Shunt conductance from all nodes to ground (stepped down)
- src: Source stepping from 0->100% (with GMIN fallback at factor=0)

Reference: VACASK lib/hmtpgmin.cpp and lib/hmtpsrc.cpp
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Tuple, Optional

import jax.numpy as jnp
from jax import Array

from jax_spice.analysis.solver import NRConfig, NRResult, newton_solve


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
    build_residual_fn: Callable[[float, float], Callable],
    build_jacobian_fn: Callable[[float, float], Callable],
    V_init: Array,
    config: HomotopyConfig,
    nr_config: NRConfig,
    mode: str = "gdev",
) -> HomotopyResult:
    """VACASK-style adaptive GMIN stepping.

    This algorithm gradually reduces GMIN from a high starting value
    down to the target, with adaptive step adjustment based on
    convergence behavior.

    Args:
        build_residual_fn: Function (gmin, gshunt) -> residual_fn
        build_jacobian_fn: Function (gmin, gshunt) -> jacobian_fn
        V_init: Initial voltage guess
        config: Homotopy configuration
        nr_config: Newton-Raphson configuration
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

    for step in range(config.gmin_max_steps):
        homotopy_steps += 1

        # Build residual/jacobian with current gmin
        # Convert to Python float to ensure compatibility with type-checked functions
        if mode == "gdev":
            effective_gmin = float(config.gmin + at_gmin)
            gshunt = 0.0
        else:  # gshunt mode
            effective_gmin = float(config.gmin)
            gshunt = float(at_gmin)

        _debug_print(config, 2, f"  [step {step}] Building residual fn (gmin={effective_gmin:.2e}, gshunt={gshunt:.2e})")
        residual_fn = build_residual_fn(effective_gmin, gshunt)
        _debug_print(config, 2, f"  [step {step}] Building jacobian fn")
        jacobian_fn = build_jacobian_fn(effective_gmin, gshunt)

        # Run NR solver
        _debug_print(config, 2, f"  [step {step}] Running NR solver...")
        result = newton_solve(residual_fn, jacobian_fn, V, nr_config)
        _debug_print(config, 2, f"  [step {step}] NR solver returned")
        total_iterations += result.iterations

        if result.converged:
            continuation = True
            V_good = result.V
            good_gmin = at_gmin

            _debug_print(
                config,
                1,
                f"Homotopy: {mode}={at_gmin:.2e}, step {homotopy_steps} "
                f"converged in {result.iterations} iterations",
            )
            # Show voltage solution summary at high debug level
            if config.debug >= 2:
                import jax.numpy as jnp
                V_min = float(jnp.min(result.V))
                V_max = float(jnp.max(result.V))
                V_mean = float(jnp.mean(result.V))
                print(f"    V: min={V_min:.4f}, max={V_max:.4f}, mean={V_mean:.4f}", flush=True)

            if at_gmin <= target_gmin:
                # Success - reached target
                _debug_print(config, 1, f"Homotopy: {mode} stepping succeeded")
                break

            # Adaptive factor adjustment - more conservative than VACASK
            # Don't increase factor on fast convergence for difficult circuits
            if result.iterations <= nr_config.max_iterations // 4:
                # Converging quickly - keep factor unchanged (conservative)
                pass
            elif result.iterations > nr_config.max_iterations * 3 // 4:
                # Converging slowly - decrease step size
                factor = max(jnp.sqrt(factor), config.gmin_factor_min)

            # Update gmin
            if at_gmin / factor < target_gmin:
                factor = at_gmin / target_gmin
                at_gmin = target_gmin
            else:
                at_gmin = at_gmin / factor

            V = result.V  # Use as initial guess for next step
        else:
            _debug_print(
                config,
                1,
                f"Homotopy: {mode}={at_gmin:.2e}, step {homotopy_steps} "
                f"failed to converge in {result.iterations} iterations",
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

    # Final solve at original gmin (VACASK hmtpgmin.cpp lines 157-172)
    if continuation:
        residual_fn = build_residual_fn(float(config.gmin), 0.0)
        jacobian_fn = build_jacobian_fn(float(config.gmin), 0.0)
        result = newton_solve(residual_fn, jacobian_fn, V_good, nr_config)
        total_iterations += result.iterations
        homotopy_steps += 1

        _debug_print(
            config,
            1,
            f"Homotopy: {mode} final step "
            f"{'converged' if result.converged else 'failed'} in {result.iterations} iterations",
        )

        return HomotopyResult(
            converged=result.converged,
            V=result.V if result.converged else V_good,
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
    build_residual_fn: Callable[[float, float, float], Callable],
    build_jacobian_fn: Callable[[float, float, float], Callable],
    V_init: Array,
    config: HomotopyConfig,
    nr_config: NRConfig,
) -> HomotopyResult:
    """VACASK-style adaptive source stepping with GMIN fallback.

    This algorithm gradually ramps voltage/current sources from 0 to 100%.
    If the initial solve at source_factor=0 fails, it falls back to
    GMIN stepping first.

    Args:
        build_residual_fn: Function (source_scale, gmin, gshunt) -> residual_fn
        build_jacobian_fn: Function (source_scale, gmin, gshunt) -> jacobian_fn
        V_init: Initial voltage guess
        config: Homotopy configuration
        nr_config: Newton-Raphson configuration

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

    # Initial solve at source_factor=0 (VACASK hmtpsrc.cpp lines 64-69)
    residual_fn = build_residual_fn(0.0, float(config.gmin), 0.0)
    jacobian_fn = build_jacobian_fn(0.0, float(config.gmin), 0.0)
    result = newton_solve(residual_fn, jacobian_fn, V, nr_config)
    total_iterations += result.iterations
    homotopy_steps += 1

    _debug_print(
        config,
        1,
        f"Homotopy: srcfact=0.00, initial solve "
        f"{'converged' if result.converged else 'failed'} in {result.iterations} iterations",
    )

    if not result.converged:
        # Fallback to GMIN stepping at source_factor=0 (VACASK hmtpsrc.cpp lines 71-88)
        _debug_print(config, 1, "Homotopy: Trying gdev stepping at source_factor=0")

        # Build gmin-parameterized functions for source_scale=0
        def build_residual_with_gmin(gmin: float, gshunt: float) -> Callable:
            return build_residual_fn(0.0, gmin, gshunt)

        def build_jacobian_with_gmin(gmin: float, gshunt: float) -> Callable:
            return build_jacobian_fn(0.0, gmin, gshunt)

        gmin_result = gmin_stepping(
            build_residual_with_gmin,
            build_jacobian_with_gmin,
            V,
            config,
            nr_config,
            mode="gdev",
        )
        total_iterations += gmin_result.iterations
        homotopy_steps += gmin_result.homotopy_steps

        if not gmin_result.converged:
            _debug_print(config, 1, "Homotopy: Trying gshunt stepping at source_factor=0")
            gmin_result = gmin_stepping(
                build_residual_with_gmin,
                build_jacobian_with_gmin,
                V,
                config,
                nr_config,
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

        result = gmin_result
        V_good = result.V

    else:
        V_good = result.V

    # Source stepping loop (VACASK hmtpsrc.cpp lines 99-164)
    for step in range(config.source_max_steps):
        new_factor = min(good_factor + raise_step, 1.0)

        residual_fn = build_residual_fn(float(new_factor), float(config.gmin), 0.0)
        jacobian_fn = build_jacobian_fn(float(new_factor), float(config.gmin), 0.0)
        result = newton_solve(residual_fn, jacobian_fn, V_good, nr_config)
        total_iterations += result.iterations
        homotopy_steps += 1

        if result.converged:
            V_good = result.V
            good_factor = new_factor

            _debug_print(
                config,
                1,
                f"Homotopy: srcfact={new_factor:.2f}, step {homotopy_steps} "
                f"converged in {result.iterations} iterations",
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

            # Adaptive step adjustment (VACASK hmtpsrc.cpp lines 135-144)
            if result.iterations <= nr_config.max_iterations // 4:
                raise_step *= config.source_scale
            elif result.iterations > nr_config.max_iterations * 3 // 4:
                raise_step = max(raise_step / config.source_scale, config.source_step_min)
        else:
            _debug_print(
                config,
                1,
                f"Homotopy: srcfact={new_factor:.2f}, step {homotopy_steps} "
                f"failed to converge in {result.iterations} iterations",
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
    build_residual_fn: Callable,
    build_jacobian_fn: Callable,
    V_init: Array,
    config: HomotopyConfig,
    nr_config: NRConfig,
) -> HomotopyResult:
    """Run VACASK-style homotopy chain: gdev -> gshunt -> src.

    Tries each algorithm in sequence until one succeeds.

    Args:
        build_residual_fn: Function (source_scale, gmin, gshunt) -> residual_fn
        build_jacobian_fn: Function (source_scale, gmin, gshunt) -> jacobian_fn
        V_init: Initial voltage guess
        config: Homotopy configuration
        nr_config: Newton-Raphson configuration

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
            # Build gmin-only parameterized functions (source_scale=1.0)
            def build_residual_gmin(gmin: float, gshunt: float) -> Callable:
                return build_residual_fn(1.0, gmin, gshunt)

            def build_jacobian_gmin(gmin: float, gshunt: float) -> Callable:
                return build_jacobian_fn(1.0, gmin, gshunt)

            result = gmin_stepping(
                build_residual_gmin,
                build_jacobian_gmin,
                V,
                config,
                nr_config,
                mode="gdev",
            )
        elif algorithm == "gshunt":
            # Build gshunt-only parameterized functions (source_scale=1.0)
            def build_residual_gshunt(gmin: float, gshunt: float) -> Callable:
                return build_residual_fn(1.0, gmin, gshunt)

            def build_jacobian_gshunt(gmin: float, gshunt: float) -> Callable:
                return build_jacobian_fn(1.0, gmin, gshunt)

            result = gmin_stepping(
                build_residual_gshunt,
                build_jacobian_gshunt,
                V,
                config,
                nr_config,
                mode="gshunt",
            )
        elif algorithm == "src":
            result = source_stepping(
                build_residual_fn,
                build_jacobian_fn,
                V,
                config,
                nr_config,
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
