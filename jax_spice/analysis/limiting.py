"""SPICE-compatible voltage limiting functions for NR convergence.

These functions implement the classic SPICE limiting algorithms (pnjlim, fetlim)
that compress large voltage changes during Newton-Raphson iteration to improve
convergence. Without limiting, large voltage steps can cause device models to
evaluate at unrealistic operating points, leading to poor convergence.

The key insight is that PN junction currents are exponential in voltage:
    I = Is * (exp(V/Vt) - 1)

A 1V change in junction voltage can cause current to change by a factor of
e^(1/0.026) ≈ 2e16. By using logarithmic compression, we limit the step size
while still making progress toward the solution.

Reference: VACASK limitfunctions.cpp DEVpnjlim (improved Buermen algorithm)
"""

import jax.numpy as jnp
from jax import Array


def pnjlim(vnew: Array, vold: Array, vt: float = 0.026, vcrit: float = 0.6) -> Array:
    """PN junction voltage limiting (logarithmic damping).

    Matches VACASK's DEVpnjlim (limitfunctions.cpp:72-146), which is the
    Buermen-improved version of SPICE3's algorithm. Three cases:

    1. vnew > vcrit AND |delta| > 2*vt (forward, large change):
       - vold > 0, delta > 0: vold + vt * log(1 + delta/vt)
       - vold > 0, delta < 0: vold - vt * log(1 - delta/vt)
       - vold <= 0: vt * log(vnew/vt)
    2. vnew < 0 (negative voltage clamping):
       - vold > 0: clamp to max(vnew, -vold - 1)
       - vold <= 0: clamp to max(vnew, 2*vold - 1)
    3. Otherwise: no limiting

    Args:
        vnew: Proposed new voltage (from NR step)
        vold: Previous voltage
        vt: Thermal voltage (kT/q, ≈ 0.026V at 300K)
        vcrit: Critical voltage above which limiting is applied (typ. 0.6V)

    Returns:
        Limited voltage that is closer to vold than vnew when the step is large
    """
    delta_v = vnew - vold
    arg = delta_v / vt

    # Case 1: vnew > vcrit AND large change → logarithmic compression
    fwd_condition = (vnew > vcrit) & (jnp.abs(delta_v) > 2 * vt)

    # Forward bias (vold > 0): separate positive/negative step branches
    # Positive step: vold + vt * log(1 + arg)  — compresses large increases
    # Negative step: vold - vt * log(1 - arg)  — compresses large decreases
    # Both branches produce values closer to vold than vnew.
    limited_fwd_pos = vold + vt * jnp.log(jnp.maximum(1.0 + arg, 1e-30))
    limited_fwd_neg = vold - vt * jnp.log(jnp.maximum(1.0 - arg, 1e-30))
    limited_forward = jnp.where(arg >= 0, limited_fwd_pos, limited_fwd_neg)

    # Reverse recovery (vold <= 0): compress vnew directly
    limited_reverse = jnp.where(
        vnew > 0,
        vt * jnp.log(jnp.maximum(vnew / vt, 1e-30)),
        vcrit,
    )
    limited_fwd = jnp.where(vold > 0, limited_forward, limited_reverse)

    # Case 2: vnew < 0 → negative voltage clamping (ngspice extension)
    neg_clamp = jnp.where(vold > 0, -vold - 1.0, 2.0 * vold - 1.0)
    limited_neg = jnp.maximum(vnew, neg_clamp)

    # Combine all cases
    result = jnp.where(
        fwd_condition,
        limited_fwd,
        jnp.where(vnew < 0, limited_neg, vnew),
    )

    return result


def fetlim(vnew: Array, vold: Array, vto: float = 0.5) -> Array:
    """FET gate-source voltage limiting (region-based).

    Limits gate-source voltage changes based on operating region.
    In the on-region (Vgs > Vto), larger steps are allowed.
    In the off-region, steps are limited to 0.5V.

    Args:
        vnew: Proposed new Vgs voltage
        vold: Previous Vgs voltage
        vto: Threshold voltage (default 0.5V for NMOS)

    Returns:
        Limited voltage
    """
    # Determine operating region
    in_on_region = vold >= vto

    # Allow larger steps when well into on-region
    # Step limit = 2 * |vold - vto| + 2 when on, 0.5 when off
    max_step = jnp.where(in_on_region, 2 * jnp.abs(vold - vto) + 2, 0.5)

    delta = vnew - vold
    limited = vold + jnp.clip(delta, -max_step, max_step)

    return limited


def apply_voltage_damping(
    V_new: Array,
    V_old: Array,
    vt: float = 0.026,
    vcrit: float = 0.6,
    max_step: float = 0.3,
    nr_damping: float = 1.0,
) -> Array:
    """Apply voltage damping to all voltage updates.

    This combines three damping strategies:
    1. Global NR damping: scale the step by nr_damping factor
    2. pnjlim-style logarithmic compression for PN junction regions
       (where vnew > vcrit and vold > 0)
    3. General step limiting for large voltage changes that don't
       trigger pnjlim (important for MOSFETs where Vgs may start at 0)

    The global NR damping is applied first, then pnjlim, then general limiting.
    This order ensures that even with nr_damping=1.0, the pnjlim and step
    limiting still provide protection against large voltage jumps.

    Args:
        V_new: Proposed new voltage vector (from NR step, excluding ground)
        V_old: Previous voltage vector (excluding ground)
        vt: Thermal voltage (kT/q ≈ 0.026V at 300K)
        vcrit: Critical voltage above which pnjlim is applied
        max_step: Maximum voltage step for general limiting (default 0.3V)
        nr_damping: Global NR damping factor (default 1.0 = no damping).
                    Values < 1.0 reduce step size (e.g., 0.5 = half steps).

    Returns:
        Damped voltage vector
    """
    # First apply global NR damping by scaling the step
    if nr_damping != 1.0:
        delta_raw = V_new - V_old
        V_new = V_old + nr_damping * delta_raw

    # Then apply pnjlim for PN junction regions
    result = pnjlim(V_new, V_old, vt, vcrit)

    # Then apply general step limiting for nodes where pnjlim didn't help
    # This catches MOSFET Vgs starting from 0V where vold > 0 condition fails
    delta = result - V_old
    abs_delta = jnp.abs(delta)

    # Only apply step limiting where:
    # 1. The change is large (> max_step)
    # 2. pnjlim didn't already reduce it significantly
    needs_limiting = abs_delta > max_step

    # Logarithmic compression for large steps (like pnjlim but without vold>0 condition)
    # This compresses delta while preserving direction
    compressed_delta = jnp.sign(delta) * max_step * jnp.log1p(abs_delta / max_step)
    limited = V_old + compressed_delta

    # Apply step limiting only where needed
    result = jnp.where(needs_limiting, limited, result)

    return result
