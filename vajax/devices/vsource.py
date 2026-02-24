"""Voltage and current source functions for VA-JAX

Provides pure functions for computing source waveforms:
- pulse_voltage: Calculate pulse waveform voltage at time t
- pulse_voltage_jax: JAX-compatible pulse voltage for JIT compilation
- vsource_batch: Vectorized voltage source evaluation for batched processing
- isource_batch: Vectorized current source evaluation for batched processing
"""

from typing import Tuple

import jax.numpy as jnp
from jax import Array


def pulse_voltage(t: float, pulse_params: Tuple[float, ...]) -> float:
    """Calculate pulse voltage at time t

    SPICE PULSE syntax: PULSE(v0 v1 td tr tf pw per)

    Args:
        t: Current time
        pulse_params: (v0, v1, td, tr, tf, pw, per)
            v0: Initial/low voltage
            v1: Pulsed/high voltage
            td: Delay time
            tr: Rise time
            tf: Fall time
            pw: Pulse width
            per: Period (0 for single pulse)

    Returns:
        Voltage at time t
    """
    # Unpack parameters with defaults
    v0 = pulse_params[0] if len(pulse_params) > 0 else 0.0
    v1 = pulse_params[1] if len(pulse_params) > 1 else 1.0
    td = pulse_params[2] if len(pulse_params) > 2 else 0.0
    tr = pulse_params[3] if len(pulse_params) > 3 else 1e-12
    tf = pulse_params[4] if len(pulse_params) > 4 else 1e-12
    pw = pulse_params[5] if len(pulse_params) > 5 else 1e-9
    per = pulse_params[6] if len(pulse_params) > 6 else 0.0

    # Ensure minimum rise/fall times for numerical stability
    tr = max(tr, 1e-15)
    tf = max(tf, 1e-15)

    # Before delay: return v0
    if t < td:
        return v0

    # Calculate time within period
    t_rel = t - td
    if per > 0:
        t_rel = t_rel % per

    # Piecewise linear waveform
    if t_rel < tr:
        # Rising edge
        return v0 + (v1 - v0) * (t_rel / tr)
    elif t_rel < tr + pw:
        # High level
        return v1
    elif t_rel < tr + pw + tf:
        # Falling edge
        t_fall = t_rel - tr - pw
        return v1 - (v1 - v0) * (t_fall / tf)
    else:
        # Low level
        return v0


def pulse_voltage_jax(
    t: Array, v0: Array, v1: Array, td: Array, tr: Array, tf: Array, pw: Array, per: Array
) -> Array:
    """JAX-compatible pulse voltage calculation

    Uses jnp.where for differentiability and JIT compatibility.

    Args:
        t: Time array
        v0: Low voltage
        v1: High voltage
        td: Delay time
        tr: Rise time
        tf: Fall time
        pw: Pulse width
        per: Period (0 for single pulse)

    Returns:
        Voltage array at times t
    """
    # Ensure minimum rise/fall times
    tr = jnp.maximum(tr, 1e-15)
    tf = jnp.maximum(tf, 1e-15)

    # Before delay
    before_delay = t < td

    # Time relative to delay
    t_rel = t - td

    # Handle periodic waveform
    t_rel = jnp.where(per > 0, t_rel % per, t_rel)

    # Rising edge
    in_rise = (t_rel >= 0) & (t_rel < tr)
    v_rise = v0 + (v1 - v0) * (t_rel / tr)

    # High level
    in_high = (t_rel >= tr) & (t_rel < tr + pw)

    # Falling edge
    in_fall = (t_rel >= tr + pw) & (t_rel < tr + pw + tf)
    t_fall = t_rel - tr - pw
    v_fall = v1 - (v1 - v0) * (t_fall / tf)

    # Construct output
    v = jnp.where(
        before_delay,
        v0,
        jnp.where(in_rise, v_rise, jnp.where(in_high, v1, jnp.where(in_fall, v_fall, v0))),
    )

    return v


# =============================================================================
# Vectorized Batch Functions
# =============================================================================


def vsource_batch(V_batch: Array, V_target: Array, G_BIG: float = 1e12) -> Tuple[Array, Array]:
    """Vectorized voltage source evaluation for batch processing

    Evaluates multiple voltage sources in parallel using JAX operations.

    Args:
        V_batch: Terminal voltages (n, 2) - [[V_p, V_n], ...] for each source
        V_target: Target voltages (n,) - voltage each source enforces
        G_BIG: Large conductance for voltage enforcement (default: 1e12)

    Returns:
        Tuple of:
            I_batch: Current at positive terminal (n,) - current flowing into p
            G_batch: Conductance values (n,) - all equal to G_BIG

    Note:
        The voltage source forces V_p - V_n = V_target by injecting current:
        I = G_BIG * ((V_p - V_n) - V_target)

        Stamps into MNA system:
        - Residual: f[p] += I, f[n] -= I
        - Jacobian: G[p,p] += G, G[p,n] -= G, G[n,p] -= G, G[n,n] += G
    """
    V_actual = V_batch[:, 0] - V_batch[:, 1]  # V_p - V_n for each source
    I = G_BIG * (V_actual - V_target)
    G = jnp.full_like(I, G_BIG)
    return I, G


def isource_batch(I_target: Array) -> Array:
    """Vectorized current source evaluation for batch processing

    Args:
        I_target: Target currents (n,) - current each source injects

    Returns:
        I_batch: Currents (n,) - same as I_target (current sources are trivial)

    Note:
        Current sources have no conductance contribution.
        Stamps into MNA system:
        - Residual: f[p] -= I, f[n] += I
        - Jacobian: no contribution (all zeros)
    """
    return I_target
