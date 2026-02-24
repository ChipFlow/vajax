"""Source device handling for VA-JAX.

This module handles voltage and current source devices:
- Time-varying source functions (DC, pulse, sine, PWL, exp, AM, FM)
- COO stamp templates and collection for source stamping
- DC value extraction

All other devices (resistors, capacitors, diodes, etc.) are handled via OpenVAF.
"""

import ast
import logging
from typing import Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp

from vajax import get_float_dtype

logger = logging.getLogger(__name__)


# Source function creators - each returns a JIT-compatible function


def create_dc_source(dc_val: float) -> Callable:
    """Create a DC source function."""
    return lambda t, v=dc_val: v


def create_pulse_source(
    val0: float,
    val1: float,
    rise: float,
    fall: float,
    width: float,
    period: float,
    delay: float,
) -> Callable:
    """Create a pulse source function.

    Uses jnp.where for GPU-compatible conditionals.
    """

    def pulse_fn(t, v0=val0, v1=val1, r=rise, f=fall, w=width, p=period, d=delay):
        t_in_period = (t - d) % p
        rising = v0 + (v1 - v0) * t_in_period / r
        falling = v1 - (v1 - v0) * (t_in_period - r - w) / f
        return jnp.where(
            t < d,
            v0,
            jnp.where(
                t_in_period < r,
                rising,
                jnp.where(
                    t_in_period < r + w,
                    v1,
                    jnp.where(t_in_period < r + w + f, falling, v0),
                ),
            ),
        )

    return pulse_fn


def create_sine_source(
    sinedc: float,
    ampl: float,
    freq: float,
    phase: float,
) -> Callable:
    """Create a sine source function."""

    def sine_fn(t, dc=sinedc, a=ampl, f=freq, ph=phase):
        return dc + a * jnp.sin(2 * jnp.pi * f * t + ph)

    return sine_fn


def create_pwl_source(
    wave: List[float],
    offset: float,
    scale: float,
    stretch: float,
    pwlperiod: float,
) -> Callable:
    """Create a piecewise linear (PWL) source function.

    Args:
        wave: List of [t1, v1, t2, v2, ...] time-value pairs
        offset: Added to all values
        scale: Multiplied with all values
        stretch: Multiplied with all times
        pwlperiod: If > 0, makes the source periodic
    """
    wave_arr = jnp.array(wave, dtype=get_float_dtype())
    times = wave_arr[0::2] * stretch
    values = wave_arr[1::2] * scale + offset

    def pwl_fn(t, times=times, values=values, period=pwlperiod):
        if period > 0:
            t = t % period
        return jnp.interp(t, times, values)

    return pwl_fn


def create_exp_source(
    val0: float,
    val1: float,
    delay: float,
    td2: float,
    tau1: float,
    tau2: float,
) -> Callable:
    """Create an exponential source function (rise then fall)."""

    def exp_fn(t, v0=val0, v1=val1, d=delay, td=td2, t1=tau1, t2=tau2):
        t_fall = d + td
        rising = v0 + (v1 - v0) * (1 - jnp.exp(-(t - d) / t1))
        rising_at_fall = v0 + (v1 - v0) * (1 - jnp.exp(-td / t1))
        falling = rising_at_fall + (v0 - rising_at_fall) * (1 - jnp.exp(-(t - t_fall) / t2))
        return jnp.where(t < d, v0, jnp.where(t < t_fall, rising, falling))

    return exp_fn


def create_am_source(
    sinedc: float,
    ampl: float,
    freq: float,
    phase: float,
    delay: float,
    modfreq: float,
    modphase: float,
    modindex: float,
) -> Callable:
    """Create an amplitude modulation (AM) source function."""

    def am_fn(
        t,
        dc=sinedc,
        a=ampl,
        f=freq,
        ph=phase,
        d=delay,
        mf=modfreq,
        mph=modphase,
        mi=modindex,
    ):
        initial = dc + a * jnp.sin(ph * jnp.pi / 180) * (1 + mi * jnp.sin(mph * jnp.pi / 180))
        carrier = jnp.sin(2 * jnp.pi * f * (t - d) + ph * jnp.pi / 180)
        modulation = 1 + mi * jnp.sin(2 * jnp.pi * mf * (t - d) + mph * jnp.pi / 180)
        return jnp.where(t < d, initial, dc + a * carrier * modulation)

    return am_fn


def create_fm_source(
    sinedc: float,
    ampl: float,
    freq: float,
    phase: float,
    delay: float,
    modfreq: float,
    modphase: float,
    modindex: float,
) -> Callable:
    """Create a frequency modulation (FM) source function."""

    def fm_fn(
        t,
        dc=sinedc,
        a=ampl,
        f=freq,
        ph=phase,
        d=delay,
        mf=modfreq,
        mph=modphase,
        mi=modindex,
    ):
        initial = dc + a * jnp.sin(ph * jnp.pi / 180 + mi * jnp.sin(mph * jnp.pi / 180))
        carrier_phase = 2 * jnp.pi * f * (t - d) + ph * jnp.pi / 180
        mod_contribution = mi * jnp.sin(2 * jnp.pi * mf * (t - d) + mph * jnp.pi / 180)
        return jnp.where(t < d, initial, dc + a * jnp.sin(carrier_phase + mod_contribution))

    return fm_fn


def build_source_fn(
    devices: List[Dict],
    parse_number: Callable[[Any], float],
) -> Callable[[float], Dict[str, float]]:
    """Build time-varying source function from device parameters.

    Args:
        devices: List of device dictionaries with 'model', 'name', 'params'
        parse_number: Function to parse SPICE numbers (e.g., '1u' -> 1e-6)

    Returns:
        Function that takes time t and returns dict of {source_name: value}
    """
    sources = {}

    for dev in devices:
        if dev["model"] not in ("vsource", "isource"):
            continue

        params = dev["params"]
        source_type = str(params.get("type", "dc")).lower()
        name = dev["name"]

        logger.debug(f"  Source {name}: type={source_type}")

        if source_type in ("dc", "0", "0.0", ""):
            dc_val = params.get("dc", 0)
            sources[name] = create_dc_source(dc_val)

        elif source_type == "pulse":
            sources[name] = create_pulse_source(
                val0=parse_number(params.get("val0", 0)),
                val1=parse_number(params.get("val1", 1)),
                rise=parse_number(params.get("rise", 1e-9)),
                fall=parse_number(params.get("fall", 1e-9)),
                width=parse_number(params.get("width", 1e-6)),
                period=parse_number(params.get("period", 2e-6)),
                delay=parse_number(params.get("delay", 0)),
            )

        elif source_type == "sine":
            sources[name] = create_sine_source(
                sinedc=parse_number(params.get("sinedc", 0)),
                ampl=parse_number(params.get("ampl", 1)),
                freq=parse_number(params.get("freq", 1e3)),
                phase=parse_number(params.get("phase", 0)),
            )

        elif source_type == "pwl":
            wave = params.get("wave", [0, 0])
            if isinstance(wave, str):
                try:
                    wave = ast.literal_eval(wave)
                except (ValueError, SyntaxError) as e:
                    raise ValueError(f"Invalid PWL wave format '{wave}': {e}") from e

            sources[name] = create_pwl_source(
                wave=wave,
                offset=parse_number(params.get("offset", 0)),
                scale=parse_number(params.get("scale", 1)),
                stretch=parse_number(params.get("stretch", 1)),
                pwlperiod=parse_number(params.get("pwlperiod", 0)),
            )

        elif source_type == "exp":
            sources[name] = create_exp_source(
                val0=parse_number(params.get("val0", 0)),
                val1=parse_number(params.get("val1", 1)),
                delay=parse_number(params.get("delay", 0)),
                td2=parse_number(params.get("td2", 1e-9)),
                tau1=parse_number(params.get("tau1", 1e-10)),
                tau2=parse_number(params.get("tau2", 1e-10)),
            )

        elif source_type == "am":
            sources[name] = create_am_source(
                sinedc=parse_number(params.get("sinedc", 0)),
                ampl=parse_number(params.get("ampl", 1)),
                freq=parse_number(params.get("freq", 1e3)),
                phase=parse_number(params.get("phase", params.get("sinephase", 0))),
                delay=parse_number(params.get("delay", 0)),
                modfreq=parse_number(params.get("modfreq", 1e3)),
                modphase=parse_number(params.get("modphase", 0)),
                modindex=parse_number(params.get("modindex", 0.5)),
            )

        elif source_type == "fm":
            sources[name] = create_fm_source(
                sinedc=parse_number(params.get("sinedc", 0)),
                ampl=parse_number(params.get("ampl", 1)),
                freq=parse_number(params.get("freq", 1e3)),
                phase=parse_number(params.get("phase", params.get("sinephase", 0))),
                delay=parse_number(params.get("delay", 0)),
                modfreq=parse_number(params.get("modfreq", 1e3)),
                modphase=parse_number(params.get("modphase", 0)),
                modindex=parse_number(params.get("modindex", 0.5)),
            )

        else:
            raise ValueError(f"Unknown source type '{source_type}' for device {name}")

    def source_fn(t):
        return {name: fn(t) for name, fn in sources.items()}

    return source_fn


def get_source_fn_for_device(
    dev: Dict,
    parse_number: Callable[[Any], float],
) -> Callable | None:
    """Get the source function for a single device, or None if not a source.

    Args:
        dev: Device dictionary with 'model', 'params'
        parse_number: Function to parse SPICE numbers

    Returns:
        Source function or None
    """
    if dev["model"] not in ("vsource", "isource"):
        return None

    params = dev["params"]
    source_type = str(params.get("type", "dc")).lower()

    if source_type in ("dc", "0", "0.0", ""):
        return create_dc_source(params.get("dc", 0))

    elif source_type == "pulse":
        return create_pulse_source(
            val0=parse_number(params.get("val0", 0)),
            val1=parse_number(params.get("val1", 1)),
            rise=parse_number(params.get("rise", 1e-9)),
            fall=parse_number(params.get("fall", 1e-9)),
            width=parse_number(params.get("width", 1e-6)),
            period=parse_number(params.get("period", 2e-6)),
            delay=parse_number(params.get("delay", 0)),
        )

    elif source_type == "sine":
        return create_sine_source(
            sinedc=parse_number(params.get("sinedc", 0)),
            ampl=parse_number(params.get("ampl", 1)),
            freq=parse_number(params.get("freq", 1e6)),  # Note: default differs here
            phase=parse_number(params.get("phase", 0)),
        )

    return None


def prepare_source_devices_coo(
    source_devices: List[Dict],
    ground: int,
    n_unknowns: int,
) -> Dict[str, Any]:
    """Pre-compute data structures and stamp templates for source devices.

    All other devices (resistor, capacitor, diode) are handled via OpenVAF.
    This function only handles vsource and isource.

    Pre-computes static index arrays so runtime collection is fully vectorized
    with no Python loops.

    For 2-terminal devices (p, n), the stamp pattern is:
    - Residual: f[p] += I, f[n] -= I (2 entries, masked by ground)
    - Jacobian: J[p,p] += G, J[p,n] -= G, J[n,p] -= G, J[n,n] += G (4 entries)

    Args:
        source_devices: List of source device dicts
        ground: Ground node index
        n_unknowns: Number of unknowns in system

    Returns:
        Dict with device data and pre-computed stamp templates
    """
    logger.debug("Preparing source devices COO")

    # Group by model type
    by_type: Dict[str, List[Dict]] = {}
    for dev in source_devices:
        model = dev["model"]
        if model in ("vsource", "isource"):
            if model not in by_type:
                by_type[model] = []
            by_type[model].append(dev)

    result = {}
    for model, devs in by_type.items():
        logger.debug(f"COO for {model}")
        n = len(devs)

        # Extract node indices as JAX arrays
        node_p = jnp.array([d["nodes"][0] for d in devs], dtype=jnp.int32)
        node_n = jnp.array([d["nodes"][1] for d in devs], dtype=jnp.int32)
        names = [d["name"] for d in devs]

        # Pre-compute stamp templates for 2-terminal devices
        # Residual indices: [p-1, n-1] for each device, -1 if grounded
        logger.debug("Pre-computing stamp templates")
        f_idx_p = jnp.where(node_p != ground, node_p - 1, -1)
        f_idx_n = jnp.where(node_n != ground, node_n - 1, -1)
        f_indices = jnp.stack([f_idx_p, f_idx_n], axis=1)
        f_signs = jnp.array([1.0, -1.0])

        # Jacobian indices for 4-entry stamp pattern
        mask_p = node_p != ground
        mask_n = node_n != ground
        mask_both = mask_p & mask_n

        j_row_pp = jnp.where(mask_p, node_p - 1, -1)
        j_row_pn = jnp.where(mask_both, node_p - 1, -1)
        j_row_nn = jnp.where(mask_n, node_n - 1, -1)
        j_row_np = jnp.where(mask_both, node_n - 1, -1)

        j_col_pp = jnp.where(mask_p, node_p - 1, -1)
        j_col_pn = jnp.where(mask_both, node_n - 1, -1)
        j_col_nn = jnp.where(mask_n, node_n - 1, -1)
        j_col_np = jnp.where(mask_both, node_p - 1, -1)

        j_rows = jnp.stack([j_row_pp, j_row_pn, j_row_nn, j_row_np], axis=1)
        j_cols = jnp.stack([j_col_pp, j_col_pn, j_col_nn, j_col_np], axis=1)
        j_signs = jnp.array([1.0, -1.0, 1.0, -1.0])

        base_data = {
            "node_p": node_p,
            "node_n": node_n,
            "n": n,
            "f_indices": f_indices,
            "f_signs": f_signs,
            "j_rows": j_rows,
            "j_cols": j_cols,
            "j_signs": j_signs,
        }

        if model == "vsource":
            dc = jnp.array([d["params"].get("dc", 0.0) for d in devs], dtype=get_float_dtype())
            result["vsource"] = {**base_data, "dc": dc, "names": names}
        elif model == "isource":
            dc = jnp.array([d["params"].get("dc", 0.0) for d in devs], dtype=get_float_dtype())
            result["isource"] = {
                "node_p": node_p,
                "node_n": node_n,
                "n": n,
                "names": names,
                "dc": dc,
                "f_indices": f_indices,
                "f_signs": f_signs,
                # No Jacobian for current sources
                "j_rows": jnp.zeros((n, 0), dtype=jnp.int32),
                "j_cols": jnp.zeros((n, 0), dtype=jnp.int32),
                "j_signs": jnp.array([]),
            }
        logger.debug("Stamp templates complete")

    return result


def collect_source_devices_coo(
    device_data: Dict[str, Any],
    V: jax.Array,
    vsource_vals: jax.Array,
    isource_vals: jax.Array,
    f_indices: List,
    f_values: List,
    j_rows: List,
    j_cols: List,
    j_vals: List,
) -> None:
    """Collect COO triplets from source devices using fully vectorized operations.

    All other devices (resistor, capacitor, diode) are handled via OpenVAF.
    This function only handles vsource and isource.

    Uses pre-computed stamp templates from prepare_source_devices_coo.
    Fully vectorized - no Python loops, all JAX operations.

    Args:
        device_data: Pre-computed stamp templates
        V: Current voltage vector
        vsource_vals: JAX array of voltage source target values
        isource_vals: JAX array of current source values
        f_indices, f_values: Lists to append residual COO data
        j_rows, j_cols, j_vals: Lists to append Jacobian COO data
    """

    def _stamp_two_terminal(d: Dict, I: jax.Array, G: jax.Array):
        """Vectorized stamp for 2-terminal devices with current I and conductance G."""
        # Residual: shape (n, 2) -> flatten to (2*n,)
        f_vals = I[:, None] * d["f_signs"][None, :]
        f_idx = d["f_indices"].ravel()
        f_val = f_vals.ravel()

        # Jacobian: shape (n, 4) -> flatten to (4*n,)
        j_vals_arr = G[:, None] * d["j_signs"][None, :]
        j_row = d["j_rows"].ravel()
        j_col = d["j_cols"].ravel()
        j_val = j_vals_arr.ravel()

        # Filter valid entries (index >= 0)
        f_valid = f_idx >= 0
        j_valid = j_row >= 0

        f_indices.append(jnp.where(f_valid, f_idx, 0))
        f_values.append(jnp.where(f_valid, f_val, 0.0))
        j_rows.append(jnp.where(j_valid, j_row, 0))
        j_cols.append(jnp.where(j_valid, j_col, 0))
        j_vals.append(jnp.where(j_valid, j_val, 0.0))

    # Voltage sources: I = G * (Vp - Vn - Vtarget), G = 1e12
    if "vsource" in device_data and vsource_vals.size > 0:
        d = device_data["vsource"]
        G = 1e12
        Vp, Vn = V[d["node_p"]], V[d["node_n"]]
        I = G * (Vp - Vn - vsource_vals)
        G_arr = jnp.full(d["n"], G)
        _stamp_two_terminal(d, I, G_arr)

    # Current sources (residual only, no Jacobian)
    if "isource" in device_data and isource_vals.size > 0:
        d = device_data["isource"]
        f_vals = isource_vals[:, None] * jnp.array([1.0, -1.0])[None, :]
        f_idx = d["f_indices"].ravel()
        f_val = f_vals.ravel()
        f_valid = f_idx >= 0
        f_indices.append(jnp.where(f_valid, f_idx, 0))
        f_values.append(jnp.where(f_valid, f_val, 0.0))


def get_dc_source_values(
    devices: List[Dict],
    n_vsources: int,
    n_isources: int,
) -> Tuple[jax.Array, jax.Array]:
    """Extract DC values from voltage and current sources.

    Args:
        devices: List of device dictionaries
        n_vsources: Number of voltage sources
        n_isources: Number of current sources

    Returns:
        Tuple of (vsource_dc_vals, isource_dc_vals) as JAX arrays
    """
    vsource_dc_vals = jnp.zeros(n_vsources, dtype=get_float_dtype())
    isource_dc_vals = jnp.zeros(n_isources, dtype=get_float_dtype())

    vsource_idx = 0
    isource_idx = 0
    for dev in devices:
        if dev["model"] == "vsource":
            dc_val = dev["params"].get("dc", 0.0)
            vsource_dc_vals = vsource_dc_vals.at[vsource_idx].set(float(dc_val))
            vsource_idx += 1
        elif dev["model"] == "isource":
            source_type = str(dev["params"].get("type", "dc")).lower()
            dc_val = dev["params"].get("val0" if source_type == "pulse" else "dc", 0.0)
            isource_dc_vals = isource_dc_vals.at[isource_idx].set(float(dc_val))
            isource_idx += 1

    return vsource_dc_vals, isource_dc_vals


def get_vdd_value(devices: List[Dict]) -> float:
    """Find the maximum DC voltage from voltage sources (VDD).

    Args:
        devices: List of device dictionaries

    Returns:
        Maximum DC voltage value
    """
    vdd_value = 0.0
    for dev in devices:
        if dev["model"] == "vsource":
            dc_val = dev["params"].get("dc", 0.0)
            if dc_val > vdd_value:
                vdd_value = dc_val
    return vdd_value
