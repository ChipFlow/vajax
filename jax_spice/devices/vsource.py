"""Voltage source models for JAX-SPICE

Includes DC, pulse, and piecewise-linear (PWL) voltage sources.
Essential for transient analysis stimulus.
"""

from typing import Dict, Optional, Tuple, TYPE_CHECKING
import jax.numpy as jnp
from jax import Array

from jax_spice.devices.base import DeviceStamps

if TYPE_CHECKING:
    from jax_spice.analysis.context import AnalysisContext


class VoltageSource:
    """Independent voltage source with various waveform types

    Voltage sources are modeled as ideal (zero internal resistance).
    In MNA formulation, they add a branch current as an unknown.

    For simple stamping, we model as large conductance G_big = 1e12
    forcing V = Vs via: I = G_big * (V - Vs)

    Parameters:
        dc: DC voltage value (default: 0)
        ac: AC magnitude for AC analysis (default: 0)
        pulse: Pulse parameters tuple (v0, v1, td, tr, tf, pw, per)

    Terminals:
        p: Positive terminal
        n: Negative terminal
    """

    terminals: Tuple[str, str] = ('p', 'n')

    # Large conductance for voltage source stamping
    G_BIG = 1e12

    def __init__(
        self,
        dc: float = 0.0,
        ac: float = 0.0,
        pulse: Optional[Tuple[float, ...]] = None,
    ):
        """Initialize voltage source

        Args:
            dc: DC voltage level
            ac: AC analysis magnitude
            pulse: Pulse parameters (v0, v1, delay, rise, fall, width, period)
                   v0: Initial/low voltage
                   v1: Pulsed/high voltage
                   delay: Time delay before first pulse
                   rise: Rise time (0 to v1)
                   fall: Fall time (v1 to 0)
                   width: Pulse width at high level
                   period: Period for repetition (0 = single pulse)
        """
        self.dc = dc
        self.ac = ac
        self.pulse = pulse

    def get_voltage(self, time: Optional[float] = None) -> float:
        """Get voltage at specified time

        Args:
            time: Simulation time (None for DC analysis)

        Returns:
            Voltage value at the given time
        """
        if time is None or self.pulse is None:
            return self.dc

        return pulse_voltage(time, self.pulse)

    def evaluate(
        self,
        voltages: Dict[str, float],
        params: Optional[Dict[str, float]] = None,
        context: Optional["AnalysisContext"] = None,
    ) -> DeviceStamps:
        """Evaluate voltage source at given terminal voltages

        Args:
            voltages: Dictionary with 'p' and 'n' terminal voltages
            params: Optional parameter overrides (can override 'dc')
            context: Analysis context (provides current time for transient)

        Returns:
            DeviceStamps forcing the specified voltage
        """
        # Get parameters
        if params is None:
            params = {}

        # Get time from context
        time = None
        if context is not None:
            time = getattr(context, 'time', None)

        # Calculate target voltage
        if 'v' in params:
            V_target = params['v']
        elif time is not None and self.pulse is not None:
            V_target = pulse_voltage(time, self.pulse)
        else:
            V_target = params.get('dc', self.dc)

        # Get terminal voltages
        Vp = voltages.get('p', 0.0)
        Vn = voltages.get('n', 0.0)
        V_actual = Vp - Vn

        # Force voltage via large conductance
        # I = G_BIG * (V_actual - V_target)
        # This creates a current that drives V_actual toward V_target
        G = self.G_BIG
        I = G * (V_actual - V_target)

        return DeviceStamps(
            currents={'p': I, 'n': -I},
            conductances={
                ('p', 'p'): G, ('p', 'n'): -G,
                ('n', 'p'): -G, ('n', 'n'): G
            }
        )


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


def pulse_voltage_jax(t: Array, v0: Array, v1: Array, td: Array,
                      tr: Array, tf: Array, pw: Array, per: Array) -> Array:
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
    v = jnp.where(before_delay, v0,
            jnp.where(in_rise, v_rise,
                jnp.where(in_high, v1,
                    jnp.where(in_fall, v_fall, v0))))

    return v


class CurrentSource:
    """Independent current source

    Parameters:
        dc: DC current value (default: 0)
        ac: AC magnitude for AC analysis (default: 0)

    Terminals:
        p: Positive terminal (current flows into p)
        n: Negative terminal (current flows out of n)
    """

    terminals: Tuple[str, str] = ('p', 'n')

    def __init__(self, dc: float = 0.0, ac: float = 0.0):
        """Initialize current source

        Args:
            dc: DC current level
            ac: AC analysis magnitude
        """
        self.dc = dc
        self.ac = ac

    def evaluate(
        self,
        voltages: Dict[str, float],
        params: Optional[Dict[str, float]] = None,
        context: Optional["AnalysisContext"] = None,
    ) -> DeviceStamps:
        """Evaluate current source

        Current sources have no conductance contribution (ideal).

        Args:
            voltages: Dictionary with 'p' and 'n' terminal voltages
            params: Optional parameter overrides (can override 'dc')
            context: Analysis context

        Returns:
            DeviceStamps with specified current
        """
        if params is None:
            params = {}

        I = params.get('i', params.get('dc', self.dc))

        # Current source: current flows from p to n
        return DeviceStamps(
            currents={'p': -I, 'n': I},  # Convention: positive I flows into p
            conductances={
                ('p', 'p'): jnp.array(0.0), ('p', 'n'): jnp.array(0.0),
                ('n', 'p'): jnp.array(0.0), ('n', 'n'): jnp.array(0.0)
            }
        )
