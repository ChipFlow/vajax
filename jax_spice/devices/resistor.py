"""Resistor device model for JAX-SPICE

Simple two-terminal linear resistor with optional temperature dependence.
"""

from typing import Dict, Optional, Tuple, TYPE_CHECKING
import jax.numpy as jnp
from jax import Array

from jax_spice.devices.base import DeviceStamps

if TYPE_CHECKING:
    from jax_spice.analysis.context import AnalysisContext


class Resistor:
    """Two-terminal resistor device

    I = V/R where V = V(+) - V(-)
    G = 1/R (conductance)

    Parameters:
        R: Resistance in Ohms (default: 1000)
        tc1: First-order temperature coefficient (default: 0)
        tc2: Second-order temperature coefficient (default: 0)
        tnom: Nominal temperature in Kelvin (default: 300.15)

    Terminals:
        p: Positive terminal
        n: Negative terminal
    """

    terminals: Tuple[str, str] = ('p', 'n')

    def __init__(self, R: float = 1000.0, tc1: float = 0.0, tc2: float = 0.0,
                 tnom: float = 300.15):
        """Initialize resistor with parameters

        Args:
            R: Resistance value in Ohms
            tc1: First-order temperature coefficient
            tc2: Second-order temperature coefficient
            tnom: Nominal temperature in Kelvin
        """
        self.R = R
        self.tc1 = tc1
        self.tc2 = tc2
        self.tnom = tnom

    def evaluate(
        self,
        voltages: Dict[str, float],
        params: Optional[Dict[str, float]] = None,
        context: Optional["AnalysisContext"] = None,
    ) -> DeviceStamps:
        """Evaluate resistor at given terminal voltages

        Args:
            voltages: Dictionary with 'p' and 'n' terminal voltages
            params: Optional parameter overrides
            context: Analysis context (temperature)

        Returns:
            DeviceStamps with current and conductance
        """
        # Get parameters (allow override)
        if params is None:
            params = {}
        R = params.get('R', self.R)
        tc1 = params.get('tc1', self.tc1)
        tc2 = params.get('tc2', self.tc2)
        tnom = params.get('tnom', self.tnom)

        # Get temperature
        T = context.temperature if context else 300.15

        # Temperature-adjusted resistance
        dT = T - tnom
        R_t = R * (1.0 + tc1 * dT + tc2 * dT * dT)

        # Ensure minimum resistance for numerical stability
        R_t = jnp.maximum(R_t, 1e-12)

        # Calculate current and conductance
        Vp = voltages.get('p', 0.0)
        Vn = voltages.get('n', 0.0)
        V = Vp - Vn

        G = 1.0 / R_t
        I = G * V

        return DeviceStamps(
            currents={'p': I, 'n': -I},
            conductances={
                ('p', 'p'): G, ('p', 'n'): -G,
                ('n', 'p'): -G, ('n', 'n'): G
            }
        )


def resistor(Vp: Array, Vn: Array, R: Array) -> Tuple[Array, Array]:
    """Functional resistor model

    Pure function for use in JAX computations.

    Args:
        Vp: Positive terminal voltage
        Vn: Negative terminal voltage
        R: Resistance in Ohms

    Returns:
        Tuple of (current, conductance) from p to n terminal
    """
    G = 1.0 / jnp.maximum(R, 1e-12)
    I = G * (Vp - Vn)
    return I, G
