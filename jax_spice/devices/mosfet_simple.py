"""Simplified MOSFET model with automatic differentiation

This is a BSIM-like model with the essential physics for analog circuit simulation:
- Threshold voltage with body effect
- Mobility degradation
- Velocity saturation
- Channel length modulation
- Subthreshold conduction

All derivatives (gm, gds, gmb) are computed automatically via JAX!
"""

from typing import Dict, NamedTuple, Optional, TYPE_CHECKING
import jax
import jax.numpy as jnp
from jax import Array

from jax_spice.devices.base import DeviceStamps

if TYPE_CHECKING:
    from jax_spice.analysis.context import AnalysisContext


class MOSFETParams(NamedTuple):
    """MOSFET model parameters"""
    # Geometry
    W: float = 1e-6  # Width (m)
    L: float = 0.25e-6  # Length (m)

    # Threshold voltage
    Vth0: float = 0.4  # Threshold voltage at Vbs=0 (V)
    gamma: float = 0.5  # Body effect coefficient (V^0.5)
    phiB: float = 0.9  # Surface potential (V)

    # Mobility
    u0: float = 400e-4  # Low-field mobility (m^2/V/s) - 400 cm^2/V/s
    theta: float = 0.2  # Mobility degradation (1/V)

    # Saturation
    vsat: float = 1e5  # Saturation velocity (m/s)
    a0: float = 1.0  # Saturation parameter

    # Channel length modulation
    lambda_: float = 0.05  # Channel length modulation (1/V)

    # Process parameters
    tox: float = 5e-9  # Oxide thickness (m)
    epsilon_ox: float = 3.9 * 8.854e-12  # Oxide permittivity (F/m)

    # Subthreshold
    n_sub: float = 1.5  # Subthreshold slope factor
    Ioff: float = 1e-12  # Off current at Vgs=0 (A/um)

    # Temperature
    temp: float = 300.0  # Temperature (K)

    # Type
    pmos: bool = False  # True for PMOS, False for NMOS

    @property
    def Cox(self):
        """Oxide capacitance per unit area (F/m^2)"""
        return self.epsilon_ox / self.tox

    @property
    def Vt(self):
        """Thermal voltage (V)"""
        return 1.381e-23 * self.temp / 1.602e-19  # kT/q

    @property
    def beta(self):
        """Transconductance parameter before mobility degradation"""
        return self.u0 * self.Cox * self.W / self.L


def mosfet_ids(Vgs: float, Vds: float, Vbs: float, params: MOSFETParams) -> float:
    """MOSFET drain current

    Args:
        Vgs: Gate-source voltage (V)
        Vds: Drain-source voltage (V)
        Vbs: Bulk-source voltage (V)
        params: MOSFET parameters

    Returns:
        Drain current Ids (A)

    This function is pure and differentiable - JAX can automatically compute:
    - gm = dIds/dVgs (transconductance)
    - gds = dIds/dVds (output conductance)
    - gmb = dIds/dVbs (body effect transconductance)
    """
    # Handle PMOS by flipping voltages
    if params.pmos:
        Vgs, Vds, Vbs = -Vgs, -Vds, -Vbs

    # Threshold voltage with body effect
    sqrt_term = jnp.sqrt(jnp.maximum(2 * params.phiB - Vbs, 0.1))
    sqrt_2phiB = jnp.sqrt(2 * params.phiB)
    Vth = params.Vth0 + params.gamma * (sqrt_term - sqrt_2phiB)

    # Gate overdrive
    Vgst = Vgs - Vth

    # Subthreshold region (Vgst < 0)
    # Exponential current below threshold
    Ioff_scaled = params.Ioff * params.W / 1e-6  # Scale by width in microns
    I_sub = Ioff_scaled * jnp.exp(Vgst / (params.n_sub * params.Vt))

    # Strong inversion (Vgst > 0)
    # Mobility degradation due to vertical field
    Eeff = (Vgst + Vds / 2) / params.tox
    ueff = params.u0 / (1 + params.theta * Eeff * params.tox)

    # Updated beta with mobility degradation
    beta_eff = ueff * params.Cox * params.W / params.L

    # Saturation voltage with velocity saturation
    # Classical: Vdsat = Vgst
    # With velocity saturation: Vdsat = Vgst / (1 + Vgst/EcritL)
    Ecrit = params.vsat / ueff  # Critical field
    EcritL = Ecrit * params.L
    Vdsat = Vgst / (1 + params.a0 * Vgst / EcritL)

    # Smooth transition between linear and saturation using tanh
    # This avoids discontinuity at Vds = Vdsat
    Vdseff = Vdsat * jnp.tanh(Vds / jnp.maximum(Vdsat, 0.01))

    # Drain current in strong inversion
    I_strong = beta_eff * Vgst * Vdseff * (1 + params.lambda_ * Vds)

    # Smooth transition between subthreshold and strong inversion
    # Use smooth maximum function
    transition_width = 4 * params.Vt
    I_total = jnp.logaddexp(
        jnp.log(I_sub + 1e-30),
        jnp.log(I_strong + 1e-30)
    )
    I_total = jnp.exp(I_total)

    # Ensure current is positive and in correct direction for PMOS
    Ids = jnp.maximum(I_total, 0.0)
    if params.pmos:
        Ids = -Ids

    return Ids


class MOSFETSimple:
    """Simplified MOSFET device model with automatic differentiation

    Usage:
        ```python
        # Create NMOS
        nmos = MOSFETSimple(W=10e-6, L=0.25e-6, Vth0=0.4)

        # Evaluate at operating point
        voltages = {'d': 1.2, 'g': 1.5, 's': 0.0, 'b': 0.0}
        stamps = nmos.evaluate(voltages)

        # Currents and conductances are computed automatically
        print(f"Ids = {stamps.currents['d']:.6f} A")
        print(f"gm = {stamps.conductances[('d', 'g')]:.6f} S")
        ```
    """

    terminals = ('d', 'g', 's', 'b')

    def __init__(self, **kwargs):
        """Initialize MOSFET with given parameters

        Args:
            **kwargs: Parameter overrides (W, L, Vth0, etc.)
        """
        # Create params with defaults, then override
        self.params = MOSFETParams(**kwargs)

    def evaluate(
        self,
        voltages: Dict[str, float],
        context: Optional["AnalysisContext"] = None,
    ) -> DeviceStamps:
        """Evaluate MOSFET at given terminal voltages

        Args:
            voltages: Dict with keys 'd', 'g', 's', 'b' for terminal voltages
            context: Analysis context (unused - MOSFET has no dynamic terms)

        Returns:
            DeviceStamps with currents and conductances
        """
        # Note: context is unused in this model since it has no ddt() terms.
        # It's included for protocol compliance and future extension.
        # Extract voltages (source is reference)
        Vd = voltages['d']
        Vg = voltages['g']
        Vs = voltages['s']
        Vb = voltages['b']

        # Convert to device voltages (source-referenced)
        Vgs = Vg - Vs
        Vds = Vd - Vs
        Vbs = Vb - Vs

        # Compute drain current
        Ids = mosfet_ids(Vgs, Vds, Vbs, self.params)

        # Automatic differentiation for small-signal parameters!
        # This is the magic of JAX - no manual derivative computation
        gm = jax.grad(mosfet_ids, argnums=0)(Vgs, Vds, Vbs, self.params)   # dIds/dVgs
        gds = jax.grad(mosfet_ids, argnums=1)(Vgs, Vds, Vbs, self.params)  # dIds/dVds
        gmb = jax.grad(mosfet_ids, argnums=2)(Vgs, Vds, Vbs, self.params)  # dIds/dVbs

        # Build current stamps (KCL: current out of each terminal)
        currents = {
            'd': Ids,      # Current out of drain
            'g': 0.0,      # No DC gate current
            's': -Ids,     # Current out of source (negative of drain)
            'b': 0.0,      # No bulk current in simple model
        }

        # Build conductance stamps (Jacobian matrix entries)
        # Format: (to_node, from_node) -> conductance
        # Convention: I[to] += G * V[from]
        conductances = {
            # Drain current dependencies
            ('d', 'g'): gm,      # Transconductance
            ('d', 'd'): gds,     # Output conductance
            ('d', 's'): -(gm + gds + gmb),  # Source coupling
            ('d', 'b'): gmb,     # Body effect

            # Source current dependencies (negative of drain)
            ('s', 'g'): -gm,
            ('s', 'd'): -gds,
            ('s', 's'): (gm + gds + gmb),
            ('s', 'b'): -gmb,

            # Gate and bulk have no current
            ('g', 'g'): 0.0,
            ('b', 'b'): 0.0,
        }

        return DeviceStamps(currents=currents, conductances=conductances)

    def __repr__(self):
        ptype = "PMOS" if self.params.pmos else "NMOS"
        return f"MOSFET{ptype}(W={self.params.W*1e6:.2f}um, L={self.params.L*1e6:.3f}um)"
