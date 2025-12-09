"""Simplified MOSFET model with automatic differentiation

This is a BSIM-like model with the essential physics for analog circuit simulation:
- Threshold voltage with body effect
- Mobility degradation
- Velocity saturation
- Channel length modulation
- Subthreshold conduction

All derivatives (gm, gds, gmb) are computed automatically via JAX!
"""

from typing import Dict, NamedTuple, Optional, Tuple, TYPE_CHECKING
import jax
import jax.numpy as jnp
from jax import Array

from jax_spice.devices.base import DeviceStamps

if TYPE_CHECKING:
    from jax_spice.analysis.context import AnalysisContext


# =============================================================================
# MOSFET Parameters
# =============================================================================


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


# =============================================================================
# Vectorized Batch Functions
# =============================================================================


def _mosfet_ids_batched(
    Vgs: Array,
    Vds: Array,
    Vbs: Array,
    W: Array,
    L: Array,
    Vth0: Array,
    gamma: Array,
    phiB: Array,
    u0: Array,
    theta: Array,
    vsat: Array,
    a0: Array,
    lambda_: Array,
    tox: Array,
    epsilon_ox: Array,
    n_sub: Array,
    Ioff: Array,
    temp: Array,
    pmos: Array,
) -> Array:
    """MOSFET drain current - pure JAX function for batched evaluation.

    All arguments are arrays of shape (n,) for n devices.
    This function handles PMOS via pmos array (1.0 for PMOS, 0.0 for NMOS).

    Returns:
        Ids: Drain current (n,)
    """
    # PMOS sign: flip voltages for PMOS (pmos=1), keep for NMOS (pmos=0)
    sign = 1.0 - 2.0 * pmos  # +1 for NMOS, -1 for PMOS
    Vgs_eff = sign * Vgs
    Vds_eff = sign * Vds
    Vbs_eff = sign * Vbs

    # Thermal voltage
    k_B = 1.381e-23
    q = 1.602e-19
    Vt = k_B * temp / q

    # Oxide capacitance
    Cox = epsilon_ox / tox

    # Threshold voltage with body effect
    sqrt_term = jnp.sqrt(jnp.maximum(2 * phiB - Vbs_eff, 0.1))
    sqrt_2phiB = jnp.sqrt(2 * phiB)
    Vth = Vth0 + gamma * (sqrt_term - sqrt_2phiB)

    # Gate overdrive
    Vgst = Vgs_eff - Vth

    # Subthreshold region (Vgst < 0)
    # The exponential model is only valid for weak inversion (Vgst < 0)
    # For Vgst >= 0, cap the subthreshold current at Ioff (the knee point)
    Ioff_scaled = Ioff * W / 1e-6
    Vgst_sub = jnp.minimum(Vgst, 0.0)  # Cap at 0 to prevent huge exponential
    I_sub = Ioff_scaled * jnp.exp(Vgst_sub / (n_sub * Vt))

    # Strong inversion - use effective overdrive that's zero in cutoff
    # This prevents negative Vgst from causing spurious positive current
    Vgst_pos = jnp.maximum(Vgst, 0.0)

    # Effective field for mobility degradation
    # Use only Vgst (not Vds) to avoid negative gds issues
    # This simplification ensures gds remains positive while preserving key physics
    Eeff = Vgst_pos / tox
    ueff = u0 / (1 + theta * jnp.maximum(Eeff, 0.0) * tox)

    beta_eff = ueff * Cox * W / L

    # Saturation voltage with velocity saturation
    Ecrit = vsat / ueff
    EcritL = Ecrit * L
    # Use Vgst_pos for Vdsat to avoid negative saturation voltage
    Vdsat = Vgst_pos / (1 + a0 * Vgst_pos / EcritL + 1e-9)  # Add small term to avoid division issues

    # Smooth transition linear/saturation
    Vdseff = Vdsat * jnp.tanh(Vds_eff / jnp.maximum(Vdsat, 0.01))

    # Drain current in strong inversion (using Vgst_pos)
    # Use |Vds_eff| for channel length modulation to ensure positive gds
    # This is more physically accurate: CLM increases current with |Vds| magnitude
    I_strong = beta_eff * Vgst_pos * Vdseff * (1 + lambda_ * jnp.abs(Vds_eff))

    # Smooth transition subthreshold/strong
    # Ensure I_strong is non-negative before log
    I_strong_safe = jnp.maximum(I_strong, 1e-30)
    I_total = jnp.exp(jnp.logaddexp(
        jnp.log(I_sub + 1e-30),
        jnp.log(I_strong_safe)
    ))

    # Ensure positive, then flip sign for PMOS
    Ids = jnp.maximum(I_total, 0.0) * sign

    return Ids


def mosfet_batch(
    V_batch: Array,
    params: Dict[str, Array],
) -> Tuple[Array, Array, Array, Array]:
    """Vectorized MOSFET evaluation for batch processing.

    Evaluates multiple MOSFETs in parallel using JAX operations.
    This is the GPU-friendly batch version for use in GPU-native stamping.

    Args:
        V_batch: Terminal voltages (n, 4) - [[V_d, V_g, V_s, V_b], ...] per device
        params: Dict with (n,) arrays for each parameter:
            - W: Width (m)
            - L: Length (m)
            - Vth0: Threshold voltage (V)
            - gamma: Body effect coefficient (V^0.5)
            - phiB: Surface potential (V)
            - u0: Low-field mobility (m^2/V/s)
            - theta: Mobility degradation (1/V)
            - vsat: Saturation velocity (m/s)
            - a0: Saturation parameter
            - lambda_: Channel length modulation (1/V)
            - tox: Oxide thickness (m)
            - epsilon_ox: Oxide permittivity (F/m)
            - n_sub: Subthreshold slope factor
            - Ioff: Off current (A/um)
            - temp: Temperature (K)
            - pmos: 1.0 for PMOS, 0.0 for NMOS

    Returns:
        Tuple of (n,) arrays:
            Ids: Drain current
            gm: Transconductance (dIds/dVgs)
            gds: Output conductance (dIds/dVds)
            gmb: Body transconductance (dIds/dVbs)

    Note:
        Stamps into MNA system (4-terminal device):
        - Residual: f[d] += Ids, f[s] -= Ids
        - Jacobian: gm, gds, gmb stamps (see MOSFETSimple.evaluate for pattern)
    """
    # Extract voltages (source-referenced)
    Vd = V_batch[:, 0]
    Vg = V_batch[:, 1]
    Vs = V_batch[:, 2]
    Vb = V_batch[:, 3]

    Vgs = Vg - Vs
    Vds = Vd - Vs
    Vbs = Vb - Vs

    # Extract parameters
    W = params['W']
    L = params['L']
    Vth0 = params['Vth0']
    gamma = params['gamma']
    phiB = params['phiB']
    u0 = params['u0']
    theta = params['theta']
    vsat = params['vsat']
    a0 = params['a0']
    lambda_ = params['lambda_']
    tox = params['tox']
    epsilon_ox = params['epsilon_ox']
    n_sub = params['n_sub']
    Ioff = params['Ioff']
    temp = params['temp']
    pmos = params['pmos']

    # Compute Ids
    Ids = _mosfet_ids_batched(
        Vgs, Vds, Vbs,
        W, L, Vth0, gamma, phiB, u0, theta, vsat, a0, lambda_,
        tox, epsilon_ox, n_sub, Ioff, temp, pmos
    )

    # Compute derivatives via JAX autodiff
    # gm = dIds/dVgs
    grad_fn = jax.grad(_mosfet_ids_batched, argnums=0)
    gm = jax.vmap(lambda vgs, vds, vbs, w, l, vth0, g, pb, u, th, vs, a, lam, tx, eox, ns, io, t, p:
        grad_fn(vgs, vds, vbs, w, l, vth0, g, pb, u, th, vs, a, lam, tx, eox, ns, io, t, p)
    )(Vgs, Vds, Vbs, W, L, Vth0, gamma, phiB, u0, theta, vsat, a0, lambda_,
      tox, epsilon_ox, n_sub, Ioff, temp, pmos)

    # gds = dIds/dVds
    grad_fn = jax.grad(_mosfet_ids_batched, argnums=1)
    gds = jax.vmap(lambda vgs, vds, vbs, w, l, vth0, g, pb, u, th, vs, a, lam, tx, eox, ns, io, t, p:
        grad_fn(vgs, vds, vbs, w, l, vth0, g, pb, u, th, vs, a, lam, tx, eox, ns, io, t, p)
    )(Vgs, Vds, Vbs, W, L, Vth0, gamma, phiB, u0, theta, vsat, a0, lambda_,
      tox, epsilon_ox, n_sub, Ioff, temp, pmos)

    # gmb = dIds/dVbs
    grad_fn = jax.grad(_mosfet_ids_batched, argnums=2)
    gmb = jax.vmap(lambda vgs, vds, vbs, w, l, vth0, g, pb, u, th, vs, a, lam, tx, eox, ns, io, t, p:
        grad_fn(vgs, vds, vbs, w, l, vth0, g, pb, u, th, vs, a, lam, tx, eox, ns, io, t, p)
    )(Vgs, Vds, Vbs, W, L, Vth0, gamma, phiB, u0, theta, vsat, a0, lambda_,
      tox, epsilon_ox, n_sub, Ioff, temp, pmos)

    return Ids, gm, gds, gmb
