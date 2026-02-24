"""Integration methods for transient analysis.

Supports multiple numerical integration methods for computing charge derivatives:
- Backward Euler (be): First-order implicit, unconditionally stable
- Trapezoidal (trap): Second-order A-stable, good for oscillatory circuits
- Gear2/BDF2: Second-order L-stable, good for stiff problems

The integration formula computes dQ/dt from charge history:
    dQ/dt = c0 * Q + sum(c[i] * Q_history[i]) + sum(d[i] * dQdt_history[i])

Where:
    c0: Leading coefficient for current charge
    c[i]: Coefficients for past charges
    d[i]: Coefficients for past derivatives (only for trap)
"""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

from jax import Array


class IntegrationMethod(Enum):
    """Supported integration methods for transient analysis."""

    BACKWARD_EULER = "be"
    TRAPEZOIDAL = "trap"
    GEAR2 = "gear2"
    BDF2 = "bdf2"  # Alias for gear2

    @classmethod
    def from_string(cls, s: str) -> "IntegrationMethod":
        """Parse integration method from string.

        Handles various aliases used in SPICE simulators.
        """
        s_lower = s.lower().strip().strip("\"'")
        aliases = {
            "be": cls.BACKWARD_EULER,
            "euler": cls.BACKWARD_EULER,
            "backward_euler": cls.BACKWARD_EULER,
            "trap": cls.TRAPEZOIDAL,
            "trapezoidal": cls.TRAPEZOIDAL,
            "am2": cls.TRAPEZOIDAL,  # Adams-Moulton order 2
            "gear2": cls.GEAR2,
            "gear": cls.GEAR2,
            "bdf2": cls.GEAR2,
            "bdf": cls.GEAR2,
        }
        if s_lower in aliases:
            return aliases[s_lower]
        raise ValueError(f"Unknown integration method: {s}. Supported: be, trap, gear2")


class IntegrationCoeffs(NamedTuple):
    """Integration coefficients for a specific method and timestep.

    The integration formula is:
        dQ/dt = c0 * Q_new + c1 * Q_prev + c2 * Q_prev2 + d1 * dQdt_prev

    Where coefficients depend on the method:
        BE:    dQ/dt = (Q_new - Q_prev) / dt
               c0 = 1/dt, c1 = -1/dt, c2 = 0, d1 = 0
               error_coeff = 1/2

        Trap:  dQ/dt = 2/dt * (Q_new - Q_prev) - dQdt_prev
               c0 = 2/dt, c1 = -2/dt, c2 = 0, d1 = -1
               error_coeff = 1/12

        Gear2: dQ/dt = (3*Q_new - 4*Q_prev + Q_prev2) / (2*dt)
               c0 = 3/(2*dt), c1 = -4/(2*dt), c2 = 1/(2*dt), d1 = 0
               error_coeff = 2/9

    The error_coeff is used for Local Truncation Error (LTE) estimation
    in adaptive timestep control.
    """

    c0: float  # Coefficient for Q_new (leading coefficient)
    c1: float  # Coefficient for Q_prev
    c2: float  # Coefficient for Q_prev2 (only Gear2)
    d1: float  # Coefficient for dQdt_prev (only trap)
    history_depth: int  # Number of past Q values needed (1 for BE/trap, 2 for Gear2)
    needs_dqdt_history: bool  # Whether dQdt history is needed (only trap)
    error_coeff: float = 0.5  # LTE error coefficient (default BE value)


def compute_coefficients(method: IntegrationMethod, dt: float) -> IntegrationCoeffs:
    """Compute integration coefficients for a given method and timestep.

    Args:
        method: Integration method to use
        dt: Timestep size

    Returns:
        IntegrationCoeffs with all coefficients
    """
    inv_dt = 1.0 / dt

    if method == IntegrationMethod.BACKWARD_EULER:
        # dQ/dt = (Q_new - Q_prev) / dt
        # Error coefficient: 1/2 (first-order method)
        return IntegrationCoeffs(
            c0=inv_dt,
            c1=-inv_dt,
            c2=0.0,
            d1=0.0,
            history_depth=1,
            needs_dqdt_history=False,
            error_coeff=0.5,
        )
    elif method == IntegrationMethod.TRAPEZOIDAL:
        # dQ/dt = 2/dt * (Q_new - Q_prev) - dQdt_prev
        # Error coefficient: 1/12 (second-order method)
        return IntegrationCoeffs(
            c0=2.0 * inv_dt,
            c1=-2.0 * inv_dt,
            c2=0.0,
            d1=-1.0,
            history_depth=1,
            needs_dqdt_history=True,
            error_coeff=1.0 / 12.0,
        )
    elif method in (IntegrationMethod.GEAR2, IntegrationMethod.BDF2):
        # dQ/dt = (3*Q_new - 4*Q_prev + Q_prev2) / (2*dt)
        # Error coefficient: 2/9 (second-order method)
        return IntegrationCoeffs(
            c0=1.5 * inv_dt,
            c1=-2.0 * inv_dt,
            c2=0.5 * inv_dt,
            history_depth=2,
            d1=0.0,
            needs_dqdt_history=False,
            error_coeff=2.0 / 9.0,
        )
    else:
        raise ValueError(f"Unknown integration method: {method}")


def apply_integration(
    Q_new: Array,
    Q_prev: Array,
    coeffs: IntegrationCoeffs,
    Q_prev2: Array | None = None,
    dQdt_prev: Array | None = None,
) -> Array:
    """Apply integration formula to compute dQ/dt.

    Args:
        Q_new: Current charge vector (shape: n_unknowns)
        Q_prev: Previous charge vector
        coeffs: Integration coefficients
        Q_prev2: Second previous charge (for Gear2)
        dQdt_prev: Previous dQ/dt (for trap)

    Returns:
        dQ/dt vector
    """
    dQdt = coeffs.c0 * Q_new + coeffs.c1 * Q_prev

    if coeffs.history_depth >= 2 and Q_prev2 is not None:
        dQdt = dQdt + coeffs.c2 * Q_prev2

    if coeffs.needs_dqdt_history and dQdt_prev is not None:
        dQdt = dQdt + coeffs.d1 * dQdt_prev

    return dQdt


@dataclass
class IntegrationState:
    """State for multi-step integration methods.

    Tracks charge and derivative history for higher-order methods.
    """

    Q_prev: Array  # Q at t_{n-1}
    Q_prev2: Array | None = None  # Q at t_{n-2} (for Gear2)
    dQdt_prev: Array | None = None  # dQ/dt at t_{n-1} (for trap)

    def update(
        self,
        Q_new: Array,
        dQdt_new: Array,
        method: IntegrationMethod,
    ) -> "IntegrationState":
        """Update state after a successful timestep.

        Args:
            Q_new: New charge vector (becomes Q_prev)
            dQdt_new: New dQ/dt vector (becomes dQdt_prev for trap)
            method: Integration method being used

        Returns:
            Updated IntegrationState
        """
        if method in (IntegrationMethod.GEAR2, IntegrationMethod.BDF2):
            return IntegrationState(
                Q_prev=Q_new,
                Q_prev2=self.Q_prev,
                dQdt_prev=None,
            )
        elif method == IntegrationMethod.TRAPEZOIDAL:
            return IntegrationState(
                Q_prev=Q_new,
                Q_prev2=None,
                dQdt_prev=dQdt_new,
            )
        else:
            # Backward Euler - just track Q_prev
            return IntegrationState(
                Q_prev=Q_new,
                Q_prev2=None,
                dQdt_prev=None,
            )


def get_method_from_options(options_params: dict) -> IntegrationMethod:
    """Extract integration method from parsed options directive.

    Args:
        options_params: Dictionary from OptionsDirective.params

    Returns:
        IntegrationMethod, defaults to TRAPEZOIDAL if not specified (VACASK default)
    """
    tran_method = options_params.get("tran_method", "trap")
    # Strip quotes if present
    if isinstance(tran_method, str):
        tran_method = tran_method.strip("\"'")
    return IntegrationMethod.from_string(tran_method)
