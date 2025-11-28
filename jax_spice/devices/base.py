"""Base device interface for JAX-SPICE

All device models implement this interface, providing current and conductance
stamps for circuit simulation. Inspired by SAX's functional approach.
"""

from typing import NamedTuple, Protocol, Dict, Tuple, Optional, TYPE_CHECKING
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from jax_spice.analysis.context import AnalysisContext


class DeviceStamps(NamedTuple):
    """Device contribution to circuit equations

    Attributes:
        currents: Dictionary mapping terminal names to current contributions
        conductances: Dictionary mapping (terminal1, terminal2) pairs to conductance values
        charges: Optional dictionary for dynamic elements (capacitance/charge storage)
    """
    currents: Dict[str, Array]
    conductances: Dict[Tuple[str, str], Array]
    charges: Dict[str, Array] | None = None


class Device(Protocol):
    """Base protocol for all device models

    Device models are pure functions that:
    1. Take terminal voltages and parameters as input
    2. Return DeviceStamps (currents, conductances, optional charges)
    3. Are automatically differentiable via JAX
    4. Can be batched with jax.vmap()
    5. Can be JIT compiled with jax.jit()

    Example:
        ```python
        # Define a simple resistor
        def resistor(V1: float, V2: float, R: float) -> DeviceStamps:
            I = (V1 - V2) / R
            G = 1.0 / R
            return DeviceStamps(
                currents={'1': I, '2': -I},
                conductances={('1', '1'): G, ('1', '2'): -G,
                              ('2', '1'): -G, ('2', '2'): G}
            )

        # Use with JAX
        import jax
        V1, V2, R = 1.0, 0.0, 1000.0
        stamps = resistor(V1, V2, R)
        print(f"Current: {stamps.currents['1']}")  # 1 mA

        # Automatic differentiation
        dI_dV1 = jax.grad(lambda v: resistor(v, V2, R).currents['1'])(V1)
        print(f"Conductance: {dI_dV1}")  # 1/R = 1 mS
        ```
    """

    terminals: Tuple[str, ...]
    """Terminal names for this device (e.g., ('d', 'g', 's', 'b') for MOSFET)"""

    def evaluate(
        self,
        voltages: Dict[str, float],
        params: Optional[Dict[str, float]] = None,
        context: Optional["AnalysisContext"] = None,
    ) -> DeviceStamps:
        """Evaluate device at given terminal voltages

        Args:
            voltages: Dictionary mapping terminal names to voltages
            params: Dictionary of device parameters (optional, device may have defaults)
            context: Analysis context for modifying behavior based on analysis type
                     (e.g., skip ddt() terms during DC analysis)

        Returns:
            DeviceStamps containing currents and conductances

        Note:
            This function must be pure (no side effects) and JAX-compatible.
            Conductances are typically computed via jax.grad() automatically.

            Devices with dynamic terms (ddt()) should check context.is_dc to
            determine whether to skip those terms during DC analysis.
        """
        ...


class LinearDevice:
    """Helper class for linear devices (resistors, capacitors, etc.)

    Linear devices have constant conductances independent of voltage.
    """

    @staticmethod
    def two_terminal(value: float, V1: float, V2: float, terminal_names: Tuple[str, str] = ('1', '2')) -> DeviceStamps:
        """Generic two-terminal linear element

        Args:
            value: Conductance (1/R for resistor, C for capacitor, etc.)
            V1, V2: Terminal voltages
            terminal_names: Names for the two terminals

        Returns:
            DeviceStamps with linear I-V relationship
        """
        G = value
        I = G * (V1 - V2)

        t1, t2 = terminal_names
        return DeviceStamps(
            currents={t1: I, t2: -I},
            conductances={
                (t1, t1): G, (t1, t2): -G,
                (t2, t1): -G, (t2, t2): G
            }
        )
