"""AC (Small-Signal) Analysis for JAX-SPICE

Implements AC frequency sweep analysis following the VACASK algorithm:

1. Compute DC operating point: f(x0) = 0
2. Linearize at DC: Jr = resistive Jacobian, Jc = reactive Jacobian
3. For each frequency f:
   - omega = 2*pi*f
   - J_ac = Jr + j*omega*Jc (complex matrix)
   - U = AC source excitations (complex phasors)
   - Solve: J_ac * X = U for complex node voltages

The reactive Jacobian Jc contains capacitance entries (dq/dv terms) from
devices like capacitors and MOSFETs. These are already extracted by the
OpenVAF device evaluation (jacobian_react output).

Frequency sweep modes (matching VACASK):
- 'lin': Linear step from 'from' to 'to' with step size 'step'
- 'dec': Logarithmic, 'points' frequencies per decade
- 'oct': Logarithmic, 'points' frequencies per octave
- 'list': Explicit frequency values in 'values' array
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax_spice import get_float_dtype

logger = logging.getLogger(__name__)


@dataclass
class ACConfig:
    """AC analysis configuration.

    Attributes:
        freq_start: Starting frequency in Hz
        freq_stop: Ending frequency in Hz
        mode: Sweep mode - 'lin', 'dec', 'oct', or 'list'
        points: Number of points per decade/octave (for 'dec'/'oct' modes)
        step: Frequency step (for 'lin' mode)
        values: Explicit frequency list (for 'list' mode)
    """
    freq_start: float = 1.0
    freq_stop: float = 1e6
    mode: str = 'dec'  # 'lin', 'dec', 'oct', 'list'
    points: int = 10   # points per decade/octave
    step: Optional[float] = None  # for linear mode
    values: Optional[List[float]] = None  # for list mode


@dataclass
class ACResult:
    """AC analysis results.

    Attributes:
        frequencies: Array of frequencies (Hz), shape (n_freqs,)
        voltages: Dict mapping node name (str) to complex phasor array, shape (n_freqs,).
                  Node names come from the netlist (e.g., 'vdd', 'out', 'inp').
        currents: Dict mapping branch name to complex current array, shape (n_freqs,)
        dc_voltages: DC operating point voltages
    """
    frequencies: Array
    voltages: Dict[str, Array]
    currents: Dict[str, Array]
    dc_voltages: Array


def generate_frequencies(config: ACConfig) -> Array:
    """Generate frequency sweep array based on configuration.

    Args:
        config: AC analysis configuration

    Returns:
        JAX array of frequencies in Hz
    """
    if config.mode == 'list':
        if config.values is None:
            raise ValueError("'values' must be provided for 'list' mode")
        return jnp.array(config.values, dtype=get_float_dtype())

    elif config.mode == 'lin':
        if config.step is None:
            raise ValueError("'step' must be provided for 'lin' mode")
        n_points = int((config.freq_stop - config.freq_start) / config.step) + 1
        return jnp.linspace(config.freq_start, config.freq_stop, n_points)

    elif config.mode == 'dec':
        # points per decade
        n_decades = jnp.log10(config.freq_stop / config.freq_start)
        n_points = int(n_decades * config.points) + 1
        return jnp.logspace(
            jnp.log10(config.freq_start),
            jnp.log10(config.freq_stop),
            n_points
        )

    elif config.mode == 'oct':
        # points per octave (octave = factor of 2)
        n_octaves = jnp.log2(config.freq_stop / config.freq_start)
        n_points = int(n_octaves * config.points) + 1
        # logspace with base 2
        exponents = jnp.linspace(
            jnp.log2(config.freq_start),
            jnp.log2(config.freq_stop),
            n_points
        )
        return jnp.power(2.0, exponents)

    else:
        raise ValueError(f"Unknown sweep mode: {config.mode}")


def build_ac_excitation(
    ac_sources: List[Dict],
    node_indices: Dict[str, int],
    n_unknowns: int,
) -> Array:
    """Build AC excitation vector from voltage sources.

    In JAX-SPICE, voltage sources are modeled using high conductance G=1e12:
        I = G * (V_pos - V_neg - V_target)

    For AC analysis, the linearized equation is:
        Jr * delta_V = U

    where U = G * ac_phasor for voltage source nodes.

    Args:
        ac_sources: List of AC source specifications with keys:
            - 'name': Source name
            - 'pos_node': Positive terminal node index
            - 'neg_node': Negative terminal node index
            - 'mag': AC magnitude (default 1.0)
            - 'phase': AC phase in degrees (default 0.0)
        node_indices: Map of node names to indices
        n_unknowns: Total number of unknowns (nodes-1 + branches)

    Returns:
        Complex excitation vector U, shape (n_unknowns,)
    """
    U = jnp.zeros(n_unknowns, dtype=jnp.complex128)

    # High conductance used for voltage sources
    G = 1e12

    for source in ac_sources:
        mag = source.get('mag', 1.0)
        phase_deg = source.get('phase', 0.0)
        phase_rad = jnp.deg2rad(phase_deg)

        # Complex phasor: mag * e^(j*phase)
        phasor = mag * jnp.exp(1j * phase_rad)

        pos_node = source.get('pos_node', 0)
        neg_node = source.get('neg_node', 0)

        # For voltage source with high conductance model:
        # Residual: I = G * (V_pos - V_neg - V_target)
        # AC excitation: U = G * ac_phasor (applied to pos_node, -G to neg_node)
        if pos_node > 0:
            # pos_node - 1 because index 0 is ground (excluded from unknowns)
            U = U.at[pos_node - 1].set(U[pos_node - 1] + G * phasor)
        if neg_node > 0:
            U = U.at[neg_node - 1].set(U[neg_node - 1] - G * phasor)

    return U


def solve_ac_single_frequency(
    Jr: Array,
    Jc: Array,
    U: Array,
    omega,  # float or Array (scalar) - no type annotation to allow vmap tracer
) -> Array:
    """Solve AC system at a single frequency.

    Solves: (Jr + j*omega*Jc) * X = U

    Args:
        Jr: Resistive Jacobian (real), shape (n, n)
        Jc: Reactive Jacobian (capacitances, real), shape (n, n)
        U: Excitation vector (complex), shape (n,)
        omega: Angular frequency 2*pi*f (float or JAX scalar)

    Returns:
        Complex voltage phasors X, shape (n,)
    """
    # Build complex AC matrix
    J_ac = Jr.astype(jnp.complex128) + 1j * omega * Jc.astype(jnp.complex128)

    # Add regularization for numerical stability
    n = J_ac.shape[0]
    J_ac = J_ac + 1e-15 * jnp.eye(n, dtype=jnp.complex128)

    # Solve using LU decomposition (jax.scipy.linalg.solve)
    X = jax.scipy.linalg.solve(J_ac, U)

    return X


def solve_ac_sweep(
    Jr: Array,
    Jc: Array,
    U: Array,
    frequencies: Array,
) -> Array:
    """Solve AC system for all frequencies in sweep.

    Vectorized over frequencies using vmap for efficiency.

    Args:
        Jr: Resistive Jacobian (real), shape (n, n)
        Jc: Reactive Jacobian (capacitances, real), shape (n, n)
        U: Excitation vector (complex), shape (n,)
        frequencies: Frequency array in Hz, shape (n_freqs,)

    Returns:
        Complex voltage phasors, shape (n_freqs, n)
    """
    # Convert frequencies to angular frequencies
    omegas = 2.0 * jnp.pi * frequencies

    # Vectorize the single-frequency solver over omega
    def solve_at_omega(omega):
        return solve_ac_single_frequency(Jr, Jc, U, omega)

    # Use vmap for efficient vectorization
    X_all = jax.vmap(solve_at_omega)(omegas)

    return X_all


def run_ac_analysis(
    Jr: Array,
    Jc: Array,
    ac_sources: List[Dict],
    config: ACConfig,
    node_names: Optional[Dict[str, int]] = None,
    dc_voltages: Optional[Array] = None,
) -> ACResult:
    """Run complete AC analysis.

    Args:
        Jr: Resistive Jacobian at DC operating point, shape (n, n)
        Jc: Reactive (capacitance) Jacobian at DC OP, shape (n, n)
        ac_sources: List of AC source specifications
        config: AC analysis configuration
        node_names: Optional mapping of node names to indices
        dc_voltages: Optional DC operating point voltages

    Returns:
        ACResult with frequencies and complex phasors
    """
    n_unknowns = Jr.shape[0]

    # Generate frequency sweep
    frequencies = generate_frequencies(config)
    logger.info(f"AC analysis: {len(frequencies)} frequencies, "
                f"{config.freq_start:.2e} to {config.freq_stop:.2e} Hz")

    # Build excitation vector
    U = build_ac_excitation(ac_sources, node_names or {}, n_unknowns)

    # Solve for all frequencies
    X_all = solve_ac_sweep(Jr, Jc, U, frequencies)

    # Build result dictionaries - use string names only
    voltages: Dict[str, Array] = {}

    # Index by name from node_names mapping
    # Note: AC solution array excludes ground, so node idx maps to array position idx-1
    if node_names:
        for name, idx in node_names.items():
            if idx > 0 and idx <= n_unknowns:
                voltages[name] = X_all[:, idx - 1]

    return ACResult(
        frequencies=frequencies,
        voltages=voltages,
        currents={},  # Branch currents computed separately
        dc_voltages=dc_voltages if dc_voltages is not None else jnp.zeros(n_unknowns),
    )
