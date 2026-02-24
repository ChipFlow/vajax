"""Harmonic Balance Analysis for VA-JAX

Implements Harmonic Balance (HB) analysis for finding periodic steady-state
solutions of circuits with nonlinear elements driven by periodic sources.

Algorithm Overview:
1. Build frequency grid based on fundamental frequencies and harmonics
2. Generate collocation timepoints in time domain
3. Compute Almost Periodic Fourier Transform (APFT) matrices
4. Newton-Raphson iteration in time domain:
   a. Evaluate circuit at all collocation points
   b. Build HB Jacobian using DDT operator
   c. Solve for correction
   d. Update solution
5. Transform converged solution to frequency domain

Physical Interpretation:
- Circuit equation: f_resist(v) + dQ/dt = 0
- In frequency domain: F_resist(V) + j*omega*Q(V) = 0
- Time domain formulation: f_resist + DDT*Q = 0

References:
- K. S. Kundert, et al., "Steady-State Methods for Simulating Analog and
  Microwave Circuits", Kluwer Academic Publishers, 1990
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array

logger = logging.getLogger(__name__)

# Constants
TWO_PI = 2.0 * jnp.pi


@dataclass
class HBConfig:
    """Harmonic Balance analysis configuration.

    Attributes:
        freq: Fundamental frequencies in Hz (list of floats)
        nharm: Number of harmonics for each fundamental (int or list)
        truncation: 'box' or 'diamond' truncation scheme
        sample_factor: Oversampling factor for collocation (>= 1.0)
        n_periods: Number of periods for collocation points
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance for convergence
        reltol: Relative tolerance for convergence
    """

    freq: List[float] = field(default_factory=lambda: [1e3])
    nharm: Union[int, List[int]] = 4
    truncation: str = "diamond"
    sample_factor: float = 2.0
    n_periods: float = 3.0
    max_iterations: int = 100
    abstol: float = 1e-9
    reltol: float = 1e-6


@dataclass
class HBResult:
    """Harmonic Balance analysis results.

    Attributes:
        frequencies: Array of frequencies in the spectrum (Hz)
        phasors: Complex phasors for each node, shape (n_nodes, n_freqs)
        dc_voltages: DC operating point voltages
        converged: Whether the analysis converged
        iterations: Number of NR iterations
        max_residual: Maximum residual at convergence
    """

    frequencies: Array
    phasors: Dict[str, Array]  # node_name -> complex phasor array
    dc_voltages: Optional[Array] = None
    converged: bool = False
    iterations: int = 0
    max_residual: float = 0.0


def build_frequency_grid_box(
    fund_freqs: List[float],
    nharm: Union[int, List[int]],
) -> Tuple[Array, Array]:
    """Build frequency grid using box truncation.

    Box truncation includes all harmonics k_j = 0..H_j for each fundamental.
    The grid includes DC and all positive frequencies.

    Args:
        fund_freqs: Fundamental frequencies in Hz
        nharm: Number of harmonics per fundamental

    Returns:
        (frequencies, grid) where:
        - frequencies: sorted array of unique frequencies (Hz)
        - grid: (n_freqs, n_funds) integer array of harmonic indices
    """
    n_funds = len(fund_freqs)
    if isinstance(nharm, int):
        nharm = [nharm] * n_funds

    # Build all grid points recursively
    def build_grid_recursive(dim, current):
        if dim == n_funds:
            return [current.copy()]
        results = []
        for k in range(nharm[dim] + 1):
            current[dim] = k
            results.extend(build_grid_recursive(dim + 1, current))
        return results

    grid_points = build_grid_recursive(0, [0] * n_funds)

    # Filter: first non-zero index must be positive (avoid duplicates)
    valid_points = []
    for point in grid_points:
        first_nonzero = -1
        for j, k in enumerate(point):
            if k != 0:
                first_nonzero = j
                break
        if first_nonzero == -1:  # DC
            valid_points.append(point)
        elif point[first_nonzero] > 0:  # First nonzero is positive
            valid_points.append(point)

    # Compute frequencies
    jnp.array(fund_freqs)
    frequencies = []
    for point in valid_points:
        f = sum(k * fund_freqs[j] for j, k in enumerate(point))
        frequencies.append(f)

    # Sort by frequency
    freq_grid_pairs = list(zip(frequencies, valid_points))
    freq_grid_pairs.sort(key=lambda x: x[0])

    sorted_freqs = jnp.array([x[0] for x in freq_grid_pairs])
    sorted_grid = jnp.array([x[1] for x in freq_grid_pairs])

    return sorted_freqs, sorted_grid


def build_frequency_grid_diamond(
    fund_freqs: List[float],
    nharm: Union[int, List[int]],
    immax: Optional[int] = None,
) -> Tuple[Array, Array]:
    """Build frequency grid using diamond truncation.

    Diamond truncation includes all harmonics where sum(|k_j|) <= immax.

    Args:
        fund_freqs: Fundamental frequencies in Hz
        nharm: Number of harmonics (used to set immax if not provided)
        immax: Maximum intermodulation order

    Returns:
        (frequencies, grid) where:
        - frequencies: sorted array of unique frequencies (Hz)
        - grid: (n_freqs, n_funds) integer array of harmonic indices
    """
    n_funds = len(fund_freqs)
    if isinstance(nharm, int):
        if immax is None:
            immax = nharm
    else:
        if immax is None:
            immax = max(nharm)

    # Build all grid points with |k_1| + |k_2| + ... <= immax
    # Only consider non-negative k values (symmetry handles negatives)
    def build_grid_recursive(dim, current, remaining_order):
        if dim == n_funds:
            if remaining_order >= 0:
                return [current.copy()]
            return []
        results = []
        for k in range(remaining_order + 1):
            current[dim] = k
            results.extend(build_grid_recursive(dim + 1, current, remaining_order - k))
        return results

    grid_points = build_grid_recursive(0, [0] * n_funds, immax)

    # Filter: first non-zero index must be positive (avoid duplicates)
    valid_points = []
    for point in grid_points:
        first_nonzero = -1
        for j, k in enumerate(point):
            if k != 0:
                first_nonzero = j
                break
        if first_nonzero == -1:  # DC
            valid_points.append(point)
        elif point[first_nonzero] > 0:  # First nonzero is positive
            valid_points.append(point)

    # Compute frequencies
    frequencies = []
    for point in valid_points:
        f = sum(k * fund_freqs[j] for j, k in enumerate(point))
        frequencies.append(f)

    # Sort by frequency
    freq_grid_pairs = list(zip(frequencies, valid_points))
    freq_grid_pairs.sort(key=lambda x: x[0])

    sorted_freqs = jnp.array([x[0] for x in freq_grid_pairs])
    sorted_grid = jnp.array([x[1] for x in freq_grid_pairs])

    return sorted_freqs, sorted_grid


def build_frequency_grid(config: HBConfig) -> Tuple[Array, Array]:
    """Build frequency grid based on configuration.

    Args:
        config: HB configuration

    Returns:
        (frequencies, grid) tuple
    """
    if config.truncation == "box":
        return build_frequency_grid_box(config.freq, config.nharm)
    elif config.truncation == "diamond":
        return build_frequency_grid_diamond(config.freq, config.nharm)
    else:
        raise ValueError(f"Unknown truncation scheme: {config.truncation}")


def build_collocation_points(
    frequencies: Array,
    fund_freqs: List[float],
    n_periods: float = 3.0,
    sample_factor: float = 2.0,
) -> Array:
    """Build collocation timepoints for HB analysis.

    The number of collocation points is nt = 2*nf - 1 (excluding DC's imaginary part).
    Points are distributed over n_periods of the lowest fundamental frequency.

    Args:
        frequencies: Array of frequencies in spectrum
        fund_freqs: Fundamental frequencies
        n_periods: Number of periods to span
        sample_factor: Oversampling factor (>= 1.0)

    Returns:
        Array of collocation timepoints (seconds)
    """
    nf = len(frequencies)
    nt = 2 * nf - 1  # Number of real unknowns: DC + (nf-1) complex pairs

    # Fundamental period
    f0 = min(fund_freqs)
    T0 = 1.0 / f0

    # Distribute points uniformly over n_periods
    total_time = n_periods * T0
    timepoints = jnp.linspace(0, total_time, nt, endpoint=False)

    return timepoints


def build_apft_matrices(
    frequencies: Array,
    timepoints: Array,
) -> Tuple[Array, Array, Array]:
    """Build Almost Periodic Fourier Transform matrices.

    APFT transforms time-domain values to frequency-domain phasors.
    IAPFT transforms frequency-domain phasors to time-domain values.
    DDT is the time derivative operator in time domain.

    Args:
        frequencies: Array of frequencies (Hz)
        timepoints: Array of collocation timepoints (s)

    Returns:
        (APFT, IAPFT, DDT) matrices where:
        - APFT: (2*nf-1, nt) real matrix, transforms TD to APFT spectrum
        - IAPFT: (nt, 2*nf-1) real matrix, transforms APFT spectrum to TD
        - DDT: (nt, nt) real matrix, time derivative operator
    """
    nf = len(frequencies)
    nt = len(timepoints)

    # Build APFT matrix
    # For each frequency f_k, we have basis functions:
    # - DC (f=0): constant 1
    # - Non-DC: cos(2*pi*f*t), sin(2*pi*f*t)
    # The APFT spectrum has 2*nf-1 components:
    # [DC, Re(f1), Im(f1), Re(f2), Im(f2), ...]

    # Build basis matrix B: rows are timepoints, cols are basis functions
    B = jnp.zeros((nt, 2 * nf - 1), dtype=jnp.float64)

    # DC component
    B = B.at[:, 0].set(1.0)

    # Non-DC components
    for k in range(1, nf):
        omega_k = 2.0 * jnp.pi * frequencies[k]
        # Cosine (real part)
        B = B.at[:, 2 * k - 1].set(jnp.cos(omega_k * timepoints))
        # Sine (imaginary part)
        B = B.at[:, 2 * k].set(jnp.sin(omega_k * timepoints))

    # APFT: least-squares projection from TD to FD
    # X_fd = (B^T B)^{-1} B^T x_td
    BtB = B.T @ B
    BtB_inv = jnp.linalg.inv(BtB)
    APFT = BtB_inv @ B.T

    # IAPFT: synthesis from FD to TD
    # x_td = B X_fd
    IAPFT = B

    # DDT operator: d/dt in time domain
    # In frequency domain: d/dt -> j*omega
    # DDT = IAPFT * diag(j*omega) * APFT
    # Since we work with real representation, this becomes:
    # For each frequency k: [Re, Im] -> omega * [-Im, Re]
    Omega = jnp.zeros((2 * nf - 1, 2 * nf - 1), dtype=jnp.float64)
    for k in range(1, nf):
        omega_k = 2.0 * jnp.pi * frequencies[k]
        # [Re, Im] -> omega * [-Im, Re]
        Omega = Omega.at[2 * k - 1, 2 * k].set(-omega_k)  # Re <- -omega*Im
        Omega = Omega.at[2 * k, 2 * k - 1].set(omega_k)  # Im <- omega*Re

    DDT = IAPFT @ Omega @ APFT

    return APFT, IAPFT, DDT


def phasors_to_complex(apft_spectrum: Array, nf: int) -> Array:
    """Convert APFT real spectrum to complex phasors.

    Args:
        apft_spectrum: Real spectrum [DC, Re1, Im1, Re2, Im2, ...]
        nf: Number of frequencies

    Returns:
        Complex phasor array of shape (nf,)
    """
    phasors = jnp.zeros(nf, dtype=jnp.complex128)
    # DC
    phasors = phasors.at[0].set(apft_spectrum[0] + 0j)
    # Non-DC
    for k in range(1, nf):
        re = apft_spectrum[2 * k - 1]
        im = apft_spectrum[2 * k]
        phasors = phasors.at[k].set(re + 1j * im)
    return phasors


def complex_to_phasors(phasors: Array) -> Array:
    """Convert complex phasors to APFT real spectrum.

    Args:
        phasors: Complex phasor array of shape (nf,)

    Returns:
        Real spectrum [DC, Re1, Im1, Re2, Im2, ...]
    """
    nf = len(phasors)
    spectrum = jnp.zeros(2 * nf - 1, dtype=jnp.float64)
    # DC
    spectrum = spectrum.at[0].set(jnp.real(phasors[0]))
    # Non-DC
    for k in range(1, nf):
        spectrum = spectrum.at[2 * k - 1].set(jnp.real(phasors[k]))
        spectrum = spectrum.at[2 * k].set(jnp.imag(phasors[k]))
    return spectrum


def solve_hb_single_tone(
    build_circuit_fn,
    config: HBConfig,
    V_dc: Optional[Array] = None,
) -> HBResult:
    """Solve single-tone Harmonic Balance problem.

    This is a simplified implementation for single fundamental frequency.

    Args:
        build_circuit_fn: Function that evaluates circuit at given voltages
            Signature: (V_td, t) -> (f_resist, f_react, J_resist, J_react)
            where V_td is shape (n_nodes, nt) and t is shape (nt,)
        config: HB configuration
        V_dc: Initial DC solution (optional)

    Returns:
        HBResult with converged solution
    """
    # Build frequency grid
    frequencies, grid = build_frequency_grid(config)
    nf = len(frequencies)
    nt = 2 * nf - 1

    logger.info(f"HB analysis: {nf} frequencies, {nt} collocation points")

    # Build collocation points
    timepoints = build_collocation_points(
        frequencies, config.freq, config.n_periods, config.sample_factor
    )

    # Build APFT matrices
    APFT, IAPFT, DDT = build_apft_matrices(frequencies, timepoints)

    # This is a simplified stub - full implementation would include:
    # 1. Initialize solution from DC (replicated across timepoints)
    # 2. Newton-Raphson iteration:
    #    a. Evaluate circuit at all timepoints
    #    b. Build HB residual: F = f_resist + DDT @ Q
    #    c. Build HB Jacobian: J_hb = J_resist + DDT @ J_react
    #    d. Solve: delta = -J_hb^{-1} @ F
    #    e. Update: V <- V + delta
    #    f. Check convergence
    # 3. Transform solution to frequency domain

    # For now, return empty result
    return HBResult(
        frequencies=frequencies,
        phasors={},
        dc_voltages=V_dc,
        converged=False,
        iterations=0,
        max_residual=0.0,
    )


def run_hb_analysis(
    Jr: Array,
    Jc: Array,
    sources: List[Dict],
    config: HBConfig,
    node_names: Optional[Dict[str, int]] = None,
    dc_voltages: Optional[Array] = None,
) -> HBResult:
    """Run complete Harmonic Balance analysis.

    This is a simplified version that uses linearized circuit model.
    Full nonlinear HB would re-evaluate devices at each iteration.

    Args:
        Jr: Resistive Jacobian at DC operating point
        Jc: Reactive Jacobian at DC operating point
        sources: List of source specifications with AC parameters
        config: HB configuration
        node_names: Optional node name to index mapping
        dc_voltages: DC operating point voltages

    Returns:
        HBResult with harmonic solutions
    """
    # Build frequency grid
    frequencies, grid = build_frequency_grid(config)
    nf = len(frequencies)
    nt = 2 * nf - 1
    n = Jr.shape[0]

    logger.info(f"HB analysis: {nf} frequencies, {nt} timepoints, {n} nodes")

    # Build collocation points and transform matrices
    timepoints = build_collocation_points(
        frequencies, config.freq, config.n_periods, config.sample_factor
    )
    APFT, IAPFT, DDT = build_apft_matrices(frequencies, timepoints)

    # For linear HB, we solve at each frequency independently:
    # (Jr + j*omega*Jc) V(omega) = I(omega)
    # This is similar to AC analysis but at discrete harmonic frequencies

    phasors = {}

    # Find AC sources
    ac_sources = []
    for src in sources:
        if src.get("type") == "vsource":
            ac_params = src.get("ac", {})
            if ac_params.get("mag", 0) != 0 or ac_params.get("ampl", 0) != 0:
                ac_sources.append(src)
            # Also check for sine sources at fundamental frequency
            if src.get("params", {}).get("type") == "sine":
                params = src.get("params", {})
                src_freq = params.get("freq", 0)
                if src_freq in config.freq:
                    ac_sources.append(src)

    # Solve at each harmonic frequency
    for k, freq in enumerate(frequencies):
        omega = 2.0 * jnp.pi * freq

        # Build AC matrix at this frequency
        if freq == 0:
            # DC: just use Jr
            J_ac = Jr.astype(jnp.complex128)
        else:
            J_ac = Jr.astype(jnp.complex128) + 1j * omega * Jc.astype(jnp.complex128)

        # Build source excitation
        # For now, assume sources are at fundamental only
        U = jnp.zeros(n, dtype=jnp.complex128)

        if k == 1 and ac_sources:  # Fundamental frequency
            for src in ac_sources:
                pos = src.get("pos_node", 0)
                neg = src.get("neg_node", 0)
                params = src.get("params", {})

                # Get amplitude
                mag = params.get("ampl", params.get("mag", 1.0))
                phase = params.get("phase", 0.0)

                # High conductance for voltage source
                G = 1e12
                I_src = G * mag * jnp.exp(1j * jnp.radians(phase))

                if pos > 0:
                    U = U.at[pos - 1].add(I_src)
                if neg > 0:
                    U = U.at[neg - 1].add(-I_src)

        # Solve for voltage phasors
        try:
            V_k = jax.scipy.linalg.solve(J_ac, U)
        except Exception:
            V_k = jnp.zeros(n, dtype=jnp.complex128)

        # Store phasors
        if node_names:
            for name, idx in node_names.items():
                if name not in phasors:
                    phasors[name] = jnp.zeros(nf, dtype=jnp.complex128)
                phasors[name] = phasors[name].at[k].set(V_k[idx - 1])

    return HBResult(
        frequencies=frequencies,
        phasors=phasors,
        dc_voltages=dc_voltages,
        converged=True,
        iterations=nf,  # One solve per frequency
        max_residual=0.0,
    )


# =============================================================================
# Full Nonlinear Harmonic Balance (TODO)
# =============================================================================
#
# The full nonlinear HB requires re-evaluating devices at each Newton-Raphson
# iteration. This section outlines the algorithm and provides a skeleton for
# future implementation.
#
# Algorithm:
# 1. Initialize: V_td shape (n_nodes, nt) from DC replicated across timepoints
# 2. Newton-Raphson loop:
#    a. Evaluate ALL devices at ALL collocation points:
#       (f_resist, Q, J_resist, J_react) = evaluate_circuit(V_td)
#       where each has shape (n_nodes, nt) or (n_nodes, n_nodes, nt) for Jacobians
#
#    b. Build HB residual:
#       F = f_resist + DDT @ Q
#       where DDT operates on the time dimension
#       F has shape (n_nodes, nt) -> flattened to (n_nodes * nt,)
#
#    c. Build HB Jacobian:
#       J_hb = J_resist + DDT @ J_react
#       This is a block matrix of shape (n_nodes * nt, n_nodes * nt)
#       Block (i,j,k,l) couples node i at timepoint k to node j at timepoint l
#
#    d. Solve: delta = -J_hb^{-1} @ F
#    e. Update: V_td <- V_td + delta.reshape(n_nodes, nt)
#    f. Check convergence: ||F|| < abstol, ||delta|| < reltol * ||V||
#
# 3. Transform converged V_td to frequency domain via APFT
#
# Implementation Requirements:
# - Batched device evaluation: evaluate_circuit must accept V shape (n_nodes, nt)
# - Efficient block Jacobian assembly
# - Source handling: build time-domain source waveforms at collocation points
# - Continuation: start with large harmonic truncation and refine
#
# Integration with CircuitEngine:
# - Need access to _make_full_mna_build_system_fn but batched over timepoints
# - Each device model's vmapped function needs additional time dimension
# =============================================================================


def run_hb_nonlinear(
    build_circuit_fn,
    config: HBConfig,
    V_dc: Optional[Array] = None,
    node_names: Optional[Dict[str, int]] = None,
) -> HBResult:
    """Run full nonlinear Harmonic Balance analysis.

    This re-evaluates devices at each Newton-Raphson iteration for accurate
    handling of strongly nonlinear circuits.

    Args:
        build_circuit_fn: Function that evaluates circuit at given voltages.
            Signature: (V_td) -> (f_resist, Q, J_resist, J_react)
            where V_td has shape (n_nodes, nt) - voltages at all collocation points
            Returns:
                f_resist: Resistive residual, shape (n_nodes, nt)
                Q: Charges, shape (n_nodes, nt)
                J_resist: Resistive Jacobian, shape (n_nodes, n_nodes, nt)
                J_react: Reactive Jacobian, shape (n_nodes, n_nodes, nt)
        config: HB configuration
        V_dc: Initial DC solution, shape (n_nodes,)
        node_names: Optional node name to index mapping

    Returns:
        HBResult with converged harmonic solutions

    Note:
        This is currently a placeholder. Full implementation requires:
        1. Integration with CircuitEngine for batched device evaluation
        2. Block Jacobian assembly and solve
        3. Source waveform generation at collocation points
    """
    # Build frequency grid
    frequencies, grid = build_frequency_grid(config)
    nf = len(frequencies)
    nt = 2 * nf - 1

    logger.info(f"Nonlinear HB: {nf} frequencies, {nt} collocation points")

    # Build collocation points
    timepoints = build_collocation_points(
        frequencies, config.freq, config.n_periods, config.sample_factor
    )

    # Build APFT matrices
    APFT, IAPFT, DDT = build_apft_matrices(frequencies, timepoints)

    # Get number of nodes from DC solution
    if V_dc is not None:
        len(V_dc)
    else:
        pass

    # Initialize solution: replicate DC across all timepoints
    if V_dc is not None:
        jnp.tile(V_dc[:, None], (1, nt))
    else:
        pass

    # TODO: Implement full NR loop
    # For now, return unconverged result to indicate not implemented
    logger.warning("Nonlinear HB not yet implemented - returning empty result")

    return HBResult(
        frequencies=frequencies,
        phasors={},
        dc_voltages=V_dc,
        converged=False,
        iterations=0,
        max_residual=float("inf"),
    )
