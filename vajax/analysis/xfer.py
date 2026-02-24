"""Transfer Function Analyses for VA-JAX

Implements three related small-signal analyses:

1. DCINC (DC Incremental): Small-signal DC response to source excitations
   - Solves: Jr·dx = du where du comes from source 'mag' parameters
   - Returns incremental nodal voltages

2. DCXF (DC Transfer Function): DC small-signal transfer function metrics
   - For each source, computes: tf (transfer function), zin (input impedance), yin (input admittance)
   - Uses unity excitation per source

3. ACXF (AC Transfer Function): Frequency-dependent transfer function
   - Same as DCXF but over frequency sweep
   - Includes reactive effects: (Jr + j*omega*Jc)·X = U
   - Returns complex-valued tf, zin, yin
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import jax
import jax.numpy as jnp
from jax import Array

logger = logging.getLogger(__name__)


def extract_all_sources(devices: List[Dict]) -> List[Dict]:
    """Extract all independent sources for transfer function analysis.

    Args:
        devices: List of device dictionaries from circuit parsing

    Returns:
        List of source specifications with type, name, nodes, mag, dc
    """
    sources = []

    for dev in devices:
        if dev.get("model") in ("vsource", "isource"):
            params = dev.get("params", {})
            nodes = dev.get("nodes", [0, 0])

            sources.append(
                {
                    "name": dev["name"],
                    "type": dev["model"],
                    "pos_node": nodes[0] if len(nodes) > 0 else 0,
                    "neg_node": nodes[1] if len(nodes) > 1 else 0,
                    "mag": float(params.get("mag", 0.0)),
                    "dc": float(params.get("dc", 0.0)),
                }
            )

    return sources


# =============================================================================
# DCINC (DC Incremental) Analysis
# =============================================================================


@dataclass
class DCIncConfig:
    """DCINC analysis configuration.

    Attributes:
        writeop: If True, also return DC operating point data
    """

    writeop: bool = False


@dataclass
class DCIncResult:
    """DCINC analysis results.

    Attributes:
        incremental_voltages: Dict mapping node name (str) to incremental voltage.
                              Node names come from the netlist.
        dc_voltages: DC operating point voltages (if writeop=True)
    """

    incremental_voltages: Dict[str, float]
    dc_voltages: Optional[Array] = None


def solve_dcinc(
    Jr: Array,
    excitation: Array,
    node_names: Optional[Dict[str, int]] = None,
    dc_voltages: Optional[Array] = None,
) -> DCIncResult:
    """Solve DC incremental analysis.

    Solves: Jr·dx = du for incremental nodal voltages.

    Args:
        Jr: Resistive Jacobian at DC operating point, shape (n, n)
        excitation: Incremental excitation vector, shape (n,)
        node_names: Optional mapping of node names to indices
        dc_voltages: Optional DC operating point voltages

    Returns:
        DCIncResult with incremental voltages
    """
    n = Jr.shape[0]

    # Add regularization for numerical stability
    Jr_reg = Jr + 1e-15 * jnp.eye(n)

    # Solve Jr·dx = du
    dx = jax.scipy.linalg.solve(Jr_reg, excitation)

    # Build result dictionary - use string names only
    voltages: Dict[str, float] = {}

    # Index by name from node_names mapping
    # Note: Solution array excludes ground, so node idx maps to array position idx-1
    if node_names:
        for name, idx in node_names.items():
            if idx > 0 and idx <= n:
                voltages[name] = float(dx[idx - 1])

    return DCIncResult(
        incremental_voltages=voltages,
        dc_voltages=dc_voltages,
    )


def build_dcinc_excitation(
    sources: List[Dict],
    n_unknowns: int,
) -> Array:
    """Build DC incremental excitation vector from source 'mag' parameters.

    For voltage sources modeled with high conductance G:
        Excitation = G * mag (applied to positive node, -G*mag to negative)

    For current sources:
        Excitation = mag (applied directly as current injection)

    Args:
        sources: List of source specifications with keys:
            - 'type': 'vsource' or 'isource'
            - 'pos_node': Positive terminal node index
            - 'neg_node': Negative terminal node index
            - 'mag': Incremental excitation magnitude
        n_unknowns: Total number of unknowns

    Returns:
        Excitation vector, shape (n_unknowns,)
    """
    du = jnp.zeros(n_unknowns, dtype=jnp.float64)

    # High conductance used for voltage sources
    G = 1e12

    for source in sources:
        mag = source.get("mag", 0.0)
        if mag == 0.0:
            continue

        source_type = source.get("type", "vsource")
        pos_node = source.get("pos_node", 0)
        neg_node = source.get("neg_node", 0)

        if source_type == "vsource":
            # Voltage source: du = G * mag
            if pos_node > 0:
                du = du.at[pos_node - 1].set(du[pos_node - 1] + G * mag)
            if neg_node > 0:
                du = du.at[neg_node - 1].set(du[neg_node - 1] - G * mag)
        else:
            # Current source: du = mag (direct current injection)
            if pos_node > 0:
                du = du.at[pos_node - 1].set(du[pos_node - 1] + mag)
            if neg_node > 0:
                du = du.at[neg_node - 1].set(du[neg_node - 1] - mag)

    return du


# =============================================================================
# DCXF (DC Transfer Function) Analysis
# =============================================================================


@dataclass
class DCXFConfig:
    """DCXF analysis configuration.

    Attributes:
        out: Output node specification (node name or index)
        writeop: If True, also return DC operating point data
    """

    out: Union[str, int] = 1
    writeop: bool = False


@dataclass
class DCXFResult:
    """DCXF analysis results.

    For each source, provides:
    - tf: Transfer function (Vout/Vsrc for voltage sources, Vout/Isrc for current sources)
    - zin: Input impedance seen by source
    - yin: Input admittance (1/zin)

    Attributes:
        tf: Dict mapping source name to transfer function value
        zin: Dict mapping source name to input impedance
        yin: Dict mapping source name to input admittance
        out_node: The output node used
        dc_voltages: DC operating point voltages (if writeop=True)
    """

    tf: Dict[str, float]
    zin: Dict[str, float]
    yin: Dict[str, float]
    out_node: Union[str, int]
    dc_voltages: Optional[Array] = None


def solve_dcxf(
    Jr: Array,
    sources: List[Dict],
    out_node: Union[str, int],
    node_names: Optional[Dict[str, int]] = None,
    dc_voltages: Optional[Array] = None,
) -> DCXFResult:
    """Solve DC transfer function analysis.

    For each source, computes:
    - tf: Vout / Vsrc (or Vout / Isrc for current sources)
    - zin: Input impedance at source terminals
    - yin: 1 / zin

    Args:
        Jr: Resistive Jacobian at DC operating point, shape (n, n)
        sources: List of source specifications
        out_node: Output node (name or index)
        node_names: Optional mapping of node names to indices
        dc_voltages: Optional DC operating point voltages

    Returns:
        DCXFResult with tf, zin, yin for each source
    """
    n = Jr.shape[0]
    G = 1e12  # High conductance for voltage sources

    # Regularize Jacobian
    Jr_reg = Jr + 1e-15 * jnp.eye(n)

    # Determine output node index
    if isinstance(out_node, str):
        if node_names and out_node in node_names:
            out_idx = node_names[out_node] - 1  # Convert to 0-based
        else:
            out_idx = int(out_node) - 1
    else:
        out_idx = int(out_node) - 1

    tf_dict: Dict[str, float] = {}
    zin_dict: Dict[str, float] = {}
    yin_dict: Dict[str, float] = {}

    for source in sources:
        name = source.get("name", "unknown")
        source_type = source.get("type", "vsource")
        pos_node = source.get("pos_node", 0)
        neg_node = source.get("neg_node", 0)

        # Build unity excitation for this source
        du = jnp.zeros(n, dtype=jnp.float64)

        if source_type == "vsource":
            # Unity voltage excitation
            if pos_node > 0:
                du = du.at[pos_node - 1].set(G)
            if neg_node > 0:
                du = du.at[neg_node - 1].set(-G)
        else:
            # Unity current excitation
            if pos_node > 0:
                du = du.at[pos_node - 1].set(1.0)
            if neg_node > 0:
                du = du.at[neg_node - 1].set(-1.0)

        # Solve Jr·dx = du
        dx = jax.scipy.linalg.solve(Jr_reg, du)

        # Extract output voltage
        v_out = float(dx[out_idx]) if 0 <= out_idx < n else 0.0

        # Compute transfer function
        if source_type == "vsource":
            # tf = Vout / Vsrc (Vsrc = 1V)
            tf = v_out
        else:
            # tf = Vout / Isrc (Isrc = 1A)
            tf = v_out

        # Compute input impedance
        # For voltage source: Zin = Vsrc / Isrc
        # Isrc = G * (Vpos - Vneg - Vsrc) at DC, but we want small-signal
        # Input impedance = (Vpos - Vneg) / Isrc_into_source
        if source_type == "vsource":
            v_pos = float(dx[pos_node - 1]) if pos_node > 0 else 0.0
            v_neg = float(dx[neg_node - 1]) if neg_node > 0 else 0.0
            delta_v = v_pos - v_neg
            # Current through source: I = G * (1 - delta_v) for unit excitation
            # But we measure at the terminals, so Zin = 1V / I
            # For small-signal: dI = G * (dVsrc - (dVpos - dVneg)) = G * (1 - delta_v)
            i_src = G * (1.0 - delta_v)
            zin = 1.0 / i_src if abs(i_src) > 1e-30 else 1e30
        else:
            # For current source: Zin = Vpos - Vneg (voltage across for 1A)
            v_pos = float(dx[pos_node - 1]) if pos_node > 0 else 0.0
            v_neg = float(dx[neg_node - 1]) if neg_node > 0 else 0.0
            zin = v_pos - v_neg

        yin = 1.0 / zin if abs(zin) > 1e-30 else 1e30

        tf_dict[name] = tf
        zin_dict[name] = zin
        yin_dict[name] = yin

    return DCXFResult(
        tf=tf_dict,
        zin=zin_dict,
        yin=yin_dict,
        out_node=out_node,
        dc_voltages=dc_voltages,
    )


# =============================================================================
# ACXF (AC Transfer Function) Analysis
# =============================================================================


@dataclass
class ACXFConfig:
    """ACXF analysis configuration.

    Attributes:
        out: Output node specification (node name or index)
        freq_start: Starting frequency in Hz
        freq_stop: Ending frequency in Hz
        mode: Sweep mode - 'lin', 'dec', 'oct', or 'list'
        points: Number of points per decade/octave
        step: Frequency step for 'lin' mode
        values: Explicit frequency list for 'list' mode
        writeop: If True, also return DC operating point data
    """

    out: Union[str, int] = 1
    freq_start: float = 1.0
    freq_stop: float = 1e6
    mode: str = "dec"
    points: int = 10
    step: Optional[float] = None
    values: Optional[List[float]] = None
    writeop: bool = False


@dataclass
class ACXFResult:
    """ACXF analysis results.

    For each source, provides complex-valued:
    - tf: Transfer function over frequency
    - zin: Input impedance over frequency
    - yin: Input admittance over frequency

    Attributes:
        frequencies: Array of frequencies (Hz), shape (n_freqs,)
        tf: Dict mapping source name to complex tf array, shape (n_freqs,)
        zin: Dict mapping source name to complex zin array, shape (n_freqs,)
        yin: Dict mapping source name to complex yin array, shape (n_freqs,)
        out_node: The output node used
        dc_voltages: DC operating point voltages (if writeop=True)
    """

    frequencies: Array
    tf: Dict[str, Array]
    zin: Dict[str, Array]
    yin: Dict[str, Array]
    out_node: Union[str, int]
    dc_voltages: Optional[Array] = None


def solve_acxf_single_freq(
    Jr: Array,
    Jc: Array,
    sources: List[Dict],
    out_idx: int,
    omega: float,
) -> Dict[str, Dict[str, complex]]:
    """Solve ACXF at a single frequency.

    Args:
        Jr: Resistive Jacobian, shape (n, n)
        Jc: Reactive Jacobian, shape (n, n)
        sources: List of source specifications
        out_idx: Output node index (0-based)
        omega: Angular frequency (rad/s)

    Returns:
        Dict with 'tf', 'zin', 'yin' dicts for each source
    """
    n = Jr.shape[0]
    G = 1e12  # High conductance for voltage sources

    # Build complex AC matrix
    J_ac = Jr.astype(jnp.complex128) + 1j * omega * Jc.astype(jnp.complex128)

    # Add regularization
    J_ac = J_ac + 1e-15 * jnp.eye(n, dtype=jnp.complex128)

    results: Dict[str, Dict[str, complex]] = {}

    for source in sources:
        name = source.get("name", "unknown")
        source_type = source.get("type", "vsource")
        pos_node = source.get("pos_node", 0)
        neg_node = source.get("neg_node", 0)

        # Build unity excitation for this source
        U = jnp.zeros(n, dtype=jnp.complex128)

        if source_type == "vsource":
            if pos_node > 0:
                U = U.at[pos_node - 1].set(G + 0j)
            if neg_node > 0:
                U = U.at[neg_node - 1].set(-G + 0j)
        else:
            if pos_node > 0:
                U = U.at[pos_node - 1].set(1.0 + 0j)
            if neg_node > 0:
                U = U.at[neg_node - 1].set(-1.0 + 0j)

        # Solve J_ac·X = U
        X = jax.scipy.linalg.solve(J_ac, U)

        # Extract output voltage
        v_out = X[out_idx] if 0 <= out_idx < n else 0.0 + 0j

        # Transfer function
        tf = complex(v_out)

        # Input impedance
        if source_type == "vsource":
            v_pos = X[pos_node - 1] if pos_node > 0 else 0.0 + 0j
            v_neg = X[neg_node - 1] if neg_node > 0 else 0.0 + 0j
            delta_v = v_pos - v_neg
            i_src = G * (1.0 - delta_v)
            zin = 1.0 / i_src if abs(i_src) > 1e-30 else 1e30 + 0j
        else:
            v_pos = X[pos_node - 1] if pos_node > 0 else 0.0 + 0j
            v_neg = X[neg_node - 1] if neg_node > 0 else 0.0 + 0j
            zin = v_pos - v_neg

        yin = 1.0 / zin if abs(zin) > 1e-30 else 1e30 + 0j

        results[name] = {
            "tf": complex(tf),
            "zin": complex(zin),
            "yin": complex(yin),
        }

    return results


def solve_acxf(
    Jr: Array,
    Jc: Array,
    sources: List[Dict],
    config: ACXFConfig,
    node_names: Optional[Dict[str, int]] = None,
    dc_voltages: Optional[Array] = None,
) -> ACXFResult:
    """Solve AC transfer function analysis over frequency sweep.

    Args:
        Jr: Resistive Jacobian at DC operating point, shape (n, n)
        Jc: Reactive Jacobian at DC operating point, shape (n, n)
        sources: List of source specifications
        config: ACXF configuration
        node_names: Optional mapping of node names to indices
        dc_voltages: Optional DC operating point voltages

    Returns:
        ACXFResult with complex tf, zin, yin over frequency
    """
    from vajax.analysis.ac import ACConfig, generate_frequencies

    # Generate frequencies using AC module's generator
    ac_config = ACConfig(
        freq_start=config.freq_start,
        freq_stop=config.freq_stop,
        mode=config.mode,
        points=config.points,
        step=config.step,
        values=config.values,
    )
    frequencies = generate_frequencies(ac_config)
    n_freqs = len(frequencies)

    logger.info(
        f"ACXF analysis: {n_freqs} frequencies, "
        f"{config.freq_start:.2e} to {config.freq_stop:.2e} Hz"
    )

    # Determine output node index
    out_node = config.out
    if isinstance(out_node, str):
        if node_names and out_node in node_names:
            out_idx = node_names[out_node] - 1
        else:
            out_idx = int(out_node) - 1
    else:
        out_idx = int(out_node) - 1

    # Get source names for result initialization
    source_names = [s.get("name", f"src{i}") for i, s in enumerate(sources)]

    # Initialize result arrays
    tf_arrays: Dict[str, List[complex]] = {name: [] for name in source_names}
    zin_arrays: Dict[str, List[complex]] = {name: [] for name in source_names}
    yin_arrays: Dict[str, List[complex]] = {name: [] for name in source_names}

    # Solve for each frequency
    # Note: Could use vmap here but the loop is clearer and works with variable sources
    for freq in frequencies:
        omega = 2.0 * jnp.pi * freq
        results = solve_acxf_single_freq(Jr, Jc, sources, out_idx, float(omega))

        for name in source_names:
            if name in results:
                tf_arrays[name].append(results[name]["tf"])
                zin_arrays[name].append(results[name]["zin"])
                yin_arrays[name].append(results[name]["yin"])
            else:
                tf_arrays[name].append(0.0 + 0j)
                zin_arrays[name].append(0.0 + 0j)
                yin_arrays[name].append(0.0 + 0j)

    # Convert to JAX arrays
    tf_dict = {name: jnp.array(vals, dtype=jnp.complex128) for name, vals in tf_arrays.items()}
    zin_dict = {name: jnp.array(vals, dtype=jnp.complex128) for name, vals in zin_arrays.items()}
    yin_dict = {name: jnp.array(vals, dtype=jnp.complex128) for name, vals in yin_arrays.items()}

    return ACXFResult(
        frequencies=frequencies,
        tf=tf_dict,
        zin=zin_dict,
        yin=yin_dict,
        out_node=out_node,
        dc_voltages=dc_voltages,
    )
