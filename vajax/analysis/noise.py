"""Noise Analysis for VA-JAX

Implements small-signal noise analysis following VACASK algorithm:

1. Compute DC operating point
2. Linearize circuit: Jr (resistive) and Jc (reactive) Jacobians
3. For each frequency:
   a. Build AC matrix: Jr + j*omega*Jc
   b. Compute power gain from input source to output
   c. For each noise source:
      - Compute noise PSD (power spectral density)
      - Apply unity excitation from noise source
      - Solve for transfer function to output
      - Contribution = |transfer|^2 * PSD
   d. Sum all contributions for total output noise

Noise source types:
- Thermal (white noise): PSD = 4*k*T/R for resistors
- Shot (white noise): PSD = 2*q*I for diodes/transistors
- Flicker (1/f noise): PSD = Kf*I^Af/f^Ef

Physical constants:
- k (Boltzmann): 1.380649e-23 J/K
- q (electron charge): 1.602176634e-19 C
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array

from vajax.config import K_BOLTZMANN, Q_ELECTRON

logger = logging.getLogger(__name__)

T_NOMINAL = 300.15  # 27°C in Kelvin

# Thermal voltage at 300K for DC current estimation
_VT_300K = K_BOLTZMANN * T_NOMINAL / Q_ELECTRON


def extract_dc_currents(
    devices: List[Dict],
    V_dc: Array,
) -> Dict[str, float]:
    """Extract DC currents through devices from operating point.

    Used for noise analysis to determine shot/flicker noise levels.

    Args:
        devices: List of device dictionaries from circuit parsing
        V_dc: DC operating point voltages (excluding ground, so node i -> V_dc[i-1])

    Returns:
        Dict mapping device name to DC current
    """
    dc_currents: Dict[str, float] = {}

    for dev in devices:
        name = dev.get("name", "")
        model = dev.get("model", "")
        params = dev.get("params", {})
        nodes = dev.get("nodes", [0, 0])

        pos_node = nodes[0] if len(nodes) > 0 else 0
        neg_node = nodes[1] if len(nodes) > 1 else 0

        # Get node voltages (index adjusted for ground exclusion)
        v_pos = float(V_dc[pos_node - 1]) if pos_node > 0 and pos_node <= len(V_dc) else 0.0
        v_neg = float(V_dc[neg_node - 1]) if neg_node > 0 and neg_node <= len(V_dc) else 0.0
        v_diff = v_pos - v_neg

        if model == "resistor":
            r = float(params.get("r", 1000.0))
            if r > 0:
                dc_currents[name] = v_diff / r

        elif model in ("diode", "d"):
            # Diode current: I = Is * (exp(V/nVT) - 1)
            Is = float(params.get("is", 1e-14))
            n = float(params.get("n", 1.0))
            vt = _VT_300K
            if v_diff > -5 * n * vt:  # Avoid overflow
                dc_currents[name] = float(Is * (jnp.exp(v_diff / (n * vt)) - 1))
            else:
                dc_currents[name] = -Is

    return dc_currents


@dataclass
class NoiseConfig:
    """Noise analysis configuration.

    Attributes:
        out: Output node specification (node name or index)
        input_source: Name of input source for power gain calculation
        freq_start: Starting frequency in Hz
        freq_stop: Ending frequency in Hz
        mode: Sweep mode - 'lin', 'dec', 'oct', or 'list'
        points: Number of points per decade/octave
        step: Frequency step for 'lin' mode
        values: Explicit frequency list for 'list' mode
        temperature: Circuit temperature in Kelvin
    """

    out: Union[str, int] = 1
    input_source: str = ""
    freq_start: float = 1.0
    freq_stop: float = 1e6
    mode: str = "dec"
    points: int = 10
    step: Optional[float] = None
    values: Optional[List[float]] = None
    temperature: float = T_NOMINAL


@dataclass
class NoiseSource:
    """A noise source in the circuit.

    Attributes:
        name: Source name (e.g., "r1")
        source_type: Type of noise ("thermal", "shot", "flicker")
        device_name: Parent device name
        pos_node: Positive node index
        neg_node: Negative node index
        psd_params: Parameters for PSD calculation
            - thermal: {'resistance': R, 'temperature': T}
            - shot: {'current': I}
            - flicker: {'current': I, 'kf': Kf, 'af': Af, 'ef': Ef}
    """

    name: str
    source_type: str
    device_name: str
    pos_node: int
    neg_node: int
    psd_params: Dict


@dataclass
class NoiseResult:
    """Noise analysis results.

    Attributes:
        frequencies: Array of frequencies (Hz), shape (n_freqs,)
        output_noise: Total output noise PSD (V^2/Hz), shape (n_freqs,)
        power_gain: Power gain from input to output, shape (n_freqs,)
        contributions: Dict mapping device name to noise contribution, shape (n_freqs,)
        detailed_contributions: Dict mapping (device, type) to contribution
        dc_voltages: DC operating point voltages
    """

    frequencies: Array
    output_noise: Array
    power_gain: Array
    contributions: Dict[str, Array]
    detailed_contributions: Dict[Tuple[str, str], Array]
    dc_voltages: Optional[Array] = None


def compute_thermal_noise_psd(resistance, temperature=T_NOMINAL):
    """Compute thermal noise PSD for a resistor.

    Nyquist-Johnson thermal noise: PSD = 4*k*T/R (A^2/Hz)

    Args:
        resistance: Resistance in Ohms
        temperature: Temperature in Kelvin

    Returns:
        Power spectral density in A^2/Hz
    """
    resistance = float(resistance)
    temperature = float(temperature)
    if resistance <= 0:
        return 0.0
    return 4.0 * K_BOLTZMANN * temperature / resistance


def compute_shot_noise_psd(current):
    """Compute shot noise PSD for a PN junction.

    Schottky shot noise: PSD = 2*q*|I| (A^2/Hz)

    Args:
        current: DC current through junction in Amperes

    Returns:
        Power spectral density in A^2/Hz
    """
    current = float(current)
    return 2.0 * Q_ELECTRON * abs(current)


def compute_flicker_noise_psd(
    current,
    frequency,
    kf=0.0,
    af=1.0,
    ef=1.0,
):
    """Compute flicker (1/f) noise PSD.

    Hooge flicker noise: PSD = Kf * |I|^Af / f^Ef (A^2/Hz)

    Args:
        current: DC current in Amperes
        frequency: Frequency in Hz
        kf: Flicker noise coefficient
        af: Current exponent (typically 1-2)
        ef: Frequency exponent (typically 1)

    Returns:
        Power spectral density in A^2/Hz
    """
    current = float(current)
    frequency = float(frequency)
    kf = float(kf)
    af = float(af)
    ef = float(ef)
    if kf <= 0 or frequency <= 0:
        return 0.0
    return kf * (abs(current) ** af) / (frequency**ef)


def extract_noise_sources(
    devices: List[Dict],
    dc_currents: Dict[str, float],
    temperature: float = T_NOMINAL,
) -> List[NoiseSource]:
    """Extract noise sources from circuit devices.

    Args:
        devices: List of device specifications
        dc_currents: Dict mapping device name to DC current
        temperature: Circuit temperature in Kelvin

    Returns:
        List of NoiseSource objects
    """
    noise_sources = []

    for dev in devices:
        name = dev.get("name", "")
        model = dev.get("model", "")
        params = dev.get("params", {})
        nodes = dev.get("nodes", [0, 0])

        pos_node = nodes[0] if len(nodes) > 0 else 0
        neg_node = nodes[1] if len(nodes) > 1 else 0

        if model == "resistor":
            # Thermal noise for resistor
            r = float(params.get("r", 1000.0))
            has_noise = params.get("has_noise", 1)

            if has_noise and r > 0:
                noise_sources.append(
                    NoiseSource(
                        name=f"{name}_thermal",
                        source_type="thermal",
                        device_name=name,
                        pos_node=pos_node,
                        neg_node=neg_node,
                        psd_params={"resistance": r, "temperature": temperature},
                    )
                )

        elif model in ("diode", "d"):
            # Shot and flicker noise for diode
            i_dc = dc_currents.get(name, 0.0)

            # Shot noise
            noise_sources.append(
                NoiseSource(
                    name=f"{name}_shot",
                    source_type="shot",
                    device_name=name,
                    pos_node=pos_node,
                    neg_node=neg_node,
                    psd_params={"current": i_dc},
                )
            )

            # Flicker noise
            kf = float(params.get("kf", 0.0))
            af = float(params.get("af", 1.0))
            ef = float(params.get("ef", 1.0))

            if kf > 0:
                noise_sources.append(
                    NoiseSource(
                        name=f"{name}_flicker",
                        source_type="flicker",
                        device_name=name,
                        pos_node=pos_node,
                        neg_node=neg_node,
                        psd_params={"current": i_dc, "kf": kf, "af": af, "ef": ef},
                    )
                )

    return noise_sources


def compute_noise_psd(source: NoiseSource, frequency: float) -> float:
    """Compute PSD for a noise source at given frequency.

    Args:
        source: NoiseSource object
        frequency: Frequency in Hz

    Returns:
        Power spectral density in A^2/Hz
    """
    if source.source_type == "thermal":
        return compute_thermal_noise_psd(
            source.psd_params.get("resistance", 1000.0),
            source.psd_params.get("temperature", T_NOMINAL),
        )
    elif source.source_type == "shot":
        return compute_shot_noise_psd(
            source.psd_params.get("current", 0.0),
        )
    elif source.source_type == "flicker":
        return compute_flicker_noise_psd(
            source.psd_params.get("current", 0.0),
            frequency,
            source.psd_params.get("kf", 0.0),
            source.psd_params.get("af", 1.0),
            source.psd_params.get("ef", 1.0),
        )
    else:
        return 0.0


def solve_noise_single_freq(
    Jr: Array,
    Jc: Array,
    noise_sources: List[NoiseSource],
    input_source: Optional[Dict],
    out_idx: int,
    frequency: float,
) -> Tuple[float, float, Dict[str, float], Dict[Tuple[str, str], float]]:
    """Solve noise analysis at a single frequency.

    Args:
        Jr: Resistive Jacobian, shape (n, n)
        Jc: Reactive Jacobian, shape (n, n)
        noise_sources: List of noise sources
        input_source: Input source specification (for power gain)
        out_idx: Output node index (0-based)
        frequency: Frequency in Hz

    Returns:
        Tuple of (output_noise, power_gain, device_contributions, detailed_contributions)
    """
    n = Jr.shape[0]
    omega = 2.0 * jnp.pi * frequency
    G = 1e12  # High conductance for voltage sources

    # Build complex AC matrix
    J_ac = Jr.astype(jnp.complex128) + 1j * omega * Jc.astype(jnp.complex128)
    J_ac = J_ac + 1e-15 * jnp.eye(n, dtype=jnp.complex128)

    # Compute power gain from input source
    power_gain = 0.0
    if input_source:
        pos_node = input_source.get("pos_node", 0)
        neg_node = input_source.get("neg_node", 0)
        source_type = input_source.get("type", "vsource")

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

        X = jax.scipy.linalg.solve(J_ac, U)
        v_out = X[out_idx] if 0 <= out_idx < n else 0.0 + 0j
        power_gain = float(abs(v_out) ** 2)

    # Compute noise contributions
    output_noise = 0.0
    device_contributions: Dict[str, float] = {}
    detailed_contributions: Dict[Tuple[str, str], float] = {}

    for source in noise_sources:
        # Compute PSD at this frequency
        psd = compute_noise_psd(source, frequency)
        if psd <= 0:
            continue

        # Build excitation for noise source (unity current injection)
        U = jnp.zeros(n, dtype=jnp.complex128)
        pos_node = source.pos_node
        neg_node = source.neg_node

        if pos_node > 0:
            U = U.at[pos_node - 1].set(1.0 + 0j)
        if neg_node > 0:
            U = U.at[neg_node - 1].set(-1.0 + 0j)

        # Solve for transfer function
        X = jax.scipy.linalg.solve(J_ac, U)
        v_out = X[out_idx] if 0 <= out_idx < n else 0.0 + 0j

        # Power gain from noise source to output
        gain = float(abs(v_out) ** 2)

        # Noise contribution
        contribution = gain * psd

        # Accumulate
        output_noise += contribution

        # Per-device contribution
        dev_name = source.device_name
        device_contributions[dev_name] = device_contributions.get(dev_name, 0.0) + contribution

        # Detailed contribution
        key = (source.device_name, source.source_type)
        detailed_contributions[key] = detailed_contributions.get(key, 0.0) + contribution

    return output_noise, power_gain, device_contributions, detailed_contributions


def run_noise_analysis(
    Jr: Array,
    Jc: Array,
    noise_sources: List[NoiseSource],
    input_source: Optional[Dict],
    config: NoiseConfig,
    node_names: Optional[Dict[str, int]] = None,
    dc_voltages: Optional[Array] = None,
) -> NoiseResult:
    """Run complete noise analysis.

    Args:
        Jr: Resistive Jacobian at DC operating point, shape (n, n)
        Jc: Reactive Jacobian at DC operating point, shape (n, n)
        noise_sources: List of noise sources
        input_source: Input source for power gain (or None)
        config: Noise analysis configuration
        node_names: Optional mapping of node names to indices
        dc_voltages: Optional DC operating point voltages

    Returns:
        NoiseResult with output noise, power gain, and contributions
    """
    from vajax.analysis.ac import ACConfig, generate_frequencies

    # Generate frequencies
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
        f"Noise analysis: {n_freqs} frequencies, "
        f"{config.freq_start:.2e} to {config.freq_stop:.2e} Hz, "
        f"{len(noise_sources)} noise sources"
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

    # Initialize result arrays
    output_noise_list = []
    power_gain_list = []

    # Get all device names
    device_names = set(s.device_name for s in noise_sources)
    device_contribs = {name: [] for name in device_names}

    # Get all (device, type) pairs
    detail_keys = set((s.device_name, s.source_type) for s in noise_sources)
    detailed_contribs = {key: [] for key in detail_keys}

    # Sweep frequencies
    for freq in frequencies:
        onoise, pgain, dev_c, det_c = solve_noise_single_freq(
            Jr, Jc, noise_sources, input_source, out_idx, float(freq)
        )

        output_noise_list.append(onoise)
        power_gain_list.append(pgain)

        for name in device_names:
            device_contribs[name].append(dev_c.get(name, 0.0))

        for key in detail_keys:
            detailed_contribs[key].append(det_c.get(key, 0.0))

    # Convert to arrays
    output_noise = jnp.array(output_noise_list, dtype=jnp.float64)
    power_gain = jnp.array(power_gain_list, dtype=jnp.float64)

    contributions = {
        name: jnp.array(vals, dtype=jnp.float64) for name, vals in device_contribs.items()
    }

    detailed = {key: jnp.array(vals, dtype=jnp.float64) for key, vals in detailed_contribs.items()}

    return NoiseResult(
        frequencies=frequencies,
        output_noise=output_noise,
        power_gain=power_gain,
        contributions=contributions,
        detailed_contributions=detailed,
        dc_voltages=dc_voltages,
    )
