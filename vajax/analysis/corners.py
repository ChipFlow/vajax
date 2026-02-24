"""Corner analysis data structures and utilities.

Provides PVT (Process/Voltage/Temperature) corner sweep capabilities:
- ProcessCorner: Defines parameter scaling for FF/TT/SS corners
- VoltageCorner: Defines supply voltage variations
- CornerConfig: Complete corner specification (process + voltage + temperature)
- CornerSweepResult: Aggregated results from multiple corner simulations

Example usage:
    from vajax.analysis.corners import (
        CornerConfig, create_standard_corners, PROCESS_CORNERS
    )

    # Create a single corner
    corner = CornerConfig(
        name='FF_hot',
        process=PROCESS_CORNERS['FF'],
        temperature=398.15  # 125°C
    )

    # Create standard PVT corners
    corners = create_standard_corners(
        processes=['FF', 'TT', 'SS'],
        temperatures=[233.15, 300.15, 398.15],  # -40C, 27C, 125C
        vdd_scales=[0.9, 1.0, 1.1]
    )

    # Run simulation across corners
    results = engine.run_corners(corners, t_stop=1e-3, dt=1e-6)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from vajax.config import DEFAULT_TEMPERATURE_K


@dataclass
class ProcessCorner:
    """Process corner specification.

    Defines parameter scaling factors for different process corners.
    FF = Fast-Fast, TT = Typical-Typical, SS = Slow-Slow, etc.

    Attributes:
        name: Corner name (e.g., 'FF', 'TT', 'SS')
        mobility_scale: Scaling factor for mobility params (mu0, uo). >1 = faster
        vth_shift: Threshold voltage shift in Volts. <0 = faster (lower threshold)
        tox_scale: Gate oxide thickness scaling. <1 = faster (thinner oxide)
        length_delta: Channel length adjustment in meters
        model_params: Per-model parameter overrides {model_name: {param: value}}
    """

    name: str
    mobility_scale: float = 1.0
    vth_shift: float = 0.0
    tox_scale: float = 1.0
    length_delta: float = 0.0
    model_params: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class VoltageCorner:
    """Voltage corner specification.

    Defines supply voltage variations for corner analysis.

    Attributes:
        name: Corner name (e.g., 'nom', 'low', 'high')
        vdd_scale: VDD scaling factor (1.0 = nominal)
        source_values: Explicit values for specific sources {source_name: voltage}
    """

    name: str
    vdd_scale: float = 1.0
    source_values: Dict[str, float] = field(default_factory=dict)


@dataclass
class CornerConfig:
    """Complete corner configuration (PVT).

    Combines process, voltage, and temperature specifications.

    Attributes:
        name: Descriptive corner name
        process: Process corner specification (or None for nominal)
        voltage: Voltage corner specification (or None for nominal)
        temperature: Simulation temperature in Kelvin
    """

    name: str
    process: Optional[ProcessCorner] = None
    voltage: Optional[VoltageCorner] = None
    temperature: float = DEFAULT_TEMPERATURE_K  # 27°C


@dataclass
class CornerResult:
    """Results from a single corner simulation.

    Attributes:
        corner: Corner configuration used
        result: Simulation result (TransientResult, DCResult, etc.)
        converged: Whether simulation converged successfully
        stats: Additional statistics and metadata
    """

    corner: CornerConfig
    result: Any  # TransientResult, DCResult, etc.
    converged: bool
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CornerSweepResult:
    """Aggregated results from corner sweep.

    Attributes:
        corners: List of corner configurations run
        results: List of CornerResult objects
    """

    corners: List[CornerConfig]
    results: List[CornerResult]

    @property
    def num_corners(self) -> int:
        """Number of corners in the sweep."""
        return len(self.corners)

    @property
    def num_converged(self) -> int:
        """Number of corners that converged."""
        return sum(1 for r in self.results if r.converged)

    @property
    def all_converged(self) -> bool:
        """Whether all corners converged."""
        return self.num_converged == self.num_corners

    def get_result(self, corner_name: str) -> Optional[CornerResult]:
        """Get result for a specific corner by name.

        Args:
            corner_name: Name of the corner

        Returns:
            CornerResult or None if not found
        """
        for result in self.results:
            if result.corner.name == corner_name:
                return result
        return None

    def converged_results(self) -> List[CornerResult]:
        """Get only the converged results."""
        return [r for r in self.results if r.converged]


# Standard process corners
PROCESS_CORNERS: Dict[str, ProcessCorner] = {
    "FF": ProcessCorner(
        name="FF",
        mobility_scale=1.15,
        vth_shift=-0.05,
        tox_scale=0.95,
    ),
    "TT": ProcessCorner(
        name="TT",
        # All nominal (no scaling)
    ),
    "SS": ProcessCorner(
        name="SS",
        mobility_scale=0.85,
        vth_shift=0.05,
        tox_scale=1.05,
    ),
    "FS": ProcessCorner(
        name="FS",
        mobility_scale=1.15,  # Fast NMOS
        vth_shift=0.05,  # Slow PMOS (higher |Vth|)
    ),
    "SF": ProcessCorner(
        name="SF",
        mobility_scale=0.85,  # Slow NMOS
        vth_shift=-0.05,  # Fast PMOS (lower |Vth|)
    ),
}

# Standard temperature corners (in Kelvin)
TEMPERATURE_CORNERS: Dict[str, float] = {
    "cold": 233.15,  # -40°C
    "room": 300.15,  # 27°C
    "hot": 398.15,  # 125°C
}


def create_standard_corners(
    processes: Optional[List[str]] = None,
    temperatures: Optional[List[float]] = None,
    vdd_scales: Optional[List[float]] = None,
) -> List[CornerConfig]:
    """Create corner configurations from standard specifications.

    Generates all combinations of process/voltage/temperature corners.

    Args:
        processes: List of process corner names ('FF', 'TT', 'SS', 'FS', 'SF')
                   Default: ['TT']
        temperatures: List of temperatures in Kelvin
                      Default: [300.15] (room temperature)
        vdd_scales: List of VDD scaling factors
                    Default: [1.0] (nominal)

    Returns:
        List of CornerConfig for all combinations
    """
    if processes is None:
        processes = ["TT"]
    if temperatures is None:
        temperatures = [DEFAULT_TEMPERATURE_K]
    if vdd_scales is None:
        vdd_scales = [1.0]

    corners = []

    for proc_name in processes:
        process = PROCESS_CORNERS.get(proc_name)
        if process is None and proc_name != "TT":
            raise ValueError(f"Unknown process corner: {proc_name}")

        for temp in temperatures:
            for vdd in vdd_scales:
                # Create descriptive name
                temp_c = temp - 273.15
                if temp_c == -40:
                    temp_str = "m40C"
                elif temp_c == 27:
                    temp_str = "27C"
                elif temp_c == 125:
                    temp_str = "125C"
                else:
                    temp_str = f"{temp_c:.0f}C"

                vdd_str = f"V{vdd * 100:.0f}pct"
                name = f"{proc_name}_{temp_str}_{vdd_str}"

                voltage = VoltageCorner(name=f"VDD_{vdd}", vdd_scale=vdd)

                corners.append(
                    CornerConfig(
                        name=name,
                        process=process,
                        voltage=voltage,
                        temperature=temp,
                    )
                )

    return corners


def apply_process_corner(devices: List[Dict[str, Any]], corner: Optional[ProcessCorner]) -> None:
    """Apply process corner scaling to device parameters.

    Modifies device parameters in place based on corner specification.

    Args:
        devices: List of device dicts with 'params', 'is_openvaf', 'model' keys
        corner: Process corner to apply (or None for nominal)
    """
    if corner is None:
        return

    for dev in devices:
        if not dev.get("is_openvaf", False):
            continue

        params = dev.get("params", {})
        model = dev.get("model", "")

        # Apply mobility scaling
        mobility_params = ("uo", "mu0", "u0", "betn", "betp", "mue")
        for param in mobility_params:
            if param in params:
                params[param] = float(params[param]) * corner.mobility_scale

        # Apply Vth shift
        vth_params = ("vth0", "vfb", "delvto", "dvt0")
        for param in vth_params:
            if param in params:
                params[param] = float(params[param]) + corner.vth_shift

        # Apply Tox scaling
        tox_params = ("tox", "toxe", "toxo", "toxp")
        for param in tox_params:
            if param in params:
                params[param] = float(params[param]) * corner.tox_scale

        # Apply length delta
        if corner.length_delta != 0:
            if "l" in params:
                params["l"] = float(params["l"]) + corner.length_delta

        # Apply model-specific overrides
        if model in corner.model_params:
            for param, value in corner.model_params[model].items():
                params[param] = value


def apply_voltage_corner(devices: List[Dict[str, Any]], corner: Optional[VoltageCorner]) -> None:
    """Apply voltage corner scaling to source devices.

    Modifies voltage source DC values based on corner specification.

    Args:
        devices: List of device dicts with 'model', 'name', 'params' keys
        corner: Voltage corner to apply (or None for nominal)
    """
    if corner is None:
        return

    for dev in devices:
        if dev.get("model") != "vsource":
            continue

        name = dev.get("name", "")
        params = dev.get("params", {})

        # Check for explicit source value
        if name in corner.source_values:
            params["dc"] = corner.source_values[name]
        elif "dc" in params:
            # Apply general VDD scaling
            params["dc"] = float(params["dc"]) * corner.vdd_scale


def create_pvt_corners(
    processes: Optional[List[str]] = None,
    temperatures: Optional[List[str]] = None,
    voltages: Optional[List[float]] = None,
) -> List[CornerConfig]:
    """Create standard PVT corner matrix.

    Convenience function for typical PVT analysis.

    Args:
        processes: Process corners (default: ['FF', 'TT', 'SS'])
        temperatures: Temperature corner names (default: ['cold', 'room', 'hot'])
        voltages: VDD scales (default: [0.9, 1.0, 1.1])

    Returns:
        List of CornerConfig (default: 27 corners = 3 × 3 × 3)
    """
    if processes is None:
        processes = ["FF", "TT", "SS"]

    if temperatures is None:
        temp_values = [
            TEMPERATURE_CORNERS["cold"],
            TEMPERATURE_CORNERS["room"],
            TEMPERATURE_CORNERS["hot"],
        ]
    else:
        temp_values = [
            TEMPERATURE_CORNERS[t] if t in TEMPERATURE_CORNERS else float(t) for t in temperatures
        ]

    if voltages is None:
        voltages = [0.9, 1.0, 1.1]

    return create_standard_corners(
        processes=processes,
        temperatures=temp_values,
        vdd_scales=voltages,
    )
