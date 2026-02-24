"""Simulation options with unified Python API and netlist parsing.

This module provides a centralized definition of all simulation options with:
- Default values
- Type validation
- Parsing from netlist `options` directive
- Python API via properties

Example usage:
    # Via Python API
    engine = CircuitEngine(sim_path)
    engine.options.nr_damping = 0.5
    engine.options.tran_method = IntegrationMethod.TRAP

    # Via netlist options directive
    # options nr_damping=0.5 tran_method="trap"
"""

from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, Optional, Tuple

from vajax.analysis.integration import IntegrationMethod


def _parse_bool(value: Any) -> bool:
    """Parse a boolean from various input types."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    return bool(value)


@dataclass
class SimulationOptions:
    """Centralized simulation options.

    All options can be set via:
    1. Python API: `engine.options.nr_damping = 0.5`
    2. Netlist: `options nr_damping=0.5`

    Options are validated on assignment. Invalid values raise ValueError.

    Temperature convention:
    - User-facing ``temp`` field is in Celsius (matching VACASK ``temp`` option).
    - Internal code converts to Kelvin: ``temperature_k = temp + 273.15``.
    """

    # Simulation temperature (Celsius, matching VACASK 'temp' option)
    temp: float = 27.0
    """Simulation temperature (°C). Default 27°C (300.15K).
    Maps to $temperature (in K) for VA models."""

    # Newton-Raphson solver options
    nr_damping: float = 1.0
    """NR step damping factor. 1.0 = full steps, 0.5 = half steps. Must be in (0, 1]."""

    nr_convtol: float = 0.01
    """NR convergence tolerance multiplier on abstol. VACASK default 0.01.
    Effective tolerance = abstol * nr_convtol."""

    # NR iteration limits
    op_itl: int = 100
    """Max NR iterations for DC operating point (non-continuation). VACASK default 100."""

    op_itlcont: int = 50
    """Max NR iterations in continuation/homotopy mode. VACASK default 50."""

    tran_itl: int = 20
    """Max NR iterations per transient timepoint. Increased from VACASK default (10) for stiff circuits."""

    # Timestep cut factor on NR failure
    tran_ft: float = 0.25
    """Timestep cut factor when NR fails. New dt = dt * tran_ft. VACASK default 0.25."""

    # Integration method
    tran_method: IntegrationMethod = IntegrationMethod.TRAPEZOIDAL
    """Transient integration method (backward_euler, trap, gear2). VACASK default trap."""

    # LTE (Local Truncation Error) control
    tran_lteratio: float = 3.5
    """LTE ratio for adaptive timestep. Higher = larger steps allowed. VACASK default 3.5."""

    tran_redofactor: float = 2.5
    """Factor to reduce timestep when LTE exceeds threshold. VACASK default 2.5."""

    # Timestep control
    tran_fs: float = 0.25
    """Timestep safety factor. Lower = more conservative steps."""

    tran_minpts: int = 50
    """Minimum output points (caps max_dt). VACASK default 50."""

    # Tolerances
    reltol: float = 1e-3
    """Relative tolerance for convergence checks."""

    abstol: float = 1e-12
    """Absolute current tolerance (A). VACASK default 1e-12."""

    vntol: float = 1e-6
    """Absolute voltage tolerance (V). Used for delta convergence. VACASK default 1e-6."""

    # Conductance options
    tran_gshunt: float = 0.0
    """Shunt conductance to ground for numerical stability."""

    gmin: float = 1e-12
    """Minimum conductance for matrix conditioning."""

    # Analysis parameters (typically from .tran directive)
    step: Optional[float] = None
    """Requested output timestep."""

    stop: Optional[float] = None
    """Simulation stop time."""

    maxstep: Optional[float] = None
    """Maximum internal timestep."""

    icmode: str = "op"
    """Initial condition mode: 'op' (DC operating point), 'ic' (use .ic), or 'uic' (use IC)."""

    # Homotopy chain control
    op_homotopy: Tuple[str, ...] = ("gdev", "gshunt", "src")
    """Homotopy algorithms to try (in order) when plain OP fails."""

    op_srchomotopy: Tuple[str, ...] = ("gdev", "gshunt")
    """Homotopy algorithms to try when source stepping fails at factor=0."""

    op_skipinitial: bool = False
    """Skip plain OP, go straight to homotopy."""

    # Homotopy gmin stepping
    homotopy_gminsteps: int = 100
    """Max gmin stepping steps."""

    homotopy_gminfactor: float = 10.0
    """Initial gmin reduction factor per step."""

    homotopy_maxgminfactor: float = 10.0
    """Max gmin reduction factor (adaptive ceiling)."""

    homotopy_mingminfactor: float = 1.00005
    """Min gmin reduction factor (give-up threshold)."""

    homotopy_startgmin: float = 1e-3
    """Starting gmin value for gdev/gshunt stepping."""

    homotopy_maxgmin: float = 1e2
    """Max gmin value (fail threshold)."""

    homotopy_mingmin: float = 1e-15
    """Target gmin value (success threshold)."""

    # Homotopy source stepping
    homotopy_srcsteps: int = 100
    """Max source stepping steps."""

    homotopy_srcstep: float = 0.01
    """Initial source step size (fraction of full source)."""

    homotopy_srcscale: float = 3.0
    """Source step scaling factor on successful convergence."""

    homotopy_minsrcstep: float = 1e-7
    """Min source step size (give-up threshold)."""

    def __post_init__(self):
        """Validate all options after initialization."""
        self._validate_all()

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute with validation."""
        if name == "temp" and value <= -273.15:
            raise ValueError(f"temp must be > -273.15°C (absolute zero), got {value}")
        if name == "nr_damping" and not (0 < value <= 1.0):
            raise ValueError(f"nr_damping must be in (0, 1], got {value}")
        if name == "nr_convtol" and value <= 0:
            raise ValueError(f"nr_convtol must be positive, got {value}")
        if name == "op_itl" and value < 1:
            raise ValueError(f"op_itl must be >= 1, got {value}")
        if name == "op_itlcont" and value < 1:
            raise ValueError(f"op_itlcont must be >= 1, got {value}")
        if name == "tran_itl" and value < 1:
            raise ValueError(f"tran_itl must be >= 1, got {value}")
        if name == "tran_ft" and not (0 < value < 1):
            raise ValueError(f"tran_ft must be in (0, 1), got {value}")
        if name == "tran_lteratio" and value <= 0:
            raise ValueError(f"tran_lteratio must be positive, got {value}")
        if name == "tran_redofactor" and value <= 1:
            raise ValueError(f"tran_redofactor must be > 1, got {value}")
        if name == "tran_fs" and not (0 < value <= 1.0):
            raise ValueError(f"tran_fs must be in (0, 1], got {value}")
        if name == "tran_minpts" and value < 1:
            raise ValueError(f"tran_minpts must be >= 1, got {value}")
        if name == "reltol" and value <= 0:
            raise ValueError(f"reltol must be positive, got {value}")
        if name == "abstol" and value <= 0:
            raise ValueError(f"abstol must be positive, got {value}")
        if name == "vntol" and value <= 0:
            raise ValueError(f"vntol must be positive, got {value}")
        if name == "gmin" and value < 0:
            raise ValueError(f"gmin must be non-negative, got {value}")
        if name == "tran_gshunt" and value < 0:
            raise ValueError(f"tran_gshunt must be non-negative, got {value}")
        if name == "icmode" and value not in ("op", "ic", "uic"):
            raise ValueError(f"icmode must be 'op', 'ic', or 'uic', got {value}")

        object.__setattr__(self, name, value)

    def _validate_all(self):
        """Validate all option values."""
        if self.temp <= -273.15:
            raise ValueError(f"temp must be > -273.15°C, got {self.temp}")
        if not (0 < self.nr_damping <= 1.0):
            raise ValueError(f"nr_damping must be in (0, 1], got {self.nr_damping}")
        if self.nr_convtol <= 0:
            raise ValueError(f"nr_convtol must be positive, got {self.nr_convtol}")
        if self.op_itl < 1:
            raise ValueError(f"op_itl must be >= 1, got {self.op_itl}")
        if self.op_itlcont < 1:
            raise ValueError(f"op_itlcont must be >= 1, got {self.op_itlcont}")
        if self.tran_itl < 1:
            raise ValueError(f"tran_itl must be >= 1, got {self.tran_itl}")
        if not (0 < self.tran_ft < 1):
            raise ValueError(f"tran_ft must be in (0, 1), got {self.tran_ft}")
        if self.tran_lteratio <= 0:
            raise ValueError(f"tran_lteratio must be positive, got {self.tran_lteratio}")
        if self.tran_redofactor <= 1:
            raise ValueError(f"tran_redofactor must be > 1, got {self.tran_redofactor}")
        if not (0 < self.tran_fs <= 1.0):
            raise ValueError(f"tran_fs must be in (0, 1], got {self.tran_fs}")
        if self.tran_minpts < 1:
            raise ValueError(f"tran_minpts must be >= 1, got {self.tran_minpts}")
        if self.reltol <= 0:
            raise ValueError(f"reltol must be positive, got {self.reltol}")
        if self.abstol <= 0:
            raise ValueError(f"abstol must be positive, got {self.abstol}")
        if self.vntol <= 0:
            raise ValueError(f"vntol must be positive, got {self.vntol}")
        if self.gmin < 0:
            raise ValueError(f"gmin must be non-negative, got {self.gmin}")
        if self.tran_gshunt < 0:
            raise ValueError(f"tran_gshunt must be non-negative, got {self.tran_gshunt}")
        if self.icmode not in ("op", "ic", "uic"):
            raise ValueError(f"icmode must be 'op', 'ic', or 'uic', got {self.icmode}")

    def set(self, name: str, value: Any) -> None:
        """Set an option by name with validation.

        Args:
            name: Option name (e.g., 'nr_damping')
            value: Option value (will be converted to appropriate type)

        Raises:
            ValueError: If option name is unknown or value is invalid
        """
        if not hasattr(self, name):
            raise ValueError(f"Unknown option: {name}")

        # Get the field type for conversion
        field_type = None
        for f in fields(self):
            if f.name == name:
                field_type = f.type
                break

        # Convert value to appropriate type
        if field_type == float or field_type == Optional[float]:
            value = float(value) if value is not None else None
        elif field_type == int:
            value = int(value)
        elif field_type == bool:
            value = _parse_bool(value)
        elif field_type == str:
            value = str(value).strip("\"'")
        elif field_type == IntegrationMethod:
            if isinstance(value, str):
                value = IntegrationMethod.from_string(value)

        # Set and validate
        setattr(self, name, value)
        self._validate_all()

    def get(self, name: str, default: Any = None) -> Any:
        """Get an option value by name.

        Args:
            name: Option name
            default: Default value if option is None

        Returns:
            Option value or default
        """
        value = getattr(self, name, default)
        return default if value is None else value

    def update_from_netlist(
        self, opts: Dict[str, Any], parse_number: Callable[[str], float] = float
    ) -> None:
        """Update options from netlist options directive.

        Args:
            opts: Dictionary from OptionsDirective.params
            parse_number: Function to parse SPICE numbers (e.g., '1u' -> 1e-6)
        """
        # Map of netlist option names to our field names (if different)
        name_map = {
            # All names are the same currently, but this allows for aliases
        }

        # Fields that should be parsed as floats
        _float_fields = {
            "temp",
            "step",
            "stop",
            "maxstep",
            "nr_damping",
            "nr_convtol",
            "tran_ft",
            "tran_lteratio",
            "tran_redofactor",
            "tran_fs",
            "reltol",
            "abstol",
            "vntol",
            "tran_gshunt",
            "gmin",
            "homotopy_gminfactor",
            "homotopy_maxgminfactor",
            "homotopy_mingminfactor",
            "homotopy_startgmin",
            "homotopy_maxgmin",
            "homotopy_mingmin",
            "homotopy_srcstep",
            "homotopy_srcscale",
            "homotopy_minsrcstep",
        }

        # Fields that should be parsed as ints
        _int_fields = {
            "op_itl",
            "op_itlcont",
            "tran_itl",
            "tran_minpts",
            "homotopy_gminsteps",
            "homotopy_srcsteps",
        }

        for opt_name, opt_value in opts.items():
            # Map name if needed
            field_name = name_map.get(opt_name, opt_name)

            # Skip unknown options (might be for other tools)
            if not hasattr(self, field_name):
                continue

            # Parse numeric values using SPICE number parser
            try:
                if field_name in _float_fields:
                    opt_value = parse_number(opt_value)
                elif field_name in _int_fields:
                    opt_value = int(parse_number(opt_value))

                self.set(field_name, opt_value)
            except (ValueError, TypeError) as e:
                # Log warning but don't fail - allows forward compatibility
                import logging

                logging.getLogger("vajax").warning(
                    f"Failed to parse option {opt_name}={opt_value}: {e}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert options to dictionary.

        Returns:
            Dictionary of all option values
        """
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            # Convert IntegrationMethod to string for serialization
            if isinstance(value, IntegrationMethod):
                value = value.value
            # Convert tuples to lists for JSON serialization
            if isinstance(value, tuple):
                value = list(value)
            result[f.name] = value
        return result

    def copy(self) -> "SimulationOptions":
        """Create a copy of these options.

        Returns:
            New SimulationOptions instance with same values
        """
        return SimulationOptions(**{f.name: getattr(self, f.name) for f in fields(self)})
