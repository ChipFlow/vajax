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

from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional, Union

from jax_spice.analysis.integration import IntegrationMethod


def _parse_bool(value: Any) -> bool:
    """Parse a boolean from various input types."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    return bool(value)


@dataclass
class SimulationOptions:
    """Centralized simulation options.

    All options can be set via:
    1. Python API: `engine.options.nr_damping = 0.5`
    2. Netlist: `options nr_damping=0.5`

    Options are validated on assignment. Invalid values raise ValueError.
    """

    # Newton-Raphson solver options
    nr_damping: float = 1.0
    """NR step damping factor. 1.0 = full steps, 0.5 = half steps. Must be in (0, 1]."""

    nr_convtol: float = 1e4
    """NR convergence tolerance (residual threshold). Default matches ~10nV with G=1e12."""

    # Integration method
    tran_method: IntegrationMethod = IntegrationMethod.BACKWARD_EULER
    """Transient integration method (backward_euler, trap, gear2)."""

    # LTE (Local Truncation Error) control
    tran_lteratio: float = 10.0
    """LTE ratio for adaptive timestep. Higher = larger steps allowed."""

    tran_redofactor: float = 4.0
    """Factor to reduce timestep when LTE exceeds threshold."""

    # Timestep control
    tran_fs: float = 0.25
    """Timestep safety factor. Lower = more conservative steps."""

    tran_minpts: int = 2
    """Minimum points per smallest time constant."""

    # Tolerances
    reltol: float = 1e-3
    """Relative tolerance for convergence checks."""

    abstol: float = 1e-12
    """Absolute tolerance for convergence checks."""

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

    icmode: str = 'op'
    """Initial condition mode: 'op' (DC operating point) or 'ic' (use .ic)."""

    def __post_init__(self):
        """Validate all options after initialization."""
        self._validate_all()

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute with validation."""
        # Validate specific fields before setting to avoid partial state
        if name == 'nr_damping' and not (0 < value <= 1.0):
            raise ValueError(f"nr_damping must be in (0, 1], got {value}")
        if name == 'nr_convtol' and value <= 0:
            raise ValueError(f"nr_convtol must be positive, got {value}")
        if name == 'tran_lteratio' and value <= 0:
            raise ValueError(f"tran_lteratio must be positive, got {value}")
        if name == 'tran_redofactor' and value <= 1:
            raise ValueError(f"tran_redofactor must be > 1, got {value}")
        if name == 'tran_fs' and not (0 < value <= 1.0):
            raise ValueError(f"tran_fs must be in (0, 1], got {value}")
        if name == 'tran_minpts' and value < 1:
            raise ValueError(f"tran_minpts must be >= 1, got {value}")
        if name == 'reltol' and value <= 0:
            raise ValueError(f"reltol must be positive, got {value}")
        if name == 'abstol' and value <= 0:
            raise ValueError(f"abstol must be positive, got {value}")
        if name == 'gmin' and value < 0:
            raise ValueError(f"gmin must be non-negative, got {value}")
        if name == 'tran_gshunt' and value < 0:
            raise ValueError(f"tran_gshunt must be non-negative, got {value}")
        if name == 'icmode' and value not in ('op', 'ic'):
            raise ValueError(f"icmode must be 'op' or 'ic', got {value}")

        # Use object.__setattr__ to set the value
        object.__setattr__(self, name, value)

    def _validate_all(self):
        """Validate all option values."""
        # nr_damping: must be in (0, 1]
        if not (0 < self.nr_damping <= 1.0):
            raise ValueError(f"nr_damping must be in (0, 1], got {self.nr_damping}")

        # nr_convtol: must be positive
        if self.nr_convtol <= 0:
            raise ValueError(f"nr_convtol must be positive, got {self.nr_convtol}")

        # tran_lteratio: must be positive
        if self.tran_lteratio <= 0:
            raise ValueError(f"tran_lteratio must be positive, got {self.tran_lteratio}")

        # tran_redofactor: must be > 1
        if self.tran_redofactor <= 1:
            raise ValueError(f"tran_redofactor must be > 1, got {self.tran_redofactor}")

        # tran_fs: must be in (0, 1]
        if not (0 < self.tran_fs <= 1.0):
            raise ValueError(f"tran_fs must be in (0, 1], got {self.tran_fs}")

        # tran_minpts: must be positive integer
        if self.tran_minpts < 1:
            raise ValueError(f"tran_minpts must be >= 1, got {self.tran_minpts}")

        # Tolerances: must be positive
        if self.reltol <= 0:
            raise ValueError(f"reltol must be positive, got {self.reltol}")
        if self.abstol <= 0:
            raise ValueError(f"abstol must be positive, got {self.abstol}")

        # gmin/tran_gshunt: must be non-negative
        if self.gmin < 0:
            raise ValueError(f"gmin must be non-negative, got {self.gmin}")
        if self.tran_gshunt < 0:
            raise ValueError(f"tran_gshunt must be non-negative, got {self.tran_gshunt}")

        # icmode: must be valid
        if self.icmode not in ('op', 'ic'):
            raise ValueError(f"icmode must be 'op' or 'ic', got {self.icmode}")

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
            value = str(value).strip('"\'')
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

    def update_from_netlist(self, opts: Dict[str, Any], parse_number: callable = float) -> None:
        """Update options from netlist options directive.

        Args:
            opts: Dictionary from OptionsDirective.params
            parse_number: Function to parse SPICE numbers (e.g., '1u' -> 1e-6)
        """
        # Map of netlist option names to our field names (if different)
        name_map = {
            # All names are the same currently, but this allows for aliases
        }

        for opt_name, opt_value in opts.items():
            # Map name if needed
            field_name = name_map.get(opt_name, opt_name)

            # Skip unknown options (might be for other tools)
            if not hasattr(self, field_name):
                continue

            # Parse numeric values using SPICE number parser
            try:
                if field_name in ('step', 'stop', 'maxstep', 'nr_damping', 'nr_convtol',
                                  'tran_lteratio', 'tran_redofactor', 'tran_fs',
                                  'reltol', 'abstol', 'tran_gshunt', 'gmin'):
                    opt_value = parse_number(opt_value)
                elif field_name == 'tran_minpts':
                    opt_value = int(parse_number(opt_value))

                self.set(field_name, opt_value)
            except (ValueError, TypeError) as e:
                # Log warning but don't fail - allows forward compatibility
                import logging
                logging.getLogger('jax_spice').warning(
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
            result[f.name] = value
        return result

    def copy(self) -> 'SimulationOptions':
        """Create a copy of these options.

        Returns:
            New SimulationOptions instance with same values
        """
        return SimulationOptions(**{f.name: getattr(self, f.name) for f in fields(self)})
