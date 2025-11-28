"""Analysis context for JAX-SPICE

Provides context information to device models during evaluation, allowing
them to modify behavior based on the type of analysis being performed.

For example, OSDI models with ddt() terms (like reverse recovery) should
skip dynamic contributions during DC analysis to avoid convergence issues.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
from jax import Array


class AnalysisType(Enum):
    """Type of circuit analysis being performed"""
    DC = auto()           # DC operating point or DC sweep
    AC = auto()           # Small-signal AC analysis (future)
    TRANSIENT = auto()    # Time-domain transient analysis (future)


@dataclass(frozen=True)
class AnalysisContext:
    """Context information passed to devices during evaluation

    This provides devices with information about the current analysis type
    and any relevant parameters. Devices can use this to modify their
    behavior - for example, skipping ddt() terms during DC analysis.

    Attributes:
        analysis_type: The type of analysis being performed
        time: Current simulation time (for transient analysis, None for DC/AC)
        time_step: Current time step dt (for transient, None otherwise)
        frequency: Analysis frequency (for AC analysis, None otherwise)
        temperature: Circuit temperature in Kelvin (default 300K = 27C)
    """
    analysis_type: AnalysisType
    time: Optional[Array] = None
    time_step: Optional[Array] = None
    frequency: Optional[Array] = None
    temperature: float = 300.0

    @property
    def is_dc(self) -> bool:
        """True if this is a DC analysis (skip dynamic terms)"""
        return self.analysis_type == AnalysisType.DC

    @property
    def is_transient(self) -> bool:
        """True if this is transient analysis (include ddt terms)"""
        return self.analysis_type == AnalysisType.TRANSIENT

    @property
    def is_ac(self) -> bool:
        """True if this is AC small-signal analysis"""
        return self.analysis_type == AnalysisType.AC

    @classmethod
    def dc(cls, temperature: float = 300.0) -> "AnalysisContext":
        """Create a DC analysis context

        Args:
            temperature: Circuit temperature in Kelvin (default 300K)

        Returns:
            AnalysisContext configured for DC analysis
        """
        return cls(analysis_type=AnalysisType.DC, temperature=temperature)

    @classmethod
    def transient(
        cls,
        time: Array,
        time_step: Array,
        temperature: float = 300.0
    ) -> "AnalysisContext":
        """Create a transient analysis context

        Args:
            time: Current simulation time
            time_step: Time step (dt) for ddt() calculations
            temperature: Circuit temperature in Kelvin

        Returns:
            AnalysisContext configured for transient analysis
        """
        return cls(
            analysis_type=AnalysisType.TRANSIENT,
            time=time,
            time_step=time_step,
            temperature=temperature
        )

    @classmethod
    def ac(cls, frequency: Array, temperature: float = 300.0) -> "AnalysisContext":
        """Create an AC analysis context

        Args:
            frequency: Analysis frequency in Hz
            temperature: Circuit temperature in Kelvin

        Returns:
            AnalysisContext configured for AC analysis
        """
        return cls(
            analysis_type=AnalysisType.AC,
            frequency=frequency,
            temperature=temperature
        )
