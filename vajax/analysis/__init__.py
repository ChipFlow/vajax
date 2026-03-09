"""Analysis engines for VAJAX

Provides DC operating point, transient, and AC (small-signal) analysis
using OpenVAF-compiled devices.
"""

# Public API - main entry points
# AC analysis
from vajax.analysis.ac import ACConfig, ACResult, generate_frequencies

# Corner analysis
from vajax.analysis.corners import (
    CornerConfig,
    CornerResult,
    CornerSweepResult,
    create_pvt_corners,
    create_standard_corners,
)
from vajax.analysis.engine import CircuitEngine, DCSweepResult, TransientResult, warmup_models

# GPU backend
from vajax.analysis.gpu_backend import BackendConfig, is_gpu_available, select_backend

# Harmonic Balance analysis
from vajax.analysis.hb import HBConfig, HBResult

# Noise analysis
from vajax.analysis.noise import NoiseConfig, NoiseResult

# Transient configuration
from vajax.analysis.transient import AdaptiveConfig

# Transfer function analyses
from vajax.analysis.xfer import (
    ACXFConfig,
    ACXFResult,
    DCIncConfig,
    DCIncResult,
    DCXFConfig,
    DCXFResult,
)

__all__ = [
    # Core
    "CircuitEngine",
    "DCSweepResult",
    "TransientResult",
    "warmup_models",
    # Transient configuration
    "AdaptiveConfig",
    # AC analysis
    "ACConfig",
    "ACResult",
    "generate_frequencies",
    # Noise analysis
    "NoiseConfig",
    "NoiseResult",
    # Transfer function analyses
    "DCIncConfig",
    "DCIncResult",
    "DCXFConfig",
    "DCXFResult",
    "ACXFConfig",
    "ACXFResult",
    # Corner analysis
    "CornerConfig",
    "CornerResult",
    "CornerSweepResult",
    "create_pvt_corners",
    "create_standard_corners",
    # GPU backend
    "BackendConfig",
    "is_gpu_available",
    "select_backend",
    # Harmonic Balance
    "HBConfig",
    "HBResult",
]
