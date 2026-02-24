"""Analysis engines for VA-JAX

Provides DC operating point, transient, and AC (small-signal) analysis
using OpenVAF-compiled devices.
"""

from vajax.analysis.ac import (
    ACConfig,
    ACResult,
    generate_frequencies,
    run_ac_analysis,
)
from vajax.analysis.context import AnalysisContext
from vajax.analysis.corners import (
    PROCESS_CORNERS,
    TEMPERATURE_CORNERS,
    CornerConfig,
    CornerResult,
    CornerSweepResult,
    ProcessCorner,
    VoltageCorner,
    create_pvt_corners,
    create_standard_corners,
)
from vajax.analysis.engine import CircuitEngine, TransientResult, warmup_models
from vajax.analysis.gpu_backend import (
    BackendConfig,
    is_gpu_available,
    select_backend,
)
from vajax.analysis.hb import (
    HBConfig,
    HBResult,
    build_apft_matrices,
    build_collocation_points,
    build_frequency_grid,
    run_hb_analysis,
)
from vajax.analysis.homotopy import (
    HomotopyConfig,
    HomotopyResult,
    gmin_stepping,
    run_homotopy_chain,
    source_stepping,
)
from vajax.analysis.mna import DeviceInfo, DeviceType, eval_param_simple
from vajax.analysis.noise import (
    NoiseConfig,
    NoiseResult,
    NoiseSource,
    compute_flicker_noise_psd,
    compute_shot_noise_psd,
    compute_thermal_noise_psd,
    extract_noise_sources,
    run_noise_analysis,
)
from vajax.analysis.solver import (
    NRConfig,
    NRResult,
    newton_solve,
    newton_solve_with_system,
)
from vajax.analysis.sparse import sparse_solve, sparse_solve_csr
from vajax.analysis.transient import (
    AdaptiveConfig,
    FullMNAStrategy,
    TransientSetup,
    TransientStrategy,
)
from vajax.analysis.xfer import (
    # ACXF
    ACXFConfig,
    ACXFResult,
    # DCINC
    DCIncConfig,
    DCIncResult,
    # DCXF
    DCXFConfig,
    DCXFResult,
    build_dcinc_excitation,
    solve_acxf,
    solve_dcinc,
    solve_dcxf,
)

__all__ = [
    "AnalysisContext",
    "CircuitEngine",
    "TransientResult",
    "warmup_models",
    # MNA types
    "DeviceInfo",
    "DeviceType",
    "eval_param_simple",
    # Transient strategies
    "TransientStrategy",
    "TransientSetup",
    "FullMNAStrategy",
    "AdaptiveConfig",
    # AC analysis
    "ACConfig",
    "ACResult",
    "generate_frequencies",
    "run_ac_analysis",
    # Sparse solver
    "sparse_solve",
    "sparse_solve_csr",
    # GPU backend
    "select_backend",
    "is_gpu_available",
    "BackendConfig",
    # Unified solver
    "NRConfig",
    "NRResult",
    "newton_solve",
    "newton_solve_with_system",
    # Homotopy
    "HomotopyConfig",
    "HomotopyResult",
    "run_homotopy_chain",
    "gmin_stepping",
    "source_stepping",
    # Transfer function analyses
    "DCIncConfig",
    "DCIncResult",
    "solve_dcinc",
    "build_dcinc_excitation",
    "DCXFConfig",
    "DCXFResult",
    "solve_dcxf",
    "ACXFConfig",
    "ACXFResult",
    "solve_acxf",
    # Noise analysis
    "NoiseConfig",
    "NoiseResult",
    "NoiseSource",
    "run_noise_analysis",
    "extract_noise_sources",
    "compute_thermal_noise_psd",
    "compute_shot_noise_psd",
    "compute_flicker_noise_psd",
    # Harmonic Balance analysis
    "HBConfig",
    "HBResult",
    "build_frequency_grid",
    "build_collocation_points",
    "build_apft_matrices",
    "run_hb_analysis",
    # Corner analysis
    "ProcessCorner",
    "VoltageCorner",
    "CornerConfig",
    "CornerResult",
    "CornerSweepResult",
    "PROCESS_CORNERS",
    "TEMPERATURE_CORNERS",
    "create_standard_corners",
    "create_pvt_corners",
]
