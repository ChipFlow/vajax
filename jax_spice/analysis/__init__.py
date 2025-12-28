"""Analysis engines for JAX-SPICE

Provides DC operating point and transient analysis using OpenVAF-compiled devices.
"""

from jax_spice.analysis.context import AnalysisContext
from jax_spice.analysis.engine import CircuitEngine
from jax_spice.analysis.mna import DeviceInfo, DeviceType, eval_param_simple
from jax_spice.analysis.transient import (
    TransientStrategy,
    TransientSetup,
    PythonLoopStrategy,
    ScanStrategy,
)
from jax_spice.analysis.sparse import sparse_solve, sparse_solve_csr
from jax_spice.analysis.gpu_backend import (
    select_backend,
    is_gpu_available,
    BackendConfig,
)
from jax_spice.analysis.solver import (
    NRConfig,
    NRResult,
    newton_solve,
    newton_solve_with_system,
)
from jax_spice.analysis.homotopy import (
    HomotopyConfig,
    HomotopyResult,
    run_homotopy_chain,
    gmin_stepping,
    source_stepping,
)

__all__ = [
    "AnalysisContext",
    "CircuitEngine",
    # MNA types
    "DeviceInfo",
    "DeviceType",
    "eval_param_simple",
    # Transient strategies
    "TransientStrategy",
    "TransientSetup",
    "PythonLoopStrategy",
    "ScanStrategy",
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
]
