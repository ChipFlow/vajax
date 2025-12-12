"""Analysis engines for JAX-SPICE

Provides DC operating point and transient analysis.
"""

from jax_spice.analysis.context import AnalysisContext
from jax_spice.analysis.mna import MNASystem
from jax_spice.analysis.dc import (
    dc_operating_point,
    dc_operating_point_sparse,
    dc_operating_point_gpu,
)
from jax_spice.analysis.transient import (
    transient_analysis,
    transient_analysis_jit,
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

__all__ = [
    "AnalysisContext",
    "MNASystem",
    "dc_operating_point",
    "dc_operating_point_sparse",
    "dc_operating_point_gpu",
    "transient_analysis",
    "transient_analysis_jit",
    "sparse_solve",
    "sparse_solve_csr",
    "select_backend",
    "is_gpu_available",
    "BackendConfig",
    # Unified solver
    "NRConfig",
    "NRResult",
    "newton_solve",
    "newton_solve_with_system",
]
