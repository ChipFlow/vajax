"""Debug utilities for JAX-SPICE."""

from jax_spice.debug.trace_monitor import (
    TraceScope,
    get_trace_counts,
    monitor_dict,
    print_traces,
    report_traces,
    reset_traces,
    trace_monitor,
)

from jax_spice.debug.jacobian import (
    JacobianComparison,
    compare_jacobians,
    jax_to_dense_jacobian,
    osdi_to_dense_jacobian,
    print_jacobian_structure,
)

from jax_spice.debug.mir_tracer import (
    MIRTracer,
    ValueInfo,
    trace_model,
)

from jax_spice.debug.param_analyzer import (
    ParamAnalyzer,
    ParamInfo,
    KindSummary,
    analyze_model,
)

# MIR analysis imports are optional (require networkx/pydot)
try:
    from jax_spice.debug.mir_analysis import (
        analyze_phi_block,
        find_branch_points,
        find_paths_to_block,
        find_phi_blocks,
        generate_mir_dot,
        get_cfg_summary,
        load_mir_cfg,
    )

    _HAS_MIR_ANALYSIS = True
except ImportError:
    _HAS_MIR_ANALYSIS = False

__all__ = [
    # Trace monitoring
    "trace_monitor",
    "reset_traces",
    "get_trace_counts",
    "report_traces",
    "print_traces",
    "TraceScope",
    "monitor_dict",
    # Jacobian comparison
    "JacobianComparison",
    "compare_jacobians",
    "osdi_to_dense_jacobian",
    "jax_to_dense_jacobian",
    "print_jacobian_structure",
    # MIR value tracing
    "MIRTracer",
    "ValueInfo",
    "trace_model",
    # Param analysis
    "ParamAnalyzer",
    "ParamInfo",
    "KindSummary",
    "analyze_model",
    # MIR analysis (optional)
    "generate_mir_dot",
    "load_mir_cfg",
    "find_phi_blocks",
    "find_branch_points",
    "get_cfg_summary",
    "find_paths_to_block",
    "analyze_phi_block",
]
