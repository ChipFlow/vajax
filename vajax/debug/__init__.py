"""Debug utilities for VA-JAX."""

from vajax.debug.jacobian import (
    JacobianComparison,
    compare_jacobians,
    jax_to_dense_jacobian,
    osdi_to_dense_jacobian,
    print_jacobian_structure,
)
from vajax.debug.mir_inspector import (
    MIRInspector,
    ParamSummary,
    PHIInfo,
    inspect_model,
)
from vajax.debug.mir_tracer import (
    MIRTracer,
    ValueInfo,
    trace_model,
)
from vajax.debug.model_comparison import (
    CacheAnalysis,
    ComparisonResult,
    ModelComparator,
    quick_compare,
)
from vajax.debug.param_analyzer import (
    KindSummary,
    ParamAnalyzer,
    ParamInfo,
    analyze_model,
)
from vajax.debug.simulation_tracer import (
    DeviceParamsTrace,
    NodeAllocation,
    SimulationTracer,
    VoltageMapping,
    trace_simulation,
)
from vajax.debug.trace_monitor import (
    TraceScope,
    get_trace_counts,
    monitor_dict,
    print_traces,
    report_traces,
    reset_traces,
    trace_monitor,
)
from vajax.debug.transient_diagnostics import (
    ConvergenceSweepResult,
    StepRecord,
    StepTraceSummary,
    VACASKStepRecord,
    capture_step_trace,
    convergence_sweep,
    parse_debug_output,
    parse_vacask_debug_output,
    print_step_summary,
)

# MIR analysis imports are optional (require networkx/pydot)
try:
    from vajax.debug.mir_analysis import (
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

# MIR graph imports are optional (require networkx)
try:
    from vajax.debug.mir_graph import MIRGraph

    _HAS_MIR_GRAPH = True
except ImportError:
    _HAS_MIR_GRAPH = False

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
    # Model comparison
    "ModelComparator",
    "ComparisonResult",
    "CacheAnalysis",
    "quick_compare",
    # MIR inspection
    "MIRInspector",
    "ParamSummary",
    "PHIInfo",
    "inspect_model",
    # Simulation tracing
    "SimulationTracer",
    "NodeAllocation",
    "VoltageMapping",
    "DeviceParamsTrace",
    "trace_simulation",
    # MIR analysis (optional)
    "generate_mir_dot",
    "load_mir_cfg",
    "find_phi_blocks",
    "find_branch_points",
    "get_cfg_summary",
    "find_paths_to_block",
    "analyze_phi_block",
    # MIR graph (optional)
    "MIRGraph",
    # Transient diagnostics
    "StepRecord",
    "StepTraceSummary",
    "VACASKStepRecord",
    "ConvergenceSweepResult",
    "parse_debug_output",
    "capture_step_trace",
    "parse_vacask_debug_output",
    "convergence_sweep",
    "print_step_summary",
]
