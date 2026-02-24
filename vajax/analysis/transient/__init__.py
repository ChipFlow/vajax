"""Transient analysis for VA-JAX.

This module provides transient analysis using full Modified Nodal Analysis (MNA)
with explicit branch currents for voltage sources. This provides:

- Better numerical conditioning (no G=1e12 high-G approximation)
- More accurate current extraction (branch currents are primary unknowns)
- Smoother dI/dt transitions matching VACASK reference

Usage:
    from vajax.analysis.transient import (
        FullMNAStrategy, AdaptiveConfig, extract_results
    )

    # Run simulation - returns full arrays for zero-copy performance
    config = AdaptiveConfig(lte_ratio=3.5, redo_factor=2.5)
    strategy = FullMNAStrategy(runner, config=config)
    times, V_out, stats = strategy.run(t_stop=1e-6, dt=1e-9)

    # Extract sliced results for plotting (converts to numpy)
    times_np, voltages, currents = extract_results(times, V_out, stats)
    V_out_node = voltages['out']
    I_VDD = currents['vdd']
"""

from .adaptive import (
    AdaptiveConfig,
    # Shared LTE functions
    compute_lte_timestep_jax,
    predict_voltage_jax,
)
from .base import TransientSetup, TransientStrategy
from .full_mna import FullMNAStrategy, extract_results

__all__ = [
    "TransientStrategy",
    "TransientSetup",
    "FullMNAStrategy",
    "AdaptiveConfig",
    "extract_results",
    # Shared LTE functions
    "compute_lte_timestep_jax",
    "predict_voltage_jax",
]
