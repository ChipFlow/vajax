"""Transient analysis for JAX-SPICE.

This module provides transient analysis capabilities using OpenVAF-compiled devices:

Strategy classes for transient simulation:

- PythonLoopStrategy: Traditional Python for-loop with JIT-compiled NR solver
  - Full convergence tracking per timestep
  - Easy to debug and profile
  - Moderate performance (~0.5ms/step)

- ScanStrategy: Fully JIT-compiled using lax.scan
  - Best performance (~0.1ms/step on CPU)
  - 5x+ speedup over Python loop
  - Limited per-step debugging

- AdaptiveStrategy: LTE-based adaptive timestep control
  - Automatically adjusts timestep based on solution accuracy
  - Smaller steps during fast transients, larger steps during slow evolution
  - Uses predictor-corrector scheme with polynomial extrapolation
  - Compatible with VACASK for validation

Strategy Usage:
    from jax_spice.analysis.transient import PythonLoopStrategy, ScanStrategy

    # Using Python loop (default, more debugging info)
    strategy = PythonLoopStrategy(runner, use_sparse=False)
    times, voltages, stats = strategy.run(t_stop=1e-6, dt=1e-9)

    # Using lax.scan (faster, but requires matching warmup steps)
    strategy = ScanStrategy(runner, use_sparse=False)
    times, voltages, stats = strategy.run(t_stop=1e-6, dt=1e-9)

    # Using adaptive timestep (best accuracy, automatic step sizing)
    from jax_spice.analysis.transient import AdaptiveStrategy, AdaptiveConfig
    config = AdaptiveConfig(lte_ratio=3.5, redo_factor=2.5)
    strategy = AdaptiveStrategy(runner, config=config)
    times, voltages, stats = strategy.run(t_stop=1e-6, dt=1e-9)
"""

# Strategy classes for OpenVAF-based transient
from .adaptive import AdaptiveConfig, AdaptiveStrategy
from .base import TransientSetup, TransientStrategy
from .python_loop import PythonLoopStrategy
from .scan import ScanStrategy

__all__ = [
    # Strategy classes for OpenVAF
    "TransientStrategy",
    "TransientSetup",
    "PythonLoopStrategy",
    "ScanStrategy",
    "AdaptiveStrategy",
    "AdaptiveConfig",
]
