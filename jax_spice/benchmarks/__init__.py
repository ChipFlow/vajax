"""Circuit simulation and benchmarking utilities.

Exports:
    CircuitEngine: Core circuit simulation engine
    BenchmarkRunner: Benchmark runner with timing utilities
"""

from .runner import CircuitEngine, BenchmarkRunner

__all__ = ['CircuitEngine', 'BenchmarkRunner']
