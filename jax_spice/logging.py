"""Logging configuration for JAX-SPICE.

Provides two logging modes:
- Default: WARNING level only (quiet)
- Performance tracing: INFO level with flush and memory stats (for Cloud Run visibility)

Usage:
    from jax_spice.logging import logger, enable_performance_logging

    # Default - only warnings
    logger.warning("This will show")
    logger.info("This won't show")

    # Enable for performance tracing (e.g., Cloud Run)
    enable_performance_logging()
    logger.info("Now this shows with memory stats and flushes immediately")

    # Enable with perf_counter timestamps (for correlating with Perfetto traces)
    enable_performance_logging(with_perf_counter=True)
    logger.info("Now shows: [1234.567890] message")
"""

import logging
import sys
import time
import tracemalloc

import jax

# Create the jax_spice logger
logger = logging.getLogger("jax_spice")

# Default: WARNING level only (quiet operation)
logger.setLevel(logging.WARNING)

# Add a default handler if none exists
if not logger.handlers:
    _default_handler = logging.StreamHandler(sys.stdout)
    _default_handler.setLevel(logging.WARNING)
    _default_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_default_handler)


def _get_memory_stats() -> str:
    """Get GPU and CPU memory stats string."""
    parts = []

    # CPU memory via tracemalloc
    if tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        parts.append(f"CPU:{current/1024/1024:.0f}MB")

    # GPU memory via JAX (works for CUDA devices)
    try:
        for dev in jax.devices():
            # Check for GPU/CUDA devices (platform can be 'gpu', 'cuda', etc.)
            if dev.platform != 'cpu':
                stats = dev.memory_stats()
                if stats and 'bytes_in_use' in stats:
                    current_mb = stats['bytes_in_use'] / 1024 / 1024
                    parts.append(f"{dev.platform}:{current_mb:.0f}MB")
    except Exception:
        pass

    return f"[{' '.join(parts)}]" if parts else ""


class FlushingHandler(logging.StreamHandler):
    """StreamHandler that flushes after every emit (for Cloud Run log visibility)."""

    def emit(self, record):
        super().emit(record)
        self.flush()


class MemoryLoggingHandler(logging.StreamHandler):
    """StreamHandler that prepends memory stats and flushes after every emit."""

    def emit(self, record):
        # Prepend memory stats to the message
        mem_stats = _get_memory_stats()
        if mem_stats:
            record.msg = f"{mem_stats} {record.msg}"
        super().emit(record)
        self.flush()


class PerfCounterHandler(logging.StreamHandler):
    """StreamHandler that prepends time.perf_counter() and flushes after every emit.

    Useful for correlating log messages with Perfetto trace timestamps.
    """

    def __init__(self, stream=None, with_memory: bool = False):
        super().__init__(stream)
        self.with_memory = with_memory

    def emit(self, record):
        # Prepend memory stats if enabled
        if self.with_memory:
            mem_stats = _get_memory_stats()
            if mem_stats:
                record.msg = f"{mem_stats} [{time.perf_counter():.6f}] {record.msg}"
            else:
                record.msg = f"[{time.perf_counter():.6f}] {record.msg}"
        else:
            record.msg = f"[{time.perf_counter():.6f}] {record.msg}"
        super().emit(record)
        self.flush()


def enable_performance_logging(with_memory: bool = True, with_perf_counter: bool = False):
    """Enable DEBUG level logging with immediate flush for performance tracing.

    Use this when running on Cloud Run or when you need to see logs
    in real-time during long-running operations.

    Args:
        with_memory: If True, prepend CPU/GPU memory stats to each log line.
        with_perf_counter: If True, prepend time.perf_counter() timestamps.
            Useful for correlating log messages with Perfetto trace timestamps.
    """
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Start tracemalloc for CPU memory tracking
    if with_memory and not tracemalloc.is_tracing():
        tracemalloc.start()

    # Add appropriate handler
    if with_perf_counter:
        handler = PerfCounterHandler(sys.stdout, with_memory=with_memory)
    elif with_memory:
        handler = MemoryLoggingHandler(sys.stdout)
    else:
        handler = FlushingHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def set_log_level(level: int):
    """Set the logging level.

    Args:
        level: logging.DEBUG, logging.INFO, logging.WARNING, etc.
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
