"""Profiling utilities for JAX-SPICE.

Provides decorators and context managers for CUDA (nsys) and JAX/XLA profiling.

Usage:
    from jax_spice.profiling import profile, ProfileConfig

    # Simple decorator (uses environment variables for configuration)
    @profile
    def my_function():
        ...

    # Decorator with explicit configuration
    @profile(jax=True, cuda=True, trace_dir="/tmp/traces")
    def my_function():
        ...

    # Context manager
    with profile_section("transient_analysis"):
        result = run_transient(...)

Environment Variables:
    JAX_SPICE_PROFILE_JAX: Enable JAX/XLA profiling (1 or true)
    JAX_SPICE_PROFILE_CUDA: Enable CUDA profiling (1 or true)
    JAX_SPICE_PROFILE_DIR: Directory for trace output (default: /tmp/jax-spice-traces)

For nsys profiling, run with:
    nsys-jax -o profile.zip --capture-range=cudaProfilerApi python your_script.py

For JAX profiling, traces can be viewed in:
    - Perfetto (https://ui.perfetto.dev/)
    - TensorBoard (tensorboard --logdir=/tmp/jax-spice-traces)
"""

import functools
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

import jax

from jax_spice._logging import logger

F = TypeVar('F', bound=Callable[..., Any])


def _env_bool(name: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    val = os.environ.get(name, '').lower()
    if val in ('1', 'true', 'yes', 'on'):
        return True
    if val in ('0', 'false', 'no', 'off'):
        return False
    return default


@dataclass
class ProfileConfig:
    """Configuration for profiling.

    Attributes:
        jax: Enable JAX/XLA profiling (Perfetto/TensorBoard traces)
        cuda: Enable CUDA profiling (for nsys-jax)
        trace_dir: Directory for trace output
        create_perfetto_link: Create Perfetto UI link for JAX traces
        name: Name prefix for trace files/annotations
    """
    jax: bool = field(default_factory=lambda: _env_bool('JAX_SPICE_PROFILE_JAX'))
    cuda: bool = field(default_factory=lambda: _env_bool('JAX_SPICE_PROFILE_CUDA'))
    trace_dir: str = field(default_factory=lambda: os.environ.get(
        'JAX_SPICE_PROFILE_DIR', '/tmp/jax-spice-traces'))
    create_perfetto_link: bool = False
    name: Optional[str] = None

    @property
    def enabled(self) -> bool:
        """Return True if any profiling is enabled."""
        return self.jax or self.cuda


# Global config (can be modified at runtime)
_global_config = ProfileConfig()


def get_config() -> ProfileConfig:
    """Get the global profiling configuration."""
    return _global_config


def set_config(config: ProfileConfig) -> None:
    """Set the global profiling configuration."""
    global _global_config
    _global_config = config


def enable_profiling(jax: bool = True, cuda: bool = True,
                     trace_dir: Optional[str] = None) -> None:
    """Enable profiling globally.

    Args:
        jax: Enable JAX/XLA profiling
        cuda: Enable CUDA profiling
        trace_dir: Directory for trace output
    """
    global _global_config
    _global_config.jax = jax
    _global_config.cuda = cuda
    if trace_dir:
        _global_config.trace_dir = trace_dir


def disable_profiling() -> None:
    """Disable all profiling globally."""
    global _global_config
    _global_config.jax = False
    _global_config.cuda = False


# CUDA profiler API (optional dependency)
_cuda_profiler_available = False
_cudaProfilerStart = None
_cudaProfilerStop = None

try:
    from cuda.cudart import cudaProfilerStart as _cudaProfilerStart_raw
    from cuda.cudart import cudaProfilerStop as _cudaProfilerStop_raw

    def _cudaProfilerStart():
        err, = _cudaProfilerStart_raw()
        if err.value != 0:
            logger.warning(f"cudaProfilerStart failed with error {err}")

    def _cudaProfilerStop():
        err, = _cudaProfilerStop_raw()
        if err.value != 0:
            logger.warning(f"cudaProfilerStop failed with error {err}")

    _cuda_profiler_available = True
    logger.debug("CUDA profiler API available (cuda-python)")
except ImportError:
    pass

if not _cuda_profiler_available:
    try:
        from cuda_profiler_api import cudaProfilerStart as _cudaProfilerStart
        from cuda_profiler_api import cudaProfilerStop as _cudaProfilerStop
        _cuda_profiler_available = True
        logger.debug("CUDA profiler API available (cuda-profiler-api)")
    except ImportError:
        pass


def _has_gpu() -> bool:
    """Check if GPU device is available."""
    try:
        return any(d.platform != 'cpu' for d in jax.devices())
    except Exception:
        return False


@contextmanager
def profile_section(name: str, config: Optional[ProfileConfig] = None):
    """Context manager for profiling a code section.

    Args:
        name: Name for the profiled section (used in trace annotations)
        config: Profiling configuration (uses global config if None)

    Example:
        with profile_section("solve_linear_system"):
            result = solve(A, b)
            result.block_until_ready()

    Note:
        JAX profiling works on both CPU and GPU. CUDA profiling is only
        available when running on GPU with CUDA.
    """
    cfg = config or _global_config

    if not cfg.enabled:
        yield
        return

    # JAX profiling works on both CPU and GPU
    should_jax_profile = cfg.jax

    trace_path = Path(cfg.trace_dir) / name if should_jax_profile else None
    jax_trace = None
    cuda_started = False

    try:
        # Start JAX profiling (GPU only)
        if should_jax_profile:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Starting JAX trace: {trace_path}")
            try:
                jax_trace = jax.profiler.trace(
                    str(trace_path),
                    create_perfetto_link=cfg.create_perfetto_link
                )
                jax_trace.__enter__()
            except Exception as e:
                logger.warning(f"Failed to start JAX trace: {e}")
                jax_trace = None

        # Start CUDA profiling
        if cfg.cuda and _cuda_profiler_available:
            logger.info(f"Starting CUDA profiler for: {name}")
            _cudaProfilerStart()
            cuda_started = True

        yield

    finally:
        # Stop CUDA profiling
        if cuda_started:
            # Ensure GPU operations complete before stopping
            for dev in jax.devices():
                if dev.platform != 'cpu':
                    try:
                        jax.device_put(0, dev).block_until_ready()
                    except Exception:
                        pass
            _cudaProfilerStop()
            logger.info(f"Stopped CUDA profiler for: {name}")

        # Stop JAX profiling
        if jax_trace is not None:
            jax_trace.__exit__(None, None, None)
            logger.info(f"JAX trace saved to: {trace_path}")


def profile(
    func: Optional[F] = None,
    *,
    jax: Optional[bool] = None,
    cuda: Optional[bool] = None,
    trace_dir: Optional[str] = None,
    name: Optional[str] = None,
) -> Union[F, Callable[[F], F]]:
    """Decorator for profiling a function.

    Can be used with or without arguments:

        @profile
        def my_function():
            ...

        @profile(jax=True, cuda=True)
        def my_function():
            ...

    Args:
        func: Function to decorate (when used without arguments)
        jax: Enable JAX profiling (overrides global config if set)
        cuda: Enable CUDA profiling (overrides global config if set)
        trace_dir: Directory for traces (overrides global config if set)
        name: Name for the profile (defaults to function name)

    Returns:
        Decorated function or decorator
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Build config, starting from global and overriding
            cfg = ProfileConfig(
                jax=jax if jax is not None else _global_config.jax,
                cuda=cuda if cuda is not None else _global_config.cuda,
                trace_dir=trace_dir or _global_config.trace_dir,
                create_perfetto_link=_global_config.create_perfetto_link,
                name=name or fn.__name__,
            )

            if not cfg.enabled:
                return fn(*args, **kwargs)

            with profile_section(cfg.name, cfg):
                return fn(*args, **kwargs)

        return wrapper  # type: ignore

    # Handle both @profile and @profile(...) syntax
    if func is not None:
        return decorator(func)
    return decorator


class ProfileTimer:
    """Timer that integrates with profiling for measuring code sections.

    Example:
        timer = ProfileTimer("transient_solve")
        timer.start()
        result = solve(...)
        result.block_until_ready()
        timer.stop()
        print(f"Elapsed: {timer.elapsed_ms:.2f}ms")
    """

    def __init__(self, name: str, config: Optional[ProfileConfig] = None):
        self.name = name
        self.config = config or _global_config
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._trace = None
        self._cuda_started = False

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        if self._start_time is None or self._end_time is None:
            return 0.0
        return (self._end_time - self._start_time) * 1000

    def start(self) -> 'ProfileTimer':
        """Start timing and profiling."""
        import time
        self._start_time = time.perf_counter()

        # JAX profiling works best on GPU - skip on CPU
        should_jax_profile = self.config.jax and _has_gpu()
        if should_jax_profile:
            trace_path = Path(self.config.trace_dir) / self.name
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._trace = jax.profiler.trace(
                    str(trace_path),
                    create_perfetto_link=self.config.create_perfetto_link
                )
                self._trace.__enter__()
            except Exception as e:
                logger.warning(f"Failed to start JAX trace: {e}")
                self._trace = None

        if self.config.cuda and _cuda_profiler_available:
            _cudaProfilerStart()
            self._cuda_started = True

        return self

    def stop(self) -> 'ProfileTimer':
        """Stop timing and profiling."""
        import time

        # Ensure GPU operations complete
        for dev in jax.devices():
            if dev.platform != 'cpu':
                try:
                    jax.device_put(0, dev).block_until_ready()
                except Exception:
                    pass

        self._end_time = time.perf_counter()

        if self._cuda_started:
            _cudaProfilerStop()

        if self._trace is not None:
            self._trace.__exit__(None, None, None)

        return self

    def __enter__(self) -> 'ProfileTimer':
        return self.start()

    def __exit__(self, *args) -> None:
        self.stop()
