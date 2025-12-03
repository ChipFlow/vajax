"""JAX Performance Profiler for CPU vs GPU benchmarking

Provides utilities to measure:
- Execution time on different backends (CPU vs GPU)
- Host<->Device data transfer overhead
- JIT compilation time
- Memory allocation patterns
"""

import time
import functools
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Dict, List
from contextlib import contextmanager
import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class TimingResult:
    """Result from a single timing measurement"""
    name: str
    backend: str
    wall_time_ms: float
    jit_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    transfer_time_ms: float = 0.0
    n_iterations: int = 1

    @property
    def time_per_iter_ms(self) -> float:
        return self.wall_time_ms / self.n_iterations

    def __repr__(self) -> str:
        return (f"TimingResult({self.name}, {self.backend}, "
                f"wall={self.wall_time_ms:.3f}ms, "
                f"jit={self.jit_time_ms:.3f}ms, "
                f"exec={self.execution_time_ms:.3f}ms, "
                f"transfer={self.transfer_time_ms:.3f}ms)")


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results"""
    name: str
    results: List[TimingResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: TimingResult):
        self.results.append(result)

    def get_by_backend(self, backend: str) -> List[TimingResult]:
        return [r for r in self.results if r.backend == backend]

    def cpu_vs_gpu_speedup(self) -> Optional[float]:
        """Calculate GPU speedup over CPU (exec time only, excluding JIT)"""
        cpu_results = self.get_by_backend('cpu')
        gpu_results = self.get_by_backend('gpu')

        if not cpu_results or not gpu_results:
            return None

        cpu_time = cpu_results[0].execution_time_ms
        gpu_time = gpu_results[0].execution_time_ms

        if gpu_time == 0:
            return float('inf')
        return cpu_time / gpu_time

    def summary(self) -> str:
        """Generate a human-readable summary"""
        lines = [f"\n{'='*60}", f"Benchmark: {self.name}", f"{'='*60}"]

        if self.metadata:
            lines.append("\nMetadata:")
            for k, v in self.metadata.items():
                lines.append(f"  {k}: {v}")

        lines.append("\nResults:")
        for r in self.results:
            lines.append(f"\n  [{r.backend.upper()}] {r.name}")
            lines.append(f"    Wall time:     {r.wall_time_ms:>10.3f} ms")
            if r.jit_time_ms > 0:
                lines.append(f"    JIT compile:   {r.jit_time_ms:>10.3f} ms")
            lines.append(f"    Execution:     {r.execution_time_ms:>10.3f} ms")
            if r.transfer_time_ms > 0:
                lines.append(f"    Data transfer: {r.transfer_time_ms:>10.3f} ms")
            if r.n_iterations > 1:
                lines.append(f"    Per iteration: {r.time_per_iter_ms:>10.3f} ms")

        speedup = self.cpu_vs_gpu_speedup()
        if speedup is not None:
            lines.append(f"\nGPU Speedup: {speedup:.2f}x")

        lines.append(f"{'='*60}\n")
        return "\n".join(lines)


class JAXProfiler:
    """Profiler for JAX operations with CPU/GPU comparison"""

    def __init__(self, warmup_iterations: int = 2, benchmark_iterations: int = 10):
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self._current_report: Optional[BenchmarkReport] = None

    def get_available_backends(self) -> List[str]:
        """Get list of available JAX backends"""
        backends = []

        # Check for CPU backend explicitly (may not be in default devices)
        try:
            cpu_devices = jax.devices('cpu')
            if cpu_devices:
                backends.append('cpu')
        except RuntimeError:
            pass

        # Check for GPU/CUDA backend in default devices
        for device in jax.devices():
            platform = device.platform.lower()
            if platform in ('gpu', 'cuda'):
                backends.append('gpu')
                break

        return backends

    def _get_device_for_backend(self, backend: str):
        """Get a device for the specified backend"""
        if backend == 'cpu':
            try:
                cpu_devices = jax.devices('cpu')
                if cpu_devices:
                    return cpu_devices[0]
            except RuntimeError:
                pass
            raise RuntimeError("No CPU device found")

        elif backend == 'gpu':
            for device in jax.devices():
                platform = device.platform.lower()
                if platform in ('gpu', 'cuda'):
                    return device
            raise RuntimeError("No GPU device found")

        raise RuntimeError(f"Unknown backend: {backend}")

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about available devices"""
        info = {
            'default_backend': jax.default_backend(),
            'devices': {}
        }

        # Get CPU devices explicitly
        try:
            cpu_devs = jax.devices('cpu')
            info['devices']['cpu'] = [
                {'id': d.id, 'platform': d.platform, 'device_kind': d.device_kind}
                for d in cpu_devs
            ]
        except RuntimeError:
            pass

        # Get GPU devices from default devices
        gpu_devices = []
        for d in jax.devices():
            if d.platform.lower() in ('gpu', 'cuda'):
                gpu_devices.append({
                    'id': d.id,
                    'platform': d.platform,
                    'device_kind': d.device_kind,
                })
        if gpu_devices:
            info['devices']['gpu'] = gpu_devices

        return info

    @contextmanager
    def benchmark(self, name: str, **metadata):
        """Context manager for creating a benchmark report"""
        self._current_report = BenchmarkReport(name=name, metadata=metadata)
        try:
            yield self._current_report
        finally:
            self._current_report = None

    def time_function(
        self,
        fn: Callable,
        args: tuple = (),
        kwargs: dict = None,
        name: str = "function",
        backend: str = None,
        include_jit: bool = True,
    ) -> TimingResult:
        """Time a JAX function with detailed breakdown

        Args:
            fn: Function to benchmark
            args: Positional arguments
            kwargs: Keyword arguments
            name: Name for this timing
            backend: 'cpu', 'gpu', or None for default
            include_jit: Whether to measure JIT time separately

        Returns:
            TimingResult with timing breakdown
        """
        kwargs = kwargs or {}

        if backend is None:
            backend = jax.default_backend()
            # Normalize cuda -> gpu for consistency
            if backend == 'cuda':
                backend = 'gpu'

        # Move input data to target device
        device = self._get_device_for_backend(backend)
        transfer_start = time.perf_counter()

        device_args = jax.tree.map(
            lambda x: jax.device_put(x, device) if isinstance(x, (jnp.ndarray, np.ndarray)) else x,
            args
        )
        device_kwargs = jax.tree.map(
            lambda x: jax.device_put(x, device) if isinstance(x, (jnp.ndarray, np.ndarray)) else x,
            kwargs
        )

        # Block until transfer complete
        jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, device_args)
        jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, device_kwargs)

        transfer_time = (time.perf_counter() - transfer_start) * 1000

        # JIT compilation (first call)
        jit_time = 0.0
        if include_jit:
            jit_start = time.perf_counter()
            result = fn(*device_args, **device_kwargs)
            # Block until computation complete
            jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, result)
            jit_time = (time.perf_counter() - jit_start) * 1000

        # Warmup iterations (JIT already done)
        for _ in range(self.warmup_iterations):
            result = fn(*device_args, **device_kwargs)
            jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, result)

        # Benchmark iterations
        exec_start = time.perf_counter()
        for _ in range(self.benchmark_iterations):
            result = fn(*device_args, **device_kwargs)
            jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, result)
        exec_time = (time.perf_counter() - exec_start) * 1000

        wall_time = transfer_time + jit_time + exec_time

        timing = TimingResult(
            name=name,
            backend=backend,
            wall_time_ms=wall_time,
            jit_time_ms=jit_time,
            execution_time_ms=exec_time,
            transfer_time_ms=transfer_time,
            n_iterations=self.benchmark_iterations,
        )

        if self._current_report is not None:
            self._current_report.add_result(timing)

        return timing

    def compare_backends(
        self,
        fn: Callable,
        args: tuple = (),
        kwargs: dict = None,
        name: str = "comparison",
        backends: List[str] = None,
    ) -> BenchmarkReport:
        """Compare function execution across different backends

        Args:
            fn: Function to benchmark
            args: Positional arguments
            kwargs: Keyword arguments
            name: Name for this comparison
            backends: List of backends to compare, or None for all available

        Returns:
            BenchmarkReport with results for each backend
        """
        kwargs = kwargs or {}
        backends = backends or self.get_available_backends()

        report = BenchmarkReport(
            name=name,
            metadata={
                'warmup_iterations': self.warmup_iterations,
                'benchmark_iterations': self.benchmark_iterations,
                **self.get_device_info(),
            }
        )

        for backend in backends:
            try:
                result = self.time_function(
                    fn, args, kwargs,
                    name=f"{name}_{backend}",
                    backend=backend,
                )
                report.add_result(result)
            except Exception as e:
                print(f"Warning: Failed to benchmark on {backend}: {e}")

        return report


def measure_transfer_overhead(
    array_sizes: List[int] = None,
    dtype=jnp.float64,
) -> Dict[str, List[Dict[str, float]]]:
    """Measure CPU<->GPU transfer overhead for different array sizes

    Args:
        array_sizes: List of array sizes to test
        dtype: Data type for arrays

    Returns:
        Dictionary with transfer timing data
    """
    if array_sizes is None:
        array_sizes = [100, 1000, 10000, 100000, 1000000]

    results = {
        'sizes': array_sizes,
        'cpu_to_gpu': [],
        'gpu_to_cpu': [],
        'bytes': [],
    }

    # Find GPU device from default devices
    gpu_device = None
    for device in jax.devices():
        platform = device.platform.lower()
        if platform in ('gpu', 'cuda'):
            gpu_device = device
            break

    if gpu_device is None:
        print("GPU not available for transfer measurement")
        return results

    # Get CPU device explicitly
    cpu_device = None
    try:
        cpu_devices = jax.devices('cpu')
        if cpu_devices:
            cpu_device = cpu_devices[0]
    except RuntimeError:
        pass

    if cpu_device is None:
        print("CPU device not available")
        return results

    for size in array_sizes:
        # Create array on CPU
        cpu_array = jax.device_put(jnp.ones(size, dtype=dtype), cpu_device)
        cpu_array.block_until_ready()

        # Measure CPU -> GPU transfer
        start = time.perf_counter()
        gpu_array = jax.device_put(cpu_array, gpu_device)
        gpu_array.block_until_ready()
        cpu_to_gpu_time = (time.perf_counter() - start) * 1000

        # Measure GPU -> CPU transfer
        start = time.perf_counter()
        cpu_result = jax.device_put(gpu_array, cpu_device)
        cpu_result.block_until_ready()
        gpu_to_cpu_time = (time.perf_counter() - start) * 1000

        bytes_transferred = size * dtype.dtype.itemsize

        results['cpu_to_gpu'].append({
            'time_ms': cpu_to_gpu_time,
            'bandwidth_gb_s': (bytes_transferred / cpu_to_gpu_time) / 1e6,
        })
        results['gpu_to_cpu'].append({
            'time_ms': gpu_to_cpu_time,
            'bandwidth_gb_s': (bytes_transferred / gpu_to_cpu_time) / 1e6,
        })
        results['bytes'].append(bytes_transferred)

    return results


def benchmark_decorator(
    name: str = None,
    warmup: int = 2,
    iterations: int = 10,
    compare_backends: bool = True,
):
    """Decorator to automatically benchmark a function

    Usage:
        @benchmark_decorator(name="my_function", compare_backends=True)
        def my_function(x, y):
            return x + y
    """
    def decorator(fn: Callable) -> Callable:
        fn_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            profiler = JAXProfiler(warmup_iterations=warmup, benchmark_iterations=iterations)

            if compare_backends:
                report = profiler.compare_backends(fn, args, kwargs, name=fn_name)
                print(report.summary())
            else:
                result = profiler.time_function(fn, args, kwargs, name=fn_name)
                print(result)

            # Actually execute and return the result
            return fn(*args, **kwargs)

        return wrapper
    return decorator


# Convenience function for quick benchmarks
def quick_benchmark(fn: Callable, *args, **kwargs) -> BenchmarkReport:
    """Quick benchmark comparing CPU and GPU execution

    Usage:
        report = quick_benchmark(my_function, arg1, arg2, kwarg1=value)
        print(report.summary())
    """
    profiler = JAXProfiler(warmup_iterations=2, benchmark_iterations=5)
    return profiler.compare_backends(fn, args, kwargs, name=fn.__name__)
