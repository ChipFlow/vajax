"""Debug utility to detect JAX function retracing.

Inspired by equinox.debug.assert_max_traces.
Use this to find which functions are being retraced unexpectedly.

Usage:
    from vajax.debug.trace_monitor import trace_monitor, report_traces

    # Wrap a function to monitor its traces
    @trace_monitor("my_function")
    def my_function(x):
        return x * 2

    # Or wrap inline
    monitored_fn = trace_monitor("build_system")(build_system_fn)

    # After running, print report
    report_traces()
"""

import functools
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Global storage for trace counts
_trace_counts: Dict[str, int] = defaultdict(int)
_trace_shapes: Dict[str, List[Tuple[Any, ...]]] = defaultdict(list)
_trace_warnings: Dict[str, bool] = defaultdict(bool)


def _get_shape_signature(args: tuple, kwargs: dict) -> Tuple[Any, ...]:
    """Extract shape signature from arguments for comparison."""

    def get_shape(x):
        if hasattr(x, "shape"):
            return ("array", x.shape, x.dtype)
        elif isinstance(x, (list, tuple)):
            return (type(x).__name__, tuple(get_shape(item) for item in x))
        elif isinstance(x, dict):
            return ("dict", tuple((k, get_shape(v)) for k, v in sorted(x.items())))
        elif callable(x):
            return ("callable", id(x))  # Different function objects = different trace
        else:
            return (type(x).__name__, x)

    arg_shapes = tuple(get_shape(arg) for arg in args)
    kwarg_shapes = tuple((k, get_shape(v)) for k, v in sorted(kwargs.items()))
    return (arg_shapes, kwarg_shapes)


def trace_monitor(
    name: str,
    max_traces: int = 5,
    warn_on_retrace: bool = True,
) -> Callable:
    """Decorator to monitor how many times a function is traced.

    Args:
        name: Identifier for this function in reports
        max_traces: Warn after this many traces (default 5)
        warn_on_retrace: If True, log warning when same shapes cause retrace

    Returns:
        Decorator that wraps the function with trace monitoring
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # This code runs at trace time (not execution time)
            shape_sig = _get_shape_signature(args, kwargs)

            _trace_counts[name] += 1
            count = _trace_counts[name]

            # Check if we've seen these shapes before
            seen_before = shape_sig in _trace_shapes[name]
            _trace_shapes[name].append(shape_sig)

            if count > max_traces and not _trace_warnings[name]:
                _trace_warnings[name] = True
                logger.warning(
                    f"RETRACE WARNING: '{name}' traced {count} times (max_traces={max_traces})"
                )
                if seen_before and warn_on_retrace:
                    logger.warning(
                        "  Same shape signature seen before - possible unnecessary retrace!"
                    )

            return fn(*args, **kwargs)

        return wrapper

    return decorator


def reset_traces():
    """Reset all trace counters."""
    _trace_counts.clear()
    _trace_shapes.clear()
    _trace_warnings.clear()


def get_trace_counts() -> Dict[str, int]:
    """Get current trace counts for all monitored functions."""
    return dict(_trace_counts)


def report_traces(min_traces: int = 1) -> str:
    """Generate a report of trace counts.

    Args:
        min_traces: Only include functions traced at least this many times

    Returns:
        Formatted report string
    """
    lines = ["=== JAX Trace Report ==="]

    if not _trace_counts:
        lines.append("No traces recorded.")
        return "\n".join(lines)

    # Sort by count descending
    sorted_counts = sorted(_trace_counts.items(), key=lambda x: x[1], reverse=True)

    for name, count in sorted_counts:
        if count >= min_traces:
            unique_shapes = len(set(_trace_shapes[name]))
            lines.append(f"  {name}: {count} traces ({unique_shapes} unique signatures)")

    total = sum(_trace_counts.values())
    lines.append(f"\nTotal traces: {total}")

    return "\n".join(lines)


def print_traces(min_traces: int = 1):
    """Print trace report to stdout."""
    print(report_traces(min_traces))


# Context manager for scoped monitoring
class TraceScope:
    """Context manager to monitor traces within a scope.

    Usage:
        with TraceScope("transient_run"):
            engine.run_transient(...)
        # Automatically prints report on exit
    """

    def __init__(self, name: str, print_report: bool = True, min_traces: int = 2):
        self.name = name
        self.print_report = print_report
        self.min_traces = min_traces
        self._start_counts: Dict[str, int] = {}

    def __enter__(self):
        self._start_counts = dict(_trace_counts)
        return self

    def __exit__(self, *args):
        if self.print_report:
            print(f"\n=== Trace Report for '{self.name}' ===")
            for name, count in _trace_counts.items():
                start = self._start_counts.get(name, 0)
                delta = count - start
                if delta >= self.min_traces:
                    unique = len(set(_trace_shapes[name][start:]))
                    print(f"  {name}: +{delta} traces ({unique} unique)")


# Utility to wrap all functions in a dict
def monitor_dict(
    fns: Dict[str, Callable],
    prefix: str = "",
    max_traces: int = 5,
) -> Dict[str, Callable]:
    """Wrap all functions in a dict with trace monitoring.

    Args:
        fns: Dict of name -> function
        prefix: Prefix for monitor names
        max_traces: Max traces before warning

    Returns:
        Dict with same keys but monitored functions
    """
    result = {}
    for name, fn in fns.items():
        monitor_name = f"{prefix}{name}" if prefix else name
        result[name] = trace_monitor(monitor_name, max_traces)(fn)
    return result
