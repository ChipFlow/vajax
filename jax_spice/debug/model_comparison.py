"""Model comparison utilities for debugging OSDI vs JAX discrepancies.

This module provides tools for comparing device model outputs between
OSDI (reference) and JAX (translated) implementations.

Usage:
    from jax_spice.debug.model_comparison import ModelComparator

    comparator = ModelComparator(va_path, osdi_path, params)
    report = comparator.compare_at_bias([0.5, 0.6, 0.0, 0.0])
    print(report)
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Ensure JAX uses 64-bit floats
os.environ.setdefault('JAX_ENABLE_X64', 'true')


@dataclass
class ComparisonResult:
    """Result of comparing OSDI vs JAX at a single operating point."""

    voltages: List[float]

    # Residuals
    osdi_resist: List[float]
    jax_resist: List[float]
    resist_diff: List[float]
    resist_max_abs_diff: float
    resist_max_rel_diff: float

    # Jacobians
    osdi_jac_resist: List[float]
    jax_jac_resist: List[float]
    jac_max_abs_diff: float
    jac_nonzero_osdi: int
    jac_nonzero_jax: int

    # Summary
    passed: bool
    issues: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"=== Comparison at V={self.voltages} ===",
            "",
            "Residuals:",
            f"  OSDI max: {max(abs(v) for v in self.osdi_resist):.6e}",
            f"  JAX max:  {max(abs(v) for v in self.jax_resist):.6e}",
            f"  Max abs diff: {self.resist_max_abs_diff:.6e}",
            f"  Max rel diff: {self.resist_max_rel_diff:.2%}",
            "",
            "Jacobian:",
            f"  OSDI non-zeros: {self.jac_nonzero_osdi}",
            f"  JAX non-zeros:  {self.jac_nonzero_jax}",
            f"  Max abs diff:   {self.jac_max_abs_diff:.6e}",
            "",
            f"Passed: {self.passed}",
        ]
        if self.issues:
            lines.append("")
            lines.append("Issues:")
            for issue in self.issues:
                lines.append(f"  - {issue}")
        return "\n".join(lines)


@dataclass
class CacheAnalysis:
    """Analysis of JAX cache values."""

    size: int
    nonzero_count: int
    has_inf: int
    has_nan: int
    large_values: List[Tuple[int, float]]  # (index, value) for |value| > 1e10
    temperature_related: List[Tuple[int, float, float]]  # (index, value, implied_temp)

    def __str__(self) -> str:
        lines = [
            "=== Cache Analysis ===",
            f"Size: {self.size}",
            f"Non-zero: {self.nonzero_count}",
            f"Has inf: {self.has_inf}",
            f"Has nan: {self.has_nan}",
        ]
        if self.large_values:
            lines.append("")
            lines.append(f"Large values (|v| > 1e10): {len(self.large_values)}")
            for idx, val in self.large_values[:10]:
                lines.append(f"  cache[{idx}] = {val:.6e}")
        if self.temperature_related:
            lines.append("")
            lines.append("Temperature-related values (VT range):")
            for idx, val, implied_t in self.temperature_related:
                lines.append(f"  cache[{idx}] = {val:.8f} (implies T={implied_t:.1f}K)")
        return "\n".join(lines)


class ModelComparator:
    """Compare OSDI and JAX model implementations."""

    # Physical constants
    K_B = 1.38064852e-23  # Boltzmann constant
    Q = 1.602176634e-19   # Elementary charge

    def __init__(
        self,
        va_path: Path,
        osdi_path: Path,
        params: Dict[str, float],
        temperature: float = 300.0,
    ):
        """Initialize comparator.

        Args:
            va_path: Path to Verilog-A source file
            osdi_path: Path to compiled OSDI file
            params: Device parameters (e.g., {'TYPE': 1, 'W': 1e-6, 'L': 1e-7})
            temperature: Device temperature in Kelvin
        """
        self.va_path = Path(va_path)
        self.osdi_path = Path(osdi_path)
        self.params = params
        self.temperature = temperature

        self._osdi_eval = None
        self._jax_eval = None
        self._jax_cache = None
        self._jax_metadata = None
        self._n_nodes = None

    def _ensure_evaluators(self):
        """Lazily create evaluators."""
        if self._osdi_eval is not None:
            return

        # Import here to avoid circular imports
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tests'))

        from test_osdi_jax_comparison import create_osdi_evaluator, create_jax_evaluator

        self._osdi_eval, _, _, _, self._jacobian_keys, self._n_nodes = \
            create_osdi_evaluator(self.osdi_path, self.params, self.temperature)

        self._jax_eval, _, self._jax_metadata = \
            create_jax_evaluator(self.va_path, self.params, self.temperature)

        # Get cache for analysis
        _, _, _, debug_info = create_jax_evaluator(
            self.va_path, self.params, self.temperature, return_debug_info=True
        )
        self._jax_cache = debug_info['cache']

    def compare_at_bias(
        self,
        voltages: List[float],
        rtol: float = 1e-4,
        atol: float = 1e-12,
    ) -> ComparisonResult:
        """Compare OSDI and JAX outputs at a specific bias point.

        Args:
            voltages: Terminal voltages [V1, V2, ...]
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison

        Returns:
            ComparisonResult with detailed comparison data
        """
        self._ensure_evaluators()

        osdi_res = self._osdi_eval(voltages)
        jax_res = self._jax_eval(voltages)

        osdi_resist = list(osdi_res[0])
        jax_resist = [float(v) for v in jax_res[0]]

        osdi_jac = list(osdi_res[2])
        jax_jac = [float(v) for v in jax_res[2]]

        # Calculate differences
        resist_diff = [o - j for o, j in zip(osdi_resist, jax_resist)]
        resist_max_abs = max(abs(d) for d in resist_diff)

        # Relative diff (avoid div by zero)
        resist_max_rel = 0.0
        for o, j in zip(osdi_resist, jax_resist):
            if abs(o) > atol:
                resist_max_rel = max(resist_max_rel, abs(o - j) / abs(o))

        # Jacobian comparison
        jac_diff = max(abs(o - j) for o, j in zip(osdi_jac, jax_jac)) if osdi_jac and jax_jac else 0.0
        jac_nz_osdi = sum(1 for v in osdi_jac if abs(v) > 1e-20)
        jac_nz_jax = sum(1 for v in jax_jac if abs(v) > 1e-20)

        # Check for issues
        issues = []
        passed = True

        if resist_max_rel > rtol:
            issues.append(f"Residual relative error {resist_max_rel:.2%} exceeds {rtol:.2%}")
            passed = False

        if jac_nz_osdi > 0 and jac_nz_jax == 0:
            issues.append(f"JAX Jacobian is all zeros but OSDI has {jac_nz_osdi} non-zeros")
            passed = False

        if jac_nz_osdi > 0 and jac_nz_jax > 0:
            if abs(jac_nz_osdi - jac_nz_jax) > jac_nz_osdi * 0.1:
                issues.append(f"Jacobian sparsity mismatch: OSDI={jac_nz_osdi}, JAX={jac_nz_jax}")

        return ComparisonResult(
            voltages=voltages,
            osdi_resist=osdi_resist,
            jax_resist=jax_resist,
            resist_diff=resist_diff,
            resist_max_abs_diff=resist_max_abs,
            resist_max_rel_diff=resist_max_rel,
            osdi_jac_resist=osdi_jac,
            jax_jac_resist=jax_jac,
            jac_max_abs_diff=jac_diff,
            jac_nonzero_osdi=jac_nz_osdi,
            jac_nonzero_jax=jac_nz_jax,
            passed=passed,
            issues=issues,
        )

    def sweep_comparison(
        self,
        base_voltages: List[float],
        sweep_index: int,
        sweep_values: List[float],
    ) -> List[ComparisonResult]:
        """Compare across a voltage sweep.

        Args:
            base_voltages: Base terminal voltages
            sweep_index: Index of voltage to sweep
            sweep_values: Values to sweep through

        Returns:
            List of ComparisonResult for each sweep point
        """
        results = []
        for v in sweep_values:
            voltages = list(base_voltages)
            voltages[sweep_index] = v
            results.append(self.compare_at_bias(voltages))
        return results

    def analyze_cache(self) -> CacheAnalysis:
        """Analyze JAX cache values for potential issues."""
        self._ensure_evaluators()

        cache = np.array(self._jax_cache)

        # Basic stats
        nonzero = np.sum(np.abs(cache) > 1e-20)
        has_inf = np.sum(np.isinf(cache))
        has_nan = np.sum(np.isnan(cache))

        # Large values
        large_values = []
        for i, v in enumerate(cache):
            if abs(v) > 1e10 and not np.isinf(v):
                large_values.append((i, float(v)))

        # Temperature-related (VT range 0.02-0.03)
        temp_related = []
        for i, v in enumerate(cache):
            if 0.02 < v < 0.03:
                implied_t = v * self.Q / self.K_B
                temp_related.append((i, float(v), implied_t))

        return CacheAnalysis(
            size=len(cache),
            nonzero_count=int(nonzero),
            has_inf=int(has_inf),
            has_nan=int(has_nan),
            large_values=large_values,
            temperature_related=temp_related,
        )

    def print_residual_table(self, voltages: List[float]):
        """Print a side-by-side residual comparison table."""
        self._ensure_evaluators()

        osdi_res = self._osdi_eval(voltages)
        jax_res = self._jax_eval(voltages)

        node_names = self._jax_metadata.get('node_names', [])

        print(f"\n=== Residual Comparison at V={voltages} ===")
        print(f"{'Node':<6} {'Name':<10} {'OSDI':>14} {'JAX':>14} {'Diff':>14}")
        print("-" * 60)

        for i in range(self._n_nodes):
            name = node_names[i] if i < len(node_names) else f"n{i}"
            osdi_v = osdi_res[0][i]
            jax_v = float(jax_res[0][i])
            diff = osdi_v - jax_v
            print(f"{i:<6} {name:<10} {osdi_v:>14.4e} {jax_v:>14.4e} {diff:>14.4e}")


def quick_compare(
    va_path: Path,
    osdi_path: Path,
    params: Dict[str, float],
    voltages: List[float],
    temperature: float = 300.0,
) -> ComparisonResult:
    """Quick one-shot comparison.

    Usage:
        result = quick_compare(
            va_path="vendor/OpenVAF/integration_tests/PSP102/psp102.va",
            osdi_path="/tmp/osdi_jax_test_cache/psp102.osdi",
            params={'TYPE': 1, 'W': 1e-6, 'L': 1e-7},
            voltages=[0.5, 0.6, 0.0, 0.0],
        )
        print(result)
    """
    comparator = ModelComparator(va_path, osdi_path, params, temperature)
    return comparator.compare_at_bias(voltages)
