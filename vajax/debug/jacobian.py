"""Jacobian format conversion and comparison utilities.

OSDI and openvaf_jax return Jacobians in different formats:
- OSDI: Column-major sparse (only has_resist=True entries)
- JAX: Row-major dense (all N×N entries)

This module provides utilities to convert between formats and compare Jacobians.
"""

from typing import NamedTuple, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike


class JacobianComparison(NamedTuple):
    """Result of comparing two Jacobians."""

    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    osdi_nonzero_count: int
    jax_nonzero_count: int
    mismatched_positions: list[tuple[int, int]]  # (row, col) pairs
    report: str


def osdi_to_dense_jacobian(
    osdi_jac: Sequence[float],
    n_nodes: int,
    jacobian_keys: Sequence[dict],
    reactive: bool = False,
) -> np.ndarray:
    """Convert OSDI sparse column-major Jacobian to dense row-major matrix.

    OSDI returns Jacobian entries in column-major order, only for entries
    where has_resist=True (or has_react=True for reactive).

    Args:
        osdi_jac: Sparse Jacobian array from OSDI eval()
        n_nodes: Number of nodes (matrix is n_nodes × n_nodes)
        jacobian_keys: List of dicts with 'row', 'col', 'has_resist', 'has_react'
                      from OSDI get_jacobian() or openvaf module descriptor
        reactive: If True, use has_react flag instead of has_resist

    Returns:
        Dense n_nodes × n_nodes numpy array in row-major order
    """
    dense = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    # OSDI returns entries for has_resist=True (or has_react=True) entries
    flag_key = "has_react" if reactive else "has_resist"
    osdi_idx = 0
    for key in jacobian_keys:
        if key.get(flag_key, False):
            row = key["row"]
            col = key["col"]
            if osdi_idx < len(osdi_jac):
                dense[row, col] = osdi_jac[osdi_idx]
                osdi_idx += 1

    return dense


def jax_to_dense_jacobian(
    jax_jac: ArrayLike,
    n_nodes: int,
    jacobian_keys: Optional[Sequence[dict]] = None,
) -> np.ndarray:
    """Convert JAX flat row-major Jacobian to dense matrix.

    JAX returns entries in row-major order based on jacobian_keys structure.
    The array length equals the number of jacobian_keys (one entry per key).

    Args:
        jax_jac: Flat Jacobian array from JAX eval
        n_nodes: Number of nodes
        jacobian_keys: If provided, maps JAX entries to (row, col) positions.
                      If None, assumes N×N dense row-major order.

    Returns:
        Dense n_nodes × n_nodes numpy array
    """
    jax_arr = np.asarray(jax_jac)
    dense = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    if jacobian_keys is None:
        # Assume N×N row-major
        if len(jax_arr) != n_nodes * n_nodes:
            raise ValueError(
                f"JAX Jacobian length {len(jax_arr)} != {n_nodes}² = {n_nodes * n_nodes}. "
                "Provide jacobian_keys for non-square Jacobians."
            )
        return jax_arr.reshape((n_nodes, n_nodes))

    # JAX uses row-major order over all jacobian_keys
    # Each key has a (row, col) position
    if len(jax_arr) != len(jacobian_keys):
        raise ValueError(
            f"JAX Jacobian length {len(jax_arr)} != jacobian_keys length {len(jacobian_keys)}"
        )

    for i, key in enumerate(jacobian_keys):
        row = key["row"]
        col = key["col"]
        dense[row, col] = jax_arr[i]

    return dense


def compare_jacobians(
    osdi_jac: Sequence[float],
    jax_jac: ArrayLike,
    n_nodes: int,
    jacobian_keys: Sequence[dict],
    rtol: float = 1e-4,
    atol: float = 1e-10,
    reactive: bool = False,
) -> JacobianComparison:
    """Compare OSDI and JAX Jacobians, handling format differences.

    Converts both to dense row-major format and compares element-by-element.

    Args:
        osdi_jac: Sparse Jacobian from OSDI
        jax_jac: Dense Jacobian from JAX
        n_nodes: Number of nodes
        jacobian_keys: Jacobian structure from OSDI
        rtol: Relative tolerance
        atol: Absolute tolerance
        reactive: If True, compare reactive (dQ/dV) instead of resistive (dI/dV)

    Returns:
        JacobianComparison with detailed results
    """
    # Convert to dense matrices
    osdi_dense = osdi_to_dense_jacobian(osdi_jac, n_nodes, jacobian_keys, reactive=reactive)
    jax_dense = jax_to_dense_jacobian(jax_jac, n_nodes, jacobian_keys)

    # Count non-zeros
    osdi_nonzero = np.sum(np.abs(osdi_dense) > atol)
    jax_nonzero = np.sum(np.abs(jax_dense) > atol)

    # Compute differences
    abs_diff = np.abs(osdi_dense - jax_dense)
    max_abs_diff = float(np.max(abs_diff))

    # Relative difference (avoid division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = abs_diff / np.maximum(np.abs(osdi_dense), atol)
        rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0.0)
    max_rel_diff = float(np.max(rel_diff))

    # Find mismatched positions
    mismatched = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            osdi_val = osdi_dense[i, j]
            jax_val = jax_dense[i, j]
            if not np.isclose(osdi_val, jax_val, rtol=rtol, atol=atol):
                # Check if it's a structural mismatch (one zero, one non-zero)
                osdi_is_zero = abs(osdi_val) < atol
                jax_is_zero = abs(jax_val) < atol
                if osdi_is_zero != jax_is_zero:
                    mismatched.append((i, j))

    # Determine pass/fail
    passed = max_abs_diff < atol or max_rel_diff < rtol

    # Build report
    jac_type = "Reactive (dQ/dV)" if reactive else "Resistive (dI/dV)"
    report_lines = [
        f"{jac_type} Jacobian Comparison ({n_nodes}×{n_nodes}):",
        f"  OSDI non-zeros: {osdi_nonzero}",
        f"  JAX non-zeros:  {jax_nonzero}",
        f"  Max abs diff:   {max_abs_diff:.6e}",
        f"  Max rel diff:   {max_rel_diff:.6e}",
        f"  Passed:         {passed}",
    ]

    if mismatched:
        report_lines.append(f"  Structural mismatches: {len(mismatched)}")
        for i, j in mismatched[:10]:  # Show first 10
            report_lines.append(
                f"    [{i},{j}]: OSDI={osdi_dense[i, j]:.6e}, JAX={jax_dense[i, j]:.6e}"
            )
        if len(mismatched) > 10:
            report_lines.append(f"    ... and {len(mismatched) - 10} more")

    return JacobianComparison(
        passed=passed,
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        osdi_nonzero_count=int(osdi_nonzero),
        jax_nonzero_count=int(jax_nonzero),
        mismatched_positions=mismatched,
        report="\n".join(report_lines),
    )


def print_jacobian_structure(
    jacobian_keys: list[dict],
    n_nodes: int,
    name: str = "Jacobian",
) -> None:
    """Print the structure of a Jacobian from its keys.

    Args:
        jacobian_keys: List of dicts with row, col, has_resist, has_react
        n_nodes: Number of nodes
        name: Name for display
    """
    print(f"\n{name} structure ({n_nodes}×{n_nodes}):")

    # Build sparsity pattern
    resist_pattern = np.zeros((n_nodes, n_nodes), dtype=int)
    react_pattern = np.zeros((n_nodes, n_nodes), dtype=int)

    for key in jacobian_keys:
        row = key["row"]
        col = key["col"]
        if key.get("has_resist", False):
            resist_pattern[row, col] = 1
        if key.get("has_react", False):
            react_pattern[row, col] = 1

    print(f"  Resist entries: {np.sum(resist_pattern)}")
    print(f"  React entries:  {np.sum(react_pattern)}")

    # Print patterns
    print("\n  Resist pattern (X = has entry):")
    for i in range(n_nodes):
        row_str = "  "
        for j in range(n_nodes):
            row_str += "X " if resist_pattern[i, j] else ". "
        print(row_str)
