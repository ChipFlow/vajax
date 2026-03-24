"""Capture a sequence of (Jacobian, RHS) pairs from VAJAX benchmarks.

Instruments the NR solver to capture every linear system Ax=b solved
during transient simulation, and writes them as Matrix Market files
suitable for sparse solver testing (e.g. BaSpaCho).

For dense solvers (small circuits like ring), captures the full Jacobian.
For sparse solvers (large circuits like c6288), captures CSR data and
reconstructs the sparse matrix using the pre-computed CSR structure.

Usage:
    JAX_PLATFORMS=cpu uv run scripts/capture_benchmark_matrices.py ring
    JAX_PLATFORMS=cpu uv run scripts/capture_benchmark_matrices.py c6288 --max-captures 20
    JAX_PLATFORMS=cpu uv run scripts/capture_benchmark_matrices.py ring --output-dir /tmp/ring_mtx
"""

import argparse
import os
import sys
from pathlib import Path

# Force CPU and float64 before any JAX import
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import scipy.sparse

from vajax.analysis import CircuitEngine
from vajax.benchmarks.registry import get_benchmark, list_benchmarks

# ---------------------------------------------------------------------------
# Matrix Market writers
# ---------------------------------------------------------------------------


def write_mm_dense(path: Path, matrix: np.ndarray, comment: str = ""):
    """Write a dense matrix as Matrix Market coordinate format.

    Only writes non-zero entries.  1-indexed row/col as per the spec.
    """
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)

    rows, cols = matrix.shape
    nz_rows, nz_cols = np.nonzero(matrix)
    nnz = len(nz_rows)

    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        if comment:
            for line in comment.splitlines():
                f.write(f"%{line}\n")
        f.write(f"{rows} {cols} {nnz}\n")
        for i, j in zip(nz_rows, nz_cols):
            f.write(f"{i + 1} {j + 1} {matrix[i, j]:.17E}\n")


def write_mm_csr(
    path: Path,
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    comment: str = "",
):
    """Write a CSR matrix as Matrix Market coordinate format."""
    rows, cols = shape
    # Count non-zeros (skip exact zeros in data)
    nz_mask = data != 0.0
    nnz = int(np.sum(nz_mask))

    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        if comment:
            for line in comment.splitlines():
                f.write(f"%{line}\n")
        f.write(f"{rows} {cols} {nnz}\n")
        for row in range(rows):
            for idx in range(indptr[row], indptr[row + 1]):
                val = data[idx]
                if val != 0.0:
                    col = indices[idx]
                    f.write(f"{row + 1} {col + 1} {val:.17E}\n")


def write_mm_vector(path: Path, vec: np.ndarray, comment: str = ""):
    """Write a vector as Matrix Market coordinate format (n x 1 matrix)."""
    n = len(vec)
    nz = np.nonzero(vec)[0]
    nnz = len(nz)

    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        if comment:
            for line in comment.splitlines():
                f.write(f"%{line}\n")
        f.write(f"{n} 1 {nnz}\n")
        for i in nz:
            f.write(f"{i + 1} 1 {vec[i]:.17E}\n")


# ---------------------------------------------------------------------------
# Capture state
# ---------------------------------------------------------------------------

captured_systems: list[tuple[np.ndarray, np.ndarray]] = []
capture_counter = 0
max_captures_limit = 50

# For sparse solvers: stash CSR structure here
csr_structure: dict = {}


def capture_callback(J_or_data: jax.Array, f: jax.Array):
    """Called from inside the JIT-compiled NR loop via jax.debug.callback."""
    global capture_counter
    if capture_counter >= max_captures_limit:
        return
    captured_systems.append((np.asarray(J_or_data).copy(), np.asarray(f).copy()))
    capture_counter += 1


# ---------------------------------------------------------------------------
# Monkey-patches
# ---------------------------------------------------------------------------

import vajax.analysis.solver_factories as sf

_original_make_nr_common = sf._make_nr_solver_common


def patched_make_nr_solver_common(*, linear_solve_fn, **kwargs):
    """Wrap linear_solve_fn with capture callback."""

    def instrumented_linear_solve(J_or_data, f):
        jax.debug.callback(capture_callback, J_or_data, f)
        return linear_solve_fn(J_or_data, f)

    return _original_make_nr_common(
        linear_solve_fn=instrumented_linear_solve,
        **kwargs,
    )


sf._make_nr_solver_common = patched_make_nr_solver_common

# Also intercept UMFPACK factory to grab CSR structure
_original_make_umfpack = sf.make_umfpack_ffi_full_mna_solver


def _extract_csr_args(args, kwargs):
    """Extract CSR structure from factory call args.

    Signature: (build_system_jit, n_nodes, n_vsources, nse, *, bcsr_indptr, bcsr_indices, ...)
    First 4 are positional, rest are keyword.
    """
    n_nodes = args[1] if len(args) > 1 else kwargs.get("n_nodes")
    n_vsources = args[2] if len(args) > 2 else kwargs.get("n_vsources")
    bcsr_indptr = kwargs.get("bcsr_indptr")
    bcsr_indices = kwargs.get("bcsr_indices")
    # Fallback: positional args 4, 5
    if bcsr_indptr is None and len(args) > 4:
        bcsr_indptr = args[4]
    if bcsr_indices is None and len(args) > 5:
        bcsr_indices = args[5]
    return n_nodes, n_vsources, bcsr_indptr, bcsr_indices


def patched_make_umfpack(*args, **kwargs):
    """Capture CSR structure before building the solver."""
    n_nodes, n_vsources, bcsr_indptr, bcsr_indices = _extract_csr_args(args, kwargs)

    if bcsr_indptr is not None and n_nodes is not None:
        n_unknowns = n_nodes - 1
        n_augmented = n_unknowns + n_vsources
        csr_structure["indptr"] = np.asarray(bcsr_indptr).copy()
        csr_structure["indices"] = np.asarray(bcsr_indices).copy()
        csr_structure["shape"] = (n_augmented, n_augmented)
        print(
            f"  Captured CSR structure: {n_augmented}x{n_augmented}, "
            f"nnz_slots={len(csr_structure['indices'])}"
        )

    return _original_make_umfpack(*args, **kwargs)


sf.make_umfpack_ffi_full_mna_solver = patched_make_umfpack

# Also intercept Spineax factory for GPU sparse
_original_make_spineax = sf.make_spineax_full_mna_solver


def patched_make_spineax(*args, **kwargs):
    """Capture CSR structure from Spineax factory."""
    n_nodes, n_vsources, bcsr_indptr, bcsr_indices = _extract_csr_args(args, kwargs)

    if bcsr_indptr is not None and n_nodes is not None:
        n_unknowns = n_nodes - 1
        n_augmented = n_unknowns + n_vsources
        csr_structure["indptr"] = np.asarray(bcsr_indptr).copy()
        csr_structure["indices"] = np.asarray(bcsr_indices).copy()
        csr_structure["shape"] = (n_augmented, n_augmented)
        print(
            f"  Captured CSR structure: {n_augmented}x{n_augmented}, "
            f"nnz_slots={len(csr_structure['indices'])}"
        )

    return _original_make_spineax(*args, **kwargs)


sf.make_spineax_full_mna_solver = patched_make_spineax

# Patch in full_mna.py's namespace too (it imports by name at module level)
import vajax.analysis.transient.full_mna as _full_mna

_full_mna.make_umfpack_ffi_full_mna_solver = patched_make_umfpack
_full_mna.make_spineax_full_mna_solver = patched_make_spineax


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_systems(
    out_dir: Path,
    benchmark_name: str,
    engine: "CircuitEngine",
    t_stop: float,
    dt: float,
    num_steps: int,
    is_sparse: bool,
):
    """Write all captured systems as Matrix Market files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_sparse:
        assert "indptr" in csr_structure, "CSR structure not captured"
        indptr = csr_structure["indptr"]
        indices = csr_structure["indices"]
        shape = csr_structure["shape"]
        n = shape[0]
    else:
        n = captured_systems[0][0].shape[0]

    nnz_list = []

    for idx, (J_or_data, f) in enumerate(captured_systems):
        if is_sparse:
            # J_or_data is 1D CSR values array
            nz_mask = J_or_data != 0.0
            nnz = int(np.sum(nz_mask))
            nnz_list.append(nnz)
            comment = (
                f" {benchmark_name} benchmark, NR linear system #{idx}\n"
                f" matrix size: {n}x{n}, nnz={nnz} (CSR with {len(J_or_data)} slots)"
            )
            write_mm_csr(
                out_dir / f"jacobian_{idx:04d}.mtx",
                J_or_data,
                indices,
                indptr,
                shape,
                comment,
            )
        else:
            # J_or_data is 2D dense matrix
            nnz = int(np.count_nonzero(J_or_data))
            nnz_list.append(nnz)
            comment = (
                f" {benchmark_name} benchmark, NR linear system #{idx}\n"
                f" matrix size: {J_or_data.shape[0]}x{J_or_data.shape[1]}, nnz={nnz}"
            )
            write_mm_dense(out_dir / f"jacobian_{idx:04d}.mtx", J_or_data, comment)

        write_mm_vector(
            out_dir / f"rhs_{idx:04d}.mtx",
            f,
            f" {benchmark_name} benchmark, RHS vector #{idx}",
        )

        if (idx + 1) % 10 == 0:
            print(f"  Wrote system {idx + 1}/{len(captured_systems)}")

    # Summary
    summary_path = out_dir / "README.md"
    avg_nnz = np.mean(nnz_list)
    with open(summary_path, "w") as fh:
        fh.write(f"# {benchmark_name.upper()} Benchmark Matrix Sequence\n\n")
        fh.write(f"Circuit: {benchmark_name} ({engine.num_nodes} external nodes)\n")
        fh.write(f"Matrix size: {n} x {n}\n")
        fh.write(f"Solver: {'sparse (UMFPACK CSR)' if is_sparse else 'dense'}\n")
        fh.write(f"Systems captured: {len(captured_systems)}\n")
        fh.write(f"Simulation: t_stop={t_stop:.2e}s, dt={dt:.2e}s\n")
        fh.write(f"Steps: {num_steps}\n")
        fh.write(f"Avg nnz: {avg_nnz:.0f} ({avg_nnz / (n * n) * 100:.4f}%)\n\n")
        fh.write("## Files\n\n")
        fh.write("Each system `i` consists of:\n")
        fh.write("- `jacobian_NNNN.mtx` - Jacobian matrix J (Matrix Market coordinate format)\n")
        fh.write("- `rhs_NNNN.mtx` - RHS vector f (the solve is J @ delta = -f)\n\n")
        fh.write("## Sparsity\n\n")
        fh.write("| System | nnz | density |\n")
        fh.write("|--------|-----|--------|\n")
        for idx, nnz in enumerate(nnz_list):
            density = nnz / (n * n) * 100
            fh.write(f"| {idx:4d} | {nnz:6d} | {density:.4f}% |\n")

    return n, nnz_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global max_captures_limit

    parser = argparse.ArgumentParser(
        description="Capture NR linear systems from VAJAX benchmarks as Matrix Market files"
    )
    parser.add_argument(
        "benchmark",
        help="Benchmark name (e.g. ring, c6288, graetz)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: baspacho test_data/<benchmark>_sequence)",
    )
    parser.add_argument(
        "--max-captures",
        type=int,
        default=50,
        help="Maximum number of (J, b) pairs to capture",
    )
    parser.add_argument(
        "--t-stop",
        type=float,
        default=None,
        help="Override transient stop time",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Force dense solver (default: auto based on circuit size)",
    )
    args = parser.parse_args()
    max_captures_limit = args.max_captures

    benchmark_name = args.benchmark

    if args.output_dir is None:
        out_dir = (
            Path.home()
            / "Code/ChipFlow/reference/baspacho/test_data"
            / f"{benchmark_name}_sequence"
        )
    else:
        out_dir = args.output_dir

    # --- Set up benchmark ---
    info = get_benchmark(benchmark_name)
    if info is None:
        print(f"Benchmark '{benchmark_name}' not found")
        print(f"available benchmarks are: {list_benchmarks()}")
        exit(1)

    engine = CircuitEngine(info.sim_path)
    engine.parse()

    t_stop = args.t_stop if args.t_stop is not None else info.t_stop
    dt = info.dt
    use_sparse = info.is_large and not args.dense

    print(f"Benchmark: {benchmark_name}")
    print(f"  External nodes: {engine.num_nodes}")
    print(f"  Devices: {len(engine.devices)}")
    print(f"  Solver: {'sparse' if use_sparse else 'dense'}")
    print(f"  Transient: t_stop={t_stop:.2e}s, dt={dt:.2e}s")
    print(f"  Max captures: {max_captures_limit}")

    # Prepare and run
    engine.prepare(t_stop=t_stop, dt=dt, use_sparse=use_sparse)
    print(f"  Augmented system size will be reported by solver factory above")
    print("Running simulation...")
    result = engine.run_transient()

    convergence = result.stats.get("convergence_rate", 0) * 100
    print(f"  Steps: {result.num_steps}, convergence: {convergence:.0f}%")
    print(f"  Captured {len(captured_systems)} linear systems")

    if not captured_systems:
        print("ERROR: No systems captured!", file=sys.stderr)
        sys.exit(1)

    # --- Write output ---
    print(f"Writing to {out_dir} ...")
    n, nnz_list = write_systems(
        out_dir,
        benchmark_name,
        engine,
        t_stop,
        dt,
        result.num_steps,
        use_sparse,
    )

    avg_nnz = np.mean(nnz_list)
    print(
        f"\nDone: {len(captured_systems)} systems, {n}x{n} matrices, "
        f"avg nnz={avg_nnz:.0f} ({avg_nnz / (n * n) * 100:.4f}% dense)"
    )
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
