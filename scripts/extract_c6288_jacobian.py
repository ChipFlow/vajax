#!/usr/bin/env python3
"""Extract sparse Jacobian + RHS from c6288 for BaSpaCho testing.

Runs c6288 through DC operating point solve, then captures the Jacobian
matrix and RHS vector at the first NR iteration of the first transient step.
Exports in Matrix Market (.mtx) and NumPy (.npz) formats.

Usage:
    JAX_PLATFORMS=cpu uv run python scripts/extract_c6288_jacobian.py

Output files in c6288_jacobian/:
    jacobian.mtx      - Jacobian in Matrix Market format (COO, real general)
    jacobian.npz      - CSR arrays: indptr, indices, data, rhs, shape
    metadata.txt      - Human-readable summary
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import logging
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from vajax import configure_precision

configure_precision(force_x64=True)

from vajax.analysis.engine import CircuitEngine
from vajax.benchmarks.registry import get_benchmark

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_jacobian(output_dir: Path = Path("c6288_jacobian")) -> None:
    """Extract Jacobian from c6288 at DC operating point."""
    output_dir.mkdir(exist_ok=True)

    # Load c6288 benchmark
    info = get_benchmark("c6288")
    assert info is not None, "c6288 benchmark not found"
    logger.info(f"Loading {info.name}: {info.title}")

    engine = CircuitEngine(info.sim_path)
    engine.parse()
    logger.info(f"Parsed: {engine.num_nodes} nodes, {len(engine.devices)} devices")

    # Build transient setup to get build_system function
    setup = engine._build_transient_setup(backend="cpu", use_dense=False)
    n_unknowns = setup["n_unknowns"]
    source_device_data = setup["source_device_data"]
    static_inputs_cache = setup["static_inputs_cache"]

    # Create build_system function (use dense=False for COO output)
    build_system_fn, device_arrays, total_limit_states = engine._make_mna_build_system_fn(
        source_device_data=source_device_data,
        vmapped_fns={},
        static_inputs_cache=static_inputs_cache,
        n_unknowns=n_unknowns,
        use_dense=False,  # Sparse COO output
    )

    # Also create a dense version for easy export
    build_system_dense, _, _ = engine._make_mna_build_system_fn(
        source_device_data=source_device_data,
        vmapped_fns={},
        static_inputs_cache=static_inputs_cache,
        n_unknowns=n_unknowns,
        use_dense=True,
    )

    # Determine system dimensions
    n_vsources = len(source_device_data.get("vsource", {}).get("names", []))
    n_total = n_unknowns + 1  # +1 for ground
    n_augmented = n_unknowns + n_vsources

    logger.info(
        f"System size: {n_unknowns} unknowns + {n_vsources} vsources = {n_augmented} augmented"
    )

    # Get DC source values
    vsource_dc_vals, isource_dc_vals = engine._get_dc_source_values(n_vsources, 0)
    vdd = float(engine._get_vdd_value())
    logger.info(f"VDD = {vdd}V, {n_vsources} voltage source(s)")

    # Initialize solution: mid-rail voltages (reasonable starting point)
    V_init = jnp.full(n_total, vdd / 2, dtype=jnp.float64)
    V_init = V_init.at[0].set(0.0)  # Ground = 0V

    X_init = jnp.zeros(n_total + n_vsources, dtype=jnp.float64)
    X_init = X_init.at[:n_total].set(V_init)

    # Zero charges, DC mode (integ_c0=0)
    Q_prev = jnp.zeros(n_unknowns, dtype=jnp.float64)
    limit_state = (
        jnp.zeros(total_limit_states, dtype=jnp.float64) if total_limit_states > 0 else None
    )

    logger.info("Evaluating build_system (first NR iteration at mid-rail)...")

    # Build Jacobian at initial point (DC: integ_c0=0)
    J_dense, f, Q, I_vs, lim_out, max_res = build_system_dense(
        X_init,
        vsource_dc_vals,
        isource_dc_vals,
        Q_prev,
        0.0,  # integ_c0 = 0 for DC
        device_arrays,
        engine.options.gmin,  # gmin
        0.0,  # gshunt
        0.0,
        0.0,
        None,
        0.0,
        None,  # integration history (unused for DC)
        limit_state,
        0,  # nr_iteration = 0
    )

    # Block until computation is done
    J_np = np.array(jax.block_until_ready(J_dense))
    f_np = np.array(jax.block_until_ready(f))

    logger.info(f"Jacobian shape: {J_np.shape}")
    logger.info(f"RHS shape: {f_np.shape}")
    logger.info(f"max|f| = {np.max(np.abs(f_np)):.6e}")

    # Convert to scipy CSR for analysis and export
    from scipy.io import mmwrite
    from scipy.sparse import csr_matrix

    J_csr = csr_matrix(J_np)
    J_csr.eliminate_zeros()
    nnz = J_csr.nnz
    density = nnz / (n_augmented * n_augmented) * 100

    logger.info(f"Sparsity: {nnz} nonzeros, {density:.4f}% density")
    logger.info(f"Avg nonzeros/row: {nnz / n_augmented:.1f}")

    # Diagonal analysis
    diag = J_csr.diagonal()
    logger.info(f"Diagonal: min={np.min(diag):.6e}, max={np.max(diag):.6e}")
    zero_diag = np.sum(np.abs(diag) < 1e-20)
    if zero_diag > 0:
        logger.warning(f"  {zero_diag} near-zero diagonal entries!")

    # Symmetry check (BaSpaCho might care)
    sym_diff = J_csr - J_csr.T
    sym_norm = np.max(np.abs(sym_diff.data)) if sym_diff.nnz > 0 else 0.0
    logger.info(
        f"Symmetry: max|J - J^T| = {sym_norm:.6e} ({'symmetric' if sym_norm < 1e-12 else 'unsymmetric'})"
    )

    # --- Export Matrix Market format ---
    mtx_path = output_dir / "jacobian.mtx"
    mmwrite(str(mtx_path), J_csr, comment="c6288 Jacobian at DC mid-rail, first NR iteration")
    logger.info(f"Wrote {mtx_path} ({mtx_path.stat().st_size / 1024:.0f} KB)")

    # --- Export RHS in Matrix Market format ---
    from scipy.sparse import csc_matrix

    rhs_sparse = csc_matrix(f_np.reshape(-1, 1))
    rhs_path = output_dir / "rhs.mtx"
    mmwrite(str(rhs_path), rhs_sparse, comment="c6288 RHS at DC mid-rail, first NR iteration")
    logger.info(f"Wrote {rhs_path}")

    # --- Export NumPy format (CSR arrays + RHS) ---
    npz_path = output_dir / "jacobian.npz"
    np.savez_compressed(
        str(npz_path),
        indptr=J_csr.indptr.astype(np.int32),
        indices=J_csr.indices.astype(np.int32),
        data=J_csr.data.astype(np.float64),
        rhs=f_np.astype(np.float64),
        shape=np.array(J_csr.shape, dtype=np.int32),
    )
    logger.info(f"Wrote {npz_path} ({npz_path.stat().st_size / 1024:.0f} KB)")

    # --- Export solution vector (for verification) ---
    x_path = output_dir / "x_init.npy"
    np.save(str(x_path), np.array(X_init))

    # --- Metadata ---
    meta_path = output_dir / "metadata.txt"
    with open(meta_path, "w") as mf:
        mf.write("c6288 Jacobian Export for BaSpaCho Testing\n")
        mf.write("=" * 50 + "\n\n")
        mf.write(f"Circuit: {info.name} ({info.title})\n")
        mf.write(f"Operating point: DC, mid-rail ({vdd / 2:.2f}V), first NR iteration\n\n")
        mf.write(f"Matrix dimensions: {n_augmented} x {n_augmented}\n")
        mf.write(f"  n_unknowns (node voltages): {n_unknowns}\n")
        mf.write(f"  n_vsources (branch currents): {n_vsources}\n")
        mf.write(f"  n_augmented (total): {n_augmented}\n\n")
        mf.write(f"Nonzeros: {nnz}\n")
        mf.write(f"Density: {density:.4f}%\n")
        mf.write(f"Avg nnz/row: {nnz / n_augmented:.1f}\n\n")
        mf.write(f"Diagonal range: [{np.min(diag):.6e}, {np.max(diag):.6e}]\n")
        mf.write(f"Near-zero diagonals: {zero_diag}\n")
        mf.write(f"Symmetry: max|J - J^T| = {sym_norm:.6e}\n\n")
        mf.write(f"RHS norm: max|f| = {np.max(np.abs(f_np)):.6e}\n")
        mf.write(f"RHS 2-norm: ||f||_2 = {np.linalg.norm(f_np):.6e}\n\n")
        mf.write("Files:\n")
        mf.write("  jacobian.mtx  - Matrix Market format (COO, real general)\n")
        mf.write("  rhs.mtx       - RHS vector in Matrix Market format\n")
        mf.write("  jacobian.npz  - NumPy compressed: indptr, indices, data, rhs, shape\n")
        mf.write("  x_init.npy    - Initial solution vector\n\n")
        mf.write("Usage with BaSpaCho:\n")
        mf.write("  Load jacobian.mtx as CSR, solve J @ delta = -rhs\n")
        mf.write("  Matrix is real, unsymmetric (MNA with transconductance)\n")
        mf.write("  Requires LU factorization (not Cholesky)\n")

    logger.info(f"Wrote {meta_path}")
    logger.info(f"\nDone! Files in {output_dir}/")

    # Print a quick summary useful for BaSpaCho
    print(f"\n{'=' * 60}")
    print("BaSpaCho Test Case: c6288 Jacobian")
    print(f"{'=' * 60}")
    print(f"  Size:       {n_augmented} x {n_augmented}")
    print(f"  Nonzeros:   {nnz}")
    print(f"  Density:    {density:.4f}%")
    print(f"  Symmetric:  {'yes' if sym_norm < 1e-12 else 'no'}")
    print("  Solver:     LU (unsymmetric)")
    print(f"  Files:      {output_dir}/jacobian.mtx (Matrix Market)")
    print(f"              {output_dir}/jacobian.npz (NumPy CSR)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    extract_jacobian()
