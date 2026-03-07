# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Analyze parallelism opportunities in VAJAX simulation matrices.

For IREE & Baspacho test case context: given a circuit's Jacobian sparsity
pattern, reports what parallelism can be exploited during factorization,
assembly, and device evaluation.

Key outputs:
- Elimination tree: dependency structure for sparse factorization
- Level-set parallelism: how many columns can be processed simultaneously
- Supernodal structure: dense blocks exploitable with BLAS-3
- Fill-in analysis: memory requirements for factorization
- Device evaluation parallelism: scatter pattern from vmap'd device evals
- Pattern stability: sparsity is fixed across all NR iterations

Usage:
    # Analyze from a benchmark (captures matrices + device info)
    JAX_PLATFORMS=cpu uv run scripts/analyze_parallelism.py ring
    JAX_PLATFORMS=cpu uv run scripts/analyze_parallelism.py c6288

    # Analyze existing Matrix Market file
    uv run scripts/analyze_parallelism.py --from-mtx path/to/jacobian_0000.mtx

    # Output to specific directory
    JAX_PLATFORMS=cpu uv run scripts/analyze_parallelism.py ring --output-dir /tmp/ring_par
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import scipy.io
import scipy.sparse as sp
import scipy.sparse.linalg

# ---------------------------------------------------------------------------
# Elimination tree
# ---------------------------------------------------------------------------


def symmetrize_pattern(A: sp.spmatrix) -> sp.csc_matrix:
    """Compute |A| + |A^T| as a binary pattern (no values, just structure)."""
    A_csc = sp.csc_matrix(A)
    # Binary pattern: set all values to 1
    A_bin = sp.csc_matrix(
        (np.ones(A_csc.nnz), A_csc.indices, A_csc.indptr), shape=A_csc.shape
    )
    A_sym = A_bin + A_bin.T
    # Re-binarize (eliminates any 2s from diagonal overlap)
    A_sym.data[:] = 1.0
    A_sym.eliminate_zeros()
    return sp.csc_matrix(A_sym)


def compute_etree(A_csc: sp.csc_matrix) -> np.ndarray:
    """Compute elimination tree of a symmetric matrix.

    Uses Liu's algorithm with path compression (union-find).
    Only the upper triangle is used.

    Args:
        A_csc: Symmetric matrix in CSC format

    Returns:
        parent array where parent[i] is the parent of column i,
        or -1 for root(s)
    """
    n = A_csc.shape[0]
    parent = np.full(n, -1, dtype=np.int64)
    ancestor = np.arange(n, dtype=np.int64)

    indptr = A_csc.indptr
    indices = A_csc.indices

    for k in range(n):
        for ptr in range(indptr[k], indptr[k + 1]):
            i = indices[ptr]
            if i >= k:
                continue
            # Find root of i with path compression
            r = i
            while ancestor[r] != r:
                r = ancestor[r]
            if r != k:
                parent[r] = k
                ancestor[r] = k
            # Path compression for i
            r = i
            while ancestor[r] != k:
                t = ancestor[r]
                ancestor[r] = k
                r = t

    return parent


def compute_level_sets(parent: np.ndarray) -> list[list[int]]:
    """Compute level sets from an elimination tree.

    Level 0 = leaves, higher levels = closer to root.
    Columns at the same level have no dependencies and can be
    processed in parallel.

    Returns:
        List of levels, where levels[d] = list of column indices at depth d
        (depth measured from leaves, so leaves are at depth 0)
    """
    n = len(parent)
    # Compute depth from root first
    depth_from_root = np.full(n, -1, dtype=np.int64)

    # Find roots
    roots = np.where(parent == -1)[0]
    for r in roots:
        depth_from_root[r] = 0

    # BFS from roots to compute depth_from_root
    # Build children lists for top-down traversal
    children = [[] for _ in range(n)]
    for i in range(n):
        if parent[i] != -1:
            children[parent[i]].append(i)

    queue = list(roots)
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        for child in children[node]:
            depth_from_root[child] = depth_from_root[node] + 1
            queue.append(child)

    max_depth = int(np.max(depth_from_root)) if n > 0 else 0

    # Convert to bottom-up levels (leaves = 0)
    depth_from_leaves = max_depth - depth_from_root

    levels: list[list[int]] = [[] for _ in range(max_depth + 1)]
    for i in range(n):
        levels[depth_from_leaves[i]].append(i)

    return levels


def compute_etree_stats(parent: np.ndarray, levels: list[list[int]]) -> dict:
    """Compute statistics about elimination tree parallelism."""
    n = len(parent)
    widths = [len(level) for level in levels]
    height = len(levels)

    # Count leaves (nodes with no children)
    has_child = np.zeros(n, dtype=bool)
    for i in range(n):
        if parent[i] != -1:
            has_child[parent[i]] = True
    n_leaves = int(np.sum(~has_child))

    # Subtree sizes
    subtree_size = np.ones(n, dtype=np.int64)
    # Process bottom-up: levels[0] are leaves
    for level in levels:
        for node in level:
            if parent[node] != -1:
                subtree_size[parent[node]] += subtree_size[node]

    return {
        "height": height,
        "n_leaves": n_leaves,
        "max_parallelism": max(widths) if widths else 0,
        "avg_parallelism": float(np.mean(widths)) if widths else 0,
        "min_parallelism": min(widths) if widths else 0,
        "level_widths": widths,
        "subtree_size_stats": {
            "min": int(np.min(subtree_size)) if n > 0 else 0,
            "max": int(np.max(subtree_size)) if n > 0 else 0,
            "mean": float(np.mean(subtree_size)) if n > 0 else 0,
            "median": float(np.median(subtree_size)) if n > 0 else 0,
        },
    }


# ---------------------------------------------------------------------------
# Supernodal detection
# ---------------------------------------------------------------------------


def detect_supernodes(parent: np.ndarray, A_csc: sp.csc_matrix) -> list[list[int]]:
    """Detect fundamental supernodes in the elimination tree.

    A fundamental supernode is a maximal chain of consecutive columns
    j, j+1, ..., j+k where:
    - parent[j] = j+1, parent[j+1] = j+2, ..., parent[j+k-1] = j+k
    - The columns have nested sparsity patterns (each is a subset of the next)

    These can be factored as dense blocks using BLAS-3 operations.
    """
    n = A_csc.shape[0]
    if n == 0:
        return []

    # Count children per node
    n_children = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if parent[i] != -1:
            n_children[parent[i]] += 1

    # A node starts a new supernode if:
    # - It has more than one child, OR
    # - It is not the only child of its parent, OR
    # - parent[i] != i + 1
    is_supernode_start = np.ones(n, dtype=bool)
    for i in range(n - 1):
        if parent[i] == i + 1 and n_children[i + 1] == 1:
            is_supernode_start[i + 1] = False

    supernodes: list[list[int]] = []
    current: list[int] = []
    for i in range(n):
        if is_supernode_start[i]:
            if current:
                supernodes.append(current)
            current = [i]
        else:
            current.append(i)
    if current:
        supernodes.append(current)

    return supernodes


def supernode_stats(supernodes: list[list[int]]) -> dict:
    """Compute statistics about supernodal structure."""
    sizes = [len(s) for s in supernodes]
    size_counts = Counter(sizes)

    # Bucket into histogram ranges
    histogram = {}
    for size, count in sorted(size_counts.items()):
        if size == 1:
            histogram["1"] = histogram.get("1", 0) + count
        elif size <= 4:
            histogram["2-4"] = histogram.get("2-4", 0) + count
        elif size <= 10:
            histogram["5-10"] = histogram.get("5-10", 0) + count
        elif size <= 50:
            histogram["11-50"] = histogram.get("11-50", 0) + count
        else:
            histogram["51+"] = histogram.get("51+", 0) + count

    return {
        "count": len(supernodes),
        "largest": max(sizes) if sizes else 0,
        "mean_size": float(np.mean(sizes)) if sizes else 0,
        "median_size": float(np.median(sizes)) if sizes else 0,
        "size_histogram": histogram,
    }


# ---------------------------------------------------------------------------
# Fill-in analysis
# ---------------------------------------------------------------------------


def fill_in_analysis(A: sp.spmatrix) -> dict:
    """Analyze fill-in from LU factorization using scipy's SuperLU.

    Uses MMD_AT_PLUS_A ordering for fill-reducing permutation.
    """
    A_csc = sp.csc_matrix(A, dtype=np.float64)
    n = A_csc.shape[0]
    original_nnz = A_csc.nnz

    results = {}

    for ordering_name, permc_spec in [
        ("MMD_AT_PLUS_A", "MMD_AT_PLUS_A"),
        ("COLAMD", "COLAMD"),
    ]:
        try:
            lu = scipy.sparse.linalg.splu(
                A_csc,
                permc_spec=permc_spec,
                options={"SymmetricMode": False},
            )
            l_nnz = lu.L.nnz
            u_nnz = lu.U.nnz
            factor_nnz = l_nnz + u_nnz - n  # subtract diagonal counted twice

            results[ordering_name] = {
                "L_nnz": l_nnz,
                "U_nnz": u_nnz,
                "factor_nnz": factor_nnz,
                "fill_ratio": factor_nnz / max(original_nnz, 1),
                "fill_in": factor_nnz - original_nnz,
            }
        except Exception as e:
            results[ordering_name] = {"error": str(e)}

    return {
        "original_nnz": original_nnz,
        "orderings": results,
        "best_ordering": min(
            (k for k, v in results.items() if "error" not in v),
            key=lambda k: results[k]["factor_nnz"],
            default=None,
        ),
    }


# ---------------------------------------------------------------------------
# Matrix structure analysis
# ---------------------------------------------------------------------------


def matrix_structure_analysis(A: sp.spmatrix) -> dict:
    """Analyze structural properties of the matrix."""
    A_csc = sp.csc_matrix(A)
    A_csr = sp.csr_matrix(A)
    n = A_csc.shape[0]

    # Bandwidth
    rows, cols = A_csc.nonzero()
    if len(rows) > 0:
        bandwidth = int(np.max(np.abs(rows - cols)))
        profile = int(np.sum(np.abs(rows - cols)))
    else:
        bandwidth = 0
        profile = 0

    # Degree distribution (treating matrix as adjacency matrix)
    row_nnz = np.diff(A_csr.indptr)
    col_nnz = np.diff(A_csc.indptr)

    # Symmetry check
    A_T = A_csc.T
    sym_diff = A_csc - A_T
    sym_diff.eliminate_zeros()
    is_structurally_symmetric = sym_diff.nnz == 0

    # Check for numerical symmetry
    if is_structurally_symmetric:
        val_diff = np.max(np.abs(A_csc.data - A_T.tocsc().data)) if A_csc.nnz > 0 else 0
        is_numerically_symmetric = val_diff < 1e-10
    else:
        is_numerically_symmetric = False

    # Connected components (treating as undirected graph)
    A_sym_pattern = symmetrize_pattern(A_csc)
    n_components, labels = sp.csgraph.connected_components(A_sym_pattern, directed=False)

    component_sizes = Counter(labels.tolist())
    component_size_list = sorted(component_sizes.values(), reverse=True)

    # Diagonal dominance check
    diag = np.abs(A_csc.diagonal())
    row_sums = np.array(np.abs(A_csr).sum(axis=1)).ravel()
    off_diag_sums = row_sums - diag
    diag_dominant_rows = int(np.sum(diag >= off_diag_sums))

    return {
        "size": n,
        "nnz": A_csc.nnz,
        "density_pct": A_csc.nnz / (n * n) * 100 if n > 0 else 0,
        "bandwidth": bandwidth,
        "profile": profile,
        "is_structurally_symmetric": is_structurally_symmetric,
        "is_numerically_symmetric": is_numerically_symmetric,
        "connected_components": n_components,
        "component_sizes": component_size_list[:10],  # Top 10
        "diagonal_dominance": {
            "dominant_rows": diag_dominant_rows,
            "total_rows": n,
            "pct": diag_dominant_rows / n * 100 if n > 0 else 0,
        },
        "degree_stats": {
            "row_min": int(np.min(row_nnz)) if n > 0 else 0,
            "row_max": int(np.max(row_nnz)) if n > 0 else 0,
            "row_mean": float(np.mean(row_nnz)) if n > 0 else 0,
            "col_min": int(np.min(col_nnz)) if n > 0 else 0,
            "col_max": int(np.max(col_nnz)) if n > 0 else 0,
            "col_mean": float(np.mean(col_nnz)) if n > 0 else 0,
        },
    }


# ---------------------------------------------------------------------------
# RCM ordering analysis
# ---------------------------------------------------------------------------


def rcm_analysis(A: sp.spmatrix) -> dict:
    """Analyze effect of Reverse Cuthill-McKee ordering."""
    A_sym = symmetrize_pattern(A)

    try:
        perm = sp.csgraph.reverse_cuthill_mckee(A_sym, symmetric_mode=True)
        A_rcm = A_sym[perm][:, perm]

        rows_orig, cols_orig = A_sym.nonzero()
        rows_rcm, cols_rcm = A_rcm.nonzero()

        bw_orig = int(np.max(np.abs(rows_orig - cols_orig))) if len(rows_orig) > 0 else 0
        bw_rcm = int(np.max(np.abs(rows_rcm - cols_rcm))) if len(rows_rcm) > 0 else 0

        return {
            "bandwidth_original": bw_orig,
            "bandwidth_rcm": bw_rcm,
            "bandwidth_reduction_pct": (1 - bw_rcm / max(bw_orig, 1)) * 100,
            "permutation_available": True,
        }
    except Exception as e:
        return {"error": str(e), "permutation_available": False}


# ---------------------------------------------------------------------------
# Device scatter pattern analysis
# ---------------------------------------------------------------------------


def analyze_device_scatter(engine) -> dict:
    """Analyze device-to-matrix scatter patterns for assembly parallelism.

    Examines the stamp index mappings to determine:
    - How many matrix positions are written by multiple devices (conflicts)
    - Maximum fan-in to any single position
    - Independence structure between device evaluations
    """
    setup = engine._build_transient_setup(backend="cpu", use_dense=True)
    static_inputs_cache = setup["static_inputs_cache"]
    n_unknowns = setup["n_unknowns"]

    model_info = {}
    # Global position → set of unique (model_type, device_idx) writers
    global_position_writers: dict[tuple[int, int], set[tuple[str, int]]] = {}

    for model_type, (voltage_indices, stamp_indices, *_rest) in static_inputs_cache.items():
        jac_rows = np.asarray(stamp_indices["jac_row_indices"])
        jac_cols = np.asarray(stamp_indices["jac_col_indices"])
        res_indices = np.asarray(stamp_indices["res_indices"])

        n_devices = jac_rows.shape[0]
        n_jac_entries = jac_rows.shape[1]
        n_residuals = res_indices.shape[1]

        # Count unique positions per device
        positions_per_device = []
        for dev_idx in range(n_devices):
            valid = (jac_rows[dev_idx] >= 0) & (jac_cols[dev_idx] >= 0)
            unique_pos = set()
            for j in range(n_jac_entries):
                if valid[j]:
                    pos = (int(jac_rows[dev_idx, j]), int(jac_cols[dev_idx, j]))
                    unique_pos.add(pos)
                    if pos not in global_position_writers:
                        global_position_writers[pos] = set()
                    global_position_writers[pos].add((model_type, dev_idx))
            positions_per_device.append(len(unique_pos))

        # Count touched nodes per device (for residual fan-out)
        nodes_per_device = []
        for dev_idx in range(n_devices):
            valid_nodes = set()
            for r in range(n_residuals):
                idx = int(res_indices[dev_idx, r])
                if idx >= 0:
                    valid_nodes.add(idx)
            nodes_per_device.append(len(valid_nodes))

        model_info[model_type] = {
            "n_devices": n_devices,
            "jac_entries_per_device": n_jac_entries,
            "residuals_per_device": n_residuals,
            "unique_positions_per_device": {
                "min": min(positions_per_device) if positions_per_device else 0,
                "max": max(positions_per_device) if positions_per_device else 0,
                "mean": float(np.mean(positions_per_device)) if positions_per_device else 0,
            },
            "nodes_per_device": {
                "min": min(nodes_per_device) if nodes_per_device else 0,
                "max": max(nodes_per_device) if nodes_per_device else 0,
                "mean": float(np.mean(nodes_per_device)) if nodes_per_device else 0,
            },
        }

    # Analyze scatter conflicts (unique devices per position)
    fan_in_counts = [len(writers) for writers in global_position_writers.values()]
    fan_in_counter = Counter(fan_in_counts)
    conflict_positions = sum(1 for c in fan_in_counts if c > 1)

    # Build device conflict graph: two devices conflict if they write to
    # the same matrix position
    n_total_devices = sum(info["n_devices"] for info in model_info.values())
    conflict_edges = 0
    for writers in global_position_writers.values():
        n_writers = len(writers)
        if n_writers > 1:
            conflict_edges += n_writers * (n_writers - 1) // 2

    return {
        "n_unknowns": n_unknowns,
        "total_devices": n_total_devices,
        "model_types": model_info,
        "scatter_conflicts": {
            "total_positions": len(global_position_writers),
            "conflict_positions": conflict_positions,
            "conflict_pct": conflict_positions / max(len(global_position_writers), 1) * 100,
            "max_fan_in": max(fan_in_counts) if fan_in_counts else 0,
            "fan_in_distribution": {str(k): v for k, v in sorted(fan_in_counter.items())},
        },
        "device_conflict_graph": {
            "n_nodes": n_total_devices,
            "n_edges": conflict_edges,
            "note": "Edges connect devices that write to the same matrix position",
        },
    }


# ---------------------------------------------------------------------------
# Pattern stability check
# ---------------------------------------------------------------------------


def check_pattern_stability(matrices: list[sp.spmatrix]) -> dict:
    """Verify that sparsity pattern is identical across NR iterations.

    This is a key property for IREE: the pattern is fixed, only values change,
    so symbolic analysis can be compiled once and reused.
    """
    if len(matrices) < 2:
        return {
            "is_fixed": True,
            "n_samples": len(matrices),
            "note": "Only one matrix available, cannot verify stability",
        }

    ref = sp.csc_matrix(matrices[0])
    ref_pattern = set(zip(*ref.nonzero()))

    all_match = True
    first_mismatch = None

    for idx, M in enumerate(matrices[1:], 1):
        M_csc = sp.csc_matrix(M)
        M_pattern = set(zip(*M_csc.nonzero()))

        if M_pattern != ref_pattern:
            all_match = False
            added = M_pattern - ref_pattern
            removed = ref_pattern - M_pattern
            first_mismatch = {
                "index": idx,
                "added_entries": len(added),
                "removed_entries": len(removed),
            }
            break

    # Value variation statistics (how much do values change across iterations?)
    if all_match and len(matrices) >= 2:
        values = np.column_stack([sp.csc_matrix(M).data for M in matrices])
        rel_variation = np.std(values, axis=1) / (np.abs(np.mean(values, axis=1)) + 1e-30)
        value_stats = {
            "mean_relative_variation": float(np.mean(rel_variation)),
            "max_relative_variation": float(np.max(rel_variation)),
            "median_relative_variation": float(np.median(rel_variation)),
        }
    else:
        value_stats = None

    return {
        "is_fixed": all_match,
        "n_samples": len(matrices),
        "first_mismatch": first_mismatch,
        "value_variation": value_stats,
        "note": (
            "Sparsity pattern is identical across all samples — symbolic "
            "factorization can be compiled once and reused for all NR iterations"
            if all_match
            else "WARNING: Sparsity pattern changes between iterations"
        ),
    }


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------


def analyze_matrix(
    A: sp.spmatrix,
    name: str = "",
    all_matrices: list[sp.spmatrix] | None = None,
) -> dict:
    """Run full parallelism analysis on a Jacobian matrix.

    Args:
        A: The Jacobian matrix (any sparse format)
        name: Circuit/benchmark name for labeling
        all_matrices: Optional list of matrices for pattern stability check

    Returns:
        Dict with all analysis results
    """
    A_csc = sp.csc_matrix(A, dtype=np.float64)
    n = A_csc.shape[0]

    print(f"Analyzing {n}x{n} matrix ({A_csc.nnz} nonzeros)...")

    # 1. Matrix structure
    print("  Matrix structure...")
    structure = matrix_structure_analysis(A_csc)

    # 2. Elimination tree on symmetrized pattern
    print("  Elimination tree...")
    A_sym = symmetrize_pattern(A_csc)
    parent = compute_etree(A_sym)
    levels = compute_level_sets(parent)
    etree_stats = compute_etree_stats(parent, levels)

    # 3. Supernodes
    print("  Supernodal detection...")
    supernodes = detect_supernodes(parent, A_sym)
    snode_stats = supernode_stats(supernodes)

    # 4. Fill-in analysis
    print("  Fill-in analysis (SuperLU)...")
    fill = fill_in_analysis(A_csc)

    # 5. RCM ordering
    print("  RCM ordering...")
    rcm = rcm_analysis(A_csc)

    # 6. Pattern stability
    stability = None
    if all_matrices and len(all_matrices) > 1:
        print(f"  Pattern stability ({len(all_matrices)} samples)...")
        stability = check_pattern_stability(all_matrices)

    # Compute parallelism summary
    # "Work" at each level = width (number of independent columns)
    # Total sequential steps = height
    # Total work = n (all columns must be processed)
    # Parallelism efficiency = n / height (ideal speedup from parallelism)
    parallelism_efficiency = n / max(etree_stats["height"], 1)

    analysis = {
        "name": name,
        "matrix": structure,
        "_etree_parent": parent,  # Full array, not serialized to JSON
        "elimination_tree": {
            **etree_stats,
            "parallelism_efficiency": parallelism_efficiency,
            "parent_array_sample": parent[:min(50, n)].tolist(),
            "note": (
                f"Height {etree_stats['height']} levels with max width "
                f"{etree_stats['max_parallelism']}. Columns at the same level "
                f"can be factored in parallel. Efficiency = n/height = "
                f"{parallelism_efficiency:.1f}x theoretical speedup."
            ),
        },
        "supernodes": snode_stats,
        "fill_in": fill,
        "rcm_ordering": rcm,
    }

    if stability is not None:
        analysis["pattern_stability"] = stability

    return analysis


# ---------------------------------------------------------------------------
# Device eval branch analysis
# ---------------------------------------------------------------------------


def analyze_eval_branches(engine) -> dict:
    """Analyze jnp.where branches in compiled device eval functions.

    For each model type, checks the compiled model's parameter split to determine:
    - How many device configurations exist (e.g., NMOS vs PMOS)
    - Whether all eval branches are statically determinable at setup time
    - How much specialization is possible

    This does NOT require dumping/parsing generated code — it analyzes the
    actual shared_params, device_params, and device_cache arrays to determine
    how many unique device variants exist.
    """
    result = {}

    for model_type, compiled in engine._compiled_models.items():
        if "shared_params" not in compiled:
            continue

        sp = np.asarray(compiled["shared_params"])
        dp = np.asarray(compiled["device_params"])
        sc = np.asarray(compiled.get("shared_cache", np.array([])))
        dc = np.asarray(compiled.get("device_cache", np.empty((dp.shape[0], 0))))
        vp = np.asarray(compiled.get("voltage_positions_in_varying", np.array([], dtype=int)))

        n_devices = dp.shape[0]
        n_varying = dp.shape[1] if dp.ndim > 1 else 0
        n_voltages = len(vp)
        n_static_varying = n_varying - n_voltages

        # Identify non-voltage varying param columns
        voltage_cols = set(vp.tolist()) if len(vp) > 0 else set()
        static_cols = sorted(set(range(n_varying)) - voltage_cols)

        # Count unique device configurations (static params only)
        if static_cols and n_devices > 1:
            static_dp = dp[:, static_cols]
            unique_configs, config_indices, config_counts = np.unique(
                static_dp, axis=0, return_inverse=True, return_counts=True
            )
            n_unique_configs = len(unique_configs)
            config_sizes = config_counts.tolist()
        elif n_devices > 1:
            # No static varying params — all devices identical
            n_unique_configs = 1
            config_sizes = [n_devices]
        else:
            n_unique_configs = 1
            config_sizes = [1]

        # Check device_cache uniformity
        n_cache_cols = dc.shape[1] if dc.ndim > 1 else 0
        if n_cache_cols > 0 and n_devices > 1:
            cache_uniform = int(np.sum(np.all(dc == dc[0:1, :], axis=0)))
            cache_varying = n_cache_cols - cache_uniform

            # Count unique cache configurations
            unique_dc, dc_indices = np.unique(dc, axis=0, return_inverse=True)
            n_unique_cache = len(unique_dc)
        else:
            cache_uniform = n_cache_cols
            cache_varying = 0
            n_unique_cache = 1

        # Get param names for the varying columns if available
        param_names = compiled.get("param_names", [])
        param_kinds = compiled.get("param_kinds", [])
        varying_indices = compiled.get("varying_indices", [])

        varying_param_info = []
        for col_idx, orig_idx in enumerate(varying_indices):
            if col_idx in voltage_cols:
                continue
            name = param_names[orig_idx] if orig_idx < len(param_names) else f"param_{orig_idx}"
            kind = param_kinds[orig_idx] if orig_idx < len(param_kinds) else "unknown"
            if n_devices > 1:
                vals = dp[:, col_idx]
                unique_vals = np.unique(vals)
                varying_param_info.append({
                    "name": name,
                    "kind": kind,
                    "n_unique": len(unique_vals),
                    "values": unique_vals.tolist() if len(unique_vals) <= 10 else f"{len(unique_vals)} values",
                })

        result[model_type] = {
            "n_devices": n_devices,
            "n_shared_params": len(sp),
            "n_varying_params": n_varying,
            "n_voltage_params": n_voltages,
            "n_static_varying_params": n_static_varying,
            "n_shared_cache": len(sc) if sc.ndim == 1 else (sc.shape[1] if sc.ndim > 1 else 0),
            "n_device_cache_cols": n_cache_cols,
            "cache_uniform_cols": cache_uniform,
            "cache_varying_cols": cache_varying,
            "n_unique_param_configs": n_unique_configs,
            "n_unique_cache_configs": n_unique_cache,
            "config_sizes": config_sizes,
            "varying_static_params": varying_param_info,
            "specialization_note": (
                f"All {n_devices} devices can be grouped into {n_unique_configs} "
                f"specialized eval variant(s). Branches conditioned on shared_params "
                f"({len(sp)} params) and device configuration ({n_static_varying} "
                f"static varying params) can be resolved at compile time, eliminating "
                f"jnp.where overhead for straight-line GPU kernels."
            ),
        }

    return result


# ---------------------------------------------------------------------------
# Benchmark mode: run simulation and analyze
# ---------------------------------------------------------------------------


def analyze_benchmark(
    benchmark_name: str,
    max_captures: int = 20,
    t_stop_override: float | None = None,
) -> dict:
    """Run a benchmark simulation, capture matrices, and analyze parallelism.

    Also captures device scatter pattern information.
    """
    import jax

    from vajax.analysis import CircuitEngine
    from vajax.benchmarks.registry import get_benchmark

    info = get_benchmark(benchmark_name)
    assert info is not None, f"Benchmark '{benchmark_name}' not found"

    engine = CircuitEngine(info.sim_path)
    engine.parse()

    use_sparse = info.is_large
    dt = info.dt
    # Use override, or run just enough steps to capture max_captures matrices
    # (~5 NR iterations per timestep, so max_captures/5 timesteps plus margin)
    if t_stop_override is not None:
        t_stop = t_stop_override
    else:
        # Short simulation: enough for max_captures NR systems
        min_steps = max_captures * 2  # ~2x margin (5 NR iters, capture early ones)
        t_stop = min(dt * min_steps, info.t_stop)

    print(f"Benchmark: {benchmark_name}")
    print(f"  Nodes: {engine.num_nodes}, Devices: {len(engine.devices)}")
    print(f"  Solver: {'sparse' if use_sparse else 'dense'}")
    print(f"  Transient: t_stop={t_stop:.2e}s, dt={dt:.2e}s")

    # Suppress step-by-step logging during simulation
    import logging

    logging.getLogger("vajax").setLevel(logging.WARNING)

    # --- Device scatter analysis (before running simulation) ---
    print("\nAnalyzing device scatter patterns...")
    device_scatter = analyze_device_scatter(engine)

    # --- Capture matrices via monkey-patching ---
    import vajax.analysis.solver_factories as sf

    captured_systems: list[tuple[np.ndarray, np.ndarray]] = []
    csr_info: dict = {}
    capture_count = [0]

    def capture_cb(J_or_data: jax.Array, f: jax.Array):
        if capture_count[0] >= max_captures:
            return
        captured_systems.append((np.asarray(J_or_data).copy(), np.asarray(f).copy()))
        capture_count[0] += 1

    original_make_nr = sf._make_nr_solver_common

    def patched_nr(*, linear_solve_fn, **kwargs):
        def instrumented(J_or_data, f):
            jax.debug.callback(capture_cb, J_or_data, f)
            return linear_solve_fn(J_or_data, f)

        return original_make_nr(linear_solve_fn=instrumented, **kwargs)

    sf._make_nr_solver_common = patched_nr

    # Intercept sparse factories for CSR structure
    for factory_name in ("make_umfpack_ffi_full_mna_solver", "make_spineax_full_mna_solver"):
        original = getattr(sf, factory_name)

        def make_patched(orig):
            def patched(*args, **kwargs):
                n_nodes = args[1] if len(args) > 1 else kwargs.get("n_nodes")
                n_vsources = args[2] if len(args) > 2 else kwargs.get("n_vsources")
                bcsr_indptr = kwargs.get("bcsr_indptr")
                bcsr_indices = kwargs.get("bcsr_indices")
                if bcsr_indptr is None and len(args) > 4:
                    bcsr_indptr = args[4]
                if bcsr_indices is None and len(args) > 5:
                    bcsr_indices = args[5]
                if bcsr_indptr is not None and n_nodes is not None:
                    n_aug = n_nodes - 1 + n_vsources
                    csr_info["indptr"] = np.asarray(bcsr_indptr).copy()
                    csr_info["indices"] = np.asarray(bcsr_indices).copy()
                    csr_info["shape"] = (n_aug, n_aug)
                return orig(*args, **kwargs)

            return patched

        patched = make_patched(original)
        setattr(sf, factory_name, patched)

        import vajax.analysis.transient.full_mna as _full_mna

        setattr(_full_mna, factory_name, patched)

    # --- Run simulation ---
    print("\nRunning simulation...")
    engine.prepare(t_stop=t_stop, dt=dt, use_sparse=use_sparse)
    result = engine.run_transient()

    convergence = result.stats.get("convergence_rate", 0) * 100
    print(f"  Steps: {result.num_steps}, convergence: {convergence:.0f}%")
    print(f"  Captured {len(captured_systems)} linear systems")

    if not captured_systems:
        print("ERROR: No systems captured!", file=sys.stderr)
        sys.exit(1)

    # --- Build sparse matrices from captured data ---
    matrices: list[sp.spmatrix] = []
    for J_or_data, f in captured_systems:
        if use_sparse and "indptr" in csr_info:
            mat = sp.csr_matrix(
                (J_or_data, csr_info["indices"], csr_info["indptr"]),
                shape=csr_info["shape"],
            )
        else:
            mat = sp.csc_matrix(J_or_data)
        matrices.append(mat)

    # --- Analyze ---
    print("\nAnalyzing first captured matrix...")
    analysis = analyze_matrix(matrices[0], name=benchmark_name, all_matrices=matrices)

    # Add device scatter info
    analysis["device_parallelism"] = device_scatter

    # Add circuit info
    analysis["circuit"] = {
        "name": benchmark_name,
        "n_external_nodes": engine.num_nodes,
        "n_devices": len(engine.devices),
        "n_unknowns": device_scatter["n_unknowns"],
        "simulation": {
            "t_stop": t_stop,
            "dt": dt,
            "steps": result.num_steps,
            "convergence_pct": convergence,
        },
    }

    # Eval branch specialization analysis
    print("\nAnalyzing eval function branches...")
    branch_analysis = analyze_eval_branches(engine)
    analysis["eval_specialization"] = branch_analysis

    # Add compilation note for IREE
    analysis["iree_notes"] = {
        "pattern_is_fixed": analysis.get("pattern_stability", {}).get("is_fixed", True),
        "same_pattern_every_nr_iteration": True,
        "values_change_every_iteration": True,
        "typical_nr_iterations_per_step": "3-8",
        "typical_timesteps": f"{result.num_steps}",
        "total_solves": len(captured_systems),
        "recommendation": (
            "The sparsity pattern is determined at circuit parse time and never changes. "
            "Symbolic factorization (ordering, elimination tree, memory allocation) can "
            "be compiled once. Only numerical factorization needs to run per NR iteration. "
            f"For this circuit: {result.num_steps} timesteps x ~5 NR iters = "
            f"~{result.num_steps * 5} factorizations with identical structure."
        ),
    }

    return analysis


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_analysis(analysis: dict, output_dir: Path):
    """Write analysis results to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON output (machine-readable)
    json_path = output_dir / "parallelism_analysis.json"

    # Remove internal data not suitable for JSON
    analysis_json = {k: v for k, v in analysis.items() if not k.startswith("_")}
    analysis_json = json.loads(json.dumps(analysis_json, default=str))
    widths = analysis_json.get("elimination_tree", {}).get("level_widths", [])
    if len(widths) > 100:
        analysis_json["elimination_tree"]["level_widths_truncated"] = widths[:50] + ["..."] + widths[-50:]
        del analysis_json["elimination_tree"]["level_widths"]

    with open(json_path, "w") as f:
        json.dump(analysis_json, f, indent=2)
    print(f"  JSON: {json_path}")

    # Human-readable summary
    summary_path = output_dir / "parallelism_summary.txt"
    with open(summary_path, "w") as f:
        name = analysis.get("name", "unknown")
        f.write(f"{'=' * 70}\n")
        f.write(f"Parallelism Analysis: {name}\n")
        f.write(f"{'=' * 70}\n\n")

        mat = analysis["matrix"]
        f.write(f"Matrix: {mat['size']}x{mat['size']}, {mat['nnz']} nonzeros ({mat['density_pct']:.4f}%)\n")
        f.write(f"Bandwidth: {mat['bandwidth']}, Symmetric: {mat['is_structurally_symmetric']}\n")
        f.write(f"Connected components: {mat['connected_components']}\n")
        deg = mat["degree_stats"]
        f.write(f"Row degree: min={deg['row_min']}, max={deg['row_max']}, mean={deg['row_mean']:.1f}\n")
        f.write(f"Diagonal dominance: {mat['diagonal_dominance']['pct']:.1f}% of rows\n\n")

        et = analysis["elimination_tree"]
        f.write("--- Elimination Tree ---\n")
        f.write(f"Height (sequential steps): {et['height']}\n")
        f.write(f"Leaves: {et['n_leaves']}\n")
        f.write(f"Max parallelism (widest level): {et['max_parallelism']}\n")
        f.write(f"Avg parallelism: {et['avg_parallelism']:.1f}\n")
        f.write(f"Parallelism efficiency (n/height): {et['parallelism_efficiency']:.1f}x\n")
        st = et["subtree_size_stats"]
        f.write(f"Subtree sizes: min={st['min']}, max={st['max']}, median={st['median']:.0f}\n\n")

        sn = analysis["supernodes"]
        f.write("--- Supernodes ---\n")
        f.write(f"Count: {sn['count']} supernodes\n")
        f.write(f"Largest: {sn['largest']} columns\n")
        f.write(f"Mean size: {sn['mean_size']:.1f}\n")
        f.write(f"Size distribution: {sn['size_histogram']}\n\n")

        fi = analysis["fill_in"]
        f.write("--- Fill-in (LU factorization) ---\n")
        f.write(f"Original nnz: {fi['original_nnz']}\n")
        for order_name, order_data in fi["orderings"].items():
            if "error" not in order_data:
                f.write(
                    f"  {order_name}: factor_nnz={order_data['factor_nnz']}, "
                    f"fill_ratio={order_data['fill_ratio']:.2f}x, "
                    f"fill_in=+{order_data['fill_in']}\n"
                )
        if fi["best_ordering"]:
            f.write(f"Best ordering: {fi['best_ordering']}\n")
        f.write("\n")

        rcm = analysis.get("rcm_ordering", {})
        if rcm.get("permutation_available"):
            f.write("--- RCM Ordering ---\n")
            f.write(f"Bandwidth: {rcm['bandwidth_original']} -> {rcm['bandwidth_rcm']} ")
            f.write(f"({rcm['bandwidth_reduction_pct']:.1f}% reduction)\n\n")

        ps = analysis.get("pattern_stability")
        if ps:
            f.write("--- Pattern Stability ---\n")
            f.write(f"Fixed pattern: {ps['is_fixed']} ({ps['n_samples']} samples)\n")
            if ps.get("value_variation"):
                vv = ps["value_variation"]
                f.write(f"Value variation: mean_rel={vv['mean_relative_variation']:.4f}, ")
                f.write(f"max_rel={vv['max_relative_variation']:.4f}\n")
            f.write(f"{ps['note']}\n\n")

        dp = analysis.get("device_parallelism")
        if dp:
            f.write("--- Device Evaluation Parallelism ---\n")
            f.write(f"Total devices: {dp['total_devices']}\n")
            for mt, mi in dp["model_types"].items():
                f.write(f"  {mt}: {mi['n_devices']} devices, ")
                f.write(f"{mi['jac_entries_per_device']} Jacobian entries/device, ")
                f.write(f"{mi['nodes_per_device']['mean']:.0f} nodes/device\n")
            sc = dp["scatter_conflicts"]
            f.write(f"Scatter conflicts: {sc['conflict_positions']}/{sc['total_positions']} positions ")
            f.write(f"({sc['conflict_pct']:.1f}%), max fan-in={sc['max_fan_in']}\n")
            f.write(f"Fan-in distribution: {sc['fan_in_distribution']}\n\n")

        es = analysis.get("eval_specialization")
        if es:
            f.write("--- Eval Branch Specialization ---\n")
            for mt, info in es.items():
                f.write(f"  {mt}: {info['n_devices']} devices\n")
                f.write(f"    Params: {info['n_shared_params']} shared, ")
                f.write(f"{info['n_voltage_params']} voltage, ")
                f.write(f"{info['n_static_varying_params']} static-varying\n")
                f.write(f"    Cache: {info['n_shared_cache']} shared, ")
                f.write(f"{info['cache_varying_cols']} device-varying\n")
                f.write(f"    Unique device configs: {info['n_unique_param_configs']}")
                f.write(f" (sizes: {info['config_sizes']})\n")
                if info.get("varying_static_params"):
                    for vp in info["varying_static_params"]:
                        f.write(f"      {vp['name']} ({vp['kind']}): {vp['n_unique']} unique values\n")
                f.write(f"    {info['specialization_note']}\n")
            f.write("\n")

        notes = analysis.get("iree_notes")
        if notes:
            f.write("--- IREE/Baspacho Notes ---\n")
            f.write(f"{notes['recommendation']}\n")

    print(f"  Summary: {summary_path}")

    # Write full elimination tree parent array (useful for solver development)
    if "_etree_parent" in analysis:
        etree_path = output_dir / "etree_parent.npy"
        np.save(etree_path, analysis["_etree_parent"])
        print(f"  Etree parent: {etree_path} ({len(analysis['_etree_parent'])} nodes)")

    # Write level-set widths as CSV (for plotting)
    widths = analysis.get("elimination_tree", {}).get("level_widths", [])
    if widths:
        widths_path = output_dir / "level_set_widths.csv"
        with open(widths_path, "w") as f:
            f.write("level,width\n")
            for i, w in enumerate(widths):
                f.write(f"{i},{w}\n")
        print(f"  Level widths: {widths_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Analyze parallelism opportunities in VAJAX simulation matrices"
    )
    parser.add_argument(
        "benchmark",
        nargs="?",
        help="Benchmark name (e.g. ring, c6288, graetz)",
    )
    parser.add_argument(
        "--from-mtx",
        type=Path,
        nargs="+",
        help="Analyze existing Matrix Market file(s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: /tmp/claude/<benchmark>_parallelism)",
    )
    parser.add_argument(
        "--max-captures",
        type=int,
        default=20,
        help="Max NR systems to capture for pattern stability check",
    )
    parser.add_argument(
        "--t-stop",
        type=float,
        default=None,
        help="Override transient stop time (default: auto-short for analysis)",
    )
    args = parser.parse_args()

    if args.from_mtx:
        # Load from Matrix Market files
        matrices = []
        for path in args.from_mtx:
            print(f"Loading {path}...")
            matrices.append(scipy.io.mmread(path))

        name = args.from_mtx[0].stem.replace("jacobian_", "")
        analysis = analyze_matrix(matrices[0], name=name, all_matrices=matrices)

        out_dir = args.output_dir or Path(f"/tmp/claude/{name}_parallelism")
        write_analysis(analysis, out_dir)

    elif args.benchmark:
        analysis = analyze_benchmark(
            args.benchmark,
            max_captures=args.max_captures,
            t_stop_override=args.t_stop,
        )
        out_dir = args.output_dir or Path(f"/tmp/claude/{args.benchmark}_parallelism")
        write_analysis(analysis, out_dir)

    else:
        parser.print_help()
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
