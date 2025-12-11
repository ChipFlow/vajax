"""GPU-native DC operating point analysis with analytical Jacobians

This module implements a fully GPU-resident Newton-Raphson solver that uses
analytical Jacobian computation. The analytical approach computes explicit
gm/gds stamps for each device, avoiding autodiff issues that cause convergence
problems with floating nodes.

Key advantages:
1. Convergence with floating nodes (series NMOS stacks, AND gates, etc.)
2. Explicit gds_min ensures non-singular Jacobians in cutoff
3. All iterations can run on GPU with jax.lax.while_loop
4. No autodiff overhead for Jacobian computation
"""

from typing import Callable, Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import sparse as jsparse
import numpy as np

from jax_spice.analysis.mna import MNASystem, DeviceInfo, DeviceType
from jax_spice.analysis.context import AnalysisContext


def _bcoo_to_csr(bcoo_matrix, n: int) -> Tuple[Array, Array, Array]:
    """Convert BCOO sparse matrix to CSR arrays for spsolve.

    BCOO format: data (n_nnz,), indices (n_nnz, 2) with [row, col]
    CSR format: data (n_nnz,), indices (n_nnz,) columns, indptr (n+1,) row pointers

    Args:
        bcoo_matrix: JAX BCOO sparse matrix
        n: Number of rows (= number of columns for square matrix)

    Returns:
        Tuple of (data, indices, indptr) in CSR format
    """
    # Extract BCOO components
    data = bcoo_matrix.data
    bcoo_indices = bcoo_matrix.indices  # (n_nnz, 2) with [row, col]

    rows = bcoo_indices[:, 0]
    cols = bcoo_indices[:, 1]

    # Sort by row then column for CSR format
    # Use lexsort: sort by cols first (secondary), then rows (primary)
    sort_idx = jnp.lexsort((cols, rows))
    data = data[sort_idx]
    rows = rows[sort_idx]
    cols = cols[sort_idx]

    # Build indptr: count entries per row
    # indptr[i] = number of entries in rows 0..i-1
    row_counts = jnp.zeros(n + 1, dtype=jnp.int32)
    row_counts = row_counts.at[rows + 1].add(1)
    indptr = jnp.cumsum(row_counts)

    return data, cols.astype(jnp.int32), indptr


def eval_param_simple(value, vdd: float = 1.2, defaults: dict = None):
    """Simple parameter evaluation for common cases.

    Handles:
    - Numbers (int, float)
    - String 'vdd' -> vdd value
    - String '0' -> 0.0
    - String 'w', 'l' -> default MOSFET dimensions
    - SPICE number suffixes (1u, 100n, etc.)
    """
    if defaults is None:
        defaults = {
            'w': 1e-6,      # Default MOSFET width = 1u
            'l': 0.2e-6,    # Default MOSFET length = 0.2u
            'ld': 0.5e-6,   # Default drain extension
            'ls': 0.5e-6,   # Default source extension
        }

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        value_lower = value.lower().strip()

        # Common parameter references
        if value_lower == 'vdd':
            return vdd
        if value_lower in ('0', '0.0', 'vss', 'gnd'):
            return 0.0
        if value_lower in defaults:
            return defaults[value_lower]

        # SPICE number suffixes
        suffixes = {
            't': 1e12, 'g': 1e9, 'meg': 1e6, 'k': 1e3,
            'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15
        }
        for suffix, mult in sorted(suffixes.items(), key=lambda x: -len(x[0])):
            if value_lower.endswith(suffix):
                try:
                    return float(value_lower[:-len(suffix)]) * mult
                except ValueError:
                    pass

        # Try direct conversion
        try:
            return float(value)
        except ValueError:
            pass

    return 0.0


@dataclass
class AnalyticalResidualFunction:
    """Encapsulates a residual function with analytical Jacobian.

    Computes both residual and Jacobian in a single pass using
    analytical device models with explicit gm/gds computation.
    """
    num_nodes: int
    residual_and_jacobian_fn: Callable[[Array], Tuple[Array, jsparse.BCOO]]
    sparsity_indices: Tuple[Array, Array]

    @property
    def n_reduced(self) -> int:
        return self.num_nodes - 1


def build_analytical_residual_and_jacobian_fn(
    system: MNASystem,
    gmin: float = 1e-9,
    vdd_scale: float = 1.0,
    vdd: float = 1.2,
) -> AnalyticalResidualFunction:
    """Build residual and Jacobian function with analytical device models.

    This avoids autodiff entirely by computing analytical Jacobian stamps
    for each device type. This fixes convergence issues caused by autodiff
    giving near-zero conductances in cutoff regions.

    Args:
        system: MNA system with devices
        gmin: Minimum conductance from each node to ground
        vdd_scale: Scale factor for voltage sources
        vdd: Supply voltage

    Returns:
        AnalyticalResidualFunction with combined residual/Jacobian function
    """
    num_nodes = system.num_nodes
    n_reduced = num_nodes - 1

    # Collect device data
    vsource_data = {'node_p': [], 'node_n': [], 'v_target': []}
    resistor_data = {'node_p': [], 'node_n': [], 'conductance': []}
    mosfet_data = {'node_d': [], 'node_g': [], 'node_s': [], 'node_b': [],
                   'W': [], 'L': [], 'is_pmos': []}

    # Sparsity pattern
    sparsity_set = set()

    for device in system.devices:
        model_lower = device.model_name.lower()

        is_vsource = (
            'vsource' in model_lower or 'vdc' in model_lower or
            model_lower == 'v' or
            (model_lower.startswith('v') and len(model_lower) <= 2)
        )

        is_nmos = 'nmos' in model_lower or (model_lower.endswith('n') and 'psp' in model_lower)
        is_pmos = 'pmos' in model_lower or (model_lower.endswith('p') and 'psp' in model_lower)
        is_mosfet = is_nmos or is_pmos

        if is_vsource:
            node_p = device.node_indices[0]
            node_n = device.node_indices[1]
            v_raw = device.params.get('v', device.params.get('dc', 0.0))
            v_target = eval_param_simple(v_raw, vdd=vdd) * vdd_scale

            vsource_data['node_p'].append(node_p)
            vsource_data['node_n'].append(node_n)
            vsource_data['v_target'].append(v_target)

            # Sparsity: vsource stamps 2x2 block
            for ni in [node_p, node_n]:
                for nj in [node_p, node_n]:
                    if ni > 0 and nj > 0:
                        sparsity_set.add((ni - 1, nj - 1))

        elif is_mosfet:
            node_d = device.node_indices[0]
            node_g = device.node_indices[1]
            node_s = device.node_indices[2]
            node_b = device.node_indices[3] if len(device.node_indices) > 3 else node_s

            W = eval_param_simple(device.params.get('w', 1e-6), vdd=vdd)
            L = eval_param_simple(device.params.get('l', 0.2e-6), vdd=vdd)
            W = max(W, 1e-9)
            L = max(L, 1e-9)

            mosfet_data['node_d'].append(node_d)
            mosfet_data['node_g'].append(node_g)
            mosfet_data['node_s'].append(node_s)
            mosfet_data['node_b'].append(node_b)
            mosfet_data['W'].append(float(W))
            mosfet_data['L'].append(float(L))
            mosfet_data['is_pmos'].append(is_pmos)

            # MOSFET Jacobian: D,S rows affected by D,G,S,B columns
            terminals = [node_d, node_g, node_s, node_b]
            current_nodes = [node_d, node_s]
            for ni in current_nodes:
                for nj in terminals:
                    if ni > 0 and nj > 0:
                        sparsity_set.add((ni - 1, nj - 1))

        else:
            is_resistor = (
                model_lower == 'r' or model_lower == 'resistor' or
                (model_lower.startswith('r') and len(model_lower) <= 2)
            )
            if is_resistor:
                node_p = device.node_indices[0]
                node_n = device.node_indices[1]
                r_value = eval_param_simple(
                    device.params.get('r', device.params.get('value', 1e6)), vdd=vdd
                )
                r_value = max(r_value, 1e-9)

                resistor_data['node_p'].append(node_p)
                resistor_data['node_n'].append(node_n)
                resistor_data['conductance'].append(1.0 / r_value)

                for ni in [node_p, node_n]:
                    for nj in [node_p, node_n]:
                        if ni > 0 and nj > 0:
                            sparsity_set.add((ni - 1, nj - 1))

    # Add GMIN diagonal
    for i in range(n_reduced):
        sparsity_set.add((i, i))

    # Build sorted sparsity arrays
    sparsity_list = sorted(sparsity_set)
    sparsity_rows = jnp.array([r for r, c in sparsity_list], dtype=jnp.int32)
    sparsity_cols = jnp.array([c for r, c in sparsity_list], dtype=jnp.int32)
    n_nnz = len(sparsity_list)

    # Build index lookup for fast stamping
    sparsity_to_idx = {(r, c): i for i, (r, c) in enumerate(sparsity_list)}

    # Convert to JAX arrays
    vsource_arrays = {
        'node_p': jnp.array(vsource_data['node_p'], dtype=jnp.int32),
        'node_n': jnp.array(vsource_data['node_n'], dtype=jnp.int32),
        'v_target': jnp.array(vsource_data['v_target'], dtype=jnp.float64),
    }
    resistor_arrays = {
        'node_p': jnp.array(resistor_data['node_p'], dtype=jnp.int32),
        'node_n': jnp.array(resistor_data['node_n'], dtype=jnp.int32),
        'conductance': jnp.array(resistor_data['conductance'], dtype=jnp.float64),
    }
    mosfet_arrays = {
        'node_d': jnp.array(mosfet_data['node_d'], dtype=jnp.int32),
        'node_g': jnp.array(mosfet_data['node_g'], dtype=jnp.int32),
        'node_s': jnp.array(mosfet_data['node_s'], dtype=jnp.int32),
        'node_b': jnp.array(mosfet_data['node_b'], dtype=jnp.int32),
        'W': jnp.array(mosfet_data['W'], dtype=jnp.float64),
        'L': jnp.array(mosfet_data['L'], dtype=jnp.float64),
        'is_pmos': jnp.array(mosfet_data['is_pmos'], dtype=jnp.bool_),
    }

    # Precompute Jacobian stamp indices for each device type
    # Voltage source stamps
    vsource_jac_indices = []
    for i in range(len(vsource_data['node_p'])):
        np_i, nn_i = vsource_data['node_p'][i], vsource_data['node_n'][i]
        indices = {}
        for (ni, nj), sign in [((np_i, np_i), 1), ((np_i, nn_i), -1),
                                ((nn_i, np_i), -1), ((nn_i, nn_i), 1)]:
            if ni > 0 and nj > 0:
                idx = sparsity_to_idx.get((ni - 1, nj - 1))
                if idx is not None:
                    indices[(ni, nj)] = (idx, sign)
        vsource_jac_indices.append(indices)

    # Resistor stamps
    resistor_jac_indices = []
    for i in range(len(resistor_data['node_p'])):
        np_i, nn_i = resistor_data['node_p'][i], resistor_data['node_n'][i]
        indices = {}
        for (ni, nj), sign in [((np_i, np_i), 1), ((np_i, nn_i), -1),
                                ((nn_i, np_i), -1), ((nn_i, nn_i), 1)]:
            if ni > 0 and nj > 0:
                idx = sparsity_to_idx.get((ni - 1, nj - 1))
                if idx is not None:
                    indices[(ni, nj)] = (idx, sign)
        resistor_jac_indices.append(indices)

    # MOSFET stamps: D and S rows, D/G/S/B columns
    mosfet_jac_indices = []
    for i in range(len(mosfet_data['node_d'])):
        nd = mosfet_data['node_d'][i]
        ng = mosfet_data['node_g'][i]
        ns = mosfet_data['node_s'][i]
        nb = mosfet_data['node_b'][i]
        indices = {}
        # Key: (row_terminal, col_terminal) -> (sparse_idx, ...)
        for row_node, row_name in [(nd, 'D'), (ns, 'S')]:
            for col_node, col_name in [(nd, 'D'), (ng, 'G'), (ns, 'S'), (nb, 'B')]:
                if row_node > 0 and col_node > 0:
                    idx = sparsity_to_idx.get((row_node - 1, col_node - 1))
                    if idx is not None:
                        indices[(row_name, col_name)] = idx
        mosfet_jac_indices.append(indices)

    # GMIN diagonal indices
    gmin_indices = [sparsity_to_idx[(i, i)] for i in range(n_reduced)]

    def residual_and_jacobian_fn(V_reduced: Array) -> Tuple[Array, jsparse.BCOO]:
        """Compute residual and analytical Jacobian."""
        V_full = jnp.concatenate([jnp.array([0.0]), V_reduced])

        residual = jnp.zeros(num_nodes, dtype=V_full.dtype)
        jac_data = jnp.zeros(n_nnz, dtype=V_full.dtype)

        # === Voltage sources ===
        if len(vsource_data['node_p']) > 0:
            G_big = 1e6
            Vp = V_full[vsource_arrays['node_p']]
            Vn = V_full[vsource_arrays['node_n']]
            V_actual = Vp - Vn
            I = G_big * (V_actual - vsource_arrays['v_target'])

            residual = residual.at[vsource_arrays['node_p']].add(I)
            residual = residual.at[vsource_arrays['node_n']].add(-I)

            # Jacobian stamps
            for i, indices in enumerate(vsource_jac_indices):
                for (ni, nj), (idx, sign) in indices.items():
                    jac_data = jac_data.at[idx].add(sign * G_big)

        # === Resistors ===
        if len(resistor_data['node_p']) > 0:
            Vp = V_full[resistor_arrays['node_p']]
            Vn = V_full[resistor_arrays['node_n']]
            I = resistor_arrays['conductance'] * (Vp - Vn)

            residual = residual.at[resistor_arrays['node_p']].add(I)
            residual = residual.at[resistor_arrays['node_n']].add(-I)

            for i, indices in enumerate(resistor_jac_indices):
                G = float(resistor_data['conductance'][i])
                for (ni, nj), (idx, sign) in indices.items():
                    jac_data = jac_data.at[idx].add(sign * G)

        # === MOSFETs with analytical Jacobian ===
        # Use level-1 Shichman-Hodges model with explicit Jacobian stamps
        # Sign convention: residual = current OUT of node (KCL: sum out = 0)
        if len(mosfet_data['node_d']) > 0:
            Vd = V_full[mosfet_arrays['node_d']]
            Vg = V_full[mosfet_arrays['node_g']]
            Vs = V_full[mosfet_arrays['node_s']]
            Vb = V_full[mosfet_arrays['node_b']]
            W = mosfet_arrays['W']
            L = mosfet_arrays['L']
            is_pmos = mosfet_arrays['is_pmos']

            # Level-1 parameters
            vth0 = 0.4
            kp = 200e-6
            lambda_ = 0.01
            gds_min = 1e-9  # Minimum conductance for cutoff

            # Compute terminal voltages (always positive Vgs, Vds for the model)
            Vgs_nmos = Vg - Vs
            Vds_nmos = Vd - Vs
            Vgs_pmos = Vs - Vg
            Vds_pmos = Vs - Vd

            Vgs = jnp.where(is_pmos, Vgs_pmos, Vgs_nmos)
            Vds = jnp.where(is_pmos, Vds_pmos, Vds_nmos)

            beta = kp * W / L
            Vov = Vgs - vth0

            # Compute Id, gm, gds analytically for each region
            # Cutoff: Vov <= 0
            Id_cutoff = gds_min * Vds
            gm_cutoff = jnp.zeros_like(Vov)
            gds_cutoff = jnp.full_like(Vov, gds_min)

            # Linear: Vov > 0 and Vds < Vov
            Id_lin = beta * (Vov * Vds - 0.5 * Vds * Vds) * (1 + lambda_ * Vds)
            gm_lin = beta * Vds * (1 + lambda_ * Vds)
            gds_lin = beta * (Vov - Vds) * (1 + lambda_ * Vds) + \
                      beta * (Vov * Vds - 0.5 * Vds * Vds) * lambda_

            # Saturation: Vov > 0 and Vds >= Vov
            Id_sat = 0.5 * beta * Vov * Vov * (1 + lambda_ * Vds)
            gm_sat = beta * Vov * (1 + lambda_ * Vds)
            gds_sat = 0.5 * beta * Vov * Vov * lambda_

            # Select region
            in_cutoff = Vov <= 0
            in_linear = (~in_cutoff) & (Vds < Vov)

            Id = jnp.where(in_cutoff, Id_cutoff,
                          jnp.where(in_linear, Id_lin, Id_sat))
            gm = jnp.where(in_cutoff, gm_cutoff,
                          jnp.where(in_linear, gm_lin, gm_sat))
            gds = jnp.where(in_cutoff, gds_cutoff,
                           jnp.where(in_linear, gds_lin, gds_sat))

            # Add minimum conductance (always active for numerical stability)
            Id = Id + gds_min * Vds
            gds = jnp.maximum(gds + gds_min, gds_min)
            gm = jnp.maximum(gm, 1e-12)

            # Current sign convention: residual = current OUT of node
            # NMOS: Ids flows D→S, so current LEAVES drain (-), ENTERS source (+)
            # PMOS: Ids flows S→D, so current ENTERS drain (+), LEAVES source (-)
            # BUT we negate because residual convention is current OUT
            Id_drain = jnp.where(is_pmos, -Id, Id)   # PMOS: -Id (in), NMOS: +Id (out)
            Id_source = jnp.where(is_pmos, Id, -Id)  # PMOS: +Id (out), NMOS: -Id (in)

            residual = residual.at[mosfet_arrays['node_d']].add(Id_drain)
            residual = residual.at[mosfet_arrays['node_s']].add(Id_source)

            # Analytical Jacobian stamps using conductance matrix
            # These are the SAME for both NMOS and PMOS (conductances are always positive)
            # The sign differences are handled by the current convention above
            for i, indices in enumerate(mosfet_jac_indices):
                gm_i = float(gm[i])
                gds_i = float(gds[i])

                # Standard MOSFET conductance stamps (Y-matrix)
                stamps = {
                    ('D', 'D'): gds_i,
                    ('D', 'G'): gm_i,
                    ('D', 'S'): -gds_i - gm_i,
                    ('S', 'D'): -gds_i,
                    ('S', 'G'): -gm_i,
                    ('S', 'S'): gds_i + gm_i,
                }

                for (row, col), val in stamps.items():
                    if (row, col) in indices:
                        idx = indices[(row, col)]
                        jac_data = jac_data.at[idx].add(val)

        # === GMIN diagonal ===
        residual = residual.at[1:].add(gmin * V_reduced)
        for i, idx in enumerate(gmin_indices):
            jac_data = jac_data.at[idx].add(gmin)

        # Build sparse Jacobian
        jac_indices = jnp.stack([sparsity_rows, sparsity_cols], axis=-1)
        J = jsparse.BCOO((jac_data, jac_indices), shape=(n_reduced, n_reduced))

        return residual[1:], J

    return AnalyticalResidualFunction(
        num_nodes=num_nodes,
        residual_and_jacobian_fn=residual_and_jacobian_fn,
        sparsity_indices=(sparsity_rows, sparsity_cols),
    )


def dc_operating_point_analytical(
    system: MNASystem,
    initial_guess: Optional[Array] = None,
    max_iterations: int = 100,
    abstol: float = 1e-9,
    reltol: float = 1e-3,
    damping: float = 1.0,
    vdd: float = 1.2,
    gmin: float = 1e-9,
    verbose: bool = False,
) -> Tuple[Array, Dict]:
    """DC operating point using analytical Jacobians (no autodiff).

    This solver uses analytically computed Jacobian stamps instead of
    automatic differentiation. This ensures proper conductance values
    even in cutoff regions, fixing convergence issues with floating nodes.

    Args:
        system: MNA system with devices
        initial_guess: Initial voltage estimate
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance
        reltol: Relative tolerance
        damping: Damping factor
        vdd: Supply voltage
        gmin: Minimum conductance
        verbose: Print iteration details

    Returns:
        Tuple of (solution, info)
    """
    n = system.num_nodes
    n_reduced = n - 1

    # Build analytical residual/Jacobian function
    analytical_fn = build_analytical_residual_and_jacobian_fn(
        system, gmin=gmin, vdd=vdd
    )
    res_jac_fn = analytical_fn.residual_and_jacobian_fn

    # Initialize
    if initial_guess is not None:
        V = jnp.array(initial_guess[1:], dtype=jnp.float64)
    else:
        V = jnp.zeros(n_reduced, dtype=jnp.float64)
        for name, idx in system.node_names.items():
            if 'vdd' in name.lower() and idx > 0:
                V = V.at[idx - 1].set(vdd)

    converged = False
    iterations = 0
    residual_norm = 1e20
    delta_norm = 0.0
    residual_history = []

    for iteration in range(max_iterations):
        f, J = res_jac_fn(V)
        residual_norm = float(jnp.max(jnp.abs(f)))
        residual_history.append(residual_norm)

        if verbose and iteration < 20:
            print(f"  Iter {iteration}: residual={residual_norm:.2e}")

        if residual_norm < abstol:
            converged = True
            iterations = iteration + 1
            break

        # Solve J * delta_V = -f
        backend = jax.default_backend()
        if backend in ('gpu', 'cuda'):
            from jax.experimental.sparse.linalg import spsolve as jax_spsolve
            J_data, J_indices, J_indptr = _bcoo_to_csr(J, n_reduced)
            delta_V = jax_spsolve(J_data, J_indices, J_indptr, -f, tol=0)
        else:
            J_dense = J.todense()
            try:
                delta_V = jnp.linalg.solve(J_dense, -f)
            except Exception:
                reg = 1e-10 * jnp.eye(n_reduced)
                delta_V = jnp.linalg.solve(J_dense + reg, -f)

        # Update with damping and limiting
        max_step = 2.0
        max_delta = jnp.max(jnp.abs(delta_V))
        step_scale = jnp.minimum(damping, max_step / (max_delta + 1e-15))

        V = V + step_scale * delta_V
        V = jnp.clip(V, -vdd * 2.0, vdd * 2.0)

        iterations = iteration + 1
        delta_norm = float(jnp.max(jnp.abs(step_scale * delta_V)))
        v_norm = float(jnp.max(jnp.abs(V)))

        if delta_norm < abstol + reltol * max(v_norm, 1.0):
            converged = True
            break

    V_full = jnp.concatenate([jnp.array([0.0]), V])

    info = {
        'converged': converged,
        'iterations': iterations,
        'residual_norm': residual_norm,
        'delta_norm': delta_norm,
        'residual_history': residual_history,
        'method': 'analytical_jacobian',
    }

    return V_full, info
