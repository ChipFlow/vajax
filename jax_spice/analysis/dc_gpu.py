"""GPU-native DC operating point analysis using sparsejac

This module implements a fully GPU-resident Newton-Raphson solver that keeps
all data on GPU and uses automatic differentiation via sparsejac to compute
sparse Jacobians efficiently.

Key advantages over the standard sparse solver:
1. No Python loops during iteration - everything compiled via JAX
2. Sparse Jacobian computed via graph coloring, not explicit stamping
3. All iterations run on GPU with jax.lax.while_loop
4. Convergence checks don't force GPU->CPU sync
"""

from typing import Callable, Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import sparse as jsparse
import numpy as np

try:
    import sparsejac
    HAS_SPARSEJAC = True
except ImportError:
    HAS_SPARSEJAC = False
    sparsejac = None

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


@dataclass
class GPUResidualFunction:
    """Encapsulates a pure JAX residual function for a circuit.

    The residual function f(V) returns the current imbalance at each node.
    For DC operating point, we solve f(V) = 0.

    This class captures all circuit topology and parameters as static data,
    returning a pure function that only depends on voltage array V.
    """
    num_nodes: int
    residual_fn: Callable[[Array], Array]
    sparsity_indices: Tuple[Array, Array]  # (row_indices, col_indices)

    @property
    def n_reduced(self) -> int:
        """Number of nodes excluding ground"""
        return self.num_nodes - 1


def build_vsource_residual_fn(
    node_p_indices: Array,
    node_n_indices: Array,
    v_targets: Array,
    g_big: float = 1e6,
) -> Callable[[Array], Array]:
    """Build residual contribution from voltage sources.

    Args:
        node_p_indices: Positive terminal node indices (n_sources,)
        node_n_indices: Negative terminal node indices (n_sources,)
        v_targets: Target voltages (n_sources,)
        g_big: Large conductance for voltage enforcement

    Returns:
        Function f(V) -> residual_contribution (num_nodes,)
    """
    def residual_fn(V: Array, num_nodes: int) -> Array:
        # Get terminal voltages
        Vp = V[node_p_indices]
        Vn = V[node_n_indices]
        V_actual = Vp - Vn

        # Current: I = G_big * (V_actual - V_target)
        I = g_big * (V_actual - v_targets)

        # Stamp into residual: f[p] += I, f[n] -= I
        residual = jnp.zeros(num_nodes, dtype=V.dtype)
        residual = residual.at[node_p_indices].add(I)
        residual = residual.at[node_n_indices].add(-I)

        return residual

    return residual_fn


def build_mosfet_residual_fn(
    node_d_indices: Array,
    node_g_indices: Array,
    node_s_indices: Array,
    node_b_indices: Array,
    W_values: Array,
    L_values: Array,
    is_pmos: Array,
    vth0: float = 0.4,
    kp: float = 200e-6,
    lambda_: float = 0.01,
) -> Callable[[Array], Array]:
    """Build residual contribution from MOSFETs using level-1 model.

    This implements a simplified level-1 MOSFET model suitable for digital circuits.
    The model is fully differentiable for automatic Jacobian computation.

    Args:
        node_d_indices: Drain node indices (n_mosfets,)
        node_g_indices: Gate node indices (n_mosfets,)
        node_s_indices: Source node indices (n_mosfets,)
        node_b_indices: Bulk node indices (n_mosfets,)
        W_values: Width values (n_mosfets,)
        L_values: Length values (n_mosfets,)
        is_pmos: Boolean array, True for PMOS (n_mosfets,)
        vth0: Threshold voltage magnitude
        kp: Transconductance parameter
        lambda_: Channel length modulation

    Returns:
        Function f(V) -> residual_contribution (num_nodes,)
    """
    def mosfet_ids_batch(Vgs: Array, Vds: Array, W: Array, L: Array) -> Array:
        """Batch MOSFET drain current computation (level-1 model)"""
        beta = kp * W / L

        # Gate overdrive
        Vgst = Vgs - vth0

        # Smooth cutoff region (subthreshold)
        # Use soft-plus for smooth transition
        Vgst_eff = jnp.maximum(Vgst, 0.0) + 0.01 * jnp.log1p(jnp.exp(Vgst / 0.01))

        # Saturation voltage
        Vdsat = jnp.maximum(Vgst_eff, 1e-6)

        # Linear vs saturation (smooth transition)
        Vds_eff = Vds * jnp.tanh(Vds / jnp.maximum(Vdsat, 0.01))

        # Drain current with channel length modulation
        Ids = beta * Vgst_eff * Vds_eff * (1 + lambda_ * jnp.abs(Vds))

        # Ensure non-negative current
        Ids = jnp.maximum(Ids, 0.0)

        return Ids

    def residual_fn(V: Array, num_nodes: int) -> Array:
        # Get terminal voltages
        Vd = V[node_d_indices]
        Vg = V[node_g_indices]
        Vs = V[node_s_indices]
        Vb = V[node_b_indices]

        # Compute device voltages
        # For NMOS: Vgs = Vg - Vs, Vds = Vd - Vs
        # For PMOS: swap D/S and invert
        Vgs_nmos = Vg - Vs
        Vds_nmos = Vd - Vs

        Vgs_pmos = Vs - Vg  # PMOS gate-source
        Vds_pmos = Vs - Vd  # PMOS drain-source (source is higher)

        # Select based on device type
        Vgs = jnp.where(is_pmos, Vgs_pmos, Vgs_nmos)
        Vds = jnp.where(is_pmos, Vds_pmos, Vds_nmos)

        # Compute drain current
        Ids = mosfet_ids_batch(Vgs, Vds, W_values, L_values)

        # Stamp into residual
        # NMOS: current flows D -> S (Id out of D, into S)
        # PMOS: current flows S -> D (Id into D, out of S) after sign flip
        residual = jnp.zeros(num_nodes, dtype=V.dtype)

        # Convention: positive residual = current INTO node
        # NMOS: current flows D→S when ON, so -Ids leaves D, +Ids enters S
        # PMOS: current flows S→D when ON, so +Ids enters D, -Ids leaves S
        Id_out = jnp.where(is_pmos, Ids, -Ids)
        Is_out = jnp.where(is_pmos, -Ids, Ids)

        residual = residual.at[node_d_indices].add(Id_out)
        residual = residual.at[node_s_indices].add(Is_out)

        return residual

    return residual_fn


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


def build_circuit_residual_fn(
    system: MNASystem,
    gmin: float = 1e-9,
    vdd_scale: float = 1.0,
    vdd: float = 1.2,
) -> GPUResidualFunction:
    """Build a GPU-compatible residual function from an MNA system.

    This function analyzes the circuit topology and creates a pure JAX function
    that computes the residual f(V) = KCL current imbalance at each node.

    Args:
        system: MNA system with devices
        gmin: Minimum conductance from each node to ground
        vdd_scale: Scale factor for voltage sources (for source stepping)

    Returns:
        GPUResidualFunction with the residual function and sparsity pattern
    """
    num_nodes = system.num_nodes
    n_reduced = num_nodes - 1  # Exclude ground

    # Collect device data by type
    vsource_data = {'node_p': [], 'node_n': [], 'v_target': []}
    resistor_data = {'node_p': [], 'node_n': [], 'conductance': []}
    mosfet_data = {'node_d': [], 'node_g': [], 'node_s': [], 'node_b': [],
                   'W': [], 'L': [], 'is_pmos': []}

    # Also collect sparsity pattern
    sparsity_rows = []
    sparsity_cols = []

    for device in system.devices:
        model_lower = device.model_name.lower()

        # Detect voltage sources: 'vsource', 'vdc', 'v', or model name starting with 'v'
        is_vsource = (
            'vsource' in model_lower or
            'vdc' in model_lower or
            model_lower == 'v' or
            (model_lower.startswith('v') and len(model_lower) <= 2)
        )

        # Detect MOSFETs: 'nmos', 'pmos', 'psp103n', 'psp103p', etc.
        is_nmos = 'nmos' in model_lower or model_lower.endswith('n') and 'psp' in model_lower
        is_pmos = 'pmos' in model_lower or model_lower.endswith('p') and 'psp' in model_lower
        is_mosfet = is_nmos or is_pmos

        if is_vsource:
            # Voltage source
            node_p = device.node_indices[0]
            node_n = device.node_indices[1]
            v_raw = device.params.get('v', device.params.get('dc', 0.0))
            v_target = eval_param_simple(v_raw, vdd=vdd) * vdd_scale

            vsource_data['node_p'].append(node_p)
            vsource_data['node_n'].append(node_n)
            vsource_data['v_target'].append(v_target)

            # Voltage source sparsity: G[p,p], G[p,n], G[n,p], G[n,n]
            for ni in [node_p, node_n]:
                for nj in [node_p, node_n]:
                    if ni > 0 and nj > 0:  # Skip ground
                        sparsity_rows.append(ni - 1)
                        sparsity_cols.append(nj - 1)

        elif is_mosfet:
            # MOSFET - is_pmos already set above
            node_d = device.node_indices[0]  # D
            node_g = device.node_indices[1]  # G
            node_s = device.node_indices[2]  # S
            node_b = device.node_indices[3] if len(device.node_indices) > 3 else node_s

            W = eval_param_simple(device.params.get('w', 1e-6), vdd=vdd)
            L = eval_param_simple(device.params.get('l', 0.2e-6), vdd=vdd)

            # Ensure positive values
            W = max(W, 1e-9)
            L = max(L, 1e-9)

            mosfet_data['node_d'].append(node_d)
            mosfet_data['node_g'].append(node_g)
            mosfet_data['node_s'].append(node_s)
            mosfet_data['node_b'].append(node_b)
            mosfet_data['W'].append(float(W))
            mosfet_data['L'].append(float(L))
            mosfet_data['is_pmos'].append(is_pmos)

            # MOSFET sparsity: affects D, G, S, B nodes
            # Jacobian entries: dId/dVd, dId/dVg, dId/dVs, dId/dVb
            #                   dIs/dVd, dIs/dVg, dIs/dVs, dIs/dVb
            terminals = [node_d, node_g, node_s, node_b]
            current_nodes = [node_d, node_s]  # Only D and S have current
            for ni in current_nodes:
                for nj in terminals:
                    if ni > 0 and nj > 0:  # Skip ground
                        sparsity_rows.append(ni - 1)
                        sparsity_cols.append(nj - 1)

        else:
            # Check for resistor
            is_resistor = (
                model_lower == 'r' or
                model_lower == 'resistor' or
                model_lower.startswith('r') and len(model_lower) <= 2
            )
            if is_resistor:
                node_p = device.node_indices[0]
                node_n = device.node_indices[1]
                r_value = eval_param_simple(device.params.get('r', device.params.get('value', 1e6)), vdd=vdd)
                r_value = max(r_value, 1e-9)  # Minimum resistance

                resistor_data['node_p'].append(node_p)
                resistor_data['node_n'].append(node_n)
                resistor_data['conductance'].append(1.0 / r_value)

                # Resistor sparsity: G[p,p], G[p,n], G[n,p], G[n,n]
                for ni in [node_p, node_n]:
                    for nj in [node_p, node_n]:
                        if ni > 0 and nj > 0:  # Skip ground
                            sparsity_rows.append(ni - 1)
                            sparsity_cols.append(nj - 1)

    # Add GMIN diagonal entries to sparsity
    for i in range(n_reduced):
        sparsity_rows.append(i)
        sparsity_cols.append(i)

    # Convert to JAX arrays
    vsource_arrays = {
        'node_p': jnp.array(vsource_data['node_p'], dtype=jnp.int32),
        'node_n': jnp.array(vsource_data['node_n'], dtype=jnp.int32),
        'v_target': jnp.array(vsource_data['v_target'], dtype=jnp.float64),
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

    resistor_arrays = {
        'node_p': jnp.array(resistor_data['node_p'], dtype=jnp.int32),
        'node_n': jnp.array(resistor_data['node_n'], dtype=jnp.int32),
        'conductance': jnp.array(resistor_data['conductance'], dtype=jnp.float64),
    }

    # Build the combined residual function
    def residual_fn(V_reduced: Array) -> Array:
        """Compute residual f(V) for the circuit.

        Args:
            V_reduced: Node voltages excluding ground (num_nodes - 1,)

        Returns:
            Residual vector (num_nodes - 1,)
        """
        # Prepend ground voltage (0)
        V_full = jnp.concatenate([jnp.array([0.0]), V_reduced])

        residual = jnp.zeros(num_nodes, dtype=V_full.dtype)

        # Voltage sources
        if len(vsource_data['node_p']) > 0:
            Vp = V_full[vsource_arrays['node_p']]
            Vn = V_full[vsource_arrays['node_n']]
            V_actual = Vp - Vn
            G_big = 1e6
            I = G_big * (V_actual - vsource_arrays['v_target'])
            residual = residual.at[vsource_arrays['node_p']].add(I)
            residual = residual.at[vsource_arrays['node_n']].add(-I)

        # MOSFETs
        if len(mosfet_data['node_d']) > 0:
            Vd = V_full[mosfet_arrays['node_d']]
            Vg = V_full[mosfet_arrays['node_g']]
            Vs = V_full[mosfet_arrays['node_s']]
            Vb = V_full[mosfet_arrays['node_b']]
            W = mosfet_arrays['W']
            L = mosfet_arrays['L']
            is_pmos = mosfet_arrays['is_pmos']

            # Level-1 MOSFET model parameters
            vth0 = 0.4
            kp = 200e-6
            lambda_ = 0.01

            # Device voltages (handle NMOS/PMOS)
            Vgs_nmos = Vg - Vs
            Vds_nmos = Vd - Vs
            Vgs_pmos = Vs - Vg
            Vds_pmos = Vs - Vd

            Vgs = jnp.where(is_pmos, Vgs_pmos, Vgs_nmos)
            Vds = jnp.where(is_pmos, Vds_pmos, Vds_nmos)

            # Level-1 drain current
            beta = kp * W / L
            Vgst = Vgs - vth0

            # Smooth subthreshold with soft-plus
            alpha = 10.0  # Smoothing parameter
            Vgst_eff = jnp.where(
                Vgst > 0,
                Vgst,
                (1/alpha) * jnp.log1p(jnp.exp(alpha * Vgst))
            )

            # Saturation
            Vdsat = jnp.maximum(Vgst_eff, 1e-6)
            Vds_eff = Vdsat * jnp.tanh(jnp.abs(Vds) / jnp.maximum(Vdsat, 0.01))

            # Drain current
            Ids = beta * Vgst_eff * Vds_eff * (1 + lambda_ * jnp.abs(Vds))
            Ids = jnp.maximum(Ids, 0.0)

            # Add minimum off-state conductance for numerical stability
            # This provides a DC path for floating nodes (like series NMOS stacks)
            # VACASK uses gds_min = 1e-9, we match that for consistency
            gds_min = 1e-9
            Ids_leakage = gds_min * Vds
            Ids = Ids + Ids_leakage

            # Stamp currents
            # Convention: positive residual = current INTO node
            # NMOS: current flows D→S when ON, so -Ids leaves D, +Ids enters S
            # PMOS: current flows S→D when ON, so +Ids enters D, -Ids leaves S
            Id_out = jnp.where(is_pmos, Ids, -Ids)
            Is_out = jnp.where(is_pmos, -Ids, Ids)

            residual = residual.at[mosfet_arrays['node_d']].add(Id_out)
            residual = residual.at[mosfet_arrays['node_s']].add(Is_out)

        # Resistors
        if len(resistor_data['node_p']) > 0:
            Vp = V_full[resistor_arrays['node_p']]
            Vn = V_full[resistor_arrays['node_n']]
            I = resistor_arrays['conductance'] * (Vp - Vn)
            residual = residual.at[resistor_arrays['node_p']].add(I)
            residual = residual.at[resistor_arrays['node_n']].add(-I)

        # GMIN: small conductance from each node to ground
        residual = residual.at[1:].add(gmin * V_reduced)

        # Return reduced residual (exclude ground)
        return residual[1:]

    # Build sparsity pattern (deduplicated)
    sparsity_set = set(zip(sparsity_rows, sparsity_cols))
    sparsity_rows = jnp.array([r for r, c in sorted(sparsity_set)], dtype=jnp.int32)
    sparsity_cols = jnp.array([c for r, c in sorted(sparsity_set)], dtype=jnp.int32)

    return GPUResidualFunction(
        num_nodes=num_nodes,
        residual_fn=residual_fn,
        sparsity_indices=(sparsity_rows, sparsity_cols),
    )


def dc_operating_point_gpu(
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
    """GPU-native DC operating point using sparsejac.

    This solver keeps all computation on GPU using JAX operations.
    The Jacobian is computed via automatic differentiation with sparsejac,
    which uses graph coloring to efficiently compute only non-zero entries.

    Args:
        system: MNA system with devices
        initial_guess: Initial voltage estimate (shape: [num_nodes])
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance for convergence
        reltol: Relative tolerance for convergence
        damping: Damping factor (0 < damping <= 1)
        vdd: Supply voltage for initialization and clamping
        gmin: Minimum conductance from each node to ground
        verbose: Print iteration details

    Returns:
        Tuple of (solution, info) where:
            solution: Node voltages (shape: [num_nodes])
            info: Dict with convergence information
    """
    if not HAS_SPARSEJAC:
        raise ImportError("sparsejac is required for GPU-native DC analysis. "
                         "Install with: pip install sparsejac")

    n = system.num_nodes
    n_reduced = n - 1

    # Build the GPU residual function
    gpu_fn = build_circuit_residual_fn(system, gmin=gmin, vdd=vdd)
    residual_fn = gpu_fn.residual_fn

    # Build sparsity pattern for sparsejac
    sparsity_rows, sparsity_cols = gpu_fn.sparsity_indices

    # Create BCOO sparsity matrix for sparsejac
    n_nnz = len(sparsity_rows)
    sparsity_data = jnp.ones(n_nnz, dtype=jnp.float64)
    sparsity_indices = jnp.stack([sparsity_rows, sparsity_cols], axis=-1)
    sparsity = jsparse.BCOO((sparsity_data, sparsity_indices), shape=(n_reduced, n_reduced))

    # Create sparse Jacobian function using sparsejac
    jacobian_fn = sparsejac.jacrev(residual_fn, sparsity=sparsity)

    # Initialize solution
    if initial_guess is not None:
        V = jnp.array(initial_guess[1:], dtype=jnp.float64)  # Skip ground
    else:
        V = jnp.zeros(n_reduced, dtype=jnp.float64)
        # Initialize vdd nodes
        for name, idx in system.node_names.items():
            if 'vdd' in name.lower() and idx > 0:
                V = V.at[idx - 1].set(vdd)

    # Run Newton-Raphson iterations
    converged = False
    iterations = 0
    residual_norm = 1e20
    delta_norm = 0.0
    residual_history = []

    for iteration in range(max_iterations):
        # Compute residual
        f = residual_fn(V)
        residual_norm = float(jnp.max(jnp.abs(f)))
        residual_history.append(residual_norm)

        if verbose and iteration < 20:
            print(f"  Iter {iteration}: residual={residual_norm:.2e}")

        if residual_norm < abstol:
            converged = True
            iterations = iteration + 1
            break

        # Compute Jacobian using sparsejac (returns BCOO)
        J = jacobian_fn(V)

        # Use sparse solver on GPU, dense solver on CPU
        backend = jax.default_backend()
        if backend in ('gpu', 'cuda'):
            # Convert BCOO to CSR arrays for sparse solver
            # BCOO has: J.data (values), J.indices (n_nnz, 2) with [row, col]
            # CSR needs: data, col_indices, row_indptr
            from jax.experimental.sparse.linalg import spsolve as jax_spsolve
            J_data, J_csr_indices, J_csr_indptr = _bcoo_to_csr(J, n_reduced)
            delta_V = jax_spsolve(J_data, J_csr_indices, J_csr_indptr, -f, tol=0)
        else:
            # CPU: use dense solver (spsolve falls back to scipy anyway)
            J_dense = J.todense()

            try:
                delta_V = jnp.linalg.solve(J_dense, -f)
            except Exception:
                # Add regularization if singular
                reg = 1e-10 * jnp.eye(n_reduced)
                delta_V = jnp.linalg.solve(J_dense + reg, -f)

        # Apply damping with voltage limiting
        max_step = 2.0
        max_delta = jnp.max(jnp.abs(delta_V))
        step_scale = jnp.minimum(damping, max_step / (max_delta + 1e-15))

        # Update solution
        V = V + step_scale * delta_V

        # Clamp to reasonable range
        v_clamp = vdd * 2.0
        V = jnp.clip(V, -v_clamp, v_clamp)

        iterations = iteration + 1

        # Check delta for convergence
        delta_norm = float(jnp.max(jnp.abs(step_scale * delta_V)))
        v_norm = float(jnp.max(jnp.abs(V)))

        if delta_norm < abstol + reltol * max(v_norm, 1.0):
            converged = True
            break

    # Build full voltage vector with ground
    V_full = jnp.concatenate([jnp.array([0.0]), V])

    info = {
        'converged': converged,
        'iterations': iterations,
        'residual_norm': residual_norm,
        'delta_norm': delta_norm,
        'residual_history': residual_history,
        'method': 'gpu_sparsejac',
    }

    return V_full, info


def dc_operating_point_gpu_source_stepping(
    system: MNASystem,
    initial_guess: Optional[Array] = None,
    vdd_target: float = 1.2,
    vdd_steps: int = 12,
    max_iterations_per_step: int = 50,
    abstol: float = 1e-9,
    reltol: float = 1e-3,
    damping: float = 1.0,
    gmin: float = 1e-6,
    gmin_fallback: float = 1e-3,
    verbose: bool = False,
) -> Tuple[Array, Dict]:
    """GPU-native DC operating point with source stepping for difficult circuits.

    Source stepping is a homotopy method that gradually ramps the supply voltage
    from 0 to the target value. This is particularly effective for large digital
    circuits with many cascaded stages.

    This version uses sparsejac for automatic Jacobian computation and keeps
    all Newton-Raphson iterations on GPU.

    Args:
        system: MNA system with devices
        initial_guess: Initial voltage estimate (shape: [num_nodes])
        vdd_target: Target supply voltage (default 1.2V)
        vdd_steps: Number of voltage steps (default 12)
        max_iterations_per_step: Max NR iterations per source step
        abstol: Absolute tolerance for convergence
        reltol: Relative tolerance for convergence
        damping: Damping factor (0 < damping <= 1)
        gmin: GMIN value for matrix conditioning
        gmin_fallback: Higher GMIN for fallback on difficult steps
        verbose: Print progress information

    Returns:
        Tuple of (solution, info) where:
            solution: Node voltages (shape: [num_nodes])
            info: Dict with convergence information including source_steps
    """
    if not HAS_SPARSEJAC:
        raise ImportError("sparsejac is required for GPU-native DC analysis. "
                         "Install with: pip install sparsejac")

    n = system.num_nodes
    n_reduced = n - 1

    # Initialize solution to zeros
    if initial_guess is not None:
        V = np.array(initial_guess, dtype=np.float64)
    else:
        V = np.zeros(n, dtype=np.float64)

    # Find vdd node indices
    vdd_node_indices = []
    for name, idx in system.node_names.items():
        name_lower = name.lower()
        if 'vdd' in name_lower and name_lower not in ('vss', 'gnd', '0'):
            vdd_node_indices.append(idx)

    total_iterations = 0
    source_steps = 0
    all_residual_history = []

    # Generate voltage steps
    vdd_values = np.linspace(vdd_target / vdd_steps, vdd_target, vdd_steps)

    if verbose:
        print(f"GPU Source stepping: 0 -> {vdd_target:.2f}V in {vdd_steps} steps", flush=True)

    converged_at_target = False
    last_info = {'converged': False, 'iterations': 0, 'residual_norm': 1e20}

    for step_idx, vdd_step in enumerate(vdd_values):
        source_steps += 1
        is_final_step = (step_idx == len(vdd_values) - 1)
        vdd_scale = vdd_step / vdd_target

        # Set vdd nodes to current step voltage
        for idx in vdd_node_indices:
            V[idx] = vdd_step

        if verbose:
            print(f"  Step {source_steps}: Vdd={vdd_step:.3f}V", flush=True)

        # Use relaxed tolerance for intermediate steps
        step_abstol = abstol if is_final_step else max(abstol, 1e-4)

        # Build residual function for this vdd_scale
        gpu_fn = build_circuit_residual_fn(
            system, gmin=gmin, vdd_scale=vdd_scale, vdd=vdd_target
        )

        # Run GPU Newton-Raphson at this step
        V_jax, info = _gpu_newton_raphson(
            gpu_fn,
            initial_guess=V,
            max_iterations=max_iterations_per_step,
            abstol=step_abstol,
            reltol=reltol,
            damping=damping,
            vdd=vdd_step,
        )

        V = np.array(V_jax)
        total_iterations += info['iterations']
        all_residual_history.extend(info['residual_history'])
        last_info = info

        if verbose:
            print(f"    -> iter={info['iterations']}, residual={info['residual_norm']:.2e}, "
                  f"converged={info['converged']}", flush=True)

        # Fallback with higher GMIN for difficult steps
        if not info['converged']:
            if not is_final_step and info['residual_norm'] < 1e-3:
                if verbose:
                    print(f"    Accepting partial convergence", flush=True)
            else:
                if verbose:
                    print(f"    Trying with higher GMIN ({gmin_fallback:.0e})...", flush=True)

                gpu_fn_h = build_circuit_residual_fn(
                    system, gmin=gmin_fallback, vdd_scale=vdd_scale, vdd=vdd_target
                )

                V_jax_h, info_h = _gpu_newton_raphson(
                    gpu_fn_h,
                    initial_guess=V,
                    max_iterations=max_iterations_per_step * 2,
                    abstol=1e-3 if not is_final_step else abstol,
                    reltol=reltol,
                    damping=damping,
                    vdd=vdd_step,
                )

                if info_h['residual_norm'] < info['residual_norm']:
                    V = np.array(V_jax_h)
                    total_iterations += info_h['iterations']
                    all_residual_history.extend(info_h['residual_history'])
                    last_info = info_h

                if verbose:
                    print(f"      -> iter={info_h['iterations']}, "
                          f"residual={info_h['residual_norm']:.2e}", flush=True)

        if is_final_step and last_info['converged']:
            converged_at_target = True

    result_info = {
        'converged': converged_at_target,
        'iterations': total_iterations,
        'source_steps': source_steps,
        'final_vdd': vdd_values[-1],
        'residual_norm': last_info['residual_norm'],
        'delta_norm': last_info.get('delta_norm', 0.0),
        'residual_history': all_residual_history,
        'method': 'gpu_sparsejac_source_stepping',
    }

    if verbose:
        print(f"  Complete: steps={source_steps}, iter={total_iterations}, "
              f"converged={converged_at_target}", flush=True)

    return jnp.array(V), result_info


def _gpu_newton_raphson(
    gpu_fn: GPUResidualFunction,
    initial_guess: Array,
    max_iterations: int = 100,
    abstol: float = 1e-9,
    reltol: float = 1e-3,
    damping: float = 1.0,
    vdd: float = 1.2,
) -> Tuple[Array, Dict]:
    """Internal GPU Newton-Raphson solver using sparsejac.

    Args:
        gpu_fn: GPU residual function with sparsity pattern
        initial_guess: Initial voltage estimate (full array including ground)
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance
        reltol: Relative tolerance
        damping: Damping factor
        vdd: Current supply voltage for clamping

    Returns:
        Tuple of (solution, info)
    """
    n_reduced = gpu_fn.num_nodes - 1
    residual_fn = gpu_fn.residual_fn

    # Build sparsity pattern for sparsejac
    sparsity_rows, sparsity_cols = gpu_fn.sparsity_indices
    n_nnz = len(sparsity_rows)
    sparsity_data = jnp.ones(n_nnz, dtype=jnp.float64)
    sparsity_indices = jnp.stack([sparsity_rows, sparsity_cols], axis=-1)
    sparsity = jsparse.BCOO((sparsity_data, sparsity_indices), shape=(n_reduced, n_reduced))

    # Create sparse Jacobian function
    jacobian_fn = sparsejac.jacrev(residual_fn, sparsity=sparsity)

    # Initialize (skip ground)
    V = jnp.array(initial_guess[1:], dtype=jnp.float64)

    converged = False
    iterations = 0
    residual_norm = 1e20
    delta_norm = 0.0
    residual_history = []

    for iteration in range(max_iterations):
        # Compute residual
        f = residual_fn(V)
        residual_norm = float(jnp.max(jnp.abs(f)))
        residual_history.append(residual_norm)

        if residual_norm < abstol:
            converged = True
            iterations = iteration + 1
            break

        # Compute Jacobian
        J = jacobian_fn(V)

        # Solve (dense on CPU, sparse on GPU)
        backend = jax.default_backend()
        if backend in ('gpu', 'cuda'):
            from jax.experimental.sparse.linalg import spsolve as jax_spsolve
            J_data, J_csr_indices, J_csr_indptr = _bcoo_to_csr(J, n_reduced)
            delta_V = jax_spsolve(J_data, J_csr_indices, J_csr_indptr, -f, tol=0)
        else:
            J_dense = J.todense()
            try:
                delta_V = jnp.linalg.solve(J_dense, -f)
            except Exception:
                reg = 1e-10 * jnp.eye(n_reduced)
                delta_V = jnp.linalg.solve(J_dense + reg, -f)

        # Apply damping with voltage limiting
        max_step = 2.0
        max_delta = jnp.max(jnp.abs(delta_V))
        step_scale = jnp.minimum(damping, max_step / (max_delta + 1e-15))

        V = V + step_scale * delta_V

        # Clamp
        v_clamp = max(vdd * 2.0, 0.5)
        V = jnp.clip(V, -v_clamp, v_clamp)

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
    }

    return V_full, info


def dc_operating_point_gpu_jit(
    system: MNASystem,
    initial_guess: Optional[Array] = None,
    max_iterations: int = 100,
    abstol: float = 1e-9,
    damping: float = 1.0,
    vdd: float = 1.2,
    gmin: float = 1e-9,
) -> Tuple[Array, Dict]:
    """Fully JIT-compiled GPU DC solver using jax.lax.while_loop.

    This version compiles the entire Newton-Raphson loop with JAX,
    keeping all computation on GPU without Python interpreter overhead.

    Note: This requires the solver to be trace-able by JAX, which means
    no Python conditionals or dynamic shapes during iteration.

    Args:
        system: MNA system with devices
        initial_guess: Initial voltage estimate
        max_iterations: Maximum iterations (static - must be known at compile time)
        abstol: Absolute tolerance
        damping: Damping factor
        vdd: Supply voltage
        gmin: GMIN conductance

    Returns:
        Tuple of (solution, info)
    """
    if not HAS_SPARSEJAC:
        raise ImportError("sparsejac is required")

    n = system.num_nodes
    n_reduced = n - 1

    # Build residual function and Jacobian
    gpu_fn = build_circuit_residual_fn(system, gmin=gmin, vdd=vdd)
    residual_fn = gpu_fn.residual_fn

    sparsity_rows, sparsity_cols = gpu_fn.sparsity_indices
    n_nnz = len(sparsity_rows)
    sparsity_data = jnp.ones(n_nnz, dtype=jnp.float64)
    sparsity_indices = jnp.stack([sparsity_rows, sparsity_cols], axis=-1)
    sparsity = jsparse.BCOO((sparsity_data, sparsity_indices), shape=(n_reduced, n_reduced))

    jacobian_fn = sparsejac.jacrev(residual_fn, sparsity=sparsity)

    # Initialize
    if initial_guess is not None:
        V_init = jnp.array(initial_guess[1:], dtype=jnp.float64)
    else:
        V_init = jnp.zeros(n_reduced, dtype=jnp.float64)
        for name, idx in system.node_names.items():
            if 'vdd' in name.lower() and idx > 0:
                V_init = V_init.at[idx - 1].set(vdd)

    v_clamp = vdd * 2.0

    # Define single Newton step
    def newton_step(carry):
        V, iteration = carry

        # Compute residual and Jacobian
        f = residual_fn(V)
        J = jacobian_fn(V)
        J_dense = J.todense()

        # Solve for update
        delta_V = jnp.linalg.solve(J_dense + 1e-12 * jnp.eye(n_reduced), -f)

        # Apply damping
        max_delta = jnp.max(jnp.abs(delta_V))
        step_scale = jnp.minimum(damping, 2.0 / (max_delta + 1e-15))

        # Update and clamp
        V_new = V + step_scale * delta_V
        V_new = jnp.clip(V_new, -v_clamp, v_clamp)

        return (V_new, iteration + 1)

    # Condition function
    def cond_fn(carry):
        V, iteration = carry
        f = residual_fn(V)
        residual_norm = jnp.max(jnp.abs(f))
        return jnp.logical_and(residual_norm >= abstol, iteration < max_iterations)

    # Run the loop
    V_final, iterations = jax.lax.while_loop(cond_fn, newton_step, (V_init, 0))

    # Compute final residual
    f_final = residual_fn(V_final)
    residual_norm = float(jnp.max(jnp.abs(f_final)))
    converged = residual_norm < abstol

    V_full = jnp.concatenate([jnp.array([0.0]), V_final])

    info = {
        'converged': converged,
        'iterations': int(iterations),
        'residual_norm': residual_norm,
        'method': 'gpu_sparsejac_jit',
    }

    return V_full, info


def dc_operating_point_gpu_vectorized(
    system: MNASystem,
    initial_guess: Optional[Array] = None,
    max_iterations: int = 100,
    abstol: float = 1e-9,
    reltol: float = 1e-3,
    vdd: float = 1.2,
    gmin: float = 1e-9,
    verbose: bool = False,
) -> Tuple[Array, Dict]:
    """GPU DC solver using vectorized device evaluation from MNASystem.

    This solver uses build_gpu_residual_fn() which evaluates all devices
    of the same type in parallel using mosfet_batch() and other vectorized
    device functions. The Jacobian is computed via JAX autodiff.

    This uses the more sophisticated BSIM-like MOSFET model from mosfet_simple.py
    rather than the simpler Level-1 model.

    Args:
        system: MNA system with devices
        initial_guess: Initial voltage estimate (shape: [num_nodes])
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance for convergence
        reltol: Relative tolerance for convergence
        vdd: Supply voltage
        gmin: Minimum conductance from each node to ground
        verbose: Print iteration details

    Returns:
        Tuple of (solution, info) where:
            solution: Node voltages (shape: [num_nodes])
            info: Dict with convergence information
    """
    if sparsejac is None:
        raise ImportError("sparsejac is required for GPU-native DC analysis. "
                         "Install with: pip install sparsejac")

    n = system.num_nodes
    n_reduced = n - 1

    # Build device groups for vectorized evaluation
    system.build_device_groups(vdd=vdd)

    # Build the GPU residual function using vectorized device evaluation
    gpu_residual_fn = system.build_gpu_residual_fn(vdd=vdd, gmin=gmin)

    # Wrapper that takes reduced voltage vector (excluding ground)
    def residual_fn(V_reduced: Array) -> Array:
        V_full = jnp.concatenate([jnp.array([0.0]), V_reduced])
        return gpu_residual_fn(V_full)

    # Build sparsity pattern for efficient sparse Jacobian
    sparsity_rows, sparsity_cols = system.build_sparsity_pattern()

    # Create BCOO sparsity matrix for sparsejac
    n_nnz = len(sparsity_rows)
    sparsity_data = jnp.ones(n_nnz, dtype=jnp.float64)
    sparsity_indices = jnp.stack([sparsity_rows, sparsity_cols], axis=-1)
    sparsity = jsparse.BCOO((sparsity_data, sparsity_indices), shape=(n_reduced, n_reduced))

    # Create sparse Jacobian function using sparsejac
    jacobian_fn = sparsejac.jacrev(residual_fn, sparsity=sparsity)

    # Initialize solution
    if initial_guess is not None:
        V = jnp.array(initial_guess[1:], dtype=jnp.float64)
    else:
        # Initialize all internal nodes to vdd/2 for better starting point
        # This helps avoid the extreme Jacobian entries when nodes are at 0
        V = jnp.full(n_reduced, vdd / 2, dtype=jnp.float64)
        # Initialize vdd nodes to full vdd
        for name, idx in system.node_names.items():
            if 'vdd' in name.lower() and idx > 0:
                V = V.at[idx - 1].set(vdd)
            # Initialize vss and ground-referenced nodes to 0
            elif 'vss' in name.lower() and idx > 0:
                V = V.at[idx - 1].set(0.0)
            # Initialize input nodes to 0 (assuming low inputs)
            elif 'in' in name.lower() and idx > 0:
                V = V.at[idx - 1].set(0.0)

    # Direct Newton-Raphson without source stepping
    # Source stepping doesn't work well with BSIM-like models at low VDD
    converged = False
    iterations = 0
    residual_norm = 1e20
    delta_norm = 0.0
    residual_history = []

    for iteration in range(max_iterations):
        f = residual_fn(V)
        residual_norm = float(jnp.max(jnp.abs(f)))
        residual_history.append(residual_norm)

        if verbose and iterations < 5:
            print(f"  Iter {iterations}: residual={residual_norm:.2e}, V[out]={float(V[2]) if len(V) > 2 else 0:.4f}")

        # Check convergence
        # Account for GMIN contribution: GMIN * Vnode can add to residual
        gmin_floor = gmin * float(jnp.max(jnp.abs(V)))
        effective_tol = max(abstol, gmin_floor * 2.0)
        if residual_norm < effective_tol:
            converged = True
            break

        # Compute sparse Jacobian using sparsejac (returns BCOO)
        J = jacobian_fn(V)

        # Solve linear system
        # Use sparse solver on GPU, dense solver on CPU
        backend = jax.default_backend()
        if backend in ('gpu', 'cuda'):
            # Convert BCOO to CSR arrays for sparse solver
            from jax.experimental.sparse.linalg import spsolve as jax_spsolve
            J_data, J_csr_indices, J_csr_indptr = _bcoo_to_csr(J, n_reduced)
            delta_V = jax_spsolve(J_data, J_csr_indices, J_csr_indptr, -f, tol=0)
        else:
            # CPU: use dense solver
            J_dense = J.todense()
            try:
                delta_V = jnp.linalg.solve(J_dense, -f)
            except Exception:
                # Add regularization if singular
                J_reg = J_dense + 1e-12 * jnp.eye(n_reduced)
                delta_V = jnp.linalg.solve(J_reg, -f)

        # Voltage limiting
        max_step = 0.5 * vdd
        max_delta = float(jnp.max(jnp.abs(delta_V)))
        if verbose and iterations < 5:
            print(f"    raw delta_V[out] = {float(delta_V[2]) if len(delta_V) > 2 else 0:.4f}, max_delta = {max_delta:.4f}")
        if max_delta > max_step:
            delta_V = delta_V * (max_step / max_delta)
            if verbose and iterations < 5:
                print(f"    limited delta_V[out] = {float(delta_V[2]) if len(delta_V) > 2 else 0:.4f}")

        # Line search backtracking to prevent residual from increasing
        alpha = 1.0
        V_new = V + alpha * delta_V
        V_new = jnp.clip(V_new, -vdd * 2.0, vdd * 2.0)
        f_new = residual_fn(V_new)
        new_residual = float(jnp.max(jnp.abs(f_new)))

        # Backtrack only if residual increases dramatically (more than 5x)
        # and we're not already close to convergence
        backtrack_count = 0
        while (new_residual > 5.0 * residual_norm and
               residual_norm > 1e-6 and
               backtrack_count < 3 and
               alpha > 0.1):
            alpha *= 0.5
            V_new = V + alpha * delta_V
            V_new = jnp.clip(V_new, -vdd * 2.0, vdd * 2.0)
            f_new = residual_fn(V_new)
            new_residual = float(jnp.max(jnp.abs(f_new)))
            backtrack_count += 1

        if verbose and iterations < 5 and backtrack_count > 0:
            print(f"    backtracked {backtrack_count}x, alpha={alpha:.3f}")

        # Update solution
        V = V_new
        delta_norm = float(jnp.max(jnp.abs(alpha * delta_V)))

        # Clamp to reasonable range
        v_clamp = vdd * 2.0
        V = jnp.clip(V, -v_clamp, v_clamp)

        iterations += 1

    # Build full voltage vector with ground
    V_full = jnp.concatenate([jnp.array([0.0]), V])

    info = {
        'converged': converged,
        'iterations': iterations,
        'residual_norm': residual_norm,
        'delta_norm': delta_norm,
        'residual_history': residual_history,
        'method': 'gpu_vectorized',
    }

    return V_full, info


def dc_operating_point_gpu_vectorized_source_stepping(
    system: MNASystem,
    initial_guess: Optional[Array] = None,
    vdd_target: float = 1.2,
    vdd_steps: int = 6,
    max_iterations_per_step: int = 50,
    abstol: float = 1e-9,
    reltol: float = 1e-3,
    gmin: float = 1e-9,
    gmin_fallback: float = 1e-6,
    verbose: bool = False,
) -> Tuple[Array, Dict]:
    """GPU DC solver with BSIM model and source stepping.

    This combines the vectorized BSIM-like MOSFET model from mosfet_simple.py
    with source stepping for robust convergence on multi-stage circuits.

    Args:
        system: MNA system with devices
        initial_guess: Initial voltage estimate (shape: [num_nodes])
        vdd_target: Target supply voltage (default 1.2V)
        vdd_steps: Number of voltage steps (default 6)
        max_iterations_per_step: Max NR iterations per source step
        abstol: Absolute tolerance for convergence
        reltol: Relative tolerance for convergence
        gmin: GMIN value for matrix conditioning
        gmin_fallback: Higher GMIN for fallback on difficult steps
        verbose: Print progress information

    Returns:
        Tuple of (solution, info)
    """
    if sparsejac is None:
        raise ImportError("sparsejac is required for GPU-native DC analysis. "
                         "Install with: pip install sparsejac")

    n = system.num_nodes
    n_reduced = n - 1

    # Find vdd node indices
    vdd_node_indices = []
    for name, idx in system.node_names.items():
        name_lower = name.lower()
        if 'vdd' in name_lower and name_lower not in ('vss', 'gnd', '0'):
            vdd_node_indices.append(idx)

    # Initialize solution
    if initial_guess is not None:
        V = np.array(initial_guess, dtype=np.float64)
    else:
        V = np.zeros(n, dtype=np.float64)

    total_iterations = 0
    source_steps = 0
    all_residual_history = []

    # Generate voltage steps
    vdd_values = np.linspace(vdd_target / vdd_steps, vdd_target, vdd_steps)

    if verbose:
        print(f"BSIM Source stepping: 0 -> {vdd_target:.2f}V in {vdd_steps} steps", flush=True)

    converged_at_target = False
    last_info = {'converged': False, 'iterations': 0, 'residual_norm': 1e20}

    for step_idx, vdd_step in enumerate(vdd_values):
        source_steps += 1
        is_final_step = (step_idx == len(vdd_values) - 1)

        # Set vdd nodes to current step voltage
        for idx in vdd_node_indices:
            V[idx] = vdd_step

        if verbose:
            print(f"  Step {source_steps}: Vdd={vdd_step:.3f}V", flush=True)

        # Use relaxed tolerance for intermediate steps
        step_abstol = abstol if is_final_step else max(abstol, 1e-4)
        current_gmin = gmin

        # Build device groups for this VDD
        system.build_device_groups(vdd=vdd_step)

        # Run Newton-Raphson at this step
        V_jax, info = _newton_raphson_bsim(
            system,
            initial_guess=V,
            max_iterations=max_iterations_per_step,
            abstol=step_abstol,
            reltol=reltol,
            vdd=vdd_step,
            gmin=current_gmin,
        )

        V = np.array(V_jax)
        total_iterations += info['iterations']
        all_residual_history.extend(info['residual_history'])
        last_info = info

        if verbose:
            print(f"    -> iter={info['iterations']}, residual={info['residual_norm']:.2e}, "
                  f"converged={info['converged']}", flush=True)

        # Fallback with higher GMIN for difficult steps
        if not info['converged']:
            if not is_final_step and info['residual_norm'] < 1e-3:
                if verbose:
                    print(f"    Accepting partial convergence", flush=True)
            else:
                if verbose:
                    print(f"    Trying with higher GMIN ({gmin_fallback:.0e})...", flush=True)

                V_jax_h, info_h = _newton_raphson_bsim(
                    system,
                    initial_guess=V,
                    max_iterations=max_iterations_per_step * 2,
                    abstol=1e-3 if not is_final_step else abstol,
                    reltol=reltol,
                    vdd=vdd_step,
                    gmin=gmin_fallback,
                )

                if info_h['residual_norm'] < info['residual_norm']:
                    V = np.array(V_jax_h)
                    total_iterations += info_h['iterations']
                    all_residual_history.extend(info_h['residual_history'])
                    last_info = info_h

                if verbose:
                    print(f"      -> iter={info_h['iterations']}, "
                          f"residual={info_h['residual_norm']:.2e}", flush=True)

        if is_final_step and last_info['converged']:
            converged_at_target = True

    result_info = {
        'converged': converged_at_target,
        'iterations': total_iterations,
        'source_steps': source_steps,
        'final_vdd': vdd_values[-1],
        'residual_norm': last_info['residual_norm'],
        'delta_norm': last_info.get('delta_norm', 0.0),
        'residual_history': all_residual_history,
        'method': 'gpu_vectorized_bsim_source_stepping',
    }

    if verbose:
        print(f"  Complete: steps={source_steps}, iter={total_iterations}, "
              f"converged={converged_at_target}", flush=True)

    return jnp.array(V), result_info


def _newton_raphson_bsim(
    system: MNASystem,
    initial_guess: Array,
    max_iterations: int = 50,
    abstol: float = 1e-9,
    reltol: float = 1e-3,
    vdd: float = 1.2,
    gmin: float = 1e-9,
) -> Tuple[Array, Dict]:
    """Internal Newton-Raphson solver using BSIM model with sparsejac.

    Args:
        system: MNA system with device groups built
        initial_guess: Initial voltage estimate (full array including ground)
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance
        reltol: Relative tolerance
        vdd: Current supply voltage for clamping
        gmin: GMIN conductance

    Returns:
        Tuple of (solution, info)
    """
    n = system.num_nodes
    n_reduced = n - 1

    # Build the GPU residual function using vectorized device evaluation
    gpu_residual_fn = system.build_gpu_residual_fn(vdd=vdd, gmin=gmin)

    # Wrapper that takes reduced voltage vector (excluding ground)
    def residual_fn(V_reduced: Array) -> Array:
        V_full = jnp.concatenate([jnp.array([0.0]), V_reduced])
        return gpu_residual_fn(V_full)

    # Build sparsity pattern for efficient sparse Jacobian
    sparsity_rows, sparsity_cols = system.build_sparsity_pattern()

    # Create BCOO sparsity matrix for sparsejac
    n_nnz = len(sparsity_rows)
    sparsity_data = jnp.ones(n_nnz, dtype=jnp.float64)
    sparsity_indices = jnp.stack([sparsity_rows, sparsity_cols], axis=-1)
    sparsity = jsparse.BCOO((sparsity_data, sparsity_indices), shape=(n_reduced, n_reduced))

    # Create sparse Jacobian function using sparsejac
    jacobian_fn = sparsejac.jacrev(residual_fn, sparsity=sparsity)

    # Initialize (skip ground)
    V = jnp.array(initial_guess[1:], dtype=jnp.float64)

    converged = False
    iterations = 0
    residual_norm = 1e20
    delta_norm = 0.0
    residual_history = []

    for iteration in range(max_iterations):
        # Compute residual
        f = residual_fn(V)
        residual_norm = float(jnp.max(jnp.abs(f)))
        residual_history.append(residual_norm)

        # Check convergence with GMIN floor
        gmin_floor = gmin * float(jnp.max(jnp.abs(V)))
        effective_tol = max(abstol, gmin_floor * 2.0)
        if residual_norm < effective_tol:
            converged = True
            iterations = iteration + 1
            break

        # Compute Jacobian using sparsejac (returns BCOO)
        J = jacobian_fn(V)

        # Solve (dense on CPU, sparse on GPU)
        backend = jax.default_backend()
        if backend in ('gpu', 'cuda'):
            from jax.experimental.sparse.linalg import spsolve as jax_spsolve
            J_data, J_csr_indices, J_csr_indptr = _bcoo_to_csr(J, n_reduced)
            delta_V = jax_spsolve(J_data, J_csr_indices, J_csr_indptr, -f, tol=0)
        else:
            J_dense = J.todense()
            try:
                delta_V = jnp.linalg.solve(J_dense, -f)
            except Exception:
                reg = 1e-10 * jnp.eye(n_reduced)
                delta_V = jnp.linalg.solve(J_dense + reg, -f)

        # Voltage limiting
        max_step = 0.5 * vdd
        max_delta = float(jnp.max(jnp.abs(delta_V)))
        if max_delta > max_step:
            delta_V = delta_V * (max_step / max_delta)

        # Line search backtracking
        alpha = 1.0
        V_new = V + alpha * delta_V
        V_new = jnp.clip(V_new, -vdd * 2.0, vdd * 2.0)
        f_new = residual_fn(V_new)
        new_residual = float(jnp.max(jnp.abs(f_new)))

        backtrack_count = 0
        while (new_residual > 5.0 * residual_norm and
               residual_norm > 1e-6 and
               backtrack_count < 3 and
               alpha > 0.1):
            alpha *= 0.5
            V_new = V + alpha * delta_V
            V_new = jnp.clip(V_new, -vdd * 2.0, vdd * 2.0)
            f_new = residual_fn(V_new)
            new_residual = float(jnp.max(jnp.abs(f_new)))
            backtrack_count += 1

        V = V_new
        delta_norm = float(jnp.max(jnp.abs(alpha * delta_V)))
        iterations = iteration + 1

    V_full = jnp.concatenate([jnp.array([0.0]), V])

    info = {
        'converged': converged,
        'iterations': iterations,
        'residual_norm': residual_norm,
        'delta_norm': delta_norm,
        'residual_history': residual_history,
    }

    return V_full, info


# =============================================================================
# Analytical Jacobian Solver (no autodiff)
# =============================================================================

@dataclass
class AnalyticalResidualFunction:
    """Encapsulates a residual function with analytical Jacobian.

    Unlike GPUResidualFunction which relies on sparsejac for autodiff,
    this computes both residual and Jacobian in a single pass using
    analytical device models.
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

            # Compute terminal voltages
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
            in_saturation = (~in_cutoff) & (Vds >= Vov)

            Id = jnp.where(in_cutoff, Id_cutoff,
                          jnp.where(in_linear, Id_lin, Id_sat))
            gm = jnp.where(in_cutoff, gm_cutoff,
                          jnp.where(in_linear, gm_lin, gm_sat))
            gds = jnp.where(in_cutoff, gds_cutoff,
                           jnp.where(in_linear, gds_lin, gds_sat))

            # Add minimum conductance
            Id = Id + gds_min * Vds
            gds = jnp.maximum(gds + gds_min, gds_min)
            gm = jnp.maximum(gm, 1e-12)

            # Current sign convention
            Id_drain = jnp.where(is_pmos, Id, -Id)
            Id_source = jnp.where(is_pmos, -Id, Id)

            residual = residual.at[mosfet_arrays['node_d']].add(Id_drain)
            residual = residual.at[mosfet_arrays['node_s']].add(Id_source)

            # Analytical Jacobian stamps
            # For NMOS: dId_drain/dVd = -gds, dId_drain/dVg = -gm, dId_drain/dVs = gds+gm
            # For PMOS: dId_drain/dVd = gds, dId_drain/dVg = gm, dId_drain/dVs = -gds-gm
            for i, indices in enumerate(mosfet_jac_indices):
                is_p = bool(mosfet_data['is_pmos'][i])
                gm_i = float(gm[i])
                gds_i = float(gds[i])

                # Conductance matrix entries (analytical!)
                if is_p:
                    # PMOS
                    stamps = {
                        ('D', 'D'): gds_i,
                        ('D', 'G'): gm_i,
                        ('D', 'S'): -gds_i - gm_i,
                        ('S', 'D'): -gds_i,
                        ('S', 'G'): -gm_i,
                        ('S', 'S'): gds_i + gm_i,
                    }
                else:
                    # NMOS
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
