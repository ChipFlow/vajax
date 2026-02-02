"""Full MNA transient analysis strategy with adaptive timestep.

This strategy uses true Modified Nodal Analysis (MNA) with branch currents
as explicit unknowns, instead of the high-G (G=1e12) voltage source
approximation used in other strategies.

Benefits:
- More accurate current extraction (no numerical noise from G=1e12)
- Smoother dI/dt transitions matching VACASK reference
- Better conditioned matrices for ill-conditioned circuits

The augmented system has structure:

    ┌───────────────┐   ┌───┐   ┌───────┐
    │  G + c0*C   B │   │ V │   │ f_node│
    │               │ × │   │ = │       │
    │    B^T      0 │   │ J │   │ E - V │
    └───────────────┘   └───┘   └───────┘

Where:
- G = device conductance matrix (n×n)
- C = device capacitance matrix (n×n)
- B = incidence matrix mapping currents to nodes (n×m)
- V = node voltages (n×1)
- J = branch currents (m×1) - these are the primary unknowns for vsources
"""

import dataclasses
import time as time_module
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from jax_spice._logging import logger
from jax_spice.analysis.solver_factories import (
    make_dense_full_mna_solver,
    make_spineax_full_mna_solver,
    make_umfpack_ffi_full_mna_solver,
)


# Check if Spineax (GPU sparse solver) is available
def is_spineax_available() -> bool:
    """Check if Spineax/cuDSS GPU sparse solver is available."""
    try:
        from spineax.cudss.solver import CuDSSSolver  # noqa: F401

        return True
    except ImportError:
        return False


from jax_spice.analysis.integration import IntegrationMethod

from .adaptive import AdaptiveConfig, compute_lte_timestep_jax, predict_voltage_jax
from .base import TransientSetup, TransientStrategy

DEFAULT_MAX_STEPS = 10000

# Memory budget fraction - use 70% of GPU memory for output buffers
# to leave headroom for state arrays, Jacobian, etc.
GPU_MEMORY_BUDGET_FRACTION = 0.7


def compute_checkpoint_interval(
    n_external: int,
    n_vsources: int,
    max_steps: int,
    dtype: Any = jnp.float64,
    target_memory_gb: Optional[float] = None,
) -> Optional[int]:
    """Compute optimal checkpoint interval based on available GPU memory.

    Args:
        n_external: Number of external (user-visible) nodes
        n_vsources: Number of voltage source branch currents
        max_steps: Total number of steps requested
        dtype: Data type for arrays (default: float64)
        target_memory_gb: Override for target memory in GB (for testing).
                         If None, auto-detect from GPU.

    Returns:
        Checkpoint interval, or None if checkpointing not needed (fits in memory)
    """
    # Bytes per element
    bytes_per_elem = 8 if dtype == jnp.float64 else 4

    # Memory per step: times (1) + V_out (n_external) + I_out (n_vsources)
    mem_per_step = (1 + n_external + max(n_vsources, 1)) * bytes_per_elem

    # Try to get GPU memory
    if target_memory_gb is None:
        try:
            devices = jax.devices("gpu")
            if devices:
                # Get memory stats from first GPU
                device = devices[0]
                if hasattr(device, "memory_stats"):
                    stats = device.memory_stats()
                    if stats and "bytes_limit" in stats:
                        target_memory_gb = stats["bytes_limit"] / (1024**3)
        except (RuntimeError, TypeError):
            pass  # No GPU available

        # Conservative defaults if can't detect
        if target_memory_gb is None:
            # Check if we're on GPU backend
            try:
                if jax.default_backend() == "gpu":
                    target_memory_gb = 16.0  # Conservative estimate for typical GPU
                else:
                    return None  # CPU doesn't need checkpointing
            except Exception:
                return None

    # Calculate budget for output arrays
    budget_bytes = int(target_memory_gb * (1024**3) * GPU_MEMORY_BUDGET_FRACTION)

    # Maximum steps that fit in budget
    max_steps_in_budget = budget_bytes // mem_per_step

    if max_steps_in_budget >= max_steps:
        # Fits in memory without checkpointing
        return None

    # Round down to nice number for cache efficiency
    # Use power of 2 or multiple of 1000
    interval = max(1000, max_steps_in_budget)
    # Round to nearest 1000
    interval = (interval // 1000) * 1000
    interval = max(1000, interval)  # Minimum 1000 steps per checkpoint

    return interval


def extract_results(
    times: jax.Array, V_out: jax.Array, stats: Dict
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Extract sliced results from run() output.

    Converts full JAX arrays to sliced numpy arrays for plotting/analysis.
    This performs host transfer and slicing, so call after simulation is complete.

    Args:
        times: Full time array from run()
        V_out: Full voltage array from run()
        stats: Stats dict from run()

    Returns:
        Tuple of (times, voltages, currents) where:
        - times: numpy array of valid time points
        - voltages: Dict mapping node name to numpy voltage array
        - currents: Dict mapping source name to numpy current array
    """
    n = stats["n_steps"]

    # Convert to numpy and slice
    times_np = np.asarray(times)[:n]
    V_np = np.asarray(V_out)[:n]

    voltages = {name: V_np[:, idx] for name, idx in stats["node_indices"].items()}

    currents = {}
    if "I_out" in stats and stats["current_indices"]:
        I_np = np.asarray(stats["I_out"])[:n]
        currents = {name: I_np[:, idx] for name, idx in stats["current_indices"].items()}

    return times_np, voltages, currents


class FullMNAStrategy(TransientStrategy):
    """Full MNA transient with LTE-based adaptive timestep control.

    Uses true Modified Nodal Analysis (MNA) with branch currents as explicit
    unknowns, combined with adaptive timestep control for efficiency.

    Benefits:
    - Better numerical conditioning (no G=1e12 high-G approximation)
    - More accurate current extraction (branch currents are primary unknowns)
    - Smoother dI/dt transitions matching VACASK reference
    - Automatic timestep adjustment based on local truncation error

    Uses lax.while_loop for early termination when t >= t_stop.

    Example:
        runner = CircuitEngine(sim_path)
        runner.parse()

        config = AdaptiveConfig(lte_ratio=0.5, min_dt=1e-15, max_dt=1e-9)
        strategy = FullMNAStrategy(runner, use_sparse=False, config=config)

        times, voltages, stats = strategy.run(t_stop=1e-6, dt=1e-9)
        print(f"Accepted: {stats['accepted_steps']}, Rejected: {stats['rejected_steps']}")
        print(f"I_VDD: {stats['currents']['vdd']}")  # Direct branch current
    """

    @property
    def name(self) -> str:
        """Human-readable strategy name for logging."""
        return "full_mna"

    def __init__(
        self,
        runner,
        use_sparse: bool = False,
        backend: str = "cpu",
        config: Optional[AdaptiveConfig] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        """Initialize FullMNAStrategy.

        Args:
            runner: CircuitEngine instance with parsed netlist
            use_sparse: Use sparse solver (for large circuits)
            backend: Compute backend ('cpu', 'gpu')
            config: Adaptive timestep configuration
            max_steps: Maximum steps for simulation. This value is fixed after
                initialization to ensure JIT cache hits. Changing it would cause
                expensive recompilation (~30s).
        """
        super().__init__(runner, use_sparse=use_sparse, backend=backend)
        self._cached_full_mna_solver: Optional[Callable] = None
        self._cached_full_mna_key: Optional[Tuple] = None
        self.config = config or self._build_config_from_runner()
        # max_steps is fixed at init to avoid JIT recompilation
        self.max_steps = max_steps
        # Cache JIT-compiled while_loop keyed by circuit structure
        # Note: Using Any since JIT-compiled functions have complex types
        self._jit_run_while_cache: Dict[tuple, Any] = {}
        # Build reverse lookup for node names (index -> name)
        self._idx_to_name: Dict[int, str] = {}
        # Cache for _init_mid_rail result (expensive for large circuits)
        self._cached_init_V0: Optional[jax.Array] = None
        self._cached_init_V0_n_total: Optional[int] = None

    def _build_config_from_runner(self) -> AdaptiveConfig:
        """Build AdaptiveConfig from runner's analysis_params (netlist options).

        Maps VACASK-style options to AdaptiveConfig fields:
        - tran_lteratio -> lte_ratio
        - tran_redofactor -> redo_factor
        - nr_convtol -> nr_convtol
        - tran_gshunt -> gshunt_init
        - reltol -> reltol
        - abstol -> abstol
        - tran_fs -> tran_fs
        - tran_minpts -> tran_minpts
        - maxstep -> max_dt
        - tran_method -> integration_method
        """
        params = getattr(self.runner, "analysis_params", {})
        kwargs = {}

        # LTE options
        if "tran_lteratio" in params:
            kwargs["lte_ratio"] = float(params["tran_lteratio"])
        if "tran_redofactor" in params:
            kwargs["redo_factor"] = float(params["tran_redofactor"])

        # NR options
        if "nr_convtol" in params:
            kwargs["nr_convtol"] = float(params["nr_convtol"])

        # GSHUNT options
        if "tran_gshunt" in params:
            kwargs["gshunt_init"] = float(params["tran_gshunt"])

        # Tolerance options
        if "reltol" in params:
            kwargs["reltol"] = float(params["reltol"])
        if "abstol" in params:
            kwargs["abstol"] = float(params["abstol"])

        # Timestep control options
        if "tran_fs" in params:
            kwargs["tran_fs"] = float(params["tran_fs"])
        if "tran_minpts" in params:
            kwargs["tran_minpts"] = int(params["tran_minpts"])
        if "maxstep" in params:
            kwargs["max_dt"] = float(params["maxstep"])

        # Integration method
        if "tran_method" in params:
            kwargs["integration_method"] = params["tran_method"]

        return AdaptiveConfig(**kwargs)

    def _get_node_name(self, idx: int) -> str:
        """Look up node name by index, with lazy initialization."""
        if not self._idx_to_name:
            for name, node_idx in self.runner.node_names.items():
                self._idx_to_name[node_idx] = name
        return self._idx_to_name.get(idx, f"node{idx}")

    def ensure_setup(self) -> TransientSetup:
        """Ensure transient setup is initialized with full MNA data.

        Extends base setup with branch current information for full MNA.
        """
        # Get base setup first (this populates runner caches)
        base_setup = super().ensure_setup()

        # Add full MNA branch data
        from jax_spice.analysis.mna import MNABranchData

        branch_data = MNABranchData.from_devices(self.runner.devices, self.runner.node_names)

        # Augment setup with full MNA info
        self._setup = TransientSetup(
            n_total=base_setup.n_total,
            n_unknowns=base_setup.n_unknowns,
            n_external=base_setup.n_external,
            device_internal_nodes=base_setup.device_internal_nodes,
            source_fn=base_setup.source_fn,
            source_device_data=base_setup.source_device_data,
            openvaf_by_type=base_setup.openvaf_by_type,
            vmapped_fns=base_setup.vmapped_fns,
            static_inputs_cache=base_setup.static_inputs_cache,
            icmode=base_setup.icmode,
            # Full MNA fields
            n_branches=branch_data.n_branches,
            n_augmented=base_setup.n_unknowns + branch_data.n_branches,
            use_full_mna=True,
            branch_data=branch_data,
            branch_node_p=(
                jnp.array(branch_data.node_p, dtype=jnp.int32) if branch_data.node_p else None
            ),
            branch_node_n=(
                jnp.array(branch_data.node_n, dtype=jnp.int32) if branch_data.node_n else None
            ),
        )
        return self._setup

    def _ensure_full_mna_solver(self, setup: TransientSetup) -> Callable:
        """Ensure full MNA solver is created and cached."""
        n_nodes = setup.n_unknowns + 1
        n_vsources = setup.n_branches

        cache_key = (n_nodes, n_vsources, self.use_dense)

        if self._cached_full_mna_solver is not None and self._cached_full_mna_key == cache_key:
            return self._cached_full_mna_solver

        # Create full MNA build_system function
        build_system_fn, device_arrays, total_limit_states = (
            self.runner._make_full_mna_build_system_fn(
                setup.source_device_data,
                setup.vmapped_fns,
                setup.static_inputs_cache,
                setup.n_unknowns,
                use_dense=self.use_dense,
            )
        )

        # Store device arrays and limit state size for solver
        self._device_arrays_full_mna = device_arrays
        self._total_limit_states = total_limit_states

        # JIT compile build_system
        build_system_jit = jax.jit(build_system_fn)

        # Collect NOI node indices
        noi_indices = []
        if setup.device_internal_nodes:
            for dev_name, internal_nodes in setup.device_internal_nodes.items():
                if "node4" in internal_nodes:  # NOI is node4 in PSP103
                    noi_indices.append(internal_nodes["node4"])
        noi_indices = jnp.array(noi_indices, dtype=jnp.int32) if noi_indices else None

        # Create full MNA solver
        # Note: For full MNA, residuals are in Amperes (node) and Volts (branch)
        # Use tighter tolerance than high-G version (which scales by 1e12)
        # abstol=1e-6 means 1µA current error and 1µV voltage error
        if self.use_dense:
            nr_solve = make_dense_full_mna_solver(
                build_system_jit,
                n_nodes,
                n_vsources,
                noi_indices=noi_indices,
                max_iterations=100,
                abstol=1e-6,
                max_step=1.0,
                total_limit_states=total_limit_states,
            )
        else:
            # Sparse path: compute COO→CSR mapping from trial run
            n_augmented = setup.n_unknowns + n_vsources

            # Trial run to get sparse structure
            X_trial = jnp.zeros(n_nodes + n_vsources, dtype=jnp.float64)
            vsource_trial = (
                jnp.zeros(n_vsources, dtype=jnp.float64)
                if n_vsources > 0
                else jnp.zeros(0, dtype=jnp.float64)
            )
            isource_trial = jnp.zeros(0, dtype=jnp.float64)
            Q_trial = jnp.zeros(setup.n_unknowns, dtype=jnp.float64)

            J_bcoo_trial, _, _, _, _ = build_system_fn(
                X_trial,
                vsource_trial,
                isource_trial,
                Q_trial,
                0.0,
                device_arrays,
                1e-12,
                0.0,
                0.0,
                0.0,
                None,
                0.0,
                None,
                None,
            )

            # Extract COO indices
            coo_rows = np.array(J_bcoo_trial.indices[:, 0])
            coo_cols = np.array(J_bcoo_trial.indices[:, 1])
            n_coo = len(coo_rows)

            # Compute linear indices for duplicate detection
            linear_idx = coo_rows * n_augmented + coo_cols

            # Sort COO by linear index (groups duplicates together)
            coo_sort_perm = np.argsort(linear_idx)
            sorted_linear = linear_idx[coo_sort_perm]

            # Find unique entries
            unique_linear, coo_to_unique = np.unique(sorted_linear, return_inverse=True)
            nse = len(unique_linear)

            # Convert sorted unique linear indices back to row/col
            unique_rows = unique_linear // n_augmented
            unique_cols = unique_linear % n_augmented

            # Build CSR indptr and indices
            csr_indptr = np.zeros(n_augmented + 1, dtype=np.int32)
            for row in unique_rows:
                csr_indptr[row + 1] += 1
            csr_indptr = np.cumsum(csr_indptr).astype(np.int32)
            csr_indices = unique_cols.astype(np.int32)

            # Segment IDs for summing duplicates
            csr_segment_ids = coo_to_unique.astype(np.int32)

            logger.info(f"Full MNA sparse: {n_coo} COO -> {nse} CSR entries")

            # Use UMFPACK FFI solver (workspace dependency, always available)
            bcsr_indptr_jax = jnp.array(csr_indptr, dtype=jnp.int32)
            bcsr_indices_jax = jnp.array(csr_indices, dtype=jnp.int32)
            coo_sort_perm_jax = jnp.array(coo_sort_perm, dtype=jnp.int32)
            csr_segment_ids_jax = jnp.array(csr_segment_ids, dtype=jnp.int32)

            # Use Spineax/cuDSS on CUDA, UMFPACK FFI otherwise
            on_cuda = jax.default_backend() in ("cuda", "gpu")

            if on_cuda and is_spineax_available():
                logger.info("Using Spineax/cuDSS solver (GPU sparse)")
                nr_solve = make_spineax_full_mna_solver(
                    build_system_jit,
                    n_nodes,
                    n_vsources,
                    nse,
                    bcsr_indptr=bcsr_indptr_jax,
                    bcsr_indices=bcsr_indices_jax,
                    noi_indices=noi_indices,
                    max_iterations=100,
                    abstol=1e-6,
                    max_step=1.0,
                    coo_sort_perm=coo_sort_perm_jax,
                    csr_segment_ids=csr_segment_ids_jax,
                )
            else:
                logger.info("Using UMFPACK FFI solver (zero callback overhead)")
                nr_solve = make_umfpack_ffi_full_mna_solver(
                    build_system_jit,
                    n_nodes,
                    n_vsources,
                    nse,
                    bcsr_indptr=bcsr_indptr_jax,
                    bcsr_indices=bcsr_indices_jax,
                    noi_indices=noi_indices,
                    max_iterations=100,
                    abstol=1e-6,
                    max_step=1.0,
                    coo_sort_perm=coo_sort_perm_jax,
                    csr_segment_ids=csr_segment_ids_jax,
                )

        self._cached_full_mna_solver = nr_solve
        self._cached_full_mna_key = cache_key
        self._cached_build_system_jit = build_system_jit  # Store for Q_init computation
        logger.info(f"Created full MNA solver: V({n_nodes}) + I({n_vsources})")

        return nr_solve

    def _init_mid_rail(self, setup: TransientSetup, n_total: int) -> jax.Array:
        """Initialize voltage vector with mid-rail values.

        Provides a good starting point for DC convergence.
        Results are cached since circuit topology doesn't change between runs.

        Args:
            setup: TransientSetup with device info
            n_total: Total number of nodes

        Returns:
            V0: Initial voltage vector
        """
        # Check cache - circuit topology doesn't change between runs
        if self._cached_init_V0 is not None and self._cached_init_V0_n_total == n_total:
            return self._cached_init_V0

        vdd_value = self.runner._get_vdd_value()
        mid_rail = vdd_value / 2.0
        V = jnp.full(n_total, mid_rail, dtype=jnp.float64)
        V = V.at[0].set(0.0)  # Ground is always 0

        # Set VDD/GND nodes
        for name, idx in self.runner.node_names.items():
            name_lower = name.lower()
            if "vdd" in name_lower or "vcc" in name_lower:
                V = V.at[idx].set(vdd_value)
            elif name_lower in ("gnd", "vss", "0"):
                V = V.at[idx].set(0.0)

        # Initialize voltage source nodes to target values
        for dev in self.runner.devices:
            if dev["model"] == "vsource":
                nodes = dev.get("nodes", [])
                if len(nodes) >= 2:
                    p_node, n_node = nodes[0], nodes[1]
                    dc_val = float(dev["params"].get("dc", 0.0))
                    if n_node == 0 and p_node > 0:
                        V = V.at[p_node].set(dc_val)

        # Initialize NOI nodes to 0V
        if setup.device_internal_nodes:
            for dev_name, internal_nodes in setup.device_internal_nodes.items():
                if "node4" in internal_nodes:
                    noi_idx = internal_nodes["node4"]
                    V = V.at[noi_idx].set(0.0)

        # Cache result
        self._cached_init_V0 = V
        self._cached_init_V0_n_total = n_total

        return V

    def warmup(self, dt: Optional[float] = None) -> float:
        """Pre-compile the JIT function by running a minimal simulation.

        This triggers JIT compilation so subsequent run() calls are fast.
        Uses self.max_steps set at initialization.

        Args:
            dt: Timestep to use. If None, uses 'step' from netlist.

        Returns:
            Wall time for warmup (compilation + execution)
        """
        # Use netlist step as default
        if dt is None:
            params = getattr(self.runner, "analysis_params", {})
            dt = params.get("step", 1e-12)

        logger.info(f"{self.name}: Starting warmup (max_steps={self.max_steps})")

        t_start = time_module.perf_counter()
        # Run tiny simulation - just enough to trigger compilation
        self.run(t_stop=dt, dt=dt)
        wall_time = time_module.perf_counter() - t_start

        logger.info(f"{self.name}: Warmup complete in {wall_time:.2f}s")
        return wall_time

    def run(
        self,
        t_stop: Optional[float] = None,
        dt: Optional[float] = None,
        checkpoint_interval: Optional[int] = None,
    ) -> Tuple[jax.Array, jax.Array, Dict]:
        """Run adaptive transient analysis with full MNA.

        Args:
            t_stop: Simulation stop time in seconds. If None, uses 'stop' from
                netlist analysis line.
            dt: Initial time step in seconds. If None, uses 'step' from netlist
                analysis line.
            checkpoint_interval: If set, use GPU memory checkpointing with this
                many steps per buffer. Results are periodically copied to CPU
                to avoid GPU OOM on large circuits. Recommended for circuits
                with many nodes (>1000) and long simulations (>10000 steps).

        Note:
            max_steps is set at strategy initialization to avoid JIT recompilation.
            To change it, create a new strategy instance.

        Returns:
            Tuple of (times, V_out, stats) where:
            - times: Full time array [max_steps] - valid data is [:n_steps]
            - V_out: Full voltage array [max_steps, n_nodes] - valid data is [:n_steps]
            - stats: Dict with:
                - n_steps: Number of valid timesteps
                - I_out: Full current array [max_steps, n_vsources]
                - node_indices: Dict mapping node name -> column index in V_out
                - current_indices: Dict mapping source name -> column index in I_out
                - Other statistics (wall_time, rejected_steps, etc.)

        Example:
            # Use netlist defaults
            times, V_out, stats = strategy.run()

            # Or override
            times, V_out, stats = strategy.run(t_stop=1e-6, dt=1e-9)

            n = stats['n_steps']
            # Get voltage at node 'out': V_out[:n, stats['node_indices']['out']]
            # Or use helper: extract_voltages(times, V_out, stats)
        """
        # Use fixed max_steps from initialization (avoids JIT recompilation)
        max_steps = self.max_steps

        # Use netlist analysis params as defaults
        params = getattr(self.runner, "analysis_params", {})
        if t_stop is None:
            t_stop = params.get("stop")
            if t_stop is None:
                raise ValueError("t_stop not specified and 'stop' not found in netlist")
        if dt is None:
            dt = params.get("step")
            if dt is None:
                raise ValueError("dt not specified and 'step' not found in netlist")
        setup = self.ensure_setup()
        nr_solve = self._ensure_full_mna_solver(setup)

        n_total = setup.n_total
        n_unknowns = setup.n_unknowns
        n_vsources = setup.n_branches
        n_external = setup.n_external
        source_fn = setup.source_fn
        config = self.config

        # Compute effective hmax based on tran_minpts (VACASK default: 50)
        # This ensures at least tran_minpts output points in the simulation
        if config.tran_minpts > 0:
            hmax_from_minpts = t_stop / config.tran_minpts
            effective_max_dt = min(config.max_dt, hmax_from_minpts)
            if effective_max_dt < config.max_dt:
                logger.info(
                    f"{self.name}: tran_minpts={config.tran_minpts} -> hmax={effective_max_dt:.2e}s"
                )
                config = dataclasses.replace(config, max_dt=effective_max_dt)

        # Apply initial timestep scaling (VACASK uses tran_fs=0.25 by default)
        # This helps with startup transients by using smaller initial steps
        if config.tran_fs != 1.0:
            dt_original = dt
            dt = dt * config.tran_fs
            logger.info(
                f"{self.name}: Applied tran_fs={config.tran_fs} scaling: dt={dt_original:.2e}s -> {dt:.2e}s"
            )

        device_arrays = self._device_arrays_full_mna
        dtype = jnp.float64

        # Build JIT-compatible source evaluator
        jit_source_eval = self._make_jit_source_eval(setup, source_fn)

        # Initialize solution
        X0 = jnp.zeros(n_total + n_vsources, dtype=dtype)
        X0 = X0.at[:n_total].set(self._init_mid_rail(setup, n_total))
        Q_init = jnp.zeros(n_unknowns, dtype=dtype)
        I_vsource_dc = jnp.zeros(max(n_vsources, 1), dtype=dtype)

        # DC operating point (skip if icmode='uic')
        vsource_vals_init, isource_vals_init = self._build_source_arrays(source_fn(0.0))

        if setup.icmode == "uic":
            # Use Initial Conditions - skip DC solve, start from 0V
            # This matches VACASK behavior: all nodes start at 0V, voltage sources
            # force their values. The initial state may be inconsistent (like SPICE3).
            logger.info(f"{self.name}: icmode='uic' - skipping DC solve, using initial conditions")

            # All nodes start at 0V (VACASK behavior)
            X0 = X0.at[:n_total].set(0.0)

            # Compute initial charges consistent with initial voltages
            # Note: The state may not be at DC equilibrium - this is expected for UIC
            # Use gmin from netlist options (default 1e-12)
            dc_gmin = getattr(self.runner.options, "gmin", 1e-12)
            _, _, Q_init, I_vsource_dc, _ = self._cached_build_system_jit(
                X0,
                vsource_vals_init,
                isource_vals_init,
                jnp.zeros(n_unknowns, dtype=dtype),
                0.0,
                device_arrays,
                dc_gmin,
                0.0,
                0.0,
                0.0,
                None,
                0.0,
                None,
                None,  # Last None is limit_state_in
            )
            logger.info(
                f"{self.name}: Initial state - Q_max={float(jnp.max(jnp.abs(Q_init))):.2e}, "
                f"I_vdd={float(I_vsource_dc[0]) if len(I_vsource_dc) > 0 else 0:.2e}A"
            )
        else:
            # Compute DC operating point
            # Use gmin from netlist options (default 1e-12)
            dc_gmin = getattr(self.runner.options, "gmin", 1e-12)
            X_dc, _, dc_converged, dc_residual, Q_dc, _, I_vsource_dc, _ = nr_solve(
                X0,
                vsource_vals_init,
                isource_vals_init,
                jnp.zeros(n_unknowns, dtype=dtype),
                0.0,
                device_arrays,
                dc_gmin,
                0.0,
                0.0,
                0.0,
                None,
                0.0,
                None,
                limit_state_in=None,  # Device-level limiting state (optional)
            )

            if dc_converged:
                X0 = X_dc
                Q_init = Q_dc
                logger.info(f"{self.name}: DC converged, V[1]={float(X0[1]):.4f}V")
            else:
                logger.warning(f"{self.name}: DC did not converge (residual={dc_residual:.2e})")

        # History buffers
        max_history = config.max_order + 2
        V_history = jnp.zeros((max_history, n_total), dtype=dtype)
        V_history = V_history.at[0].set(X0[:n_total])
        dt_history = jnp.full(max_history, dt, dtype=dtype)

        # Determine if we should use checkpoint mode
        # checkpoint_interval can be:
        #   - None: auto-detect based on GPU memory
        #   - int > 0: use specified interval
        #   - int <= 0: disable checkpointing

        effective_checkpoint_interval = checkpoint_interval

        # Auto-detect checkpoint interval based on GPU memory
        if checkpoint_interval is None:
            effective_checkpoint_interval = compute_checkpoint_interval(
                n_external, n_vsources, max_steps, dtype
            )
            if effective_checkpoint_interval is not None:
                logger.info(
                    f"{self.name}: Auto-enabled GPU checkpointing "
                    f"(interval={effective_checkpoint_interval})"
                )

        use_checkpoints = (
            effective_checkpoint_interval is not None
            and effective_checkpoint_interval > 0
            and max_steps > effective_checkpoint_interval
        )

        if use_checkpoints:
            # Checkpoint mode: use smaller buffers and outer Python loop
            return self._run_with_checkpoints(
                setup=setup,
                nr_solve=nr_solve,
                jit_source_eval=jit_source_eval,
                device_arrays=device_arrays,
                X0=X0,
                Q_init=Q_init,
                I_vsource_dc=I_vsource_dc,
                V_history=V_history,
                dt_history=dt_history,
                t_stop=t_stop,
                dt=dt,
                max_steps=max_steps,
                checkpoint_interval=effective_checkpoint_interval,
                n_total=n_total,
                n_unknowns=n_unknowns,
                n_external=n_external,
                n_vsources=n_vsources,
                config=config,
                dtype=dtype,
            )

        # Standard mode: single while_loop with full output arrays
        # Output arrays
        times_out = jnp.zeros(max_steps, dtype=dtype)
        times_out = times_out.at[0].set(0.0)
        V_out = jnp.zeros((max_steps, n_external), dtype=dtype)
        V_out = V_out.at[0].set(X0[:n_external])
        I_out = jnp.zeros((max_steps, max(n_vsources, 1)), dtype=dtype)
        if n_vsources > 0:
            I_out = I_out.at[0].set(I_vsource_dc[:n_vsources])

        # Initial state
        # Initialize historic max with DC solution voltages
        V_max_historic_init = jnp.abs(X0[:n_total])
        # Initialize device-level limiting state (flat array for all devices)
        total_limit_states = self._total_limit_states
        if total_limit_states > 0:
            limit_state_init = jnp.zeros(total_limit_states, dtype=dtype)
        else:
            limit_state_init = jnp.zeros(0, dtype=dtype)
        init_state = FullMNAState(
            t=jnp.array(0.0, dtype=dtype),
            dt=jnp.array(dt, dtype=dtype),
            X=X0,
            Q_prev=Q_init,
            dQdt_prev=jnp.zeros(n_unknowns, dtype=dtype),
            Q_prev2=jnp.zeros(n_unknowns, dtype=dtype),
            V_history=V_history,
            dt_history=dt_history,
            history_count=jnp.array(1, dtype=jnp.int32),
            times_out=times_out,
            V_out=V_out,
            I_out=I_out,
            step_idx=jnp.array(1, dtype=jnp.int32),  # Start at 1 (0 is initial)
            warmup_count=jnp.array(0, dtype=jnp.int32),
            t_stop=jnp.array(t_stop, dtype=dtype),
            max_dt=jnp.array(config.max_dt, dtype=dtype),
            total_nr_iters=jnp.array(0, dtype=jnp.int32),
            rejected_steps=jnp.array(0, dtype=jnp.int32),
            min_dt_used=jnp.array(dt, dtype=dtype),
            max_dt_used=jnp.array(dt, dtype=dtype),
            consecutive_rejects=jnp.array(0, dtype=jnp.int32),
            V_max_historic=V_max_historic_init,
            limit_state=limit_state_init,
        )

        # Cache key does NOT include t_stop or max_dt since they're passed dynamically via state
        cache_key = (max_steps, n_total, n_unknowns, n_external, n_vsources, dtype)

        if cache_key not in self._jit_run_while_cache:
            # LRU eviction: limit cache size to prevent unbounded growth
            MAX_JIT_CACHE_SIZE = 8
            if len(self._jit_run_while_cache) >= MAX_JIT_CACHE_SIZE:
                oldest_key = next(iter(self._jit_run_while_cache))
                del self._jit_run_while_cache[oldest_key]
                logger.debug(f"{self.name}: Evicted oldest JIT cache entry")

            # Create while_loop functions using module-level function
            cond_fn, body_fn = _make_full_mna_while_loop_fns(
                nr_solve,
                jit_source_eval,
                device_arrays,
                config,
                n_total,
                n_unknowns,
                n_external,
                n_vsources,
                max_steps,
                config.warmup_steps,
                dtype,
            )

            # JIT-compile the while_loop for performance
            @jax.jit
            def run_while(state):
                return lax.while_loop(cond_fn, body_fn, state)

            self._jit_run_while_cache[cache_key] = run_while
            logger.debug(
                f"{self.name}: Created JIT runner for max_steps={max_steps} (t_stop is dynamic)"
            )

        run_while = self._jit_run_while_cache[cache_key]

        # Run simulation
        logger.info(
            f"{self.name}: Starting simulation (t_stop={t_stop:.2e}s, dt={dt:.2e}s, "
            f"max_steps={max_steps}, {'sparse' if self.use_sparse else 'dense'})"
        )
        t_start = time_module.perf_counter()

        final_state = run_while(init_state)

        # Block until computation completes for accurate timing
        final_state.step_idx.block_until_ready()
        wall_time = time_module.perf_counter() - t_start

        # Return full arrays - users slice with [:n_steps] if needed
        n_steps = int(final_state.step_idx)
        times = final_state.times_out
        V_out = final_state.V_out
        I_out = final_state.I_out

        # Build node name -> column index mapping for V_out
        node_indices: Dict[str, int] = {}
        for name, idx in self.runner.node_names.items():
            if 0 < idx < n_external:
                node_indices[name] = idx

        # Build current name -> column index mapping for I_out
        current_indices: Dict[str, int] = {}
        if setup.branch_data and n_vsources > 0:
            current_indices = dict(setup.branch_data.name_to_idx)

        rejected = int(final_state.rejected_steps)
        stats = {
            "n_steps": n_steps,  # Valid data is [:n_steps]
            "total_timesteps": n_steps,
            "accepted_steps": n_steps,
            "rejected_steps": rejected,
            "total_nr_iterations": int(final_state.total_nr_iters),
            "avg_nr_iterations": float(final_state.total_nr_iters) / max(n_steps, 1),
            "wall_time": wall_time,
            "time_per_step_ms": wall_time / n_steps * 1000 if n_steps > 0 else 0,
            "min_dt_used": float(final_state.min_dt_used),
            "max_dt_used": float(final_state.max_dt_used),
            "convergence_rate": n_steps / max(n_steps + rejected, 1),
            "strategy": "adaptive_full_mna",
            "solver": "sparse" if self.use_sparse else "dense",
            "V_out": V_out,  # Full voltage array [max_steps, n_nodes]
            "I_out": I_out,  # Full current array [max_steps, n_vsources]
            "node_indices": node_indices,  # name -> column index for V_out
            "current_indices": current_indices,  # name -> column index for I_out
        }

        logger.info(
            f"{self.name}: Completed {n_steps} steps in {wall_time:.3f}s "
            f"({stats['time_per_step_ms']:.2f}ms/step, "
            f"{int(final_state.rejected_steps)} rejected, "
            f"dt range [{float(final_state.min_dt_used):.2e}, "
            f"{float(final_state.max_dt_used):.2e}])"
        )

        return times, V_out, stats

    def _run_with_checkpoints(
        self,
        setup: TransientSetup,
        nr_solve: Callable,
        jit_source_eval: Callable,
        device_arrays: Any,
        X0: jax.Array,
        Q_init: jax.Array,
        I_vsource_dc: jax.Array,
        V_history: jax.Array,
        dt_history: jax.Array,
        t_stop: float,
        dt: float,
        max_steps: int,
        checkpoint_interval: int,
        n_total: int,
        n_unknowns: int,
        n_external: int,
        n_vsources: int,
        config: AdaptiveConfig,
        dtype: Any,
    ) -> Tuple[jax.Array, jax.Array, Dict]:
        """Run simulation with periodic checkpoints to CPU memory.

        Uses smaller GPU buffers and periodically transfers results to CPU,
        enabling simulations that would otherwise exceed GPU memory.
        """
        # CPU accumulators for results (numpy arrays)
        cpu_times: list = []
        cpu_V: list = []
        cpu_I: list = []

        # Track total statistics across checkpoints
        total_steps = 0
        total_nr_iters = 0
        total_rejected = 0
        min_dt_global = dt
        max_dt_global = dt

        # Current simulation state (carried across checkpoints)
        current_t = 0.0
        current_dt = dt
        current_X = X0
        current_Q_prev = Q_init
        current_dQdt_prev = jnp.zeros(n_unknowns, dtype=dtype)
        current_Q_prev2 = jnp.zeros(n_unknowns, dtype=dtype)
        current_V_history = V_history
        current_dt_history = dt_history
        current_history_count = 1
        current_warmup_count = 0
        current_V_max_historic = jnp.abs(X0[:n_total])
        # Initialize device-level limiting state for checkpoint mode
        total_limit_states = self._total_limit_states
        if total_limit_states > 0:
            current_limit_state = jnp.zeros(total_limit_states, dtype=dtype)
        else:
            current_limit_state = jnp.zeros(0, dtype=dtype)

        # Get or create JIT-compiled while_loop for checkpoint_interval size
        cache_key = (checkpoint_interval, n_total, n_unknowns, n_external, n_vsources, dtype)
        if cache_key not in self._jit_run_while_cache:
            MAX_JIT_CACHE_SIZE = 8
            if len(self._jit_run_while_cache) >= MAX_JIT_CACHE_SIZE:
                oldest_key = next(iter(self._jit_run_while_cache))
                del self._jit_run_while_cache[oldest_key]

            cond_fn, body_fn = _make_full_mna_while_loop_fns(
                nr_solve,
                jit_source_eval,
                device_arrays,
                config,
                n_total,
                n_unknowns,
                n_external,
                n_vsources,
                checkpoint_interval,
                config.warmup_steps,
                dtype,
            )

            @jax.jit
            def run_while(state):
                return lax.while_loop(cond_fn, body_fn, state)

            self._jit_run_while_cache[cache_key] = run_while

        run_while = self._jit_run_while_cache[cache_key]

        # Log checkpoint mode
        n_checkpoints = (max_steps + checkpoint_interval - 1) // checkpoint_interval
        buffer_mb = checkpoint_interval * n_external * 8 / (1024 * 1024)
        logger.info(
            f"{self.name}: Checkpoint mode - {checkpoint_interval} steps/buffer "
            f"({buffer_mb:.1f}MB), up to {n_checkpoints} checkpoints"
        )

        t_start = time_module.perf_counter()
        checkpoint_num = 0
        first_checkpoint = True

        while total_steps < max_steps and current_t < t_stop:
            checkpoint_num += 1
            steps_remaining = max_steps - total_steps

            # Create output buffers for this checkpoint
            buffer_size = min(checkpoint_interval, steps_remaining)
            times_out = jnp.zeros(buffer_size, dtype=dtype)
            V_out = jnp.zeros((buffer_size, n_external), dtype=dtype)
            I_out = jnp.zeros((buffer_size, max(n_vsources, 1)), dtype=dtype)

            # For first checkpoint, include initial conditions at index 0
            if first_checkpoint:
                times_out = times_out.at[0].set(current_t)
                V_out = V_out.at[0].set(current_X[:n_external])
                if n_vsources > 0:
                    I_out = I_out.at[0].set(I_vsource_dc[:n_vsources])
                start_idx = 1
            else:
                start_idx = 0

            # Build state for this checkpoint
            state = FullMNAState(
                t=jnp.array(current_t, dtype=dtype),
                dt=jnp.array(current_dt, dtype=dtype),
                X=current_X,
                Q_prev=current_Q_prev,
                dQdt_prev=current_dQdt_prev,
                Q_prev2=current_Q_prev2,
                V_history=current_V_history,
                dt_history=current_dt_history,
                history_count=jnp.array(current_history_count, dtype=jnp.int32),
                times_out=times_out,
                V_out=V_out,
                I_out=I_out,
                step_idx=jnp.array(start_idx, dtype=jnp.int32),
                warmup_count=jnp.array(current_warmup_count, dtype=jnp.int32),
                t_stop=jnp.array(t_stop, dtype=dtype),
                max_dt=jnp.array(config.max_dt, dtype=dtype),
                total_nr_iters=jnp.array(0, dtype=jnp.int32),
                rejected_steps=jnp.array(0, dtype=jnp.int32),
                min_dt_used=jnp.array(current_dt, dtype=dtype),
                max_dt_used=jnp.array(current_dt, dtype=dtype),
                consecutive_rejects=jnp.array(0, dtype=jnp.int32),
                V_max_historic=current_V_max_historic,
                limit_state=current_limit_state,
            )

            # Run while_loop for this checkpoint
            final_state = run_while(state)

            # Block and extract results to CPU
            final_state.step_idx.block_until_ready()
            n_filled = int(final_state.step_idx)

            # Copy valid results to CPU numpy arrays
            if n_filled > 0:
                cpu_times.append(np.array(final_state.times_out[:n_filled]))
                cpu_V.append(np.array(final_state.V_out[:n_filled]))
                cpu_I.append(np.array(final_state.I_out[:n_filled]))

            # Update statistics
            total_steps += n_filled
            total_nr_iters += int(final_state.total_nr_iters)
            total_rejected += int(final_state.rejected_steps)
            min_dt_global = min(min_dt_global, float(final_state.min_dt_used))
            max_dt_global = max(max_dt_global, float(final_state.max_dt_used))

            # Carry forward simulation state for next checkpoint
            current_t = float(final_state.t)
            current_dt = float(final_state.dt)
            current_X = final_state.X
            current_Q_prev = final_state.Q_prev
            current_dQdt_prev = final_state.dQdt_prev
            current_Q_prev2 = final_state.Q_prev2
            current_V_history = final_state.V_history
            current_dt_history = final_state.dt_history
            current_history_count = int(final_state.history_count)
            current_warmup_count = int(final_state.warmup_count)
            current_V_max_historic = final_state.V_max_historic
            current_limit_state = final_state.limit_state

            first_checkpoint = False

            logger.debug(
                f"{self.name}: Checkpoint {checkpoint_num} - {n_filled} steps, "
                f"t={current_t:.2e}s, total={total_steps}"
            )

            # Check if simulation completed (reached t_stop before filling buffer)
            if current_t >= t_stop:
                break

        wall_time = time_module.perf_counter() - t_start

        # Concatenate all checkpoints into final arrays
        if cpu_times:
            times_np = np.concatenate(cpu_times)
            V_np = np.concatenate(cpu_V, axis=0)
            I_np = np.concatenate(cpu_I, axis=0)
        else:
            times_np = np.zeros(0, dtype=np.float64)
            V_np = np.zeros((0, n_external), dtype=np.float64)
            I_np = np.zeros((0, max(n_vsources, 1)), dtype=np.float64)

        # Convert to JAX arrays for consistent return type
        times = jnp.array(times_np)
        V_out = jnp.array(V_np)
        I_out = jnp.array(I_np)

        # Build node/current name mappings
        node_indices: Dict[str, int] = {}
        for name, idx in self.runner.node_names.items():
            if 0 < idx < n_external:
                node_indices[name] = idx

        current_indices: Dict[str, int] = {}
        if setup.branch_data and n_vsources > 0:
            current_indices = dict(setup.branch_data.name_to_idx)

        n_steps = total_steps
        stats = {
            "n_steps": n_steps,
            "total_timesteps": n_steps,
            "accepted_steps": n_steps,
            "rejected_steps": total_rejected,
            "total_nr_iterations": total_nr_iters,
            "avg_nr_iterations": total_nr_iters / max(n_steps, 1),
            "wall_time": wall_time,
            "time_per_step_ms": wall_time / n_steps * 1000 if n_steps > 0 else 0,
            "min_dt_used": min_dt_global,
            "max_dt_used": max_dt_global,
            "convergence_rate": n_steps / max(n_steps + total_rejected, 1),
            "strategy": "adaptive_full_mna_checkpointed",
            "solver": "sparse" if self.use_sparse else "dense",
            "V_out": V_out,
            "I_out": I_out,
            "node_indices": node_indices,
            "current_indices": current_indices,
            "checkpoints": checkpoint_num,
            "checkpoint_interval": checkpoint_interval,
        }

        logger.info(
            f"{self.name}: Completed {n_steps} steps in {wall_time:.3f}s "
            f"({stats['time_per_step_ms']:.2f}ms/step, {checkpoint_num} checkpoints, "
            f"{total_rejected} rejected, dt range [{min_dt_global:.2e}, {max_dt_global:.2e}])"
        )

        return times, V_out, stats

    def _make_jit_source_eval(self, setup: TransientSetup, source_fn: Callable) -> Callable:
        """Create JIT-compatible source evaluator.

        Uses jnp.stack instead of jnp.array to handle traced values properly.
        """
        source_data = setup.source_device_data
        vsource_data = source_data.get("vsource", {"names": [], "dc_values": []})
        isource_data = source_data.get("isource", {"names": [], "dc_values": []})

        vsource_names = vsource_data.get("names", [])
        isource_names = isource_data.get("names", [])
        n_vsources = len(vsource_names)
        n_isources = len(isource_names)

        dtype = jnp.float64

        if n_vsources == 0 and n_isources == 0:

            def eval_sources(t):
                return jnp.zeros(0, dtype=dtype), jnp.zeros(0, dtype=dtype)

            return eval_sources

        # Get DC values as JAX arrays for fallback
        vsource_dc = jnp.array(vsource_data.get("dc_values", [0.0] * n_vsources), dtype=dtype)
        isource_dc = jnp.array(isource_data.get("dc_values", [0.0] * n_isources), dtype=dtype)

        def eval_sources(t):
            source_values = source_fn(t)

            # Build vsource array using jnp.stack (JIT-compatible)
            if n_vsources == 0:
                vsource_vals = jnp.zeros(0, dtype=dtype)
            elif n_vsources == 1:
                val = source_values.get(vsource_names[0], vsource_dc[0])
                vsource_vals = jnp.array([val], dtype=dtype)
            else:
                vals = [source_values.get(name, dc) for name, dc in zip(vsource_names, vsource_dc)]
                vsource_vals = jnp.stack(vals)

            # Build isource array using jnp.stack (JIT-compatible)
            if n_isources == 0:
                isource_vals = jnp.zeros(0, dtype=dtype)
            elif n_isources == 1:
                val = source_values.get(isource_names[0], isource_dc[0])
                isource_vals = jnp.array([val], dtype=dtype)
            else:
                vals = [source_values.get(name, dc) for name, dc in zip(isource_names, isource_dc)]
                isource_vals = jnp.stack(vals)

            return vsource_vals, isource_vals

        return eval_sources


class FullMNAState(NamedTuple):
    """State for while-loop based full MNA adaptive timestep."""

    t: jax.Array  # Current time
    dt: jax.Array  # Current timestep
    X: jax.Array  # Current solution [V; I_branch]
    Q_prev: jax.Array  # Previous charge state
    dQdt_prev: jax.Array  # Previous dQ/dt
    Q_prev2: jax.Array  # Charge two steps ago (for Gear2)
    V_history: jax.Array  # History of voltage vectors (max_history, n_total)
    dt_history: jax.Array  # History of timesteps (max_history,)
    history_count: jax.Array  # Number of valid history entries
    times_out: jax.Array  # Output time array
    V_out: jax.Array  # Output voltage array (max_steps, n_external)
    I_out: jax.Array  # Output current array (max_steps, n_vsources)
    step_idx: jax.Array  # Current output step index
    warmup_count: jax.Array  # Warmup steps completed
    t_stop: jax.Array  # Target stop time (passed through)
    max_dt: jax.Array  # Maximum timestep (passed through, from tran_minpts)
    total_nr_iters: jax.Array  # Total NR iterations
    rejected_steps: jax.Array  # Number of rejected steps
    min_dt_used: jax.Array  # Minimum dt actually used
    max_dt_used: jax.Array  # Maximum dt actually used
    consecutive_rejects: jax.Array  # Consecutive LTE rejections (reset on accept)
    V_max_historic: jax.Array  # Historic max |V| per node (for LTE tolerance)
    limit_state: jax.Array  # Device-level limiting state (flat array for all devices)


def _make_full_mna_while_loop_fns(
    nr_solve,
    jit_source_eval: Callable,
    device_arrays,
    config: AdaptiveConfig,
    n_total: int,
    n_unknowns: int,
    n_external: int,
    n_vsources: int,
    max_steps: int,
    warmup_steps: int,
    dtype,
):
    """Create cond and body functions for full MNA while_loop adaptive timestep.

    This is a module-level function to enable JAX JIT caching. The functions
    are parameterized by circuit structure (n_total, etc.) but t_stop is
    passed dynamically via state to allow reuse.
    """
    max_history = config.max_order + 2

    # Extract gshunt config for ramping
    gshunt_init = config.gshunt_init
    gshunt_target = config.gshunt_target
    gshunt_steps = config.gshunt_steps

    # Progress reporting
    progress_interval = config.progress_interval

    def _progress_callback(step, t, t_stop, dt, rejected):
        """Print progress during transient simulation."""
        pct = 100.0 * float(t) / float(t_stop) if float(t_stop) > 0 else 0.0
        print(
            f"Step {int(step):6d}: t={float(t):.3e}s ({pct:5.1f}%), "
            f"dt={float(dt):.2e}s, rejected={int(rejected)}"
        )

    def cond_fn(state: FullMNAState) -> jax.Array:
        """Continue while t < t_stop and step_idx < max_steps."""
        return (state.t < state.t_stop) & (state.step_idx < max_steps)

    # Extract integration method - compute coefficients in body_fn
    integ_method = config.integration_method

    def body_fn(state: FullMNAState) -> FullMNAState:
        """One iteration of full MNA adaptive timestep loop."""
        t = state.t
        dt_cur = state.dt
        X = state.X
        Q_prev = state.Q_prev
        dQdt_prev = state.dQdt_prev
        Q_prev2 = state.Q_prev2

        # Compute integration coefficients based on method
        # Using compile-time constants since method is fixed per run
        inv_dt = 1.0 / dt_cur
        if integ_method == IntegrationMethod.BACKWARD_EULER:
            # BE: dQ/dt = (Q_new - Q_prev) / dt
            c0 = inv_dt
            c1 = -inv_dt
            d1 = 0.0
            c2 = 0.0
            error_coeff_integ = -0.5
        elif integ_method == IntegrationMethod.TRAPEZOIDAL:
            # Trap: dQ/dt = 2/dt * (Q_new - Q_prev) - dQdt_prev
            c0 = 2.0 * inv_dt
            c1 = -2.0 * inv_dt
            d1 = -1.0
            c2 = 0.0
            # VACASK uses 1/24 (h^3/24 LTE coeff) which with pred_err_coeff~-1
            # gives factor ~ 0.04, matching VACASK's LTE scaling
            error_coeff_integ = 1.0 / 24.0
        else:  # GEAR2/BDF2
            # Gear2: dQ/dt = (3*Q_new - 4*Q_prev + Q_prev2) / (2*dt)
            c0 = 1.5 * inv_dt
            c1 = -2.0 * inv_dt
            d1 = 0.0
            c2 = 0.5 * inv_dt
            error_coeff_integ = -2.0 / 9.0

        t_next = t + dt_cur
        vsource_vals, isource_vals = jit_source_eval(t_next)

        # Prediction using shared function
        warmup_complete = state.warmup_count >= warmup_steps
        can_predict = warmup_complete & (state.history_count >= 2)

        V_pred, pred_err_coeff = predict_voltage_jax(
            state.V_history,
            state.dt_history,
            state.history_count,
            dt_cur,
            config.max_order,
            debug=config.debug_lte,
            debug_node=24,
        )

        # Initialize X with predicted voltages
        X_init = jnp.where(can_predict, X.at[:n_total].set(V_pred), X)

        # Compute current gshunt (linear ramp from init to target)
        ramp_progress = jnp.where(
            gshunt_steps > 0,
            jnp.clip(state.step_idx / gshunt_steps, 0.0, 1.0),
            1.0,  # If no steps, use target immediately
        )
        current_gshunt = gshunt_init + ramp_progress * (gshunt_target - gshunt_init)

        # Newton-Raphson solve with device-level limiting state
        X_new, iterations, converged, max_f, Q, dQdt_out, I_vsource, limit_state_out = nr_solve(
            X_init,
            vsource_vals,
            isource_vals,
            Q_prev,
            c0,
            device_arrays,
            1e-12,
            current_gshunt,
            c1,
            d1,
            dQdt_prev,
            c2,
            Q_prev2,
            limit_state_in=state.limit_state,
        )

        new_total_nr_iters = state.total_nr_iters + jnp.int32(iterations)

        # Handle NR failure
        nr_failed = ~converged
        at_min_dt = dt_cur <= config.min_dt
        # When NR fails at min_dt, we accept the step (to advance time) but DON'T use the
        # bad solution - we keep the previous solution to prevent state corruption.
        # This is tracked by nr_failed_at_min_dt and used in the state update below.
        nr_failed_at_min_dt = nr_failed & at_min_dt

        # LTE estimation (on voltage part only) using shared function
        V_new = X_new[:n_total]
        dt_lte, lte_norm = compute_lte_timestep_jax(
            V_new,
            V_pred,
            pred_err_coeff,
            dt_cur,
            state.history_count,
            config,
            error_coeff_integ,
            debug_lte=config.debug_lte,
            step_idx=state.step_idx,
            V_max_historic=state.V_max_historic,  # Use historic max for tolerance (VACASK relrefAlllocal)
        )

        # Accept/reject decision
        lte_reject = (dt_cur / dt_lte > config.redo_factor) & can_predict & converged
        nr_reject = nr_failed & ~at_min_dt

        # Check for forced acceptance after too many consecutive LTE rejects
        # This prevents timestep from collapsing to femtoseconds
        force_accept = state.consecutive_rejects >= config.max_consecutive_rejects

        reject_step = (lte_reject | nr_reject) & ~force_accept
        accept_step = ~reject_step

        # Track consecutive LTE rejects (reset on acceptance, increment on LTE reject)
        new_consecutive_rejects = jnp.where(
            accept_step,
            jnp.array(0, dtype=jnp.int32),
            jnp.where(lte_reject, state.consecutive_rejects + 1, state.consecutive_rejects),
        )

        # Debug per-step logging (for VACASK comparison)
        if config.debug_steps:

            def _debug_step_callback(
                step,
                t_next_val,
                dt_val,
                nr_iters,
                residual,
                can_pred,
                lte_norm_val,
                lte_rej,
                nr_rej,
                accept,
            ):
                t_ps = float(t_next_val) * 1e12
                dt_ps = float(dt_val) * 1e12
                float(dt_val / (dt_val / max(float(lte_norm_val), 1e-30)))  # Approximate
                if not can_pred:
                    lte_status = "Cannot estimate"
                elif lte_rej:
                    lte_status = f"LTE/tol={float(lte_norm_val):.1f} → REJECT"
                else:
                    lte_status = f"LTE/tol={float(lte_norm_val):.2f}"
                status = "REJECT" if not accept else "accept"
                print(
                    f"Step {int(step) + 1:3d}: t={t_ps:8.3f}ps dt={dt_ps:8.4f}ps "
                    f"NR={int(nr_iters):2d} res={float(residual):.2e} {lte_status:<20} [{status}]"
                )

            jax.debug.callback(
                _debug_step_callback,
                state.step_idx,
                t_next,
                dt_cur,
                iterations,
                max_f,
                can_predict,
                lte_norm,
                lte_reject,
                nr_reject,
                accept_step,
            )

        # Compute new dt
        # During warmup (can_predict=False), keep current dt to avoid LTE-driven timestep collapse
        # Only use dt_lte once we have enough history for meaningful LTE estimates
        dt_from_lte = jnp.where(can_predict, dt_lte, dt_cur)
        new_dt = jnp.where(nr_failed, jnp.maximum(dt_cur / 2, config.min_dt), dt_from_lte)
        # Use state.max_dt (dynamic) instead of config.max_dt (static) for hmax
        new_dt = jnp.clip(new_dt, config.min_dt, state.max_dt)
        new_dt = jnp.minimum(new_dt, state.t_stop - t_next)

        # Update state
        # When NR fails at min_dt, we accept the step (advance time) but keep the previous
        # solution to prevent state corruption from bad NR results
        use_new_solution = accept_step & ~nr_failed_at_min_dt
        new_t = jnp.where(accept_step, t_next, t)
        new_X = jnp.where(use_new_solution, X_new, X)
        new_Q_prev = jnp.where(use_new_solution, Q, Q_prev)
        new_dQdt_prev = jnp.where(use_new_solution, dQdt_out, dQdt_prev)
        new_Q_prev2 = jnp.where(use_new_solution, Q_prev, Q_prev2)

        # Update history - only when we're actually using a new valid solution
        # When NR fails at min_dt, don't corrupt the predictor history
        new_V_history = jnp.where(
            use_new_solution, jnp.roll(state.V_history, 1, axis=0).at[0].set(V_new), state.V_history
        )
        new_dt_history = jnp.where(
            use_new_solution, jnp.roll(state.dt_history, 1).at[0].set(dt_cur), state.dt_history
        )
        new_history_count = jnp.where(
            use_new_solution, jnp.minimum(state.history_count + 1, max_history), state.history_count
        )
        new_warmup_count = jnp.where(use_new_solution, state.warmup_count + 1, state.warmup_count)

        # Update historic max voltage per node (for VACASK-compatible LTE tolerance)
        # This implements relrefAlllocal: tolerance based on max |V| seen across all time
        # Only update when we have a valid solution
        new_V_max_historic = jnp.where(
            use_new_solution,
            jnp.maximum(state.V_max_historic, jnp.abs(V_new)),
            state.V_max_historic,
        )

        # Update outputs
        # Compute the voltage to record - use new_X which is the actual solution we're using
        # (either X_new if converged, or previous X if NR failed at min_dt)
        V_to_record = new_X[:n_external]
        new_times_out = jnp.where(
            accept_step, state.times_out.at[state.step_idx].set(t_next), state.times_out
        )
        new_V_out = jnp.where(
            accept_step, state.V_out.at[state.step_idx].set(V_to_record), state.V_out
        )
        # For currents, use zero if NR failed at min_dt (current from bad solution is unreliable)
        I_to_record = jnp.where(
            nr_failed_at_min_dt,
            jnp.zeros(n_vsources, dtype=dtype) if n_vsources > 0 else jnp.zeros(1, dtype=dtype),
            I_vsource[:n_vsources] if n_vsources > 0 else jnp.zeros(1, dtype=dtype),
        )
        new_I_out = jnp.where(
            accept_step, state.I_out.at[state.step_idx].set(I_to_record), state.I_out
        )
        new_step_idx = jnp.where(accept_step, state.step_idx + 1, state.step_idx)

        # Statistics
        new_rejected = state.rejected_steps + jnp.where(reject_step, 1, 0)
        new_min_dt = jnp.where(
            accept_step, jnp.minimum(state.min_dt_used, dt_cur), state.min_dt_used
        )
        new_max_dt = jnp.where(
            accept_step, jnp.maximum(state.max_dt_used, dt_cur), state.max_dt_used
        )

        # Progress reporting (every progress_interval steps)
        # NOTE: Use static Python if check to avoid tracing the callback when disabled.
        # jax.lax.cond would still trace the callback, preventing XLA cache.
        if progress_interval > 0:
            should_report = new_step_idx % progress_interval == 0
            jax.lax.cond(
                should_report,
                lambda: jax.debug.callback(
                    _progress_callback, new_step_idx, new_t, state.t_stop, new_dt, new_rejected
                ),
                lambda: None,
            )

        # Update limit_state: use new state when we accept the step, keep old state on rejection
        new_limit_state = (
            jnp.where(use_new_solution, limit_state_out, state.limit_state)
            if state.limit_state.size > 0
            else state.limit_state
        )

        return FullMNAState(
            t=new_t,
            dt=new_dt,
            X=new_X,  # pyright: ignore[reportArgumentType] - lax.cond typing
            Q_prev=new_Q_prev,  # pyright: ignore[reportArgumentType] - lax.cond typing
            dQdt_prev=new_dQdt_prev,  # pyright: ignore[reportArgumentType] - lax.cond typing
            Q_prev2=new_Q_prev2,
            V_history=new_V_history,
            dt_history=new_dt_history,
            history_count=new_history_count,
            times_out=new_times_out,
            V_out=new_V_out,
            I_out=new_I_out,
            step_idx=new_step_idx,
            warmup_count=new_warmup_count,
            t_stop=state.t_stop,
            max_dt=state.max_dt,
            total_nr_iters=new_total_nr_iters,
            rejected_steps=new_rejected,
            min_dt_used=new_min_dt,
            max_dt_used=new_max_dt,
            consecutive_rejects=new_consecutive_rejects,
            V_max_historic=new_V_max_historic,
            limit_state=new_limit_state,
        )

    return cond_fn, body_fn
