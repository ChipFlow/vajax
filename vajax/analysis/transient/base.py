"""Base class for transient analysis strategies.

This module defines the TransientStrategy interface and shared setup logic
for running transient analysis with OpenVAF-compiled devices.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from vajax._logging import logger

if TYPE_CHECKING:
    from vajax.analysis.engine import CircuitEngine
    from vajax.analysis.mna import MNABranchData


@dataclass
class TransientSetup:
    """Cached transient simulation setup data.

    Contains all the precomputed data needed to run transient analysis,
    independent of the specific strategy (Python loop vs lax.scan).

    Attributes:
        n_total: Total number of nodes (external + internal)
        n_unknowns: Number of node voltage unknowns (n_total - 1, excluding ground)
        n_external: Number of external (user-visible) nodes
        device_internal_nodes: Mapping of device names to internal node indices
        source_fn: Function that evaluates time-varying sources at time t
        source_device_data: Pre-computed COO data for source devices
        openvaf_by_type: Devices grouped by OpenVAF model type
        vmapped_fns: Vmapped evaluation functions for each model type
        static_inputs_cache: Cached static inputs for each model type

        # Full MNA branch current fields
        n_branches: Number of branch current unknowns (= number of vsources)
        n_augmented: Total augmented system size (n_unknowns + n_branches)
        use_full_mna: Whether to use full MNA (True) or high-G approximation (False)
        branch_data: MNABranchData with vsource branch current info
        branch_node_p: JAX array of positive terminal indices for vsources
        branch_node_n: JAX array of negative terminal indices for vsources
    """

    n_total: int
    n_unknowns: int
    n_external: int
    device_internal_nodes: Dict[str, Dict[str, int]]
    source_fn: Callable[[float], Dict[str, float]]
    source_device_data: Dict[str, Any]
    openvaf_by_type: Dict[str, List[Dict]]
    vmapped_fns: Dict[str, Callable]
    static_inputs_cache: Dict[str, Tuple]

    # Full MNA branch current fields (optional, default to high-G mode)
    n_branches: int = 0
    n_augmented: int = 0
    use_full_mna: bool = False
    branch_data: Optional["MNABranchData"] = None
    branch_node_p: Optional[jax.Array] = None
    branch_node_n: Optional[jax.Array] = None

    # Initial condition mode: 'op' (compute DC) or 'uic' (use initial conditions)
    icmode: str = "op"

    def __post_init__(self):
        """Compute derived fields after initialization."""
        if self.n_augmented == 0:
            self.n_augmented = self.n_unknowns + self.n_branches


class TransientStrategy(ABC):
    """Abstract base class for transient analysis strategies.

    The primary implementation is FullMNAStrategy which provides:
    - Full Modified Nodal Analysis with explicit branch currents
    - Adaptive timestep control based on local truncation error
    - JIT-compiled simulation loop using lax.while_loop
    """

    def __init__(self, runner: "CircuitEngine", use_sparse: bool = False, backend: str = "cpu"):
        """Initialize the strategy.

        Args:
            runner: CircuitEngine instance with parsed circuit
            use_sparse: If True, use sparse solver; if False, use dense solver
            backend: 'cpu' or 'gpu' for device evaluation
        """
        self.runner = runner
        self.use_sparse = use_sparse
        self.backend = backend
        self.use_dense = not use_sparse

        # Cached data (lazy initialized)
        self._setup: Optional[TransientSetup] = None
        self._nr_solve: Optional[Callable] = None
        self._setup_key: Optional[str] = None

    @property
    def name(self) -> str:
        """Human-readable strategy name for logging."""
        return self.__class__.__name__

    def _get_setup_key(self) -> str:
        """Generate cache key for setup data."""
        return f"{self.runner.num_nodes}_{len(self.runner.devices)}_{self.use_dense}_{self.backend}"

    def ensure_setup(self) -> TransientSetup:
        """Ensure transient setup is initialized, using cache if available.

        Returns cached setup if available and valid, otherwise builds new setup.
        This method is idempotent and can be called multiple times.

        Returns:
            TransientSetup with all precomputed data needed for simulation
        """
        setup_key = self._get_setup_key()

        # Get icmode from analysis params (default 'op' = compute DC)
        icmode = self.runner.analysis_params.get("icmode", "op")

        # Check runner's cache first (shared across strategies)
        if (
            self.runner._transient_setup_cache is not None
            and self.runner._transient_setup_key == setup_key
        ):
            cache = self.runner._transient_setup_cache
            self._setup = TransientSetup(
                n_total=cache["n_total"],
                n_unknowns=cache["n_unknowns"],
                n_external=self.runner.num_nodes,
                device_internal_nodes=cache["device_internal_nodes"],
                source_fn=cache["source_fn"],
                source_device_data=cache["source_device_data"],
                openvaf_by_type=cache["openvaf_by_type"],
                vmapped_fns=cache["vmapped_fns"],
                static_inputs_cache=cache["static_inputs_cache"],
                icmode=icmode,
            )
            logger.debug(f"{self.name}: Reusing cached setup")
            return self._setup

        # Build new setup via runner's setup method
        logger.info(f"{self.name}: Building transient setup...")
        cache = self.runner._build_transient_setup(backend=self.backend, use_dense=self.use_dense)
        self._setup = TransientSetup(
            n_total=cache["n_total"],
            n_unknowns=cache["n_unknowns"],
            n_external=self.runner.num_nodes,
            device_internal_nodes=cache["device_internal_nodes"],
            source_fn=cache["source_fn"],
            source_device_data=cache["source_device_data"],
            openvaf_by_type=cache["openvaf_by_type"],
            vmapped_fns=cache["vmapped_fns"],
            static_inputs_cache=cache["static_inputs_cache"],
            icmode=icmode,
        )
        self._setup_key = setup_key
        return self._setup

    def ensure_solver(self) -> Callable:
        """Ensure NR solver is initialized.

        This is a hook for strategies to implement solver initialization.
        The default implementation raises NotImplementedError since each
        strategy (e.g., FullMNAStrategy) builds its own solver with the
        appropriate formulation.

        Returns:
            JIT-compiled Newton-Raphson solver function

        Raises:
            NotImplementedError: Subclasses must implement solver creation
        """
        raise NotImplementedError(
            f"{self.name} must implement ensure_solver() or use a solver-specific method"
        )

    @abstractmethod
    def run(
        self, t_stop: float, dt: float, max_steps: int = 10000
    ) -> Tuple[jax.Array, Dict[int, jax.Array], Dict]:
        """Run transient analysis.

        Args:
            t_stop: Simulation stop time in seconds
            dt: Time step in seconds
            max_steps: Maximum number of time steps (for limiting long simulations)

        Returns:
            Tuple of (times, voltages, stats) where:
            - times: JAX array of time points
            - voltages: Dict mapping node index to voltage array
            - stats: Dict with convergence info (total_timesteps, wall_time, etc.)
        """
        pass

    def _build_source_arrays(self, source_values: Dict) -> Tuple[jax.Array, jax.Array]:
        """Convert source_values dict to JAX arrays for solver input.

        Args:
            source_values: Dict mapping source name to value

        Returns:
            Tuple of (vsource_vals, isource_vals) as JAX arrays
        """
        setup = self._setup
        assert setup is not None, "Setup not initialized - call ensure_setup() first"
        source_device_data = setup.source_device_data

        if "vsource" in source_device_data:
            d = source_device_data["vsource"]
            vsource_vals = jnp.array(
                [source_values.get(name, float(dc)) for name, dc in zip(d["names"], d["dc"])]
            )
        else:
            vsource_vals = jnp.array([])

        if "isource" in source_device_data:
            d = source_device_data["isource"]
            isource_vals = jnp.array(
                [source_values.get(name, float(dc)) for name, dc in zip(d["names"], d["dc"])]
            )
        else:
            isource_vals = jnp.array([])

        return vsource_vals, isource_vals

    def _compute_num_timesteps(self, t_stop: float, dt: float) -> int:
        """Compute number of timesteps for given parameters.

        Uses round() to avoid floating-point comparison issues.
        Includes both t=0 and t=t_stop.

        Args:
            t_stop: Simulation stop time
            dt: Time step

        Returns:
            Number of timesteps (including initial and final points)
        """
        return int(round(t_stop / dt)) + 1
