"""Base class for transient analysis strategies.

This module defines the TransientStrategy interface and shared setup logic
for running transient analysis with OpenVAF-compiled devices.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Optional

import jax
import jax.numpy as jnp

from jax_spice.logging import logger

if TYPE_CHECKING:
    from jax_spice.benchmarks.runner import VACASKBenchmarkRunner


@dataclass
class TransientSetup:
    """Cached transient simulation setup data.

    Contains all the precomputed data needed to run transient analysis,
    independent of the specific strategy (Python loop vs lax.scan).

    Attributes:
        n_total: Total number of nodes (external + internal)
        n_unknowns: Number of unknowns (n_total - 1, excluding ground)
        n_external: Number of external (user-visible) nodes
        device_internal_nodes: Mapping of device names to internal node indices
        source_fn: Function that evaluates time-varying sources at time t
        source_device_data: Pre-computed COO data for source devices
        openvaf_by_type: Devices grouped by OpenVAF model type
        vmapped_fns: Vmapped evaluation functions for each model type
        static_inputs_cache: Cached static inputs for each model type
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


class TransientStrategy(ABC):
    """Abstract base class for transient analysis strategies.

    Subclasses implement different approaches to the timestep loop:
    - PythonLoopStrategy: Traditional Python for-loop with JIT-compiled NR solver
    - ScanStrategy: Fully JIT-compiled using lax.scan

    All strategies share:
    - The same circuit setup process
    - The same Newton-Raphson solver creation
    - The same return value format

    The key difference is how the timestep loop is implemented:
    - Python loop: More debugging info, ~0.5ms/step
    - lax.scan: Less debugging info, ~0.1ms/step (5x faster)
    """

    def __init__(self, runner: 'VACASKBenchmarkRunner',
                 use_sparse: bool = False, backend: str = "cpu"):
        """Initialize the strategy.

        Args:
            runner: VACASKBenchmarkRunner instance with parsed circuit
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

        # Check runner's cache first (shared across strategies)
        if (self.runner._transient_setup_cache is not None and
            self.runner._transient_setup_key == setup_key):
            cache = self.runner._transient_setup_cache
            self._setup = TransientSetup(
                n_total=cache['n_total'],
                n_unknowns=cache['n_unknowns'],
                n_external=self.runner.num_nodes,
                device_internal_nodes=cache['device_internal_nodes'],
                source_fn=cache['source_fn'],
                source_device_data=cache['source_device_data'],
                openvaf_by_type=cache['openvaf_by_type'],
                vmapped_fns=cache['vmapped_fns'],
                static_inputs_cache=cache['static_inputs_cache'],
            )
            logger.debug(f"{self.name}: Reusing cached setup")
            return self._setup

        # Build new setup via runner's hybrid method (with 0 steps)
        logger.info(f"{self.name}: Building transient setup...")
        self.runner._run_transient_hybrid(t_stop=0, dt=1e-12, backend=self.backend, use_dense=self.use_dense)

        # Extract from runner's cache
        cache = self.runner._transient_setup_cache
        self._setup = TransientSetup(
            n_total=cache['n_total'],
            n_unknowns=cache['n_unknowns'],
            n_external=self.runner.num_nodes,
            device_internal_nodes=cache['device_internal_nodes'],
            source_fn=cache['source_fn'],
            source_device_data=cache['source_device_data'],
            openvaf_by_type=cache['openvaf_by_type'],
            vmapped_fns=cache['vmapped_fns'],
            static_inputs_cache=cache['static_inputs_cache'],
        )
        self._setup_key = setup_key
        return self._setup

    def ensure_solver(self) -> Callable:
        """Ensure NR solver is initialized, using cache if available.

        Returns cached solver if available, otherwise builds and caches new solver.

        Returns:
            JIT-compiled Newton-Raphson solver function
        """
        setup = self.ensure_setup()

        n_vsources = len(setup.source_device_data.get('vsource', {}).get('names', []))
        n_isources = len(setup.source_device_data.get('isource', {}).get('names', []))
        n_nodes = setup.n_unknowns + 1

        cache_key = (n_nodes, n_vsources, n_isources, self.use_dense)

        # Check runner's cached solver
        if hasattr(self.runner, '_cached_nr_solve') and self.runner._cached_solver_key == cache_key:
            self._nr_solve = self.runner._cached_nr_solve
            logger.debug(f"{self.name}: Reusing cached NR solver")
            return self._nr_solve

        # Build solver via runner (will cache it)
        logger.info(f"{self.name}: Building NR solver...")
        self.runner._run_transient_hybrid(t_stop=1e-12, dt=1e-12, backend=self.backend, use_dense=self.use_dense)
        self._nr_solve = self.runner._cached_nr_solve
        return self._nr_solve

    @abstractmethod
    def run(self, t_stop: float, dt: float,
            max_steps: int = 10000) -> Tuple[jax.Array, Dict[int, jax.Array], Dict]:
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
        source_device_data = setup.source_device_data

        if 'vsource' in source_device_data:
            d = source_device_data['vsource']
            vsource_vals = jnp.array([
                source_values.get(name, float(dc))
                for name, dc in zip(d['names'], d['dc'])
            ])
        else:
            vsource_vals = jnp.array([])

        if 'isource' in source_device_data:
            d = source_device_data['isource']
            isource_vals = jnp.array([
                source_values.get(name, float(dc))
                for name, dc in zip(d['names'], d['dc'])
            ])
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
