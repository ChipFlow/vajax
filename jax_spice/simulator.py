"""JAX-SPICE Circuit Simulator API.

Simple, unified interface for running circuit simulations.

Example:
    from jax_spice import Simulator

    sim = Simulator("circuit.sim").parse()

    # Warmup JIT compilation (required for accurate benchmarks)
    sim.warmup(t_stop=1e-9, dt=1e-12)

    # Run simulation
    result = sim.transient(t_stop=1e-9, dt=1e-12)
    print(f"V(1) at end: {result.voltage(1)[-1]}")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from jax import Array


@dataclass
class TransientResult:
    """Result of a transient simulation.

    Attributes:
        times: Array of time points (shape: num_steps)
        voltages: Dict mapping node index to voltage array (shape: num_steps)
        stats: Dict with convergence metrics (convergence_rate, wall_time, etc.)
    """

    times: Array
    voltages: Dict[int, Array]
    stats: Dict[str, Any]

    @property
    def num_steps(self) -> int:
        """Number of timesteps in the simulation."""
        return len(self.times)

    def voltage(self, node: Union[int, str]) -> Array:
        """Get voltage waveform at a specific node.

        Args:
            node: Node index (int) or name (str)

        Returns:
            Voltage array over time
        """
        if isinstance(node, str):
            raise ValueError(
                f"Node name lookup not yet supported. Use node index. "
                f"Available: {list(self.voltages.keys())}"
            )
        return self.voltages[node]


class Simulator:
    """JAX-SPICE circuit simulator.

    Provides a simple interface for parsing and simulating circuits.
    Uses CircuitEngine internally for the actual simulation.

    Example:
        sim = Simulator("circuit.sim")
        sim.parse()

        # Warmup (recommended for large circuits / benchmarks)
        sim.warmup(t_stop=1e-9, dt=1e-12)

        # Run simulation
        result = sim.transient(t_stop=1e-9, dt=1e-12)
        print(f"V(1) = {result.voltage(1)[-1]:.6f}")

    Attributes:
        circuit_path: Path to the circuit .sim file
        num_nodes: Number of circuit nodes (after parsing)
        node_names: Mapping of node names to indices
    """

    def __init__(self, circuit_path: Union[Path, str]):
        """Initialize simulator with a circuit file path.

        Args:
            circuit_path: Path to VACASK-format .sim file
        """
        self._circuit_path = Path(circuit_path)
        self._runner = None
        self._warmed_up = False
        self._warmup_config: Optional[tuple] = None

    def parse(self) -> "Simulator":
        """Parse the circuit file.

        Returns:
            Self for method chaining
        """
        from jax_spice.analysis.engine import CircuitEngine

        self._runner = CircuitEngine(self._circuit_path)
        self._runner.parse()
        return self

    @property
    def circuit_path(self) -> Path:
        """Path to the circuit file."""
        return self._circuit_path

    @property
    def num_nodes(self) -> int:
        """Number of circuit nodes (excluding ground)."""
        self._check_parsed()
        return self._runner.num_nodes

    @property
    def node_names(self) -> Dict[str, int]:
        """Mapping of node names to indices."""
        self._check_parsed()
        return self._runner.node_names

    @property
    def analysis_params(self) -> Dict[str, Any]:
        """Analysis parameters from the circuit file (dt, stop time, etc.)."""
        self._check_parsed()
        return self._runner.analysis_params

    @property
    def devices(self) -> list:
        """List of parsed devices with model info.

        Each device is a dict with keys like 'name', 'model', 'ports', 'params'.
        Useful for inspecting circuit structure and device types.
        """
        self._check_parsed()
        return self._runner.devices

    @property
    def is_warmed_up(self) -> bool:
        """Whether JIT warmup has been performed."""
        return self._warmed_up

    def warmup(
        self,
        t_stop: float,
        dt: float,
        *,
        use_sparse: Optional[bool] = None,
        use_scan: bool = True,
    ) -> "Simulator":
        """Warmup JIT compilation.

        Call this once before timed benchmark runs to ensure JIT compilation
        doesn't affect timing measurements. For large circuits (>1000 nodes),
        this can take 10-30 seconds but makes subsequent runs much faster.

        Args:
            t_stop: Stop time in seconds
            dt: Timestep in seconds
            use_sparse: Use sparse solver (auto-detect if None)
            use_scan: Use lax.scan for better performance (default True)

        Returns:
            Self for method chaining
        """
        self._check_parsed()
        self._runner.run_transient(
            t_stop=t_stop,
            dt=dt,
            max_steps=int(t_stop / dt) + 10,
            use_sparse=use_sparse,
            use_while_loop=use_scan,
        )
        self._warmed_up = True
        self._warmup_config = (t_stop, dt, use_sparse, use_scan)
        return self

    def transient(
        self,
        t_stop: float,
        dt: float,
        *,
        use_sparse: Optional[bool] = None,
        use_scan: bool = True,
        profile_config: Optional[Any] = None,
    ) -> TransientResult:
        """Run transient simulation.

        For accurate benchmark timing, call warmup() first to pre-compile
        the JIT code.

        Args:
            t_stop: Stop time in seconds
            dt: Timestep in seconds
            use_sparse: Use sparse solver (auto-detect based on circuit size if None)
            use_scan: Use lax.scan for better performance (default True)
            profile_config: Optional ProfileConfig for JAX/CUDA profiling

        Returns:
            TransientResult with times, voltages, and stats
        """
        self._check_parsed()

        times, voltages, stats = self._runner.run_transient(
            t_stop=t_stop,
            dt=dt,
            max_steps=int(t_stop / dt) + 10,
            use_sparse=use_sparse,
            use_while_loop=use_scan,
            profile_config=profile_config,
        )

        return TransientResult(times=times, voltages=voltages, stats=stats)

    def _check_parsed(self) -> None:
        """Raise error if circuit hasn't been parsed yet."""
        if self._runner is None:
            raise RuntimeError(
                "Circuit not parsed. Call parse() first: "
                "Simulator(path).parse().transient(...)"
            )

    @property
    def _internal_runner(self):
        """Access to internal runner for advanced use (tests, debugging).

        This provides access to internal methods like _setup_internal_nodes()
        for node collapse testing. Not part of public API.
        """
        self._check_parsed()
        return self._runner
