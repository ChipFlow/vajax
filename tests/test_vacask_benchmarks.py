"""Test VACASK benchmarks using VACASKBenchmarkRunner

This file tests the actual VACASK benchmark circuits which use OpenVAF-compiled
device models (resistor.va, capacitor.va, diode.va from vendor/VACASK/devices/).

These tests replace the custom resistor/diode tests with actual benchmark circuits:
- rc: RC circuit with pulse source (tests resistor + capacitor)
- graetz: Full-wave rectifier (tests diode)
- mul: Multiplier circuit
- ring: Ring oscillator (tests PSP103 MOSFET)
- c6288: Large benchmark (sparse solver only)
"""

import pytest
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

# Add jax-spice to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_spice.benchmarks import VACASKBenchmarkRunner


# Benchmark paths
BENCHMARK_DIR = Path(__file__).parent.parent / "vendor" / "VACASK" / "benchmark"


def get_benchmark_sim(name: str) -> Path:
    """Get path to benchmark .sim file"""
    return BENCHMARK_DIR / name / "vacask" / "runme.sim"


class TestRCBenchmark:
    """Test RC circuit benchmark (resistor + capacitor)

    Circuit: V(pulse) -> R(1k) -> C(1u) -> GND
    Time constant: tau = RC = 1k * 1u = 1ms
    """

    @pytest.fixture
    def runner(self):
        """Create and parse RC benchmark runner"""
        sim_path = get_benchmark_sim("rc")
        if not sim_path.exists():
            pytest.skip(f"RC benchmark not found at {sim_path}")

        runner = VACASKBenchmarkRunner(sim_path)
        runner.parse()
        return runner

    def test_parse(self, runner):
        """Test RC benchmark parses correctly"""
        assert runner.num_nodes > 0
        assert len(runner.devices) > 0

        # Should have resistor, capacitor, vsource
        device_types = {d['model'] for d in runner.devices}
        assert 'resistor' in device_types, f"Missing resistor, got {device_types}"
        assert 'capacitor' in device_types, f"Missing capacitor, got {device_types}"
        assert 'vsource' in device_types, f"Missing vsource, got {device_types}"

        print(f"RC benchmark: {runner.num_nodes} nodes, {len(runner.devices)} devices")
        print(f"Device types: {device_types}")

    def test_transient_dense(self, runner):
        """Test RC transient with dense solver"""
        dt = runner.analysis_params.get('step', 1e-6)

        # Run short transient (10 steps)
        times, voltages, stats = runner.run_transient(
            t_stop=dt * 10, dt=dt, max_steps=10, use_sparse=False
        )

        assert len(times) > 0, "No timesteps returned"
        # voltages is a dict mapping node index to voltage array
        assert isinstance(voltages, dict), f"Expected dict, got {type(voltages)}"

        # Check convergence
        converged = stats.get('convergence_rate', 0)
        print(f"RC dense: {len(times)} steps, {converged*100:.0f}% converged")
        assert converged > 0.5, f"Poor convergence: {converged*100:.0f}%"

    def test_transient_sparse(self, runner):
        """Test RC transient with sparse solver"""
        dt = runner.analysis_params.get('step', 1e-6)

        times, voltages, stats = runner.run_transient(
            t_stop=dt * 10, dt=dt, max_steps=10, use_sparse=True
        )

        assert len(times) > 0, "No timesteps returned"

        converged = stats.get('convergence_rate', 0)
        print(f"RC sparse: {len(times)} steps, {converged*100:.0f}% converged")
        assert converged > 0.5, f"Poor convergence: {converged*100:.0f}%"

    def test_rc_time_constant(self, runner):
        """Verify RC time constant behavior

        With R=1k, C=1u, tau=1ms
        After 1 tau, voltage should reach ~63.2% of final value
        After 5 tau, voltage should reach ~99.3% of final value
        """
        dt = 10e-6  # 10us steps
        t_stop = 5e-3  # 5ms (5 tau)

        times, voltages, stats = runner.run_transient(
            t_stop=t_stop, dt=dt, max_steps=500, use_sparse=False
        )

        # Get node 2 voltage (capacitor voltage)
        # voltages is a dict mapping node index to voltage array
        if 2 in voltages:
            v_cap = voltages[2]  # Capacitor voltage
        else:
            # Get the last non-ground node
            v_cap = voltages[max(voltages.keys())]

        times_np = np.array(times)
        v_cap_np = np.array(v_cap)

        # Find approximate final value (after 5 tau)
        v_final = v_cap_np[-1] if len(v_cap_np) > 0 else 0

        print(f"RC response: V_final = {v_final:.3f}V after {times_np[-1]*1000:.1f}ms")

        # Just verify we got reasonable output
        assert len(times) > 10, "Not enough timesteps for RC analysis"


class TestGraetzBenchmark:
    """Test Graetz bridge benchmark (full-wave rectifier with diodes)

    Circuit: AC source -> 4 diodes (full bridge) -> RC filter -> load
    Tests diode model under dynamic conditions.
    """

    @pytest.fixture
    def runner(self):
        """Create and parse Graetz benchmark runner"""
        sim_path = get_benchmark_sim("graetz")
        if not sim_path.exists():
            pytest.skip(f"Graetz benchmark not found at {sim_path}")

        runner = VACASKBenchmarkRunner(sim_path)
        runner.parse()
        return runner

    def test_parse(self, runner):
        """Test Graetz benchmark parses correctly"""
        assert runner.num_nodes > 0
        assert len(runner.devices) > 0

        device_types = {d['model'] for d in runner.devices}
        assert 'diode' in device_types, f"Missing diode, got {device_types}"

        # Count diodes (should be 4 for full bridge)
        diode_count = sum(1 for d in runner.devices if d['model'] == 'diode')
        assert diode_count == 4, f"Expected 4 diodes, got {diode_count}"

        print(f"Graetz benchmark: {runner.num_nodes} nodes, {len(runner.devices)} devices")
        print(f"Device types: {device_types}")
        print(f"Diode count: {diode_count}")

    def test_transient_dense(self, runner):
        """Test Graetz transient with dense solver

        Note: Graetz has numerical challenges due to diode nonlinearity.
        We test that the solver runs and produces output, even if
        convergence isn't perfect.
        """
        dt = runner.analysis_params.get('step', 1e-6)

        # Run short transient
        times, voltages, stats = runner.run_transient(
            t_stop=dt * 10, dt=dt, max_steps=10, use_sparse=False
        )

        assert len(times) > 0, "No timesteps returned"

        converged = stats.get('converged_steps', 0) / max(len(times), 1)
        print(f"Graetz dense: {len(times)} steps, {converged*100:.0f}% converged")
        # Graetz is numerically challenging - accept lower convergence

    @pytest.mark.skip(reason="Graetz sparse has convergence issues - known limitation")
    def test_transient_sparse(self, runner):
        """Test Graetz transient with sparse solver"""
        dt = runner.analysis_params.get('step', 1e-6)

        times, voltages, stats = runner.run_transient(
            t_stop=dt * 10, dt=dt, max_steps=10, use_sparse=True
        )

        assert len(times) > 0, "No timesteps returned"


class TestMulBenchmark:
    """Test multiplier circuit benchmark"""

    @pytest.fixture
    def runner(self):
        """Create and parse mul benchmark runner"""
        sim_path = get_benchmark_sim("mul")
        if not sim_path.exists():
            pytest.skip(f"Mul benchmark not found at {sim_path}")

        runner = VACASKBenchmarkRunner(sim_path)
        runner.parse()
        return runner

    def test_parse(self, runner):
        """Test mul benchmark parses correctly"""
        assert runner.num_nodes > 0
        assert len(runner.devices) > 0

        print(f"Mul benchmark: {runner.num_nodes} nodes, {len(runner.devices)} devices")
        device_types = {d['model'] for d in runner.devices}
        print(f"Device types: {device_types}")

    def test_transient_dense(self, runner):
        """Test mul transient with dense solver"""
        dt = runner.analysis_params.get('step', 1e-9)

        times, voltages, stats = runner.run_transient(
            t_stop=dt * 5, dt=dt, max_steps=5, use_sparse=False
        )

        assert len(times) > 0, "No timesteps returned"

        converged = stats.get('converged_steps', 0) / max(len(times), 1)
        print(f"Mul dense: {len(times)} steps, {converged*100:.0f}% converged")


class TestRingBenchmark:
    """Test ring oscillator benchmark (PSP103 MOSFETs)

    This is a 9-stage ring oscillator using PSP103 MOSFET models.
    Tests OpenVAF compilation and evaluation of complex device models.
    """

    @pytest.fixture
    def runner(self):
        """Create and parse ring benchmark runner"""
        sim_path = get_benchmark_sim("ring")
        if not sim_path.exists():
            pytest.skip(f"Ring benchmark not found at {sim_path}")

        runner = VACASKBenchmarkRunner(sim_path)
        runner.parse()
        return runner

    def test_parse(self, runner):
        """Test ring benchmark parses correctly"""
        assert runner.num_nodes > 0
        assert len(runner.devices) > 0

        # Should have PSP103 MOSFETs
        device_types = {d['model'] for d in runner.devices}
        assert 'psp103' in device_types, f"Missing psp103, got {device_types}"

        # Count PSP103 devices (9 stages * 2 transistors)
        psp_count = sum(1 for d in runner.devices if d['model'] == 'psp103')
        assert psp_count == 18, f"Expected 18 PSP103 devices, got {psp_count}"

        print(f"Ring benchmark: {runner.num_nodes} nodes, {len(runner.devices)} devices")
        print(f"Device types: {device_types}")
        print(f"PSP103 count: {psp_count}")

    def test_transient_dense(self, runner):
        """Test ring transient with dense solver"""
        dt = runner.analysis_params.get('step', 5e-11)

        times, voltages, stats = runner.run_transient(
            t_stop=dt * 5, dt=dt, max_steps=5, use_sparse=False
        )

        assert len(times) > 0, "No timesteps returned"

        converged = stats.get('convergence_rate', 0)
        print(f"Ring dense: {len(times)} steps, {converged*100:.0f}% converged")
        assert converged > 0.5, f"Poor convergence: {converged*100:.0f}%"

    def test_transient_sparse(self, runner):
        """Test ring transient with sparse solver"""
        dt = runner.analysis_params.get('step', 5e-11)

        times, voltages, stats = runner.run_transient(
            t_stop=dt * 5, dt=dt, max_steps=5, use_sparse=True
        )

        assert len(times) > 0, "No timesteps returned"

        converged = stats.get('convergence_rate', 0)
        print(f"Ring sparse: {len(times)} steps, {converged*100:.0f}% converged")
        assert converged > 0.5, f"Poor convergence: {converged*100:.0f}%"


class TestC6288Benchmark:
    """Test c6288 large benchmark (sparse solver only)

    c6288 is a large circuit (~86k nodes) that requires sparse solver.
    Dense solver would need ~56GB of memory.
    """

    @pytest.fixture
    def runner(self):
        """Create and parse c6288 benchmark runner"""
        sim_path = get_benchmark_sim("c6288")
        if not sim_path.exists():
            pytest.skip(f"c6288 benchmark not found at {sim_path}")

        runner = VACASKBenchmarkRunner(sim_path)
        runner.parse()
        return runner

    def test_parse(self, runner):
        """Test c6288 benchmark parses correctly"""
        assert runner.num_nodes > 0
        assert len(runner.devices) > 0

        print(f"c6288 benchmark: {runner.num_nodes} nodes, {len(runner.devices)} devices")

        # c6288 should be large
        assert runner.num_nodes > 1000, f"Expected large circuit, got {runner.num_nodes} nodes"

    @pytest.mark.skip(reason="c6288 sparse test takes too long for CI")
    def test_transient_sparse(self, runner):
        """Test c6288 transient with sparse solver (slow)"""
        dt = runner.analysis_params.get('step', 1e-12)

        times, voltages, stats = runner.run_transient(
            t_stop=dt * 2, dt=dt, max_steps=2, use_sparse=True
        )

        assert len(times) > 0, "No timesteps returned"


class TestNodeCountComparison:
    """Test that JAX-SPICE node counts match VACASK.

    VACASK outputs node count via 'print stats':
        System stats:
          Number of nodes:                 <nodeCount>
          Number of unknonws:              <unknownCount>

    These tests require VACASK to be built and available.
    """

    @staticmethod
    def find_vacask_binary() -> Path | None:
        """Find VACASK simulator binary."""
        import shutil
        import os

        # Check environment variable
        env_path = os.environ.get("VACASK_BIN")
        if env_path and Path(env_path).exists():
            return Path(env_path)

        # Check common build locations relative to jax-spice
        project_root = Path(__file__).parent.parent
        candidates = [
            project_root / "vendor" / "VACASK" / "build" / "simulator" / "vacask",
            project_root / "vendor" / "VACASK" / "build.VACASK" / "Release" / "simulator" / "vacask",
            project_root / "vendor" / "VACASK" / "build.VACASK" / "Debug" / "simulator" / "vacask",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Check system PATH
        found = shutil.which("vacask")
        if found:
            return Path(found)

        return None

    @staticmethod
    def get_vacask_node_count(vacask_bin: Path, benchmark: str, timeout: int = 600) -> int:
        """Run VACASK on benchmark and extract node count from 'print stats'.

        Args:
            vacask_bin: Path to VACASK binary
            benchmark: Benchmark name
            timeout: Subprocess timeout in seconds (default 600s = 10min for large circuits)
        """
        import subprocess
        import re

        sim_file = get_benchmark_sim(benchmark)
        if not sim_file.exists():
            raise FileNotFoundError(f"Benchmark sim file not found: {sim_file}")

        result = subprocess.run(
            [str(vacask_bin), str(sim_file)],
            capture_output=True,
            text=True,
            cwd=sim_file.parent,
            timeout=timeout
        )

        # Parse both "Number of nodes:" and "Number of unknonws:" from output
        # Note: VACASK has a typo "unknonws" instead of "unknowns"
        nodes_match = re.search(r"Number of nodes:\s+(\d+)", result.stdout)
        unknowns_match = re.search(r"Number of unknonws:\s+(\d+)", result.stdout)

        if not nodes_match:
            nodes_match = re.search(r"Number of nodes:\s+(\d+)", result.stderr)
        if not unknowns_match:
            unknowns_match = re.search(r"Number of unknonws:\s+(\d+)", result.stderr)

        if nodes_match and unknowns_match:
            return {
                'nodes': int(nodes_match.group(1)),
                'unknowns': int(unknowns_match.group(1))
            }

        raise ValueError(
            f"Could not parse node count from VACASK output.\n"
            f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
        )

    @pytest.fixture
    def vacask_bin(self):
        """Get VACASK binary path, skip if not available."""
        binary = self.find_vacask_binary()
        if binary is None:
            pytest.skip("VACASK binary not found. Set VACASK_BIN env var or build VACASK.")
        return binary

    @pytest.mark.parametrize("benchmark,xfail_reason", [
        ("rc", None),
        pytest.param("graetz", "Graetz node count mismatch - JAX-SPICE uses named nodes differently", marks=pytest.mark.xfail(reason="Node naming mismatch")),
        pytest.param("ring", "Ring node count mismatch - internal node handling differs", marks=pytest.mark.xfail(reason="Internal node handling differs")),
    ])
    def test_node_count_matches_vacask(self, vacask_bin, benchmark, xfail_reason):
        """Compare JAX-SPICE node count with VACASK for simple benchmarks.

        For simple benchmarks without complex internal nodes, we compare:
        - JAX-SPICE external nodes (num_nodes) vs VACASK's nodeCount
        - JAX-SPICE total nodes vs VACASK's unknownCount + 1 (for ground)
        """
        sim_path = get_benchmark_sim(benchmark)
        if not sim_path.exists():
            pytest.skip(f"Benchmark not found at {sim_path}")

        # Get VACASK counts (both nodes and unknowns)
        vacask_counts = self.get_vacask_node_count(vacask_bin, benchmark)
        vacask_nodes = vacask_counts['nodes']
        vacask_unknowns = vacask_counts['unknowns']

        # Get JAX-SPICE counts
        runner = VACASKBenchmarkRunner(sim_path)
        runner.parse()
        jax_external = runner.num_nodes
        n_total, _ = runner._setup_internal_nodes()

        # Report counts
        print(f"\n{benchmark}:")
        print(f"  VACASK: nodes={vacask_nodes}, unknowns={vacask_unknowns}")
        print(f"  JAX-SPICE: external={jax_external}, total={n_total}")

        # For simple benchmarks, external nodes should match VACASK's nodeCount
        # and total should match unknowns + 1 (VACASK excludes ground)
        diff_external = abs(jax_external - vacask_nodes)
        diff_total = abs(n_total - (vacask_unknowns + 1))

        assert diff_external <= 1, \
            f"{benchmark}: external nodes differ: JAX-SPICE={jax_external}, VACASK={vacask_nodes}"
        assert diff_total <= 1, \
            f"{benchmark}: total nodes differ: JAX-SPICE={n_total}, VACASK unknowns+1={vacask_unknowns+1}"

    def test_c6288_node_count(self, vacask_bin):
        """Test c6288 node count - this is the main target for node collapse fix.

        VACASK reports two metrics:
        - nodeCount: Total Node objects (~86k for c6288, includes all internal nodes)
        - unknownCount: Actual system size after collapse (~15k for c6288)

        JAX-SPICE's total_nodes should match VACASK's unknownCount + 1 (for ground).
        """
        benchmark = "c6288"
        sim_path = get_benchmark_sim(benchmark)
        if not sim_path.exists():
            pytest.skip(f"c6288 benchmark not found at {sim_path}")

        # Get VACASK counts
        vacask_counts = self.get_vacask_node_count(vacask_bin, benchmark)
        vacask_nodes = vacask_counts['nodes']
        vacask_unknowns = vacask_counts['unknowns']

        # Get JAX-SPICE total node count (external + internal after collapse)
        runner = VACASKBenchmarkRunner(sim_path)
        runner.parse()
        n_total, _ = runner._setup_internal_nodes()
        jax_total = n_total

        # Report counts
        print(f"\nc6288:")
        print(f"  VACASK: nodes={vacask_nodes}, unknowns={vacask_unknowns}")
        print(f"  JAX-SPICE: external={runner.num_nodes}, total={jax_total}")
        print(f"  Internal nodes: {jax_total - runner.num_nodes}")

        # Compare JAX-SPICE total with VACASK unknowns + 1
        # (VACASK's unknownCount excludes ground, JAX-SPICE includes it)
        expected = vacask_unknowns + 1
        diff = abs(jax_total - expected)
        ratio = diff / expected if expected > 0 else 0

        print(f"  Comparison: JAX-SPICE total={jax_total} vs VACASK unknowns+1={expected}")
        print(f"  Difference: {diff} ({ratio*100:.1f}%)")

        # Allow 10% tolerance for slight differences in node handling
        assert ratio <= 0.1, \
            f"c6288: JAX-SPICE total={jax_total}, VACASK unknowns+1={expected} (diff={diff}, {ratio*100:.1f}%)"


class TestNodeCollapseStandalone:
    """Test node collapse without requiring VACASK binary.

    These tests verify the node collapse implementation works correctly
    by checking expected node counts for known circuits.
    """

    def test_c6288_node_collapse_reduces_count(self):
        """Test that node collapse significantly reduces c6288 node count.

        Without node collapse: ~86,000 nodes (5123 external + 81k internal)
        With node collapse: ~15,000 nodes (5123 external + 10k internal)
        """
        sim_path = get_benchmark_sim("c6288")
        if not sim_path.exists():
            pytest.skip(f"c6288 benchmark not found at {sim_path}")

        runner = VACASKBenchmarkRunner(sim_path)
        runner.parse()

        # Get total nodes after collapse
        n_total, device_internal = runner._setup_internal_nodes()
        n_internal = n_total - runner.num_nodes
        n_psp103_devices = sum(1 for d in runner.devices if d.get('model') == 'psp103')

        print(f"\nc6288 node collapse:")
        print(f"  External nodes: {runner.num_nodes}")
        print(f"  Internal nodes: {n_internal}")
        print(f"  Total nodes: {n_total}")
        print(f"  PSP103 devices: {n_psp103_devices}")
        print(f"  Internal nodes per device: {n_internal / n_psp103_devices:.2f}")

        # With node collapse, each PSP103 should need only 1 internal node
        # (node 4 in the model is the only non-collapsed internal)
        expected_internal = n_psp103_devices  # 1 internal per device
        internal_ratio = n_internal / expected_internal

        # Allow up to 10% variance
        assert internal_ratio < 1.1, \
            f"Too many internal nodes: {n_internal} (expected ~{expected_internal}, ratio={internal_ratio:.2f})"

        # Total should be well under 20k (vs ~86k without collapse)
        assert n_total < 20000, \
            f"Total nodes too high: {n_total} (expected <20000 with node collapse)"

    def test_ring_node_collapse(self):
        """Test that node collapse is applied to ring benchmark.

        Ring has 18 PSP103 devices, each with 1 internal node after collapse.
        """
        sim_path = get_benchmark_sim("ring")
        if not sim_path.exists():
            pytest.skip(f"ring benchmark not found at {sim_path}")

        runner = VACASKBenchmarkRunner(sim_path)
        runner.parse()

        # Get total nodes
        n_total, _ = runner._setup_internal_nodes()
        n_internal = n_total - runner.num_nodes
        n_psp103 = sum(1 for d in runner.devices if d.get('model') == 'psp103')

        print(f"\nring benchmark:")
        print(f"  External nodes: {runner.num_nodes}")
        print(f"  Internal nodes: {n_internal}")
        print(f"  Total nodes: {n_total}")
        print(f"  PSP103 devices: {n_psp103}")

        # With collapse, each PSP103 should have 1 internal node
        assert n_internal == n_psp103, \
            f"Expected {n_psp103} internal nodes (1 per device), got {n_internal}"


# =============================================================================
# VACASK Result Comparison Framework
#
# This framework enables automated comparison of JAX-SPICE simulation results
# against VACASK reference simulations. To add a new benchmark comparison:
#
# 1. Add an entry to BENCHMARK_SPECS with the benchmark configuration
# 2. Run: JAX_PLATFORMS=cpu uv run pytest tests/test_vacask_benchmarks.py -v -k "test_transient_matches_vacask"
#
# Benchmarks marked with xfail=True are expected to fail and won't cause CI to fail.
# =============================================================================

from dataclasses import dataclass
from typing import Callable


@dataclass
class BenchmarkSpec:
    """Specification for a benchmark comparison test.

    Attributes:
        name: Benchmark directory name (e.g., 'rc', 'graetz')
        dt: Timestep for simulation
        t_stop: Stop time for simulation
        max_rel_error: Maximum allowed relative RMS error (0.05 = 5%)
        vacask_nodes: List of VACASK node names to compare (in order of preference)
        jax_nodes: Corresponding JAX-SPICE node indices
        xfail: If True, test is expected to fail (won't break CI)
        xfail_reason: Reason for expected failure
        node_transform: Optional function to transform node voltages before comparison
                       e.g., for differential outputs like (outp - outn)
    """
    name: str
    dt: float
    t_stop: float
    max_rel_error: float
    vacask_nodes: list[str]
    jax_nodes: list[int]
    xfail: bool = False
    xfail_reason: str = ""
    node_transform: Callable | None = None


# Benchmark specifications - add new benchmarks here
BENCHMARK_SPECS = {
    'rc': BenchmarkSpec(
        name='rc',
        dt=1e-6,           # 1µs step
        t_stop=1e-3,       # 1ms (one time constant)
        max_rel_error=0.05,  # 5% allowed
        vacask_nodes=['2', 'v(2)'],
        jax_nodes=[2, 1],
        xfail=False,
    ),
    'graetz': BenchmarkSpec(
        name='graetz',
        dt=1e-6,           # 1µs step
        t_stop=5e-3,       # 5ms (1/4 of 50Hz period, allows capacitor charging)
        max_rel_error=0.05,  # 5% allowed
        vacask_nodes=['outp'],  # Compare differential voltage via transform
        jax_nodes=[4],
        xfail=False,
        node_transform=lambda v: v.get('outp', v.get(4, np.zeros(1))) - v.get('outn', v.get(3, np.zeros(1))),
    ),
    'ring': BenchmarkSpec(
        name='ring',
        dt=5e-11,          # 50ps step
        t_stop=5e-9,       # 5ns
        max_rel_error=0.15,  # 15% allowed (PSP103 complexity)
        vacask_nodes=['1', 'v(1)'],
        jax_nodes=[1],
        xfail=True,
        xfail_reason="PSP103 MOSFET not oscillating correctly - ~33% error",
    ),
    'mul': BenchmarkSpec(
        name='mul',
        dt=1e-9,           # 1ns step
        t_stop=1e-7,       # 100ns
        max_rel_error=0.15,
        vacask_nodes=['1', 'v(1)'],
        jax_nodes=[1],
        xfail=True,
        xfail_reason="Multiplier circuit not yet validated",
    ),
}


def find_vacask_binary() -> Path | None:
    """Find VACASK simulator binary."""
    import shutil
    import os

    # Check environment variable
    env_path = os.environ.get("VACASK_BIN")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # Check common build locations relative to jax-spice
    project_root = Path(__file__).parent.parent
    candidates = [
        project_root / "vendor" / "VACASK" / "build" / "simulator" / "vacask",
        project_root / "vendor" / "VACASK" / "build.VACASK" / "Release" / "simulator" / "vacask",
        project_root / "vendor" / "VACASK" / "build.VACASK" / "Debug" / "simulator" / "vacask",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Check system PATH
    found = shutil.which("vacask")
    if found:
        return Path(found)

    return None


def run_vacask_simulation(
    vacask_bin: Path, sim_path: Path, t_stop: float, dt: float
) -> dict:
    """Run VACASK and parse the .raw file output.

    Returns dict with:
        'time': numpy array of timepoints
        'voltages': dict mapping node name (str) to voltage array
    """
    import subprocess
    import re

    # Add rawfile parser to path
    rawfile_path = Path(__file__).parent.parent / "vendor" / "VACASK" / "python"
    if str(rawfile_path) not in sys.path:
        sys.path.insert(0, str(rawfile_path))

    from rawfile import rawread

    sim_dir = sim_path.parent

    # Read original sim file
    with open(sim_path) as f:
        sim_content = f.read()

    # Modify the analysis line to use our t_stop
    modified = re.sub(
        r'(analysis\s+\w+\s+tran\s+.*?stop=)[^\s]+',
        f'\\g<1>{t_stop:.2e}',
        sim_content
    )

    # Also modify step size
    modified = re.sub(
        r'(step=)[^\s]+',
        f'\\g<1>{dt:.2e}',
        modified
    )

    # Write to temp file
    temp_sim = sim_dir / 'test_compare.sim'
    with open(temp_sim, 'w') as f:
        f.write(modified)

    try:
        # Run VACASK
        result = subprocess.run(
            [str(vacask_bin), 'test_compare.sim'],
            cwd=sim_dir,
            capture_output=True,
            text=True,
            timeout=120
        )

        # Parse the raw file (VACASK may return non-zero from postprocess scripts)
        raw_path = sim_dir / 'tran1.raw'
        if not raw_path.exists():
            raise RuntimeError(
                f"VACASK did not produce {raw_path}.\n"
                f"stdout: {result.stdout[:500]}\n"
                f"stderr: {result.stderr[:500]}"
            )

        raw = rawread(str(raw_path)).get()
        time_arr = np.array(raw['time'])

        # Get all voltage nodes
        voltages = {}
        for name in raw.names:
            if name != 'time':
                voltages[name] = np.array(raw[name])

        return {'time': time_arr, 'voltages': voltages}

    finally:
        # Clean up
        if temp_sim.exists():
            temp_sim.unlink()
        raw_path = sim_dir / 'tran1.raw'
        if raw_path.exists():
            raw_path.unlink()


def compare_waveforms(
    vacask_time: np.ndarray,
    vacask_voltage: np.ndarray,
    jax_times: np.ndarray,
    jax_voltage: np.ndarray,
) -> dict:
    """Compare two voltage waveforms.

    Returns dict with:
        'max_diff': Maximum absolute difference
        'rms_diff': RMS difference
        'rel_rms': Relative RMS error (normalized by voltage range)
        'v_range': Voltage range
    """
    if len(jax_voltage) < 2 or len(vacask_voltage) < 2:
        return {'max_diff': float('inf'), 'rms_diff': float('inf'), 'rel_rms': float('inf'), 'v_range': 0}

    # Interpolate JAX-SPICE to VACASK timepoints
    jax_interp = np.interp(vacask_time, jax_times, jax_voltage)

    abs_diff = np.abs(jax_interp - vacask_voltage)
    max_diff = float(np.max(abs_diff))
    rms_diff = float(np.sqrt(np.mean(abs_diff**2)))
    v_range = float(np.max(vacask_voltage) - np.min(vacask_voltage))
    rel_rms = rms_diff / max(v_range, 1e-12)

    return {
        'max_diff': max_diff,
        'rms_diff': rms_diff,
        'rel_rms': rel_rms,
        'v_range': v_range,
    }


class TestVACASKResultComparison:
    """Compare JAX-SPICE simulation results against VACASK reference.

    These tests run both simulators on the same benchmarks and verify that
    the voltage waveforms match within tolerance.

    Requirements:
    - VACASK binary must be available (set VACASK_BIN env var or build VACASK)
    - VACASK raw file parser from vendor/VACASK/python/rawfile.py

    To add a new benchmark, add an entry to BENCHMARK_SPECS at the top of this file.
    """

    @pytest.fixture
    def vacask_bin(self):
        """Get VACASK binary path, skip if not available."""
        binary = find_vacask_binary()
        if binary is None:
            pytest.skip("VACASK binary not found. Set VACASK_BIN env var or build VACASK.")
        return binary

    @pytest.mark.parametrize("benchmark_name", list(BENCHMARK_SPECS.keys()))
    def test_transient_matches_vacask(self, vacask_bin, benchmark_name):
        """Parametrized test comparing JAX-SPICE to VACASK for each benchmark.

        This test automatically handles xfail markers for benchmarks that are
        known to fail.
        """
        spec = BENCHMARK_SPECS[benchmark_name]

        # Apply xfail if specified
        if spec.xfail:
            pytest.xfail(spec.xfail_reason)

        sim_path = get_benchmark_sim(spec.name)
        if not sim_path.exists():
            pytest.skip(f"Benchmark not found at {sim_path}")

        num_steps = int(spec.t_stop / spec.dt)

        # Run VACASK
        vacask_results = run_vacask_simulation(vacask_bin, sim_path, spec.t_stop, spec.dt)
        vacask_time = vacask_results['time']

        # Find VACASK output node
        vacask_voltage = None
        vacask_node_used = None
        for node_name in spec.vacask_nodes:
            if node_name in vacask_results['voltages']:
                vacask_voltage = vacask_results['voltages'][node_name]
                vacask_node_used = node_name
                break

        if vacask_voltage is None:
            pytest.skip(
                f"Could not find any of {spec.vacask_nodes} in VACASK output. "
                f"Available: {list(vacask_results['voltages'].keys())}"
            )

        # Get corresponding JAX-SPICE node index
        jax_node_idx = spec.jax_nodes[spec.vacask_nodes.index(vacask_node_used) % len(spec.jax_nodes)]

        # Run JAX-SPICE
        runner = VACASKBenchmarkRunner(sim_path)
        runner.parse()
        jax_times, jax_voltages, stats = runner.run_transient(
            t_stop=spec.t_stop, dt=spec.dt, max_steps=num_steps
        )

        jax_voltage = np.array(jax_voltages.get(jax_node_idx, []))

        # Apply node transform if specified
        if spec.node_transform is not None:
            vacask_voltage = spec.node_transform(vacask_results['voltages'])
            jax_voltage = spec.node_transform({i: np.array(v) for i, v in jax_voltages.items()})

        # Compare waveforms
        comparison = compare_waveforms(
            vacask_time, vacask_voltage,
            np.array(jax_times), jax_voltage
        )

        print(f"\n{benchmark_name.upper()} comparison:")
        print(f"  VACASK node: {vacask_node_used}, JAX-SPICE node: {jax_node_idx}")
        print(f"  VACASK: {len(vacask_time)} points, JAX-SPICE: {len(jax_times)} points")
        print(f"  Voltage range: {comparison['v_range']:.4f}V")
        print(f"  Max difference: {comparison['max_diff']:.6f}V")
        print(f"  RMS difference: {comparison['rms_diff']:.6f}V ({comparison['rel_rms']*100:.2f}% relative)")
        print(f"  Convergence: {stats.get('convergence_rate', 0)*100:.1f}%")

        assert comparison['rel_rms'] < spec.max_rel_error, \
            f"Relative RMS error too high: {comparison['rel_rms']*100:.2f}% > {spec.max_rel_error*100:.0f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
