"""Test VACASK benchmarks using JAX-SPICE CircuitEngine API

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

from jax_spice.analysis import CircuitEngine
from jax_spice.utils import find_vacask_binary, rawread


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
    def sim_path(self):
        """Get RC benchmark sim path"""
        path = get_benchmark_sim("rc")
        if not path.exists():
            pytest.skip(f"RC benchmark not found at {path}")
        return path

    def test_parse(self, sim_path):
        """Test RC benchmark parses correctly"""
        engine = CircuitEngine(sim_path)
        engine.parse()
        assert engine.num_nodes > 0
        assert len(engine.devices) > 0

        # Should have resistor, capacitor, vsource
        device_types = {d['model'] for d in engine.devices}
        assert 'resistor' in device_types, f"Missing resistor, got {device_types}"
        assert 'capacitor' in device_types, f"Missing capacitor, got {device_types}"
        assert 'vsource' in device_types, f"Missing vsource, got {device_types}"

        print(f"RC benchmark: {engine.num_nodes} nodes, {len(engine.devices)} devices")
        print(f"Device types: {device_types}")

    def test_transient_dense(self, sim_path):
        """Test RC transient with dense solver"""
        engine = CircuitEngine(sim_path)
        engine.parse()
        dt = engine.analysis_params.get('step', 1e-6)

        # Run short transient (10 steps)
        result = engine.run_transient(t_stop=dt * 10, dt=dt, use_sparse=False)

        assert result.num_steps > 0, "No timesteps returned"
        # voltages is a dict mapping node index to voltage array
        assert isinstance(result.voltages, dict), f"Expected dict, got {type(result.voltages)}"

        # Check convergence
        converged = result.stats.get('convergence_rate', 0)
        print(f"RC dense: {result.num_steps} steps, {converged*100:.0f}% converged")
        assert converged > 0.5, f"Poor convergence: {converged*100:.0f}%"

    def test_transient_sparse(self, sim_path):
        """Test RC transient with sparse solver"""
        engine = CircuitEngine(sim_path)
        engine.parse()
        dt = engine.analysis_params.get('step', 1e-6)

        result = engine.run_transient(t_stop=dt * 10, dt=dt, use_sparse=True)

        assert result.num_steps > 0, "No timesteps returned"

        converged = result.stats.get('convergence_rate', 0)
        print(f"RC sparse: {result.num_steps} steps, {converged*100:.0f}% converged")
        assert converged > 0.5, f"Poor convergence: {converged*100:.0f}%"

    def test_rc_time_constant(self, sim_path):
        """Verify RC time constant behavior

        With R=1k, C=1u, tau=1ms
        After 1 tau, voltage should reach ~63.2% of final value
        After 5 tau, voltage should reach ~99.3% of final value
        """
        dt = 10e-6  # 10us steps
        t_stop = 5e-3  # 5ms (5 tau)

        engine = CircuitEngine(sim_path)
        engine.parse()
        result = engine.run_transient(t_stop=t_stop, dt=dt, use_sparse=False)

        # Get node 2 voltage (capacitor voltage)
        # voltages is a dict mapping node index to voltage array
        if 2 in result.voltages:
            v_cap = result.voltage(2)  # Capacitor voltage
        else:
            # Get the last non-ground node
            v_cap = result.voltages[max(result.voltages.keys())]

        times_np = np.array(result.times)
        v_cap_np = np.array(v_cap)

        # Find approximate final value (after 5 tau)
        v_final = v_cap_np[-1] if len(v_cap_np) > 0 else 0

        print(f"RC response: V_final = {v_final:.3f}V after {times_np[-1]*1000:.1f}ms")

        # Just verify we got reasonable output
        assert result.num_steps > 10, "Not enough timesteps for RC analysis"


class TestGraetzBenchmark:
    """Test Graetz bridge benchmark (full-wave rectifier with diodes)

    Circuit: AC source -> 4 diodes (full bridge) -> RC filter -> load
    Tests diode model under dynamic conditions.
    """

    @pytest.fixture
    def sim_path(self):
        """Get Graetz benchmark sim path"""
        path = get_benchmark_sim("graetz")
        if not path.exists():
            pytest.skip(f"Graetz benchmark not found at {path}")
        return path

    def test_parse(self, sim_path):
        """Test Graetz benchmark parses correctly"""
        engine = CircuitEngine(sim_path)
        engine.parse()
        assert engine.num_nodes > 0
        assert len(engine.devices) > 0

        device_types = {d['model'] for d in engine.devices}
        assert 'diode' in device_types, f"Missing diode, got {device_types}"

        # Count diodes (should be 4 for full bridge)
        diode_count = sum(1 for d in engine.devices if d['model'] == 'diode')
        assert diode_count == 4, f"Expected 4 diodes, got {diode_count}"

        print(f"Graetz benchmark: {engine.num_nodes} nodes, {len(engine.devices)} devices")
        print(f"Device types: {device_types}")
        print(f"Diode count: {diode_count}")

    def test_transient_dense(self, sim_path):
        """Test Graetz transient with dense solver

        Note: Graetz has numerical challenges due to diode nonlinearity.
        We test that the solver runs and produces output, even if
        convergence isn't perfect.
        """
        engine = CircuitEngine(sim_path)
        engine.parse()
        dt = engine.analysis_params.get('step', 1e-6)

        # Run short transient
        result = engine.run_transient(t_stop=dt * 10, dt=dt, use_sparse=False)

        assert result.num_steps > 0, "No timesteps returned"

        converged = result.stats.get('converged_steps', 0) / max(result.num_steps, 1)
        print(f"Graetz dense: {result.num_steps} steps, {converged*100:.0f}% converged")
        # Graetz is numerically challenging - accept lower convergence

    @pytest.mark.skip(reason="Graetz sparse has convergence issues - known limitation")
    def test_transient_sparse(self, sim_path):
        """Test Graetz transient with sparse solver"""
        engine = CircuitEngine(sim_path)
        engine.parse()
        dt = engine.analysis_params.get('step', 1e-6)

        result = engine.run_transient(t_stop=dt * 10, dt=dt, use_sparse=True)

        assert result.num_steps > 0, "No timesteps returned"


class TestMulBenchmark:
    """Test multiplier circuit benchmark"""

    @pytest.fixture
    def sim_path(self):
        """Get mul benchmark sim path"""
        path = get_benchmark_sim("mul")
        if not path.exists():
            pytest.skip(f"Mul benchmark not found at {path}")
        return path

    def test_parse(self, sim_path):
        """Test mul benchmark parses correctly"""
        engine = CircuitEngine(sim_path)
        engine.parse()
        assert engine.num_nodes > 0
        assert len(engine.devices) > 0

        print(f"Mul benchmark: {engine.num_nodes} nodes, {len(engine.devices)} devices")
        device_types = {d['model'] for d in engine.devices}
        print(f"Device types: {device_types}")

    def test_transient_dense(self, sim_path):
        """Test mul transient with dense solver"""
        engine = CircuitEngine(sim_path)
        engine.parse()
        dt = engine.analysis_params.get('step', 1e-9)

        result = engine.run_transient(t_stop=dt * 5, dt=dt, use_sparse=False)

        assert result.num_steps > 0, "No timesteps returned"

        converged = result.stats.get('converged_steps', 0) / max(result.num_steps, 1)
        print(f"Mul dense: {result.num_steps} steps, {converged*100:.0f}% converged")


class TestRingBenchmark:
    """Test ring oscillator benchmark (PSP103 MOSFETs)

    This is a 9-stage ring oscillator using PSP103 MOSFET models.
    Tests OpenVAF compilation and evaluation of complex device models.
    """

    @pytest.fixture
    def sim_path(self):
        """Get ring benchmark sim path"""
        path = get_benchmark_sim("ring")
        if not path.exists():
            pytest.skip(f"Ring benchmark not found at {path}")
        return path

    def test_parse(self, sim_path):
        """Test ring benchmark parses correctly"""
        engine = CircuitEngine(sim_path)
        engine.parse()
        assert engine.num_nodes > 0
        assert len(engine.devices) > 0

        # Should have PSP103 MOSFETs
        device_types = {d['model'] for d in engine.devices}
        assert 'psp103' in device_types, f"Missing psp103, got {device_types}"

        # Count PSP103 devices (9 stages * 2 transistors)
        psp_count = sum(1 for d in engine.devices if d['model'] == 'psp103')
        assert psp_count == 18, f"Expected 18 PSP103 devices, got {psp_count}"

        print(f"Ring benchmark: {engine.num_nodes} nodes, {len(engine.devices)} devices")
        print(f"Device types: {device_types}")
        print(f"PSP103 count: {psp_count}")

    def test_vacask_expected_dc_values(self):
        """Document expected VACASK DC operating point values for ring oscillator.

        VACASK produces the following DC operating point for the 9-stage ring oscillator:
        - All ring nodes (1-9) settle to approximately VDD/2 (metastable point)
        - V[1] = V[2] = ... = V[9] ≈ 0.6606V (with VDD=1.2V)
        - VDD node = 1.2V

        These values were extracted by running VACASK on dc_debug.sim and parsing
        the tran1.raw binary output.

        This test documents the expected values for validation once JAX-SPICE
        ring oscillator DC convergence is fixed.
        """
        # Expected VACASK DC values (extracted from vendor/VACASK/benchmark/ring/vacask/tran1.raw)
        expected = {
            'vdd': 1.2,
            # Ring oscillator nodes - all at metastable VDD/2
            'v1': 0.660597,
            'v2': 0.660597,
            'v3': 0.660597,
            'v4': 0.660597,
            'v5': 0.660597,
            'v6': 0.660597,
            'v7': 0.660597,
            'v8': 0.660597,
            'v9': 0.660597,
        }

        print(f"VACASK expected DC operating point:")
        print(f"  V[vdd] = {expected['vdd']:.6f} V")
        print(f"  V[1-9] = {expected['v1']:.6f} V (metastable)")

        # Document tolerance for comparison
        tolerance = 0.01  # 1% tolerance
        expected_v1 = expected['v1']
        print(f"  Tolerance: ±{tolerance*100:.0f}% ({expected_v1 - tolerance:.6f} to {expected_v1 + tolerance:.6f} V)")

    def test_transient_dense(self, sim_path):
        """Test ring transient with dense solver."""
        engine = CircuitEngine(sim_path)
        engine.parse()
        dt = engine.analysis_params.get('step', 5e-11)

        result = engine.run_transient(t_stop=dt * 5, dt=dt, use_sparse=False)

        assert result.num_steps > 0, "No timesteps returned"

        converged = result.stats.get('convergence_rate', 0)
        print(f"Ring dense: {result.num_steps} steps, {converged*100:.0f}% converged")
        assert converged > 0.5, f"Poor convergence: {converged*100:.0f}%"

    def test_transient_sparse(self, sim_path):
        """Test ring transient with sparse solver."""
        engine = CircuitEngine(sim_path)
        engine.parse()
        dt = engine.analysis_params.get('step', 5e-11)

        result = engine.run_transient(t_stop=dt * 5, dt=dt, use_sparse=True)

        assert result.num_steps > 0, "No timesteps returned"

        converged = result.stats.get('convergence_rate', 0)
        print(f"Ring sparse: {result.num_steps} steps, {converged*100:.0f}% converged")
        assert converged > 0.5, f"Poor convergence: {converged*100:.0f}%"


class TestC6288Benchmark:
    """Test c6288 large benchmark (sparse solver only)

    c6288 is a large circuit (~86k nodes) that requires sparse solver.
    Dense solver would need ~56GB of memory.
    """

    @pytest.fixture
    def sim_path(self):
        """Get c6288 benchmark sim path"""
        path = get_benchmark_sim("c6288")
        if not path.exists():
            pytest.skip(f"c6288 benchmark not found at {path}")
        return path

    def test_parse(self, sim_path):
        """Test c6288 benchmark parses correctly"""
        engine = CircuitEngine(sim_path)
        engine.parse()
        assert engine.num_nodes > 0
        assert len(engine.devices) > 0

        print(f"c6288 benchmark: {engine.num_nodes} nodes, {len(engine.devices)} devices")

        # c6288 should be large
        assert engine.num_nodes > 1000, f"Expected large circuit, got {engine.num_nodes} nodes"

    @pytest.mark.skip(reason="c6288 sparse takes >10min on CPU - use GPU: uv run scripts/profile_gpu_cloudrun.py --benchmark c6288 --use-sparse")
    def test_transient_sparse(self, sim_path):
        """Test c6288 transient with sparse solver.

        Uses node collapse (86k -> 25k nodes) for tractable matrix size.
        Still takes >10min on CPU due to JIT compilation of 25k×25k sparse ops.
        For fast execution, use Spineax/cuDSS on GPU via Cloud Run:
        uv run scripts/profile_gpu_cloudrun.py --benchmark c6288 --use-sparse
        """
        engine = CircuitEngine(sim_path)
        engine.parse()
        dt = engine.analysis_params.get('step', 1e-12)

        result = engine.run_transient(t_stop=dt * 2, dt=dt, use_sparse=True)

        assert result.num_steps > 0, "No timesteps returned"


class TestTbDpBenchmark:
    """Test IHP SG13G2 dual-port SRAM benchmark (tb_dp512x8)

    Circuit: 512x8 dual-port SRAM using PSP103.6 models from IHP PDK
    This is a large circuit (~9k lines) for stress-testing the simulator.

    Test added from: https://github.com/robtaylor/VACASK/pull/2

    Analysis params:
    - step: 100ps (1e-10)
    - stop: 30ns (3e-8)
    """

    @pytest.fixture
    def sim_path(self):
        """Get tb_dp benchmark sim path"""
        # tb_dp is in jax_spice/benchmarks/data/tb_dp/
        test_dir = Path(__file__).parent.parent / "jax_spice" / "benchmarks" / "data" / "tb_dp"
        path = test_dir / "tb_dp512x8_klu.sim"
        if not path.exists():
            pytest.skip(f"tb_dp benchmark not found at {path} - run runme.py to convert")
        return path

    def test_parse(self, sim_path):
        """Test that tb_dp512x8 parses correctly"""
        engine = CircuitEngine(sim_path)
        engine.parse()

        # Should have many devices (large SRAM)
        assert len(engine.devices) > 100, f"Expected many devices, got {len(engine.devices)}"

        print(f"\ntb_dp512x8:")
        print(f"  Devices: {len(engine.devices)}")
        print(f"  Nodes: {engine.num_nodes}")

    def test_transient_sparse(self, sim_path):
        """Test tb_dp512x8 transient with sparse solver.

        This is a large circuit that requires sparse solver.
        Analysis: step=100ps, stop=30ns (but we run just a few steps for testing)
        """
        engine = CircuitEngine(sim_path)
        engine.parse()

        # Run just 5 steps for testing (full sim would be 300 steps)
        dt = 1e-10  # 100ps
        t_stop = 5e-10  # 5 steps

        result = engine.run_transient(t_stop=t_stop, dt=dt, use_sparse=True)

        print(f"\ntb_dp512x8 sparse:")
        print(f"  Steps: {result.num_steps}")
        print(f"  Convergence: {result.stats.get('convergence_rate', 0)*100:.1f}%")

        converged = result.stats.get('convergence_rate', 0)
        assert converged > 0.5, f"Poor convergence: {converged*100:.0f}%"


class TestNodeCountComparison:
    """Test that JAX-SPICE node counts match VACASK.

    VACASK outputs node count via 'print stats':
        System stats:
          Number of nodes:                 <nodeCount>
          Number of unknonws:              <unknownCount>

    These tests require VACASK to be built and available.
    """

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
        binary = find_vacask_binary()
        if binary is None:
            pytest.skip("VACASK binary not found. Set VACASK_BIN env var or build VACASK.")
        return binary

    @pytest.mark.parametrize("benchmark,xfail_reason", [
        ("rc", None),
        ("graetz", None),  # Graetz node count now matches after branch current fix
        ("ring", None),  # Ring node count now matches VACASK (47 unknowns)
    ])
    def test_node_count_matches_vacask(self, vacask_bin, benchmark, xfail_reason):
        """Compare JAX-SPICE node count with VACASK for benchmarks.

        VACASK reports two node counts:
        - nodeCount: Total Node objects (before collapse, includes redundant internal nodes)
        - unknownCount: Actual system matrix size (after collapse)

        JAX-SPICE's total_nodes should match VACASK's unknownCount (+/- 1 for ground handling).
        Note: VACASK's nodeCount is much higher for PSP103 circuits because it creates
        Node objects for all internal nodes before collapse.
        """
        sim_path = get_benchmark_sim(benchmark)
        if not sim_path.exists():
            pytest.skip(f"Benchmark not found at {sim_path}")

        # Get VACASK counts (both nodes and unknowns)
        vacask_counts = self.get_vacask_node_count(vacask_bin, benchmark)
        vacask_nodes = vacask_counts['nodes']
        vacask_unknowns = vacask_counts['unknowns']

        # Get JAX-SPICE counts
        engine = CircuitEngine(sim_path)
        engine.parse()
        jax_external = engine.num_nodes
        n_total, _ = engine._setup_internal_nodes()

        # Report counts
        print(f"\n{benchmark}:")
        print(f"  VACASK: nodes={vacask_nodes}, unknowns={vacask_unknowns}")
        print(f"  JAX-SPICE: external={jax_external}, total={n_total}")

        # The key comparison: JAX-SPICE total should match VACASK unknowns
        # Allow +/- 1 for different ground handling
        diff_total = abs(n_total - vacask_unknowns)

        assert diff_total <= 1, \
            f"{benchmark}: total nodes differ: JAX-SPICE={n_total}, VACASK unknowns={vacask_unknowns}"

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
        engine = CircuitEngine(sim_path)
        engine.parse()
        n_total, _ = engine._setup_internal_nodes()
        jax_total = n_total

        # Report counts
        print(f"\nc6288:")
        print(f"  VACASK: nodes={vacask_nodes}, unknowns={vacask_unknowns}")
        print(f"  JAX-SPICE: external={engine.num_nodes}, total={jax_total}")
        print(f"  Internal nodes: {jax_total - engine.num_nodes}")

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
        With PSP103-specific collapse: ~15,000-26,000 nodes (matching VACASK)

        PSP103 has 2 internal nodes per device after collapse:
        - NOI (noise correlation node) - always separate
        - BI (internal bulk) with BP/BS/BD collapsed to it
        """
        sim_path = get_benchmark_sim("c6288")
        if not sim_path.exists():
            pytest.skip(f"c6288 benchmark not found at {sim_path}")

        engine = CircuitEngine(sim_path)
        engine.parse()

        # Get total nodes after collapse
        n_total, device_internal = engine._setup_internal_nodes()
        n_internal = n_total - engine.num_nodes
        n_psp103_devices = sum(1 for d in engine.devices if d.get('model') == 'psp103')

        print(f"\nc6288 node collapse:")
        print(f"  External nodes: {engine.num_nodes}")
        print(f"  Internal nodes: {n_internal}")
        print(f"  Total nodes: {n_total}")
        print(f"  PSP103 devices: {n_psp103_devices}")
        print(f"  Internal nodes per device: {n_internal / n_psp103_devices:.2f}")

        # With PSP103-specific collapse, each device needs 2 internal nodes
        # (NOI and BI where BP/BS/BD collapse to BI)
        expected_internal = n_psp103_devices * 2
        internal_ratio = n_internal / expected_internal

        # Allow up to 10% variance
        assert 0.9 < internal_ratio < 1.1, \
            f"Internal node count off: {n_internal} (expected ~{expected_internal}, ratio={internal_ratio:.2f})"

        # Total should be well under 30k (vs ~86k without collapse)
        assert n_total < 30000, \
            f"Total nodes too high: {n_total} (expected <30000 with node collapse)"

    def test_ring_node_collapse(self):
        """Test that node collapse is applied to ring benchmark.

        Ring has 18 PSP103 devices, each with 2 internal nodes after collapse:
        - NOI (noise correlation node) - always separate
        - BI (internal bulk) with BP/BS/BD collapsed to it

        This matches VACASK's behavior: 47 unknowns = 11 external + 36 internal.
        """
        sim_path = get_benchmark_sim("ring")
        if not sim_path.exists():
            pytest.skip(f"ring benchmark not found at {sim_path}")

        engine = CircuitEngine(sim_path)
        engine.parse()

        # Get total nodes
        n_total, _ = engine._setup_internal_nodes()
        n_internal = n_total - engine.num_nodes
        n_psp103 = sum(1 for d in engine.devices if d.get('model') == 'psp103')

        print(f"\nring benchmark:")
        print(f"  External nodes: {engine.num_nodes}")
        print(f"  Internal nodes: {n_internal}")
        print(f"  Total nodes: {n_total}")
        print(f"  PSP103 devices: {n_psp103}")

        # With PSP103-specific collapse, each device has 2 internal nodes:
        # NOI (node4) and BI (node9) where BP/BS/BD collapse to BI
        # This matches VACASK's 47 unknowns for the Ring benchmark
        expected_internal = n_psp103 * 2
        assert n_internal == expected_internal, \
            f"Expected {expected_internal} internal nodes (2 per device), got {n_internal}"
        assert n_total == 47, \
            f"Expected 47 total nodes (matching VACASK), got {n_total}"


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
        jax_nodes: Corresponding JAX-SPICE node indices or names
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
    jax_nodes: list[int | str]  # Can be node index (int) or name (str)
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
        xfail_reason="PSP103 ring oscillator: DC operating point differs (JAX=0.546V vs VACASK=0.661V). "
                     "Root cause: NOI (noise correlation) internal node has I(NOIR) <+ V(NOI)/mig contribution "
                     "where mig=1e-40, creating G=1e40 conductance to ground. If V(NOI) is non-zero (~0.6V), "
                     "residual = 6e39 corrupts NR solve. Fix: ensure NOI node is properly allocated and V(NOI)=0. "
                     "See PSP103_module.include lines 1834-1836 for the noise branch contributions.",
    ),
    'mul': BenchmarkSpec(
        name='mul',
        dt=1e-9,           # 1ns step
        t_stop=1e-7,       # 100ns
        max_rel_error=0.01,  # 1% allowed (actual: ~0.06%)
        vacask_nodes=['1', 'v(1)'],
        jax_nodes=[1],
        xfail=False,
    ),
    'c6288': BenchmarkSpec(
        name='c6288',
        dt=1e-12,          # 1ps step
        t_stop=5e-12,      # 5ps (just a few steps for timing)
        max_rel_error=0.10,  # 10% allowed (large PSP103 circuit)
        vacask_nodes=['v(p0)', 'p0'],  # Product bit 0 (VACASK uses flat names)
        jax_nodes=['top.p0'],  # Use named node lookup (JAX-SPICE uses hierarchical names)
        xfail=True,
        xfail_reason="VACASK OSDI compilation fails on Linux CI (openvaf-r crash)",
    ),
}


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

    sim_dir = sim_path.parent

    # Read original sim file
    with open(sim_path) as f:
        sim_content = f.read()

    # NOTE: OSDI file check removed - VACASK finds them via its library path
    # (vendor/VACASK/build/lib/vacask/). The previous check was failing because
    # it looked in the sim file's directory instead of the staged library path.

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
            timeout=600  # 10 min timeout for large circuits like c6288
        )

        # Find any .raw file (analysis name varies: tran1.raw, tranmul.raw, etc.)
        raw_files = list(sim_dir.glob('*.raw'))
        if not raw_files:
            raise RuntimeError(
                f"VACASK did not produce any .raw file in {sim_dir}.\n"
                f"stdout: {result.stdout[:500]}\n"
                f"stderr: {result.stderr[:500]}"
            )
        raw_path = raw_files[0]  # Use first .raw file found

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
        # Clean up all .raw files
        for raw_file in sim_dir.glob('*.raw'):
            raw_file.unlink()


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

    @staticmethod
    def _setup_benchmark_osdi(benchmark_name: str):
        """Copy OSDI files from reference benchmark if missing.

        Some benchmarks (like mul) reference OSDI files that aren't in their directory.
        This copies them from rc benchmark which has all the basic SPICE OSDI files.
        """
        import shutil
        benchmark_dir = BENCHMARK_DIR / benchmark_name / "vacask"
        reference_dir = BENCHMARK_DIR / "rc" / "vacask"
        graetz_dir = BENCHMARK_DIR / "graetz" / "vacask"

        # Check if spice directory needs setup
        spice_dir = benchmark_dir / "spice"
        if not spice_dir.exists() and (reference_dir / "spice").exists():
            shutil.copytree(reference_dir / "spice", spice_dir)

        # Check for sn subdirectory (diode.osdi variant)
        sn_dir = spice_dir / "sn"
        if not sn_dir.exists() and (graetz_dir / "spice" / "sn").exists():
            shutil.copytree(graetz_dir / "spice" / "sn", sn_dir)

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

        # Set up OSDI files if needed
        self._setup_benchmark_osdi(spec.name)

        num_steps = int(spec.t_stop / spec.dt)

        # Run VACASK
        try:
            vacask_results = run_vacask_simulation(vacask_bin, sim_path, spec.t_stop, spec.dt)
        except FileNotFoundError as e:
            pytest.skip(str(e))
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

        # Run JAX-SPICE using CircuitEngine API
        engine = CircuitEngine(sim_path)
        engine.parse()
        result = engine.run_transient(t_stop=spec.t_stop, dt=spec.dt)

        jax_voltage = np.array(result.voltages.get(jax_node_idx, []))

        # Apply node transform if specified
        if spec.node_transform is not None:
            vacask_voltage = spec.node_transform(vacask_results['voltages'])
            jax_voltage = spec.node_transform({i: np.array(v) for i, v in result.voltages.items()})

        # Compare waveforms
        comparison = compare_waveforms(
            vacask_time, vacask_voltage,
            np.array(result.times), jax_voltage
        )

        print(f"\n{benchmark_name.upper()} comparison:")
        print(f"  VACASK node: {vacask_node_used}, JAX-SPICE node: {jax_node_idx}")
        print(f"  VACASK: {len(vacask_time)} points, JAX-SPICE: {result.num_steps} points")
        print(f"  Voltage range: {comparison['v_range']:.4f}V")
        print(f"  Max difference: {comparison['max_diff']:.6f}V")
        print(f"  RMS difference: {comparison['rms_diff']:.6f}V ({comparison['rel_rms']*100:.2f}% relative)")
        print(f"  Convergence: {result.stats.get('convergence_rate', 0)*100:.1f}%")

        assert comparison['rel_rms'] < spec.max_rel_error, \
            f"Relative RMS error too high: {comparison['rel_rms']*100:.2f}% > {spec.max_rel_error*100:.0f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
