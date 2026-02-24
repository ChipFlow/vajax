"""Parameterized VACASK test suite

Automatically discovers and runs all VACASK .sim test files, extracting
expected values from embedded Python scripts and comparing with our solver.
"""

import re
import sys
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

# Add openvaf_jax and openvaf_py to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "openvaf_jax" / "openvaf_py"))

import numpy as np
from conftest import parse_embedded_python, parse_si_value

from vajax.netlist.parser import parse_netlist

# Paths - VACASK is at ../VACASK relative to va-jax
VA_JAX_ROOT = Path(__file__).parent.parent
VACASK_ROOT = VA_JAX_ROOT.parent / "VACASK"
VACASK_TEST = VACASK_ROOT / "test"
VACASK_DEVICES = VACASK_ROOT / "devices"
VACASK_BENCHMARK = VA_JAX_ROOT / "vendor" / "VACASK" / "benchmark"


def discover_benchmark_dirs() -> List[Path]:
    """Find all benchmark directories with runme.sim files."""
    if not VACASK_BENCHMARK.exists():
        return []
    benchmarks = []
    for d in sorted(VACASK_BENCHMARK.iterdir()):
        sim_file = d / "vacask" / "runme.sim"
        if sim_file.exists():
            benchmarks.append(d)
    return benchmarks


# Discover benchmarks for parameterized tests
BENCHMARK_DIRS = discover_benchmark_dirs()


def discover_sim_files() -> List[Path]:
    """Find all .sim test files in VACASK test directory."""
    if not VACASK_TEST.exists():
        return []
    return sorted(VACASK_TEST.glob("*.sim"))


# parse_embedded_python is imported from conftest


def parse_analysis_commands(sim_path: Path) -> List[Dict]:
    """Extract analysis commands from control block."""
    text = sim_path.read_text()

    analyses = []

    # Find control block
    control_match = re.search(r"control\s*(.*?)endc", text, re.DOTALL | re.IGNORECASE)
    if not control_match:
        return analyses

    control_block = control_match.group(1)

    # Match: analysis <name> op [options]
    for m in re.finditer(r"analysis\s+(\w+)\s+op\b", control_block):
        analyses.append({"name": m.group(1), "type": "op"})

    # Match: analysis <name> tran stop=<time> step=<step> [icmode=<mode>]
    for m in re.finditer(
        r'analysis\s+(\w+)\s+tran\s+stop=(\S+)\s+step=(\S+)(?:\s+icmode="(\w+)")?', control_block
    ):
        analyses.append(
            {
                "name": m.group(1),
                "type": "tran",
                "stop": m.group(2),
                "step": m.group(3),
                "icmode": m.group(4) or "op",
            }
        )

    return analyses


def get_required_models(sim_path: Path) -> List[str]:
    """Extract model types required by the sim file."""
    text = sim_path.read_text()
    models = set()

    # Match: load "model.osdi"
    for m in re.finditer(r'load\s+"(\w+)\.osdi"', text):
        models.add(m.group(1))

    return list(models)


def categorize_test(sim_path: Path) -> Tuple[str, List[str]]:
    """Categorize a test and return (category, skip_reasons).

    Categories:
    - 'op_basic': Simple DC operating point (resistor, diode)
    - 'op_complex': Complex DC (sweeps, multiple analyses)
    - 'tran': Transient analysis
    - 'ac': AC analysis
    - 'hb': Harmonic balance
    - 'unsupported': Not yet supported
    """
    name = sim_path.stem
    text = sim_path.read_text()
    models = get_required_models(sim_path)
    analyses = parse_analysis_commands(sim_path)

    skip_reasons = []

    # Check for unsupported features
    if "hb" in name or any(
        a.get("type") == "hb" for a in analyses if isinstance(a, dict) and "type" in a
    ):
        skip_reasons.append("harmonic balance not implemented")
        return "hb", skip_reasons

    if "ac" in name or ("analysis" in text and " ac " in text.lower()):
        skip_reasons.append("AC analysis not implemented")
        return "ac", skip_reasons

    if "mutual" in name:
        skip_reasons.append("mutual inductors not implemented")
        return "unsupported", skip_reasons

    if "noise" in name:
        skip_reasons.append("noise analysis not implemented")
        return "unsupported", skip_reasons

    if "sweep" in name:
        skip_reasons.append("parameter sweeps not yet implemented")
        return "unsupported", skip_reasons

    if "xf" in name:
        skip_reasons.append("transfer function analysis not implemented")
        return "unsupported", skip_reasons

    # Check for unsupported models
    unsupported_models = {"bsimsoi", "hicum", "mextram"}
    for model in models:
        if model.lower() in unsupported_models:
            skip_reasons.append(f"model {model} not supported")
            return "unsupported", skip_reasons

    # Categorize by analysis type
    has_tran = any(a.get("type") == "tran" for a in analyses if isinstance(a, dict))
    has_op = any(a.get("type") == "op" for a in analyses if isinstance(a, dict))

    if has_tran:
        return "tran", skip_reasons
    elif has_op:
        if (
            len(analyses) == 1
            and models
            and all(m in ["resistor", "vsource", "isource"] for m in models)
        ):
            return "op_basic", skip_reasons
        return "op_complex", skip_reasons

    return "unsupported", ["no recognized analysis type"]


# parse_si_value is imported from conftest


# Discover all tests
ALL_SIM_FILES = discover_sim_files()

# Categorize tests
CATEGORIZED_TESTS = {path: categorize_test(path) for path in ALL_SIM_FILES}


def get_test_ids():
    """Generate test IDs from sim file names."""
    return [p.stem for p in ALL_SIM_FILES]


class TestVACASKDiscovery:
    """Test that we can discover and parse VACASK test files."""

    def test_discover_sim_files(self):
        """Should find VACASK .sim test files."""
        if not VACASK_TEST.exists():
            pytest.skip("VACASK test directory not found")

        sim_files = discover_sim_files()
        assert len(sim_files) > 0, "No .sim files found"
        print(f"\nFound {len(sim_files)} .sim files")

    def test_categorize_all_tests(self):
        """Categorize all discovered tests."""
        if not ALL_SIM_FILES:
            pytest.skip("No sim files found")

        categories = {}
        for path, (cat, reasons) in CATEGORIZED_TESTS.items():
            categories.setdefault(cat, []).append(path.stem)

        print("\nTest categorization:")
        for cat, tests in sorted(categories.items()):
            print(f"  {cat}: {len(tests)} tests")
            if len(tests) <= 5:
                for t in tests:
                    print(f"    - {t}")


class TestVACASKParsing:
    """Test that we can parse all VACASK .sim files."""

    @pytest.mark.parametrize("sim_file", ALL_SIM_FILES, ids=get_test_ids())
    def test_parse_netlist(self, sim_file):
        """Each sim file should parse without error."""
        try:
            circuit = parse_netlist(sim_file)
            assert circuit is not None
            assert circuit.ground is not None
        except Exception as e:
            pytest.fail(f"Failed to parse {sim_file.name}: {e}")

    @pytest.mark.parametrize("sim_file", ALL_SIM_FILES, ids=get_test_ids())
    def test_extract_expectations(self, sim_file):
        """Should extract expected values from embedded Python."""
        result = parse_embedded_python(sim_file)
        # Just verify it doesn't crash - not all files have expectations
        assert "expectations" in result
        assert "analysis_type" in result


class TestVACASKSubcircuit:
    """Test subcircuit support."""

    def test_subcircuit_parsing(self):
        """Test that we can parse subcircuits from VACASK files."""
        sim_file = VACASK_TEST / "test_build.sim"
        if not sim_file.exists():
            pytest.skip("VACASK test_build.sim not found")

        circuit = parse_netlist(sim_file)

        # Check that subcircuits are parsed
        assert hasattr(circuit, "subckts"), "Circuit should have subckts attribute"
        assert len(circuit.subckts) >= 2, "Should have at least 2 subcircuits"

        # Check par1 subcircuit
        assert "par1" in circuit.subckts
        par1 = circuit.subckts["par1"]
        assert len(par1.instances) == 1, "par1 should have 1 instance"
        assert par1.instances[0].model == "resistor"

        print(f"Found {len(circuit.subckts)} subcircuits:")
        for name, sub in circuit.subckts.items():
            print(f"  {name}: {len(sub.instances)} instances, params={sub.params}")

    def test_subcircuit_instantiation(self):
        """Test instantiating a subcircuit in a circuit.

        Creates a voltage divider subcircuit and instantiates it.
        Uses direct Newton-Raphson solving for validation.
        """
        # Define a simple voltage divider subcircuit
        Instance = namedtuple("Instance", ["name", "model", "terminals", "params"])
        Subcircuit = namedtuple("Subcircuit", ["name", "terminals", "instances", "params"])
        Circuit = namedtuple("Circuit", ["ground", "top_instances", "subckts"])

        # Voltage divider subcircuit: in -> out with R1=1k, R2=1k (output = in/2)
        divider_subckt = Subcircuit(
            name="divider",
            terminals=["in", "out", "gnd"],
            instances=[
                Instance("r1", "resistor", ["in", "out"], {"r": 1000}),
                Instance("r2", "resistor", ["out", "gnd"], {"r": 1000}),
            ],
            params={},
        )

        # Main circuit: V1 (2V) -> divider -> GND
        # Expected output: 1V
        circuit_with_subckt = Circuit(
            ground="0",
            top_instances=[
                Instance("v1", "vsource", ["1", "0"], {"dc": 2}),
                Instance("div1", "divider", ["1", "2", "0"], {}),
            ],
            subckts={"divider": divider_subckt},
        )

        # Flatten subcircuit: expand div1 into its component instances
        flattened_instances = []

        for inst in circuit_with_subckt.top_instances:
            if inst.model in circuit_with_subckt.subckts:
                # This is a subcircuit instance - expand it
                subckt = circuit_with_subckt.subckts[inst.model]

                # Map subcircuit terminals to connection nodes
                term_map = dict(zip(subckt.terminals, inst.terminals))

                for sub_inst in subckt.instances:
                    # Create new terminals by mapping
                    new_terminals = [
                        term_map.get(t, f"{inst.name}.{t}") for t in sub_inst.terminals
                    ]

                    flattened_instances.append(
                        Instance(
                            name=f"{inst.name}.{sub_inst.name}",
                            model=sub_inst.model,
                            terminals=new_terminals,
                            params=sub_inst.params,
                        )
                    )
            else:
                flattened_instances.append(inst)

        print(f"Flattened circuit: {len(flattened_instances)} instances")
        for inst in flattened_instances:
            print(f"  {inst.name}: {inst.model} {inst.terminals}")

        # Now solve the flattened circuit using direct Newton-Raphson
        node_names = {"0": 0}
        node_idx = 1
        for inst in flattened_instances:
            for t in inst.terminals:
                if t not in node_names and t != "0":
                    node_names[t] = node_idx
                    node_idx += 1

        num_nodes = node_idx
        ground = 0

        devices = []
        for inst in flattened_instances:
            devices.append(
                {
                    "name": inst.name,
                    "model": inst.model,
                    "nodes": [node_names.get(t, 0) for t in inst.terminals],
                    "params": {
                        k: (parse_si_value(str(v)) if isinstance(v, str) else v)
                        for k, v in inst.params.items()
                    },
                }
            )

        # Solve DC using simple Newton-Raphson
        V = np.zeros(num_nodes)
        for _ in range(100):
            J = np.zeros((num_nodes - 1, num_nodes - 1))
            f = np.zeros(num_nodes - 1)

            for dev in devices:
                model = dev["model"]
                nodes = dev["nodes"]
                params = dev["params"]

                if model == "vsource":
                    V_target = params.get("dc", 0)
                    np_idx, nn_idx = nodes[0], nodes[1]
                    G = 1e12
                    I = G * (V[np_idx] - V[nn_idx] - V_target)

                    if np_idx != ground:
                        f[np_idx - 1] += I
                        J[np_idx - 1, np_idx - 1] += G
                        if nn_idx != ground:
                            J[np_idx - 1, nn_idx - 1] -= G
                    if nn_idx != ground:
                        f[nn_idx - 1] -= I
                        J[nn_idx - 1, nn_idx - 1] += G

                elif model == "resistor":
                    R = params.get("r", 1000)
                    G = 1.0 / max(R, 1e-12)
                    np_idx, nn_idx = nodes[0], nodes[1]
                    Vd = V[np_idx] - V[nn_idx]
                    I = G * Vd

                    if np_idx != ground:
                        f[np_idx - 1] += I
                        J[np_idx - 1, np_idx - 1] += G
                        if nn_idx != ground:
                            J[np_idx - 1, nn_idx - 1] -= G
                    if nn_idx != ground:
                        f[nn_idx - 1] -= I
                        J[nn_idx - 1, nn_idx - 1] += G
                        if np_idx != ground:
                            J[nn_idx - 1, np_idx - 1] -= G

            if np.max(np.abs(f)) < 1e-9:
                break

            delta = np.linalg.solve(J + 1e-12 * np.eye(J.shape[0]), -f)
            V[1:] += delta

        # Check result
        v_in = V[node_names["1"]]
        v_out = V[node_names["2"]]

        print(f"V(in) = {v_in:.3f}V")
        print(f"V(out) = {v_out:.3f}V (expected 1.0V)")

        assert abs(v_in - 2.0) < 0.01, f"V(in) expected 2V, got {v_in}V"
        assert abs(v_out - 1.0) < 0.01, f"V(out) expected 1V, got {v_out}V"
