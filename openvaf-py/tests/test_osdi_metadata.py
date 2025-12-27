"""Tests for OSDI metadata comparison between openvaf-py and OpenVAF snapshots.

This module validates that openvaf-py correctly exposes Verilog-A model metadata
matching the OSDI descriptor format used by OpenVAF.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import openvaf_py
from snap_parser import load_snap_file, parse_snap_file

# Base directories
OPENVAF_DIR = Path(__file__).parent.parent / "vendor" / "OpenVAF"
INTEGRATION_TESTS_DIR = OPENVAF_DIR / "integration_tests"
SNAP_DIR = OPENVAF_DIR / "openvaf" / "test_data" / "osdi"


# Models to test - mapping from snapshot name to Verilog-A file path
# Start with simple models that have straightforward comparison
TEST_MODELS = [
    ("resistor", INTEGRATION_TESTS_DIR / "RESISTOR" / "resistor.va"),
    ("diode", INTEGRATION_TESTS_DIR / "DIODE" / "diode.va"),
    ("cccs", INTEGRATION_TESTS_DIR / "CCCS" / "cccs.va"),
    ("vccs", INTEGRATION_TESTS_DIR / "VCCS" / "vccs.va"),
    ("current_source", INTEGRATION_TESTS_DIR / "CURRENT_SOURCE" / "current_source.va"),
]


def get_osdi_descriptor(va_path: Path) -> dict:
    """Compile a Verilog-A file and return its OSDI descriptor."""
    modules = openvaf_py.compile_va(str(va_path))
    if not modules:
        raise ValueError(f"No modules found in {va_path}")
    return modules[0].get_osdi_descriptor()


class TestParameterMetadata:
    """Tests for parameter metadata correctness."""

    @pytest.mark.parametrize("model_name,va_path", TEST_MODELS)
    def test_parameter_names_match(self, model_name: str, va_path: Path):
        """Verify parameter names match OSDI reference (excluding built-ins)."""
        if not va_path.exists():
            pytest.skip(f"VA file not found: {va_path}")

        snap_path = SNAP_DIR / f"{model_name}.snap"
        if not snap_path.exists():
            pytest.skip(f"Snapshot not found: {snap_path}")

        expected = load_snap_file(model_name, SNAP_DIR)
        actual = get_osdi_descriptor(va_path)

        # Filter out built-in parameters (starting with $) from expected
        expected_params = [p.name for p in expected.params if not p.name.startswith("$")]
        actual_params = [p["name"] for p in actual["params"]]

        assert set(actual_params) == set(expected_params), (
            f"Parameter mismatch for {model_name}:\n"
            f"  Expected: {expected_params}\n"
            f"  Actual: {actual_params}"
        )

    @pytest.mark.parametrize("model_name,va_path", TEST_MODELS)
    def test_parameter_units_match(self, model_name: str, va_path: Path):
        """Verify parameter units match OSDI reference."""
        if not va_path.exists():
            pytest.skip(f"VA file not found: {va_path}")

        snap_path = SNAP_DIR / f"{model_name}.snap"
        if not snap_path.exists():
            pytest.skip(f"Snapshot not found: {snap_path}")

        expected = load_snap_file(model_name, SNAP_DIR)
        actual = get_osdi_descriptor(va_path)

        # Build lookup by name
        expected_units = {p.name: p.units for p in expected.params if not p.name.startswith("$")}
        actual_units = {p["name"]: p["units"] for p in actual["params"]}

        for name in expected_units:
            if name in actual_units:
                assert actual_units[name] == expected_units[name], (
                    f"Unit mismatch for param '{name}' in {model_name}:\n"
                    f"  Expected: '{expected_units[name]}'\n"
                    f"  Actual: '{actual_units[name]}'"
                )

    @pytest.mark.parametrize("model_name,va_path", TEST_MODELS)
    def test_parameter_descriptions_match(self, model_name: str, va_path: Path):
        """Verify parameter descriptions match OSDI reference."""
        if not va_path.exists():
            pytest.skip(f"VA file not found: {va_path}")

        snap_path = SNAP_DIR / f"{model_name}.snap"
        if not snap_path.exists():
            pytest.skip(f"Snapshot not found: {snap_path}")

        expected = load_snap_file(model_name, SNAP_DIR)
        actual = get_osdi_descriptor(va_path)

        # Build lookup by name
        expected_desc = {p.name: p.description for p in expected.params if not p.name.startswith("$")}
        actual_desc = {p["name"]: p["description"] for p in actual["params"]}

        for name in expected_desc:
            if name in actual_desc:
                assert actual_desc[name] == expected_desc[name], (
                    f"Description mismatch for param '{name}' in {model_name}:\n"
                    f"  Expected: '{expected_desc[name]}'\n"
                    f"  Actual: '{actual_desc[name]}'"
                )

    @pytest.mark.parametrize("model_name,va_path", TEST_MODELS)
    def test_parameter_instance_flag_match(self, model_name: str, va_path: Path):
        """Verify parameter instance/model flags match OSDI reference."""
        if not va_path.exists():
            pytest.skip(f"VA file not found: {va_path}")

        snap_path = SNAP_DIR / f"{model_name}.snap"
        if not snap_path.exists():
            pytest.skip(f"Snapshot not found: {snap_path}")

        expected = load_snap_file(model_name, SNAP_DIR)
        actual = get_osdi_descriptor(va_path)

        # Build lookup by name
        expected_inst = {p.name: p.is_instance for p in expected.params if not p.name.startswith("$")}
        actual_inst = {p["name"]: p["is_instance"] for p in actual["params"]}

        for name in expected_inst:
            if name in actual_inst:
                assert actual_inst[name] == expected_inst[name], (
                    f"Instance flag mismatch for param '{name}' in {model_name}:\n"
                    f"  Expected is_instance: {expected_inst[name]}\n"
                    f"  Actual is_instance: {actual_inst[name]}"
                )


class TestNodeMetadata:
    """Tests for node metadata correctness."""

    @pytest.mark.parametrize("model_name,va_path", TEST_MODELS)
    def test_terminal_count_matches(self, model_name: str, va_path: Path):
        """Verify terminal count matches OSDI reference."""
        if not va_path.exists():
            pytest.skip(f"VA file not found: {va_path}")

        snap_path = SNAP_DIR / f"{model_name}.snap"
        if not snap_path.exists():
            pytest.skip(f"Snapshot not found: {snap_path}")

        expected = load_snap_file(model_name, SNAP_DIR)
        actual = get_osdi_descriptor(va_path)

        assert actual["num_terminals"] == expected.num_terminals, (
            f"Terminal count mismatch for {model_name}:\n"
            f"  Expected: {expected.num_terminals}\n"
            f"  Actual: {actual['num_terminals']}"
        )

    @pytest.mark.parametrize("model_name,va_path", TEST_MODELS)
    def test_node_count_matches(self, model_name: str, va_path: Path):
        """Verify total node count matches OSDI reference."""
        if not va_path.exists():
            pytest.skip(f"VA file not found: {va_path}")

        snap_path = SNAP_DIR / f"{model_name}.snap"
        if not snap_path.exists():
            pytest.skip(f"Snapshot not found: {snap_path}")

        expected = load_snap_file(model_name, SNAP_DIR)
        actual = get_osdi_descriptor(va_path)

        assert actual["num_nodes"] == len(expected.nodes), (
            f"Node count mismatch for {model_name}:\n"
            f"  Expected: {len(expected.nodes)}\n"
            f"  Actual: {actual['num_nodes']}"
        )

    @pytest.mark.parametrize("model_name,va_path", TEST_MODELS)
    def test_node_names_match(self, model_name: str, va_path: Path):
        """Verify node names match OSDI reference."""
        if not va_path.exists():
            pytest.skip(f"VA file not found: {va_path}")

        snap_path = SNAP_DIR / f"{model_name}.snap"
        if not snap_path.exists():
            pytest.skip(f"Snapshot not found: {snap_path}")

        expected = load_snap_file(model_name, SNAP_DIR)
        actual = get_osdi_descriptor(va_path)

        expected_names = [n.name for n in expected.nodes]
        actual_names = [n["name"] for n in actual["nodes"]]

        assert actual_names == expected_names, (
            f"Node names mismatch for {model_name}:\n"
            f"  Expected: {expected_names}\n"
            f"  Actual: {actual_names}"
        )


class TestJacobianMetadata:
    """Tests for Jacobian structure correctness."""

    @pytest.mark.parametrize("model_name,va_path", TEST_MODELS)
    def test_jacobian_count_matches(self, model_name: str, va_path: Path):
        """Verify Jacobian entry count matches OSDI reference."""
        if not va_path.exists():
            pytest.skip(f"VA file not found: {va_path}")

        snap_path = SNAP_DIR / f"{model_name}.snap"
        if not snap_path.exists():
            pytest.skip(f"Snapshot not found: {snap_path}")

        expected = load_snap_file(model_name, SNAP_DIR)
        actual = get_osdi_descriptor(va_path)

        assert actual["num_jacobian_entries"] == len(expected.jacobian), (
            f"Jacobian entry count mismatch for {model_name}:\n"
            f"  Expected: {len(expected.jacobian)}\n"
            f"  Actual: {actual['num_jacobian_entries']}"
        )

    @pytest.mark.parametrize("model_name,va_path", TEST_MODELS)
    def test_jacobian_sparsity_pattern_matches(self, model_name: str, va_path: Path):
        """Verify Jacobian (row, col) sparsity pattern matches OSDI reference."""
        if not va_path.exists():
            pytest.skip(f"VA file not found: {va_path}")

        snap_path = SNAP_DIR / f"{model_name}.snap"
        if not snap_path.exists():
            pytest.skip(f"Snapshot not found: {snap_path}")

        expected = load_snap_file(model_name, SNAP_DIR)
        actual = get_osdi_descriptor(va_path)

        # Build node name to index mapping from actual nodes
        node_names = [n["name"] for n in actual["nodes"]]

        # Convert expected (name, name) to (index, index) pattern
        expected_pattern = set()
        for e in expected.jacobian:
            row_idx = node_names.index(e.row) if e.row in node_names else -1
            col_idx = node_names.index(e.col) if e.col in node_names else -1
            expected_pattern.add((row_idx, col_idx))

        # Get actual pattern
        actual_pattern = {(j["row"], j["col"]) for j in actual["jacobian"]}

        assert actual_pattern == expected_pattern, (
            f"Jacobian sparsity pattern mismatch for {model_name}:\n"
            f"  Expected: {sorted(expected_pattern)}\n"
            f"  Actual: {sorted(actual_pattern)}"
        )

    @pytest.mark.parametrize("model_name,va_path", TEST_MODELS)
    def test_jacobian_flags_match(self, model_name: str, va_path: Path):
        """Verify Jacobian entry flags match OSDI reference."""
        if not va_path.exists():
            pytest.skip(f"VA file not found: {va_path}")

        snap_path = SNAP_DIR / f"{model_name}.snap"
        if not snap_path.exists():
            pytest.skip(f"Snapshot not found: {snap_path}")

        expected = load_snap_file(model_name, SNAP_DIR)
        actual = get_osdi_descriptor(va_path)

        # Build node name to index mapping
        node_names = [n["name"] for n in actual["nodes"]]

        # Build expected flags map by (row_idx, col_idx)
        expected_flags = {}
        for e in expected.jacobian:
            row_idx = node_names.index(e.row) if e.row in node_names else -1
            col_idx = node_names.index(e.col) if e.col in node_names else -1
            expected_flags[(row_idx, col_idx)] = {
                "has_resist": e.has_resist,
                "has_react": e.has_react,
                "resist_const": e.resist_const,
                "react_const": e.react_const,
            }

        # Check each actual entry
        for j in actual["jacobian"]:
            key = (j["row"], j["col"])
            if key in expected_flags:
                exp = expected_flags[key]
                assert j["has_resist"] == exp["has_resist"], f"has_resist mismatch at {key}"
                assert j["has_react"] == exp["has_react"], f"has_react mismatch at {key}"
                assert j["resist_const"] == exp["resist_const"], f"resist_const mismatch at {key}"
                assert j["react_const"] == exp["react_const"], f"react_const mismatch at {key}"


class TestCollapsiblePairs:
    """Tests for node collapse metadata."""

    @pytest.mark.parametrize("model_name,va_path", TEST_MODELS)
    def test_collapsible_count_matches(self, model_name: str, va_path: Path):
        """Verify collapsible pair count matches OSDI reference."""
        if not va_path.exists():
            pytest.skip(f"VA file not found: {va_path}")

        snap_path = SNAP_DIR / f"{model_name}.snap"
        if not snap_path.exists():
            pytest.skip(f"Snapshot not found: {snap_path}")

        expected = load_snap_file(model_name, SNAP_DIR)
        actual = get_osdi_descriptor(va_path)

        assert actual["num_collapsible"] == len(expected.collapsible), (
            f"Collapsible count mismatch for {model_name}:\n"
            f"  Expected: {len(expected.collapsible)}\n"
            f"  Actual: {actual['num_collapsible']}"
        )


class TestMiscMetadata:
    """Tests for miscellaneous metadata fields."""

    @pytest.mark.parametrize("model_name,va_path", TEST_MODELS)
    def test_num_states_matches(self, model_name: str, va_path: Path):
        """Verify num_states matches OSDI reference."""
        if not va_path.exists():
            pytest.skip(f"VA file not found: {va_path}")

        snap_path = SNAP_DIR / f"{model_name}.snap"
        if not snap_path.exists():
            pytest.skip(f"Snapshot not found: {snap_path}")

        expected = load_snap_file(model_name, SNAP_DIR)
        actual = get_osdi_descriptor(va_path)

        assert actual["num_states"] == expected.num_states, (
            f"num_states mismatch for {model_name}:\n"
            f"  Expected: {expected.num_states}\n"
            f"  Actual: {actual['num_states']}"
        )

    @pytest.mark.parametrize("model_name,va_path", TEST_MODELS)
    def test_has_bound_step_matches(self, model_name: str, va_path: Path):
        """Verify has_bound_step matches OSDI reference."""
        if not va_path.exists():
            pytest.skip(f"VA file not found: {va_path}")

        snap_path = SNAP_DIR / f"{model_name}.snap"
        if not snap_path.exists():
            pytest.skip(f"Snapshot not found: {snap_path}")

        expected = load_snap_file(model_name, SNAP_DIR)
        actual = get_osdi_descriptor(va_path)

        assert actual["has_bound_step"] == expected.has_bound_step, (
            f"has_bound_step mismatch for {model_name}:\n"
            f"  Expected: {expected.has_bound_step}\n"
            f"  Actual: {actual['has_bound_step']}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
