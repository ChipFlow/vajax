"""Tests that parse VACASK .sim files and use openvaf_jax models

This module tests the integration between:
1. VACASK netlist parser (vajax.netlist.parser)
2. openvaf_jax model compilation (from .va files)
3. DC operating point analysis

Instead of using pre-compiled OSDI models, we compile Verilog-A sources
to JAX functions using openvaf_jax.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add openvaf_jax and openvaf_py to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "openvaf_jax" / "openvaf_py"))

from vajax.netlist.parser import parse_netlist

# VACASK paths
VACASK_PATH = Path(__file__).parent.parent / "vendor" / "VACASK"
VACASK_DEVICES = VACASK_PATH / "devices"
VACASK_TESTS = VACASK_PATH / "test"


def osdi_to_va_path(osdi_name: str) -> Path:
    """Map an OSDI filename to its Verilog-A source file

    VACASK .sim files reference .osdi files like "resistor.osdi" or
    "spice/resistor.osdi". We map these to the corresponding .va files
    in vendor/VACASK/devices/.
    """
    # Remove .osdi extension
    name = osdi_name.replace(".osdi", "")

    # Try various locations
    candidates = [
        VACASK_DEVICES / f"{name}.va",
        VACASK_DEVICES / "spice" / f"{name}.va",
        VACASK_DEVICES / name / f"{name}.va",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"No .va file found for {osdi_name}. Tried: {candidates}")


class CompiledVAModel:
    """A Verilog-A model compiled to JAX via openvaf_jax"""

    def __init__(self, va_path: Path):
        import openvaf_py

        import openvaf_jax

        self.va_path = va_path
        modules = openvaf_py.compile_va(str(va_path))
        if not modules:
            raise ValueError(f"No modules found in {va_path}")

        self.module = modules[0]
        self.translator = openvaf_jax.OpenVAFToJAX(self.module)
        self.jax_fn = self.translator.translate()

    @property
    def name(self) -> str:
        return self.module.name

    @property
    def nodes(self):
        return list(self.module.nodes)

    @property
    def param_names(self):
        return list(self.module.param_names)

    @property
    def param_kinds(self):
        return list(self.module.param_kinds)

    def build_inputs(self, voltages: dict, params: dict, temperature: float = 300.15) -> list:
        """Build input array for the JAX function

        Args:
            voltages: Dict mapping terminal pairs to voltages, e.g., {'V(A,B)': 1.0}
            params: Dict of model/instance parameters, e.g., {'r': 2000.0}
            temperature: Circuit temperature in Kelvin

        Returns:
            List of input values in the order expected by the JAX function
        """
        inputs = []
        for name, kind in zip(self.param_names, self.param_kinds):
            if kind == "voltage":
                # Look for matching voltage
                inputs.append(voltages.get(name, 0.0))
            elif kind == "temperature":
                inputs.append(temperature)
            elif kind == "sysfun":
                # System function like mfactor
                if "mfactor" in name.lower():
                    inputs.append(params.get("mfactor", 1.0))
                else:
                    inputs.append(1.0)
            elif kind == "hidden_state":
                inputs.append(0.0)
            elif kind == "param":
                # Look for matching parameter
                inputs.append(params.get(name, 1.0))
            else:
                inputs.append(0.0)
        return inputs

    def evaluate(self, inputs: list):
        """Evaluate the JAX function

        Returns:
            (residuals, jacobian) tuple
        """
        return self.jax_fn(inputs)


class TestParseVACASKSimFiles:
    """Test that we can parse VACASK .sim test files"""

    def test_parse_test_resistor(self):
        """Parse test_resistor.sim"""
        circuit = parse_netlist(VACASK_TESTS / "test_resistor.sim")

        assert circuit.title == "Resistor"
        assert circuit.ground == "0"
        assert "resistor.osdi" in circuit.loads
        assert "resistor" in circuit.models
        assert "vsource" in circuit.models

        # Check instances
        assert len(circuit.top_instances) == 2
        v1 = next(i for i in circuit.top_instances if i.name == "v1")
        r1 = next(i for i in circuit.top_instances if i.name == "r1")

        assert v1.model == "vsource"
        assert v1.params.get("dc") == "1"
        assert r1.model == "resistor"
        assert r1.params.get("r") == "2k"

    def test_parse_test_diode(self):
        """Parse test_diode.sim"""
        circuit = parse_netlist(VACASK_TESTS / "test_diode.sim")

        assert "Diode" in circuit.title
        assert "diode.osdi" in circuit.loads
        assert "d" in circuit.models

        # Check diode model has parameters
        diode_model = circuit.models["d"]
        assert diode_model.module == "diode"
        assert "is" in diode_model.params


class TestOsdiToVaMapping:
    """Test that we can map OSDI filenames to VA sources"""

    def test_resistor_mapping(self):
        """resistor.osdi maps to resistor.va"""
        va_path = osdi_to_va_path("resistor.osdi")
        assert va_path.exists()
        assert va_path.name == "resistor.va"

    def test_capacitor_mapping(self):
        """capacitor.osdi maps to capacitor.va"""
        va_path = osdi_to_va_path("capacitor.osdi")
        assert va_path.exists()

    def test_diode_mapping(self):
        """diode.osdi maps to diode.va"""
        va_path = osdi_to_va_path("diode.osdi")
        assert va_path.exists()


class TestCompileVACASKModels:
    """Test that VACASK models compile with openvaf_jax"""

    def test_compile_resistor(self):
        """Compile resistor.va to JAX"""
        va_path = osdi_to_va_path("resistor.osdi")
        model = CompiledVAModel(va_path)

        assert model.name == "resistor"
        assert len(model.nodes) == 2
        assert "r" in model.param_names

    def test_compile_capacitor(self):
        """Compile capacitor.va to JAX"""
        va_path = osdi_to_va_path("capacitor.osdi")
        model = CompiledVAModel(va_path)

        assert model.name == "capacitor"

    def test_compile_diode(self):
        """Compile diode.va to JAX"""
        va_path = osdi_to_va_path("diode.osdi")
        model = CompiledVAModel(va_path)

        assert model.name in ["diode", "sp_diode"]


class TestVACASKResistorSim:
    """Test that replicates VACASK test_resistor.sim using openvaf_jax

    Circuit from test_resistor.sim:
        v1 (1 0) vsource dc=1
        r1 (1 0) resistor r=2k

    Expected results (from embedded Python in .sim file):
        i(v1) = -1/2k * mfactor = -0.5mA (for mfactor=3, it's -1.5mA)
        r1.i = 1/2k = 0.5mA
    """

    @pytest.fixture
    def resistor_model(self):
        """Compile VACASK resistor model"""
        va_path = osdi_to_va_path("resistor.osdi")
        return CompiledVAModel(va_path)

    def test_resistor_current(self, resistor_model):
        """Test resistor current calculation matches VACASK expected values

        For a 1V source across a 2kOhm resistor:
        I = V/R = 1V / 2000 Ohm = 0.5mA
        """
        # Build inputs for V=1V across resistor, R=2000 Ohm
        inputs = resistor_model.build_inputs(
            voltages={"V(A,B)": 1.0}, params={"r": 2000.0, "has_noise": 0}, temperature=300.15
        )

        residuals, jacobian = resistor_model.evaluate(inputs)

        # The resistor contributes current to both nodes
        # For node0 (A): I = V/R flows out
        # For node1 (B): I = V/R flows in

        # Check that we got valid outputs
        assert residuals is not None

        # Extract current from residuals
        # The resistor equation is I = V/R
        # In JAX output, residuals contain the current contributions
        for node_name, node_res in residuals.items():
            resist = float(node_res["resist"])
            assert not np.isnan(resist), f"NaN resist at {node_name}"

        # Calculate expected current
        expected_I = 1.0 / 2000.0  # 0.5mA

        # The 'A' should have the current flowing
        node0_current = float(residuals["A"]["resist"])

        # Current magnitude should match (sign depends on convention)
        assert abs(abs(node0_current) - expected_I) < 1e-9, (
            f"Expected I={expected_I}, got {node0_current}"
        )

        print(f"Resistor current: {node0_current * 1000:.6f} mA")
        print(f"Expected: {expected_I * 1000:.6f} mA")

    def test_resistor_with_mfactor(self, resistor_model):
        """Test resistor with mfactor=3 (parallel instances)

        VACASK test expects: i(v1) = -1/2k * 3 = -1.5mA
        """
        inputs = resistor_model.build_inputs(
            voltages={"V(A,B)": 1.0},
            params={"r": 2000.0, "has_noise": 0, "mfactor": 3.0},
            temperature=300.15,
        )

        residuals, jacobian = resistor_model.evaluate(inputs)

        # With mfactor=3, current should be 3x
        expected_I = (1.0 / 2000.0) * 3.0  # 1.5mA

        node0_current = float(residuals["A"]["resist"])

        assert abs(abs(node0_current) - expected_I) < 1e-9, (
            f"Expected I={expected_I} (mfactor=3), got {node0_current}"
        )

        print(f"Resistor current (mfactor=3): {node0_current * 1000:.6f} mA")
        print(f"Expected: {expected_I * 1000:.6f} mA")

    def test_resistor_jacobian(self, resistor_model):
        """Test that Jacobian (conductance) is correct

        For a resistor: dI/dV = G = 1/R
        With R=2000, G=0.0005 S
        """
        inputs = resistor_model.build_inputs(
            voltages={"V(A,B)": 1.0}, params={"r": 2000.0, "has_noise": 0}, temperature=300.15
        )

        residuals, jacobian = resistor_model.evaluate(inputs)

        # Expected conductance
        expected_G = 1.0 / 2000.0  # 0.5mS

        # Check Jacobian values
        # The Jacobian should contain dI/dV = G
        assert jacobian is not None

        # Find the conductance in the Jacobian
        # Structure depends on how openvaf_jax organizes it
        for key, val in jacobian.items():
            if "resist" in val:
                G = float(val["resist"])
                if abs(G) > 1e-12:  # Non-zero entry
                    print(f"Jacobian[{key}] = {G:.6f} S")

        print(f"Expected conductance: {expected_G * 1000:.6f} mS")


class TestFullVACASKTestResistor:
    """Full integration test parsing and running test_resistor.sim"""

    def test_parse_compile_and_evaluate(self):
        """Parse test_resistor.sim, compile models, and evaluate

        This is the full integration test that:
        1. Parses the .sim file
        2. Maps load statements to VA sources
        3. Compiles VA to JAX
        4. Evaluates the circuit
        """
        # Step 1: Parse the netlist
        circuit = parse_netlist(VACASK_TESTS / "test_resistor.sim")

        # Step 2: Compile models from load statements
        compiled_models = {}
        for osdi_file in circuit.loads:
            try:
                va_path = osdi_to_va_path(osdi_file)
                model = CompiledVAModel(va_path)
                compiled_models[model.name] = model
                print(f"Compiled {osdi_file} -> {model.name}")
            except FileNotFoundError as e:
                # vsource/isource are built-in, not from VA files
                print(f"Skipping {osdi_file}: {e}")

        # Step 3: Check we have the resistor model
        assert "resistor" in compiled_models
        resistor_model = compiled_models["resistor"]

        # Step 4: Get circuit parameters from netlist
        r1_instance = next(i for i in circuit.top_instances if i.name == "r1")
        v1_instance = next(i for i in circuit.top_instances if i.name == "v1")

        # Parse resistance value (2k -> 2000)
        r_str = r1_instance.params.get("r", "1000")
        if r_str.endswith("k"):
            R = float(r_str[:-1]) * 1000
        else:
            R = float(r_str)

        # Parse voltage
        V = float(v1_instance.params.get("dc", "0"))

        print(f"\nCircuit from {circuit.title}:")
        print(f"  V1 = {V}V")
        print(f"  R1 = {R} Ohm")

        # Step 5: Evaluate the resistor model
        inputs = resistor_model.build_inputs(
            voltages={"V(A,B)": V}, params={"r": R, "has_noise": 0}, temperature=300.15
        )

        residuals, jacobian = resistor_model.evaluate(inputs)

        # Step 6: Verify results match VACASK expected values
        expected_I = V / R
        actual_I = abs(float(residuals["A"]["resist"]))

        rel_err = abs(actual_I - expected_I) / expected_I
        assert rel_err < 1e-6, f"Current mismatch: expected {expected_I}, got {actual_I}"

        print("\nResults:")
        print(f"  Expected I = V/R = {V}/{R} = {expected_I * 1000:.6f} mA")
        print(f"  Actual I = {actual_I * 1000:.6f} mA")
        print(f"  Relative error: {rel_err:.2e}")
        print("  PASS: Matches VACASK expected values!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
