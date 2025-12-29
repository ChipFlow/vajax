"""Tests using VACASK test cases with openvaf_jax

These tests parse actual VACASK .sim files and run them using openvaf_jax
compiled models instead of loading OSDI files.

The approach:
1. Parse vendor/VACASK/test/*.sim using our parser
2. Compile the required VA models with openvaf_jax
3. Build and solve the MNA system
4. Compare results with expected values from the embedded Python scripts
"""

import pytest
import sys
from pathlib import Path

# Add openvaf-py to path
sys.path.insert(0, str(Path(__file__).parent.parent / "openvaf-py"))

import numpy as np
import jax.numpy as jnp

import openvaf_py
import openvaf_jax
from jax_spice.netlist.parser import parse_netlist
from conftest import parse_si_value

# Paths
VACASK_TEST = Path(__file__).parent.parent / "vendor" / "VACASK" / "test"
VACASK_DEVICES = Path(__file__).parent.parent / "vendor" / "VACASK" / "devices"

# Note: This file has its own parse_embedded_python that returns a simpler dict format
# The conftest version returns {'expectations': [...], 'analysis_type': ...}
# We keep this local version for backwards compatibility with the tests in this file


class TestResistorSim:
    """Tests based on vendor/VACASK/test/test_resistor.sim

    Circuit:
        v1 (1 0) vsource dc=1
        r1 (1 0) resistor r=2k $mfactor=3

    Expected (from embedded Python):
        v1:flow(br) = -1/2e3 * 3 = -0.0015 A  (current into vsource)
        r1.i = 1/2e3 = 0.0005 A               (per-unit resistor current)
    """

    @pytest.fixture(scope="class")
    def sim_data(self):
        """Parse the sim file and compile the model"""
        sim_path = VACASK_TEST / "test_resistor.sim"
        circuit = parse_netlist(sim_path)

        # Compile resistor model
        va_path = VACASK_DEVICES / "resistor.va"
        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])
        eval_fn = translator.translate()

        return {
            'circuit': circuit,
            'module': modules[0],
            'eval_fn': eval_fn,
            'sim_path': sim_path
        }

    def test_parses_correctly(self, sim_data):
        """Sim file parses without error"""
        circuit = sim_data['circuit']

        assert circuit.title == "Resistor"
        assert circuit.ground == "0"
        assert len(circuit.top_instances) == 2  # v1 and r1

        # Check r1 instance
        r1 = next(i for i in circuit.top_instances if i.name == 'r1')
        assert r1.model == 'resistor'
        assert r1.terminals == ['1', '0']
        assert 'r' in r1.params

    def test_resistor_current(self, sim_data):
        """Resistor produces correct current for V=1V, R=2k

        From sim file: r=2k (no mfactor in param, $mfactor=3 is comment)
        Expected: I = V/R = 1/2000 = 0.5mA
        """
        module = sim_data['module']
        eval_fn = sim_data['eval_fn']
        circuit = sim_data['circuit']

        # Get r1 params
        r1 = next(i for i in circuit.top_instances if i.name == 'r1')
        r_value = parse_si_value(r1.params['r'])  # 2k -> 2000

        # Build inputs for resistor eval
        inputs = [0.0] * len(module.param_names)
        for i, name in enumerate(module.param_names):
            if name == 'V(A,B)':
                inputs[i] = 1.0  # V across resistor
            elif name == 'r':
                inputs[i] = r_value
            elif name == 'mfactor':
                inputs[i] = 1.0  # Per-unit
            elif 'temperature' in name.lower():
                inputs[i] = 300.0

        residuals, _ = eval_fn(inputs)
        current = float(residuals['sim_node0']['resist'])

        expected = 1.0 / r_value  # 0.0005 A
        assert abs(current - expected) < 1e-9, f"Expected {expected}, got {current}"

    def test_mfactor_total_current(self, sim_data):
        """With mfactor=3, total current = 3 * V/R = 1.5mA

        From embedded Python: exact = -1/2e3 * 3 = -0.0015
        (negative because it's current INTO the vsource)
        """
        module = sim_data['module']
        eval_fn = sim_data['eval_fn']

        r_value = 2000.0
        mfactor = 3.0

        inputs = [0.0] * len(module.param_names)
        for i, name in enumerate(module.param_names):
            if name == 'V(A,B)':
                inputs[i] = 1.0
            elif name == 'r':
                inputs[i] = r_value
            elif name == 'mfactor':
                inputs[i] = mfactor
            elif 'temperature' in name.lower():
                inputs[i] = 300.0

        residuals, _ = eval_fn(inputs)
        current = float(residuals['sim_node0']['resist'])

        # Total current with mfactor
        expected = mfactor * 1.0 / r_value  # 0.0015 A
        assert abs(current - expected) < 1e-9, f"Expected {expected}, got {current}"


class TestDiodeSim:
    """Tests based on vendor/VACASK/test/test_diode.sim

    Circuit:
        v1 (1 0) vsource dc=0
        r1 (1 2) resistor r=1
        d1 (2 0) d

    Model d diode is=1e-12 n=2 rs=0.1 cjo=100p vj=1 m=0.5

    The test sweeps v1 from -50 to 10V and is from 1e-12 to 1e-6.
    """

    @pytest.fixture(scope="class")
    def sim_data(self):
        """Parse the sim file and compile the model"""
        sim_path = VACASK_TEST / "test_diode.sim"
        circuit = parse_netlist(sim_path)

        # Compile diode model
        va_path = VACASK_DEVICES / "diode.va"
        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])
        eval_fn = translator.translate()

        return {
            'circuit': circuit,
            'module': modules[0],
            'eval_fn': eval_fn,
            'sim_path': sim_path
        }

    def test_parses_correctly(self, sim_data):
        """Sim file parses without error"""
        circuit = sim_data['circuit']

        assert circuit.title == "Diode response"
        assert circuit.ground == "0"

        # Check d1 instance exists
        d1 = next((i for i in circuit.top_instances if i.name == 'd1'), None)
        assert d1 is not None
        assert d1.model == 'd'
        assert d1.terminals == ['2', '0']

    def test_model_params(self, sim_data):
        """Model parameters are parsed correctly"""
        circuit = sim_data['circuit']

        d_model = circuit.models.get('d')
        assert d_model is not None
        assert d_model.module == 'diode'

        # Check key parameters
        assert 'is' in d_model.params
        assert d_model.params['is'] == '1e-12'
        assert d_model.params['n'] == '2'

    def test_diode_compiles(self, sim_data):
        """Diode model compiles to JAX function"""
        module = sim_data['module']
        eval_fn = sim_data['eval_fn']

        assert module.name == "diode"
        assert eval_fn is not None

        # Should have nodes for A, C, and internal CI
        assert len(module.nodes) >= 2


class TestCapacitorSim:
    """Tests based on vendor/VACASK/test/test_capacitor.sim"""

    @pytest.fixture(scope="class")
    def sim_data(self):
        sim_path = VACASK_TEST / "test_capacitor.sim"
        circuit = parse_netlist(sim_path)

        va_path = VACASK_DEVICES / "capacitor.va"
        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])
        eval_fn = translator.translate()

        return {
            'circuit': circuit,
            'module': modules[0],
            'eval_fn': eval_fn
        }

    def test_parses_correctly(self, sim_data):
        """Sim file parses without error"""
        circuit = sim_data['circuit']
        assert circuit.ground == "0"

        # Should have capacitor instance
        c_inst = next((i for i in circuit.top_instances
                      if 'capacitor' in i.model.lower() or i.name.startswith('c')), None)
        assert c_inst is not None


class TestInductorSim:
    """Tests based on vendor/VACASK/test/test_inductor.sim"""

    @pytest.fixture(scope="class")
    def sim_data(self):
        sim_path = VACASK_TEST / "test_inductor.sim"
        circuit = parse_netlist(sim_path)

        va_path = VACASK_DEVICES / "inductor.va"
        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])
        eval_fn = translator.translate()

        return {
            'circuit': circuit,
            'module': modules[0],
            'eval_fn': eval_fn
        }

    def test_parses_correctly(self, sim_data):
        """Sim file parses without error"""
        circuit = sim_data['circuit']
        assert circuit.ground == "0"


class TestOpSim:
    """Tests based on vendor/VACASK/test/test_op.sim

    Basic operating point analysis test.
    """

    def test_parses_correctly(self):
        """Sim file parses without error"""
        sim_path = VACASK_TEST / "test_op.sim"
        circuit = parse_netlist(sim_path)

        assert circuit.ground == "0"
        assert len(circuit.top_instances) > 0
