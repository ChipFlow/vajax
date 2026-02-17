"""Tests for the OpenVAFToJAX translator API.

Tests the new translate_init() and translate_eval() methods with param validation.
"""

import os

os.environ["JAX_ENABLE_X64"] = "true"

from pathlib import Path

import jax.numpy as jnp
import openvaf_py
import pytest

import openvaf_jax
from jax_spice import build_simparams

# Test model paths
PROJECT_ROOT = Path(__file__).parent.parent
VACASK_DEVICES = PROJECT_ROOT / "vendor" / "VACASK" / "devices"
OPENVAF_TESTS = PROJECT_ROOT / "vendor" / "OpenVAF" / "integration_tests"


class TestParamInfo:
    """Tests for _get_param_info() helper."""

    def test_param_info_resistor(self):
        """Test param info extraction from resistor model."""
        va_path = VACASK_DEVICES / "resistor.va"
        if not va_path.exists():
            pytest.skip(f"Model not found: {va_path}")

        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])
        param_info = translator._get_param_info()

        # Resistor should have r parameter (lowercase in VACASK model)
        assert "r" in param_info, f"Expected 'r' param, got: {list(param_info.keys())}"
        r_info = param_info["r"]
        assert r_info["type"] == "real"
        assert r_info["default"] is not None

    def test_param_info_ekv_types(self):
        """Test that EKV TYPE param is correctly identified as int."""
        va_path = OPENVAF_TESTS / "EKV" / "ekv.va"
        if not va_path.exists():
            pytest.skip(f"Model not found: {va_path}")

        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])
        param_info = translator._get_param_info()

        # TYPE should be integer
        type_info = param_info.get("TYPE")
        assert type_info is not None, "EKV should have TYPE param"
        assert type_info["type"] == "int", f"TYPE should be int, got {type_info['type']}"

    def test_param_info_defaults(self):
        """Test that defaults are correctly extracted."""
        va_path = OPENVAF_TESTS / "EKV" / "ekv.va"
        if not va_path.exists():
            pytest.skip(f"Model not found: {va_path}")

        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])
        param_info = translator._get_param_info()

        # Check known defaults
        vto_info = param_info.get("VTO")
        if vto_info and vto_info["default"] is not None:
            assert vto_info["default"] == 0.5, (
                f"VTO default should be 0.5, got {vto_info['default']}"
            )


class TestTranslateInit:
    """Tests for translate_init() method."""

    def test_translate_init_basic(self):
        """Test basic translate_init functionality."""
        va_path = VACASK_DEVICES / "resistor.va"
        if not va_path.exists():
            pytest.skip(f"Model not found: {va_path}")

        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])

        init_fn, metadata = translator.translate_init(params={"R": 1000.0}, temperature=300.0)

        # Check metadata structure
        assert "param_names" in metadata
        assert "param_kinds" in metadata
        assert "init_inputs" in metadata
        assert "param_info" in metadata
        assert "warnings" in metadata

        # Run init function
        init_inputs = jnp.array(metadata["init_inputs"])
        cache, collapse = init_fn(init_inputs)
        assert cache.shape[0] >= 0  # Cache may be empty for simple models

    def test_translate_init_case_insensitive(self):
        """Test that param names are case-insensitive."""
        va_path = OPENVAF_TESTS / "EKV" / "ekv.va"
        if not va_path.exists():
            pytest.skip(f"Model not found: {va_path}")

        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])

        # Both should work
        init_fn1, meta1 = translator.translate_init(params={"VTO": 0.6})
        init_fn2, meta2 = translator.translate_init(params={"vto": 0.6})

        # Find VTO in param_info
        vto_idx = None
        for i, p in enumerate(meta1["param_info"]):
            if p["name"].upper() == "VTO":
                vto_idx = i
                break

        if vto_idx is not None:
            assert meta1["init_inputs"][vto_idx] == meta2["init_inputs"][vto_idx]

    def test_translate_init_temperature(self):
        """Test temperature handling."""
        va_path = OPENVAF_TESTS / "EKV" / "ekv.va"
        if not va_path.exists():
            pytest.skip(f"Model not found: {va_path}")

        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])

        init_fn, metadata = translator.translate_init(temperature=350.0)

        # Find $temperature in params
        temp_idx = None
        for i, p in enumerate(metadata["param_info"]):
            if p["name"] == "$temperature":
                temp_idx = i
                break

        if temp_idx is not None:
            assert metadata["init_inputs"][temp_idx] == 350.0


class TestTranslateEval:
    """Tests for translate_eval() method."""

    def test_translate_eval_basic(self):
        """Test basic translate_eval functionality."""
        va_path = VACASK_DEVICES / "resistor.va"
        if not va_path.exists():
            pytest.skip(f"Model not found: {va_path}")

        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])

        eval_fn, metadata = translator.translate_eval(params={"R": 1000.0}, temperature=300.0)

        # Check metadata structure
        assert "param_names" in metadata
        assert "param_kinds" in metadata
        assert "shared_inputs" in metadata
        assert "shared_indices" in metadata
        assert "voltage_indices" in metadata
        assert "node_names" in metadata
        assert "jacobian_keys" in metadata

    def test_translate_eval_voltage_indices(self):
        """Test that voltage indices are correctly computed."""
        va_path = OPENVAF_TESTS / "EKV" / "ekv.va"
        if not va_path.exists():
            pytest.skip(f"Model not found: {va_path}")

        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])

        eval_fn, metadata = translator.translate_eval()

        # Check that voltage indices point to voltage or implicit_unknown params
        # Both are runtime voltage values that come from the varying_params array
        voltage_kinds = ("voltage", "implicit_unknown")
        for idx in metadata["voltage_indices"]:
            kind = metadata["param_kinds"][idx]
            assert kind in voltage_kinds, (
                f"Index {idx} should be voltage/implicit_unknown, got {kind}"
            )

    def test_translate_eval_full_flow(self):
        """Test full init+eval flow with new API."""
        va_path = VACASK_DEVICES / "resistor.va"
        if not va_path.exists():
            pytest.skip(f"Model not found: {va_path}")

        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])

        # Get init function and run
        init_fn, init_meta = translator.translate_init(params={"R": 1000.0})
        init_inputs = jnp.array(init_meta["init_inputs"])
        cache, _ = init_fn(init_inputs)

        # Get eval function and run
        eval_fn, eval_meta = translator.translate_eval(params={"R": 1000.0})
        shared_inputs = jnp.array(eval_meta["shared_inputs"])

        # Build voltage inputs (1V across resistor)
        voltage_inputs = jnp.array([1.0])  # V(p,n) = 1V

        # Build simparams array
        simparams = jnp.array(build_simparams(eval_meta))

        # eval_fn signature: (shared_params, device_params, shared_cache, device_cache,
        #                     simparams, limit_state_in, limit_funcs)
        # With default cache_split=None, shared_cache is empty and device_cache is full cache
        shared_cache = jnp.array([])
        limit_state_in = jnp.array([])
        limit_funcs = {}

        result = eval_fn(
            shared_inputs,
            voltage_inputs,
            shared_cache,
            cache,
            simparams,
            limit_state_in,
            limit_funcs,
        )
        res_resist = result[0]

        # Current should be V/R = 1/1000 = 1mA
        # Residual = I - V/R for KCL, depends on model
        assert len(res_resist) >= 2  # At least 2 nodes for resistor


class TestTranslateEvalEKV:
    """Tests for translate_eval() with EKV model."""

    def test_ekv_full_flow(self):
        """Test full EKV evaluation flow."""
        va_path = OPENVAF_TESTS / "EKV" / "ekv.va"
        if not va_path.exists():
            pytest.skip(f"Model not found: {va_path}")

        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])

        # Init
        init_fn, init_meta = translator.translate_init(params={"TYPE": 1}, temperature=300.0)
        init_inputs = jnp.array(init_meta["init_inputs"])
        cache, _ = init_fn(init_inputs)

        # Eval
        eval_fn, eval_meta = translator.translate_eval(params={"TYPE": 1}, temperature=300.0)
        shared_inputs = jnp.array(eval_meta["shared_inputs"])

        # Build voltage inputs
        node_names = eval_meta["node_names"]
        V = [0.5, 0.6, 0.0, 0.0, 0.0, 0.0]  # Vds=0.5, Vgs=0.6, + internal nodes at 0

        voltage_inputs = []
        for i in eval_meta["voltage_indices"]:
            name = eval_meta["param_names"][i]
            kind = eval_meta["param_kinds"][i]

            if kind == "implicit_unknown":
                # Internal node voltage - set to 0.0 (solver would compute this)
                voltage_inputs.append(0.0)
            elif name.startswith("V("):
                inner = name[2:-1]  # Remove V( and )
                if "," in inner:
                    node_pos, node_neg = inner.split(",")
                    idx_pos = node_names.index(node_pos.strip())
                    idx_neg = node_names.index(node_neg.strip())
                    voltage_inputs.append(V[idx_pos] - V[idx_neg])
                else:
                    idx = node_names.index(inner.strip())
                    voltage_inputs.append(V[idx])
            else:
                # Unknown voltage param format - default to 0.0
                voltage_inputs.append(0.0)

        voltage_arr = jnp.array(voltage_inputs)

        # Build simparams array
        simparams = jnp.array(build_simparams(eval_meta))

        result = eval_fn(shared_inputs, voltage_arr, cache, simparams)
        res_resist = result[0]

        # Should produce non-zero drain current
        # Residual at drain node is negative when current flows OUT of the device
        # (into drain terminal), so Ids = -residual[drain_idx]
        drain_residual = float(res_resist[0])
        assert drain_residual != 0.0, "EKV should produce non-zero residual"
        # For NMOS with Vgs > Vth, current flows into drain (positive residual)
        # or out of source (checking that we have conduction)
        assert abs(drain_residual) > 1e-12, "NMOS should have significant drain current"


class TestGetParams:
    """Tests for get_params() and print_params() methods."""

    def test_get_params_diode(self):
        """Test get_params returns correct metadata for diode."""
        va_path = VACASK_DEVICES / "diode.va"
        if not va_path.exists():
            pytest.skip(f"Model not found: {va_path}")

        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])

        params = translator.get_params()

        # Should have params
        assert len(params) > 0

        # Check structure of param dict
        p = params[0]
        assert "name" in p
        assert "type" in p
        assert "default" in p
        assert "units" in p
        assert "description" in p
        assert "aliases" in p
        assert "is_instance" in p
        assert "is_model_param" in p

        # Check Is param (should be first)
        is_param = next((p for p in params if p["name"] == "Is"), None)
        assert is_param is not None
        assert is_param["type"] == "real"
        assert is_param["default"] == 1e-14
        assert is_param["units"] == "A"
        assert "Saturation" in is_param["description"]

    def test_get_params_ekv_types(self):
        """Test that get_params correctly identifies integer params."""
        va_path = OPENVAF_TESTS / "EKV" / "ekv.va"
        if not va_path.exists():
            pytest.skip(f"Model not found: {va_path}")

        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])

        params = translator.get_params()

        # TYPE should be int
        type_param = next((p for p in params if p["name"] == "TYPE"), None)
        assert type_param is not None
        assert type_param["type"] == "int"

    def test_get_params_excludes_system(self):
        """Test that get_params excludes system params by default."""
        va_path = OPENVAF_TESTS / "EKV" / "ekv.va"
        if not va_path.exists():
            pytest.skip(f"Model not found: {va_path}")

        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])

        params = translator.get_params()

        # Should not include $temperature or mfactor
        names = [p["name"] for p in params]
        assert "$temperature" not in names
        assert "mfactor" not in names

    def test_print_params(self, capsys):
        """Test print_params produces expected output."""
        va_path = VACASK_DEVICES / "diode.va"
        if not va_path.exists():
            pytest.skip(f"Model not found: {va_path}")

        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])

        translator.print_params()

        captured = capsys.readouterr()
        assert "Parameters for" in captured.out
        assert "Name" in captured.out
        assert "Type" in captured.out
        assert "Default" in captured.out
        assert "Is" in captured.out  # First param


class TestDebugMode:
    """Tests for debug mode output."""

    def test_debug_mode_init(self, capsys):
        """Test debug output for translate_init."""
        va_path = VACASK_DEVICES / "resistor.va"
        if not va_path.exists():
            pytest.skip(f"Model not found: {va_path}")

        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])

        init_fn, metadata = translator.translate_init(params={"R": 1000.0}, debug=True)

        captured = capsys.readouterr()
        assert "Init Parameters" in captured.out
        assert "Name" in captured.out

    def test_debug_mode_eval(self, capsys):
        """Test debug output for translate_eval."""
        va_path = VACASK_DEVICES / "resistor.va"
        if not va_path.exists():
            pytest.skip(f"Model not found: {va_path}")

        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])

        eval_fn, metadata = translator.translate_eval(params={"R": 1000.0}, debug=True)

        captured = capsys.readouterr()
        assert "Eval Parameters" in captured.out
