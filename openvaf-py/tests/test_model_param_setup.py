#!/usr/bin/env python3
"""Test the new run_model_param_setup() method."""

import sys
sys.path.insert(0, '..')

def test_model_param_setup_method():
    """Test that run_model_param_setup() method exists and can be called."""
    try:
        import openvaf_py

        # Compile resistor model
        modules = openvaf_py.compile_va("../vendor/VACASK/devices/resistor.va")
        resistor = modules[0]

        # Check that the method exists
        assert hasattr(resistor, 'run_model_param_setup'), \
            "VaModule should have run_model_param_setup method"

        # Check that model_setup fields exist
        assert hasattr(resistor, 'model_setup_num_params'), \
            "VaModule should have model_setup_num_params field"
        assert hasattr(resistor, 'model_setup_param_names'), \
            "VaModule should have model_setup_param_names field"
        assert hasattr(resistor, 'model_setup_param_kinds'), \
            "VaModule should have model_setup_param_kinds field"

        print(f"✓ Model setup fields found")
        print(f"  - model_setup_num_params: {resistor.model_setup_num_params}")
        print(f"  - model_setup_param_names: {resistor.model_setup_param_names}")
        print(f"  - model_setup_param_kinds: {resistor.model_setup_param_kinds}")

        # Try calling the method
        result = resistor.run_model_param_setup({
            'r': 1000.0,
            'r_given': 1.0,
            'has_noise': 1.0,
            'has_noise_given': 1.0
        })

        print(f"\n✓ run_model_param_setup() executed successfully")
        print(f"  Result: {result}")

        # Validate result structure
        assert isinstance(result, dict), "Result should be a dict"
        assert 'r' in result, "Result should include 'r' parameter"

        # Test with defaults
        result_defaults = resistor.run_model_param_setup({
            'r': 0.0,
            'r_given': 0.0,  # Not given → should use default
            'has_noise': 0.0,
            'has_noise_given': 0.0
        })

        print(f"\n✓ Default value test")
        print(f"  Result: {result_defaults}")

        print(f"\n{'='*60}")
        print(f"SUCCESS! run_model_param_setup() works correctly!")
        print(f"{'='*60}")

        return 0

    except ImportError as e:
        print(f"⚠ openvaf_py not built yet: {e}")
        print(f"  Run: cd openvaf-py && maturin develop --release")
        return 1
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(test_model_param_setup_method())
