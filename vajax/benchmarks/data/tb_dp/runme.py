#!/usr/bin/env python3
"""Convert and run tb_dp tests.

This script:
1. Converts ngspice netlists to VA-JAX format using netlist_converter
2. Adds PSP model load and control block
3. Optionally runs the simulation

Usage:
    python runme.py                    # Convert and list all available tests
    python runme.py tb_dp512x8_klu     # Convert specific test
    python runme.py --run tb_dp512x8_klu  # Convert and run specific test
    python runme.py --all              # Convert all tests
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Paths
TEST_DIR = Path(__file__).parent
PSP_MODEL_PATH = "./psp103_ihp.osdi"

# All available test cases
TEST_CASES = [
    "tb_dp512x8_klu",
    "tb_dp512x8_as_klu",
    "tb_dp512x8_sparse",
    "tb_dp512x32_klu",
    "tb_dp512x32_as_klu",
    "tb_dp512x32_klu_rshunt",
    "tb_dp512x32_klu_trapezoid",
    "tb_dp512x32_klu_trapezoid_rshunt",
]

# IHP PDK path - check common locations
IHP_PDK_CANDIDATES = [
    Path(os.environ.get("IHP_PDK_ROOT", "")) / "libs.tech/ngspice/models",
    Path.home() / "Code/ChipFlow/PDK/IHP-Open-PDK/ihp-sg13g2/libs.tech/ngspice/models",
    Path("/foss/pdks/ihp-sg13g2/libs.tech/ngspice/models"),
]


def find_ihp_pdk():
    """Find the IHP PDK models directory."""
    for candidate in IHP_PDK_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def convert_netlist(test_name: str) -> bool:
    """Convert ngspice netlist to VA-JAX format."""
    spice_file = TEST_DIR / f"{test_name}.spi"
    sim_file = TEST_DIR / f"{test_name}.sim"

    if not spice_file.exists():
        print(f"Error: {spice_file} not found")
        return False

    print(f"Converting {spice_file.name} to VA-JAX format...")

    ihp_models = find_ihp_pdk()
    if not ihp_models:
        print("Warning: IHP PDK not found. Trying conversion anyway...")
        ihp_models = IHP_PDK_CANDIDATES[0]

    # Create temp file with fixed paths
    with tempfile.NamedTemporaryFile(mode='w', suffix='.spi', delete=False) as tmp:
        content = spice_file.read_text()
        # Replace container path with local PDK path
        content = content.replace(
            "/foss/pdks/ihp-sg13g2/libs.tech/ngspice/models",
            str(ihp_models)
        )
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Import the converter from vajax
        from vajax.netlist_converter.ng2vclib import dfl
        from vajax.netlist_converter.ng2vclib.converter import Converter

        # Build configuration
        cfg = dfl.default_config()
        cfg["sourcepath"] = [".", str(ihp_models)]
        cfg["read_depth"] = None
        cfg["process_depth"] = None
        cfg["output_depth"] = None
        cfg["all_models"] = False

        # Create converter with ngspice dialect
        converter = Converter(cfg, dialect="ngspice")
        converter.convert(tmp_path, str(sim_file))

        print(f"Generated {sim_file.name}")
        return True
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        Path(tmp_path).unlink()


def add_psp_load_and_control(test_name: str) -> bool:
    """Add PSP OSDI load and control block to the converted netlist."""
    sim_file = TEST_DIR / f"{test_name}.sim"

    print("Adding PSP load and control block...")

    content = sim_file.read_text()

    # Add PSP load before capacitor load
    content = content.replace(
        'load"spice/capacitor.osdi"',
        f'load"{PSP_MODEL_PATH}"\nload"spice/capacitor.osdi"'
    )

    # Determine appropriate timestep based on test case
    if "trapezoid" in test_name:
        # Trapezoid integration uses larger timesteps
        step = "5e-10"
    else:
        step = "1e-10"

    # Add control block at the end
    control_block = f"""
// Control block
control
  analysis tran1 tran step={step} stop=3e-8
  save all
endc
"""
    content += control_block

    sim_file.write_text(content)
    print("Added PSP load and control block")
    return True


def run_simulation(test_name: str) -> bool:
    """Run the VACASK simulation."""
    sim_file = TEST_DIR / f"{test_name}.sim"

    # Check common VACASK build locations
    vacask_candidates = [
        # VA-JAX vendor submodule
        TEST_DIR.parent.parent.parent.parent / "vendor" / "VACASK" / "build" / "simulator" / "vacask",
        # Environment variable
        Path(os.environ.get("VACASK_BIN", "")),
    ]

    vacask = None
    for candidate in vacask_candidates:
        if candidate.exists():
            vacask = candidate
            break

    if not vacask:
        print("Error: VACASK executable not found")
        print("Set VACASK_BIN environment variable to the vacask binary path")
        return False

    print(f"Running simulation {sim_file.name}...")
    print("Note: This may take a while, especially for 512x32 variants.")

    result = subprocess.run(
        [str(vacask), str(sim_file)],
        cwd=TEST_DIR,
    )

    return result.returncode == 0


def convert_test(test_name: str, run: bool = False) -> bool:
    """Convert a single test case."""
    if test_name not in TEST_CASES:
        print(f"Unknown test case: {test_name}")
        print(f"Available: {', '.join(TEST_CASES)}")
        return False

    if not convert_netlist(test_name):
        return False

    if not add_psp_load_and_control(test_name):
        return False

    print(f"\nSuccess! Generated {test_name}.sim")

    if run:
        return run_simulation(test_name)
    else:
        print(f"\nTo run the simulation:")
        print(f"  cd {TEST_DIR}")
        print(f"  vacask {test_name}.sim")
        print("\nNote: Simulation is very slow in debug builds.")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert and run tb_dp dual-port SRAM tests"
    )
    parser.add_argument(
        "test_name",
        nargs="?",
        help="Test case name (e.g., tb_dp512x8_klu)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all test cases",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the simulation after conversion",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available test cases",
    )

    args = parser.parse_args()

    ihp_pdk = find_ihp_pdk()
    if not ihp_pdk:
        print("Warning: IHP PDK models not found")
        print("Searched locations:")
        for loc in IHP_PDK_CANDIDATES:
            print(f"  - {loc}")
        print("Set IHP_PDK_ROOT environment variable or adjust paths in script")

    if args.list:
        print("Available test cases:")
        for tc in TEST_CASES:
            spi = TEST_DIR / f"{tc}.spi"
            sim = TEST_DIR / f"{tc}.sim"
            status = "✓" if sim.exists() else "○"
            exists = "found" if spi.exists() else "missing"
            print(f"  {status} {tc} ({exists})")
        return 0

    if args.all:
        success = True
        for tc in TEST_CASES:
            print(f"\n{'='*60}")
            print(f"Processing {tc}")
            print('='*60)
            if not convert_test(tc, run=False):
                success = False
        return 0 if success else 1

    if args.test_name:
        return 0 if convert_test(args.test_name, run=args.run) else 1

    # Default: list available tests
    print("tb_dp Dual-Port SRAM Test Suite")
    print("="*40)
    print("\nAvailable test cases:")
    for tc in TEST_CASES:
        spi = TEST_DIR / f"{tc}.spi"
        print(f"  - {tc}")
    print(f"\nUsage: python {Path(__file__).name} [--all | test_name] [--run]")
    print("\nExamples:")
    print(f"  python {Path(__file__).name} tb_dp512x8_klu      # Convert one test")
    print(f"  python {Path(__file__).name} --all               # Convert all tests")
    print(f"  python {Path(__file__).name} --run tb_dp512x8_klu # Convert and run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
