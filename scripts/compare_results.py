#!/usr/bin/env python3
"""Compare JAX-SPICE vs VACASK simulation results."""

import subprocess
import sys
import tempfile
import re
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.numpy as jnp

# Import jax_spice first to auto-configure precision based on backend
from jax_spice.analysis import CircuitEngine


def parse_value_with_suffix(value_str: str) -> float:
    """Parse a numeric value that may have an SI suffix (e.g., '1u' -> 1e-6)."""
    suffixes = {
        'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3,
        'k': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12
    }
    value_str = value_str.strip()
    if not value_str:
        return 0.0

    # Check for suffix
    if value_str[-1] in suffixes:
        return float(value_str[:-1]) * suffixes[value_str[-1]]
    else:
        return float(value_str)


def run_vacask_short_tran(sim_path: Path, vacask_bin: Path, steps: int = 5) -> dict:
    """Run VACASK with a short transient and get final voltages."""
    content = sim_path.read_text()

    # Get step size from original analysis - match "step=VALUE" specifically
    step_match = re.search(r'\bstep\s*=\s*([\d.e+-]+[a-zA-Z]?)', content, re.IGNORECASE)
    step = parse_value_with_suffix(step_match.group(1)) if step_match else 1e-6

    # Compute short stop time
    stop = step * steps

    # Replace analysis with short transient + NR debug for solution output
    content = re.sub(
        r'analysis\s+\w+\s+tran[^\n]*',
        f'analysis tran1 tran step={step:.2e} stop={stop:.2e}',
        content,
        flags=re.IGNORECASE
    )

    # Add nr_debug option if not present
    if 'nr_debug' not in content.lower():
        content = content.replace('control\n', 'control\n  options nr_debug=2\n')
        content = content.replace('control\r\n', 'control\r\n  options nr_debug=2\r\n')

    # Remove postprocess to avoid Python errors
    content = re.sub(r'postprocess\([^)]+\)', '', content)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.sim', delete=False, dir=sim_path.parent.resolve()) as f:
        f.write(content)
        temp_sim = Path(f.name).resolve()

    try:
        result = subprocess.run(
            [str(vacask_bin.resolve()), str(temp_sim)],
            capture_output=True,
            text=True,
            cwd=sim_path.parent.resolve(),
            timeout=120
        )
        output = result.stdout + result.stderr
    finally:
        temp_sim.unlink()

    # Parse final solution from NR debug output
    # Format: "New solution in iteration N\n  node : value\n  ..."
    # We want the LAST "New solution" block (final timestep result)
    voltages = {}
    lines = output.split('\n')

    # Find all solution blocks and keep the last one
    solution_blocks = []
    i = 0
    while i < len(lines):
        if 'New solution in iteration' in lines[i]:
            block = {}
            for j in range(i + 1, len(lines)):
                sol_line = lines[j]
                match = re.match(r'\s*(\S+)\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', sol_line)
                if match:
                    node_name = match.group(1)
                    value = float(match.group(2))
                    block[node_name] = value
                elif sol_line.strip() and not sol_line.startswith(' '):
                    break
            if block:
                solution_blocks.append(block)
            i = j
        else:
            i += 1

    # Use the last solution block
    if solution_blocks:
        voltages = solution_blocks[-1]

    return voltages, output, step


def run_jaxspice(sim_path: Path, steps: int = 5) -> dict:
    """Run JAX-SPICE and return final voltages."""
    engine = CircuitEngine(sim_path)
    engine.parse()

    # Get step from analysis params
    step = engine.analysis_params.get('step', 1e-6)
    stop = step * steps

    result = engine.run_transient(
        t_stop=stop,
        dt=step,
        max_steps=steps + 10,  # Extra margin
        use_sparse=False
    )

    # Get final voltages
    node_voltages = {}
    index_to_name = {v: k for k, v in engine.node_names.items()}

    for node_idx, voltage_array in result.voltages.items():
        node_name = index_to_name.get(node_idx, str(node_idx))
        final_v = float(voltage_array[-1]) if len(voltage_array) > 0 else 0.0
        node_voltages[node_name] = final_v

    return node_voltages, engine, step


def compare_benchmarks():
    """Compare all benchmarks."""
    vacask_bin = Path("vendor/VACASK/build/simulator/vacask")
    if not vacask_bin.exists():
        print(f"VACASK binary not found at {vacask_bin}")
        return

    benchmarks = {
        'rc': {
            'sim': Path("vendor/VACASK/benchmark/rc/vacask/runme.sim"),
            'steps': 10,
            'nodes_to_compare': ['1', '2'],
            'tolerance': 0.01,  # 1% relative error
        },
        'graetz': {
            'sim': Path("vendor/VACASK/benchmark/graetz/vacask/runme.sim"),
            'steps': 10,
            'nodes_to_compare': ['inn', 'inp', 'outn', 'outp'],
            'tolerance': 0.05,  # 5% for diode circuit
        },
        'ring': {
            'sim': Path("vendor/VACASK/benchmark/ring/vacask/runme.sim"),
            'steps': 6,
            'nodes_to_compare': ['1', '2', '3', 'vdd'],
            'tolerance': 0.10,  # 10% for complex PSP103 circuit
        },
    }

    print("=" * 70)
    print("JAX-SPICE vs VACASK Results Comparison")
    print("=" * 70)
    print()

    results = []

    for name, config in benchmarks.items():
        sim_path = config['sim']
        if not sim_path.exists():
            print(f"Skipping {name}: {sim_path} not found")
            continue

        print(f"--- {name} benchmark ---")
        print(f"Sim file: {sim_path}")

        # Run JAX-SPICE
        print("  Running JAX-SPICE...", end=" ", flush=True)
        try:
            jax_voltages, engine, jax_step = run_jaxspice(sim_path, config['steps'])
            print(f"done ({len(jax_voltages)} nodes, dt={jax_step:.2e}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Run VACASK
        print("  Running VACASK...", end=" ", flush=True)
        try:
            vacask_voltages, vacask_output, vac_step = run_vacask_short_tran(
                sim_path, vacask_bin, config['steps']
            )
            if not vacask_voltages:
                print(f"FAILED: No voltages parsed")
                print(f"  Output: {vacask_output[:500]}")
                continue
            print(f"done ({len(vacask_voltages)} nodes)")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        # Compare key nodes
        print()
        print(f"  {'Node':<15} {'JAX-SPICE':>12} {'VACASK':>12} {'Diff':>12} {'Rel Err':>10}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

        max_rel_err = 0
        comparisons = []

        for node in config['nodes_to_compare']:
            # Try different case variants
            jax_v = jax_voltages.get(node) or jax_voltages.get(node.lower()) or jax_voltages.get(node.upper())
            vac_v = vacask_voltages.get(node) or vacask_voltages.get(node.lower()) or vacask_voltages.get(node.upper())

            if jax_v is not None and vac_v is not None:
                diff = jax_v - vac_v
                ref = max(abs(vac_v), 1e-9)  # Avoid division by zero
                rel_err = abs(diff) / ref
                max_rel_err = max(max_rel_err, rel_err)

                status = "✓" if rel_err < config['tolerance'] else "✗"
                print(f"  {node:<15} {jax_v:>12.6f} {vac_v:>12.6f} {diff:>+12.2e} {rel_err:>9.2%} {status}")
                comparisons.append((node, jax_v, vac_v, rel_err))
            else:
                jax_str = f"{jax_v:.6f}" if jax_v is not None else "N/A"
                vac_str = f"{vac_v:.6f}" if vac_v is not None else "N/A"
                print(f"  {node:<15} {jax_str:>12} {vac_str:>12} {'N/A':>12} {'N/A':>10}")

        # Summary for this benchmark
        print()
        tolerance = config['tolerance']
        if max_rel_err < tolerance:
            print(f"  ✓ PASS: Max relative error = {max_rel_err:.2%} (< {tolerance:.0%})")
            results.append((name, 'PASS', max_rel_err))
        elif max_rel_err < tolerance * 2:
            print(f"  ~ WARN: Max relative error = {max_rel_err:.2%}")
            results.append((name, 'WARN', max_rel_err))
        else:
            print(f"  ✗ FAIL: Max relative error = {max_rel_err:.2%}")
            results.append((name, 'FAIL', max_rel_err))
        print()

    # Final summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"{'Benchmark':<15} {'Status':<10} {'Max Rel Error':<15}")
    print(f"{'-'*15} {'-'*10} {'-'*15}")
    for name, status, err in results:
        print(f"{name:<15} {status:<10} {err:.4%}")

    passed = sum(1 for _, s, _ in results if s in ('PASS', 'WARN'))
    total = len(results)
    print()
    print(f"Passed: {passed}/{total}")


if __name__ == "__main__":
    compare_benchmarks()
