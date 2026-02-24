"""Diagnose LTE divergence between solvers at early transient steps.

Runs the ring benchmark for a few steps and dumps V_new, V_pred, and LTE
details at each step. Run locally (UMFPACK) and on GPU (cuDSS) to compare.

Usage:
    JAX_PLATFORMS=cpu uv run python scripts/diagnose_lte_divergence.py --sparse
    JAX_PLATFORMS=cuda uv run python scripts/diagnose_lte_divergence.py --sparse
"""

import argparse
import json
import sys

import jax
import jax.numpy as jnp


def run_diagnostic(use_sparse: bool, n_steps: int = 5, output_path: str | None = None):
    from vajax.analysis.engine import CircuitEngine
    from vajax.analysis.transient.adaptive import AdaptiveConfig
    from vajax.benchmarks.registry import get_benchmark

    backend = jax.default_backend()
    solver = (
        "cuDSS"
        if (use_sparse and backend in ("cuda", "gpu"))
        else "UMFPACK"
        if use_sparse
        else "dense"
    )
    print(f"Backend: {backend}, Solver: {solver}, Sparse: {use_sparse}")

    info = get_benchmark("ring")
    engine = CircuitEngine(info.sim_path)
    engine.parse()

    # We need to intercept the while_loop body to capture per-step voltages.
    # The cleanest way: use debug_lte=True which already prints V_new/V_pred,
    # plus add a custom callback that captures the full voltage vector.

    # Step 1: Get the DC solution and initial dt
    config = AdaptiveConfig(debug_steps=True, debug_lte=True)
    engine.prepare(use_sparse=use_sparse, adaptive_config=config)
    strategy = engine._prepared_strategy

    # Get initial state
    V_dc = strategy._cached_init_V0
    n_total = strategy._cached_init_V0_n_total
    print(f"\nCircuit: {n_total} nodes (including ground)")
    print("DC solution (first 15 nodes):")
    for i in range(min(15, len(V_dc))):
        print(f"  V[{i}] = {float(V_dc[i]):.10f}")

    # Step 2: Run for n_steps and capture output
    # The netlist step is 50ps, initial dt = step/4 = 12.5ps
    # Run for n_steps * 12.5ps to get roughly n_steps
    params = getattr(engine, "analysis_params", {})
    dt = params.get("step", 50e-12)
    t_stop = dt * (n_steps + 1)  # A few extra steps

    print(f"\nRunning {n_steps} steps: dt={dt * 1e12:.1f}ps, t_stop={t_stop * 1e12:.1f}ps")
    print("=" * 80)

    # Capture stdout (debug_lte prints to stdout via jax.debug.callback)
    from io import StringIO

    captured = StringIO()
    original_stdout = sys.stdout

    class TeeWriter:
        def __init__(self, orig, buf):
            self.orig = orig
            self.buf = buf

        def write(self, text):
            self.orig.write(text)
            self.buf.write(text)
            return len(text)

        def flush(self):
            self.orig.flush()
            self.buf.flush()

    sys.stdout = TeeWriter(original_stdout, captured)
    try:
        strategy.run(t_stop=t_stop, dt=dt)
    finally:
        sys.stdout = original_stdout

    # Step 3: Parse and save results
    text = captured.getvalue()

    # Extract per-step voltage data from debug_lte output
    steps_data = []
    current_step = None
    import re

    for line in text.split("\n"):
        # Match LTE Debug header
        m = re.match(r"=== LTE Debug \(step (\d+), dt=([\d.]+)ps\) ===", line)
        if m:
            current_step = {
                "step": int(m.group(1)),
                "dt_ps": float(m.group(2)),
                "nodes": [],
            }
            steps_data.append(current_step)
            continue

        # Match node LTE detail
        m = re.match(
            r"\s+Node (\d+): V_new=([\d.e+-]+)V, V_pred=([\d.e+-]+)V, diff=([\d.e+-]+)V",
            line,
        )
        if m and current_step is not None:
            current_step["nodes"].append(
                {
                    "node": int(m.group(1)),
                    "V_new": float(m.group(2)),
                    "V_pred": float(m.group(3)),
                    "diff": float(m.group(4)),
                }
            )
            continue

        # Match tolerance line
        m = re.match(
            r"\s+tol=([\d.e+-]+) \(reltol\*max\|V\|=([\d.e+-]+), abstol=([\d.e+-]+)\)",
            line,
        )
        if m and current_step is not None and current_step["nodes"]:
            current_step["nodes"][-1]["tol"] = float(m.group(1))
            current_step["nodes"][-1]["reltol_contrib"] = float(m.group(2))
            continue

        # Match LTE/norm line
        m = re.match(r"\s+LTE=([\d.e+-]+), norm_LTE=([\d.e+-]+)", line)
        if m and current_step is not None and current_step["nodes"]:
            current_step["nodes"][-1]["lte"] = float(m.group(1))
            current_step["nodes"][-1]["norm_lte"] = float(m.group(2))
            continue

        # Match max normalized LTE
        m = re.match(r"Max normalized LTE \(lte_norm\): ([\d.e+-]+)", line)
        if m and current_step is not None:
            current_step["lte_norm"] = float(m.group(1))
            continue

        # Match error coefficients
        m = re.match(
            r"Error coefficients: pred_err=([\d.e+-]+), integ_err=([\d.e+-]+), scale=([\d.e+-]+)",
            line,
        )
        if m and current_step is not None:
            current_step["pred_err"] = float(m.group(1))
            current_step["integ_err"] = float(m.group(2))
            current_step["scale"] = float(m.group(3))
            continue

    # Print summary
    print("\n" + "=" * 80)
    print(f"Captured {len(steps_data)} steps with LTE debug")
    for sd in steps_data[:n_steps]:
        print(f"\n--- Step {sd['step']} (dt={sd['dt_ps']:.4f}ps) ---")
        print(f"  lte_norm={sd.get('lte_norm', 'N/A')}")
        print(f"  scale={sd.get('scale', 'N/A')}")
        for nd in sd.get("nodes", [])[:3]:
            print(
                f"  Node {nd['node']}: V_new={nd['V_new']:.10f} V_pred={nd['V_pred']:.10f} "
                f"diff={nd['diff']:.6e} tol={nd.get('tol', 'N/A')}"
            )

    # Save to JSON
    output = {
        "backend": backend,
        "solver": solver,
        "use_sparse": use_sparse,
        "n_total": n_total,
        "dc_solution": [float(x) for x in V_dc],
        "steps": steps_data[:n_steps],
    }

    if output_path:
        from pathlib import Path

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved diagnostic to {output_path}")
    else:
        # Print JSON to stderr for capture
        print(f"\nJSON output ({len(steps_data)} steps) available with --output flag")


def compare_diagnostics(path_a: str, path_b: str):
    """Compare two diagnostic dumps side-by-side."""
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    print(f"A: {a['solver']} ({a['backend']})")
    print(f"B: {b['solver']} ({b['backend']})")
    print(f"Nodes: A={a['n_total']}, B={b['n_total']}")

    # Compare DC solutions
    dc_a = jnp.array(a["dc_solution"])
    dc_b = jnp.array(b["dc_solution"])
    dc_diff = jnp.abs(dc_a - dc_b)
    print(f"\nDC solution max|diff|: {float(jnp.max(dc_diff)):.2e}")

    # Compare per-step
    n = min(len(a["steps"]), len(b["steps"]))
    for i in range(n):
        sa = a["steps"][i]
        sb = b["steps"][i]
        print(f"\n{'=' * 70}")
        print(f"Step {sa['step']} (A) vs Step {sb['step']} (B)")
        print(f"  dt: A={sa['dt_ps']:.4f}ps  B={sb['dt_ps']:.4f}ps")
        print(f"  lte_norm: A={sa.get('lte_norm', 'N/A')}  B={sb.get('lte_norm', 'N/A')}")
        print(f"  scale: A={sa.get('scale', 'N/A')}  B={sb.get('scale', 'N/A')}")

        nodes_a = {nd["node"]: nd for nd in sa.get("nodes", [])}
        nodes_b = {nd["node"]: nd for nd in sb.get("nodes", [])}
        all_nodes = sorted(set(nodes_a.keys()) | set(nodes_b.keys()))

        for node in all_nodes[:5]:
            na = nodes_a.get(node, {})
            nb = nodes_b.get(node, {})
            v_new_a = na.get("V_new", float("nan"))
            v_new_b = nb.get("V_new", float("nan"))
            v_pred_a = na.get("V_pred", float("nan"))
            v_pred_b = nb.get("V_pred", float("nan"))
            tol_a = na.get("tol", float("nan"))
            tol_b = nb.get("tol", float("nan"))
            print(f"  Node {node}:")
            print(
                f"    V_new:  A={v_new_a:.10f}  B={v_new_b:.10f}  diff={abs(v_new_a - v_new_b):.4e}"
            )
            print(
                f"    V_pred: A={v_pred_a:.10f}  B={v_pred_b:.10f}  diff={abs(v_pred_a - v_pred_b):.4e}"
            )
            print(f"    tol:    A={tol_a:.4e}  B={tol_b:.4e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose LTE divergence between solvers")
    parser.add_argument("--sparse", action="store_true", help="Use sparse solver")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps to capture")
    parser.add_argument("--output", type=str, help="Save diagnostic JSON to this path")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("A", "B"),
        help="Compare two diagnostic JSON files",
    )
    args = parser.parse_args()

    if args.compare:
        compare_diagnostics(args.compare[0], args.compare[1])
    else:
        run_diagnostic(use_sparse=args.sparse, n_steps=args.steps, output_path=args.output)
