"""Test compilation of various Verilog-A models"""

import openvaf_py
import time
from pathlib import Path

BASE_PATH = Path("/Users/roberttaylor/Code/ChipFlow/reference/OpenVAF/integration_tests")

# Models to test (path relative to integration_tests)
MODELS = [
    # Simple
    ("RESISTOR/resistor.va", "resistor"),
    ("DIODE/diode.va", "diode"),
    ("DIODE_CMC/diode_cmc.va", "diode_cmc"),
    ("CURRENT_SOURCE/current_source.va", "isrc"),
    ("VCCS/vccs.va", "vccs"),
    ("CCCS/cccs.va", "cccs"),

    # MOSFETs
    ("EKV/ekv.va", "ekv"),
    ("BSIM3/bsim3.va", "bsim3"),
    ("BSIM4/bsim4.va", "bsim4"),
    ("BSIM6/bsim6.va", "bsim6"),
    ("BSIMBULK/bsimbulk.va", "bsimbulk"),
    ("BSIMCMG/bsimcmg.va", "bsimcmg"),
    ("BSIMSOI/bsimsoi.va", "bsimsoi"),

    # PSP
    ("PSP102/psp102.va", "psp102"),
    ("PSP103/psp103.va", "psp103"),
    ("PSP103/juncap200.va", "juncap"),

    # HiSIM
    ("HiSIM2/hisim2.va", "hisim2"),
    ("HiSIMHV/hisimhv.va", "hisimhv"),

    # BJT
    ("HICUML2/hicuml2.va", "hicum"),
    ("MEXTRAM/mextram.va", "mextram"),

    # HEMT
    ("ASMHEMT/asmhemt.va", "asmhemt"),
    ("MVSG_CMC/mvsg_cmc.va", "mvsg"),
]

def count_by_kind(param_kinds):
    """Count parameters by kind"""
    counts = {}
    for kind in param_kinds:
        counts[kind] = counts.get(kind, 0) + 1
    return counts

results = []

print("="*80)
print("OpenVAF Model Compilation Test")
print("="*80)

for va_file, name in MODELS:
    path = BASE_PATH / va_file
    if not path.exists():
        print(f"SKIP {name:15s} - file not found")
        continue

    try:
        start = time.time()
        modules = openvaf_py.compile_va(str(path))
        elapsed = time.time() - start

        if modules:
            m = modules[0]
            kind_counts = count_by_kind(m.param_kinds)

            results.append({
                'name': name,
                'module': m.name,
                'nodes': len(m.nodes),
                'params': sum(1 for k in m.param_kinds if k == 'param'),
                'hidden': sum(1 for k in m.param_kinds if k == 'hidden_state'),
                'cached': m.num_cached_values,
                'residuals': m.num_residuals,
                'jacobian': m.num_jacobian,
                'callbacks': m.get_num_func_calls(),
                'time': elapsed,
            })

            print(f"OK   {name:15s} nodes={len(m.nodes):2d} params={kind_counts.get('param', 0):4d} "
                  f"hidden={kind_counts.get('hidden_state', 0):5d} jac={m.num_jacobian:3d} "
                  f"time={elapsed:.2f}s")
        else:
            print(f"FAIL {name:15s} - no modules returned")

    except Exception as e:
        print(f"ERR  {name:15s} - {e}")

print("\n" + "="*80)
print("Summary Table")
print("="*80)
print(f"{'Model':<15s} {'Module':<20s} {'Nodes':>6s} {'Params':>7s} {'Hidden':>7s} {'Cached':>7s} {'Jac':>5s} {'Time':>6s}")
print("-"*80)

for r in results:
    print(f"{r['name']:<15s} {r['module']:<20s} {r['nodes']:>6d} {r['params']:>7d} "
          f"{r['hidden']:>7d} {r['cached']:>7d} {r['jacobian']:>5d} {r['time']:>5.2f}s")

print("-"*80)
print(f"Total models compiled: {len(results)}")

# Print complexity ranking
print("\n" + "="*80)
print("Complexity Ranking (by hidden states)")
print("="*80)
for r in sorted(results, key=lambda x: -x['hidden'])[:10]:
    print(f"{r['name']:<15s} hidden={r['hidden']:5d} params={r['params']:4d} cached={r['cached']:4d}")
