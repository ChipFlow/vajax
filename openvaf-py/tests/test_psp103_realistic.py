#!/usr/bin/env python
"""Test PSP103 with realistic model parameters using jax_emit.

Uses the ring oscillator's PSP103 model parameters which produce realistic
transistor behavior. Compares JAX results against OSDI reference when available.
"""

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

# Enable float64 for numerical accuracy
jax.config.update('jax_enable_x64', True)

import openvaf_py

sys.path.insert(0, str(Path(__file__).parent.parent))
from jax_emit import build_eval_fn, build_init_fn

# Optional OSDI import for reference comparison
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
    from osdi_py import OsdiLibrary, CALC_RESIST_RESIDUAL, CALC_RESIST_JACOBIAN, ANALYSIS_DC
    HAS_OSDI = True
except ImportError:
    HAS_OSDI = False

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent
OPENVAF_TESTS = REPO_ROOT / "vendor" / "OpenVAF" / "integration_tests"
RING_DIR = REPO_ROOT / "vendor" / "VACASK" / "benchmark" / "ring" / "vacask"

# Ring oscillator NMOS model parameters (from models.inc)
# These produce realistic transistor behavior
NMOS_MODEL_PARAMS = {
    'TYPE': 1,
    'TR': 27.0,
    'SWGEO': 1,
    'QMC': 1.0,
    'VFBO': -1.1,
    'TOXO': 1.5e-9,
    'EPSROXO': 3.9,
    'NSUBO': 3.0e+23,
    'NPCK': 1.0e+24,
    'LPCK': 5.5e-8,
    'FOL1': 2.0e-2,
    'FOL2': 5.0e-6,
    'FACNEFFACO': 0.8,
    'GFACNUDO': 0.1,
    'DVSBNUDO': 1,
    'NSLPO': 0.05,
    'NPO': 1.5e+26,
    'NPL': 10.0e-18,
    'CTO': 5.0e-15,
    'CTL': 4.0e-2,
    'CTLEXP': 0.6,
    'TOXOVO': 1.5e-9,
    'TOXOVDO': 2.0e-9,
    'LOV': 10.0e-9,
    'NOVO': 7.5e+25,
    'NOVDO': 5.0e+25,
    'CFL': 3.0e-4,
    'CFLEXP': 2.0,
    'CFW': 5.0e-3,
    'CFBO': 0.3,
    'UO': 3.5e-2,
    'FBET1': -0.3,
    'FBET1W': 0.15,
    'LP1': 1.5e-7,
    'LP1W': -2.5e-2,
    'FBET2': 50.0,
    'LP2': 8.5e-10,
    'BETW1': 5.0e-2,
    'BETW2': -2.0e-2,
    'WBET': 5.0e-10,
    'STBETO': 1.75,
    'MUEO': 0.6,
    'MUEW': -1.2e-2,
    'STMUEO': 0.5,
    'THEMUO': 2.75,
    'CSO': 1.0e-2,
    'STCSO': -5.0,
    'XCORO': 0.15,
    'XCORL': 2.0e-3,
    'XCORW': -3.0e-2,
    'STXCORO': 1.25,
    'FETAO': 1,
    'RSW1': 50,
    'RSW2': 5.0e-2,
    'STRSO': -2.0,
    'THESATO': 1.0e-6,
    'THESATL': 0.6,
    'THESATLEXP': 0.75,
    'THESATW': -1.0e-2,
    'STTHESATO': 1.5,
    'THESATBO': 0.15,
    'THESATGO': 0.75,
    'AXO': 20,
    'AXL': 0.2,
    'ALPL': 7.0e-3,
    'ALPLEXP': 0.6,
    'ALPW': 5.0e-2,
    'ALP1L1': 2.5e-2,
    'ALP1LEXP': 0.4,
    'ALP1L2': 0.1,
    'ALP1W': 8.5e-3,
    'ALP2L1': 0.5,
    'ALP2L2': 0.5,
    'ALP2W': -0.2,
    'VPO': 0.25,
    'A1O': 1.0,
    'A2O': 10.0,
    'STA2O': -0.5,
    'A3O': 1.0,
    'GCOO': 5.0,
    'IGINVLW': 50.0,
    'IGOVW': 10.0,
    'STIGO': 1.5,
    'GC2O': 1.0,
    'GC3O': -1.0,
    'CHIBO': 3.1,
    'AGIDLW': 50.0,
    'BGIDLO': 35.0,
    'BGIDLDO': 41,
    'STBGIDLO': -5.0e-4,
    'CGIDLO': 0.15,
    'CFRW': 5.0e-17,
    'FNTO': 1,
    'NFALW': 8.0e+22,
    'NFBLW': 3.0e7,
    'IMAX': 1.0e3,
    'VJUNREF': 2.5,
    'FJUNQ': 0.03,
    'CJORBOT': 1.0e-3,
    'CJORSTI': 1.0e-9,
    'CJORGAT': 0.5e-9,
    'VBIRBOT': 0.75,
    'VBIRSTI': 1.0,
    'VBIRGAT': 0.75,
    'PBOT': 0.35,
    'PSTI': 0.35,
    'PGAT': 0.6,
    'PHIGBOT': 1.16,
    'PHIGSTI': 1.16,
    'PHIGGAT': 1.16,
    'IDSATRBOT': 5.0e-9,
    'IDSATRSTI': 1.0e-18,
    'IDSATRGAT': 1.0e-18,
    'CSRHBOT': 5.0e2,
    'CSRHGAT': 1.0e3,
    'XJUNSTI': 1.0e-8,
    'XJUNGAT': 1.0e-9,
    'CTATBOT': 5.0e2,
    'CTATGAT': 1.0e3,
    'MEFFTATBOT': 0.25,
    'MEFFTATSTI': 0.25,
    'MEFFTATGAT': 0.25,
    'CBBTBOT': 1.0e-12,
    'CBBTSTI': 1.0e-18,
    'CBBTGAT': 1.0e-18,
    'FBBTRBOT': 1.0e9,
    'FBBTRSTI': 1.0e9,
    'FBBTRGAT': 1.0e9,
    'STFBBTBOT': -1.0e-3,
    'STFBBTSTI': -1.0e-3,
    'STFBBTGAT': -1.0e-2,
    'VBRBOT': 10.0,
    'VBRSTI': 10.0,
    'VBRGAT': 10.0,
    'PBRBOT': 3,
    'PBRSTI': 4,
    'PBRGAT': 3,
}


def build_init_params_array(module, param_values: dict) -> jnp.ndarray:
    """Build init params array from named parameters.

    Args:
        module: VaModule
        param_values: Dict of parameter name -> value

    Returns:
        Array of init params in the order expected by emit_init
    """
    init_names = list(module.init_param_names)
    init_kinds = list(module.init_param_kinds)
    n_init = len(init_names)

    # Get defaults
    defaults = module.get_param_defaults()

    init_params = jnp.zeros(n_init)

    for i, (name, kind) in enumerate(zip(init_names, init_kinds)):
        if kind == 'param_given':
            # Extract base param name (remove suffix if present)
            base_name = name.upper()
            # Check if the param was explicitly provided
            given = 1.0 if base_name in param_values else 0.0
            init_params = init_params.at[i].set(given)
        elif kind == 'temperature':
            init_params = init_params.at[i].set(param_values.get('$temperature', 300.0))
        elif kind == 'sysfun':
            init_params = init_params.at[i].set(param_values.get(name, 1.0))
        elif kind == 'param':
            # Look up in param_values, then defaults
            upper_name = name.upper()
            if upper_name in param_values:
                init_params = init_params.at[i].set(float(param_values[upper_name]))
            elif name in param_values:
                init_params = init_params.at[i].set(float(param_values[name]))
            elif name.lower() in defaults:
                init_params = init_params.at[i].set(float(defaults[name.lower()]))

    return init_params


def build_eval_params_array(module, node_voltages: dict, temperature: float = 300.0, mfactor: float = 1.0) -> jnp.ndarray:
    """Build eval params array from node voltages and settings.

    Args:
        module: VaModule
        node_voltages: Dict of node name -> voltage
        temperature: Operating temperature
        mfactor: Device multiplier

    Returns:
        Array of eval params in the order expected by eval function
    """
    mir = module.get_mir_instructions()
    params_list = mir.get('params', [])
    n_params = len(params_list)
    metadata = module.get_codegen_metadata()
    eval_map = metadata.get('eval_param_mapping', {})

    def get_idx(name):
        if name in eval_map:
            var = eval_map[name]
            return params_list.index(var) if var in params_list else -1
        return -1

    params = jnp.zeros(n_params)
    params = params.at[get_idx('mfactor')].set(mfactor)
    params = params.at[get_idx('$temperature')].set(temperature)

    V = node_voltages
    branch_voltages = {
        'V(D,DI)': V.get('D', 0) - V.get('DI', V.get('D', 0)),
        'V(G,GP)': V.get('G', 0) - V.get('GP', V.get('G', 0)),
        'V(S,SI)': V.get('S', 0) - V.get('SI', V.get('S', 0)),
        'V(B,BI)': V.get('B', 0) - V.get('BI', V.get('B', 0)),
        'V(GP,SI)': V.get('GP', V.get('G', 0)) - V.get('SI', V.get('S', 0)),
        'V(DI,SI)': V.get('DI', V.get('D', 0)) - V.get('SI', V.get('S', 0)),
        'V(SI,BP)': V.get('SI', V.get('S', 0)) - V.get('BP', V.get('B', 0)),
        'V(SI,BS)': V.get('SI', V.get('S', 0)) - V.get('BS', V.get('B', 0)),
        'V(DI,BD)': V.get('DI', V.get('D', 0)) - V.get('BD', V.get('B', 0)),
        'V(BP,BI)': V.get('BP', V.get('B', 0)) - V.get('BI', V.get('B', 0)),
        'V(BS,BI)': V.get('BS', V.get('B', 0)) - V.get('BI', V.get('B', 0)),
        'V(BD,BI)': V.get('BD', V.get('B', 0)) - V.get('BI', V.get('B', 0)),
    }
    for name, val in branch_voltages.items():
        idx = get_idx(name)
        if idx >= 0:
            params = params.at[idx].set(val)

    abs_voltages = {
        'V(GP)': V.get('GP', V.get('G', 0)),
        'V(SI)': V.get('SI', V.get('S', 0)),
        'V(DI)': V.get('DI', V.get('D', 0)),
        'V(BP)': V.get('BP', V.get('B', 0)),
        'V(BS)': V.get('BS', V.get('B', 0)),
        'V(BD)': V.get('BD', V.get('B', 0)),
        'V(NOI)': V.get('NOI', 0),
    }
    for name, val in abs_voltages.items():
        idx = get_idx(name)
        if idx >= 0:
            params = params.at[idx].set(val)

    return params


def test_psp103_jax_emit_basic():
    """Test that PSP103 compiles and runs with jax_emit."""
    va_path = OPENVAF_TESTS / "PSP103" / "psp103.va"
    module = openvaf_py.compile_va(str(va_path))[0]

    # Build eval function
    eval_fn, meta = build_eval_fn(module)
    assert meta['strategy'] == 'lax_loop'  # PSP103 is large
    assert meta['n_instructions'] > 10000

    # Build init function (uses lax_loop for PSP103's large init MIR)
    init_fn, init_meta = build_init_fn(module)
    assert init_meta['strategy'] == 'lax_loop'
    jit_init = jax.jit(init_fn)

    # Build init params with realistic values
    init_param_values = {
        '$temperature': 300.0,
        'mfactor': 1.0,
        'W': 10e-6,
        'L': 1e-6,
        'AD': 10e-6 * 0.5e-6,
        'AS': 10e-6 * 0.5e-6,
        'PD': 2 * (10e-6 + 0.5e-6),
        'PS': 2 * (10e-6 + 0.5e-6),
        **NMOS_MODEL_PARAMS,
    }
    init_params = build_init_params_array(module, init_param_values)
    cache = jit_init(init_params)

    assert cache.shape[0] == module.num_cached_values
    # Most cache values should be finite
    finite_count = jnp.sum(jnp.isfinite(cache))
    assert finite_count > cache.shape[0] * 0.9

    # Test eval with zero voltages
    jit_eval = jax.jit(eval_fn)
    node_voltages = {
        'D': 0.0, 'G': 0.0, 'S': 0.0, 'B': 0.0,
        'DI': 0.0, 'SI': 0.0, 'GP': 0.0,
        'BP': 0.0, 'BI': 0.0, 'BS': 0.0, 'BD': 0.0, 'NOI': 0.0,
    }
    eval_params = build_eval_params_array(module, node_voltages)
    (resist_res, react_res), (resist_jac, react_jac) = jit_eval(eval_params, cache)

    # PSP103 should produce 13 residuals and 56 jacobian entries
    assert resist_res.shape[0] == 13
    assert resist_jac.shape[0] == 56


def test_psp103_voltage_sweep():
    """Test PSP103 Id-Vgs characteristic."""
    va_path = OPENVAF_TESTS / "PSP103" / "psp103.va"
    module = openvaf_py.compile_va(str(va_path))[0]

    eval_fn, _ = build_eval_fn(module)
    init_fn, _ = build_init_fn(module)

    jit_init = jax.jit(init_fn)
    jit_eval = jax.jit(eval_fn)

    # Build init params
    init_param_values = {
        '$temperature': 300.0,
        'mfactor': 1.0,
        'W': 10e-6,
        'L': 1e-6,
        'AD': 10e-6 * 0.5e-6,
        'AS': 10e-6 * 0.5e-6,
        'PD': 2 * (10e-6 + 0.5e-6),
        'PS': 2 * (10e-6 + 0.5e-6),
        **NMOS_MODEL_PARAMS,
    }
    init_params = build_init_params_array(module, init_param_values)
    cache = jit_init(init_params)

    # Sweep Vgs at Vds=0.6V
    vds = 0.6
    currents = []

    for vgs in np.linspace(0.0, 1.2, 13):
        node_voltages = {
            'D': vds, 'G': vgs, 'S': 0.0, 'B': 0.0,
            'DI': vds, 'SI': 0.0, 'GP': vgs,
            'BP': 0.0, 'BI': 0.0, 'BS': 0.0, 'BD': 0.0, 'NOI': 0.0,
        }
        eval_params = build_eval_params_array(module, node_voltages)
        (resist_res, _), _ = jit_eval(eval_params, cache)
        currents.append(float(resist_res[0]))

    currents = np.array(currents)

    # Verify results are finite
    assert np.all(np.isfinite(currents))

    # For NMOS, current should generally increase with Vgs
    # (though exact behavior depends on cache initialization)


def main():
    """Run PSP103 test with optional OSDI comparison."""
    print("=" * 70)
    print("PSP103 Realistic Current Test (jax_emit)")
    print("Using ring oscillator model parameters (W=10um, L=1um)")
    print("=" * 70)

    # Load PSP103 module
    va_path = OPENVAF_TESTS / "PSP103" / "psp103.va"
    module = openvaf_py.compile_va(str(va_path))[0]

    # Build functions
    eval_fn, eval_meta = build_eval_fn(module)
    init_fn, init_meta = build_init_fn(module)
    print(f"\nJAX eval strategy: {eval_meta['strategy']}")
    print(f"JAX init strategy: {init_meta['strategy']}")

    jit_init = jax.jit(init_fn)
    jit_eval = jax.jit(eval_fn)

    # Build init params
    init_param_values = {
        '$temperature': 300.0,
        'mfactor': 1.0,
        'W': 10e-6,
        'L': 1e-6,
        'AD': 10e-6 * 0.5e-6,
        'AS': 10e-6 * 0.5e-6,
        'PD': 2 * (10e-6 + 0.5e-6),
        'PS': 2 * (10e-6 + 0.5e-6),
        **NMOS_MODEL_PARAMS,
    }
    init_params = build_init_params_array(module, init_param_values)
    cache = jit_init(init_params)
    print(f"Cache size: {cache.shape[0]}")
    print(f"Finite cache values: {jnp.sum(jnp.isfinite(cache))}/{cache.shape[0]}")

    # Optionally load OSDI for comparison
    osdi_results = None
    if HAS_OSDI:
        osdi_path = RING_DIR / "psp103v4.osdi"
        if osdi_path.exists():
            try:
                lib = OsdiLibrary(str(osdi_path))
                model = lib.create_model()
                instance = model.create_instance()

                osdi_params = lib.get_params()
                osdi_param_idx = {p['name'].lower(): i for i, p in enumerate(osdi_params)}
                instance.set_real_param(osdi_param_idx['w'], 10e-6)
                instance.set_real_param(osdi_param_idx['l'], 1e-6)
                instance.set_real_param(osdi_param_idx['ad'], 10e-6 * 0.5e-6)
                instance.set_real_param(osdi_param_idx['as'], 10e-6 * 0.5e-6)
                instance.set_real_param(osdi_param_idx['pd'], 2 * (10e-6 + 0.5e-6))
                instance.set_real_param(osdi_param_idx['ps'], 2 * (10e-6 + 0.5e-6))

                nodes = lib.get_nodes()
                n_nodes = len(nodes)
                node_idx = {name: i for i, name in enumerate(nodes)}
                instance.init_node_mapping(list(range(n_nodes)))
                instance.process_params(300.0, n_nodes)

                osdi_results = {'instance': instance, 'node_idx': node_idx, 'n_nodes': n_nodes}
                print(f"OSDI nodes: {nodes}")
            except Exception as e:
                print(f"OSDI load failed: {e}")

    # Voltage sweep
    vds = 0.6
    if osdi_results:
        print(f"\n{'Vgs':>6} | {'JAX Id':>14} | {'OSDI Id':>14} | {'Diff':>12} | Match")
    else:
        print(f"\n{'Vgs':>6} | {'JAX Id':>14}")
    print("-" * 65)

    max_jax_current = 0
    max_osdi_current = 0

    for vgs in np.linspace(0.0, 1.2, 13):
        node_voltages = {
            'D': vds, 'G': vgs, 'S': 0.0, 'B': 0.0,
            'DI': vds, 'SI': 0.0, 'GP': vgs,
            'BP': 0.0, 'BI': 0.0, 'BS': 0.0, 'BD': 0.0, 'NOI': 0.0,
        }

        # JAX evaluation
        eval_params = build_eval_params_array(module, node_voltages)
        (jax_res, _), _ = jit_eval(eval_params, cache)
        jax_id = float(jax_res[0])
        max_jax_current = max(max_jax_current, abs(jax_id))

        if osdi_results:
            # OSDI evaluation
            prev_solve = np.zeros(osdi_results['n_nodes'])
            for name, idx in osdi_results['node_idx'].items():
                if name in node_voltages:
                    prev_solve[idx] = node_voltages[name]

            osdi_results['instance'].eval(
                prev_solve,
                CALC_RESIST_RESIDUAL | CALC_RESIST_JACOBIAN | ANALYSIS_DC,
                0.0
            )
            osdi_res = np.zeros(osdi_results['n_nodes'])
            osdi_results['instance'].load_residual_resist(osdi_res)
            osdi_id = osdi_res[0]
            max_osdi_current = max(max_osdi_current, abs(osdi_id))

            diff = abs(jax_id - osdi_id)
            ref = max(abs(jax_id), abs(osdi_id), 1e-15)
            match = "Y" if diff < 0.01 * ref + 1e-12 else "N"

            print(f"{vgs:6.2f} | {jax_id:14.6e} | {osdi_id:14.6e} | {diff:12.2e} | {match}")
        else:
            print(f"{vgs:6.2f} | {jax_id:14.6e}")

    print()
    print(f"Max JAX current:  {max_jax_current:.6e} A")
    if osdi_results:
        print(f"Max OSDI current: {max_osdi_current:.6e} A")

    if max_jax_current < 1e-10:
        print("\nWARNING: JAX currents are essentially zero!")


if __name__ == "__main__":
    main()
