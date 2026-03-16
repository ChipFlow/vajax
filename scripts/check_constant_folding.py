#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["jax", "jaxlib"]
# ///
"""Check whether jnp.where constant folding works at jaxpr and HLO levels.

Quick diagnostic to verify that inlining shared params as Python literals
actually eliminates jnp.where branches in the compiled XLA program.
"""

import jax
import jax.numpy as jnp


def test_basic_constant_folding():
    """Test: does jnp.where with a Python bool constant-fold?"""
    print("=" * 60)
    print("Test 1: jnp.where with Python bool literal")
    print("=" * 60)

    def f_const(x):
        # Condition is a Python bool - should constant-fold
        cond = True
        return jnp.where(cond, x * 2, x * 3)

    def f_traced(x, flag):
        # Condition is a traced value - cannot constant-fold
        return jnp.where(flag > 0.0, x * 2, x * 3)

    x = jnp.ones(10)

    jaxpr_const = jax.make_jaxpr(f_const)(x)
    jaxpr_traced = jax.make_jaxpr(f_traced)(x, 1.0)

    print(f"\nConstant cond jaxpr ({len(jaxpr_const.eqns)} ops):")
    print(jaxpr_const)
    print(f"\nTraced cond jaxpr ({len(jaxpr_traced.eqns)} ops):")
    print(jaxpr_traced)

    # Check HLO
    lowered_const = jax.jit(f_const).lower(x)
    lowered_traced = jax.jit(f_traced).lower(x, 1.0)
    hlo_const = lowered_const.as_text()
    hlo_traced = lowered_traced.as_text()

    select_const = hlo_const.count("select")
    select_traced = hlo_traced.count("select")
    print(f"\nHLO select ops: constant={select_const}, traced={select_traced}")


def test_constant_through_jnp_ops():
    """Test: does constant folding survive through jnp operations?"""
    print("\n" + "=" * 60)
    print("Test 2: Constant folding through jnp operations")
    print("=" * 60)

    def f_inlined(x):
        # Simulate what our specialization does:
        # shared param inlined as literal, then used in jnp ops
        v_param = 1.5e-6  # Was: shared_params[42]
        v_computed = jnp.exp(v_param)  # jnp op on literal
        cond = v_computed > 0.5  # comparison
        return jnp.where(cond, x * 2, x * 3)

    def f_array_lookup(x, shared_params):
        # Original: shared_params array lookup
        v_param = shared_params[42]
        v_computed = jnp.exp(v_param)
        cond = v_computed > 0.5
        return jnp.where(cond, x * 2, x * 3)

    x = jnp.ones(10)
    shared = jnp.zeros(100)

    jaxpr_inlined = jax.make_jaxpr(f_inlined)(x)
    jaxpr_lookup = jax.make_jaxpr(f_array_lookup)(x, shared)

    print(f"\nInlined literal jaxpr ({len(jaxpr_inlined.eqns)} ops):")
    print(jaxpr_inlined)
    print(f"\nArray lookup jaxpr ({len(jaxpr_lookup.eqns)} ops):")
    print(jaxpr_lookup)

    # Check HLO
    lowered_inlined = jax.jit(f_inlined).lower(x)
    lowered_lookup = jax.jit(f_array_lookup).lower(x, shared)
    hlo_inlined = lowered_inlined.as_text()
    hlo_lookup = lowered_lookup.as_text()

    select_inlined = hlo_inlined.count("select")
    select_lookup = hlo_lookup.count("select")
    print(f"\nHLO select ops: inlined={select_inlined}, lookup={select_lookup}")


def test_python_float_vs_jnp():
    """Test: Python float literal vs jnp operation - what does JAX trace?"""
    print("\n" + "=" * 60)
    print("Test 3: Python float arithmetic vs jnp arithmetic")
    print("=" * 60)

    def f_python_arith(x):
        # Pure Python: should constant-fold completely
        a = 1.5e-6
        b = a * 2.0  # Python multiplication
        cond = b > 1e-6  # Python comparison -> True
        return jnp.where(cond, x * 2, x * 3)

    def f_jnp_arith(x):
        # jnp operations: might NOT constant-fold in jaxpr
        a = 1.5e-6
        b = jnp.float64(a) * 2.0  # jnp multiplication
        cond = b > 1e-6  # comparison on jnp result
        return jnp.where(cond, x * 2, x * 3)

    x = jnp.ones(10)

    jaxpr_python = jax.make_jaxpr(f_python_arith)(x)
    jaxpr_jnp = jax.make_jaxpr(f_jnp_arith)(x)

    print(f"\nPython arith jaxpr ({len(jaxpr_python.eqns)} ops):")
    print(jaxpr_python)
    print(f"\njnp arith jaxpr ({len(jaxpr_jnp.eqns)} ops):")
    print(jaxpr_jnp)

    # Check HLO for both
    lowered_python = jax.jit(f_python_arith).lower(x)
    lowered_jnp = jax.jit(f_jnp_arith).lower(x)
    hlo_python = lowered_python.as_text()
    hlo_jnp = lowered_jnp.as_text()

    select_python = hlo_python.count("select")
    select_jnp = hlo_jnp.count("select")
    print(f"\nHLO select ops: python_arith={select_python}, jnp_arith={select_jnp}")


def test_generated_code_pattern():
    """Test the ACTUAL pattern used in generated eval code.

    The generated code does:
        v123 = 1.5e-6  # inlined literal (was shared_params[42])
    Then later uses it in jnp operations.

    Key question: does assigning a Python float to a local variable,
    then using it in jnp.where, get constant-folded?
    """
    print("\n" + "=" * 60)
    print("Test 4: Actual generated code pattern (assign + jnp.where)")
    print("=" * 60)

    def f_generated_pattern(device_params):
        # This mimics the actual generated eval code pattern
        # Inlined shared params
        v100 = 1.0  # TYPE = 1 (NMOS)
        v101 = 0.0  # SWIGATE = 0
        v102 = 1.5e-6  # TOX

        # Device params (from vmap, traced)
        v200 = device_params[0]  # voltage
        _v201 = device_params[1]  # another voltage (unused, kept for array shape)

        # Computation chain (mimics what OpenVAF generates)
        v300 = jnp.exp(v102 * 1e6)  # Uses inlined literal
        v301 = v300 * v200  # Mixes with traced value

        # Branch on static param
        v400 = v100 > 0.5  # TYPE > 0.5 -> True for NMOS
        result1 = jnp.where(v400, v301, -v301)  # Should fold

        # Branch on static param through jnp op
        v401 = jnp.abs(v101)  # jnp op on inlined literal
        v402 = v401 > 0.5  # Should be False
        result2 = jnp.where(v402, result1 * 2, result1 * 3)  # Should fold

        return result1 + result2

    def f_array_pattern(device_params, shared_params):
        # Original pattern: array lookups (all traced)
        v100 = shared_params[0]
        v101 = shared_params[1]
        v102 = shared_params[2]

        v200 = device_params[0]
        _v201 = device_params[1]  # noqa: F841

        v300 = jnp.exp(v102 * 1e6)
        v301 = v300 * v200

        v400 = v100 > 0.5
        result1 = jnp.where(v400, v301, -v301)

        v401 = jnp.abs(v101)
        v402 = v401 > 0.5
        result2 = jnp.where(v402, result1 * 2, result1 * 3)

        return result1 + result2

    dp = jnp.array([0.5, 0.3])
    sp = jnp.array([1.0, 0.0, 1.5e-6])

    jaxpr_gen = jax.make_jaxpr(f_generated_pattern)(dp)
    jaxpr_arr = jax.make_jaxpr(f_array_pattern)(dp, sp)

    print(f"\nInlined pattern jaxpr ({len(jaxpr_gen.eqns)} ops):")
    print(jaxpr_gen)
    print(f"\nArray pattern jaxpr ({len(jaxpr_arr.eqns)} ops):")
    print(jaxpr_arr)

    # Check HLO
    lowered_gen = jax.jit(f_generated_pattern).lower(dp)
    lowered_arr = jax.jit(f_array_pattern).lower(dp, sp)
    hlo_gen = lowered_gen.as_text()
    hlo_arr = lowered_arr.as_text()

    select_gen = hlo_gen.count("select")
    select_arr = hlo_arr.count("select")
    print(f"\nHLO select ops: inlined={select_gen}, array={select_arr}")

    # Also count total HLO ops
    print(f"HLO lines: inlined={len(hlo_gen.splitlines())}, array={len(hlo_arr.splitlines())}")


def test_vmap_interaction():
    """Test: does constant folding survive vmap?

    This is crucial because we vmap the eval function over devices.
    """
    print("\n" + "=" * 60)
    print("Test 5: Constant folding under vmap")
    print("=" * 60)

    def f_inlined(device_params):
        v_type = 1.0  # Inlined: TYPE = NMOS
        cond = v_type > 0.5
        return jnp.where(cond, device_params[0] * 2, device_params[0] * 3)

    def f_lookup(device_params, shared_params):
        v_type = shared_params[0]
        cond = v_type > 0.5
        return jnp.where(cond, device_params[0] * 2, device_params[0] * 3)

    # vmap over batch of devices
    f_inlined_vmapped = jax.vmap(f_inlined)
    f_lookup_vmapped = jax.vmap(f_lookup, in_axes=(0, None))

    batch_dp = jnp.ones((4, 3))
    sp = jnp.array([1.0])

    jaxpr_inlined = jax.make_jaxpr(f_inlined_vmapped)(batch_dp)
    jaxpr_lookup = jax.make_jaxpr(f_lookup_vmapped)(batch_dp, sp)

    print(f"\nvmapped inlined jaxpr ({len(jaxpr_inlined.eqns)} ops):")
    print(jaxpr_inlined)
    print(f"\nvmapped lookup jaxpr ({len(jaxpr_lookup.eqns)} ops):")
    print(jaxpr_lookup)

    # HLO
    lowered_inlined = jax.jit(f_inlined_vmapped).lower(batch_dp)
    lowered_lookup = jax.jit(f_lookup_vmapped).lower(batch_dp, sp)
    hlo_inlined = lowered_inlined.as_text()
    hlo_lookup = lowered_lookup.as_text()

    select_inlined = hlo_inlined.count("select")
    select_lookup = hlo_lookup.count("select")
    print(f"\nHLO select ops: inlined={select_inlined}, lookup={select_lookup}")
    print(
        f"HLO lines: inlined={len(hlo_inlined.splitlines())}, lookup={len(hlo_lookup.splitlines())}"
    )


if __name__ == "__main__":
    print(f"JAX version: {jax.__version__}")
    print(f"Platform: {jax.default_backend()}")
    print(
        f"x64 enabled: {jax.config.x86_64_enabled if hasattr(jax.config, 'x86_64_enabled') else 'unknown'}"
    )
    print()

    test_basic_constant_folding()
    test_constant_through_jnp_ops()
    test_python_float_vs_jnp()
    test_generated_code_pattern()
    test_vmap_interaction()
