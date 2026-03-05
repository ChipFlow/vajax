# Memory Management

VAJAX compiles Verilog-A models via OpenVAF (Rust) and generates JAX functions
for GPU-accelerated simulation. Both the compilation and JIT compilation steps
allocate significant memory. This document explains the memory lifecycle and how
to manage it.

## Memory Consumers

### 1. Rust VaModule (~100-800 MB per large model)

When `openvaf_py.compile_va()` compiles a Verilog-A file, it produces a
`VaModule` — a Rust-backed object containing the parsed model IR. Large models
like BSIM4 or PSP103 allocate hundreds of megabytes in native (Rust) memory.

**Key behaviour**: The Rust allocator (jemalloc/system) retains freed pages for
internal reuse but rarely returns them to the OS. This means RSS (Resident Set
Size) only grows — even after `del module; gc.collect()`, the process RSS stays
high. However, the freed memory IS reused by subsequent Rust allocations (e.g.,
compiling the same model again uses +0 additional RSS).

### 2. OpenVAF MIR Data (~10-50 MB per model)

The `OpenVAFToJAX` translator parses the VaModule's MIR (Mid-level IR) into
Python data structures for code generation. This includes instruction lists,
constant tables, DAE system metadata, and parsed CFG blocks.

After all `translate_*()` calls complete, MIR data is no longer needed but
stays in memory unless explicitly released via `translator.release_mir_data()`.

### 3. JAX XLA Compilation Cache (~50-200 MB per JIT'd function)

Each `jax.jit()` call compiles a Python function to XLA HLO and then to machine
code. For large models, a single JIT'd eval function can use 200+ MB. These are
cached by JAX in-memory.

`jax.clear_caches()` frees this memory. Per-function clearing is available via
`jitted_fn.clear_cache()`.

### 4. Persistent On-Disk Caches

- **OpenVAF MIR cache** (`~/.cache/vajax/openvaf/`): Serialized MIR data to
  skip recompilation on subsequent runs.
- **JAX XLA cache** (configured via `jax_compilation_cache_dir`): Compiled HLO
  artifacts to skip XLA compilation.

## API Reference

### `vajax.clear_caches(include_persistent=False)`

Nuclear option — clears everything:
- Releases MIR data and Rust module references from all cached models
- Clears the global `COMPILED_MODEL_CACHE`
- Clears openvaf_jax function caches
- Clears JAX XLA in-memory caches
- Runs `gc.collect()`

Use this when switching to a completely different simulation or to free maximum
memory. After calling this, models must be recompiled from scratch.

```python
import vajax

engine = vajax.CircuitEngine(sim_file)
engine.run_transient()

# Free all memory
stats = vajax.clear_caches()
# stats = {'openvaf_compiled_models': 3, 'models_mir_released': 2, ...}
```

### `vajax.release_model(model_type)`

Selective release — frees the heavy Rust/MIR memory for one model while keeping
other models cached. The model's compiled JAX functions and metadata remain in
cache, so re-running the same circuit topology is fast. However, running a
different circuit topology with the same model type will require recompilation.

```python
# Simulate with BSIM4
engine1 = vajax.CircuitEngine(bsim4_circuit)
engine1.run_transient()

# Free BSIM4's ~800MB of Rust data, keep other models
vajax.release_model('bsim4')

# Simulate with PSP103
engine2 = vajax.CircuitEngine(psp103_circuit)
engine2.run_transient()
```

### `vajax.cleanup_disk_cache(max_age_days=30)`

Time-based LRU cleanup of the persistent on-disk OpenVAF cache. Removes cache
entries whose newest file is older than `max_age_days`.

```python
# Clean up entries older than 7 days
result = vajax.cleanup_disk_cache(max_age_days=7)
# result = {'removed': ['psp103_a1b2c3d4'], 'kept': ['resistor_e5f6g7h8']}
```

### `engine.clear_cache()`

Instance-level cache clearing on a `CircuitEngine`. Delegates to
`vajax.clear_caches()` for global caches and also clears engine-specific state.

## Memory Strategy for Common Use Cases

### Running many circuits with the same model

Keep the cache — compilation is expensive, reuse is free:

```python
for sim_file in circuit_files:
    engine = vajax.CircuitEngine(sim_file)
    engine.run_transient()
    # Don't clear caches — the compiled model is reused
```

### Switching between different models

Release the old model before loading the new one:

```python
vajax.release_model('bsim4')
engine = vajax.CircuitEngine(psp103_circuit)
```

### Long-running server / batch processing

Periodically clean up:

```python
# After each batch
vajax.clear_caches()

# On startup, clean stale disk cache
vajax.cleanup_disk_cache(max_age_days=7)
```

### CI / test environments

Clear between test modules to prevent OOM:

```python
@pytest.fixture(autouse=True)
def _cleanup():
    yield
    jax.clear_caches()
    gc.collect()
```

## Rust Allocator Behaviour

The Rust allocator retains freed pages internally rather than returning them to
the OS via `munmap`. This is normal and expected — it allows memory to be reused
efficiently by subsequent allocations.

**Implications:**
- Process RSS only grows, never shrinks (on both Linux and macOS)
- `del module; gc.collect()` frees memory for Rust reuse but RSS stays flat
- Compiling the same model after deletion reuses freed pages (+0 MB RSS)
- Compiling a different model may or may not reuse freed pages depending on
  allocation sizes

This behaviour was verified empirically: compiling BSIM4 (RSS +800 MB),
deleting, then recompiling BSIM4 shows +0 MB additional RSS.
