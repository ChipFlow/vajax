# JAX-SPICE Development Guidelines

## Pure JAX Requirement

**CRITICAL**: This is a JAX-native circuit simulator. All numerical code MUST use JAX, not numpy or scipy.

### Forbidden Patterns

Do NOT use these:
```python
# FORBIDDEN - numpy arrays
import numpy as np
np.array([1, 2, 3])
np.zeros(n)
np.linalg.solve(A, b)

# FORBIDDEN - scipy sparse
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
from scipy.sparse.linalg import spsolve, gmres

# FORBIDDEN - numpy linear algebra
np.linalg.inv(A)
np.linalg.lstsq(A, b)
```

### Required Patterns

Use these JAX equivalents:
```python
# REQUIRED - JAX arrays
import jax.numpy as jnp
from jax import Array
jnp.array([1, 2, 3])
jnp.zeros(n)
jax.scipy.linalg.solve(A, b)

# REQUIRED - JAX sparse
from jax.experimental.sparse import BCOO
from jax.experimental.sparse.linalg import spsolve

# REQUIRED - JAX linear algebra
jax.scipy.linalg.inv(A)
jax.scipy.linalg.lstsq(A, b)

# REQUIRED - Matrix-free GMRES (for large sparse systems)
from jax.scipy.sparse.linalg import gmres
def matvec(v):
    _, jvp = jax.jvp(residual_fn, (x,), (v,))
    return jvp
delta, info = gmres(matvec, -f, tol=1e-6)
```

### Why Pure JAX?

1. **GPU Acceleration**: JAX arrays stay on GPU, avoiding host-device transfers
2. **JIT Compilation**: All code can be traced and compiled
3. **Autodiff**: Jacobians via `jax.jacfwd` or matrix-free `jax.jvp`
4. **Consistency**: One API across CPU, GPU, and TPU

### Sparse Solver Strategy

For large circuits (>1000 nodes), use matrix-free Newton-Raphson with GMRES:

```python
def newton_step_sparse(residual_fn, V, tol=1e-6):
    """Sparse Newton step using matrix-free GMRES."""
    from jax.scipy.sparse.linalg import gmres

    f = residual_fn(V)

    def matvec(v):
        # Jacobian-vector product via forward-mode AD
        v_padded = jnp.concatenate([jnp.array([0.0]), v])
        _, jvp = jax.jvp(residual_fn, (V,), (v_padded,))
        return jvp

    delta, info = gmres(matvec, -f, tol=tol)
    return V.at[1:].add(delta)
```

## Test Commands

```bash
# Run all tests (CPU)
JAX_PLATFORMS=cpu uv run pytest tests/ -v

# Run VACASK benchmark tests
JAX_PLATFORMS=cpu uv run pytest tests/test_vacask_suite.py -v

# Run openvaf-py tests
cd openvaf-py && JAX_PLATFORMS=cpu ../.venv/bin/python -m pytest tests/ -v

# Profile GPU performance
uv run python scripts/profile_gpu.py --benchmark ring
```

Note: `jax_enable_x64` is set automatically on import via `jax_spice/__init__.py`.

## Key Architecture

```
jax_spice/analysis/
├── solver.py          # Newton-Raphson with lax.while_loop
├── system.py          # SystemBuilder for J,f construction
├── dc.py              # DC operating point analysis
├── transient.py       # Transient analysis (vectorized GPU path)
├── mna.py             # MNA system representation
└── sparse.py          # JAX sparse utilities (BCOO)

jax_spice/devices/
└── openvaf_device.py  # Batched OpenVAF device evaluation

jax_spice/benchmarks/
└── runner.py          # VACASK benchmark runner
```

## Migration Status: COMPLETE

All numpy/scipy dependencies have been removed from core computation:
- `jax_spice/analysis/sparse.py` - uses pure JAX for COO→CSR conversion via `jax.ops.segment_sum`
- `jax_spice/benchmarks/runner.py` - returns `jax.Array` types, uses `jnp` throughout
- `jax_spice/devices/openvaf_device.py` - uses `jax.Array` for batched inputs

The codebase is now pure JAX in all simulation paths.
