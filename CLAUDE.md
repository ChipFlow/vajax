# JAX-SPICE Development Guidelines

## Pure JAX Requirement

**CRITICAL**: This is a JAX-native circuit simulator. All numerical code MUST use JAX, not numpy or scipy.
**CRITICAL**: All functionality should be optimised to run as much on GPU as possible, minimising CPU to GPU context switching and data transfer

You should try to minimise the code base size:

 * Avoid duplication
 * Where possible unify critical paths.
 * Regularly check for duplications of code and look for any opportunities to minimise the code base without impacting the tested functionality. Conform with the user.

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

# REQUIRED - JAX sparse (BCOO/BCSR formats)
from jax.experimental.sparse import BCOO, BCSR
from jax.experimental.sparse.linalg import spsolve

# REQUIRED - JAX linear algebra
jax.scipy.linalg.inv(A)
jax.scipy.linalg.lstsq(A, b)
```

### Why Pure JAX?

1. **GPU Acceleration**: JAX arrays stay on GPU, avoiding host-device transfers
2. **JIT Compilation**: All code can be traced and compiled
3. **Autodiff**: Jacobians via `jax.jacfwd` or matrix-free `jax.jvp`
4. **Consistency**: One API across CPU, GPU, and TPU

### Sparse Solver Strategy

The simulator supports two solver modes:

1. **Dense solver** (default): Uses `jax.scipy.linalg.solve()` for small-medium circuits
2. **Sparse solver**: Uses BCOO/BCSR + `spsolve` for large circuits (>1000 nodes)

For large circuits, use sparse mode with `use_sparse=True`:

```python
from jax.experimental.sparse import BCOO, BCSR
from jax.experimental.sparse.linalg import spsolve

# Build Jacobian as BCOO, convert to BCSR for solving
J_bcoo = BCOO((j_data, jnp.stack([j_rows, j_cols], axis=1)), shape=(n, n))
J_bcsr = BCSR.from_bcoo(J_bcoo)

# Solve J @ delta = -f using sparse direct solver
delta = spsolve(J_bcsr.data, J_bcsr.indices, J_bcsr.indptr, -f, (n, n))
```

Note: c6288 benchmark (~86k nodes) requires sparse mode as dense would need ~56GB memory.

## Test Commands

```bash
# Run all tests (CPU)
JAX_PLATFORMS=cpu uv run pytest tests/ -v

# Run VACASK benchmark tests
JAX_PLATFORMS=cpu uv run pytest tests/test_vacask_suite.py -v

# Run openvaf-py tests
cd openvaf-py && JAX_PLATFORMS=cpu ../.venv/bin/python -m pytest tests/ -v

# Run and profile GPU tests on non-CUDA systems (e.g. Apple silicon)
uv run scripts/profile_gpu_cloudrun.py --benchmark ring,c6288

# Profile GPU performance on CUDA systems
uv run python scripts/profile_gpu.py --benchmark ring,c6288

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

## Device Routing

All devices are routed through OpenVAF except voltage/current sources:

- **OpenVAF path**: resistor, capacitor, diode, psp103, and other VA models
  - Batched evaluation via `vmap` for GPU efficiency
  - VA models from `vendor/VACASK/devices/` (resistor.va, capacitor.va, diode.va)
  - Complex models from `openvaf-py/vendor/OpenVAF/integration_tests/` (PSP103)

- **Source path**: vsource, isource only
  - Time-varying behavior (pulse, sine, DC)
  - Handled separately with vectorized stamping

## Migration Status: COMPLETE

All numpy/scipy dependencies have been removed from core computation:
- `jax_spice/analysis/sparse.py` - uses pure JAX for COO→CSR conversion via `jax.ops.segment_sum`
- `jax_spice/benchmarks/runner.py` - returns `jax.Array` types, uses `jnp` throughout
- `jax_spice/devices/openvaf_device.py` - uses `jax.Array` for batched inputs

The codebase is now pure JAX in all simulation paths.
