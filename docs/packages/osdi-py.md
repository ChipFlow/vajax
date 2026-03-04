# osdi-py

Python bindings for OSDI (Open Simulator Device Interface) shared libraries.

`osdi-py` loads pre-compiled `.osdi` device model libraries and exposes their evaluation
functions to Python. It serves as the reference implementation for validating `openvaf-py`
outputs.

## Installation

```bash
pip install osdi-py
```

Pre-built wheels are available for:

- Linux x86_64 and aarch64
- macOS arm64 (Apple Silicon)

## Usage

```python
import osdi_py

# Load a pre-compiled OSDI shared library
model = osdi_py.load("psp103.osdi")

# Query model metadata
print(model.num_nodes)       # Number of external nodes
print(model.num_params)      # Number of model parameters
print(model.param_names)     # Parameter names
```

## When to Use

`osdi-py` is primarily used for:

- **Validation**: Comparing JAX-compiled model outputs against OSDI reference
- **Debugging**: Isolating whether discrepancies come from compilation or evaluation
- **Testing**: The VAJAX test suite uses `osdi-py` for cross-validation

For production simulation, use `openvaf-py` which compiles models into JAX-native
functions that run on GPU.

## Supported Platforms

| Platform | Architecture | Status |
|----------|-------------|--------|
| Linux    | x86_64      | Supported |
| Linux    | aarch64     | Supported |
| macOS    | arm64       | Supported |
| Windows  | x86_64      | Not yet supported |

## Links

- [Source code](https://github.com/ChipFlow/vajax/tree/main/openvaf_jax/osdi_py)
- [PyPI](https://pypi.org/project/osdi-py/)
- [OSDI Parameter Architecture](../osdi_parameter_architecture.md) — how OSDI parameters map to device models
- [Debug Tools](../debug_tools.md) — OSDI vs JAX comparison utilities
