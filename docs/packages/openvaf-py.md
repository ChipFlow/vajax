# openvaf-py

Python bindings for the OpenVAF MIR (Mid-level Intermediate Representation) interpreter.

`openvaf-py` compiles Verilog-A device models into JAX-compatible Python functions via OpenVAF,
enabling GPU-accelerated circuit simulation with VAJAX.

## Installation

```bash
pip install openvaf-py
```

Pre-built wheels are available for:

- Linux x86_64 and aarch64
- macOS arm64 (Apple Silicon)

## Usage

```python
import openvaf_py

# Compile a Verilog-A model
module = openvaf_py.compile("resistor.va")

# Get model metadata
print(module.functions)  # Available functions (init, eval)
print(module.params)     # Model parameters
```

## How It Works

1. **Compilation**: OpenVAF compiles Verilog-A (`.va`) source into MIR
2. **Code generation**: `openvaf_jax` translates MIR instructions into JAX operations
3. **JIT compilation**: JAX traces and JIT-compiles the resulting function for CPU/GPU

This approach enables:

- Full Verilog-A language support (control flow, `$limit`, `$simparam`, etc.)
- Automatic differentiation through device models
- GPU-accelerated batched evaluation via `vmap`

## Supported Platforms

| Platform | Architecture | Status |
|----------|-------------|--------|
| Linux    | x86_64      | Supported |
| Linux    | aarch64     | Supported |
| macOS    | arm64       | Supported |
| Windows  | x86_64      | Not yet supported |

## Links

- [Source code](https://github.com/ChipFlow/vajax/tree/main/openvaf_jax/openvaf_py)
- [PyPI](https://pypi.org/project/openvaf-py/)
- [Verilog-A System Functions](../verilog_a_system_functions.md) — supported VA callbacks
- [PHI Nodes Guide](../phi_nodes_guide.md) — debugging MIR control flow
