# JAX-SPICE: GPU-Accelerated Analog Circuit Simulator

A proof-of-concept analog circuit simulator built on JAX, demonstrating:
- Automatic differentiation for device Jacobians
- GPU acceleration for large circuits
- Integration with commercial PDK Verilog-A models via OpenVAF/OSDI
- SAX-inspired functional device model API

## Quick Start

```bash
# Install dependencies (CPU only)
pip install -e .

# With GPU support
pip install -e ".[gpu]"

# With SAX integration
pip install -e ".[sax]"
```

## Example: CMOS Inverter

```python
from jax_spice import Circuit, MOSFETSimple
from jax_spice.analysis import dc_sweep

# Define circuit
ckt = Circuit()
ckt.add("M1", MOSFETSimple, W=10e-6, L=0.25e-6, type='nmos',
        d="out", g="in", s="gnd", b="gnd")
ckt.add("M2", MOSFETSimple, W=20e-6, L=0.25e-6, type='pmos',
        d="out", g="in", s="vdd", b="vdd")

# DC sweep
result = dc_sweep(ckt, sweep_var="in", start=0, stop=2.5, points=251)
```

## Architecture

### Device Models
- Pure JAX functions with automatic differentiation
- Batched evaluation with `jax.vmap()`
- JIT compilation for performance

### Verilog-A Integration
- Compile foundry `.va` models to OSDI with OpenVAF
- Wrap OSDI libraries in JAX-compatible interface
- Validation framework ensures bit-for-bit accuracy

### Solver
- Newton-Raphson DC analysis in JAX
- Sparse matrix operations (JAX experimental)
- GPU-accelerated for large circuits

## Project Status

🚧 **Work in Progress** - Proof of concept phase

- [x] Project structure
- [ ] Base device interface
- [ ] Simplified MOSFET model
- [ ] DC solver
- [ ] CMOS inverter validation vs ngspice
- [ ] OSDI wrapper for Verilog-A
- [ ] Performance benchmarks

## Comparison vs ngspice

Target: CMOS inverter from ngspice test suite (`tests/bsim3soidd/inv2.cir`)
- Reference: ngspice simulation results
- Validation: <0.1% error in DC transfer curve
- Performance: 10x+ speedup with GPU for 100+ devices

## License

MIT (prototype/research code)
