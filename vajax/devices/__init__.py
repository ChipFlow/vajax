"""Device models for VA-JAX

Provides source waveform functions and OpenVAF Verilog-A device support.
All simulation devices (resistor, capacitor, diode, MOSFET) are compiled
from Verilog-A sources using OpenVAF.
"""

from vajax.devices.vsource import (
    isource_batch,
    pulse_voltage,
    pulse_voltage_jax,
    vsource_batch,
)

# Optional Verilog-A support (requires openvaf_py)
try:
    from vajax.devices.verilog_a import VerilogADevice, compile_va

    _HAS_VERILOG_A = True
except ImportError:
    VerilogADevice = None
    compile_va = None
    _HAS_VERILOG_A = False

__all__ = [
    # Source waveforms
    "pulse_voltage",
    "pulse_voltage_jax",
    "vsource_batch",
    "isource_batch",
    # Verilog-A support
    "VerilogADevice",
    "compile_va",
]
