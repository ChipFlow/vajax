"""Device models for JAX-SPICE"""

from jax_spice.devices.base import Device, DeviceStamps
from jax_spice.devices.mosfet_simple import MOSFETSimple
from jax_spice.devices.resistor import Resistor, resistor
from jax_spice.devices.capacitor import Capacitor, capacitor, capacitor_companion
from jax_spice.devices.vsource import (
    VoltageSource, CurrentSource,
    pulse_voltage, pulse_voltage_jax,
)

# Optional Verilog-A support (requires openvaf_py)
try:
    from jax_spice.devices.verilog_a import VerilogADevice, compile_va
    from jax_spice.devices.openvaf_device import (
        VADevice,
        CompiledModelBatch,
        compile_model,
        clear_model_cache,
        HAS_OPENVAF,
    )
    _HAS_VERILOG_A = True
except ImportError:
    VerilogADevice = None
    compile_va = None
    VADevice = None
    CompiledModelBatch = None
    compile_model = None
    clear_model_cache = None
    HAS_OPENVAF = False
    _HAS_VERILOG_A = False

__all__ = [
    "Device", "DeviceStamps",
    "MOSFETSimple",
    "VerilogADevice", "compile_va",
    "VADevice", "CompiledModelBatch", "compile_model", "clear_model_cache", "HAS_OPENVAF",
    "Resistor", "resistor",
    "Capacitor", "capacitor", "capacitor_companion",
    "VoltageSource", "CurrentSource",
    "pulse_voltage", "pulse_voltage_jax",
]
