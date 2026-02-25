# Supported Devices

VAJAX supports device models compiled from Verilog-A via OpenVAF, plus built-in
voltage and current sources.

## Core Devices (Tested in VACASK Suite)

These devices are validated against VACASK and ngspice reference results:

| Device | OSDI Module | Verilog-A Source | Benchmarks |
|--------|-------------|------------------|------------|
| Resistor | `sp_resistor` | `devices/resistor.va` | rc |
| Capacitor | `sp_capacitor` | `devices/capacitor.va` | rc |
| Diode | `diode` | `devices/diode.va` | graetz, mul |
| PSP103 MOSFET | `psp103va` | `PSP103/psp103.va` | ring, c6288 |

## Additional Verilog-A Models

These models are available in `vendor/VACASK/devices/` but have less test coverage:

| Device | OSDI Module | Verilog-A Source | Status |
|--------|-------------|------------------|--------|
| BSIM3v3 | `bsim3v3` | `devices/bsim3v3.va` | Compiles, needs integration tests |
| BSIM4v8 | `bsim4v8` | `devices/bsim4v8.va` | Compiles, needs integration tests |
| BSIM-BULK 106 | `bsimbulk106` | `devices/bsimbulk106.va` | Compiles, needs integration tests |
| Inductor | `inductor` | `devices/inductor.va` | Compiles, needs integration tests |
| Op-Amp | `opamp` | `devices/opamp.va` | Compiles, needs integration tests |
| VBIC BJT | `vbic` | `devices/vbic.va` | Compiles, needs integration tests |

## Built-in Sources

Voltage and current sources are implemented directly in Python (`vajax/devices/vsource.py`),
not via Verilog-A:

| Source | Type | Parameters |
|--------|------|------------|
| DC | Voltage/Current | `dc` (value) |
| Pulse | Voltage/Current | `val0`, `val1`, `delay`, `rise`, `fall`, `width`, `period` |
| Sine | Voltage/Current | `offset`, `amplitude`, `frequency`, `delay`, `damping` |
| PWL | Voltage/Current | Time-value pairs |

### Source Example

```spice
// DC source
vdd (vdd 0) vsource dc=1.8

// Pulse source
vin (in 0) vsource dc=0 type="pulse" val0=0 val1=1.8 rise=100p fall=100p width=5n period=10n

// Sine source
vsin (sig 0) vsource dc=0 type="sin" offset=0.9 amplitude=0.1 frequency=1e6

// Current source
ibias (vdd out) isource dc=100u
```

## SPICE Model Type Mapping

When converting from SPICE netlists, VAJAX maps model types to OSDI modules
via `vajax/netlist_converter/spiceparser/elements.py`:

### Passive Devices

| SPICE Type | Family | OSDI Module |
|------------|--------|-------------|
| `r`, `res` | `r` | `spice/resistor.osdi` |
| `c` | `c` | `spice/capacitor.osdi` |
| `l` | `l` | `spice/inductor.osdi` |

### Diodes

| SPICE Type | Family | Level | OSDI Module |
|------------|--------|-------|-------------|
| `d` | `d` | 1, 3 | `spice/diode.osdi` |

### BJTs

| SPICE Type | Family | Level | OSDI Module |
|------------|--------|-------|-------------|
| `npn`, `pnp` | `bjt` | 1 (default) | `spice/bjt.osdi` (Gummel-Poon) |
| `npn`, `pnp` | `bjt` | 4, 9 | `spice/vbic.osdi` (VBIC) |

### MOSFETs

| SPICE Type | Family | Level | OSDI Module |
|------------|--------|-------|-------------|
| `nmos`, `pmos` | `mos` | 1 | `spice/mos1.osdi` |
| `nmos`, `pmos` | `mos` | 2 | `spice/mos2.osdi` |
| `nmos`, `pmos` | `mos` | 3 | `spice/mos3.osdi` |
| `nmos`, `pmos` | `mos` | 8, 49 | `spice/bsim3v3.osdi` (BSIM3) |
| `nmos`, `pmos` | `mos` | 14, 54 | `spice/bsim4.osdi` (BSIM4) |

### Other Devices

| SPICE Type | Family | OSDI Module |
|------------|--------|-------------|
| `njf`, `pjf` | `jfet` | `spice/jfet1.osdi` |
| `nmf`, `pmf` | `mes` | `spice/mes1.osdi` |

## Adding a New Device

All devices (except sources) are compiled from Verilog-A via OpenVAF. To add a new device:

1. Obtain or write a Verilog-A model (`.va` file)
2. Compile it with OpenVAF to produce an `.osdi` module
3. Reference it in a VACASK `.sim` netlist with `load "your_model.osdi"`
4. The engine will automatically wrap it via `VerilogADevice` with batched `jax.vmap` evaluation

No Python device code is needed. See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.
