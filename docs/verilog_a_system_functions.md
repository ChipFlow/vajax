# Verilog-A System Functions Analysis

Analysis of `$` system functions used in OpenVAF and VACASK `.va` files, and how they are handled in the JAX and OSDI paths.

> **Reference**: This document summarizes system functions per the [Verilog-AMS Language Reference Manual (VAMS-LRM-2023)](../vendor/OpenVAF/docs/VAMS-LRM-2023/09-system-tasks-functions.md), Section 9.

## Summary Statistics

From 135 `.va` files analyzed:

| Function | Count | Category | Description |
|----------|------:|----------|-------------|
| `$param_given` | 1891 | Device | Check if param was explicitly set |
| `$strobe` | 1434 | Debug | Print at end of timestep |
| `$simparam` | 481 | Simulation | Simulator parameters |
| `$temperature` | 274 | Simulation | Device temperature in K |
| `$limit` | 270 | Device | Limiting function for convergence |
| `$write` | 251 | Debug | Print immediately |
| `$warning` | 212 | Debug | Issue warning message |
| `$mfactor` | 191 | Simulation | Device multiplier |
| `$finish` | 92 | Debug | End simulation |
| `$fatal` | 64 | Debug | Fatal error, stop simulation |
| `$port_connected` | 57 | Device | Check if port is connected |
| `$discontinuity` | 42 | Device | Notify simulator of discontinuity |
| `$vt` | 28 | Simulation | Thermal voltage kT/q |
| `$display` | 11 | Debug | Print with newline |
| `$abstime` | 5 | Simulation | Absolute simulation time |

---

## Simulation-Critical Functions

These must be handled correctly for accurate simulation results.

### `$temperature` (274 uses)

Returns the circuit ambient temperature in Kelvin (LRM 9.15).

> **LRM Definition**: `$temperature` returns the ambient temperature as a real value in Kelvin.

| Path | Handling |
|------|----------|
| **JAX** | Param with `kind='temperature'` → uses `temperature` argument from `translate_init()`/`translate_eval()` |
| **OSDI** | `OsdiSimInfo.temperature` field |

### `$simparam` (481 uses)

Queries simulator parameters (LRM 9.15). Usage pattern: `$simparam("name", default)`

> **LRM Definition**: `$simparam()` queries the simulator for a real-valued simulation parameter named `param_name`. If `param_name` is not known and no default is supplied, an error is generated.

#### Standard Parameters (LRM Table 9-27)

| Parameter | Units | LRM Description |
|-----------|-------|-----------------|
| `gdev` | 1/Ohms | Additional conductance for conductance homotopy convergence |
| `gmin` | 1/Ohms | Minimum conductance placed in parallel with nonlinear branches |
| `imax` | Amps | Branch current threshold above which constitutive relation should be linearized |
| `imelt` | Amps | Branch current threshold indicating device failure |
| `iteration` | - | Iteration number of the analog solver |
| `scale` | - | Scale factor for device instance geometry parameters |
| `shrink` | - | Optical linear shrink factor |
| `simulatorSubversion` | - | The simulator sub-version |
| `simulatorVersion` | - | The simulator version |
| `sourceScaleFactor` | - | Multiplicative factor for independent sources (source stepping homotopy) |
| `tnom` | Celsius | Default temperature at which model parameters were extracted |
| `timeUnit` | s | Time unit as specified in 'timescale |
| `timePrecision` | s | Time precision as specified in 'timescale |

#### Usage in Vendored Models

| Parameter | Default | Count | Notes |
|-----------|---------|------:|-------|
| `gmin` | 1e-12 | 88 | Standard LRM parameter |
| `tnom` | 27 | 74 | Standard LRM parameter |
| `scale` | 1 | 41 | Standard LRM parameter |
| `iteration` | 10 | 36 | Standard LRM parameter |
| `iniLim` | -1 | 36 | **Non-standard** (SPICE-specific) |
| `oldlimit` | 0 | 34 | **Non-standard** (SPICE-specific) |
| `defl` | 1e-4 | 30 | **Non-standard** (SPICE default L) |
| `defw` | 1e-4 | 30 | **Non-standard** (SPICE default W) |
| `defas` | 0 | 30 | **Non-standard** (SPICE default AS) |
| `defad` | 0 | 30 | **Non-standard** (SPICE default AD) |
| `reltol` | 1e-3 | 10 | **Non-standard** (SPICE tolerance) |
| `epsmin` | 1e-28 | 10 | **Non-standard** |
| `abstol` | 1e-12 | 6 | **Non-standard** (SPICE tolerance) |
| `vntol` | 1e-6 | 6 | **Non-standard** (SPICE voltage tolerance) |

| Path | Handling |
|------|----------|
| **JAX** | Only `gmin` handled → `inputs[-1]`. Others return 0.0 or default. |
| **OSDI** | `OsdiSimParas` struct with `names[]` and `vals[]` arrays |

**Gap**: JAX only handles `gmin`. Models using `scale`, `tnom`, or iteration-dependent behavior may differ.

### `$mfactor` (191 uses)

Shunt multiplicity factor (LRM 9.18).

> **LRM Definition**: `$mfactor` is the shunt multiplicity factor of the instance, that is, the number of identical devices that should be combined in parallel and modeled. The value is computed by multiplying values from the top of the hierarchy down to the instance.

**Hierarchy Combination**: `$mfactor = $mfactor_specified * $mfactor_hier`

**Top-level value**: 1.0

| Path | Handling |
|------|----------|
| **JAX** | Param with `kind='sysfun'` → uses `mfactor` argument (default 1.0) |
| **OSDI** | Instance parameter in OSDI descriptor |

### `$vt` (28 uses)

Thermal voltage kT/q (LRM 9.15).

> **LRM Definition**: `$vt` can optionally have temperature (in Kelvin units) as an input argument and returns the thermal voltage (kT/q) at the given temperature. `$vt` without the optional input temperature argument returns the thermal voltage using `$temperature`.

**Formula**: `$vt(T) = k * T / q` where k = Boltzmann constant, q = electron charge

Usage patterns found:
- `$vt` or `$vt()` - 21 uses - thermal voltage at simulation temperature
- `$vt($temperature)` - 4 uses - explicit simulation temperature
- `$vt(Tnom)` - 1 use - thermal voltage at nominal temperature
- `$vt(DevTemp)` - 1 use - thermal voltage at device temperature

| Path | Handling |
|------|----------|
| **JAX** | Computed inline as `k * T / q` where T comes from temperature param |
| **OSDI** | Computed inline |

### `$abstime` (5 uses)

Absolute simulation time (LRM Table 9-7). Only relevant for transient analysis.

> **LRM Definition**: `$abstime` returns the absolute time as a real number in seconds. The absolute time at the beginning of a simulation is typically zero.

| Path | Handling |
|------|----------|
| **JAX** | Param with `kind='abstime'` → defaults to 0.0. For transient, caller updates the input. |
| **OSDI** | `OsdiSimInfo.abstime` field |

**Note**: For DC analysis, `abstime=0.0` is correct. For transient analysis, the simulator must provide the current time value.

---

## Device Behavior Functions

### `$param_given` (1891 uses)

Checks if a parameter was explicitly provided vs using default (LRM 9.19).

> **LRM Definition**: The `$param_given()` function can be used to determine whether a parameter value was obtained from the default value in its declaration statement or if that value was overridden. The return value shall be one (1) if the parameter was overridden, either by a `defparam` statement or by a module instance parameter value assignment, and zero (0) otherwise.

**Note**: The return value is constant during simulation (fixed at elaboration). This allows use in genvar expressions controlling conditional/looping behavior of analog operators.

```verilog
if ($param_given(VTH0)) begin
    // Use provided VTH0
end else begin
    // Calculate VTH0 from other params
end
```

| Path | Handling |
|------|----------|
| **JAX** | Param with `kind='param_given'` → 1.0 if param in user dict, else 0.0 |
| **OSDI** | Tracked per-parameter in OSDI descriptor |

### `$limit` (270 uses)

Convergence limiting functions (LRM 9.17.3).

> **LRM Definition**: The `$limit()` function is a special-purpose system function whose purpose, like that of `limexp()`, is to provide the simulator with help in managing convergence problems. `$limit()` has internal state containing information about the argument from iteration to iteration. It internally limits the change of the output from iteration to iteration in order to improve convergence.

**Built-in limiting algorithms**:
- `pnjlim` - Limiting arguments to exponentials (PN junction)
  - Additional args: `vte` (step size), `vcrit` (critical voltage)
- `fetlim` - Limiting potential across MOS oxide
  - Additional arg: `vto` (threshold voltage)
- User-defined functions can also be used

Common patterns:
- `$limit(V(g, ...), "fetlim", ...)` - 76 uses
- `$limit(V(b, ...), "pnjlim", ...)` - 72 uses
- `$limit(V(d_int, ...), ...)` - 40 uses

| Path | Handling |
|------|----------|
| **JAX** | Translated to JAX limiting functions |
| **OSDI** | Native OSDI limiting support |

### `$port_connected` (57 uses)

Checks if an optional port is connected (LRM 9.19).

> **LRM Definition**: The `$port_connected()` function can be used to determine whether a connection was specified for a port. The return value shall be one (1) if the port was connected to a net (by order or by name) when the module was instantiated, and zero (0) otherwise. Note that the port may be connected to a net that has no other connections, but `$port_connected()` shall still return one.

**Note**: The return value is constant during simulation (fixed at elaboration).

```verilog
if ($port_connected(bulk)) begin
    // 4-terminal device
end else begin
    // 3-terminal device (bulk tied internally)
end
```

| Path | Handling |
|------|----------|
| **JAX** | Assumed all ports connected (returns true) |
| **OSDI** | Proper port connection checking |

**Gap**: Models with optional ports may behave differently.

### `$discontinuity` (42 uses)

Notifies simulator of a discontinuity for timestep control (LRM 9.17.1).

> **LRM Definition**: The `$discontinuity` task is used to give hints to the simulator about the behavior of the module so the simulator can control its simulation algorithms to properly handle the discontinuity. `$discontinuity(i)` implies a discontinuity in the i'th derivative of the constitutive equation. `$discontinuity(0)` indicates a discontinuity in the equation, `$discontinuity(1)` indicates a discontinuity in slope, etc.

**Special case**: `$discontinuity(-1)` is used with the `$limit()` function.

```verilog
$discontinuity(-1);  // Always notify (42 uses found)
```

| Path | Handling |
|------|----------|
| **JAX** | Ignored (DC analysis only currently) |
| **OSDI** | Passed to simulator for timestep control |

---

## Debug Functions (Ignored)

These functions are for debugging/output and are safely ignored during simulation.

| Function | Count | Purpose |
|----------|------:|---------|
| `$strobe` | 1434 | Print at end of timestep |
| `$write` | 251 | Print immediately |
| `$warning` | 212 | Issue warning message |
| `$finish` | 92 | End simulation |
| `$fatal` | 64 | Fatal error |
| `$display` | 11 | Print with newline |
| `$debug` | 4 | Debug output |
| `$fopen` | 2 | Open file |
| `$fdisplay` | 2 | Write to file |

---

## Gap Analysis

### Currently Missing in JAX Path

1. **`$simparam` parameters beyond `gmin`**
   - `tnom` - May affect temperature-dependent calculations
   - `scale` - Layout scaling (41 uses)
   - `iteration` - Iteration-dependent limiting (36 uses)
   - Tolerance values - May affect limiting behavior

2. **`$port_connected`** - Always returns true
   - Risk: Models with optional ports (bulk, substrate) may behave differently

3. **`$vt(T)` with explicit temperature**
   - Need to verify correct handling of `$vt(Tnom)` vs `$vt($temperature)`

### Recommendations

1. **Add `tnom` simparam support** - Used in 74 places for temperature normalization
2. **Add `scale` simparam support** - Used in 41 places for geometry scaling
3. **Consider `$port_connected`** - May explain differences in some models
4. **Verify `$vt` temperature handling** - Ensure consistency with OSDI

---

## Implementation Details

### JAX Parameter Handling

Parameters are handled by `kind` in `_build_param_inputs()`:

| Kind | Handling |
|------|----------|
| `param` | User-provided or OSDI default |
| `voltage` | Placeholder 0.0 (set at runtime) |
| `temperature` | Uses `temperature` argument |
| `sysfun` (mfactor) | Uses `mfactor` argument |
| `param_given` | 1.0 if param in user dict, else 0.0 |
| `port_connected` | Always 1.0 (all ports assumed connected) |
| `abstime` | Defaults to 0.0 (DC); caller updates for transient |
| `hidden_state` | Placeholder 0.0 (filled by init) |
| `current` | Placeholder 0.0 |

### Special Input Array Indices

For codegen, special values accessed via negative indices:
```
inputs[-1]  →  gmin (from $simparam("gmin"))
inputs[-2]  →  analysis_type (for analysis() function)
```

### OpenVAF Compile-Time Handling

Some system functions are evaluated at compile time by OpenVAF:
- `$simparam("name", default)` with default → default value inlined
- `$discontinuity` → Simulator hint, stripped during optimization
- `$vt` → Computed inline as `k * T / q`

This means warnings for unknown `$simparam` only appear if OpenVAF preserves the call.

### OSDI Structures

```c
typedef struct OsdiSimParas {
    char** names;      // Array of simparam names
    double* vals;      // Array of simparam values
} OsdiSimParas;

typedef struct OsdiSimInfo {
    double temperature;  // Simulation temperature (K)
    double abstime;      // Absolute time (s)
    // ... other fields
} OsdiSimInfo;
```

---

## References

- [Verilog-AMS LRM 2023 - System Tasks and Functions](../vendor/OpenVAF/docs/VAMS-LRM-2023/09-system-tasks-functions.md) - Section 9
- [Verilog-A Language Reference (SIMetrix)](https://help.simetrix.co.uk/8.0/simetrix/mergedProjects/verilog_a_reference/topics/veriloga_verilog_areference_verilog_afunctions.htm)
- [OSDI v4.0 Specification](../vendor/OpenVAF/docs/osdi_v4p0.md)
- [OpenVAF Source](../vendor/OpenVAF/)

### LRM Section References

| Section | Content |
|---------|---------|
| 9.15 | Analog kernel parameter functions (`$temperature`, `$vt`, `$simparam`, `$simparam$str`) |
| 9.17 | Analog kernel control functions (`$discontinuity`, `$bound_step`, `$limit`) |
| 9.18 | Hierarchical parameter functions (`$mfactor`, `$xposition`, `$yposition`, `$angle`, `$hflip`, `$vflip`) |
| 9.19 | Explicit binding detection (`$param_given`, `$port_connected`) |
| Table 9-27 | Standard `$simparam` parameter names |
| Table 9-28 | Standard `$simparam$str` parameter names |
