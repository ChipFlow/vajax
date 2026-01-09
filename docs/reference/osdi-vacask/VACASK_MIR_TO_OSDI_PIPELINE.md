# MIR to OSDI Pipeline Analysis

This document provides a detailed analysis of the OpenVAF compilation pipeline from Mid-level Intermediate Representation (MIR) through final OSDI 0.4 shared library generation.

## Stage 1: MIR Structure (`openvaf/mir/`)

MIR is a **SSA-form intermediate representation** with:

### Core Data Structures

- **Function**: Container holding `DataFlowGraph`, `Layout`, and source locations
- **DataFlowGraph**: All instructions and values in SSA form with phi nodes
- **InstructionData**: Fixed 16-byte enum covering:
  - Arithmetic: `Fadd`, `Fmul`, `Fdiv`, `Pow`
  - Control: `Branch`, `PhiNode`, `Jump`, `Exit`
  - Calls: External function references
  - Type casts: `FIcast`, `IFcast`, etc.

### SSA Property

MIR is in Static Single Assignment form:
- Each value is assigned exactly once
- All uses refer to that single definition
- Phi nodes merge values from different control flow paths
- This enables efficient data flow analysis and optimizations

### PHI Node Structure

PHI nodes merge values from different control flow paths. OpenVAF uses a B-tree forest for efficient edge management:

**Source**: `mir/src/instructions.rs:311-314, 26-27`

```rust
pub struct PhiNode {
    pub args: ValueList,      // List of incoming values (one per edge)
    pub blocks: PhiMap,       // B-tree map: Block → index into args
}

pub type PhiForest = bforest::MapForest<Block, u32>;
pub type PhiMap = bforest::Map<Block, u32>;
```

**Key operations** (`mir/src/dfg/phis.rs`):
- `insert_phi_edge(inst, block, val)` - Add/update edge from block
- `try_remove_phi_edge_at(inst, block)` - Remove edge from block
- `phi_edges(phi)` - Iterate over all (block, value) pairs
- `phi_edge_val(phi, pred)` - Get value for specific predecessor

The `PhiMap` uses a B-tree forest data structure for O(log n) access to phi edges and handles sparse predecessor sets efficiently.

## Stage 2: Simulation Backend (`openvaf/sim_back/`)

Transforms raw MIR into DAE system form:

### 2.1 Topology Analysis

Maps `contribute` statements to nodes/branches:
- Identifies implicit equations (for nonlinear elements)
- Categorizes contributions:
  - **Resistive** (instantaneous): `I(V)` current contributions
  - **Reactive** (memory): `Q(V)` charge contributions
- Groups into `Branch` objects per 2-terminal device

### 2.2 DAE Construction

Builds `I(x) + dQ/dt = 0` system with:

```rust
pub enum SimUnknownKind {
    KirchoffLaw(Node),      // V(node) voltage
    Current(CurrentKind),   // I(branch) current
    Implicit(ImplicitEq),   // Hidden variable from implicit element
}

pub struct Residual {
    pub resist: Value,           // I(x) - resistive part
    pub react: Value,            // Q(x) - reactive part
    pub resist_small_signal: Value,
    pub react_small_signal: Value,
    pub resist_lim_rhs: Value,   // Limiting correction terms
    pub react_lim_rhs: Value,
}

pub struct MatrixEntry {
    pub row: SimUnknown,
    pub col: SimUnknown,
    pub resist: Value,  // dI/dx (resistance)
    pub react: Value,   // dQ/dx (capacitance)
}
```

### 2.3 Code Separation

Generates 3 MIR variants:
1. **Model setup**: Runs once per model definition
2. **Instance setup**: Cached values computation
3. **Evaluation**: Per Newton iteration

## Stage 3: Automatic Differentiation (`openvaf/mir_autodiff/`)

Generates Jacobian code via **reverse-mode differentiation**:

- Uses chain rule: `d/dx(f(g(x))) = (df/dg) * (dg/dx)`
- `LiveDerivatives` prunes unused derivative computations
- Special handling for:
  - Power functions (guards for x=0)
  - Phi nodes (deferred for cycles)
  - Integer/boolean ops (zero derivatives)

### Differentiation Examples

For `Fadd`:
```
If C = A + B, then:
  dC/dx = dA/dx + dB/dx
```

For `Fmul`:
```
If C = A * B, then:
  dC/dx = A * dB/dx + B * dA/dx
```

### PHI Node Handling in Autodiff

PHI nodes require special handling in autodiff because they can form cycles (back-edges in loops).

**Acyclic PHI nodes** - derivatives computed immediately:
```rust
// mir_autodiff/src/builder.rs:250-266
for derivative in derivatives.iter() {
    let edges: Vec<_> = self.func.dfg.phi_edges(&phi)
        .map(|(bb, val)| (bb, self.derivative_of(val, derivative)))
        .collect();

    // Optimization: if all edges have same derivative, skip PHI
    if edges.iter().all(|(_, val)| *val == edges[0].1) {
        self.insert_derivative(prev_order, unknown, edges[0].1);
    } else {
        let val = self.ins().phi(&edges);
        self.derivative_values.insert((prev_order, unknown), val);
    }
}
```

**Cyclical PHI nodes** - deferred resolution:
```rust
// mir_autodiff/src/builder.rs:232-247
if is_cyclical {
    // Create placeholder PHI with dummy values
    for derivative in derivatives.iter() {
        let edges: Vec<_> = self.func.dfg.phi_edges(&phi).collect();
        let val = self.ins().phi(&edges);  // Placeholder
        self.derivative_values.insert((prev_order, unknown), val);

        // Remember to fix up later
        self.cyclical_phis.push((self.dst.0, derivative));
    }
}

// After all blocks processed (builder.rs:126-134):
for (phi, derivative) in &self.cyclical_phis {
    self.func.dfg.zap_inst(*phi);  // Clear existing args
    for arg in self.func.dfg.instr_args_mut(*phi) {
        *arg = Self::derivative_of_(..., *arg, *derivative);
    }
    self.func.dfg.update_inst_uses(*phi);
}
```

### Power Function PHI Guards

For `pow(base, exp)` where `base` could be zero, derivatives would produce `NaN` or `Inf`. OpenVAF inserts guarded PHI nodes for numerical stability:

```rust
// mir_autodiff/src/builder.rs:300-315
// Create: if (base == 0) { derivative = 0 } else { derivative = computed }
let checked_val = self.ins().phi(&[
    (old_block, F_ZERO),                    // base=0 path
    (calculate_derivative_block, val)       // normal path
]);
```

## Stage 4: MIR Optimization (`openvaf/mir_opt/`)

Optimization passes in sequence:

| Pass | Purpose |
|------|---------|
| `global_value_numbering.rs` | Eliminate redundant computations |
| `const_prop.rs` | Evaluate constants at compile-time |
| `dead_code.rs` | Remove unused instructions |
| `simplify_cfg.rs` | Clean up control flow |
| `inst_combine.rs` | Algebraic simplifications |

## Stage 5: LLVM Codegen (`openvaf/mir_llvm/`)

Maps MIR to LLVM IR:

- `LLVMBackend`: Manages target triple, CPU features
- `CodegenCx`: Type system (all numerics → `f64`)
- `Builder`: Instruction emission via LLVM C API

Opcode mapping:
```
Fadd → LLVMBuildFAdd
Fmul → LLVMBuildFMul
Pow  → call @pow intrinsic
Sin  → call @sin libm
```

### PHI Resolution in LLVM Codegen

LLVM PHI nodes require all incoming values to be defined before the PHI can be completed. OpenVAF uses a **two-phase approach**:

**Source**: `mir_llvm/src/builder.rs:512-536, 626-642`

**Phase 1**: During block traversal, create placeholder PHI nodes:
```rust
// builder.rs:626-642
InstructionData::PhiNode(ref phi) => {
    // Get type from any existing edge
    let ty = self.func.dfg.phi_edges(phi)
        .find_map(|(_, val)| self.values[val].get_ty(self))
        .unwrap();

    // Create empty LLVM PHI
    let llval = LLVMBuildPhi(self.llbuilder, ty, UNNAMED);

    // Store for later completion
    self.unfinished_phis.push((phi.clone(), llval_ref));
    self.values[res] = BuilderVal::Eager(llval_ref);
}
```

**Phase 2**: After all blocks processed, populate PHI edges:
```rust
// builder.rs:512-534
for (phi, llval) in self.unfinished_phis.iter() {
    let (blocks, vals): (Vec<_>, Vec<_>) = self.func.dfg
        .phi_edges(phi)
        .map(|(bb, val)| {
            // Position before terminator to get correct value
            self.select_bb_before_terminator(bb);
            (self.blocks[bb].unwrap(), self.values[val].get(self))
        })
        .unzip();

    LLVMAddIncoming(llval, incoming_vals, incoming_blocks, vals.len());
}
self.unfinished_phis.clear();
```

**Key insight**: Values are retrieved by positioning the builder **before the terminator** of each predecessor block, ensuring we get the correct value that flows along that edge.

## Stage 6: OSDI Generation (`openvaf/osdi/`)

Generates 4 object files per module:

| File | Function | Purpose |
|------|----------|---------|
| `access_N.o` | `access()` | Parameter read/write via switch table |
| `setup_model_N.o` | `setup_model()` | Model-level initialization |
| `setup_inst_N.o` | `setup_inst()` | Instance cache computation |
| `eval_N.o` | `eval()` | Residual + Jacobian computation |

### Eval Function Structure

```c
int eval(osdi_instance_t *inst, osdi_model_t *model,
         osdi_sim_info_t *sim_info, int *flags) {
    // 1. Load V(nodes) from sim_info
    // 2. Load parameters
    // 3. Execute evaluation MIR → compute I(x), ∂I/∂x, Q(x), ∂Q/∂x
    // 4. Store to sim_info->residual[], sim_info->jacobian[]
    // 5. Return convergence status
}
```

## Stage 7: Linking (`openvaf/linker/`)

Platform-specific linking:
- Linux: `ld` or `ld64.lld`
- macOS: Xcode `ld64`
- Windows: MSVC `link.exe`

Links module objects + `stdlib_bitcode.bc` → final `.osdi` shared library.

## Data Flow Summary

```
Verilog-A → HIR → MIR → Topology/DAE → Autodiff → MIR Opt → LLVM IR → .osdi
                        ↓
                   DAE System:
                   - Unknowns: V(nodes), I(branches)
                   - Residuals: I(x) + dQ/dt
                   - Jacobian: sparse ∂I/∂x, ∂Q/∂x
```

## Key Design Insights

1. **Sparse Jacobian**: GVN identifies common subexpressions across Jacobian rows, dead code removes unused derivatives

2. **Three-tier init**: Model setup → Instance setup → Eval separation minimizes per-iteration work

3. **OSDI 0.4 extensions**: Parameter-given flags, offset-based Jacobian loading for harmonic balance, nature/discipline descriptors

---

## Deep Dive: OSDI Function Generation

This section provides detailed analysis of how the three main OSDI functions (`setup_model`, `setup_instance`, and `eval`) are generated.

### MIR Separation into Three Functions

The key insight is that OpenVAF generates **three separate MIR functions** from a single Verilog-A module, each serving a different purpose in the simulation lifecycle.

#### Source: `sim_back/src/lib.rs` - `CompiledModule::new()`

```rust
pub struct CompiledModule<'a> {
    pub eval: Function,              // Evaluation MIR (per Newton iteration)
    pub init: Initialization,        // Instance setup MIR
    pub model_param_setup: Function, // Model setup MIR
    // ... other fields
}
```

The separation happens through **operating-point dependency analysis**:

1. **Model setup MIR**: Only parameter validation and defaulting logic
2. **Instance setup MIR**: Instructions that don't depend on operating point (voltages/currents)
3. **Eval MIR**: All operating-point-dependent instructions (the hot path)

#### Source: `sim_back/src/init.rs` - `Initialization::new()`

The `Initialization` builder traverses all blocks in the eval MIR and **splits** instructions:

```rust
fn split_block(&mut self, bb: Block) {
    for inst in block_insts {
        if self.op_dependent_insts.contains(inst) {
            // Keep in eval MIR - depends on voltages/currents
            match inst {
                Branch { else_dst, .. } => {
                    // Convert to unconditional jump in init
                    init.func.ins().jump(else_dst);
                }
                // ... handle other terminators
            }
        } else {
            // Copy to init MIR - can be computed once at setup
            self.copy_instruction(inst, bb);
        }
    }
}
```

**Key data structure - `cached_vals`:**
```rust
pub struct Initialization {
    pub func: Function,                                    // The init MIR
    pub cached_vals: IndexMap<Value, CacheSlot>,          // Values computed in init, used in eval
    pub cache_slots: TiMap<CacheSlot, (ClassId, u32), Type>, // Cache slot metadata
}
```

Values that are:
- Computed in `setup_instance` (OP-independent)
- Used in `eval` (OP-dependent)

...are assigned **cache slots** and stored in the instance data structure between the two function calls.

#### Operating-Point Dependency Analysis Details

This section explains how OpenVAF determines which instructions are OP-dependent and how cache slots are allocated.

**Step 1: Mark OP-Dependent Instructions**

**Source**: `sim_back/src/context.rs:126-189`

```rust
// Mark instructions that depend on operating point
op_dependent_insts.clear();

// 1. Mark voltage/current parameters as OP-dependent
for (param, &val) in self.intern.params.iter() {
    if param.op_dependent() {  // Voltage, Current, ImplicitUnknown
        op_dependent_vals.push(val);
    }
}

// 2. Propagate taint through data flow graph
propagate_taint(
    &self.func,
    &self.dom_tree,
    &self.cfg,
    op_dependent_vals.iter().copied(),
    &mut op_dependent_insts,  // Output: set of tainted instructions
);
```

**What marks a parameter as OP-dependent?**

**Source**: `hir_lower/src/lib.rs` (ParamKind enum)

```rust
impl ParamKind {
    pub fn op_dependent(&self) -> bool {
        matches!(
            self,
            ParamKind::Voltage { .. }          // V(n1, n2)
                | ParamKind::Current(_)        // I(branch)
                | ParamKind::ImplicitUnknown(_) // Hidden state
                | ParamKind::PrevState(_)      // Integration state
                | ParamKind::NewState(_)
        )
    }
}
```

**Step 2: Split Instructions Between Init and Eval**

**Source**: `sim_back/src/init.rs:101-142`

```rust
for inst in block_insts {
    if self.op_dependent_insts.contains(inst) {
        // Keep in EVAL MIR - depends on operating point
        // Don't copy to init
    } else {
        // Copy to INIT MIR - can be computed once at setup
        self.copy_instruction(inst, bb);
    }
}
```

**Step 3: Determine Which Values to Cache**

**Source**: `sim_back/src/init.rs:194-239`

A value gets cached if:
1. **It's computed in init** (OP-independent)
2. **It's used in eval** (after dead code elimination)
3. **OR** it's an output variable (even if not used in eval)

```rust
let is_output = self.func.dfg.insts[inst].opcode() == Opcode::OptBarrier
    && self.output_values.contains(self.func.dfg.first_result(inst));

let cache_inst = !is_output
    && self.func.dfg.inst_results(inst).iter().any(|val| {
        self.func.dfg.tag(*val).is_some()  // Has a name (user variable)
    });

if is_output || cache_inst {
    // Create cache slot for this value
    let param = self.init_cache.insert_full(val, inst).0 + self.intern.params.len();
    self.func.dfg.values.make_param_at(param.into(), val);
}
```

**Step 4: Build Cache Slots with GVN**

**Source**: `sim_back/src/init.rs:240-290`

```rust
// Group cached values by GVN equivalence class
let equiv_class = inst.and_then(|inst| gvn.inst_class(inst));

self.init.cached_vals = self
    .init_cache
    .iter()
    .filter_map(|(&val, &old_inst)| {
        if self.func.dfg.value_dead(val) {
            return None;  // Value not used in eval, don't cache
        }

        let slot = ensure_cache_slot(inst, res, ty);
        Some((val, slot))
    })
    .collect();
```

**Cache Selection Algorithm Summary**:
```
1. Start with OP-dependent seeds: voltages, currents, implicit unknowns
2. Propagate taint forward through data flow graph
3. All tainted instructions stay in EVAL
4. All non-tainted instructions go to INIT
5. Values computed in INIT but used in EVAL → cached
6. GVN groups equivalent cached values into same slot
```

---

### setup_model() Generation

**Purpose**: Initialize model-level parameters, apply defaults, validate ranges.

**Source**: `osdi/src/setup.rs` - `OsdiCompilationUnit::setup_model()`

#### Function Signature
```c
void setup_model(void* handle, void* model, void* simparam, osdi_init_info* result);
```

| Parameter | Description |
|-----------|-------------|
| `handle` | Simulator callback handle |
| `model` | Pointer to model data structure |
| `simparam` | Simulation parameters interface |
| `result` | Output: initialization status and errors |

#### Generation Flow

1. **Create LLVM function prototype**:
```rust
let fun_ty = cx.ty_func(&[cx.ty_ptr(), cx.ty_ptr(), cx.ty_ptr(), cx.ty_ptr()], cx.ty_void());
let llfunc = cx.declare_ext_fn("setup_model_{sym}", fun_ty);
```

2. **Build parameter loading**:
```rust
// For each model parameter
for (i, param) in model_data.params.keys().enumerate() {
    // Load from model structure
    let loc = model_data.nth_param_loc(cx, i, model);
    builder.params[dst] = BuilderVal::Load(loc);

    // Load param_given flag
    let is_given = model_data.is_nth_param_given(cx, i, model, llbuilder);
    builder.params[dst_given] = BuilderVal::Eager(is_given);
}
```

3. **Set up callbacks for validation**:
```rust
// ParamInfo callbacks for invalid parameter errors
if let CallBackKind::ParamInfo(ParamInfoKind::Invalid, param) = call {
    let cb = CallbackFun::Prebuilt(BuiltCallbackFun {
        fun_ty: invalid_param_err.0,
        fun: invalid_param_err.1,
        state: vec![err_ptr, err_len, err_cap, err_param],
    });
    builder.callbacks[call_id] = Some(cb);
}
```

4. **Execute MIR → LLVM translation**:
```rust
builder.build_consts();  // Emit constants
builder.build_func();    // Emit all instructions from model_param_setup MIR
```

5. **Store defaulted parameter values back**:
```rust
// After MIR execution, store computed defaults
for (i, param) in model_data.params.keys().enumerate() {
    let val = intern.outputs[&PlaceKind::Param(*param)];
    let inst = func.dfg.value_def(val).unwrap_inst();
    let bb = func.layout.inst_block(inst).unwrap();
    builder.select_bb_before_terminator(bb);

    let val = builder.values[val].get(&builder);
    model_data.store_nth_param(i, model, val, llbuilder);
}
```

#### Parameter Initialization Flow Details

**Critical Implementation Detail**: Parameters ARE already in the model structure (put there by simulator from netlist).

**Simulator's Responsibilities**:

1. **Before setup_model()**: Simulator allocates model structure and writes:
   - Parameter values from netlist
   - `param_given` flags (true if user specified, false otherwise)

2. **After setup_model()**: Simulator reads validated/defaulted parameter values back.

**Where VA Defaults Get Applied**:

Verilog-A default values (from `parameter real tox = 1e-9;`) are applied in the **model_param_setup MIR**.

The MIR lowering (`hir_lower/src/parameters.rs:83-110`) creates:

```rust
if param_given {
    // User provided value - validate it
    check_param_bounds(param_val, bounds, ops, invalid_callback);
    param_val  // Use user value
} else {
    // User didn't provide - use VA default
    let default_val = lower_expression(param.init(db));  // From .va file
    default_val
}
```

**Example Timeline**:
```
Netlist: resistor r1 (n1, n2) .model myres r=1000

1. Simulator creates model structure for "myres"
2. Simulator writes: model.params[r_idx] = 1000.0, param_given[r_idx] = true
3. Simulator calls setup_model(&handle, &model, &simparam, &result)
4. setup_model loads r=1000, param_given=true
5. setup_model runs MIR: if param_given { validate(1000) } else { use_default }
6. setup_model stores validated r=1000 back to model structure
7. Simulator uses validated parameters
```

#### Callback Semantics

**ParamInfo::Invalid** callbacks push error messages to a result vector for the simulator to display.

**Source**: `osdi/src/setup.rs:113-123`

The callback function signature:
```c
void push_invalid_param_err(
    char** err_ptr,      // Pointer to error string array
    size_t* err_len,     // Current length of error array
    size_t* err_cap,     // Current capacity of error array
    uint32_t param_idx   // Which parameter is invalid
);
```

**Behavior**:
1. **Appends** error to dynamic array in `osdi_init_info` result structure
2. Does **NOT** print to console directly
3. Does **NOT** abort execution
4. Allows **multiple** errors to accumulate

The MIR usage (`hir_lower/src/parameters.rs:88-96`) creates:
```
if param_val < min || param_val > max {
    call invalid_callback(param_idx);
    // Continue execution (don't abort)
}
```

**Other ParamInfo Callbacks**:
- `ParamInfoKind::MinInclusive`: Sets minimum bound metadata
- `ParamInfoKind::MaxInclusive`: Sets maximum bound metadata
- `ParamInfoKind::MinExclusive`: Sets exclusive lower bound metadata
- `ParamInfoKind::MaxExclusive`: Sets exclusive upper bound metadata

---

### setup_instance() Generation

**Purpose**: Initialize instance-level parameters, compute cached values, handle node collapsing.

**Source**: `osdi/src/setup.rs` - `OsdiCompilationUnit::setup_instance()`

#### Function Signature
```c
void setup_instance(
    void* handle,
    void* instance,
    void* model,
    double temperature,
    int connected_terminals,
    void* simparam,
    osdi_init_info* result
);
```

| Parameter | Description |
|-----------|-------------|
| `handle` | Simulator callback handle |
| `instance` | Pointer to instance data structure |
| `model` | Pointer to model data structure |
| `temperature` | Device operating temperature |
| `connected_terminals` | Number of connected ports |
| `simparam` | Simulation parameters |
| `result` | Output: initialization status |

#### Generation Flow

1. **Generate collapse marking helper**:
```rust
fn mark_collapsed(&self) -> (&LLVMValue, &LLVMType) {
    // Creates a helper function to mark node pairs as collapsible
    let fn_type = cx.ty_func(&[cx.ty_ptr(), cx.ty_int()], cx.ty_void());
    let llfunc = cx.declare_int_c_fn("collapse_{sym}", fn_type);
    // ... stores true in collapsed[idx] bitarray
}
```

2. **Load instance/model parameters with fallback logic**:
```rust
for (i, param) in inst_data.params.keys().enumerate() {
    // Check if explicitly set on instance
    let is_inst_given = inst_data.is_nth_param_given(cx, i, instance, llbuilder);

    // If not, check model-level default
    let is_given = builder.select(
        is_inst_given,
        true_,
        model_data.is_nth_inst_param_given(cx, i, model, llbuilder)
    );

    // Load from instance or fall back to model
    let inst_val = inst_data.read_nth_param(i, instance, llbuilder);
    let model_val = model_data.read_nth_inst_param(inst_data, i, model, llbuilder);
    let val = builder.select(is_inst_given, inst_val, model_val);

    // Apply builtin defaults (like $mfactor = 1.0)
    match param {
        OsdiInstanceParam::Builtin(builtin) => {
            let default_val = builtin.default_value();
            let val = builder.select(is_given, val, cx.const_real(default_val));
            inst_data.store_nth_param(i, instance, val, llbuilder);
        }
        // ...
    }
}
```

3. **Store temperature and connected ports**:
```rust
inst_data.store_temperature(&mut builder, instance, temperature);
inst_data.store_connected_ports(&mut builder, instance, connected_terminals);
```

4. **Set up collapse hint callbacks**:
```rust
CallBackKind::CollapseHint(node1, node2) => {
    let node1_idx = dae_system.unknowns.unwrap_index(&SimUnknownKind::KirchoffLaw(node1));
    node_collapse.hint(node1_idx, node2_idx, |pair| {
        state.push(instance);
        state.push(cx.const_unsigned_int(pair));
    });
    CallbackFun::Prebuilt(BuiltCallbackFun {
        fun_ty: mark_collapsed.1,
        fun: mark_collapsed.0,
        state: state.into_boxed_slice(),
        num_state: 2,  // Call with 2 args at a time
    })
}
```

5. **Execute init MIR and store cached values**:
```rust
builder.build_consts();
builder.build_func();

// Store all values that will be needed by eval()
for (&val, &slot) in module.init.cached_vals.iter() {
    let bb = func.layout.inst_block(func.dfg.value_def(val).unwrap_inst()).unwrap();
    builder.select_bb_before_terminator(bb);

    let val = builder.values[val].get(&builder);
    inst_data.store_cache_slot(module, llbuilder, slot, instance, val);
}
```

---

### eval() Generation

**Purpose**: Compute residuals and Jacobian entries for Newton-Raphson iteration.

**Source**: `osdi/src/eval.rs` - `OsdiCompilationUnit::eval()`

#### Function Signature
```c
int eval(void* handle, void* instance, void* model, osdi_sim_info* sim_info);
```

| Parameter | Description |
|-----------|-------------|
| `handle` | Simulator callback handle |
| `instance` | Instance data (params, cache, node mapping) |
| `model` | Model data |
| `sim_info` | Simulation info (voltages, flags, result arrays) |

**Return**: Flags indicating limiting occurred, etc.

#### osdi_sim_info Structure
```c
struct osdi_sim_info {
    void* simparam;           // Simulation parameters
    double abstime;           // Current simulation time
    double* prev_result;      // Previous solution vector (node voltages)
    double* prev_state;       // Previous integration state
    double* next_state;       // Next integration state
    int flags;                // Analysis flags (CALC_RESIST_JACOBIAN, etc.)
    // ... result arrays
};
```

#### Analysis Flags
```rust
pub const CALC_RESIST_RESIDUAL: u32 = 1;     // Compute I(x)
pub const CALC_RESIST_JACOBIAN: u32 = 2;     // Compute dI/dx
pub const CALC_REACT_RESIDUAL: u32 = 4;      // Compute Q(x)
pub const CALC_REACT_JACOBIAN: u32 = 8;      // Compute dQ/dx
pub const CALC_RESIST_LIM_RHS: u32 = 16;     // Limiting RHS
pub const CALC_REACT_LIM_RHS: u32 = 32;
pub const CALC_OP: u32 = 128;                // Store operating point vars
pub const CALC_NOISE: u32 = 256;             // Compute noise
pub const INIT_LIM: u32 = 512;               // Initialize limiting
pub const ENABLE_LIM: u32 = 1024;            // Enable limiting
pub const ANALYSIS_IC: u32 = 2048;           // Initial condition analysis
```

#### Generation Flow

1. **Load previous solution (node voltages)**:
```rust
let prev_result = builder.struct_gep(sim_info_ty, sim_info, 2);
let prev_result = builder.load(cx.ty_ptr(), prev_result);

let prev_solve: TiVec<SimUnknown, _> = module.dae_system.unknowns.indices()
    .map(|node| {
        inst_data.read_node_voltage(cx, node, instance, prev_result, llbuilder)
    })
    .collect();
```

2. **Map MIR parameters to runtime values**:
```rust
let params: TiVec<_, _> = intern.params.iter().map(|(kind, val)| {
    match kind {
        ParamKind::Voltage { hi, lo } => {
            let hi = prev_solve[SimUnknownKind::KirchoffLaw(hi)];
            if let Some(lo) = lo {
                let lo = prev_solve[SimUnknownKind::KirchoffLaw(lo)];
                LLVMBuildFSub(llbuilder, hi, lo, UNNAMED)  // V(hi, lo) = V(hi) - V(lo)
            } else {
                hi  // V(hi) = V(hi) - 0
            }
        }
        ParamKind::Param(param) => {
            // Load from instance or model data
            inst_data.param_loc(cx, OsdiInstanceParam::User(param), instance)
                .unwrap_or_else(|| model_data.param_loc(cx, param, model).unwrap())
        }
        ParamKind::Temperature => inst_data.temperature_loc(cx, instance),
        ParamKind::EnableIntegration => {
            // true if (CALC_REACT_JACOBIAN && !ANALYSIS_IC)
            let is_not_dc = is_flag_set(cx, CALC_REACT_JACOBIAN, flags, llbuilder);
            let is_not_ic = is_flag_unset(cx, ANALYSIS_IC, flags, llbuilder);
            LLVMBuildAnd(llbuilder, is_not_dc, is_not_ic, UNNAMED)
        }
        // ... other parameter kinds
    }
}).collect();
```

#### Parameter Priority Details

**Instance params override model params** - this is the correct behavior for circuit simulation.

**Priority Order (Lowest to Highest)**:
1. **Model-level defaults** (from `(* type="model" *)` params in .va file)
2. **Instance-level overrides** (from `(* type="instance" *)` params)

**Code Evidence** (`osdi/src/setup.rs:366-381`):
```rust
for (i, param) in inst_data.params.keys().enumerate() {
    // Check if explicitly set on instance
    let is_inst_given = inst_data.is_nth_param_given(cx, i, instance, llbuilder);

    // If not, check model-level default
    let is_given = builder.select(
        is_inst_given,
        true_,
        model_data.is_nth_inst_param_given(cx, i, model, llbuilder)
    );

    // Load from instance or fall back to model
    let inst_val = inst_data.read_nth_param(i, instance, llbuilder);
    let model_val = model_data.read_nth_inst_param(inst_data, i, model, llbuilder);
    let val = builder.select(is_inst_given, inst_val, model_val);  // Instance wins
}
```

**Why This Makes Sense**:

Consider MOSFET with model params:
```spice
.model nmos nmos level=54 tox=1e-9 vth0=0.5  ← Model defaults

m1 d g s b nmos w=1u l=0.18u                  ← Instance-specific geometry
m2 d g s b nmos w=10u l=0.18u                 ← Different instance, same model
```

- `tox`, `vth0`: Model params (shared by m1, m2)
- `w`, `l`: Instance params (different for m1 vs m2)

Instance params **must** override model params, otherwise every device would be identical.

3. **Load cached values from instance setup**:
```rust
let cache_vals = (0..module.init.cache_slots.len()).map(|i| {
    let slot = i.into();
    let val = inst_data.load_cache_slot(module, llbuilder, slot, instance);
    BuilderVal::Eager(val)
});
params.extend(cache_vals);
```

4. **Set up limiting callbacks**:
```rust
CallBackKind::BuiltinLimit { name, num_args } => {
    let id = module.lim_table.unwrap_index(&OsdiLimFunction { name, num_args: num_args - 2 });
    CallbackFun::Prebuilt(self.lim_func(id, num_args - 2, &flags, ret_flags))
}
```

5. **Execute evaluation MIR**:
```rust
builder.build_consts();
builder.build_func();  // This computes all residuals and Jacobian entries
```

6. **Store results conditionally based on flags**:
```rust
// For both resistive and reactive parts
for reactive in [false, true] {
    let (jacobian_flag, residual_flag, lim_rhs_flag) = if reactive {
        (CALC_REACT_JACOBIAN, CALC_REACT_RESIDUAL, CALC_REACT_LIM_RHS)
    } else {
        (CALC_RESIST_JACOBIAN, CALC_RESIST_RESIDUAL, CALC_RESIST_LIM_RHS)
    };

    // Store Jacobian entries if flag set
    Self::build_store_results(&mut builder, llfunc, &flags, jacobian_flag, |builder| {
        for entry in module.dae_system.jacobian.keys() {
            inst_data.store_jacobian(entry, instance, builder, reactive);
        }
    });

    // Store residuals if flag set
    Self::build_store_results(&mut builder, llfunc, &flags, residual_flag, |builder| {
        for unknown in module.dae_system.unknowns.indices() {
            inst_data.store_residual(unknown, instance, builder, reactive);
        }
    });
}

// Store operating point variables if requested
Self::build_store_results(&mut builder, llfunc, &flags, CALC_OP, |builder| {
    for (_, &eval_output) in &inst_data.opvars {
        inst_data.store_eval_output(eval_output, instance, builder);
    }
});
```

7. **Conditional store helper**:
```rust
fn build_store_results(
    builder: &mut Builder,
    llfunc: &LLVMValue,
    flags: &MemLoc,
    flag: u32,
    store_val: impl Fn(&mut Builder),
) {
    let bb = LLVMAppendBasicBlockInContext(...);
    let next_bb = LLVMAppendBasicBlockInContext(...);

    // Check if flag is set
    let is_set = is_flag_set_mem(cx, flag, flags, llbuilder);
    LLVMBuildCondBr(llbuilder, is_set, bb, next_bb);

    // If set, execute store operations
    LLVMPositionBuilderAtEnd(llbuilder, bb);
    store_val(builder);
    LLVMBuildBr(llbuilder, next_bb);

    LLVMPositionBuilderAtEnd(llbuilder, next_bb);
}
```

---

### Instance Data Structure Layout

**Source**: `osdi/src/inst_data.rs`

The instance data structure is generated dynamically based on the model:

```rust
pub const NUM_CONST_FIELDS: u32 = 8;
pub const PARAM_GIVEN: u32 = 0;        // Bitfield: which params are set
pub const JACOBIAN_PTR_RESIST: u32 = 1; // Array of pointers to resist Jacobian storage
pub const JACOBIAN_PTR_REACT: u32 = 2;  // Array of pointers to react Jacobian storage
pub const NODE_MAPPING: u32 = 3;        // Maps internal nodes to simulator indices
pub const COLLAPSED: u32 = 4;           // Bitarray: which node pairs are collapsed
pub const TEMPERATURE: u32 = 5;         // Device temperature
pub const CONNECTED: u32 = 6;           // Number of connected ports
pub const STATE_IDX: u32 = 7;           // Integration state indices

// Dynamic fields follow:
// - Instance parameters (builtin + user)
// - Cache slots (computed in setup_instance)
// - Eval output slots (residuals, Jacobian entries, opvars)
```

The struct is built in `OsdiInstanceData::new()`:
```rust
let fields: Vec<_> = static_fields.iter()
    .chain(params.values())           // Instance parameters
    .chain(cache_slots.iter())        // Cached init values
    .chain(eval_outputs.values())     // Result storage
    .collect();

let ty = cx.ty_struct(&format!("osdi_inst_data_{name}"), &fields);
```

---

### Instance Parameter Derivation

Instance parameters come from **three sources**, combined in order in `OsdiInstanceData::new()`.

#### Source 1: Builtin System Parameters (`ParamSysFun`)

**Source**: `hir_def/src/builtin.rs:129`

```rust
pub enum ParamSysFun {
    mfactor,    // Multiplicity factor (default: 1.0)
    xposition,  // X position (default: 0.0)
    yposition,  // Y position (default: 0.0)
    angle,      // Rotation angle (default: 0.0)
    hflip,      // Horizontal flip (default: 1.0)
    vflip,      // Vertical flip (default: 1.0)
}
```

These are **only included if actually used** in the model code (checked via liveness analysis):

```rust
// osdi/src/inst_data.rs:232-238
let builtin_inst_params = ParamSysFun::iter().filter_map(|param| {
    let is_live = |intern: &HirInterner, func| {
        intern.is_param_live(func, &ParamKind::ParamSysFun(param))
    };
    let is_live = is_live(module.intern, module.eval)
        || is_live(&module.init.intern, &module.init.func);
    is_live.then_some((OsdiInstanceParam::Builtin(param), ty_f64))
});
```

#### Source 2: Aliased System Parameters

Any user-defined aliases to system parameters (via `aliasparam`):

```rust
// osdi/src/inst_data.rs:240-244
let alias_inst_params = module
    .info
    .sys_fun_alias
    .keys()
    .map(|param| (OsdiInstanceParam::Builtin(*param), ty_f64));
```

#### Source 3: User-Declared Instance Parameters

**Source**: `sim_back/src/module_info.rs:162-173`

Parameters are classified as instance vs model based on the `type` attribute in Verilog-A:

```verilog
// In Verilog-A source:
parameter real tox = 1e-9 (* type="instance" *);  // Instance param
parameter real vth0 = 0.5 (* type="model" *);     // Model param (default)
parameter real l = 1e-6;                           // Model param (no attribute = model)
```

The parsing logic in `ModuleInfo::collect()`:

```rust
// sim_back/src/module_info.rs:162-173
let type_ = param.get_attr(db, &ast, "type").and_then(|attr| {
    attr.val().and_then(|e| e.as_str_literal())
});
let is_instance = match type_.as_deref() {
    Some("instance") => true,
    Some("model") | None => false,
    Some(found) => {
        // Warning: unknown type, defaults to model
        false
    }
};
```

Then filtered when building `OsdiInstanceData`:

```rust
// osdi/src/inst_data.rs:245-247
let user_inst_params = module.info.params.iter().filter_map(|(param, info)| {
    info.is_instance.then(|| (OsdiInstanceParam::User(*param), lltype(&param.ty(db), cx)))
});
```

#### Combined Parameter Order

```rust
// osdi/src/inst_data.rs:248-249
params.extend(builtin_inst_params.chain(alias_inst_params).chain(user_inst_params));
```

The final instance params list is:
```
[live builtins] ++ [aliased sys params] ++ [user params with type="instance"]
```

#### Default Value Application

In `setup_instance()`, defaults are applied differently for builtin vs user params:

```rust
// osdi/src/setup.rs:182-196
match *param {
    OsdiInstanceParam::Builtin(builtin) => {
        // Builtin defaults defined in hir_def/src/lib.rs:39-44
        let default_val = builtin.default_value();  // e.g., mfactor → 1.0
        let val = builder.select(is_given, val, default_val);
        inst_data.store_nth_param(i, instance, val, llbuilder);
    }
    OsdiInstanceParam::User(param) => {
        // User params use Verilog-A default from parameter declaration
        // Handled by init MIR execution
    }
}
```

Builtin default values (`hir_def/src/lib.rs:39-44`):
```rust
impl ParamSysFun {
    pub fn default_value(self) -> f64 {
        match self {
            ParamSysFun::vflip | ParamSysFun::hflip | ParamSysFun::mfactor => 1f64,
            ParamSysFun::xposition | ParamSysFun::yposition | ParamSysFun::angle => 0f64,
        }
    }
}
```

---

### Evaluation Output Tracking

The eval function needs to know where each computed value should be stored:

```rust
pub enum EvalOutput {
    Calculated(EvalOutputSlot),  // Value computed in eval, stored in slot
    Const(Const, Option<Slot>),  // Compile-time constant
    Param(Param),                // Already stored in instance/model data
    Cache(CacheSlot),            // Computed in setup_instance
}
```

For residuals:
```rust
pub struct Residual {
    pub resist: Option<EvalOutputSlot>,      // I(x) storage
    pub react: Option<EvalOutputSlot>,       // Q(x) storage
    pub resist_lim_rhs: Option<EvalOutputSlot>,
    pub react_lim_rhs: Option<EvalOutputSlot>,
}
```

For Jacobian entries:
```rust
pub struct MatrixEntry {
    pub resist: Option<EvalOutput>,  // dI/dx value
    pub react: Option<EvalOutput>,   // dQ/dx value
    pub react_off: Option<Offset>,   // Offset in reactive Jacobian array
}
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│              Verilog-A Source Code                       │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │  HIR/Desugaring         │
        │  (hir_def, hir_lower)   │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  MIR Generation         │
        │  (mir_build)            │
        └────────────┬────────────┘
                     │
    ┌────────────────▼────────────────────┐
    │     Simulation Backend              │
    │  ┌────────────────────────────────┐ │
    │  │ 1. Topology Analysis           │ │
    │  │ 2. DAE System Construction     │ │
    │  │ 3. Initialization Separation   │ │
    │  └────────────────────────────────┘ │
    └────────────────┬────────────────────┘
                     │
    ┌────────────────▼──────────────────────┐
    │ Automatic Differentiation             │
    │ (mir_autodiff)                        │
    │ - Computes Jacobian ∂I/∂x             │
    │ - Handles higher-order derivatives    │
    │ - Prunes unused derivatives           │
    └────────────────┬──────────────────────┘
                     │
    ┌────────────────▼──────────────────────┐
    │ MIR Optimization                      │
    │ (mir_opt)                             │
    │ - GVN (remove redundant computes)     │
    │ - Dead code elimination               │
    │ - Constant propagation                │
    │ - Control flow simplification         │
    └────────────────┬──────────────────────┘
                     │
    ┌────────────────▼──────────────────────┐
    │ LLVM Code Generation                  │
    │ (mir_llvm)                            │
    │ - MIR → LLVM IR                       │
    │ - Function/callback setup             │
    │ - Optimization passes                 │
    └────────────────┬──────────────────────┘
                     │
    ┌────────────────▼──────────────────────┐
    │ OSDI Binding Generation               │
    │ (osdi)                                │
    │ ┌────────────────────────────────┐   │
    │ │ Access function (param access) │   │
    │ │ Setup functions (init code)    │   │
    │ │ Eval function (residual+J)     │   │
    │ │ Metadata/descriptors           │   │
    │ └────────────────────────────────┘   │
    └────────────────┬──────────────────────┘
                     │
        ┌────────────▼────────────┐
        │  Object File Emission    │
        │  (per module × 4 files)  │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Linker                  │
        │  (linker)                │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  .osdi Shared Library    │
        │  (OSDI 0.4)              │
        └──────────────────────────┘
```

---

## OSDI Function Lifecycle

The three generated functions are called at different stages of a circuit simulation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SIMULATION LIFECYCLE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    MODEL INITIALIZATION                              │    │
│  │                       (Once per model)                               │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐│    │
│  │  │  setup_model(handle, model, simparam, result)                   ││    │
│  │  │    - Load model parameters from netlist                         ││    │
│  │  │    - Apply default values                                       ││    │
│  │  │    - Validate parameter ranges                                  ││    │
│  │  │    - Store validated parameters in model struct                 ││    │
│  │  └─────────────────────────────────────────────────────────────────┘│    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   INSTANCE INITIALIZATION                            │    │
│  │                    (Once per device instance)                        │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐│    │
│  │  │  setup_instance(handle, inst, model, temp, ports, sim, result)  ││    │
│  │  │    - Load instance params (fall back to model if not set)       ││    │
│  │  │    - Store temperature, connected ports                         ││    │
│  │  │    - Execute OP-independent computations                        ││    │
│  │  │    - Store cached values in instance struct                     ││    │
│  │  │    - Handle node collapse hints                                 ││    │
│  │  └─────────────────────────────────────────────────────────────────┘│    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      NEWTON-RAPHSON LOOP                             │    │
│  │                  (Many times per operating point)                    │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐│    │
│  │  │  while (!converged) {                                           ││    │
│  │  │      eval(handle, inst, model, sim_info)  ─────────────────────▶││    │
│  │  │        - Load V(nodes) from sim_info.prev_result                ││    │
│  │  │        - Load cached values from instance                       ││    │
│  │  │        - Execute device equations (MIR)                         ││    │
│  │  │        - Compute residuals: I(x), Q(x)                          ││    │
│  │  │        - Compute Jacobian: ∂I/∂x, ∂Q/∂x                         ││    │
│  │  │        - Store results based on flags                           ││    │
│  │  │                                                                 ││    │
│  │  │      // Simulator updates V(nodes) using J⁻¹·f                  ││    │
│  │  │  }                                                              ││    │
│  │  └─────────────────────────────────────────────────────────────────┘│    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Performance Implications

| Function | Call Frequency | Optimization Priority |
|----------|----------------|----------------------|
| `setup_model` | Once per model type | Low - runs once |
| `setup_instance` | Once per device | Medium - runs O(devices) times |
| `eval` | Many per timestep | **Critical** - runs O(iterations × timesteps × devices) |

The MIR separation ensures that:
1. **Parameter validation** runs only once
2. **OP-independent calculations** (like temperature-dependent model params) run once per device
3. **OP-dependent calculations** (the hot path) are minimized

### Cache Slot Example

For a simple MOSFET, the cache might contain:
```
Cache Slot 0: Vth (threshold voltage, depends on temp/process)
Cache Slot 1: Cox (oxide capacitance, depends on geometry)
Cache Slot 2: μeff (effective mobility, depends on temp)
...
```

These are computed once in `setup_instance` and reused every `eval` call.

---

## Summary

The MIR to OSDI pipeline transforms Verilog-A into optimized native code through:

1. **MIR Generation**: SSA-form intermediate representation with automatic differentiation
2. **Operating-Point Analysis**: Separates code into model/instance/eval tiers
3. **Optimization**: GVN, DCE, constant propagation reduce Jacobian computation
4. **LLVM Codegen**: Generates efficient native code for target platform
5. **OSDI Binding**: Creates simulator-compatible interface with sparse Jacobian support

The result is a `.osdi` shared library that simulators can load to efficiently evaluate device models in Newton-Raphson iterations.

---

## Appendix: PHI Resolution Summary

PHI nodes are handled differently across the compilation pipeline:

| Context | Strategy | Key Mechanism |
|---------|----------|---------------|
| **MIR structure** | B-tree forest for block→value mapping | `PhiMap` with O(log n) lookups |
| **LLVM codegen** | Two-phase: create placeholder, then populate | `unfinished_phis` vector + `LLVMAddIncoming` |
| **Autodiff (acyclic)** | Direct derivative PHI creation | Immediate edge mapping with `derivative_of()` |
| **Autodiff (cyclic)** | Deferred: placeholder + fixup pass | `cyclical_phis` vector for back-edges |
| **Power functions** | Guarded PHI for numerical stability | `base == 0` conditional branch |

This multi-context handling ensures correct SSA semantics while enabling efficient Jacobian computation for circuit simulation.
