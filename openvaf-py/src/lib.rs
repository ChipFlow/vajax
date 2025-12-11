use std::ffi::c_void;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use basedb::diagnostics::ConsoleSink;
use hir::{CompilationDB, CompilationOpts};
use hir_lower::{CurrentKind, ParamKind};
use lasso::Rodeo;
use mir::{FuncRef, Function, Param, Value};
use mir_interpret::{Interpreter, InterpreterState, Data};
use paths::AbsPathBuf;
use sim_back::{collect_modules, CompiledModule};
use typed_index_collections::{TiSlice, TiVec};

/// Python wrapper for a compiled Verilog-A module
#[pyclass]
struct VaModule {
    /// Module name
    #[pyo3(get)]
    name: String,
    /// Parameter names in order (for eval function)
    #[pyo3(get)]
    param_names: Vec<String>,
    /// Parameter types/kinds
    #[pyo3(get)]
    param_kinds: Vec<String>,
    /// Parameter Value indices (for debugging)
    #[pyo3(get)]
    param_value_indices: Vec<u32>,
    /// Node names
    #[pyo3(get)]
    nodes: Vec<String>,
    /// Number of residual equations
    #[pyo3(get)]
    num_residuals: usize,
    /// Number of Jacobian entries
    #[pyo3(get)]
    num_jacobian: usize,
    /// Residual variable indices (as Value u32)
    residual_resist_indices: Vec<u32>,
    residual_react_indices: Vec<u32>,
    /// Jacobian variable indices (as Value u32)
    jacobian_resist_indices: Vec<u32>,
    jacobian_react_indices: Vec<u32>,
    /// Jacobian row/col structure
    jacobian_rows: Vec<u32>,
    jacobian_cols: Vec<u32>,
    /// The compiled MIR function for evaluation
    eval_func: Function,
    /// Number of eval function parameters
    #[pyo3(get)]
    func_num_params: usize,
    /// Callback descriptions (for debugging)
    #[pyo3(get)]
    callback_names: Vec<String>,

    // Init function support
    /// The init MIR function
    init_func: Function,
    /// Init function parameter names
    #[pyo3(get)]
    init_param_names: Vec<String>,
    /// Init function parameter kinds
    #[pyo3(get)]
    init_param_kinds: Vec<String>,
    /// Init function parameter Value indices
    #[pyo3(get)]
    init_param_value_indices: Vec<u32>,
    /// Number of init function parameters
    #[pyo3(get)]
    init_num_params: usize,
    /// Cache slot mapping: (init_value_index, eval_param_index)
    /// Values computed by init that are passed to eval
    cache_mapping: Vec<(u32, u32)>,
    /// Number of cached values from init
    #[pyo3(get)]
    num_cached_values: usize,
}

#[pymethods]
impl VaModule {
    /// Get residual structure as (row_indices, resist_var_indices, react_var_indices)
    fn get_residual_structure(&self) -> (Vec<usize>, Vec<u32>, Vec<u32>) {
        let rows: Vec<usize> = (0..self.num_residuals).collect();
        (rows, self.residual_resist_indices.clone(), self.residual_react_indices.clone())
    }

    /// Get Jacobian structure as (row_indices, col_indices, resist_var_indices, react_var_indices)
    fn get_jacobian_structure(&self) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
        (
            self.jacobian_rows.clone(),
            self.jacobian_cols.clone(),
            self.jacobian_resist_indices.clone(),
            self.jacobian_react_indices.clone(),
        )
    }

    /// Get all Param-defined values in the function
    /// Returns list of (param_index, value_index) tuples
    fn get_all_func_params(&self) -> Vec<(u32, u32)> {
        let mut result = Vec::new();
        for val in self.eval_func.dfg.values.iter() {
            if let mir::ValueDef::Param(p) = self.eval_func.dfg.value_def(val) {
                result.push((u32::from(p), u32::from(val)));
            }
        }
        result.sort_by_key(|(param_idx, _)| *param_idx);
        result
    }

    /// Get MIR function as string for debugging
    fn get_mir(&self, literals: Vec<String>) -> String {
        // Create a simple Rodeo with the provided literals
        let mut rodeo = lasso::Rodeo::new();
        for lit in literals {
            rodeo.get_or_intern(&lit);
        }
        self.eval_func.print(&rodeo).to_string()
    }

    /// Get number of function calls (built-in functions) in the MIR
    fn get_num_func_calls(&self) -> usize {
        self.eval_func.dfg.signatures.len()
    }

    /// Get cache mapping as list of (init_value_idx, eval_param_idx)
    fn get_cache_mapping(&self) -> Vec<(u32, u32)> {
        self.cache_mapping.clone()
    }

    /// Export MIR instructions for JAX translation
    /// Returns a dict with:
    ///   - 'constants': Dict[str, float] - named constants (v123 -> value)
    ///   - 'params': List[str] - parameter names (v16, v17, etc.)
    ///   - 'instructions': List[Dict] - instruction data
    ///   - 'blocks': Dict[str, Dict] - block information
    ///   - 'function_decls': Dict[str, Dict] - function declarations
    fn get_mir_instructions(&self) -> std::collections::HashMap<String, pyo3::PyObject> {
        use pyo3::types::{PyDict, PyList};

        Python::with_gil(|py| {
            let mut result = std::collections::HashMap::new();

            // Extract constants (float, boolean, integer, and string values)
            let constants = PyDict::new(py);
            let bool_constants = PyDict::new(py);
            let int_constants = PyDict::new(py);
            let str_constants = PyDict::new(py);
            for val in self.eval_func.dfg.values.iter() {
                if let mir::ValueDef::Const(data) = self.eval_func.dfg.value_def(val) {
                    match data {
                        mir::Const::Float(ieee64) => {
                            let float_val: f64 = ieee64.into();
                            constants.set_item(format!("v{}", u32::from(val)), float_val).unwrap();
                        }
                        mir::Const::Bool(b) => {
                            bool_constants.set_item(format!("v{}", u32::from(val)), b).unwrap();
                        }
                        mir::Const::Int(i) => {
                            int_constants.set_item(format!("v{}", u32::from(val)), i).unwrap();
                        }
                        mir::Const::Str(spur) => {
                            // String constants - store the spur key as a unique identifier
                            // These are typically model type selectors like "TYPE", "NMOS", etc.
                            // Spur is an interned string key - we use its raw key value
                            str_constants.set_item(format!("v{}", u32::from(val)), spur.into_inner()).unwrap();
                        }
                    }
                }
            }
            result.insert("constants".to_string(), constants.into());
            result.insert("bool_constants".to_string(), bool_constants.into());
            result.insert("int_constants".to_string(), int_constants.into());
            result.insert("str_constants".to_string(), str_constants.into());

            // Extract parameters as v16, v17, etc.
            // Use all func params to include Jacobian-related parameters (derivatives)
            let params = PyList::empty(py);
            let mut all_params: Vec<(u32, mir::Value)> = Vec::new();
            for val in self.eval_func.dfg.values.iter() {
                if let mir::ValueDef::Param(p) = self.eval_func.dfg.value_def(val) {
                    all_params.push((u32::from(p), val));
                }
            }
            all_params.sort_by_key(|(param_idx, _)| *param_idx);
            for (_, val) in all_params.iter() {
                params.append(format!("v{}", u32::from(*val))).unwrap();
            }
            result.insert("params".to_string(), params.into());

            // Extract instructions
            let instructions = PyList::empty(py);
            for block in self.eval_func.layout.blocks() {
                let block_name = format!("block{}", u32::from(block));

                for inst in self.eval_func.layout.block_insts(block) {
                    let inst_data = self.eval_func.dfg.insts[inst].clone();
                    let results = self.eval_func.dfg.inst_results(inst);

                    let inst_dict = PyDict::new(py);
                    inst_dict.set_item("block", &block_name).unwrap();

                    // Get result value(s)
                    if !results.is_empty() {
                        inst_dict.set_item("result", format!("v{}", u32::from(results[0]))).unwrap();
                    }

                    // Get opcode and operands
                    let opcode = inst_data.opcode();
                    inst_dict.set_item("opcode", opcode.name()).unwrap();

                    match &inst_data {
                        mir::InstructionData::Unary { arg, .. } => {
                            inst_dict.set_item("operands", vec![format!("v{}", u32::from(*arg))]).unwrap();
                        }
                        mir::InstructionData::Binary { args, .. } => {
                            inst_dict.set_item("operands", vec![
                                format!("v{}", u32::from(args[0])),
                                format!("v{}", u32::from(args[1]))
                            ]).unwrap();
                        }
                        mir::InstructionData::Call { func_ref, args } => {
                            let args_slice = args.as_slice(&self.eval_func.dfg.insts.value_lists);
                            let args_vec: Vec<String> = args_slice.iter()
                                .map(|v| format!("v{}", u32::from(*v)))
                                .collect();
                            inst_dict.set_item("operands", args_vec).unwrap();
                            inst_dict.set_item("func_ref", format!("inst{}", u32::from(*func_ref))).unwrap();
                        }
                        mir::InstructionData::PhiNode(phi) => {
                            let phi_ops = PyList::empty(py);
                            for (blk, val) in phi.edges(&self.eval_func.dfg.insts.value_lists, &self.eval_func.dfg.phi_forest) {
                                let edge = PyDict::new(py);
                                edge.set_item("value", format!("v{}", u32::from(val))).unwrap();
                                edge.set_item("block", format!("block{}", u32::from(blk))).unwrap();
                                phi_ops.append(edge).unwrap();
                            }
                            inst_dict.set_item("phi_operands", phi_ops).unwrap();
                        }
                        mir::InstructionData::Branch { cond, then_dst, else_dst, .. } => {
                            inst_dict.set_item("condition", format!("v{}", u32::from(*cond))).unwrap();
                            inst_dict.set_item("true_block", format!("block{}", u32::from(*then_dst))).unwrap();
                            inst_dict.set_item("false_block", format!("block{}", u32::from(*else_dst))).unwrap();
                        }
                        mir::InstructionData::Jump { destination } => {
                            inst_dict.set_item("destination", format!("block{}", u32::from(*destination))).unwrap();
                        }
                        mir::InstructionData::Exit => {}
                    }

                    instructions.append(inst_dict).unwrap();
                }
            }
            result.insert("instructions".to_string(), instructions.into());

            // Extract blocks
            let blocks = PyDict::new(py);
            for block in self.eval_func.layout.blocks() {
                let block_name = format!("block{}", u32::from(block));
                let block_dict = PyDict::new(py);

                // Get predecessors and successors from CFG
                let mut predecessors = Vec::new();
                let mut successors = Vec::new();

                // Check all blocks for references to this block
                for other_block in self.eval_func.layout.blocks() {
                    if let Some(inst) = self.eval_func.layout.block_insts(other_block).last() {
                        let inst_data = &self.eval_func.dfg.insts[inst];
                        match inst_data {
                            mir::InstructionData::Branch { then_dst, else_dst, .. } => {
                                if *then_dst == block || *else_dst == block {
                                    predecessors.push(format!("block{}", u32::from(other_block)));
                                }
                                if other_block == block {
                                    successors.push(format!("block{}", u32::from(*then_dst)));
                                    successors.push(format!("block{}", u32::from(*else_dst)));
                                }
                            }
                            mir::InstructionData::Jump { destination } => {
                                if *destination == block {
                                    predecessors.push(format!("block{}", u32::from(other_block)));
                                }
                                if other_block == block {
                                    successors.push(format!("block{}", u32::from(*destination)));
                                }
                            }
                            _ => {}
                        }
                    }
                }

                block_dict.set_item("predecessors", predecessors).unwrap();
                block_dict.set_item("successors", successors).unwrap();
                blocks.set_item(&block_name, block_dict).unwrap();
            }
            result.insert("blocks".to_string(), blocks.into());

            // Extract function declarations (callbacks)
            let func_decls = PyDict::new(py);
            for (i, name) in self.callback_names.iter().enumerate() {
                let decl = PyDict::new(py);
                decl.set_item("name", name.clone()).unwrap();
                // Get signature info
                if i < self.eval_func.dfg.signatures.len() {
                    let sig = &self.eval_func.dfg.signatures[mir::FuncRef::from(i as u32)];
                    decl.set_item("num_args", sig.params).unwrap();
                    decl.set_item("num_returns", sig.returns).unwrap();
                }
                func_decls.set_item(format!("inst{}", i), decl).unwrap();
            }
            result.insert("function_decls".to_string(), func_decls.into());

            result
        })
    }

    /// Export init function MIR instructions for JAX translation
    fn get_init_mir_instructions(&self) -> std::collections::HashMap<String, pyo3::PyObject> {
        use pyo3::types::{PyDict, PyList};

        Python::with_gil(|py| {
            let mut result = std::collections::HashMap::new();

            // Extract constants (float, boolean, and integer values)
            let constants = PyDict::new(py);
            let bool_constants = PyDict::new(py);
            let int_constants = PyDict::new(py);
            for val in self.init_func.dfg.values.iter() {
                if let mir::ValueDef::Const(data) = self.init_func.dfg.value_def(val) {
                    match data {
                        mir::Const::Float(ieee64) => {
                            let float_val: f64 = ieee64.into();
                            constants.set_item(format!("v{}", u32::from(val)), float_val).unwrap();
                        }
                        mir::Const::Bool(b) => {
                            bool_constants.set_item(format!("v{}", u32::from(val)), b).unwrap();
                        }
                        mir::Const::Int(i) => {
                            int_constants.set_item(format!("v{}", u32::from(val)), i).unwrap();
                        }
                        _ => {}
                    }
                }
            }
            result.insert("constants".to_string(), constants.into());
            result.insert("bool_constants".to_string(), bool_constants.into());
            result.insert("int_constants".to_string(), int_constants.into());

            // Extract parameters using the correct indices (in param order)
            let params = PyList::empty(py);
            for val_idx in self.init_param_value_indices.iter() {
                params.append(format!("v{}", val_idx)).unwrap();
            }
            result.insert("params".to_string(), params.into());

            // Extract instructions
            let instructions = PyList::empty(py);
            for block in self.init_func.layout.blocks() {
                let block_name = format!("block{}", u32::from(block));

                for inst in self.init_func.layout.block_insts(block) {
                    let inst_data = self.init_func.dfg.insts[inst].clone();
                    let results = self.init_func.dfg.inst_results(inst);

                    let inst_dict = PyDict::new(py);
                    inst_dict.set_item("block", &block_name).unwrap();

                    if !results.is_empty() {
                        inst_dict.set_item("result", format!("v{}", u32::from(results[0]))).unwrap();
                    }

                    let opcode = inst_data.opcode();
                    inst_dict.set_item("opcode", opcode.name()).unwrap();

                    match &inst_data {
                        mir::InstructionData::Unary { arg, .. } => {
                            inst_dict.set_item("operands", vec![format!("v{}", u32::from(*arg))]).unwrap();
                        }
                        mir::InstructionData::Binary { args, .. } => {
                            inst_dict.set_item("operands", vec![
                                format!("v{}", u32::from(args[0])),
                                format!("v{}", u32::from(args[1]))
                            ]).unwrap();
                        }
                        mir::InstructionData::Call { func_ref, args } => {
                            let args_slice = args.as_slice(&self.init_func.dfg.insts.value_lists);
                            let args_vec: Vec<String> = args_slice.iter()
                                .map(|v| format!("v{}", u32::from(*v)))
                                .collect();
                            inst_dict.set_item("operands", args_vec).unwrap();
                            inst_dict.set_item("func_ref", format!("inst{}", u32::from(*func_ref))).unwrap();
                        }
                        mir::InstructionData::PhiNode(phi) => {
                            let phi_ops = PyList::empty(py);
                            for (blk, val) in phi.edges(&self.init_func.dfg.insts.value_lists, &self.init_func.dfg.phi_forest) {
                                let edge = PyDict::new(py);
                                edge.set_item("value", format!("v{}", u32::from(val))).unwrap();
                                edge.set_item("block", format!("block{}", u32::from(blk))).unwrap();
                                phi_ops.append(edge).unwrap();
                            }
                            inst_dict.set_item("phi_operands", phi_ops).unwrap();
                        }
                        mir::InstructionData::Branch { cond, then_dst, else_dst, .. } => {
                            inst_dict.set_item("condition", format!("v{}", u32::from(*cond))).unwrap();
                            inst_dict.set_item("true_block", format!("block{}", u32::from(*then_dst))).unwrap();
                            inst_dict.set_item("false_block", format!("block{}", u32::from(*else_dst))).unwrap();
                        }
                        mir::InstructionData::Jump { destination } => {
                            inst_dict.set_item("destination", format!("block{}", u32::from(*destination))).unwrap();
                        }
                        mir::InstructionData::Exit => {}
                    }

                    instructions.append(inst_dict).unwrap();
                }
            }
            result.insert("instructions".to_string(), instructions.into());

            // Extract blocks with predecessors/successors (like eval function)
            let blocks = PyDict::new(py);
            for block in self.init_func.layout.blocks() {
                let block_name = format!("block{}", u32::from(block));
                let block_dict = PyDict::new(py);

                // Get predecessors and successors from CFG
                let mut predecessors = Vec::new();
                let mut successors = Vec::new();

                // Check all blocks for references to this block
                for other_block in self.init_func.layout.blocks() {
                    if let Some(inst) = self.init_func.layout.block_insts(other_block).last() {
                        let inst_data = &self.init_func.dfg.insts[inst];
                        match inst_data {
                            mir::InstructionData::Branch { then_dst, else_dst, .. } => {
                                if *then_dst == block || *else_dst == block {
                                    predecessors.push(format!("block{}", u32::from(other_block)));
                                }
                                if other_block == block {
                                    successors.push(format!("block{}", u32::from(*then_dst)));
                                    successors.push(format!("block{}", u32::from(*else_dst)));
                                }
                            }
                            mir::InstructionData::Jump { destination } => {
                                if *destination == block {
                                    predecessors.push(format!("block{}", u32::from(other_block)));
                                }
                                if other_block == block {
                                    successors.push(format!("block{}", u32::from(*destination)));
                                }
                            }
                            _ => {}
                        }
                    }
                }

                block_dict.set_item("predecessors", predecessors).unwrap();
                block_dict.set_item("successors", successors).unwrap();
                blocks.set_item(&block_name, block_dict).unwrap();
            }
            result.insert("blocks".to_string(), blocks.into());

            // Cache mapping - which init values map to which eval params
            let cache_map = PyList::empty(py);
            for (init_val, eval_param) in &self.cache_mapping {
                let entry = PyDict::new(py);
                entry.set_item("init_value", format!("v{}", init_val)).unwrap();
                entry.set_item("eval_param", *eval_param).unwrap();
                cache_map.append(entry).unwrap();
            }
            result.insert("cache_mapping".to_string(), cache_map.into());

            result
        })
    }

    /// Export DAE system (residuals and Jacobian) for JAX translation
    fn get_dae_system(&self) -> std::collections::HashMap<String, pyo3::PyObject> {
        use pyo3::types::{PyDict, PyList};

        Python::with_gil(|py| {
            let mut result = std::collections::HashMap::new();

            // Unknowns (node mapping)
            let unknowns = PyDict::new(py);
            for (i, node) in self.nodes.iter().enumerate() {
                unknowns.set_item(format!("sim_node{}", i), node.clone()).unwrap();
            }
            result.insert("unknowns".to_string(), unknowns.into());

            // Residuals
            let residuals = PyDict::new(py);
            for i in 0..self.num_residuals {
                let res = PyDict::new(py);
                res.set_item("resist", format!("v{}", self.residual_resist_indices[i])).unwrap();
                res.set_item("react", format!("v{}", self.residual_react_indices[i])).unwrap();
                residuals.set_item(format!("sim_node{}", i), res).unwrap();
            }
            result.insert("residuals".to_string(), residuals.into());

            // Jacobian
            let jacobian = PyList::empty(py);
            for i in 0..self.num_jacobian {
                let entry = PyDict::new(py);
                entry.set_item("row", format!("sim_node{}", self.jacobian_rows[i])).unwrap();
                entry.set_item("col", format!("sim_node{}", self.jacobian_cols[i])).unwrap();
                entry.set_item("resist", format!("v{}", self.jacobian_resist_indices[i])).unwrap();
                entry.set_item("react", format!("v{}", self.jacobian_react_indices[i])).unwrap();
                jacobian.append(entry).unwrap();
            }
            result.insert("jacobian".to_string(), jacobian.into());

            result
        })
    }

    /// Run init function and then eval function
    /// This is the proper way to evaluate - init computes cached values that eval needs
    ///
    /// Args:
    ///     params: Dict mapping parameter names to values
    ///             Should include both init and eval parameters
    ///
    /// Returns:
    ///     (residuals, jacobian) tuple
    fn run_init_eval(&self, params: std::collections::HashMap<String, f64>) -> PyResult<(Vec<(f64, f64)>, Vec<(u32, u32, f64, f64)>)> {
        // Stub callback for function calls
        fn stub_callback(state: &mut InterpreterState, _args: &[Value], rets: &[Value], _data: *mut c_void) {
            for &ret in rets {
                state.write(ret, 0.0f64);
            }
        }

        // === Step 1: Run init function ===
        let mut init_args: TiVec<Param, Data> = TiVec::new();
        for i in 0..self.init_num_params {
            let val = if i < self.init_param_names.len() {
                params.get(&self.init_param_names[i]).copied().unwrap_or(0.0)
            } else {
                0.0
            };
            init_args.push(Data::from(val));
        }

        let init_callbacks: Vec<(mir_interpret::Func, *mut c_void)> =
            (0..self.init_func.dfg.signatures.len())
                .map(|_| (stub_callback as mir_interpret::Func, std::ptr::null_mut()))
                .collect();

        let init_calls: &TiSlice<FuncRef, _> = TiSlice::from_ref(&init_callbacks);
        let init_args_slice: &TiSlice<Param, Data> = init_args.as_ref();
        let mut init_interp = Interpreter::new(&self.init_func, init_calls, init_args_slice);
        init_interp.run();

        // === Step 2: Build eval args from params + cached values from init ===
        let mut eval_args: TiVec<Param, Data> = TiVec::new();

        // First, add the named params
        for i in 0..self.func_num_params {
            let val = if i < self.param_names.len() {
                params.get(&self.param_names[i]).copied().unwrap_or(0.0)
            } else {
                0.0  // Will be filled from cache
            };
            eval_args.push(Data::from(val));
        }

        // Now override with cached values from init
        for (init_val_idx, eval_param_idx) in &self.cache_mapping {
            let cached_val: f64 = init_interp.state.read(Value::with_number_(*init_val_idx));
            if (*eval_param_idx as usize) < eval_args.len() {
                eval_args[Param::from(*eval_param_idx as usize)] = Data::from(cached_val);
            }
        }

        // === Step 3: Run eval function ===
        let eval_callbacks: Vec<(mir_interpret::Func, *mut c_void)> =
            (0..self.eval_func.dfg.signatures.len())
                .map(|_| (stub_callback as mir_interpret::Func, std::ptr::null_mut()))
                .collect();

        let eval_calls: &TiSlice<FuncRef, _> = TiSlice::from_ref(&eval_callbacks);
        let eval_args_slice: &TiSlice<Param, Data> = eval_args.as_ref();
        let mut eval_interp = Interpreter::new(&self.eval_func, eval_calls, eval_args_slice);
        eval_interp.run();

        // === Step 4: Extract results ===
        let mut residuals = Vec::new();
        for i in 0..self.num_residuals {
            let resist_val: f64 = eval_interp.state.read(Value::with_number_(self.residual_resist_indices[i]));
            let react_val: f64 = eval_interp.state.read(Value::with_number_(self.residual_react_indices[i]));
            residuals.push((resist_val, react_val));
        }

        let mut jacobian = Vec::new();
        for i in 0..self.num_jacobian {
            let row = self.jacobian_rows[i];
            let col = self.jacobian_cols[i];
            let resist_val: f64 = eval_interp.state.read(Value::with_number_(self.jacobian_resist_indices[i]));
            let react_val: f64 = eval_interp.state.read(Value::with_number_(self.jacobian_react_indices[i]));
            jacobian.push((row, col, resist_val, react_val));
        }

        Ok((residuals, jacobian))
    }

    fn __repr__(&self) -> String {
        format!(
            "VaModule(name='{}', params={}, nodes={}, residuals={}, jacobian={})",
            self.name,
            self.param_names.len(),
            self.nodes.len(),
            self.num_residuals,
            self.num_jacobian
        )
    }

    /// Evaluate the module with given parameter values
    ///
    /// Args:
    ///     params: Dict mapping parameter names to values (floats)
    ///
    /// Returns:
    ///     Dict with:
    ///       - 'residuals': list of (resist, react) tuples
    ///       - 'jacobian': list of (row, col, resist, react) tuples
    fn evaluate(&self, params: std::collections::HashMap<String, f64>) -> PyResult<std::collections::HashMap<String, Vec<(f64, f64)>>> {
        // Build argument array from parameter dictionary
        // Params in param_names correspond to Param indices 0..param_names.len()
        // but the function may have additional params, so we size to func_num_params
        let mut args: TiVec<Param, Data> = TiVec::new();

        for i in 0..self.func_num_params {
            let val = if i < self.param_names.len() {
                params.get(&self.param_names[i]).copied().unwrap_or(0.0)
            } else {
                0.0  // Extra params default to 0
            };
            args.push(Data::from(val));
        }

        // Create interpreter and run
        let empty_calls: &[(mir_interpret::Func, *mut std::ffi::c_void)] = &[];
        let calls = typed_index_collections::TiSlice::from_ref(empty_calls);
        let args_slice: &typed_index_collections::TiSlice<Param, Data> = args.as_ref();
        let mut interp = Interpreter::new(&self.eval_func, calls, args_slice);
        interp.run();

        // Extract residual values
        let mut residuals = Vec::new();
        for i in 0..self.num_residuals {
            let resist_val: f64 = interp.state.read(Value::with_number_(self.residual_resist_indices[i]));
            let react_val: f64 = interp.state.read(Value::with_number_(self.residual_react_indices[i]));
            residuals.push((resist_val, react_val));
        }

        // Extract jacobian values
        let mut jacobian = Vec::new();
        for i in 0..self.num_jacobian {
            let row = self.jacobian_rows[i];
            let col = self.jacobian_cols[i];
            let resist_val: f64 = interp.state.read(Value::with_number_(self.jacobian_resist_indices[i]));
            let react_val: f64 = interp.state.read(Value::with_number_(self.jacobian_react_indices[i]));
            jacobian.push((row as f64, col as f64, resist_val, react_val));
        }

        let mut result = std::collections::HashMap::new();
        result.insert("residuals".to_string(), residuals);
        // Convert jacobian to same format (just first two elements for position)
        let jac_formatted: Vec<(f64, f64)> = jacobian.iter().map(|(r, _c, resist, _react)| (*r, *resist)).collect();
        result.insert("jacobian_resist".to_string(), jac_formatted);
        let jac_react: Vec<(f64, f64)> = jacobian.iter().map(|(r, _c, _resist, react)| (*r, *react)).collect();
        result.insert("jacobian_react".to_string(), jac_react);

        Ok(result)
    }

    /// Evaluate and return full results as nested structure
    ///
    /// Args:
    ///     params: Dict mapping parameter names to values
    ///     extra_params: Optional list of values for extra unnamed parameters
    ///                   (indexed by Param index - len(param_names))
    fn evaluate_full(&self, params: std::collections::HashMap<String, f64>, extra_params: Option<Vec<f64>>) -> PyResult<(Vec<(f64, f64)>, Vec<(u32, u32, f64, f64)>)> {
        let extra = extra_params.unwrap_or_default();

        // Build argument array from parameter dictionary
        // Params in param_names correspond to Param indices 0..param_names.len()
        // but the function may have additional params, so we size to func_num_params
        let mut args: TiVec<Param, Data> = TiVec::new();

        for i in 0..self.func_num_params {
            let val = if i < self.param_names.len() {
                params.get(&self.param_names[i]).copied().unwrap_or(0.0)
            } else {
                let extra_idx = i - self.param_names.len();
                extra.get(extra_idx).copied().unwrap_or(0.0)
            };
            args.push(Data::from(val));
        }

        // Stub callback that returns 0 for all function calls
        fn stub_callback(state: &mut InterpreterState, _args: &[Value], rets: &[Value], _data: *mut c_void) {
            for &ret in rets {
                state.write(ret, 0.0f64);
            }
        }

        // Create callback entries - one for each function signature
        let num_callbacks = self.eval_func.dfg.signatures.len();
        let callbacks: Vec<(mir_interpret::Func, *mut c_void)> = (0..num_callbacks)
            .map(|_| (stub_callback as mir_interpret::Func, std::ptr::null_mut()))
            .collect();

        // Create interpreter and run
        let calls: &TiSlice<FuncRef, _> = TiSlice::from_ref(&callbacks);
        let args_slice: &TiSlice<Param, Data> = args.as_ref();
        let mut interp = Interpreter::new(&self.eval_func, calls, args_slice);
        interp.run();

        // Extract residual values
        let mut residuals = Vec::new();
        for i in 0..self.num_residuals {
            let resist_val: f64 = interp.state.read(Value::with_number_(self.residual_resist_indices[i]));
            let react_val: f64 = interp.state.read(Value::with_number_(self.residual_react_indices[i]));
            residuals.push((resist_val, react_val));
        }

        // Extract jacobian values
        let mut jacobian = Vec::new();
        for i in 0..self.num_jacobian {
            let row = self.jacobian_rows[i];
            let col = self.jacobian_cols[i];
            let resist_val: f64 = interp.state.read(Value::with_number_(self.jacobian_resist_indices[i]));
            let react_val: f64 = interp.state.read(Value::with_number_(self.jacobian_react_indices[i]));
            jacobian.push((row, col, resist_val, react_val));
        }

        Ok((residuals, jacobian))
    }
}

/// Compile a Verilog-A file and return module information
///
/// Args:
///     path: Path to the .va file
///     allow_analog_in_cond: Allow analog operators (limexp, ddt, idt) in conditionals.
///                           Default is false. Set to true for foundry models that use
///                           non-standard Verilog-A (like GF130 PDK).
#[pyfunction]
#[pyo3(signature = (path, allow_analog_in_cond=false))]
fn compile_va(path: &str, allow_analog_in_cond: bool) -> PyResult<Vec<VaModule>> {
    let input = std::path::Path::new(path)
        .canonicalize()
        .map_err(|e| PyValueError::new_err(format!("Failed to resolve path: {}", e)))?;
    let input = AbsPathBuf::assert(input);

    let opts = CompilationOpts {
        allow_analog_in_cond,
    };

    let db = CompilationDB::new_fs(input, &[], &[], &[], &opts)
        .map_err(|e| PyValueError::new_err(format!("Failed to create compilation DB: {}", e)))?;

    let modules = collect_modules(&db, false, &mut ConsoleSink::new(&db))
        .ok_or_else(|| PyValueError::new_err("Compilation failed with errors"))?;

    let mut literals = Rodeo::new();
    let mut result = Vec::new();

    for module_info in &modules {
        let compiled = CompiledModule::new(&db, module_info, &mut literals, false, false);

        // Extract parameter names, kinds, and Value indices
        let mut param_names = Vec::new();
        let mut param_kinds = Vec::new();
        let mut param_value_indices = Vec::new();

        for (kind, val) in compiled.intern.params.iter() {
            param_value_indices.push(u32::from(*val));
            let (kind_str, name) = match kind {
                ParamKind::Param(param) => ("param".to_string(), param.name(&db).to_string()),
                ParamKind::ParamGiven { param } => {
                    ("param_given".to_string(), param.name(&db).to_string())
                }
                ParamKind::Voltage { hi, lo } => {
                    let name = if let Some(lo) = lo {
                        format!("V({},{})", hi.name(&db), lo.name(&db))
                    } else {
                        format!("V({})", hi.name(&db))
                    };
                    ("voltage".to_string(), name)
                }
                ParamKind::Current(ck) => {
                    let name = match ck {
                        CurrentKind::Branch(br) => format!("I({})", br.name(&db)),
                        CurrentKind::Unnamed { hi, lo } => {
                            if let Some(lo) = lo {
                                format!("I({},{})", hi.name(&db), lo.name(&db))
                            } else {
                                format!("I({})", hi.name(&db))
                            }
                        }
                        CurrentKind::Port(n) => format!("I({})", n.name(&db)),
                    };
                    ("current".to_string(), name)
                }
                ParamKind::Temperature => ("temperature".to_string(), "$temperature".to_string()),
                ParamKind::Abstime => ("abstime".to_string(), "$abstime".to_string()),
                ParamKind::HiddenState(var) => {
                    ("hidden_state".to_string(), var.name(&db).to_string())
                }
                ParamKind::PortConnected { port } => {
                    ("port_connected".to_string(), port.name(&db).to_string())
                }
                ParamKind::ParamSysFun(param) => {
                    ("sysfun".to_string(), format!("{:?}", param))
                }
                _ => ("unknown".to_string(), "unknown".to_string()),
            };
            param_kinds.push(kind_str);
            param_names.push(name);
        }

        // Extract node names from unknowns
        let mut nodes = Vec::new();
        for kind in compiled.dae_system.unknowns.iter() {
            nodes.push(format!("{:?}", kind));
        }

        // Extract residual indices (convert Value to u32 using Into trait)
        let mut residual_resist_indices = Vec::new();
        let mut residual_react_indices = Vec::new();
        for residual in compiled.dae_system.residual.iter() {
            residual_resist_indices.push(u32::from(residual.resist));
            residual_react_indices.push(u32::from(residual.react));
        }

        // Extract Jacobian structure
        let mut jacobian_resist_indices = Vec::new();
        let mut jacobian_react_indices = Vec::new();
        let mut jacobian_rows = Vec::new();
        let mut jacobian_cols = Vec::new();
        for (_, entry) in compiled.dae_system.jacobian.iter().enumerate() {
            // SimUnknown row/col - use the index in the unknowns set
            jacobian_rows.push(u32::from(entry.row));
            jacobian_cols.push(u32::from(entry.col));
            jacobian_resist_indices.push(u32::from(entry.resist));
            jacobian_react_indices.push(u32::from(entry.react));
        }

        // Count actual Param-defined values in the eval function
        let mut max_param_idx: i32 = -1;
        for val in compiled.eval.dfg.values.iter() {
            if let mir::ValueDef::Param(p) = compiled.eval.dfg.value_def(val) {
                let p_idx: u32 = p.into();
                if p_idx as i32 > max_param_idx {
                    max_param_idx = p_idx as i32;
                }
            }
        }
        let func_num_params = if max_param_idx >= 0 { (max_param_idx + 1) as usize } else { 0 };

        // Collect callback names for debugging
        let callback_names: Vec<String> = compiled.intern.callbacks
            .iter()
            .map(|cb| format!("{:?}", cb))
            .collect();

        // === Init function support ===
        // Extract init function parameter names, kinds, and Value indices
        let mut init_param_names = Vec::new();
        let mut init_param_kinds = Vec::new();
        let mut init_param_value_indices = Vec::new();

        for (kind, val) in compiled.init.intern.params.iter() {
            init_param_value_indices.push(u32::from(*val));
            let (kind_str, name) = match kind {
                ParamKind::Param(param) => ("param".to_string(), param.name(&db).to_string()),
                ParamKind::ParamGiven { param } => {
                    ("param_given".to_string(), param.name(&db).to_string())
                }
                ParamKind::Voltage { hi, lo } => {
                    let name = if let Some(lo) = lo {
                        format!("V({},{})", hi.name(&db), lo.name(&db))
                    } else {
                        format!("V({})", hi.name(&db))
                    };
                    ("voltage".to_string(), name)
                }
                ParamKind::Current(ck) => {
                    let name = match ck {
                        CurrentKind::Branch(br) => format!("I({})", br.name(&db)),
                        CurrentKind::Unnamed { hi, lo } => {
                            if let Some(lo) = lo {
                                format!("I({},{})", hi.name(&db), lo.name(&db))
                            } else {
                                format!("I({})", hi.name(&db))
                            }
                        }
                        CurrentKind::Port(n) => format!("I({})", n.name(&db)),
                    };
                    ("current".to_string(), name)
                }
                ParamKind::Temperature => ("temperature".to_string(), "$temperature".to_string()),
                ParamKind::Abstime => ("abstime".to_string(), "$abstime".to_string()),
                ParamKind::HiddenState(var) => {
                    ("hidden_state".to_string(), var.name(&db).to_string())
                }
                ParamKind::PortConnected { port } => {
                    ("port_connected".to_string(), port.name(&db).to_string())
                }
                ParamKind::ParamSysFun(param) => {
                    ("sysfun".to_string(), format!("{:?}", param))
                }
                _ => ("unknown".to_string(), "unknown".to_string()),
            };
            init_param_kinds.push(kind_str);
            init_param_names.push(name);
        }

        // Count actual Param-defined values in the init function
        let mut init_max_param_idx: i32 = -1;
        for val in compiled.init.func.dfg.values.iter() {
            if let mir::ValueDef::Param(p) = compiled.init.func.dfg.value_def(val) {
                let p_idx: u32 = p.into();
                if p_idx as i32 > init_max_param_idx {
                    init_max_param_idx = p_idx as i32;
                }
            }
        }
        let init_num_params = if init_max_param_idx >= 0 { (init_max_param_idx + 1) as usize } else { 0 };

        // Build cache mapping: which init values map to which eval params
        // cached_vals: IndexMap<Value (in init), CacheSlot>
        // The CacheSlot.0 + intern.params.len() gives the eval Param index
        let num_eval_named_params = compiled.intern.params.len();
        let cache_mapping: Vec<(u32, u32)> = compiled.init.cached_vals
            .iter()
            .map(|(&init_val, &cache_slot)| {
                let eval_param_idx = cache_slot.0 as usize + num_eval_named_params;
                (u32::from(init_val), eval_param_idx as u32)
            })
            .collect();

        result.push(VaModule {
            name: module_info.module.name(&db).to_string(),
            param_names,
            param_kinds,
            param_value_indices,
            nodes,
            num_residuals: compiled.dae_system.residual.len(),
            num_jacobian: compiled.dae_system.jacobian.len(),
            residual_resist_indices,
            residual_react_indices,
            jacobian_resist_indices,
            jacobian_react_indices,
            jacobian_rows,
            jacobian_cols,
            eval_func: compiled.eval.clone(),
            func_num_params,
            callback_names,
            // Init function support
            init_func: compiled.init.func.clone(),
            init_param_names,
            init_param_kinds,
            init_param_value_indices,
            init_num_params,
            cache_mapping: cache_mapping.clone(),
            num_cached_values: cache_mapping.len(),
        });
    }

    Ok(result)
}

/// Python module definition
#[pymodule]
fn openvaf_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile_va, m)?)?;
    m.add_class::<VaModule>()?;
    Ok(())
}
