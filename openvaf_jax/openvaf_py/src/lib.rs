use std::ffi::c_void;
use std::collections::HashMap;
use std::sync::Arc;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use basedb::BaseDB;
use basedb::diagnostics::ConsoleSink;
use hir::{CompilationDB, CompilationOpts, Expr, Literal, Type};
use hir_def::{DefWithBodyId, ExprId};
use hir_def::body::BodySourceMap;
use hir_def::db::HirDefDB;
use hir_lower::{CurrentKind, ParamKind, PlaceKind};
use syntax::ast::UnaryOp;
use lasso::Rodeo;
use vfs::FileId;
use mir::{FuncRef, Function, Param, Value, F_ZERO, ValueDef};
use mir_interpret::{Interpreter, InterpreterState, Data};
use paths::AbsPathBuf;
use sim_back::{collect_modules, CompiledModule};
use typed_index_collections::{TiSlice, TiVec};

// OSDI flag constants (matching osdi_0_4.rs)
const PARA_TY_REAL: u32 = 0;
const PARA_TY_INT: u32 = 1;
const PARA_TY_STR: u32 = 2;
const PARA_KIND_MODEL: u32 = 0 << 30;
const PARA_KIND_INST: u32 = 1 << 30;
const JACOBIAN_ENTRY_RESIST_CONST: u32 = 1;
const JACOBIAN_ENTRY_REACT_CONST: u32 = 2;
const JACOBIAN_ENTRY_RESIST: u32 = 4;
const JACOBIAN_ENTRY_REACT: u32 = 8;

/// Parameter metadata matching OSDI descriptor
#[derive(Clone)]
struct OsdiParamInfo {
    name: String,
    aliases: Vec<String>,
    units: String,
    description: String,
    flags: u32,
    is_instance: bool,
}

/// Node metadata matching OSDI descriptor
#[derive(Clone)]
struct OsdiNodeInfo {
    name: String,
    kind: String,  // "KirchoffLaw", "BranchCurrent", or "Implicit"
    units: String,
    residual_units: String,
    is_internal: bool,
}

/// Jacobian entry metadata matching OSDI descriptor
#[derive(Clone)]
struct OsdiJacobianInfo {
    row: u32,
    col: u32,
    flags: u32,
}

/// Noise source metadata
#[derive(Clone)]
struct OsdiNoiseInfo {
    name: String,
    node1: u32,
    node2: u32,  // u32::MAX for ground
}

/// Source location data for mapping MIR back to VA source
#[derive(Clone)]
struct SourceLocData {
    file_path: String,
    line: u32,
    column: u32,
    #[allow(dead_code)]
    byte_start: u32,
    #[allow(dead_code)]
    byte_end: u32,
    source_line: Option<String>,
}

/// Source mapping data for a module
struct SourceMappingData {
    eval_body_source_map: Arc<BodySourceMap>,
    init_body_source_map: Arc<BodySourceMap>,
    #[allow(dead_code)]
    root_file: FileId,
    file_text: Arc<str>,
    file_path: String,
}

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
    /// MIR Value indices for residual components (internal, used by interpreter)
    /// Maps: residual equation index -> MIR Value index
    /// For JAX code generation, use get_dae_system()['residuals'][i]['resist_var'] instead
    residual_resist_indices: Vec<u32>,
    residual_react_indices: Vec<u32>,
    /// MIR Value indices for limiting RHS corrections
    /// These are subtracted from residuals when limiting is applied during NR iteration
    /// lim_rhs = J(lim_x) * (lim_x - x) where lim_x is the limited voltage
    residual_resist_lim_rhs_indices: Vec<u32>,
    residual_react_lim_rhs_indices: Vec<u32>,
    /// MIR Value indices for small-signal residual components (for AC analysis)
    /// These values are known to be zero in large-signal (DC) analysis
    residual_resist_small_signal_indices: Vec<u32>,
    residual_react_small_signal_indices: Vec<u32>,
    /// MIR Value indices for parameters known to be small-signal (always zero in DC)
    small_signal_param_indices: Vec<u32>,
    /// MIR Value indices for Jacobian components (internal, used by interpreter)
    /// Maps: jacobian entry index -> MIR Value index
    /// For JAX code generation, use get_dae_system()['jacobian'][i]['resist_var'] instead
    jacobian_resist_indices: Vec<u32>,
    jacobian_react_indices: Vec<u32>,
    /// Jacobian sparsity structure (row/col node indices)
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

    // Node collapse support
    /// Collapsible node pairs: (node1_idx, node2_idx_or_ground)
    /// If node2 is u32::MAX, it means collapse to ground
    #[pyo3(get)]
    collapsible_pairs: Vec<(u32, u32)>,
    /// Number of collapsible pairs
    #[pyo3(get)]
    num_collapsible: usize,
    /// Collapse decision outputs: (eq_index, value_name)
    /// Maps equation indices to the init function output values that control collapse
    #[pyo3(get)]
    collapse_decision_outputs: Vec<(u32, String)>,

    // Parameter defaults support
    /// Default values for parameters (extracted from Verilog-A source)
    /// Only includes parameters with literal default values
    param_defaults: HashMap<String, f64>,

    // String constant values (resolved from Spur keys)
    /// Maps operand name (e.g., "v123") to actual string value (e.g., "gmin")
    str_constant_values: HashMap<String, String>,

    // OSDI descriptor metadata
    /// Number of terminal nodes (ports)
    #[pyo3(get)]
    num_terminals: usize,
    /// OSDI parameter metadata
    osdi_params: Vec<OsdiParamInfo>,
    /// OSDI node metadata
    osdi_nodes: Vec<OsdiNodeInfo>,
    /// OSDI Jacobian entry metadata with flags
    osdi_jacobian: Vec<OsdiJacobianInfo>,
    /// Noise sources
    osdi_noise_sources: Vec<OsdiNoiseInfo>,
    /// Number of limiting states
    #[pyo3(get)]
    num_states: usize,
    /// Whether module has bound_step
    #[pyo3(get)]
    has_bound_step: bool,

    /// Optional source mapping data for MIR -> VA source tracking
    source_mapping: Option<SourceMappingData>,
}

#[pymethods]
impl VaModule {
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

    /// Get parameter defaults extracted from Verilog-A source
    /// Returns a dict mapping parameter name (lowercase) to default value
    /// Only includes parameters with literal (constant) default values
    fn get_param_defaults(&self) -> HashMap<String, f64> {
        self.param_defaults.clone()
    }

    /// Get resolved string constant values
    /// Returns a dict mapping operand name (e.g., "v123") to actual string (e.g., "gmin")
    fn get_str_constants(&self) -> HashMap<String, String> {
        self.str_constant_values.clone()
    }

    /// Get OSDI-compatible descriptor metadata
    /// Returns a dict matching the OSDI descriptor format with:
    ///   - 'params': list of parameter metadata dicts
    ///   - 'nodes': list of node metadata dicts
    ///   - 'jacobian': list of Jacobian entry dicts with flags
    ///   - 'collapsible': list of collapsible node pairs
    ///   - 'noise_sources': list of noise source dicts
    ///   - 'num_terminals': number of terminal nodes
    ///   - 'num_states': number of limiting states
    ///   - 'has_bound_step': whether module has bound_step
    fn get_osdi_descriptor(&self) -> std::collections::HashMap<String, pyo3::PyObject> {
        use pyo3::types::{PyDict, PyList};

        Python::with_gil(|py| {
            let mut result = std::collections::HashMap::new();

            // Parameters
            let params = PyList::empty(py);
            for param in &self.osdi_params {
                let d = PyDict::new(py);
                d.set_item("name", &param.name).unwrap();
                d.set_item("aliases", &param.aliases).unwrap();
                d.set_item("units", &param.units).unwrap();
                d.set_item("description", &param.description).unwrap();
                d.set_item("flags", param.flags).unwrap();
                d.set_item("is_instance", param.is_instance).unwrap();
                // Decode flags for convenience
                let is_model = (param.flags & PARA_KIND_INST) == 0;
                d.set_item("is_model_param", is_model).unwrap();
                params.append(d).unwrap();
            }
            result.insert("params".to_string(), params.into());

            // Nodes
            let nodes = PyList::empty(py);
            for node in &self.osdi_nodes {
                let d = PyDict::new(py);
                d.set_item("name", &node.name).unwrap();
                d.set_item("units", &node.units).unwrap();
                d.set_item("residual_units", &node.residual_units).unwrap();
                d.set_item("is_internal", node.is_internal).unwrap();
                nodes.append(d).unwrap();
            }
            result.insert("nodes".to_string(), nodes.into());

            // Jacobian entries with flags
            let jacobian = PyList::empty(py);
            for entry in &self.osdi_jacobian {
                let d = PyDict::new(py);
                d.set_item("row", entry.row).unwrap();
                d.set_item("col", entry.col).unwrap();
                d.set_item("flags", entry.flags).unwrap();
                // Decode flags for convenience
                d.set_item("has_resist", (entry.flags & JACOBIAN_ENTRY_RESIST) != 0).unwrap();
                d.set_item("has_react", (entry.flags & JACOBIAN_ENTRY_REACT) != 0).unwrap();
                d.set_item("resist_const", (entry.flags & JACOBIAN_ENTRY_RESIST_CONST) != 0).unwrap();
                d.set_item("react_const", (entry.flags & JACOBIAN_ENTRY_REACT_CONST) != 0).unwrap();
                jacobian.append(d).unwrap();
            }
            result.insert("jacobian".to_string(), jacobian.into());

            // Collapsible pairs
            let collapsible = PyList::empty(py);
            for (n1, n2) in &self.collapsible_pairs {
                let pair = PyList::empty(py);
                pair.append(*n1).unwrap();
                // n2 = u32::MAX means collapse to ground
                if *n2 == u32::MAX {
                    pair.append("gnd").unwrap();
                } else {
                    pair.append(*n2).unwrap();
                }
                collapsible.append(pair).unwrap();
            }
            result.insert("collapsible".to_string(), collapsible.into());

            // Noise sources
            let noise = PyList::empty(py);
            for src in &self.osdi_noise_sources {
                let d = PyDict::new(py);
                d.set_item("name", &src.name).unwrap();
                d.set_item("node1", src.node1).unwrap();
                if src.node2 == u32::MAX {
                    d.set_item("node2", "gnd").unwrap();
                } else {
                    d.set_item("node2", src.node2).unwrap();
                }
                noise.append(d).unwrap();
            }
            result.insert("noise_sources".to_string(), noise.into());

            // Scalar values
            result.insert("num_terminals".to_string(), self.num_terminals.into_py(py));
            result.insert("num_states".to_string(), self.num_states.into_py(py));
            result.insert("has_bound_step".to_string(), self.has_bound_step.into_py(py));
            result.insert("num_nodes".to_string(), self.osdi_nodes.len().into_py(py));
            result.insert("num_params".to_string(), self.osdi_params.len().into_py(py));
            result.insert("num_jacobian_entries".to_string(), self.osdi_jacobian.len().into_py(py));
            result.insert("num_collapsible".to_string(), self.collapsible_pairs.len().into_py(py));
            result.insert("num_noise_sources".to_string(), self.osdi_noise_sources.len().into_py(py));

            result
        })
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
                        mir::InstructionData::Branch { cond, then_dst, else_dst, loop_entry } => {
                            inst_dict.set_item("condition", format!("v{}", u32::from(*cond))).unwrap();
                            inst_dict.set_item("true_block", format!("block{}", u32::from(*then_dst))).unwrap();
                            inst_dict.set_item("false_block", format!("block{}", u32::from(*else_dst))).unwrap();
                            inst_dict.set_item("loop_entry", *loop_entry).unwrap();
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
                        mir::InstructionData::Branch { cond, then_dst, else_dst, loop_entry } => {
                            inst_dict.set_item("condition", format!("v{}", u32::from(*cond))).unwrap();
                            inst_dict.set_item("true_block", format!("block{}", u32::from(*then_dst))).unwrap();
                            inst_dict.set_item("false_block", format!("block{}", u32::from(*else_dst))).unwrap();
                            inst_dict.set_item("loop_entry", *loop_entry).unwrap();
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

    /// Export DAE system (residuals and Jacobian) with clear naming
    ///
    /// Uses:
    /// - Direct node indices and names (no synthetic sim_node{} names)
    /// - mir_{} prefix for MIR SSA values (clear that these are compiler intermediates)
    /// - Explicit node information including terminal/internal classification
    ///
    /// Returns a dict with:
    ///
    /// **Nodes** (SimUnknown mapping):
    ///   - 'nodes': List of node dicts, index = SimUnknown index
    ///       - idx: SimUnknown index (0, 1, 2, ...)
    ///       - name: VA node name ('D', 'G', 'S', 'B', 'NOI', ...)
    ///       - kind: 'KirchoffLaw', 'BranchCurrent', or 'Implicit'
    ///       - is_internal: true for internal nodes (idx >= num_terminals)
    ///   - 'terminals': List of terminal node names ['D', 'G', 'S', 'B']
    ///   - 'internal_nodes': List of internal node names ['NOI', 'GP', ...]
    ///   - 'num_terminals', 'num_internal': Counts
    ///
    /// **Residuals** (DAE equations f(x) = 0):
    ///   - 'residuals': List of residual dicts
    ///       - equation_idx: Index of this residual equation
    ///       - node_idx: SimUnknown index this residual stamps into
    ///       - node_name: VA node name for this residual
    ///       - resist_var: MIR variable name for resistive component ('mir_29')
    ///       - react_var: MIR variable name for reactive component ('mir_3')
    ///
    /// **Jacobian** (∂f/∂x sparsity and values):
    ///   - 'jacobian': List of Jacobian entry dicts
    ///       - entry_idx: Index of this Jacobian entry
    ///       - row_node_idx, col_node_idx: SimUnknown indices
    ///       - row_node_name, col_node_name: VA node names
    ///       - resist_var, react_var: MIR variable names ('mir_XX')
    ///       - has_resist, has_react: Whether components are non-zero
    ///
    /// **Collapsible pairs** (node collapse for internal nodes):
    ///   - 'collapsible_pairs': List of collapsible pair dicts
    ///       - pair_idx: Index of this collapse pair
    ///       - node1_idx, node2_idx: SimUnknown indices (node2=MAX means ground)
    ///       - node1_name, node2_name: VA node names ('G' -> 'GP')
    ///       - decision_var: MIR variable controlling collapse ('!v3729')
    ///   - 'num_collapsible': Number of collapsible pairs
    ///
    /// The MIR variable names (resist_var, react_var) reference values computed
    /// by the eval function. For JAX code generation, these map to Python variables
    /// like `v29` (strip 'mir_' prefix). For interpreter use, extract the numeric
    /// index to read from interpreter state.
    fn get_dae_system(&self) -> std::collections::HashMap<String, pyo3::PyObject> {
        use pyo3::types::{PyDict, PyList};

        Python::with_gil(|py| {
            let mut result = std::collections::HashMap::new();

            // Node information with clean names from OSDI metadata
            let nodes_list = PyList::empty(py);
            let mut terminal_names: Vec<String> = Vec::new();
            let mut internal_names: Vec<String> = Vec::new();

            for (i, node_info) in self.osdi_nodes.iter().enumerate() {
                let node_dict = PyDict::new(py);
                node_dict.set_item("idx", i).unwrap();
                node_dict.set_item("name", &node_info.name).unwrap();
                node_dict.set_item("kind", &node_info.kind).unwrap();
                node_dict.set_item("is_internal", node_info.is_internal).unwrap();
                nodes_list.append(node_dict).unwrap();

                if node_info.is_internal {
                    internal_names.push(node_info.name.clone());
                } else {
                    terminal_names.push(node_info.name.clone());
                }
            }
            result.insert("nodes".to_string(), nodes_list.into());

            // Residuals with direct indices and mir_ prefix
            let residuals_list = PyList::empty(py);
            for i in 0..self.num_residuals {
                let res_dict = PyDict::new(py);
                res_dict.set_item("equation_idx", i).unwrap();
                res_dict.set_item("node_idx", i).unwrap();  // Residual i stamps into node i
                // Get node name from OSDI metadata
                let node_name = if i < self.osdi_nodes.len() {
                    &self.osdi_nodes[i].name
                } else {
                    "unknown"
                };
                res_dict.set_item("node_name", node_name).unwrap();
                res_dict.set_item("resist_var", format!("mir_{}", self.residual_resist_indices[i])).unwrap();
                res_dict.set_item("react_var", format!("mir_{}", self.residual_react_indices[i])).unwrap();
                // Limiting RHS correction terms - subtracted from residuals during NR iteration
                res_dict.set_item("resist_lim_rhs_var", format!("mir_{}", self.residual_resist_lim_rhs_indices[i])).unwrap();
                res_dict.set_item("react_lim_rhs_var", format!("mir_{}", self.residual_react_lim_rhs_indices[i])).unwrap();
                // Small-signal components - values known to be zero in large-signal (DC) analysis
                res_dict.set_item("resist_small_signal_var", format!("mir_{}", self.residual_resist_small_signal_indices[i])).unwrap();
                res_dict.set_item("react_small_signal_var", format!("mir_{}", self.residual_react_small_signal_indices[i])).unwrap();
                residuals_list.append(res_dict).unwrap();
            }
            result.insert("residuals".to_string(), residuals_list.into());

            // Jacobian with direct indices and mir_ prefix
            let jacobian_list = PyList::empty(py);
            for i in 0..self.num_jacobian {
                let jac_dict = PyDict::new(py);
                jac_dict.set_item("entry_idx", i).unwrap();

                let row_idx = self.jacobian_rows[i] as usize;
                let col_idx = self.jacobian_cols[i] as usize;

                jac_dict.set_item("row_node_idx", row_idx).unwrap();
                jac_dict.set_item("col_node_idx", col_idx).unwrap();

                // Get node names from OSDI metadata
                let row_name = if row_idx < self.osdi_nodes.len() {
                    &self.osdi_nodes[row_idx].name
                } else {
                    "unknown"
                };
                let col_name = if col_idx < self.osdi_nodes.len() {
                    &self.osdi_nodes[col_idx].name
                } else {
                    "unknown"
                };
                jac_dict.set_item("row_node_name", row_name).unwrap();
                jac_dict.set_item("col_node_name", col_name).unwrap();

                jac_dict.set_item("resist_var", format!("mir_{}", self.jacobian_resist_indices[i])).unwrap();
                jac_dict.set_item("react_var", format!("mir_{}", self.jacobian_react_indices[i])).unwrap();

                // Add flags from OSDI jacobian info if available
                if i < self.osdi_jacobian.len() {
                    let flags = self.osdi_jacobian[i].flags;
                    jac_dict.set_item("has_resist", (flags & JACOBIAN_ENTRY_RESIST) != 0).unwrap();
                    jac_dict.set_item("has_react", (flags & JACOBIAN_ENTRY_REACT) != 0).unwrap();
                } else {
                    jac_dict.set_item("has_resist", true).unwrap();
                    jac_dict.set_item("has_react", true).unwrap();
                }

                jacobian_list.append(jac_dict).unwrap();
            }
            result.insert("jacobian".to_string(), jacobian_list.into());

            // Terminal and internal node lists
            result.insert("terminals".to_string(), terminal_names.clone().into_py(py));
            result.insert("internal_nodes".to_string(), internal_names.clone().into_py(py));
            result.insert("num_terminals".to_string(), terminal_names.len().into_py(py));
            result.insert("num_internal".to_string(), internal_names.len().into_py(py));

            // Collapsible pairs with node names resolved
            let collapsible_list = PyList::empty(py);
            for (i, (n1, n2)) in self.collapsible_pairs.iter().enumerate() {
                let pair_dict = PyDict::new(py);
                pair_dict.set_item("pair_idx", i).unwrap();
                pair_dict.set_item("node1_idx", *n1).unwrap();
                pair_dict.set_item("node2_idx", *n2).unwrap();

                // Resolve node names
                let n1_name = if (*n1 as usize) < self.osdi_nodes.len() {
                    self.osdi_nodes[*n1 as usize].name.clone()
                } else {
                    format!("unknown_{}", n1)
                };
                let n2_name = if *n2 == u32::MAX {
                    "ground".to_string()
                } else if (*n2 as usize) < self.osdi_nodes.len() {
                    self.osdi_nodes[*n2 as usize].name.clone()
                } else {
                    format!("unknown_{}", n2)
                };
                pair_dict.set_item("node1_name", n1_name).unwrap();
                pair_dict.set_item("node2_name", n2_name).unwrap();

                // Include the collapse decision variable if available
                if let Some((_, decision_var)) = self.collapse_decision_outputs.iter()
                    .find(|(idx, _)| *idx == i as u32) {
                    pair_dict.set_item("decision_var", decision_var).unwrap();
                }

                collapsible_list.append(pair_dict).unwrap();
            }
            result.insert("collapsible_pairs".to_string(), collapsible_list.into());
            result.insert("num_collapsible".to_string(), self.collapsible_pairs.len().into_py(py));

            // Small-signal parameters - MIR value indices known to be zero in large-signal analysis
            let small_signal_list = PyList::empty(py);
            for &idx in &self.small_signal_param_indices {
                small_signal_list.append(format!("mir_{}", idx)).unwrap();
            }
            result.insert("small_signal_params".to_string(), small_signal_list.into());
            result.insert("num_small_signal_params".to_string(), self.small_signal_param_indices.len().into_py(py));

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

    /// Check if source mapping data is available
    fn has_source_mapping(&self) -> bool {
        self.source_mapping.is_some()
    }

    /// Get source locations for all MIR instructions in eval function
    fn get_source_locations(&self) -> PyResult<pyo3::PyObject> {
        use pyo3::types::PyDict;

        if self.source_mapping.is_none() {
            return Err(PyValueError::new_err("Source mapping not available - recompile with source_mapping=true"));
        }

        Python::with_gil(|py| {
            let result = PyDict::new(py);
            for block in self.eval_func.layout.blocks() {
                for inst in self.eval_func.layout.block_insts(block) {
                    // Use safe get to avoid panic on out-of-bounds
                    let srcloc = match self.eval_func.srclocs.raw.get(u32::from(inst) as usize) {
                        Some(s) => *s,
                        None => continue,  // Skip instructions without source locations
                    };
                    let srcloc_raw: i32 = srcloc.0;
                    if srcloc_raw > 0 {
                        let inst_idx = u32::from(inst);
                        let inst_key = format!("inst{}", inst_idx);
                        if let Some(loc_data) = self.resolve_expr_to_source((srcloc_raw - 1) as u32, true) {
                            let loc_dict = PyDict::new(py);
                            loc_dict.set_item("file", &loc_data.file_path).unwrap();
                            loc_dict.set_item("line", loc_data.line).unwrap();
                            loc_dict.set_item("column", loc_data.column).unwrap();
                            if let Some(ref src_line) = loc_data.source_line {
                                loc_dict.set_item("source_line", src_line).unwrap();
                            }
                            result.set_item(&inst_key, loc_dict).unwrap();
                        }
                    }
                }
            }
            Ok(result.into())
        })
    }

    /// Get source location for a specific MIR value by name (e.g., "v123")
    fn get_value_source_location(&self, value_name: &str, is_eval: bool) -> PyResult<Option<pyo3::PyObject>> {
        use pyo3::types::PyDict;

        if self.source_mapping.is_none() {
            return Err(PyValueError::new_err("Source mapping not available"));
        }

        let value_idx: u32 = if value_name.starts_with("v") || value_name.starts_with("mir_") {
            let idx_str = value_name.trim_start_matches("v").trim_start_matches("mir_");
            idx_str.parse().map_err(|_| PyValueError::new_err(format!("Invalid value name: {}", value_name)))?
        } else {
            return Err(PyValueError::new_err(format!("Value name must start with 'v' or 'mir_': {}", value_name)));
        };

        let func = if is_eval { &self.eval_func } else { &self.init_func };
        let val = mir::Value::with_number_(value_idx);

        for block in func.layout.blocks() {
            for inst in func.layout.block_insts(block) {
                let results = func.dfg.inst_results(inst);
                if results.contains(&val) {
                    // Use safe get to avoid panic on out-of-bounds
                    let srcloc = match func.srclocs.raw.get(u32::from(inst) as usize) {
                        Some(s) => *s,
                        None => continue,
                    };
                    let srcloc_raw: i32 = srcloc.0;
                    if srcloc_raw > 0 {
                        if let Some(loc_data) = self.resolve_expr_to_source((srcloc_raw - 1) as u32, is_eval) {
                            return Python::with_gil(|py| {
                                let loc_dict = PyDict::new(py);
                                loc_dict.set_item("file", &loc_data.file_path).unwrap();
                                loc_dict.set_item("line", loc_data.line).unwrap();
                                loc_dict.set_item("column", loc_data.column).unwrap();
                                if let Some(ref src_line) = loc_data.source_line {
                                    loc_dict.set_item("source_line", src_line).unwrap();
                                }
                                Ok(Some(loc_dict.into()))
                            });
                        }
                    }
                }
            }
        }
        Ok(None)
    }
}

// Helper methods for source location resolution (not exposed to Python)
impl VaModule {
    fn resolve_expr_to_source(&self, expr_idx: u32, is_eval: bool) -> Option<SourceLocData> {
        let mapping = self.source_mapping.as_ref()?;
        let body_source_map = if is_eval {
            &mapping.eval_body_source_map
        } else {
            &mapping.init_body_source_map
        };

        // Check bounds before accessing expr_map_back to avoid panics
        // Note: TiVec's get() can still panic, so we check bounds first
        let len = body_source_map.expr_map_back.raw.len();
        if (expr_idx as usize) >= len {
            return None;
        }

        let _expr_id = ExprId::from(expr_idx as usize);
        // expr_map_back is ArenaMap<Expr, Option<AstPtr<ast::Expr>>>
        // Access via raw slice to avoid TiSliceIndex panics
        let ast_ptr_opt = body_source_map.expr_map_back.raw.get(expr_idx as usize)?;
        let ast_ptr = ast_ptr_opt.as_ref()?;
        let syntax_range = ast_ptr.syntax_node_ptr().range();
        let byte_start = u32::from(syntax_range.start());
        let byte_end = u32::from(syntax_range.end());
        let (line, column) = self.offset_to_line_col(&mapping.file_text, byte_start);
        let source_line = self.extract_source_line(&mapping.file_text, line);

        Some(SourceLocData {
            file_path: mapping.file_path.clone(),
            line, column, byte_start, byte_end, source_line,
        })
    }

    fn offset_to_line_col(&self, text: &str, byte_offset: u32) -> (u32, u32) {
        let mut line = 1u32;
        let mut col = 1u32;
        let mut current_offset = 0u32;
        for ch in text.chars() {
            if current_offset >= byte_offset { break; }
            if ch == '\n' { line += 1; col = 1; } else { col += 1; }
            current_offset += ch.len_utf8() as u32;
        }
        (line, col)
    }

    fn extract_source_line(&self, text: &str, line_num: u32) -> Option<String> {
        text.lines().nth((line_num - 1) as usize).map(|s| s.to_string())
    }
}

/// Compile a Verilog-A file and return module information
///
/// Args:
///     path: Path to the .va file
///     allow_analog_in_cond: Allow analog operators (limexp, ddt, idt) in conditionals.
///                           Default is false. Set to true for foundry models that use
///                           non-standard Verilog-A (like GF130 PDK).
///     allow_builtin_primitives: Allow built-in primitives like `nmos` and `pmos`.
///                               Default is false.
#[pyfunction]
#[pyo3(signature = (path, allow_analog_in_cond=false, allow_builtin_primitives=false, source_mapping=false))]
fn compile_va(path: &str, allow_analog_in_cond: bool, allow_builtin_primitives: bool, source_mapping: bool) -> PyResult<Vec<VaModule>> {
    let input = std::path::Path::new(path)
        .canonicalize()
        .map_err(|e| PyValueError::new_err(format!("Failed to resolve path: {}", e)))?;
    let input = AbsPathBuf::assert(input);

    let opts = CompilationOpts {
        allow_analog_in_cond: allow_analog_in_cond,
        allow_builtin_primitives: allow_builtin_primitives,
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
                ParamKind::ImplicitUnknown(eq) => {
                    ("implicit_unknown".to_string(), format!("{:?}", eq))
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
        let mut residual_resist_lim_rhs_indices = Vec::new();
        let mut residual_react_lim_rhs_indices = Vec::new();
        // Small-signal indices - values known to be zero in large-signal analysis
        let mut residual_resist_small_signal_indices = Vec::new();
        let mut residual_react_small_signal_indices = Vec::new();
        for residual in compiled.dae_system.residual.iter() {
            residual_resist_indices.push(u32::from(residual.resist));
            residual_react_indices.push(u32::from(residual.react));
            // Limiting RHS corrections - subtracted from residuals when limiting is applied
            residual_resist_lim_rhs_indices.push(u32::from(residual.resist_lim_rhs));
            residual_react_lim_rhs_indices.push(u32::from(residual.react_lim_rhs));
            // Small-signal parts - for AC analysis
            residual_resist_small_signal_indices.push(u32::from(residual.resist_small_signal));
            residual_react_small_signal_indices.push(u32::from(residual.react_small_signal));
        }

        // Extract small-signal parameters - parameters known to be zero in large-signal analysis
        let small_signal_param_indices: Vec<u32> = compiled.dae_system.small_signal_parameters
            .iter()
            .map(|&v| u32::from(v))
            .collect();

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
                ParamKind::ImplicitUnknown(eq) => {
                    ("implicit_unknown".to_string(), format!("{:?}", eq))
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

        // Extract collapsible pairs for node collapse support
        // Each pair is (node1_idx, node2_idx) where node2=u32::MAX means collapse to ground
        let collapsible_pairs: Vec<(u32, u32)> = compiled.node_collapse
            .pairs()
            .map(|(_, node1, node2_opt)| {
                let n1: u32 = node1.into();
                let n2: u32 = node2_opt.map(|n| n.into()).unwrap_or(u32::MAX);
                (n1, n2)
            })
            .collect();
        let num_collapsible = collapsible_pairs.len();

        // Extract collapse decision outputs from init function
        // These are the boolean values that determine which pairs actually collapse at runtime
        // Format: Vec<(pair_index, value_name, negate)> where:
        //   - pair_index: index into collapsible_pairs
        //   - value_name: the condition variable (e.g., "v3729")
        //   - negate: true if collapse happens when condition is FALSE
        let mut collapse_decision_outputs: Vec<(u32, String)> = Vec::new();

        use hir_lower::CallBackKind;

        // Build mapping from FuncRef to CollapseHint callback info
        // CollapseHint callbacks are indexed in order in intern.callbacks
        let mut collapse_hint_funcs: HashMap<mir::FuncRef, usize> = HashMap::new();
        let mut collapse_hint_index = 0usize;
        for (idx, kind) in compiled.init.intern.callbacks.iter().enumerate() {
            if matches!(kind, CallBackKind::CollapseHint(_, _)) {
                let func_ref = mir::FuncRef::from(idx as u32);
                collapse_hint_funcs.insert(func_ref, collapse_hint_index);
                collapse_hint_index += 1;
            }
        }

        // Find Call instructions to CollapseHint callbacks and their controlling conditions
        // Strategy: For each Call to a CollapseHint, find the block it's in,
        // then find the branch that targets that block
        let init_func = &compiled.init.func;

        // Build map of block -> instructions containing calls to CollapseHint
        let mut callback_blocks: HashMap<mir::Block, Vec<(mir::FuncRef, usize)>> = HashMap::new();
        for block in init_func.layout.blocks() {
            for inst in init_func.layout.block_insts(block) {
                if let mir::InstructionData::Call { func_ref, .. } = init_func.dfg.insts[inst] {
                    if let Some(&hint_idx) = collapse_hint_funcs.get(&func_ref) {
                        callback_blocks.entry(block).or_default().push((func_ref, hint_idx));
                    }
                }
            }
        }

        // For each block containing a CollapseHint call, find the branch that targets it
        for block in init_func.layout.blocks() {
            for inst in init_func.layout.block_insts(block) {
                if let mir::InstructionData::Branch { cond, then_dst, else_dst, .. } = init_func.dfg.insts[inst] {
                    // Check if either branch target contains a CollapseHint call
                    // then_dst and else_dst are Block types directly
                    for (target_block, is_true_branch) in [(then_dst, true), (else_dst, false)] {
                        if let Some(callbacks) = callback_blocks.get(&target_block) {
                            for &(_func_ref, hint_idx) in callbacks {
                                // hint_idx corresponds to collapsible_pairs index
                                let pair_idx = hint_idx as u32;
                                let cond_idx: u32 = cond.into();
                                // If callback is on FALSE branch, collapse = NOT(cond)
                                // We'll handle negation in Python by checking is_true_branch
                                // For now, store condition with a prefix indicating if it needs negation
                                let prefix = if is_true_branch { "" } else { "!" };
                                collapse_decision_outputs.push((pair_idx, format!("{}v{}", prefix, cond_idx)));
                            }
                        }
                    }
                }
            }
        }

        // Sort by pair index for consistent ordering
        collapse_decision_outputs.sort_by_key(|(idx, _)| *idx);

        // Extract parameter defaults from HIR
        // For each parameter, try to get its literal default value
        let mut param_defaults = HashMap::new();
        for (kind, _) in compiled.intern.params.iter() {
            if let ParamKind::Param(param) = kind {
                let param_name = param.name(&db).to_lowercase();
                // Get the parameter's init body which contains the default expression
                let body = param.init(&db);
                let body_ref = body.borrow();
                // The body's first entry should be the default value expression
                // Check if it's a simple literal we can extract
                if !body_ref.entry().is_empty() {
                    let expr_id = body_ref.get_entry_expr(0);

                    // Try direct literal first
                    if let Some(lit) = body_ref.as_literal(expr_id) {
                        match lit {
                            Literal::Float(ieee64) => {
                                param_defaults.insert(param_name.clone(), f64::from(*ieee64));
                            }
                            Literal::Int(i) => {
                                param_defaults.insert(param_name.clone(), *i as f64);
                            }
                            Literal::Inf => {
                                param_defaults.insert(param_name.clone(), f64::INFINITY);
                            }
                            _ => {} // Skip string literals
                        }
                    } else {
                        // Check for UnaryOp::Neg wrapping a literal (e.g., -1.0)
                        let expr = body_ref.get_expr(expr_id);
                        if let Expr::UnaryOp { expr: inner_expr, op: UnaryOp::Neg } = expr {
                            if let Some(lit) = body_ref.as_literal(inner_expr) {
                                match lit {
                                    Literal::Float(ieee64) => {
                                        param_defaults.insert(param_name.clone(), -f64::from(*ieee64));
                                    }
                                    Literal::Int(i) => {
                                        param_defaults.insert(param_name.clone(), -(*i as f64));
                                    }
                                    _ => {} // Skip other literals
                                }
                            }
                        }
                    }
                }
            }
        }

        // === OSDI Metadata Extraction ===

        // Extract OSDI-compatible parameter metadata from module_info
        let mut osdi_params: Vec<OsdiParamInfo> = Vec::new();
        for (param, param_info) in module_info.params.iter() {
            let ty = param.ty(&db);
            let type_flag = match ty.base_type() {
                Type::Real => PARA_TY_REAL,
                Type::Integer => PARA_TY_INT,
                Type::String => PARA_TY_STR,
                _ => PARA_TY_REAL,
            };
            let kind_flag = if param_info.is_instance { PARA_KIND_INST } else { PARA_KIND_MODEL };

            osdi_params.push(OsdiParamInfo {
                name: param_info.name.to_string(),
                aliases: param_info.alias.iter().map(|s| s.to_string()).collect(),
                units: param_info.unit.clone(),
                description: param_info.description.clone(),
                flags: type_flag | kind_flag,
                is_instance: param_info.is_instance,
            });
        }

        // Extract OSDI-compatible node metadata
        // Determine terminal count from module ports
        let num_terminals = module_info.module.ports(&db).len();

        let mut osdi_nodes: Vec<OsdiNodeInfo> = Vec::new();
        for (idx, unknown_kind) in compiled.dae_system.unknowns.iter_enumerated() {
            let is_internal = u32::from(idx) >= num_terminals as u32;
            // Extract node name and kind from SimUnknownKind
            let (clean_name, kind_str) = match unknown_kind {
                sim_back::SimUnknownKind::KirchoffLaw(node) => {
                    (node.name(&db).to_string(), "KirchoffLaw")
                }
                sim_back::SimUnknownKind::Current(ck) => {
                    let name = match ck {
                        CurrentKind::Branch(br) => format!("flow({})", br.name(&db)),
                        CurrentKind::Unnamed { hi, lo } => {
                            if let Some(lo) = lo {
                                format!("flow({},{})", hi.name(&db), lo.name(&db))
                            } else {
                                format!("flow({})", hi.name(&db))
                            }
                        }
                        CurrentKind::Port(node) => format!("flow(<{}>)", node.name(&db)),
                    };
                    (name, "BranchCurrent")
                }
                sim_back::SimUnknownKind::Implicit(eq) => {
                    (format!("implicit_equation_{}", u32::from(*eq)), "Implicit")
                }
            };
            // TODO: Extract units from discipline when available
            osdi_nodes.push(OsdiNodeInfo {
                name: clean_name,
                kind: kind_str.to_string(),
                units: "V".to_string(),  // Default to voltage
                residual_units: "A".to_string(),  // Default to current
                is_internal,
            });
        }

        // Extract OSDI-compatible Jacobian entry metadata with flags
        // Helper to check if a Jacobian entry value is constant
        let is_entry_const = |entry_val: mir::Value, func: &mir::Function| -> bool {
            match func.dfg.value_def(entry_val) {
                ValueDef::Const(_) => true,
                ValueDef::Param(param) => {
                    // Check if param is operation-dependent
                    compiled.intern.params.get_index(param)
                        .map_or(true, |(kind, _)| !kind.op_dependent())
                }
                _ => false,
            }
        };

        let mut osdi_jacobian: Vec<OsdiJacobianInfo> = Vec::new();
        for entry in compiled.dae_system.jacobian.iter() {
            let mut flags: u32 = 0;

            // Check resistive component
            if entry.resist != F_ZERO {
                flags |= JACOBIAN_ENTRY_RESIST;
            }
            if is_entry_const(entry.resist, &compiled.eval) {
                flags |= JACOBIAN_ENTRY_RESIST_CONST;
            }

            // Check reactive component
            if entry.react != F_ZERO {
                flags |= JACOBIAN_ENTRY_REACT;
            }
            if is_entry_const(entry.react, &compiled.eval) {
                flags |= JACOBIAN_ENTRY_REACT_CONST;
            }

            osdi_jacobian.push(OsdiJacobianInfo {
                row: u32::from(entry.row),
                col: u32::from(entry.col),
                flags,
            });
        }

        // Extract noise sources
        let osdi_noise_sources: Vec<OsdiNoiseInfo> = compiled.dae_system.noise_sources
            .iter()
            .map(|src| {
                let name = literals.resolve(&src.name).to_owned();
                OsdiNoiseInfo {
                    name,
                    node1: u32::from(src.hi),
                    node2: src.lo.map_or(u32::MAX, |lo| u32::from(lo)),
                }
            })
            .collect();

        // Check for bound_step
        // bound_step is stored in outputs with key PlaceKind::BoundStep
        let has_bound_step = compiled.intern.outputs.contains_key(&PlaceKind::BoundStep);

        // Number of limiting states
        let num_states = compiled.intern.lim_state.len();

        // Extract and resolve string constants from eval function
        let mut str_constant_values: HashMap<String, String> = HashMap::new();
        for val in compiled.eval.dfg.values.iter() {
            if let mir::ValueDef::Const(mir::Const::Str(spur)) = compiled.eval.dfg.value_def(val) {
                let operand_name = format!("v{}", u32::from(val));
                // Resolve the Spur key to actual string using the literals interner
                let resolved_str = literals.resolve(&spur).to_owned();
                str_constant_values.insert(operand_name, resolved_str);
            }
        }

        // Build source mapping data if requested
        let source_mapping_data = if source_mapping {
            let module_id = module_info.module.module_id();
            let eval_def = DefWithBodyId::ModuleId { initial: false, module: module_id };
            let (_, eval_body_source_map) = db.body_with_sourcemap(eval_def);
            let init_def = DefWithBodyId::ModuleId { initial: true, module: module_id };
            let (_, init_body_source_map) = db.body_with_sourcemap(init_def);
            let root_file = db.compilation_unit().root_file();
            let file_text = db.file_text(root_file).unwrap_or_else(|_| Arc::from(""));

            Some(SourceMappingData {
                eval_body_source_map,
                init_body_source_map,
                root_file,
                file_text,
                file_path: path.to_string(),
            })
        } else {
            None
        };

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
            residual_resist_lim_rhs_indices,
            residual_react_lim_rhs_indices,
            residual_resist_small_signal_indices,
            residual_react_small_signal_indices,
            small_signal_param_indices,
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
            // Node collapse support
            collapsible_pairs,
            num_collapsible,
            collapse_decision_outputs,
            // Parameter defaults support
            param_defaults,
            // String constant values
            str_constant_values,
            // OSDI metadata
            num_terminals,
            osdi_params,
            osdi_nodes,
            osdi_jacobian,
            osdi_noise_sources,
            num_states,
            has_bound_step,
            source_mapping: source_mapping_data,
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
