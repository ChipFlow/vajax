//! Python bindings for OSDI (Open Simulator Device Interface) shared libraries.
//!
//! This crate provides a Python interface to load and evaluate Verilog-A models
//! compiled to OSDI format by OpenVAF.

use std::alloc::{alloc_zeroed, handle_alloc_error, Layout};
use std::cell::{Cell, RefCell};
use std::ffi::{c_void, CStr, CString};
use std::mem::align_of;
use std::os::raw::c_char;
use std::panic::catch_unwind;
use std::ptr;
use std::slice;
use std::sync::Arc;

use anyhow::{bail, Result};
use libloading::{Library, Symbol};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

// ============================================================================
// OSDI 0.4 Structure Definitions (matching vendor/OpenVAF/melange/core/src/veriloga/osdi_0_4.rs)
// ============================================================================

pub const OSDI_VERSION_MAJOR_CURR: u32 = 0;
pub const OSDI_VERSION_MINOR_CURR: u32 = 4;
pub const PARA_TY_MASK: u32 = 3;
pub const PARA_TY_REAL: u32 = 0;
pub const PARA_TY_INT: u32 = 1;
pub const PARA_TY_STR: u32 = 2;
pub const PARA_KIND_MASK: u32 = 3 << 30;
pub const PARA_KIND_MODEL: u32 = 0 << 30;
pub const PARA_KIND_INST: u32 = 1 << 30;
pub const PARA_KIND_OPVAR: u32 = 2 << 30;
pub const ACCESS_FLAG_READ: u32 = 0;
pub const ACCESS_FLAG_SET: u32 = 1;
pub const ACCESS_FLAG_INSTANCE: u32 = 4;
pub const JACOBIAN_ENTRY_RESIST_CONST: u32 = 1;
pub const JACOBIAN_ENTRY_REACT_CONST: u32 = 2;
pub const JACOBIAN_ENTRY_RESIST: u32 = 4;
pub const JACOBIAN_ENTRY_REACT: u32 = 8;
pub const CALC_RESIST_RESIDUAL: u32 = 1;
pub const CALC_REACT_RESIDUAL: u32 = 2;
pub const CALC_RESIST_JACOBIAN: u32 = 4;
pub const CALC_REACT_JACOBIAN: u32 = 8;
pub const CALC_NOISE: u32 = 16;
pub const CALC_OP: u32 = 32;
pub const CALC_RESIST_LIM_RHS: u32 = 64;
pub const CALC_REACT_LIM_RHS: u32 = 128;
pub const ENABLE_LIM: u32 = 256;
pub const INIT_LIM: u32 = 512;
pub const ANALYSIS_NOISE: u32 = 1024;
pub const ANALYSIS_DC: u32 = 2048;
pub const ANALYSIS_AC: u32 = 4096;
pub const ANALYSIS_TRAN: u32 = 8192;
pub const ANALYSIS_IC: u32 = 16384;
pub const ANALYSIS_STATIC: u32 = 32768;
pub const ANALYSIS_NODESET: u32 = 65536;
pub const EVAL_RET_FLAG_LIM: u32 = 1;
pub const EVAL_RET_FLAG_FATAL: u32 = 2;
pub const EVAL_RET_FLAG_FINISH: u32 = 4;
pub const EVAL_RET_FLAG_STOP: u32 = 8;
pub const INIT_ERR_OUT_OF_BOUNDS: u32 = 1;

// Log level constants for osdi_log callback
pub const LOG_LVL_MASK: u32 = 7;
pub const LOG_LVL_DEBUG: u32 = 0;
pub const LOG_LVL_DISPLAY: u32 = 1;
pub const LOG_LVL_INFO: u32 = 2;
pub const LOG_LVL_WARN: u32 = 3;
pub const LOG_LVL_ERR: u32 = 4;
pub const LOG_LVL_FATAL: u32 = 5;
pub const LOG_FMT_ERR: u32 = 16;

#[repr(C)]
pub struct OsdiLimFunction {
    pub name: *mut c_char,
    pub num_args: u32,
    pub func_ptr: *mut c_void,
}

#[repr(C)]
pub struct OsdiSimParas {
    pub names: *mut *mut c_char,
    pub vals: *mut f64,
    pub names_str: *mut *mut c_char,
    pub vals_str: *mut *mut c_char,
}

#[repr(C)]
pub struct OsdiSimInfo {
    pub paras: OsdiSimParas,
    pub abstime: f64,
    pub prev_solve: *mut f64,
    pub prev_state: *mut f64,
    pub next_state: *mut f64,
    pub flags: u32,
}

#[repr(C)]
pub union OsdiInitErrorPayload {
    pub parameter_id: u32,
}

#[repr(C)]
pub struct OsdiInitError {
    pub code: u32,
    pub payload: OsdiInitErrorPayload,
}

#[repr(C)]
pub struct OsdiInitInfo {
    pub flags: u32,
    pub num_errors: u32,
    pub errors: *mut OsdiInitError,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OsdiNodePair {
    pub node_1: u32,
    pub node_2: u32,
}

#[repr(C)]
pub struct OsdiJacobianEntry {
    pub nodes: OsdiNodePair,
    pub react_ptr_off: u32,
    pub flags: u32,
}

#[repr(C)]
pub struct OsdiNode {
    pub name: *mut c_char,
    pub units: *mut c_char,
    pub residual_units: *mut c_char,
    pub resist_residual_off: u32,
    pub react_residual_off: u32,
    pub resist_limit_rhs_off: u32,
    pub react_limit_rhs_off: u32,
    pub is_flow: bool,
}

#[repr(C)]
pub struct OsdiParamOpvar {
    pub name: *mut *mut c_char,
    pub num_alias: u32,
    pub description: *mut c_char,
    pub units: *mut c_char,
    pub flags: u32,
    pub len: u32,
}

#[repr(C)]
pub struct OsdiNoiseSource {
    pub name: *mut c_char,
    pub nodes: OsdiNodePair,
}

#[repr(C)]
pub struct OsdiNatureRef {
    pub ref_type: u32,
    pub index: u32,
}

// Function pointer types
type AccessFn = fn(*mut c_void, *mut c_void, u32, u32) -> *mut c_void;
type SetupModelFn = fn(*mut c_void, *mut c_void, *mut OsdiSimParas, *mut OsdiInitInfo);
type SetupInstanceFn =
    fn(*mut c_void, *mut c_void, *mut c_void, f64, u32, *mut OsdiSimParas, *mut OsdiInitInfo);
type EvalFn = fn(*mut c_void, *mut c_void, *mut c_void, *mut OsdiSimInfo) -> u32;
type LoadNoiseFn = fn(*mut c_void, *mut c_void, f64, *mut f64);
type LoadResidualFn = fn(*mut c_void, *mut c_void, *mut f64);
type LoadSpiceRhsDcFn = fn(*mut c_void, *mut c_void, *mut f64, *mut f64);
type LoadSpiceRhsTranFn = fn(*mut c_void, *mut c_void, *mut f64, *mut f64, f64);
type LoadJacobianFn = fn(*mut c_void, *mut c_void);
type LoadJacobianAlphaFn = fn(*mut c_void, *mut c_void, f64);
type GivenFlagFn = fn(*mut c_void, u32) -> u32;
type WriteJacobianArrayFn = fn(*mut c_void, *mut c_void, *mut f64);
type LoadJacobianOffsetFn = fn(*mut c_void, *mut c_void, usize);

#[repr(C)]
#[non_exhaustive]
pub struct OsdiDescriptor {
    pub name: *mut c_char,
    pub num_nodes: u32,
    pub num_terminals: u32,
    pub nodes: *mut OsdiNode,
    pub num_jacobian_entries: u32,
    pub jacobian_entries: *mut OsdiJacobianEntry,
    pub num_collapsible: u32,
    pub collapsible: *mut OsdiNodePair,
    pub collapsed_offset: u32,
    pub noise_sources: *mut OsdiNoiseSource,
    pub num_noise_src: u32,
    pub num_params: u32,
    pub num_instance_params: u32,
    pub num_opvars: u32,
    pub param_opvar: *mut OsdiParamOpvar,
    pub node_mapping_offset: u32,
    pub jacobian_ptr_resist_offset: u32,
    pub num_states: u32,
    pub state_idx_off: u32,
    pub bound_step_offset: u32,
    pub instance_size: u32,
    pub model_size: u32,
    pub access: AccessFn,
    pub setup_model: SetupModelFn,
    pub setup_instance: SetupInstanceFn,
    pub eval: EvalFn,
    pub load_noise: LoadNoiseFn,
    pub load_residual_resist: LoadResidualFn,
    pub load_residual_react: LoadResidualFn,
    pub load_limit_rhs_resist: LoadResidualFn,
    pub load_limit_rhs_react: LoadResidualFn,
    pub load_spice_rhs_dc: LoadSpiceRhsDcFn,
    pub load_spice_rhs_tran: LoadSpiceRhsTranFn,
    pub load_jacobian_resist: LoadJacobianFn,
    pub load_jacobian_react: LoadJacobianAlphaFn,
    pub load_jacobian_tran: LoadJacobianAlphaFn,
    pub given_flag_model: GivenFlagFn,
    pub given_flag_instance: GivenFlagFn,
    pub num_resistive_jacobian_entries: u32,
    pub num_reactive_jacobian_entries: u32,
    pub write_jacobian_array_resist: WriteJacobianArrayFn,
    pub write_jacobian_array_react: WriteJacobianArrayFn,
    pub num_inputs: u32,
    pub inputs: *mut OsdiNodePair,
    pub load_jacobian_with_offset_resist: LoadJacobianOffsetFn,
    pub load_jacobian_with_offset_react: LoadJacobianOffsetFn,
    pub unknown_nature: *mut OsdiNatureRef,
    pub residual_nature: *mut OsdiNatureRef,
}

impl OsdiDescriptor {
    fn nodes(&self) -> &[OsdiNode] {
        unsafe { slice::from_raw_parts(self.nodes, self.num_nodes as usize) }
    }

    fn params(&self) -> &[OsdiParamOpvar] {
        unsafe { slice::from_raw_parts(self.param_opvar, self.num_params as usize) }
    }

    fn collapsible_pairs(&self) -> &[OsdiNodePair] {
        unsafe { slice::from_raw_parts(self.collapsible, self.num_collapsible as usize) }
    }

    fn matrix_entries(&self) -> &[OsdiJacobianEntry] {
        unsafe { slice::from_raw_parts(self.jacobian_entries, self.num_jacobian_entries as usize) }
    }

    fn check_init_result(&self, res: &OsdiInitInfo) -> Result<()> {
        if (res.flags & EVAL_RET_FLAG_FATAL) != 0 {
            bail!("Verilog-A $fatal was called")
        }

        if res.num_errors != 0 {
            let mut msg = String::new();
            for i in 0..res.num_errors as usize {
                let err = unsafe { &*res.errors.add(i) };
                match err.code {
                    INIT_ERR_OUT_OF_BOUNDS => {
                        let param_id = unsafe { err.payload.parameter_id };
                        let param = &self.params()[param_id as usize];
                        let name = unsafe { osdi_str(*param.name) };
                        msg.push_str(&format!(
                            "value supplied for parameter '{}' is out of bounds\n",
                            name
                        ));
                    }
                    code => msg.push_str(&format!("unknown error: {}\n", code)),
                }
            }
            bail!(msg)
        }
        Ok(())
    }
}

// ============================================================================
// Memory allocation with max alignment
// ============================================================================

#[allow(non_camel_case_types)]
type max_align_t = u128;
const MAX_ALIGN: usize = align_of::<max_align_t>();

fn aligned_size(size: usize) -> usize {
    (size + (MAX_ALIGN - 1)) / MAX_ALIGN
}

fn max_align_layout(size: usize) -> Layout {
    Layout::array::<max_align_t>(aligned_size(size)).unwrap()
}

fn osdi_alloc(size: usize) -> *mut c_void {
    if size == 0 {
        return ptr::null_mut();
    }
    let layout = max_align_layout(size);
    let data = unsafe { alloc_zeroed(layout) } as *mut c_void;
    if data.is_null() {
        handle_alloc_error(layout)
    } else {
        data
    }
}

unsafe fn osdi_dealloc(ptr: *mut c_void, size: usize) {
    if ptr.is_null() {
        return;
    }
    let layout = max_align_layout(size);
    std::alloc::dealloc(ptr as *mut u8, layout)
}

unsafe fn osdi_str(raw: *mut c_char) -> &'static str {
    CStr::from_ptr(raw)
        .to_str()
        .expect("All OSDI strings must be encoded in UTF-8")
}

fn osdi_param_type_str(flags: u32) -> &'static str {
    match flags & PARA_TY_MASK {
        PARA_TY_REAL => "real",
        PARA_TY_INT => "int",
        PARA_TY_STR => "str",
        _ => "unknown",
    }
}

// ============================================================================
// OSDI Logging
// ============================================================================

/// A log entry captured from OSDI $display/$strobe/$write calls.
#[derive(Clone, Debug)]
pub struct OsdiLogEntry {
    pub instance: String,
    pub message: String,
    pub level: u32,
    pub level_name: String,
}

thread_local! {
    /// Thread-local storage for captured log messages.
    static OSDI_LOGS: RefCell<Vec<OsdiLogEntry>> = const { RefCell::new(Vec::new()) };

    /// Whether logging is enabled (if false, logs are printed to stdout but not captured).
    static OSDI_LOG_CAPTURE: Cell<bool> = const { Cell::new(true) };
}

/// The osdi_log callback that OSDI libraries call for $display/$strobe/$write.
unsafe extern "C" fn osdi_log(handle: *mut c_void, msg: *const c_char, lvl: u32) {
    let _ = catch_unwind(|| osdi_log_impl(handle, msg, lvl));
}

unsafe fn osdi_log_impl(handle: *mut c_void, msg: *const c_char, lvl: u32) {
    let instance = if handle.is_null() {
        "unknown".to_string()
    } else {
        let instance_ptr = handle as *const c_char;
        CStr::from_ptr(instance_ptr)
            .to_str()
            .unwrap_or("invalid-utf8")
            .to_string()
    };

    let message = CStr::from_ptr(msg)
        .to_str()
        .unwrap_or("invalid-utf8")
        .to_string();

    let level_name = if (lvl & LOG_FMT_ERR) != 0 {
        "format_error".to_string()
    } else {
        match lvl & LOG_LVL_MASK {
            LOG_LVL_DEBUG => "debug".to_string(),
            LOG_LVL_DISPLAY => "display".to_string(),
            LOG_LVL_INFO => "info".to_string(),
            LOG_LVL_WARN => "warn".to_string(),
            LOG_LVL_ERR => "error".to_string(),
            LOG_LVL_FATAL => "fatal".to_string(),
            _ => format!("unknown({})", lvl),
        }
    };

    let entry = OsdiLogEntry {
        instance: instance.clone(),
        message: message.clone(),
        level: lvl,
        level_name: level_name.clone(),
    };

    // Always print to stdout for visibility
    println!("[OSDI {}] {} - {}", level_name, instance, message);

    // Capture if enabled
    OSDI_LOG_CAPTURE.with(|capture| {
        if capture.get() {
            OSDI_LOGS.with(|logs| {
                logs.borrow_mut().push(entry);
            });
        }
    });
}

/// Get all captured OSDI log entries.
fn get_osdi_logs() -> Vec<OsdiLogEntry> {
    OSDI_LOGS.with(|logs| logs.borrow().clone())
}

/// Clear all captured OSDI log entries.
fn clear_osdi_logs() {
    OSDI_LOGS.with(|logs| logs.borrow_mut().clear());
}

/// Set whether log capture is enabled.
fn set_osdi_log_capture(enabled: bool) {
    OSDI_LOG_CAPTURE.with(|capture| capture.set(enabled));
}

// ============================================================================
// Python classes
// ============================================================================

/// Represents a loaded OSDI shared library containing compiled Verilog-A models.
#[pyclass(unsendable)]
struct OsdiLibrary {
    #[allow(dead_code)]
    library: Arc<Library>,
    descriptor: &'static OsdiDescriptor,
}

#[pymethods]
impl OsdiLibrary {
    /// Load an OSDI shared library from the given path.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        // Load the library
        let library =
            unsafe { Library::new(path) }.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Set up the osdi_log callback for $display/$strobe/$write
        unsafe {
            if let Ok(osdi_log_ptr) =
                library.get::<*mut unsafe extern "C" fn(*mut c_void, *const c_char, u32)>(b"osdi_log\0")
            {
                osdi_log_ptr.write(osdi_log);
            }
        }

        // Get OSDI_NUM_DESCRIPTORS
        let num_descriptors: u32 = unsafe {
            let sym: Symbol<*const u32> = library
                .get(b"OSDI_NUM_DESCRIPTORS\0")
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to find OSDI_NUM_DESCRIPTORS: {}", e)))?;
            **sym
        };

        if num_descriptors == 0 {
            return Err(PyRuntimeError::new_err("OSDI library has no descriptors"));
        }

        // Get OSDI_DESCRIPTORS (inline array)
        let descriptor: &'static OsdiDescriptor = unsafe {
            let sym: Symbol<*const OsdiDescriptor> = library
                .get(b"OSDI_DESCRIPTORS\0")
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to find OSDI_DESCRIPTORS: {}", e)))?;
            &**sym
        };

        Ok(OsdiLibrary {
            library: Arc::new(library),
            descriptor,
        })
    }

    /// Get the device name.
    #[getter]
    fn name(&self) -> &str {
        unsafe { osdi_str(self.descriptor.name) }
    }

    /// Get the number of nodes (terminals + internal).
    #[getter]
    fn num_nodes(&self) -> u32 {
        self.descriptor.num_nodes
    }

    /// Get the number of terminal nodes.
    #[getter]
    fn num_terminals(&self) -> u32 {
        self.descriptor.num_terminals
    }

    /// Get the number of parameters.
    #[getter]
    fn num_params(&self) -> u32 {
        self.descriptor.num_params
    }

    /// Get the number of Jacobian entries.
    #[getter]
    fn num_jacobian_entries(&self) -> u32 {
        self.descriptor.num_jacobian_entries
    }

    /// Get the instance data size in bytes.
    #[getter]
    fn instance_size(&self) -> u32 {
        self.descriptor.instance_size
    }

    /// Get the model data size in bytes.
    #[getter]
    fn model_size(&self) -> u32 {
        self.descriptor.model_size
    }

    /// Get the number of state variables.
    #[getter]
    fn num_states(&self) -> u32 {
        self.descriptor.num_states
    }

    /// Get a list of node names.
    fn get_nodes(&self) -> Vec<String> {
        self.descriptor
            .nodes()
            .iter()
            .map(|n| unsafe { osdi_str(n.name) }.to_string())
            .collect()
    }

    /// Get a list of terminal names.
    fn get_terminals(&self) -> Vec<String> {
        self.descriptor.nodes()[..self.descriptor.num_terminals as usize]
            .iter()
            .map(|n| unsafe { osdi_str(n.name) }.to_string())
            .collect()
    }

    /// Get parameter metadata as a list of dicts.
    fn get_params<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let params = PyList::empty(py);
        for param in self.descriptor.params() {
            let d = PyDict::new(py);
            let name = unsafe { osdi_str(*param.name) };
            let description = unsafe { osdi_str(param.description) };
            let units = unsafe { osdi_str(param.units) };

            d.set_item("name", name).unwrap();
            d.set_item("description", description).unwrap();
            d.set_item("units", units).unwrap();
            d.set_item("type", osdi_param_type_str(param.flags)).unwrap();
            d.set_item("is_instance", (param.flags & PARA_KIND_INST) != 0)
                .unwrap();
            d.set_item("flags", param.flags).unwrap();

            // Get aliases
            let aliases: Vec<String> = unsafe {
                let alias_names = slice::from_raw_parts(param.name.add(1), param.num_alias as usize);
                alias_names.iter().map(|&a| osdi_str(a).to_string()).collect()
            };
            d.set_item("aliases", aliases).unwrap();

            params.append(d).unwrap();
        }
        params
    }

    /// Get Jacobian entries as a list of dicts.
    fn get_jacobian_entries<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let entries = PyList::empty(py);
        for entry in self.descriptor.matrix_entries() {
            let d = PyDict::new(py);
            d.set_item("row", entry.nodes.node_2).unwrap();
            d.set_item("col", entry.nodes.node_1).unwrap();
            d.set_item("flags", entry.flags).unwrap();
            d.set_item("has_resist", (entry.flags & JACOBIAN_ENTRY_RESIST) != 0)
                .unwrap();
            d.set_item("has_react", (entry.flags & JACOBIAN_ENTRY_REACT) != 0)
                .unwrap();
            entries.append(d).unwrap();
        }
        entries
    }

    /// Create a new model with this device.
    fn create_model(&self) -> PyResult<OsdiModel> {
        let data = osdi_alloc(self.descriptor.model_size as usize);
        Ok(OsdiModel {
            library: self.library.clone(),
            descriptor: self.descriptor,
            data,
        })
    }
}

/// Represents model-level data for an OSDI device.
#[pyclass(unsendable)]
struct OsdiModel {
    #[allow(dead_code)]
    library: Arc<Library>,
    descriptor: &'static OsdiDescriptor,
    data: *mut c_void,
}

impl Drop for OsdiModel {
    fn drop(&mut self) {
        unsafe { osdi_dealloc(self.data, self.descriptor.model_size as usize) }
    }
}

#[pymethods]
impl OsdiModel {
    /// Set a real-valued parameter.
    fn set_real_param(&self, param_id: u32, value: f64) -> PyResult<()> {
        let ptr = (self.descriptor.access)(ptr::null_mut(), self.data, param_id, ACCESS_FLAG_SET);
        if ptr.is_null() {
            return Err(PyValueError::new_err(format!(
                "Invalid parameter id: {}",
                param_id
            )));
        }
        unsafe { (ptr as *mut f64).write(value) };
        Ok(())
    }

    /// Set an integer parameter.
    fn set_int_param(&self, param_id: u32, value: i32) -> PyResult<()> {
        let ptr = (self.descriptor.access)(ptr::null_mut(), self.data, param_id, ACCESS_FLAG_SET);
        if ptr.is_null() {
            return Err(PyValueError::new_err(format!(
                "Invalid parameter id: {}",
                param_id
            )));
        }
        unsafe { (ptr as *mut i32).write(value) };
        Ok(())
    }

    /// Set a string parameter.
    fn set_str_param(&self, param_id: u32, value: &str) -> PyResult<()> {
        let cstr = CString::new(value)
            .map_err(|_| PyValueError::new_err("String contains null character"))?;
        let ptr = (self.descriptor.access)(ptr::null_mut(), self.data, param_id, ACCESS_FLAG_SET);
        if ptr.is_null() {
            return Err(PyValueError::new_err(format!(
                "Invalid parameter id: {}",
                param_id
            )));
        }
        // Note: This leaks memory for string parameters (same as original OSDI code)
        unsafe { (ptr as *mut *mut c_char).write(cstr.into_raw()) };
        Ok(())
    }

    /// Process model parameters (call setup_model).
    fn process_params(&self) -> PyResult<()> {
        let mut sim_params = OsdiSimParas {
            names: &mut ptr::null_mut(),
            vals: ptr::null_mut(),
            names_str: &mut ptr::null_mut(),
            vals_str: ptr::null_mut(),
        };

        let mut res = OsdiInitInfo {
            flags: 0,
            num_errors: 0,
            errors: ptr::null_mut(),
        };

        (self.descriptor.setup_model)(
            b"osdi-py\0".as_ptr() as *mut c_void,
            self.data,
            &mut sim_params,
            &mut res,
        );

        self.descriptor
            .check_init_result(&res)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Free error array if allocated
        if res.num_errors != 0 && !res.errors.is_null() {
            unsafe { libc::free(res.errors as *mut c_void) };
        }

        Ok(())
    }

    /// Create a new instance from this model.
    fn create_instance(&self) -> PyResult<OsdiInstance> {
        let data = osdi_alloc(self.descriptor.instance_size as usize);
        Ok(OsdiInstance {
            library: self.library.clone(),
            descriptor: self.descriptor,
            data,
            model_data: self.data,
        })
    }
}

/// Represents instance-level data for an OSDI device.
#[pyclass(unsendable)]
struct OsdiInstance {
    #[allow(dead_code)]
    library: Arc<Library>,
    descriptor: &'static OsdiDescriptor,
    data: *mut c_void,
    model_data: *mut c_void,
}

impl Drop for OsdiInstance {
    fn drop(&mut self) {
        unsafe { osdi_dealloc(self.data, self.descriptor.instance_size as usize) }
    }
}

impl OsdiInstance {
    fn node_mapping(&self) -> &[Cell<u32>] {
        let ptr = self.data as *mut u8;
        unsafe {
            let ptr = ptr.add(self.descriptor.node_mapping_offset as usize) as *mut Cell<u32>;
            slice::from_raw_parts(ptr, self.descriptor.num_nodes as usize)
        }
    }

    fn jacobian_ptrs_resist(&self) -> &[Cell<*mut f64>] {
        let ptr = self.data as *mut u8;
        unsafe {
            let ptr =
                ptr.add(self.descriptor.jacobian_ptr_resist_offset as usize) as *mut Cell<*mut f64>;
            slice::from_raw_parts(ptr, self.descriptor.num_jacobian_entries as usize)
        }
    }
}

#[pymethods]
impl OsdiInstance {
    /// Set a real-valued parameter on this instance.
    fn set_real_param(&self, param_id: u32, value: f64) -> PyResult<()> {
        let ptr = (self.descriptor.access)(
            self.data,
            self.model_data,
            param_id,
            ACCESS_FLAG_SET | ACCESS_FLAG_INSTANCE,
        );
        if ptr.is_null() {
            return Err(PyValueError::new_err(format!(
                "Invalid parameter id: {}",
                param_id
            )));
        }
        unsafe { (ptr as *mut f64).write(value) };
        Ok(())
    }

    /// Set an integer parameter on this instance.
    fn set_int_param(&self, param_id: u32, value: i32) -> PyResult<()> {
        let ptr = (self.descriptor.access)(
            self.data,
            self.model_data,
            param_id,
            ACCESS_FLAG_SET | ACCESS_FLAG_INSTANCE,
        );
        if ptr.is_null() {
            return Err(PyValueError::new_err(format!(
                "Invalid parameter id: {}",
                param_id
            )));
        }
        unsafe { (ptr as *mut i32).write(value) };
        Ok(())
    }

    /// Initialize node mapping (must be called before eval).
    /// terminals: list of node indices for terminal connections
    fn init_node_mapping(&self, terminals: Vec<u32>) -> PyResult<()> {
        let node_mapping = self.node_mapping();

        // Initialize all nodes to themselves first
        for (i, nm) in node_mapping.iter().enumerate() {
            nm.set(i as u32);
        }

        // Set terminal node mappings
        for (i, &terminal) in terminals.iter().enumerate() {
            if i < node_mapping.len() {
                node_mapping[i].set(terminal);
            }
        }

        Ok(())
    }

    /// Process instance parameters (call setup_instance).
    /// temperature: device temperature in Kelvin
    /// num_terminals: number of connected terminals
    fn process_params(&self, temperature: f64, num_terminals: u32) -> PyResult<()> {
        let mut sim_params = OsdiSimParas {
            names: &mut ptr::null_mut(),
            vals: ptr::null_mut(),
            names_str: &mut ptr::null_mut(),
            vals_str: ptr::null_mut(),
        };

        let mut res = OsdiInitInfo {
            flags: 0,
            num_errors: 0,
            errors: ptr::null_mut(),
        };

        (self.descriptor.setup_instance)(
            b"osdi-py\0".as_ptr() as *mut c_void,
            self.data,
            self.model_data,
            temperature,
            num_terminals,
            &mut sim_params,
            &mut res,
        );

        self.descriptor
            .check_init_result(&res)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Free error array if allocated
        if res.num_errors != 0 && !res.errors.is_null() {
            unsafe { libc::free(res.errors as *mut c_void) };
        }

        Ok(())
    }

    /// Evaluate the device model.
    /// prev_solve: array of node voltages
    /// flags: simulation flags (CALC_* and ANALYSIS_* constants)
    /// abstime: absolute simulation time
    /// gmin: minimum conductance for convergence (default 1e-12, use 0.0 for comparison tests)
    #[pyo3(signature = (prev_solve, flags, abstime, gmin=None))]
    fn eval(&self, prev_solve: Vec<f64>, flags: u32, abstime: f64, gmin: Option<f64>) -> PyResult<u32> {
        // Set up default simulation parameters (simparams)
        // These are needed for models that call $simparam() like diode's $simparam("gmin")
        let gmin_val = gmin.unwrap_or(1e-12);
        let gmin_name = b"gmin\0".as_ptr() as *mut c_char;
        let gdev_name = b"gdev\0".as_ptr() as *mut c_char;
        let mut sim_names: [*mut c_char; 3] = [gmin_name, gdev_name, ptr::null_mut()];
        let mut sim_vals: [f64; 2] = [gmin_val, 0.0]; // gmin=user-provided or 1e-12, gdev=0

        // String params (none needed for most models)
        let mut sim_names_str: [*mut c_char; 1] = [ptr::null_mut()];
        let mut sim_vals_str: [*mut c_char; 1] = [ptr::null_mut()];

        let sim_params = OsdiSimParas {
            names: sim_names.as_mut_ptr(),
            vals: sim_vals.as_mut_ptr(),
            names_str: sim_names_str.as_mut_ptr(),
            vals_str: sim_vals_str.as_mut_ptr(),
        };

        let mut info = OsdiSimInfo {
            paras: sim_params,
            abstime,
            prev_solve: prev_solve.as_ptr() as *mut f64,
            prev_state: ptr::null_mut(),
            next_state: ptr::null_mut(),
            flags,
        };

        let ret_flags = (self.descriptor.eval)(
            b"osdi-py\0".as_ptr() as *mut c_void,
            self.data,
            self.model_data,
            &mut info,
        );

        if (ret_flags & EVAL_RET_FLAG_FATAL) != 0 {
            return Err(PyRuntimeError::new_err("Simulation aborted with $fatal"));
        }

        Ok(ret_flags)
    }

    /// Load resistive residuals into the provided array.
    fn load_residual_resist(&self, residual: Vec<f64>) -> Vec<f64> {
        let mut result = residual;
        (self.descriptor.load_residual_resist)(self.data, self.model_data, result.as_mut_ptr());
        result
    }

    /// Load reactive residuals into the provided array.
    fn load_residual_react(&self, residual: Vec<f64>) -> Vec<f64> {
        let mut result = residual;
        (self.descriptor.load_residual_react)(self.data, self.model_data, result.as_mut_ptr());
        result
    }

    /// Load resistive Jacobian entries into the matrix.
    fn load_jacobian_resist(&self) {
        (self.descriptor.load_jacobian_resist)(self.data, self.model_data);
    }

    /// Load reactive Jacobian entries into the matrix with alpha scaling.
    fn load_jacobian_react(&self, alpha: f64) {
        (self.descriptor.load_jacobian_react)(self.data, self.model_data, alpha);
    }

    /// Write resistive Jacobian entries to an array (for sparse matrix building).
    fn write_jacobian_array_resist(&self) -> Vec<f64> {
        let n = self.descriptor.num_resistive_jacobian_entries as usize;
        let mut result = vec![0.0; n];
        (self.descriptor.write_jacobian_array_resist)(
            self.data,
            self.model_data,
            result.as_mut_ptr(),
        );
        result
    }

    /// Write reactive Jacobian entries to an array (for sparse matrix building).
    fn write_jacobian_array_react(&self) -> Vec<f64> {
        let n = self.descriptor.num_reactive_jacobian_entries as usize;
        let mut result = vec![0.0; n];
        (self.descriptor.write_jacobian_array_react)(
            self.data,
            self.model_data,
            result.as_mut_ptr(),
        );
        result
    }

    /// Get the number of resistive Jacobian entries.
    #[getter]
    fn num_resistive_jacobian_entries(&self) -> u32 {
        self.descriptor.num_resistive_jacobian_entries
    }

    /// Get the number of reactive Jacobian entries.
    #[getter]
    fn num_reactive_jacobian_entries(&self) -> u32 {
        self.descriptor.num_reactive_jacobian_entries
    }

    /// Get a real-valued parameter from the instance.
    fn get_real_param(&self, param_id: u32) -> PyResult<f64> {
        let ptr = (self.descriptor.access)(
            self.data,
            self.model_data,
            param_id,
            ACCESS_FLAG_READ | ACCESS_FLAG_INSTANCE,
        );
        if ptr.is_null() {
            return Err(PyValueError::new_err(format!(
                "Invalid parameter id: {}",
                param_id
            )));
        }
        Ok(unsafe { *(ptr as *const f64) })
    }

    /// Get an integer parameter from the instance.
    fn get_int_param(&self, param_id: u32) -> PyResult<i32> {
        let ptr = (self.descriptor.access)(
            self.data,
            self.model_data,
            param_id,
            ACCESS_FLAG_READ | ACCESS_FLAG_INSTANCE,
        );
        if ptr.is_null() {
            return Err(PyValueError::new_err(format!(
                "Invalid parameter id: {}",
                param_id
            )));
        }
        Ok(unsafe { *(ptr as *const i32) })
    }

    /// Get the number of state variables.
    #[getter]
    fn num_states(&self) -> u32 {
        self.descriptor.num_states
    }

    /// Get a state value by index.
    fn get_state(&self, state_idx: u32) -> PyResult<f64> {
        if state_idx >= self.descriptor.num_states {
            return Err(PyValueError::new_err(format!(
                "State index {} out of range (num_states={})",
                state_idx, self.descriptor.num_states
            )));
        }
        let ptr = self.data as *mut u8;
        let state_ptr = unsafe {
            ptr.add(self.descriptor.state_idx_off as usize)
                .add((state_idx as usize) * std::mem::size_of::<f64>()) as *const f64
        };
        Ok(unsafe { *state_ptr })
    }

    /// Dump all state values.
    fn get_all_states(&self) -> Vec<f64> {
        let mut states = Vec::with_capacity(self.descriptor.num_states as usize);
        let ptr = self.data as *mut u8;
        for i in 0..self.descriptor.num_states {
            let state_ptr = unsafe {
                ptr.add(self.descriptor.state_idx_off as usize)
                    .add((i as usize) * std::mem::size_of::<f64>()) as *const f64
            };
            states.push(unsafe { *state_ptr });
        }
        states
    }

    /// Dump all parameter values (for debugging).
    /// Returns a list of dicts with param info and current values.
    /// Tries both instance and model level access.
    fn dump_all_params<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let params = PyList::empty(py);
        for (i, param) in self.descriptor.params().iter().enumerate() {
            let d = PyDict::new(py);
            let name = unsafe { osdi_str(*param.name) };
            let param_type = osdi_param_type_str(param.flags);
            let is_instance = (param.flags & PARA_KIND_INST) != 0;

            d.set_item("id", i).unwrap();
            d.set_item("name", name).unwrap();
            d.set_item("type", param_type).unwrap();
            d.set_item("is_instance", is_instance).unwrap();

            // Try instance-level access first, then model-level
            let mut ptr = (self.descriptor.access)(
                self.data,
                self.model_data,
                i as u32,
                ACCESS_FLAG_READ | ACCESS_FLAG_INSTANCE,
            );

            // If instance access returned null, try model-level
            if ptr.is_null() {
                ptr = (self.descriptor.access)(
                    ptr::null_mut(),
                    self.model_data,
                    i as u32,
                    ACCESS_FLAG_READ,
                );
            }

            if !ptr.is_null() {
                match param.flags & PARA_TY_MASK {
                    PARA_TY_REAL => {
                        let val = unsafe { *(ptr as *const f64) };
                        d.set_item("value", val).unwrap();
                    }
                    PARA_TY_INT => {
                        let val = unsafe { *(ptr as *const i32) };
                        d.set_item("value", val).unwrap();
                    }
                    PARA_TY_STR => {
                        let val_ptr = unsafe { *(ptr as *const *const c_char) };
                        if !val_ptr.is_null() {
                            let val = unsafe { CStr::from_ptr(val_ptr).to_string_lossy() };
                            d.set_item("value", val.to_string()).unwrap();
                        } else {
                            d.set_item("value", py.None()).unwrap();
                        }
                    }
                    _ => {
                        d.set_item("value", py.None()).unwrap();
                    }
                }
            } else {
                d.set_item("value", py.None()).unwrap();
            }

            params.append(d).unwrap();
        }
        params
    }
}

// ============================================================================
// Module-level functions for OSDI logging
// ============================================================================

/// Get all captured OSDI log entries as a list of dicts.
#[pyfunction]
fn get_logs(py: Python<'_>) -> Bound<'_, PyList> {
    let logs = get_osdi_logs();
    let result = PyList::empty(py);
    for entry in logs {
        let d = PyDict::new(py);
        d.set_item("instance", &entry.instance).unwrap();
        d.set_item("message", &entry.message).unwrap();
        d.set_item("level", entry.level).unwrap();
        d.set_item("level_name", &entry.level_name).unwrap();
        result.append(d).unwrap();
    }
    result
}

/// Clear all captured OSDI log entries.
#[pyfunction]
fn clear_logs() {
    clear_osdi_logs();
}

/// Set whether log capture is enabled.
/// If disabled, logs are still printed to stdout but not captured.
#[pyfunction]
fn set_log_capture(enabled: bool) {
    set_osdi_log_capture(enabled);
}

// ============================================================================
// Python module definition
// ============================================================================

/// Python module for OSDI device access.
#[pymodule]
fn osdi_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OsdiLibrary>()?;
    m.add_class::<OsdiModel>()?;
    m.add_class::<OsdiInstance>()?;

    // Export log functions
    m.add_function(wrap_pyfunction!(get_logs, m)?)?;
    m.add_function(wrap_pyfunction!(clear_logs, m)?)?;
    m.add_function(wrap_pyfunction!(set_log_capture, m)?)?;

    // Export constants
    m.add("PARA_TY_REAL", PARA_TY_REAL)?;
    m.add("PARA_TY_INT", PARA_TY_INT)?;
    m.add("PARA_TY_STR", PARA_TY_STR)?;
    m.add("PARA_KIND_MODEL", PARA_KIND_MODEL)?;
    m.add("PARA_KIND_INST", PARA_KIND_INST)?;
    m.add("PARA_KIND_OPVAR", PARA_KIND_OPVAR)?;

    m.add("CALC_RESIST_RESIDUAL", CALC_RESIST_RESIDUAL)?;
    m.add("CALC_REACT_RESIDUAL", CALC_REACT_RESIDUAL)?;
    m.add("CALC_RESIST_JACOBIAN", CALC_RESIST_JACOBIAN)?;
    m.add("CALC_REACT_JACOBIAN", CALC_REACT_JACOBIAN)?;
    m.add("CALC_NOISE", CALC_NOISE)?;
    m.add("CALC_OP", CALC_OP)?;

    m.add("ANALYSIS_DC", ANALYSIS_DC)?;
    m.add("ANALYSIS_AC", ANALYSIS_AC)?;
    m.add("ANALYSIS_TRAN", ANALYSIS_TRAN)?;
    m.add("ANALYSIS_STATIC", ANALYSIS_STATIC)?;

    // Log level constants
    m.add("LOG_LVL_DEBUG", LOG_LVL_DEBUG)?;
    m.add("LOG_LVL_DISPLAY", LOG_LVL_DISPLAY)?;
    m.add("LOG_LVL_INFO", LOG_LVL_INFO)?;
    m.add("LOG_LVL_WARN", LOG_LVL_WARN)?;
    m.add("LOG_LVL_ERR", LOG_LVL_ERR)?;
    m.add("LOG_LVL_FATAL", LOG_LVL_FATAL)?;

    Ok(())
}
