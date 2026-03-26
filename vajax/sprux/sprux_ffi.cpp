/**
 * Sprux FFI for JAX
 *
 * XLA FFI bindings for the Sprux Metal-accelerated sparse LU solver.
 * Provides solve and SpMV operations for JAX on Apple Silicon.
 *
 * The SpruxFFISolver handles all preprocessing (BTF, equilibration),
 * f32 GPU factorization, and f64 iterative refinement internally.
 * The JAX FFI interface is identical to UMFPACK: (indptr, indices, data, b) → x.
 */

#include <cstdint>
#include <memory>
#include <mutex>

// XLA FFI API
#include "xla/ffi/api/ffi.h"

// Sprux FFI solver + capture API
#include "sprux/sprux/SpruxFFISolver.h"
#include "sprux/sprux/sprux_c_api.h"

// Nanobind for Python bindings
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
namespace ffi = xla::ffi;

// =============================================================================
// Cached solver (reused across solves with same sparsity pattern)
// =============================================================================

namespace {

struct SpruxCache {
    std::unique_ptr<Sprux::SpruxFFISolver> solver;
    int32_t n = 0;
    int32_t nnz = 0;
    std::mutex mutex;

    // Prevent crash during Python shutdown: release solver before
    // global destructors run (mutex may be destroyed first).
    ~SpruxCache() {
        solver.reset();
    }
};

// Use a pointer to avoid global destructor ordering issues.
// Leaked intentionally — the OS reclaims the memory on exit.
SpruxCache& cache() {
    static SpruxCache* instance = new SpruxCache();
    return *instance;
}

}  // namespace

// =============================================================================
// XLA FFI Handler: solve Ax = b
// =============================================================================

ffi::Error SpruxSolveF64Impl(
    ffi::Buffer<ffi::DataType::S32> csr_indptr,
    ffi::Buffer<ffi::DataType::S32> csr_indices,
    ffi::Buffer<ffi::DataType::F64> csr_data,
    ffi::Buffer<ffi::DataType::F64> b,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x
) {
    int64_t n = b.dimensions()[0];
    int64_t nnz = csr_indices.dimensions()[0];

    if (csr_indptr.dimensions()[0] != n + 1) {
        return ffi::Error::InvalidArgument(
            "csr_indptr must have length n+1");
    }

    const int32_t* indptr_ptr = csr_indptr.typed_data();
    const int32_t* indices_ptr = csr_indices.typed_data();
    const double* data_ptr = csr_data.typed_data();
    const double* b_ptr = b.typed_data();
    double* x_ptr = x->typed_data();

    int32_t n32 = static_cast<int32_t>(n);
    int32_t nnz32 = static_cast<int32_t>(nnz);

    std::lock_guard<std::mutex> lock(cache().mutex);

    // Create solver on first call (or if dimensions changed)
    if (!cache().solver || cache().n != n32 || cache().nnz != nnz32) {
        // Default to 10 refinement steps — converges to machine epsilon
        // for typical circuit Jacobians (c6288 needs ~7 steps)
        cache().solver = std::make_unique<Sprux::SpruxFFISolver>(
            n32, nnz32, indptr_ptr, indices_ptr, data_ptr,
            /*max_refine_steps=*/10);
        cache().n = n32;
        cache().nnz = nnz32;
    }

    cache().solver->solve(data_ptr, b_ptr, x_ptr);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sprux_solve_f64,
    SpruxSolveF64Impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // csr_indptr
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // csr_indices
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // csr_data
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // b
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // x
);

// =============================================================================
// XLA FFI Handler: solve with cached factorization (chord Newton)
// =============================================================================

ffi::Error SpruxSolveOnlyF64Impl(
    ffi::Buffer<ffi::DataType::S32> csr_indptr,
    ffi::Buffer<ffi::DataType::S32> csr_indices,
    ffi::Buffer<ffi::DataType::F64> csr_data,
    ffi::Buffer<ffi::DataType::F64> b,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x
) {
    int64_t n = b.dimensions()[0];

    const double* data_ptr = csr_data.typed_data();
    const double* b_ptr = b.typed_data();
    double* x_ptr = x->typed_data();

    std::lock_guard<std::mutex> lock(cache().mutex);

    if (!cache().solver) {
        return ffi::Error::Internal(
            "sprux_solve_only called before sprux_solve (no cached factorization)");
    }

    cache().solver->solveOnly(data_ptr, b_ptr, x_ptr);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sprux_solve_only_f64,
    SpruxSolveOnlyF64Impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // csr_indptr (structure, for consistency)
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // csr_indices
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // csr_data (for refinement SpMV)
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // b
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // x
);

// =============================================================================
// XLA FFI Handler: sparse matrix-vector multiply b = A @ x
// =============================================================================

ffi::Error SpruxDotF64Impl(
    ffi::Buffer<ffi::DataType::S32> csr_indptr,
    ffi::Buffer<ffi::DataType::S32> csr_indices,
    ffi::Buffer<ffi::DataType::F64> csr_data,
    ffi::Buffer<ffi::DataType::F64> x,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> b
) {
    int64_t n = x.dimensions()[0];
    int64_t nnz = csr_indices.dimensions()[0];

    const int32_t* indptr_ptr = csr_indptr.typed_data();
    const int32_t* indices_ptr = csr_indices.typed_data();
    const double* data_ptr = csr_data.typed_data();
    const double* x_ptr = x.typed_data();
    double* b_ptr = b->typed_data();

    int32_t n32 = static_cast<int32_t>(n);
    int32_t nnz32 = static_cast<int32_t>(nnz);

    std::lock_guard<std::mutex> lock(cache().mutex);

    // Create solver if needed (dot doesn't need init data, but solver
    // must exist for the CSR structure). Use the provided data as init.
    if (!cache().solver || cache().n != n32 || cache().nnz != nnz32) {
        cache().solver = std::make_unique<Sprux::SpruxFFISolver>(
            n32, nnz32, indptr_ptr, indices_ptr, data_ptr,
            /*max_refine_steps=*/10);
        cache().n = n32;
        cache().nnz = nnz32;
    }

    cache().solver->dot(data_ptr, x_ptr, b_ptr);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    sprux_dot_f64,
    SpruxDotF64Impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // csr_indptr
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // csr_indices
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // csr_data
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // x
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // b
);

// =============================================================================
// Python module definition using nanobind
// =============================================================================

NB_MODULE(sprux_jax_cpp, m) {
    m.doc() = "Sprux Metal FFI for JAX - sparse LU solver for Apple Silicon";

    // Export FFI handler capsules for JAX registration
    m.def("sprux_solve_f64", []() {
        return nb::capsule(reinterpret_cast<void*>(sprux_solve_f64));
    }, "Get FFI capsule for Sprux solve (float64)");

    m.def("sprux_dot_f64", []() {
        return nb::capsule(reinterpret_cast<void*>(sprux_dot_f64));
    }, "Get FFI capsule for sparse matrix-vector multiply (float64)");

    m.def("sprux_solve_only_f64", []() {
        return nb::capsule(reinterpret_cast<void*>(sprux_solve_only_f64));
    }, "Get FFI capsule for solve with cached factorization (chord Newton)");

    // Utility to clear solver cache
    m.def("clear_cache", []() {
        std::lock_guard<std::mutex> lock(cache().mutex);
        cache().solver.reset();
        cache().n = 0;
        cache().nnz = 0;
    }, "Clear the cached solver (call when switching circuits)");

    // Split-phase solve (for pipelined batch processing)
    m.def("begin_solve", [](
        nb::ndarray<const int32_t, nb::ndim<1>> indptr,
        nb::ndarray<const int32_t, nb::ndim<1>> indices,
        nb::ndarray<const double, nb::ndim<1>> data,
        nb::ndarray<const double, nb::ndim<1>> rhs
    ) {
        std::lock_guard<std::mutex> lock(cache().mutex);
        int32_t n32 = static_cast<int32_t>(rhs.shape(0));
        int32_t nnz32 = static_cast<int32_t>(indices.shape(0));

        if (!cache().solver || cache().n != n32 || cache().nnz != nnz32) {
            cache().solver = std::make_unique<Sprux::SpruxFFISolver>(
                n32, nnz32, indptr.data(), indices.data(), data.data(),
                /*max_refine_steps=*/10);
            cache().n = n32;
            cache().nnz = nnz32;
        }
        cache().solver->beginSolve(data.data(), rhs.data());
    }, "Submit GPU factor+solve asynchronously (call end_solve to get result)");

    m.def("end_solve", [](nb::ndarray<double, nb::ndim<1>> x_out) {
        std::lock_guard<std::mutex> lock(cache().mutex);
        if (!cache().solver) {
            throw std::runtime_error("end_solve called without begin_solve");
        }
        return cache().solver->endSolve(x_out.data());
    }, "Complete iterative refinement and write result");

    // GPU trace capture
    m.def("begin_capture", [](const char* path) {
        return sprux_begin_capture(path) != 0;
    }, "Start GPU trace capture to .gputrace file");

    m.def("end_capture", []() {
        sprux_end_capture();
    }, "Stop GPU trace capture");
}
