/**
 * UMFPACK FFI for JAX
 *
 * This module provides XLA FFI (Foreign Function Interface) bindings for UMFPACK,
 * a high-performance sparse direct solver. This eliminates the ~100ms pure_callback
 * overhead by registering UMFPACK operations directly as XLA custom calls.
 *
 * The implementation follows the klujax pattern for JAX FFI integration.
 */

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// XLA FFI API
#include "xla/ffi/api/ffi.h"

// UMFPACK headers
extern "C" {
#include "umfpack.h"
}

// Nanobind for Python bindings
#include <nanobind/nanobind.h>

namespace nb = nanobind;
namespace ffi = xla::ffi;

//==============================================================================
// Error handling utilities
//==============================================================================

namespace {

const char* UmfpackStatusString(int status) {
    switch (status) {
        case UMFPACK_OK: return "OK";
        case UMFPACK_WARNING_singular_matrix: return "WARNING: singular matrix";
        case UMFPACK_WARNING_determinant_underflow: return "WARNING: determinant underflow";
        case UMFPACK_WARNING_determinant_overflow: return "WARNING: determinant overflow";
        case UMFPACK_ERROR_out_of_memory: return "ERROR: out of memory";
        case UMFPACK_ERROR_invalid_Numeric_object: return "ERROR: invalid Numeric object";
        case UMFPACK_ERROR_invalid_Symbolic_object: return "ERROR: invalid Symbolic object";
        case UMFPACK_ERROR_argument_missing: return "ERROR: argument missing";
        case UMFPACK_ERROR_n_nonpositive: return "ERROR: n nonpositive";
        case UMFPACK_ERROR_invalid_matrix: return "ERROR: invalid matrix";
        case UMFPACK_ERROR_different_pattern: return "ERROR: different pattern";
        case UMFPACK_ERROR_invalid_system: return "ERROR: invalid system";
        case UMFPACK_ERROR_invalid_permutation: return "ERROR: invalid permutation";
        case UMFPACK_ERROR_internal_error: return "ERROR: internal error";
        case UMFPACK_ERROR_file_IO: return "ERROR: file I/O";
        default: return "UNKNOWN";
    }
}

// Convert CSR format to CSC format (UMFPACK requires column-major)
// This is done in-place in the output arrays
void CsrToCsc(
    int32_t n,           // matrix dimension (n x n)
    int32_t nnz,         // number of non-zeros
    const int32_t* csr_indptr,   // CSR row pointers (n+1)
    const int32_t* csr_indices,  // CSR column indices (nnz)
    const double* csr_data,      // CSR values (nnz)
    int32_t* csc_indptr,  // CSC column pointers (n+1) - output
    int32_t* csc_indices, // CSC row indices (nnz) - output
    double* csc_data      // CSC values (nnz) - output
) {
    // Count non-zeros per column
    std::fill_n(csc_indptr, n + 1, 0);
    for (int32_t i = 0; i < nnz; ++i) {
        csc_indptr[csr_indices[i] + 1]++;
    }

    // Cumulative sum to get column pointers
    for (int32_t j = 0; j < n; ++j) {
        csc_indptr[j + 1] += csc_indptr[j];
    }

    // Fill in row indices and values
    std::vector<int32_t> col_counts(n, 0);
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t k = csr_indptr[i]; k < csr_indptr[i + 1]; ++k) {
            int32_t j = csr_indices[k];
            int32_t pos = csc_indptr[j] + col_counts[j];
            csc_indices[pos] = i;
            csc_data[pos] = csr_data[k];
            col_counts[j]++;
        }
    }
}

}  // namespace

//==============================================================================
// Cached factorization for repeated solves
//==============================================================================

namespace {

// Cache for symbolic factorization (reused across solves with same sparsity)
struct UmfpackCache {
    void* Symbolic = nullptr;
    int32_t n = 0;
    int32_t nnz = 0;
    std::vector<int32_t> csc_indptr;
    std::vector<int32_t> csc_indices;
    double Control[UMFPACK_CONTROL];
    double Info[UMFPACK_INFO];
    std::mutex mutex;

    UmfpackCache() {
        umfpack_di_defaults(Control);
        // Optimize for speed over accuracy
        Control[UMFPACK_PIVOT_TOLERANCE] = 0.1;
    }

    ~UmfpackCache() {
        if (Symbolic) {
            umfpack_di_free_symbolic(&Symbolic);
        }
    }

    // Check if cache is valid for given dimensions
    bool IsValid(int32_t new_n, int32_t new_nnz) const {
        return Symbolic != nullptr && n == new_n && nnz == new_nnz;
    }
};

// Global cache - one per unique sparsity pattern hash
// For simplicity, we use a single global cache. For production use with
// multiple sparsity patterns, you'd want a hash map keyed by (indptr, indices).
std::unique_ptr<UmfpackCache> g_cache;
std::once_flag g_cache_init;

UmfpackCache& GetCache() {
    std::call_once(g_cache_init, []() {
        g_cache = std::make_unique<UmfpackCache>();
    });
    return *g_cache;
}

}  // namespace

//==============================================================================
// XLA FFI Handler for UMFPACK solve
//==============================================================================

// Solve Ax = b where A is sparse (CSR format) using UMFPACK
// This is the main FFI handler called from JAX
ffi::Error UmfpackSolveF64Impl(
    ffi::Buffer<ffi::DataType::S32> csr_indptr,   // CSR row pointers
    ffi::Buffer<ffi::DataType::S32> csr_indices,  // CSR column indices
    ffi::Buffer<ffi::DataType::F64> csr_data,     // CSR values
    ffi::Buffer<ffi::DataType::F64> b,            // RHS vector
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x  // Solution vector (output)
) {
    // Extract dimensions
    int64_t n = b.dimensions()[0];
    int64_t nnz = csr_indices.dimensions()[0];

    // Validate inputs
    if (csr_indptr.dimensions()[0] != n + 1) {
        return ffi::Error::InvalidArgument(
            "csr_indptr must have length n+1");
    }

    // Get pointers to data
    const int32_t* indptr_ptr = csr_indptr.typed_data();
    const int32_t* indices_ptr = csr_indices.typed_data();
    const double* data_ptr = csr_data.typed_data();
    const double* b_ptr = b.typed_data();
    double* x_ptr = x->typed_data();

    // Get cache (with thread safety)
    UmfpackCache& cache = GetCache();
    std::lock_guard<std::mutex> lock(cache.mutex);

    // Allocate temporary CSC arrays
    std::vector<int32_t> csc_indptr(n + 1);
    std::vector<int32_t> csc_indices(nnz);
    std::vector<double> csc_data(nnz);

    // Convert CSR to CSC
    CsrToCsc(
        static_cast<int32_t>(n),
        static_cast<int32_t>(nnz),
        indptr_ptr, indices_ptr, data_ptr,
        csc_indptr.data(), csc_indices.data(), csc_data.data()
    );

    // Symbolic factorization (reuse if sparsity pattern unchanged)
    bool need_symbolic = !cache.IsValid(n, nnz);

    if (need_symbolic) {
        // Free old symbolic if exists
        if (cache.Symbolic) {
            umfpack_di_free_symbolic(&cache.Symbolic);
            cache.Symbolic = nullptr;
        }

        // Store pattern for future reference
        cache.n = static_cast<int32_t>(n);
        cache.nnz = static_cast<int32_t>(nnz);
        cache.csc_indptr = csc_indptr;
        cache.csc_indices = csc_indices;

        // Perform symbolic factorization
        int status = umfpack_di_symbolic(
            static_cast<int>(n),
            static_cast<int>(n),
            csc_indptr.data(),
            csc_indices.data(),
            csc_data.data(),
            &cache.Symbolic,
            cache.Control,
            cache.Info
        );

        if (status != UMFPACK_OK) {
            return ffi::Error::Internal(
                std::string("UMFPACK symbolic: ") + UmfpackStatusString(status));
        }
    }

    // Numeric factorization (must be done every time values change)
    void* Numeric = nullptr;
    int status = umfpack_di_numeric(
        csc_indptr.data(),
        csc_indices.data(),
        csc_data.data(),
        cache.Symbolic,
        &Numeric,
        cache.Control,
        cache.Info
    );

    if (status != UMFPACK_OK && status != UMFPACK_WARNING_singular_matrix) {
        return ffi::Error::Internal(
            std::string("UMFPACK numeric: ") + UmfpackStatusString(status));
    }

    // Solve Ax = b
    status = umfpack_di_solve(
        UMFPACK_A,  // Solve Ax = b (not A^T x = b)
        csc_indptr.data(),
        csc_indices.data(),
        csc_data.data(),
        x_ptr,
        b_ptr,
        Numeric,
        cache.Control,
        cache.Info
    );

    // Clean up numeric factorization (cannot be reused)
    umfpack_di_free_numeric(&Numeric);

    // Allow singular matrix warning - the solve may still produce useful results
    // (same behavior as scikit-umfpack which issues a warning but continues)
    if (status != UMFPACK_OK && status != UMFPACK_WARNING_singular_matrix) {
        return ffi::Error::Internal(
            std::string("UMFPACK solve: ") + UmfpackStatusString(status));
    }

    return ffi::Error::Success();
}

// XLA FFI handler definition
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    umfpack_solve_f64,
    UmfpackSolveF64Impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // csr_indptr
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // csr_indices
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // csr_data
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // b
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // x
);

//==============================================================================
// Sparse matrix-vector multiply: b = A @ x
// Useful for residual computation and autodiff
//==============================================================================

ffi::Error UmfpackDotF64Impl(
    ffi::Buffer<ffi::DataType::S32> csr_indptr,   // CSR row pointers
    ffi::Buffer<ffi::DataType::S32> csr_indices,  // CSR column indices
    ffi::Buffer<ffi::DataType::F64> csr_data,     // CSR values
    ffi::Buffer<ffi::DataType::F64> x,            // Input vector
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> b  // Output vector
) {
    int64_t n = x.dimensions()[0];

    const int32_t* indptr_ptr = csr_indptr.typed_data();
    const int32_t* indices_ptr = csr_indices.typed_data();
    const double* data_ptr = csr_data.typed_data();
    const double* x_ptr = x.typed_data();
    double* b_ptr = b->typed_data();

    // CSR matrix-vector multiply: b = A @ x
    for (int64_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int32_t k = indptr_ptr[i]; k < indptr_ptr[i + 1]; ++k) {
            sum += data_ptr[k] * x_ptr[indices_ptr[k]];
        }
        b_ptr[i] = sum;
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    umfpack_dot_f64,
    UmfpackDotF64Impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // csr_indptr
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // csr_indices
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // csr_data
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // x
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // b
);

//==============================================================================
// Solve with transposed matrix: A^T x = b
// Needed for reverse-mode autodiff
//==============================================================================

ffi::Error UmfpackSolveTransposeF64Impl(
    ffi::Buffer<ffi::DataType::S32> csr_indptr,
    ffi::Buffer<ffi::DataType::S32> csr_indices,
    ffi::Buffer<ffi::DataType::F64> csr_data,
    ffi::Buffer<ffi::DataType::F64> b,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x
) {
    int64_t n = b.dimensions()[0];
    int64_t nnz = csr_indices.dimensions()[0];

    const int32_t* indptr_ptr = csr_indptr.typed_data();
    const int32_t* indices_ptr = csr_indices.typed_data();
    const double* data_ptr = csr_data.typed_data();
    const double* b_ptr = b.typed_data();
    double* x_ptr = x->typed_data();

    UmfpackCache& cache = GetCache();
    std::lock_guard<std::mutex> lock(cache.mutex);

    // Convert CSR to CSC
    std::vector<int32_t> csc_indptr(n + 1);
    std::vector<int32_t> csc_indices(nnz);
    std::vector<double> csc_data(nnz);

    CsrToCsc(
        static_cast<int32_t>(n),
        static_cast<int32_t>(nnz),
        indptr_ptr, indices_ptr, data_ptr,
        csc_indptr.data(), csc_indices.data(), csc_data.data()
    );

    // Reuse symbolic if available
    bool need_symbolic = !cache.IsValid(n, nnz);

    if (need_symbolic) {
        if (cache.Symbolic) {
            umfpack_di_free_symbolic(&cache.Symbolic);
            cache.Symbolic = nullptr;
        }

        cache.n = static_cast<int32_t>(n);
        cache.nnz = static_cast<int32_t>(nnz);
        cache.csc_indptr = csc_indptr;
        cache.csc_indices = csc_indices;

        int status = umfpack_di_symbolic(
            static_cast<int>(n), static_cast<int>(n),
            csc_indptr.data(), csc_indices.data(), csc_data.data(),
            &cache.Symbolic, cache.Control, cache.Info
        );

        if (status != UMFPACK_OK) {
            return ffi::Error::Internal(
                std::string("UMFPACK symbolic: ") + UmfpackStatusString(status));
        }
    }

    void* Numeric = nullptr;
    int status = umfpack_di_numeric(
        csc_indptr.data(), csc_indices.data(), csc_data.data(),
        cache.Symbolic, &Numeric, cache.Control, cache.Info
    );

    if (status != UMFPACK_OK && status != UMFPACK_WARNING_singular_matrix) {
        return ffi::Error::Internal(
            std::string("UMFPACK numeric: ") + UmfpackStatusString(status));
    }

    // Solve A^T x = b (transpose solve)
    status = umfpack_di_solve(
        UMFPACK_At,  // Solve A^T x = b
        csc_indptr.data(), csc_indices.data(), csc_data.data(),
        x_ptr, b_ptr, Numeric, cache.Control, cache.Info
    );

    umfpack_di_free_numeric(&Numeric);

    // Allow singular matrix warning - same as regular solve
    if (status != UMFPACK_OK && status != UMFPACK_WARNING_singular_matrix) {
        return ffi::Error::Internal(
            std::string("UMFPACK solve transpose: ") + UmfpackStatusString(status));
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    umfpack_solve_transpose_f64,
    UmfpackSolveTransposeF64Impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
);

//==============================================================================
// Clear symbolic factorization cache
// Call this when switching between matrices with different sparsity patterns
//==============================================================================

ffi::Error UmfpackClearCacheImpl() {
    UmfpackCache& cache = GetCache();
    std::lock_guard<std::mutex> lock(cache.mutex);

    if (cache.Symbolic) {
        umfpack_di_free_symbolic(&cache.Symbolic);
        cache.Symbolic = nullptr;
    }
    cache.n = 0;
    cache.nnz = 0;
    cache.csc_indptr.clear();
    cache.csc_indices.clear();

    return ffi::Error::Success();
}

//==============================================================================
// Python module definition using nanobind
//==============================================================================

NB_MODULE(umfpack_jax_cpp, m) {
    m.doc() = "UMFPACK FFI for JAX - high-performance sparse direct solver";

    // Export FFI handler capsules for JAX registration
    m.def("umfpack_solve_f64", []() {
        return nb::capsule(reinterpret_cast<void*>(umfpack_solve_f64));
    }, "Get FFI capsule for UMFPACK solve (float64)");

    m.def("umfpack_dot_f64", []() {
        return nb::capsule(reinterpret_cast<void*>(umfpack_dot_f64));
    }, "Get FFI capsule for sparse matrix-vector multiply (float64)");

    m.def("umfpack_solve_transpose_f64", []() {
        return nb::capsule(reinterpret_cast<void*>(umfpack_solve_transpose_f64));
    }, "Get FFI capsule for UMFPACK transpose solve (float64)");

    // Utility to clear factorization cache
    m.def("clear_cache", []() {
        UmfpackClearCacheImpl();
    }, "Clear the symbolic factorization cache");
}
