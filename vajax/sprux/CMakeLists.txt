cmake_minimum_required(VERSION 3.18...3.29)
project(sprux_jax_cpp LANGUAGES CXX)

# C++17 required for XLA FFI headers
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and nanobind
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

# =============================================================================
# FetchContent: build Sprux from source (Metal backend on macOS)
# Override with FETCHCONTENT_SOURCE_DIR_SPRUX for local development.
# =============================================================================
include(FetchContent)

# SuiteSparse needs old cmake_minimum_required compat with CMake 4.x
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

message(STATUS "Fetching Sprux (Metal sparse solver)")

FetchContent_Declare(
    Sprux
    GIT_REPOSITORY https://github.com/ChipFlow/Sprux.git
    GIT_TAG metal-backend
)

# Sprux build options: Metal backend, no CUDA, no tests/examples
set(SPRUX_USE_CUBLAS OFF CACHE BOOL "" FORCE)
set(SPRUX_USE_METAL ON CACHE BOOL "" FORCE)
set(SPRUX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPRUX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BLA_VENDOR Apple CACHE STRING "" FORCE)

FetchContent_MakeAvailable(Sprux)

# Get jaxlib include path for XLA FFI headers
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import jaxlib; import os; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE JAXLIB_INCLUDE_DIR
)
message(STATUS "jaxlib include dir: ${JAXLIB_INCLUDE_DIR}")

if(NOT EXISTS "${JAXLIB_INCLUDE_DIR}/xla/ffi/api/ffi.h")
    message(FATAL_ERROR "XLA FFI headers not found at ${JAXLIB_INCLUDE_DIR}")
endif()

# Build the Python extension module
nanobind_add_module(sprux_jax_cpp STABLE_ABI
    sprux_ffi.cpp
)

target_include_directories(sprux_jax_cpp PRIVATE
    ${JAXLIB_INCLUDE_DIR}
    ${sprux_SOURCE_DIR}   # Sprux repo root (for #include "sprux/sprux/...")
)

target_link_libraries(sprux_jax_cpp PRIVATE
    Sprux_static
)

# The Metal shader library (MetalKernels.metallib) is compiled by Sprux's
# CMakeLists into its binary dir. We need to install it alongside our .so
# so that SpruxFFISolver can find it at runtime.
#
# Sprux compiles SPRUX_METAL_LIBRARY_PATH into the binary as a string literal.
# Since that path points to the build dir (which won't exist after install),
# we override it to point to a path relative to the installed .so.
# At runtime, the metallib will be in the same directory as sprux_jax_cpp.so.
set(SPRUX_METALLIB "${sprux_BINARY_DIR}/sprux/sprux/MetalKernels.metallib")

# Override the compile definition on Sprux_static to use an install-relative path.
# We use $ORIGIN-style trick: the metallib is installed next to the .so.
# Since SPRUX_METAL_LIBRARY_PATH is a compile-time string, we need to find
# the metallib at runtime instead. We'll use a wrapper approach:
# install the metallib and set an environment variable or use dladdr to find it.
#
# Simplest approach: install the metallib and have sprux_jax.py set the path
# via an environment variable before the first FFI call.

# Install metallib alongside the extension
install(FILES "${SPRUX_METALLIB}" DESTINATION .)

# Install target
install(TARGETS sprux_jax_cpp LIBRARY DESTINATION .)
