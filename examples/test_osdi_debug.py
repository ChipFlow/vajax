"""Debug OSDI loading step by step"""

import ctypes
from ctypes import c_uint32, c_void_p, POINTER, cast
from pathlib import Path

osdi_path = "/Users/roberttaylor/Code/ChipFlow/reference/jax-spice/pdks/gf130/diode_rr.osdi"

print("Step 1: Load library...")
lib = ctypes.CDLL(osdi_path)
print(f"  ✓ Loaded: {lib}")

print("\nStep 2: Read globals...")
num_desc = c_uint32.in_dll(lib, "OSDI_NUM_DESCRIPTORS").value
print(f"  OSDI_NUM_DESCRIPTORS = {num_desc}")

ver_major = c_uint32.in_dll(lib, "OSDI_VERSION_MAJOR").value
ver_minor = c_uint32.in_dll(lib, "OSDI_VERSION_MINOR").value
print(f"  OSDI_VERSION = {ver_major}.{ver_minor}")

desc_size = c_uint32.in_dll(lib, "OSDI_DESCRIPTOR_SIZE").value
print(f"  OSDI_DESCRIPTOR_SIZE = {desc_size}")

print("\nStep 3: Get descriptor pointer...")
desc_ptr = c_void_p.in_dll(lib, "OSDI_DESCRIPTORS")
print(f"  OSDI_DESCRIPTORS = {hex(desc_ptr.value)}")

print("\nStep 4: Read descriptor name field...")
# The first field of OsdiDescriptor is `name` (char*)
# Let's read it directly
name_ptr_ptr = cast(desc_ptr.value, POINTER(ctypes.c_char_p))
name = name_ptr_ptr[0]
print(f"  Model name: {name.decode() if name else 'NULL'}")

print("\nStep 5: Read num_nodes and num_terminals...")
# Offsets depend on architecture - let's read the raw structure
# name (8 bytes on 64-bit) + num_nodes (4) + num_terminals (4)
base = desc_ptr.value
num_nodes_ptr = cast(base + 8, POINTER(c_uint32))
num_terminals_ptr = cast(base + 12, POINTER(c_uint32))
print(f"  num_nodes = {num_nodes_ptr[0]}")
print(f"  num_terminals = {num_terminals_ptr[0]}")

print("\n✓ Basic OSDI loading works!")
