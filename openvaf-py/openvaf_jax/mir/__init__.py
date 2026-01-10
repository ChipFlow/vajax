"""MIR data structures and analysis for OpenVAF to JAX translation.

This module provides:
- types: MIR data classes (Block, Instruction, MIRFunction)
- cfg: Control flow graph analysis (dominators, loops)
- ssa: SSA analysis (PHI resolution, branch conditions)
"""

from .types import (
    BlockId,
    ValueId,
    # Pre-allocated MIR constants
    V_GRAVESTONE,
    V_FALSE,
    V_TRUE,
    V_F_ZERO,
    V_ZERO,
    V_ONE,
    V_F_ONE,
    V_F_N_ONE,
    # Data structures
    MIRInstruction,
    PhiOperand,
    Block,
    MIRFunction,
    parse_mir_function,
)
from .cfg import CFGAnalyzer, LoopInfo
from .ssa import SSAAnalyzer, BranchInfo, PHIResolution

__all__ = [
    # types
    "BlockId",
    "ValueId",
    # Pre-allocated MIR constants
    "V_GRAVESTONE",
    "V_FALSE",
    "V_TRUE",
    "V_F_ZERO",
    "V_ZERO",
    "V_ONE",
    "V_F_ONE",
    "V_F_N_ONE",
    # Data structures
    "MIRInstruction",
    "PhiOperand",
    "Block",
    "MIRFunction",
    "parse_mir_function",
    # cfg
    "CFGAnalyzer",
    "LoopInfo",
    # ssa
    "SSAAnalyzer",
    "BranchInfo",
    "PHIResolution",
]
