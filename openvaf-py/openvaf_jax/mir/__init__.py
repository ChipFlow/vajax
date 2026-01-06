"""MIR data structures and analysis for OpenVAF to JAX translation.

This module provides:
- types: MIR data classes (Block, Instruction, MIRFunction)
- cfg: Control flow graph analysis (dominators, loops)
- ssa: SSA analysis (PHI resolution, branch conditions)
"""

from .types import (
    MIRValue,
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
    "MIRValue",
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
