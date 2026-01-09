"""Code generation for OpenVAF to JAX translation.

This module provides:
- context: Code generation context (variable tracking, constants)
- instruction: Single instruction translation
- function_builder: Complete function assembly (init/eval)
"""

from .context import CodeGenContext
from .instruction import InstructionTranslator
from .function_builder import InitFunctionBuilder, EvalFunctionBuilder

__all__ = [
    "CodeGenContext",
    "InstructionTranslator",
    "InitFunctionBuilder",
    "EvalFunctionBuilder",
]
