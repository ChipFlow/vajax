# OpenVAF Pre-Allocated MIR Constants

## Overview

OpenVAF pre-allocates common constants at fixed SSA value positions in every MIR function. These constants are NOT shown in the MIR text dump but are always present at runtime.

## Source

From `openvaf/mir/src/dfg/values.rs:57-76` in the OpenVAF repository.

## Pre-Allocated Constants

| Position | Name | Type | Value | Python Equivalent |
|----------|------|------|-------|-------------------|
| v0 | GRAVESTONE | - | - | `None` (dead value placeholder) |
| v1 | FALSE | bool | false | `False` |
| v2 | TRUE | bool | true | `True` |
| v3 | F_ZERO | f64 | 0.0 | `0.0` |
| v4 | ZERO | i32 | 0 | `0` |
| v5 | ONE | i32 | 1 | `1` |
| v6 | F_ONE | f64 | 1.0 | `1.0` |
| v7 | F_N_ONE | f64 | -1.0 | `-1.0` |

## Implications for MIR Parsing

When parsing OpenVAF MIR dumps:

1. **v1-v7 may appear in instructions without being defined** - This is normal! They're pre-allocated.

2. **PHI nodes often use v1 (FALSE) or v2 (TRUE)** - Common for validation logic where branches merge with a "failed validation" path.

3. **Parser must pre-populate constants dict** - Before parsing any blocks, initialize:
   ```python
   constants = {
       'v0': None,
       'v1': False,
       'v2': True,
       'v3': 0.0,
       'v4': 0,
       'v5': 1,
       'v6': 1.0,
       'v7': -1.0,
   }
   ```

4. **User-defined constants start at v8+** - The MIR dump will show explicit declarations for constants starting from v8 onwards.

## Example: Parameter Validation PHI Node

From resistor `model_param_setup`:

```mir
block2:
    v23 = fle v3, v19        # Check: 0.0 <= r
    br v23, block7, block9   # If true, check upper bound; else fail

block7:
    v24 = fle v19, v15       # Check: r <= inf
    jmp block9

block9:
    v25 = phi [v1, block2], [v24, block7]
    br v25, block4, block10
```

Here:
- From **block2 → block9**: Use **v1 (FALSE)** - lower bound check failed
- From **block7 → block9**: Use **v24** - result of upper bound check

The PHI node merges:
- FALSE if r < 0 (invalid)
- (r <= inf) if r >= 0 (valid range check)

## Code Generation

When generating Python from MIR, include these constants in initialization:

```python
# Initialize OpenVAF pre-allocated constants
FALSE = False
TRUE = True
ZERO = 0.0
ONE_INT = 1
ONE = 1.0
# ... rest of codegen
```

This ensures that v1, v2, etc. resolve correctly in the generated code.

## Historical Note

This discovery solved a bug where `v25 = phi [v1, block2], [v24, block7]` appeared to reference an undefined `v1`. Initially we thought:
- v1 might be a typo for v23
- v1 was missing from the MIR dump
- The dump was incomplete

The actual answer: v1 is a pre-allocated constant that's always present but never shown in dumps.

## Reference

OpenVAF source: `openvaf-py/vendor/OpenVAF/openvaf/mir/src/dfg/values.rs:57-76`
