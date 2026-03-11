# Configuration

VAJAX supports user-level and project-level configuration via TOML files, plus environment variable overrides.

## Config Files

VAJAX loads configuration from two locations (later overrides earlier):

| Source | Path | Use case |
|--------|------|----------|
| User config | `~/.config/vajax/config.toml` | Personal model paths, global preferences |
| Project config | `vajax.toml` in working directory | Per-project model paths |

On Linux/macOS, the user config follows the [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/latest/) — set `$XDG_CONFIG_HOME` to override the default `~/.config` location.

On Windows, the user config is at `%APPDATA%\vajax\config.toml`.

## Config Schema

### `[models]` — Search Paths

```toml
# ~/.config/vajax/config.toml

[models]
# Additional directories to search for .va model files and .sim includes.
# Each path is a directory that may contain .va files directly
# or in subdirectories matching the model name.
paths = [
    "/path/to/my/models",
    "/opt/vacask/devices",
    "~/pdk/models",        # Tilde expansion is supported
]
```

These paths are used for both `load` statements (Verilog-A `.va` models) and `include` statements (netlist `.sim` files). Paths that don't exist are silently skipped (logged at DEBUG level).

## Environment Variables

### Model paths

`VAJAX_MODEL_PATH` — colon-separated (`;` on Windows) list of directories to search for `.va` model files. These are searched **before** config file paths.

```bash
# Single directory
export VAJAX_MODEL_PATH=/path/to/my/models

# Multiple directories
export VAJAX_MODEL_PATH=/path/to/models1:/path/to/models2

# Windows (PowerShell)
$env:VAJAX_MODEL_PATH = "C:\models1;C:\models2"
```

### Other environment variables

| Variable | Description |
|----------|-------------|
| `VAJAX_CACHE_DIR` | Override cache directory (default: `~/.cache/vajax/`) |
| `VAJAX_NO_PROGRESS` | Set to `1` to disable progress bars |
| `JAX_PLATFORMS` | Force JAX platform: `cpu`, `cuda`, `gpu` |
| `JAX_ENABLE_X64` | Enable float64: `1` or `0` |

## Search Order

Both `load` (VA models) and `include` (netlist files) use the same search order:

1. **Netlist directory** — relative to the file containing the `load`/`include` statement
2. **`VAJAX_MODEL_PATH`** — environment variable directories
3. **Config file `[models].paths`** — project config first, then user config
4. **Bundled models** — `vajax/devices/models/` (resistor, capacitor, etc.)
5. **Vendor directories** — `vendor/OpenVAF/integration_tests`, `vendor/VACASK/devices`

The first match wins. This means you can override bundled models by placing your version in an earlier search path.

## Examples

### Point to a local PDK

```toml
# vajax.toml (in project directory)
[models]
paths = ["/opt/pdk/sky130/models"]
```

### Use VACASK models from a separate install

```toml
# ~/.config/vajax/config.toml
[models]
paths = ["/usr/local/share/vacask/devices"]
```

### Temporary override for a single run

```bash
VAJAX_MODEL_PATH=/tmp/test_models vajax circuit.sim
```

## Programmatic Access

```python
from vajax.user_config import get_config, get_model_paths

# Get merged config dict
config = get_config()

# Get resolved model search paths (existing directories only)
paths = get_model_paths()
```
