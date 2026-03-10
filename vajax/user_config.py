"""User configuration for VAJAX.

Loads config from (in priority order):
1. Environment variables (VAJAX_MODEL_PATH, etc.)
2. Project config: vajax.toml in cwd
3. User config: $XDG_CONFIG_HOME/vajax/config.toml

Higher-priority sources override lower-priority ones.
"""

import logging
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_cached_config: dict[str, Any] | None = None


def _user_config_path() -> Path:
    """Return the user config file path following XDG on Unix, %APPDATA% on Windows."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "vajax" / "config.toml"


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file, returning empty dict if missing or invalid."""
    if not path.is_file():
        return {}
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception as exc:
        log.warning("Failed to load config %s: %s", path, exc)
        return {}


def _merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge override into base (override wins for top-level keys).

    For nested dicts (like [models]), the override replaces the entire section.
    """
    merged = dict(base)
    merged.update(override)
    return merged


def get_config(*, _force_reload: bool = False) -> dict[str, Any]:
    """Load and merge config (cached after first call).

    Resolution order (later overrides earlier):
    1. User config (~/.config/vajax/config.toml)
    2. Project config (./vajax.toml)

    Environment variables are NOT merged here — they're handled
    by the specific accessor functions (e.g., get_model_paths()).
    """
    global _cached_config
    if _cached_config is not None and not _force_reload:
        return _cached_config

    user_cfg = _load_toml(_user_config_path())
    project_cfg = _load_toml(Path.cwd() / "vajax.toml")

    _cached_config = _merge_configs(user_cfg, project_cfg)
    return _cached_config


def get_model_paths() -> list[Path]:
    """Get user-configured model search paths.

    Sources (in order — earlier paths searched first):
    1. VAJAX_MODEL_PATH env var (colon-separated, semicolon on Windows)
    2. Config file [models].paths (project config overrides user config)

    Returns only directories that actually exist.
    """
    paths: list[Path] = []

    # 1. Environment variable
    env_val = os.environ.get("VAJAX_MODEL_PATH", "")
    if env_val:
        sep = ";" if sys.platform == "win32" else ":"
        for p in env_val.split(sep):
            p = p.strip()
            if p:
                path = Path(p).expanduser()
                if path.is_dir():
                    paths.append(path)
                else:
                    log.debug("VAJAX_MODEL_PATH entry does not exist: %s", path)

    # 2. Config file [models].paths
    cfg = get_config()
    cfg_paths = cfg.get("models", {}).get("paths", [])
    if isinstance(cfg_paths, list):
        for p in cfg_paths:
            if isinstance(p, str) and p:
                path = Path(p).expanduser()
                if path.is_dir():
                    paths.append(path)
                else:
                    log.debug("Config [models].paths entry does not exist: %s", path)

    return paths


def reset_config() -> None:
    """Clear the cached config. Useful for testing."""
    global _cached_config
    _cached_config = None
