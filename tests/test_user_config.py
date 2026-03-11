"""Tests for vajax.user_config — config loading, env var override, path merging."""

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

from vajax.user_config import (
    _load_toml,
    _merge_configs,
    _user_config_path,
    get_config,
    get_model_paths,
    reset_config,
)


@pytest.fixture(autouse=True)
def _reset():
    """Reset cached config before each test."""
    reset_config()
    yield
    reset_config()


class TestUserConfigPath:
    def test_xdg_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("XDG_CONFIG_HOME", None)
            if sys.platform != "win32":
                assert _user_config_path() == Path.home() / ".config" / "vajax" / "config.toml"

    def test_xdg_custom(self):
        if sys.platform == "win32":
            pytest.skip("XDG not used on Windows")
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": "/tmp/claude/xdg_test"}):
            assert _user_config_path() == Path("/tmp/claude/xdg_test/vajax/config.toml")

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
    def test_appdata_path(self):
        with mock.patch.dict(os.environ, {"APPDATA": "C:\\Users\\Test\\AppData\\Roaming"}):
            p = _user_config_path()
            assert p == Path("C:\\Users\\Test\\AppData\\Roaming\\vajax\\config.toml")


class TestLoadToml:
    def test_missing_file(self):
        assert _load_toml(Path("/nonexistent/config.toml")) == {}

    def test_valid_toml(self, tmp_path):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('[models]\npaths = ["/foo", "/bar"]\n')
        result = _load_toml(cfg_file)
        assert result == {"models": {"paths": ["/foo", "/bar"]}}

    def test_invalid_toml(self, tmp_path):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("not valid toml {{{\n")
        assert _load_toml(cfg_file) == {}


class TestMergeConfigs:
    def test_override_wins(self):
        base = {"models": {"paths": ["/base"]}}
        override = {"models": {"paths": ["/override"]}}
        assert _merge_configs(base, override) == {"models": {"paths": ["/override"]}}

    def test_disjoint_keys(self):
        base = {"models": {"paths": ["/a"]}}
        override = {"solver": {"backend": "dense"}}
        merged = _merge_configs(base, override)
        assert "models" in merged
        assert "solver" in merged


class TestGetConfig:
    def test_caches_result(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_force_reload(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg1 = get_config()
        cfg2 = get_config(_force_reload=True)
        assert cfg1 is not cfg2

    def test_project_overrides_user(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        # Create project config in cwd
        project_cfg = tmp_path / "vajax.toml"
        project_cfg.write_text('[models]\npaths = ["/project"]\n')

        # Mock user config to return a different path
        user_cfg_dir = tmp_path / "user_config"
        user_cfg_dir.mkdir()
        user_cfg_file = user_cfg_dir / "config.toml"
        user_cfg_file.write_text('[models]\npaths = ["/user"]\n')

        with mock.patch("vajax.user_config._user_config_path", return_value=user_cfg_file):
            cfg = get_config(_force_reload=True)

        # Project config should override user config
        assert cfg["models"]["paths"] == ["/project"]


class TestGetModelPaths:
    def test_empty_when_no_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("VAJAX_MODEL_PATH", raising=False)
        with mock.patch("vajax.user_config._user_config_path", return_value=tmp_path / "nope.toml"):
            paths = get_model_paths()
        assert paths == []

    def test_env_var_paths(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        dir_a = tmp_path / "models_a"
        dir_b = tmp_path / "models_b"
        dir_a.mkdir()
        dir_b.mkdir()

        sep = ";" if sys.platform == "win32" else ":"
        monkeypatch.setenv("VAJAX_MODEL_PATH", f"{dir_a}{sep}{dir_b}")

        with mock.patch("vajax.user_config._user_config_path", return_value=tmp_path / "nope.toml"):
            paths = get_model_paths()

        assert paths == [dir_a, dir_b]

    def test_env_var_skips_nonexistent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        dir_a = tmp_path / "exists"
        dir_a.mkdir()

        sep = ";" if sys.platform == "win32" else ":"
        monkeypatch.setenv("VAJAX_MODEL_PATH", f"{dir_a}{sep}/nonexistent/path")

        with mock.patch("vajax.user_config._user_config_path", return_value=tmp_path / "nope.toml"):
            paths = get_model_paths()

        assert paths == [dir_a]

    def test_config_file_paths(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("VAJAX_MODEL_PATH", raising=False)

        dir_a = tmp_path / "cfg_models"
        dir_a.mkdir()

        project_cfg = tmp_path / "vajax.toml"
        project_cfg.write_text(f'[models]\npaths = ["{dir_a}"]\n')

        with mock.patch("vajax.user_config._user_config_path", return_value=tmp_path / "nope.toml"):
            paths = get_model_paths()

        assert paths == [dir_a]

    def test_env_var_before_config(self, tmp_path, monkeypatch):
        """VAJAX_MODEL_PATH entries come before config file entries."""
        monkeypatch.chdir(tmp_path)

        env_dir = tmp_path / "env_models"
        cfg_dir = tmp_path / "cfg_models"
        env_dir.mkdir()
        cfg_dir.mkdir()

        monkeypatch.setenv("VAJAX_MODEL_PATH", str(env_dir))

        project_cfg = tmp_path / "vajax.toml"
        project_cfg.write_text(f'[models]\npaths = ["{cfg_dir}"]\n')

        with mock.patch("vajax.user_config._user_config_path", return_value=tmp_path / "nope.toml"):
            paths = get_model_paths()

        assert paths == [env_dir, cfg_dir]

    def test_tilde_expansion(self, tmp_path, monkeypatch):
        """Paths with ~ should be expanded."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("VAJAX_MODEL_PATH", raising=False)

        # Use home dir which definitely exists
        project_cfg = tmp_path / "vajax.toml"
        project_cfg.write_text('[models]\npaths = ["~"]\n')

        with mock.patch("vajax.user_config._user_config_path", return_value=tmp_path / "nope.toml"):
            paths = get_model_paths()

        assert len(paths) == 1
        assert paths[0] == Path.home()


class TestGetBasePathsIntegration:
    """Test that _get_base_paths() picks up user config."""

    def test_user_paths_inserted(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("VAJAX_MODEL_PATH", raising=False)

        user_dir = tmp_path / "my_models"
        user_dir.mkdir()

        monkeypatch.setenv("VAJAX_MODEL_PATH", str(user_dir))

        with mock.patch("vajax.user_config._user_config_path", return_value=tmp_path / "nope.toml"):
            from vajax.analysis.openvaf_models import _get_base_paths

            paths = _get_base_paths()

        assert "user_0" in paths
        assert paths["user_0"] == user_dir
        # Bundled should come first
        keys = list(paths.keys())
        assert keys.index("bundled") < keys.index("user_0")
        # Vendor should come after user
        assert keys.index("user_0") < keys.index("integration_tests")


class TestIncludeResolution:
    """Test that include statements search user-configured paths."""

    def test_include_relative_to_netlist(self, tmp_path):
        """Include found relative to netlist dir — no search path needed."""
        from vajax.netlist.parser import VACASKParser

        sub = tmp_path / "sub.sim"
        sub.write_text("* subcircuit file\n")

        result = VACASKParser._resolve_include("sub.sim", tmp_path)
        assert result == sub

    def test_include_via_search_path(self, tmp_path, monkeypatch):
        """Include not in netlist dir but found in VAJAX_MODEL_PATH."""
        from vajax.netlist.parser import VACASKParser

        # Create include file in a separate directory
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        inc_file = lib_dir / "common.sim"
        inc_file.write_text("* common definitions\n")

        # Netlist dir does NOT have the file
        netlist_dir = tmp_path / "project"
        netlist_dir.mkdir()

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("VAJAX_MODEL_PATH", str(lib_dir))

        with mock.patch("vajax.user_config._user_config_path", return_value=tmp_path / "nope.toml"):
            result = VACASKParser._resolve_include("common.sim", netlist_dir)

        assert result is not None
        assert result == inc_file

    def test_include_not_found(self, tmp_path, monkeypatch):
        """Include not found anywhere returns None."""
        from vajax.netlist.parser import VACASKParser

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("VAJAX_MODEL_PATH", raising=False)

        with mock.patch("vajax.user_config._user_config_path", return_value=tmp_path / "nope.toml"):
            result = VACASKParser._resolve_include("nonexistent.sim", tmp_path)

        assert result is None
