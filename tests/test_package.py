"""Light tests for the pip packaging layer (no GPU, no GUI, no heavy deps)."""
import sys

import pytest

import dino4dstem
from dino4dstem import cli, paths, shortcuts


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert dino4dstem.__version__ in capsys.readouterr().out


def test_parser_commands():
    p = cli.build_parser()
    assert p.parse_args([]).command is None          # bare call -> GUI
    assert p.parse_args(["assistant", "x.npy"]).cube == "x.npy"
    assert p.parse_args(["shortcuts", "--remove"]).remove


def test_app_files_found():
    root = paths.app_root()
    assert (root / "src" / "gui_dino4dstem.py").is_file()
    assert (root / "src" / "assistant_gui.py").is_file()
    assert (root / "tools" / "setup_gpu_torch.py").is_file()
    assert paths.icon("png") is not None and paths.icon("ico") is not None


def test_workspace_seeded_with_docs(tmp_path, monkeypatch):
    monkeypatch.setenv("DINO4DSTEM_HOME", str(tmp_path / "ws"))
    ws = paths.workspace()
    assert ws == tmp_path / "ws"
    assert (ws / "docs").is_dir()
    nb = paths.notebooks_dir()
    assert nb.parent == ws and any(nb.glob("*.ipynb"))


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux launchers")
def test_linux_shortcuts_roundtrip(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "Desktop").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
    monkeypatch.setenv("DINO4DSTEM_HOME", str(tmp_path / "ws"))

    made = shortcuts.create()
    assert len(made) == 6                              # 3 menu + 3 desktop
    gui = (home / "Desktop" / "DINO-4DSTEM-GUI.desktop").read_text()
    assert f'Exec="{sys.executable}" -m dino4dstem gui' in gui
    assert "Icon=" in gui

    removed = shortcuts.remove()
    assert len(removed) == 6
    assert not any((home / "Desktop").iterdir())
