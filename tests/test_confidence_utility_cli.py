import json
import subprocess
import sys
from pathlib import Path

from src.confidence_training.cli_config import inject_json_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_json_argv_injection_preserves_explicit_cli(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "utility.json"
    config.write_text(
        json.dumps(
            {
                "runs_root": "from-json",
                "output": "selection.json",
                "sweep_seed": 7,
                "dry_run": True,
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["command", "--config", str(config), "--runs-root", "from-cli"],
    )
    inject_json_config()
    assert sys.argv.count("--runs-root") == 1
    assert "from-cli" in sys.argv
    assert ["--output", "selection.json"] == sys.argv[-5:-3]
    assert "--sweep-seed" in sys.argv
    assert "--dry-run" in sys.argv


def test_select_cli_accepts_json_only(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    config = tmp_path / "select.json"
    config.write_text(
        json.dumps(
            {
                "runs_root": str(runs),
                "output": str(tmp_path / "selection.json"),
                "sweep_seed": 42,
                "dry_run": True,
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/select_confidence_runs.py"),
            "--config",
            str(config),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["entries"] == []
    assert not (tmp_path / "selection.json").exists()

