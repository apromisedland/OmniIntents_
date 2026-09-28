import json
import subprocess
import sys

import pytest

from omniintents.cli import main


def test_installed_cli_version():
    result = subprocess.run([sys.executable, "-m", "omniintents", "--version"],
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "omniintents 0.3.0"


def test_cli_demo_and_offline_reproduction(tmp_path, capsys):
    assert main(["demo", "--utterance", "Generate a new illustration"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["agent"]["agent_type"] == "generative_ai_agent"
    assert main(["reproduce", "--out", str(tmp_path / "run")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["runs"] == 3


@pytest.mark.parametrize("arguments", [
    ["demo", "--selection", "not-an-index"],
    ["demo", "--backend", "mock", "--speech-backend", "google"],
    ["validate-data", "--dataset", "file-that-does-not-exist.jsonl"],
])
def test_cli_errors_have_nonzero_status(arguments, capsys):
    assert main(arguments) == 2
    assert "error" in json.loads(capsys.readouterr().err)


def test_cli_trace_is_opt_in(tmp_path, capsys):
    path = tmp_path / "trace.json"
    assert main(["demo", "--utterance", "Save my report", "--trace-out", str(path)]) == 0
    capsys.readouterr()
    assert json.loads(path.read_text(encoding="utf-8"))["events"]


def test_import_without_optional_heavy_packages():
    command = (
        "import sys, omniintents; "
        "assert 'torch' not in sys.modules; assert 'tensorflow' not in sys.modules; "
        "assert 'google.cloud.speech' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)
