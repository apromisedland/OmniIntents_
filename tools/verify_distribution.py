"""Verify both built distributions in a fresh environment outside the checkout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path


def run(arguments: list[str], cwd: Path) -> None:
    subprocess.run(arguments, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", default="dist")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    distribution = (repository / args.dist).resolve()
    wheels = list(distribution.glob("omniintents-0.3.0-*.whl"))
    sources = list(distribution.glob("omniintents-0.3.0.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise RuntimeError("Build exactly one 0.3.0 wheel and sdist first")
    required = {"omniintents/data/smoke.jsonl", "omniintents/data/prompts/intent_examples.json",
                "omniintents/data/prompts/task_examples.json", "omniintents/data/prompts/agent_examples.json",
                "omniintents/data/prompts/definitions.json", "omniintents/data/configs/default.json"}
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
        if not required <= names or any("__pycache__" in name or name.endswith(".pyc") for name in names):
            raise RuntimeError("Wheel resources are missing or contain bytecode")
    with tarfile.open(sources[0]) as archive:
        names = {name.partition("/")[2] for name in archive.getnames()}
        if not required <= names or not {"requirements.lock", "tests/conftest.py", "tools/verify_distribution.py"} <= names:
            raise RuntimeError("Source distribution is incomplete")
        if any(name.startswith((".venv", "private/", "runs/")) for name in names):
            raise RuntimeError("Source distribution contains local artifacts")
    with tempfile.TemporaryDirectory(prefix="omniintents-delivery-") as temporary:
        root = Path(temporary)
        environment = root / "consumer"
        venv.EnvBuilder(with_pip=True).create(environment)
        interpreter = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        command = environment / ("Scripts/omniintents.exe" if os.name == "nt" else "bin/omniintents")
        run([str(interpreter), "-m", "pip", "install", "--timeout", "120", "--constraint",
             str(repository / "requirements.lock"), str(wheels[0])], root)
        run([str(interpreter), "-I", "-c",
             "import sys, omniintents; from pathlib import Path; "
             "assert Path(omniintents.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()); "
             "assert 'torch' not in sys.modules and 'tensorflow' not in sys.modules"], root)
        run([str(command), "--version"], root)
        run([str(interpreter), "-I", "-m", "omniintents", "validate-data"], root)
        run([str(interpreter), "-I", "-m", "omniintents", "reproduce", "--suite", "smoke", "--out", str(root / "smoke")], root)
        summary = json.loads((root / "smoke" / "summary.json").read_text(encoding="utf-8"))
        if any(item["errors"] for item in summary["runs"]):
            raise RuntimeError("Installed wheel smoke suite has errors")
        run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
             "--wheel-dir", str(root / "rebuilt"), str(sources[0])], root)
        rebuilt = next((root / "rebuilt").glob("*.whl"))
        with zipfile.ZipFile(wheels[0]) as original, zipfile.ZipFile(rebuilt) as rebuilt_archive:
            for name in required:
                if original.read(name) != rebuilt_archive.read(name):
                    raise RuntimeError("Source-rebuilt wheel resources differ")
        print("Wheel installation, console command, offline reproduction, and sdist rebuild verified.")


if __name__ == "__main__":
    main()
