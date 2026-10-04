import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "directory,explicit_trt", [(None, False), ("nested/models", False), ("nested/models", True)]
)
def test_checkpoint_directory(tmp_path, directory, explicit_trt):
    env = os.environ.copy()
    env.pop("CHECKPOINT_DIR", None)
    env.pop("TRT_ENGINE_DIR", None)
    if directory is not None:
        env["CHECKPOINT_DIR"] = str(tmp_path / directory)
    if explicit_trt:
        env["TRT_ENGINE_DIR"] = str(tmp_path / "custom-engines")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, os; from flux.util import CHECKPOINTS_DIR; "
            "print(json.dumps([str(CHECKPOINTS_DIR), os.environ['TRT_ENGINE_DIR']]))",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    checkpoint_dir, trt_dir = json.loads(result.stdout.strip().splitlines()[-1])
    expected = Path(directory or "checkpoints")
    if directory is not None:
        expected = tmp_path / expected
    assert Path(checkpoint_dir) == expected
    assert (tmp_path / expected).is_dir()
    assert Path(trt_dir) == (tmp_path / "custom-engines" if explicit_trt else expected / "trt_engines")
