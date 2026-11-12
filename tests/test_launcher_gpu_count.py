"""Exercise launcher GPU detection without starting training or requiring CUDA."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", ["majob.sh", "eval.sh"])
@pytest.mark.parametrize("visible,overrides,expected", [
    (1, {}, 1), (2, {}, 2), (4, {}, 4), (8, {}, 8),
    (2, {"LOCAL_WORLD_SIZE": "8"}, 2),
    (4, {"MA_NUM_GPUS": "8"}, 4),
    (8, {"MA_NUM_GPUS": "2"}, 2),
    (4, {"GPUS_PER_NODE": "2"}, 2),
    (0, {}, None),
    (2, {"GPUS_PER_NODE": "4"}, None),
    (8, {"MA_NUM_GPUS": "2", "GPUS_PER_NODE": "4"}, None),
    (4, {"GPUS_PER_NODE": "0"}, None),
    (4, {"MA_NUM_GPUS": "0"}, None),
    (4, {"GPUS_PER_NODE": "invalid"}, None),
])
def test_launcher_selects_visible_gpus(tmp_path, script, visible, overrides, expected):
    source = (ROOT / script).read_text()
    start = source.index('GPUS_PER_NODE=$("${PYTHON_BIN}" ')
    stop = source.index('\nNUM_NODES=', start)
    # Replace only the CUDA query, keeping the real launcher selection and errors.
    (tmp_path / "torch.py").write_text(
        "import os\nfrom types import SimpleNamespace\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-test-mask'\n"
        "cuda = SimpleNamespace(device_count=lambda: int(os.environ['TEST_VISIBLE_GPUS']))\n"
    )
    env = {key: value for key, value in os.environ.items()
           if key not in ("GPUS_PER_NODE", "MA_NUM_GPUS", "LOCAL_WORLD_SIZE")}
    env.update(PYTHON_BIN=sys.executable, PYTHONPATH=str(tmp_path),
               TEST_VISIBLE_GPUS=str(visible), CUDA_VISIBLE_DEVICES="GPU-test-mask", **overrides)
    shell = source[start:stop] + '\nprintf "selected=%s\\n" "$GPUS_PER_NODE"'
    result = subprocess.run(["bash", "-c", shell], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    if expected is None:
        assert result.returncode != 0
        assert "ERROR:" in result.stderr
        assert "selected=" not in result.stdout
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == f"selected={expected}"
        assert f"visible={visible}" in result.stderr
        assert f"using={expected}" in result.stderr
