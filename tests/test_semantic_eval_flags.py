"""Exercise the scripts' real config readers and eval-argument builders only."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("script", ["majob.sh", "eval.sh"])
@pytest.mark.parametrize("anchor_mode,pack_backend", [("mid", "triton"), ("multi", "torch")])
@pytest.mark.parametrize("setting, expected", [("true", "true"), ("false", "false"), ("null", "false"), (None, "false")])
def test_legacy_route_reaches_eval_arguments(tmp_path, script, setting, expected, anchor_mode, pack_backend):
    parent = tmp_path / "base.yaml"
    parent.write_text("save_path: /tmp/unused-checkpoint\nlog_kv_semantic_clusters: true\n"
                      "log_kv_cluster_k_max: 8\nlog_kv_B: 64\n"
                      f"log_kv_semantic_centroid_backend: {'parallel' if anchor_mode == 'mid' else 'null'}\n"
                      f"log_kv_semantic_summary_size: {'8' if anchor_mode == 'mid' else 'null'}\n"
                      f"log_kv_semantic_replay_updates: {'true' if anchor_mode == 'mid' else 'null'}\n"
                      f"log_kv_semantic_anchor_mode: {anchor_mode}\nlog_kv_semantic_pack_backend: {pack_backend}\n")
    if setting is not None:
        parent.write_text(parent.read_text() + f"log_kv_semantic_legacy_route: {setting}\n")
    config = tmp_path / "run.yaml"
    config.write_text("config: base.yaml\n")
    source = (Path(__file__).resolve().parents[1] / script).read_text()
    # Execute the source's extraction/assembly blocks without environment setup,
    # dependency installation, training, filesystem barriers, or GPU launches.
    if script == "majob.sh":
        start = source.index("read -r LOG_KV_B ")
        stop = source.index('\n)"', start) + len('\n)"')
        read_config = source[start:stop]
        start = source.index('LOG_KV_ARGS="')
        stop = source.index('\nDIAG_ARGS=', start)
        build_args = source[start:stop]
        print_args = "printf '%s\\n' ${EVAL_LOG_KV_ARGS}"
    else:
        start = source.index('CONFIG_EXPORTS=$("${PYTHON_BIN}" ')
        stop = source.index('eval "${CONFIG_EXPORTS}"', start) + len('eval "${CONFIG_EXPORTS}"')
        read_config = source[start:stop]
        start = source.index('LOG_KV_ARG_LIST=(')
        stop = source.index('\nDIAG_ARGS=', start)
        build_args = source[start:stop]
        print_args = "printf '%s\\n' \"${LOG_KV_ARG_LIST[@]}\""
    shell = '\n'.join(('set -e', 'CONFIG_FILE=$1', read_config, build_args, print_args))
    env = {**os.environ, "PYTHON_BIN": sys.executable,
           "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")}
    result = subprocess.run(["bash", "-c", shell, "bash", str(config)],
                            capture_output=True, text=True, check=True, env=env)
    args = result.stdout.splitlines()
    assert args.count("--log_kv_semantic_legacy_route") == 1
    assert args[args.index("--log_kv_semantic_legacy_route") + 1] == expected
    # Also catch shifts in majob's positional read list after adding a field.
    assert args[args.index("--log_kv_cluster_k_max") + 1] == "8"
    assert args[args.index("--log_kv_B") + 1] == "64"
    assert args[args.index("--log_kv_semantic_anchor_mode") + 1] == anchor_mode
    assert args[args.index("--log_kv_semantic_pack_backend") + 1] == pack_backend

    values = ("parallel", "8", "true") if anchor_mode == "mid" else ("sequential", "1", "false")
    for name, value in zip(("centroid_backend", "summary_size", "replay_updates"), values):
        assert args[args.index("--log_kv_semantic_" + name) + 1] == value
