"""Exercise the scripts' real config readers and eval-argument builders only."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("script", ["majob.sh", "eval.sh"])
@pytest.mark.parametrize("route", ["legacy_route", "unified_route"])
@pytest.mark.parametrize("anchor_mode,pack_backend", [("mid", "triton"), ("multi", "torch")])
@pytest.mark.parametrize("setting, expected", [("true", "true"), ("false", "false"), ("null", "false"), (None, "false")])
def test_semantic_route_reaches_eval_arguments(tmp_path, script, route, setting, expected, anchor_mode, pack_backend):
    parent = tmp_path / "base.yaml"
    parent.write_text("save_path: /tmp/unused-checkpoint\nlog_kv_semantic_clusters: true\n"
                      "log_kv_cluster_k_max: 8\nlog_kv_B: 64\n"
                      "log_kv_alpha_exact_tokens: 256\nlog_kv_alpha_span_max_tokens: 64\n"
                      f"log_kv_semantic_merge_passes: {4 if anchor_mode == 'mid' else 'null'}\n"
                      f"log_kv_semantic_centroid_backend: {'parallel' if anchor_mode == 'mid' else 'null'}\n"
                      f"log_kv_semantic_replay_updates: {'true' if anchor_mode == 'mid' else 'null'}\n"
                      f"log_kv_semantic_anchor_mode: {anchor_mode}\nlog_kv_semantic_pack_backend: {pack_backend}\n")
    if setting is not None:
        parent.write_text(parent.read_text() + f"log_kv_semantic_{route}: {setting}\n")
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
    assert "--log_kv_semantic_summary_size" not in args
    for name in ("legacy_route", "unified_route"):
        flag = "--log_kv_semantic_" + name
        assert args.count(flag) == 1
        assert args[args.index(flag) + 1] == (expected if name == route else "false")
    # Also catch shifts in majob's positional read list after adding a field.
    assert args[args.index("--log_kv_cluster_k_max") + 1] == "8"
    assert args[args.index("--log_kv_B") + 1] == "64"
    assert args[args.index("--log_kv_semantic_merge_passes") + 1] == ("4" if anchor_mode == "mid" else "1")
    assert args[args.index("--log_kv_alpha_exact_tokens") + 1] == "256"
    assert args[args.index("--log_kv_alpha_span_max_tokens") + 1] == "64"
    assert args[args.index("--log_kv_semantic_anchor_mode") + 1] == anchor_mode
    assert args[args.index("--log_kv_semantic_pack_backend") + 1] == pack_backend

    values = ("parallel", "true") if anchor_mode == "mid" else ("sequential", "false")
    for name, value in zip(("centroid_backend", "replay_updates"), values):
        assert args[args.index("--log_kv_semantic_" + name) + 1] == value


@pytest.mark.parametrize("script", ["majob.sh", "eval.sh"])
def test_removed_summary_config_is_rejected_before_launch(tmp_path, script):
    (tmp_path / "base.yaml").write_text(
        "save_path: /tmp/unused-checkpoint\nlog_kv_semantic_summary_size: 8\n"
    )
    config = tmp_path / "run.yaml"
    config.write_text("config: base.yaml\n")
    source = (Path(__file__).resolve().parents[1] / script).read_text()
    marker = 'RAW_SAVE_DIR=$(python ' if script == "majob.sh" else 'CONFIG_EXPORTS=$("${PYTHON_BIN}" '
    start = source.index(marker)
    stop = source.index('\n)', start) + len('\n)')
    shell = '\n'.join(('set -e', 'CONFIG_FILE=$1', source[start:stop]))
    env = {**os.environ, "PYTHON_BIN": sys.executable,
           "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")}
    result = subprocess.run(["bash", "-c", shell, "bash", str(config)],
                            capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert "log_kv_semantic_summary_size has been removed" in result.stderr
