"""Exercise actual launch/selection code without starting GPUs or installing packages."""
import ast
import __future__
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('strategy', ['fsdp', 'model_parallel', 'single'])
@pytest.mark.parametrize('wrapped', [False, True])
def test_checkpoint_loader_accepts_training_state(tmp_path, strategy, wrapped):
    import torch

    tree = ast.parse((ROOT / 'litgpt/utils.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'load_checkpoint')
    fsdp, model_parallel = type('FSDPStrategy', (), {}), type('ModelParallelStrategy', (), {})
    source = torch.nn.Linear(2, 1)
    target = torch.nn.Linear(2, 1)
    weights = source.state_dict()
    checkpoint = tmp_path / 'lit_model.pth'
    torch.save(dict(model=weights, optimizer={}, global_step=12, data_epoch=1) if wrapped else weights, checkpoint)

    def load(path, state, strict):
        state['model'].load_state_dict(torch.load(path)['model'], strict=strict)

    def load_raw(path, model, strict):
        model.load_state_dict(torch.load(path), strict=strict)

    fabric = SimpleNamespace(strategy={'fsdp': fsdp(), 'model_parallel': model_parallel(), 'single': None}[strategy],
                             device='cpu', load=Mock(side_effect=load), load_raw=Mock(side_effect=load_raw))
    scope = dict(FSDPStrategy=fsdp, ModelParallelStrategy=model_parallel, torch=torch, lazy_load=torch.load,
                 load_from_full_model_state_dict=lambda **kw: kw['model'].load_state_dict(kw['full_sd'], strict=kw['strict']))
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'litgpt/utils.py', 'exec',
                 flags=__future__.annotations.compiler_flag), scope)
    scope['load_checkpoint'](fabric, target, checkpoint)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, weights[name])
    if strategy == 'fsdp':
        assert fabric.load.call_count == int(wrapped)
        assert fabric.load_raw.call_count == int(not wrapped)


def test_semantic_fast_enables_auto_resume():
    config = yaml.safe_load((ROOT / 'exp/qwen1.7b-32k/arc_semantic_fast.yaml').read_text())
    assert config['auto_resume'] is True


@pytest.mark.parametrize('status,artifact,expected', [(17, False, 17), (17, True, 17), (0, False, 1), (0, True, 0)])
def test_majob_preserves_training_failure_before_output_check(tmp_path, status, artifact, expected):
    source = (ROOT / 'majob.sh').read_text()
    stage = source[source.index('if checkpoint_exists "${SAVE_DIR}/lit_model.pth" && checkpoint_finished; then'):
                   source.index('BARRIER_DIR=')]
    shell = f'TEST_TRAIN_STATUS={status}\n' + '''
CONFIG_FILE="config with spaces.yaml"
SAVE_DIR=$1
NODE_RANK=0
NUM_NODES=1
GPUS_PER_NODE=8
MASTER_ADDR=localhost
TRAIN_MASTER_PORT=6000
SAVE_CKPT=true
ENABLE_TENSORBOARD=true
PYTHON_BIN=checked_python
checkpoint_exists() { [ -s "$1" ]; }
checkpoint_finished() { return 1; }
checkpoint_step_label() { echo pending; }
ensure_checkpoint_tokenizer() { return 0; }
check_tensorboard() { return 0; }
sleep() { :; }
checked_python() {
    printf '<%s>\\n' "$@"
    return "$TEST_TRAIN_STATUS"
}
'''
    if artifact:
        (tmp_path / 'lit_model.pth').write_bytes(b'weights')
    result = subprocess.run(['bash', '-c', shell + stage + '\necho ENTERED_EVAL', 'bash', str(tmp_path)],
                            text=True, capture_output=True)
    assert result.returncode == expected, result.stdout + result.stderr
    assert '<-m>\n<torch.distributed.run>' in result.stdout
    assert '<config with spaces.yaml>' in result.stdout
    assert ('ENTERED_EVAL' in result.stdout) == (expected == 0)
    if status:
        assert '训练失败' in result.stdout and '训练进程返回成功' not in result.stdout


def test_tensorboard_disabled_does_not_construct_logger():
    source = (ROOT / 'demo.py').read_text()
    block = textwrap.dedent(source[source.index('    print(f"Training Python:'):source.index('    # 2. Fabric setup')])
    logger = Mock(side_effect=ModuleNotFoundError('neither tensorboard nor tensorboardx is available'))
    scope = dict(sys=sys, TensorBoardLogger=logger, enable_tensorboard=False, tensorboard_root='/tmp/unused',
                 expid='test', arch_name='Qwen/test', print=lambda *args: None)
    exec(block, scope)
    assert scope['loggers'] == []
    logger.assert_not_called()
    scope['enable_tensorboard'] = True
    with pytest.raises(ModuleNotFoundError, match=re.escape(sys.executable)):
        exec(block, scope)


@pytest.mark.parametrize('auto,resume,output,selected', [
    (False, False, False, 'base'), (False, False, True, 'base'),
    (True, False, False, 'base'), (True, False, True, 'output'),
    (True, True, True, 'resume'), (False, True, False, 'resume'),
])
def test_training_checkpoint_source_is_not_implicit_save_path(tmp_path, auto, resume, output, selected):
    source = (ROOT / 'demo.py').read_text()
    tree = ast.parse(source)
    helpers = {'_normal_path', '_checkpoint_exists', '_read_checkpoint_metadata', '_checkpoint_is_ready',
               '_checkpoint_step', '_checkpoint_sort_key', '_latest_training_checkpoint'}
    module = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in helpers],
                        type_ignores=[])
    paths = {name: tmp_path / name for name in ('base', 'resume', 'output')}
    for name, path in paths.items():
        path.mkdir()
        if name != 'output' or output:
            (path / 'lit_model.pth').write_bytes(b'weights')
    scope = dict(Path=Path, os=os, yaml=yaml, re=re, _CHECKPOINT_META_FILENAME='checkpoint_meta.yaml',
                 _STEP_CHECKPOINT_RE=re.compile(r'^step[_-](\d+)(?:_v\d+)?$'),
                 arch_name='Qwen/Qwen3-1.7B-Base', ckpt_dir=str(paths['base']),
                 resume_dir=str(paths['resume']) if resume else None, auto_resume=auto,
                 save_path=str(paths['output']), fabric=SimpleNamespace(print=lambda *args: None))
    exec(compile(module, 'demo.py', 'exec', flags=__future__.annotations.compiler_flag), scope)
    start = source.index('    checkpoint_dir = f"checkpoints/{arch_name}"')
    end = source.index('    with fabric.init_module(empty_init=True):', start)
    block = textwrap.dedent(source[start:end])
    exec(block, scope)
    assert scope['selected_ckpt_path'] == paths[selected] / 'lit_model.pth'
    # An explicit missing resume source is an input error; never use save_path
    # or silently initialize from base in its place.
    scope['resume_dir'] = str(tmp_path / 'missing-resume')
    with pytest.raises(FileNotFoundError, match='Training input checkpoint'):
        exec(block, scope)


@pytest.mark.parametrize('enabled', ['true', 'false'])
def test_tensorboard_install_failure_stops_before_training(enabled):
    source = (ROOT / 'majob.sh').read_text()
    start = source.index('    if [ "${ENABLE_TENSORBOARD}" == "true" ]; then')
    end = source.index('    "${PYTHON_BIN}" -m torch.distributed.run', start)
    shell = f'ENABLE_TENSORBOARD={enabled}\n' + '''
PYTHON_BIN=install_test
check_tensorboard() { return 1; }
install_test() { echo INSTALL_ATTEMPT; return 1; }
''' + source[start:end] + '\necho READY_FOR_TRAINING'
    result = subprocess.run(['bash', '-c', shell], capture_output=True, text=True)
    assert result.returncode == (1 if enabled == 'true' else 0)
    assert ('READY_FOR_TRAINING' in result.stdout) == (enabled == 'false')
    assert ('INSTALL_ATTEMPT' in result.stdout) == (enabled == 'true')
