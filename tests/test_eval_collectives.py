"""CPU-only regression: python tests/test_eval_collectives.py.

For multiple nodes, use the usual torchrun rendezvous arguments with this file;
no model, dataset, GPU, or lm_eval import is needed. Each worker exits after 120s.
"""
from __future__ import annotations

import ast
from datetime import timedelta
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist


def _load_collectives():
    source = Path(__file__).resolve().parents[1] / "eval.py"
    tree = ast.parse(source.read_text())
    names = {"_env_int", "_global_rank", "_world_size", "_local_rank", "_rank_label",
             "_dist_ready", "_bcast_device", "_broadcast_obj", "_wait_all_ranks"}
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names
             or isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_SYNC_SEQ" for t in node.targets)]
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LogKVLM")
    nodes.append(next(node for node in model.body if isinstance(node, ast.FunctionDef)
                      and node.name == "all_gather_results"))
    scope = dict(torch=torch, dist=dist, time=time, timedelta=timedelta, Any=Any, os=os)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return scope


def _worker():
    def timed_out():
        print(f"rank {os.environ.get('RANK')} exceeded the 120s test deadline", file=sys.stderr, flush=True)
        os._exit(124)

    watchdog = threading.Timer(120, timed_out)
    watchdog.daemon = True
    watchdog.start()
    scope = _load_collectives()
    gather, broadcast = scope["all_gather_results"], scope["_broadcast_obj"]
    local = [(1.25, False)]
    assert gather(SimpleNamespace(process_group=None), local) is local
    assert broadcast(local) is local
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world >= 2, "run with at least two ranks"
    eval_group = None
    try:
        cases = [
            [(float(i) + 0.25, i % 2 == 0) for i in range(world * 2 + 1)],
            [f"sample-{i}-中文" for i in range(world - 1)],
            [float(i) + 0.5 for i in range(world * 2 + 1)],
            [],
            [(float(i) + 0.25, i % 2 == 0) for i in range(world * 9157 + min(4, world - 1))],
        ]
        eval_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=30))
        for group_name, group in (("default", None), ("eval", eval_group)):
            assert scope["_bcast_device"](group) is None
            payload = {"source": world - 1, "results": cases[0]}
            assert broadcast(payload if rank == world - 1 else None, src=world - 1, group=group) == payload
            for repeat in range(2):
                for case_id, expected in enumerate(cases):
                    if group is eval_group and repeat == 0 and case_id == 0 and rank == 0:
                        time.sleep(0.2)
                    actual = gather(SimpleNamespace(process_group=group), expected[rank::world],
                                    tag=f"{group_name}/{repeat}/{case_id}")
                    assert actual == expected, (rank, group_name, case_id, actual)
                    for got, want in zip(actual, expected):
                        assert type(got) is type(want)
                        if isinstance(want, tuple):
                            assert tuple(map(type, got)) == tuple(map(type, want))

        dist.destroy_process_group(eval_group)
        eval_group = None
        total = torch.tensor(rank + 1)
        dist.all_reduce(total)
        assert total.item() == world * (world + 1) // 2
        if rank == 0:
            print(f"PASS: {world} Gloo ranks; order/types/empty/repeated results, delayed rank, "
                  "broadcast, and default group survives eval cleanup", flush=True)
    finally:
        if eval_group is not None:
            dist.destroy_process_group(eval_group)
        dist.destroy_process_group()
        watchdog.cancel()


def test_eval_collectives():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
               "--master-addr=127.0.0.1", f"--master-port={port}",
               "--nproc-per-node=3", "--max-restarts=0", str(Path(__file__).resolve())]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, start_new_session=True) as process:
        try:
            output, _ = process.communicate(timeout=150)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            raise AssertionError(f"distributed regression exceeded 150s:\n{output}") from None
        assert process.returncode == 0, output
        print(next(line for line in output.splitlines() if line.startswith("PASS:")))


if __name__ == "__main__":
    if "RANK" in os.environ:
        _worker()
    else:
        test_eval_collectives()
