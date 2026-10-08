"""CPU/gloo check of eval.py's heartbeat rank sync; no model/GPU required."""
import ast
import contextlib
import io
import os
import socket
import time
from pathlib import Path

import torch.distributed as dist
import torch.multiprocessing as mp

_NAMES = {"_env_int", "_global_rank", "_world_size", "_rank_label", "_local_rank", "_dist_ready", "_wait_all_ranks"}


def _load_wait_all_ranks():
    """Compile just the sync helpers from eval.py (importing it pulls in lm_eval and the HF caches)."""
    source = Path(__file__).resolve().parents[1] / "eval.py"
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in _NAMES
             or isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_SYNC_SEQ" for t in node.targets)]
    scope = dict(os=os, time=time, dist=dist)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return scope["_wait_all_ranks"]


def _worker(rank, world, port, queue):
    os.environ.update(LOGKV_EVAL_SYNC_HEARTBEAT_S="1", LOGKV_EVAL_SYNC_TIMEOUT_S="0")
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    wait_all_ranks = _load_wait_all_ranks()
    slow = rank == world - 1
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        if slow:
            time.sleep(3)
        waited = wait_all_ranks("slow rank")
        # The slow rank never reaches the second sync; the others must time out naming it.
        os.environ["LOGKV_EVAL_SYNC_TIMEOUT_S"] = "2"
        outcome = None
        if not slow:
            try:
                wait_all_ranks("missing rank")
            except RuntimeError as exc:
                outcome = str(exc)
    queue.put((rank, waited, log.getvalue(), outcome))
    dist.barrier()
    dist.destroy_process_group()


def test_wait_all_ranks_reports_and_times_out_on_missing_ranks():
    world = 3
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(rank, world, port, queue)) for rank in range(world)]
    for proc in procs:
        proc.start()
    results = {rank: (waited, log, outcome) for rank, waited, log, outcome in (queue.get(timeout=60) for _ in procs)}
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0

    for rank in range(world - 1):
        waited, log, outcome = results[rank]
        assert waited > 1.5, (rank, waited)
        assert "slow rank" in log and "仍未到达的 rank: [2]" in log, log
        assert outcome is not None and "[2]" in outcome and "LOGKV_EVAL_SYNC_TIMEOUT_S=2" in outcome, outcome
    assert results[world - 1][0] < 1.5, results[world - 1]


if __name__ == "__main__":
    test_wait_all_ranks_reports_and_times_out_on_missing_ranks()
    print("ok")
