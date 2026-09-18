[09:09:41] Epoch 1 | Step 1 | loss: 2.1924 | 2nd_scale: 0.0000 | logKV_host: route 959.374s/13440 + replay 239.529s/6720 + plan 10.493s/14336 + pack 9.913s/21504 + attn_fwd 4.485s/21504 + attn_bwd 3.120s/7168 | Time: 1267.46s
step_host (inclusive): data 0.834s | forward 500.778s | backward 765.070s | optimizer 0.049s
logKV_cuda: route 923.892s | replay 198.154s | plan 10.140s | pack 8.699s | attn_fwd 26.852s | attn_bwd 21.366s
step_cuda (inclusive): forward 500.627s | backward 765.390s | optimizer 0.045s



[10:17:31] Epoch 1 | Step 1 | loss: 2.1924 | 2nd_scale: 0.0000 | logKV_host: route 480.193s/6720 + replay 474.553s/13440 + plan 11.624s/14336 + pack 8.023s/21504 + attn_fwd 4.130s/21504 + attn_bwd 2.988s/7168 | Time: 1027.36s
step_host (inclusive): data 1.271s | forward 506.619s | backward 507.551s | optimizer 11.260s
logKV_cuda: route 465.085s | replay 409.758s | plan 10.083s | pack 8.220s | attn_fwd 26.133s | attn_bwd 21.194s
step_cuda (inclusive): forward 506.416s | backward 519.057s | optimizer 0.047s



 [11:02:07] Epoch 1 | Step 1 | loss: 2.1929 | 2nd_scale: 0.0000 | logKV_host: route 444.106s/6720 + replay 450.857s/13440 + plan 11.471s/14336 + pack 7.927s/21504 + attn_fwd 4.107s/21504 + attn_bwd 2.930s/7168 | Time: 956.12s
step_host (inclusive): data 1.280s | forward 470.406s | backward 483.723s | optimizer 0.050s
logKV_cuda: route 428.523s | replay 386.891s | plan 9.894s | pack 8.190s | attn_fwd 25.967s | attn_bwd 20.847s
step_cuda (inclusive): forward 470.211s | backward 484.018s | optimizer 0.044s

[ma-user semanticLogKV]$python unused/benchmark_log_kv_updates.py --iters 10
{"device": "NVIDIA A800-SXM4-80GB", "sequence": 32768, "chunk": 2048, "batch": 4, "groups": 8, "dim": 128, "iters": 10, "torch": "2.11.0+cu128", "cuda": "12.8"}
{"variant": "torch", "phase": "route", "median_ms": 202.086, "peak_extra_MiB": 247.12, "exact_state": true}
{"variant": "torch", "phase": "replay", "median_ms": 50.829, "peak_extra_MiB": 210.99, "exact_state": true}
Traceback (most recent call last):
  File "/data/lihourun/semanticLogKV/unused/benchmark_log_kv_updates.py", line 114, in <module>
    main()
  File "/data/lihourun/semanticLogKV/unused/benchmark_log_kv_updates.py", line 100, in main
    torch.testing.assert_close(dict(cache.named_buffers())[field], tensor, atol=0, rtol=0, msg=field)
  File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/torch/testing/_comparison.py", line 1600, in assert_close
    raise error_metas[0].to_error(msg)
AssertionError: centroid




[ma-user semanticLogKV]$python unused/benchmark_log_kv_updates.py --diagnose
{"device": "NVIDIA A800-SXM4-80GB", "sequence": 32768, "chunk": 2048, "batch": 4, "groups": 8, "dim": 128, "iters": 10, "diagnose": true, "torch": "2.11.0+cu128", "cuda": "12.8"}
Traceback (most recent call last):
  File "/data/lihourun/semanticLogKV/unused/benchmark_log_kv_updates.py", line 68, in checked_centroid
    torch.testing.assert_close(actual, expected, atol=0, rtol=0,
  File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/torch/testing/_comparison.py", line 1600, in assert_close
    raise error_metas[0].to_error(msg)
AssertionError: centroid update 1, ordinal offset=0, runs=256: centroid
Tensor-likes are not equal!

Mismatched elements: 1341 / 32768 (4.1%)
Greatest absolute difference: 7.808208465576172e-05 at index (160, 41)
Greatest relative difference: 0.07515373080968857 at index (60, 89)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/lihourun/semanticLogKV/unused/benchmark_log_kv_updates.py", line 195, in <module>
    main()
  File "/data/lihourun/semanticLogKV/unused/benchmark_log_kv_updates.py", line 146, in main
    cache.route_and_flush_batch(k, v, pos, positions_host=host, record_op_log=True)
  File "/data/lihourun/semanticLogKV/litgpt/log_kv_cache.py", line 2887, in route_and_flush_batch
    return self._route_and_flush_batch(*args, **kwargs)
  File "/data/lihourun/semanticLogKV/litgpt/log_kv_cache.py", line 2971, in _route_and_flush_batch
    self._semantic_route_three_phase(k_raw, v, positions, positions_host, record=record_op_log)
  File "/data/lihourun/semanticLogKV/litgpt/log_kv_cache.py", line 2653, in _semantic_route_three_phase
    self._semantic_route_orphans_fast(
  File "/data/lihourun/semanticLogKV/litgpt/log_kv_cache.py", line 2809, in _semantic_route_orphans_fast
    self._semantic_commit_joins(jobs, k_raw, v, positions, positions_host, record=record)
  File "/data/lihourun/semanticLogKV/litgpt/log_kv_cache.py", line 2246, in _semantic_commit_joins
    self._semantic_commit_joins_inner(jobs, k_raw, v, positions, positions_host, record=record)
  File "/data/lihourun/semanticLogKV/litgpt/log_kv_cache.py", line 2348, in _semantic_commit_joins_inner
    self._semantic_apply_join_plan(
  File "/data/lihourun/semanticLogKV/litgpt/log_kv_cache.py", line 2505, in _semantic_apply_join_plan
    fused.centroid(k_sel, centroid_flat, n_eff_flat, metadata, start, count)
  File "/data/lihourun/semanticLogKV/unused/benchmark_log_kv_updates.py", line 76, in checked_centroid
    raise AssertionError(f"{exc}\nLargest-difference inputs: {json.dumps(values)}") from exc
AssertionError: centroid update 1, ordinal offset=0, runs=256: centroid
Tensor-likes are not equal!

Mismatched elements: 1341 / 32768 (4.1%)
Greatest absolute difference: 7.808208465576172e-05 at index (160, 41)
Greatest relative difference: 0.07515373080968857 at index (60, 89)
Largest-difference inputs: {"cluster": 160, "dim": 41, "old_mu": 1.2907462120056152, "sum": 376.7734375, "pre": 3879.0, "n": 288.0, "numerator": 5383.578125, "denom": 4167.0, "actual": 1.291877269744873, "expected": 1.2919553518295288}
