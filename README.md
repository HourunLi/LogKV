LogKV timing: host counters every optimizer step; CUDA profile steps=[]. Console/TensorBoard timings are rank 0 only. CUDA spans include stream waits; forward/backward totals contain the LogKV breakdown.
[11:34:19] Epoch 1 | Step 1 | loss: 1.6328 | 2nd_scale: 0.0000 | logKV_host: route 94.780s/196 + replay 22.861s/392 + plan 0.518s/448 + pack 3.089s/672 + attn_fwd 0.154s/672 + attn_bwd 0.229s/224 | Time: 142.00s
step_host (inclusive): data 4.014s | forward 108.365s | backward 29.566s | optimizer 0.053s
