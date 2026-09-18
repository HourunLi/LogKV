[09:09:41] Epoch 1 | Step 1 | loss: 2.1924 | 2nd_scale: 0.0000 | logKV_host: route 959.374s/13440 + replay 239.529s/6720 + plan 10.493s/14336 + pack 9.913s/21504 + attn_fwd 4.485s/21504 + attn_bwd 3.120s/7168 | Time: 1267.46s
step_host (inclusive): data 0.834s | forward 500.778s | backward 765.070s | optimizer 0.049s
logKV_cuda: route 923.892s | replay 198.154s | plan 10.140s | pack 8.699s | attn_fwd 26.852s | attn_bwd 21.366s
step_cuda (inclusive): forward 500.627s | backward 765.390s | optimizer 0.045s



[10:17:31] Epoch 1 | Step 1 | loss: 2.1924 | 2nd_scale: 0.0000 | logKV_host: route 480.193s/6720 + replay 474.553s/13440 + plan 11.624s/14336 + pack 8.023s/21504 + attn_fwd 4.130s/21504 + attn_bwd 2.988s/7168 | Time: 1027.36s
step_host (inclusive): data 1.271s | forward 506.619s | backward 507.551s | optimizer 11.260s
logKV_cuda: route 465.085s | replay 409.758s | plan 10.083s | pack 8.220s | attn_fwd 26.133s | attn_bwd 21.194s
step_cuda (inclusive): forward 506.416s | backward 519.057s | optimizer 0.047s
