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


===================================================================== FAILURES =====================================================================
_______________________________________________ test_fused_route_replay_output_and_gradients[dtype0] _______________________________________________

dtype = torch.float32

    @CUDA
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_fused_route_replay_output_and_gradients(dtype):
        assert kv._triton_updates() is not None
        torch.manual_seed(53)
        q = torch.randn(2, 4, 96, 8, device="cuda", dtype=dtype, requires_grad=True)
        k = torch.randn(2, 2, 128, 8, device="cuda", dtype=dtype)[:, :, 16:112].requires_grad_()
        v = torch.randn(2, 128, 2, 8, device="cuda", dtype=dtype).transpose(1, 2)[:, :, 16:112].requires_grad_()
        upstream = torch.randn_like(q)
    
        def run():
            cache = cache_for("cuda", dtype)
            out = kv.LogKVStreamTrainingAttention.apply(q, k, v, cache, 8 ** -.5, 8, 0., k)
            grad = torch.autograd.grad(out, (q, k, v), upstream)
            return (out, *grad), {name: t.clone() for name, t in cache.named_buffers()}
    
        with patch.object(kv, "_triton_updates", return_value=None):
>           reference, buffers = run()

tests/test_log_kv_updates.py:106: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_log_kv_updates.py:101: in run
    out = kv.LogKVStreamTrainingAttention.apply(q, k, v, cache, 8 ** -.5, 8, 0., k)
/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/torch/autograd/function.py:596: in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

ctx = <torch.autograd.function.LogKVStreamTrainingAttentionBackward object at 0x7f53a15918b0>
q = tensor([[[[-7.5125e-01,  4.7407e-01,  4.9919e-01,  ...,  2.1828e-01,
            5.4525e-01,  7.5375e-01],
          [...6783e+00, -5.5390e-01,  ..., -1.2022e+00,
           -1.0956e+00,  6.4063e-01]]]], device='cuda:0', requires_grad=True)
k = tensor([[[[-1.3318e+00, -4.5381e-01, -2.1309e-01,  ..., -2.2963e-01,
            5.8557e-01,  9.5605e-01],
          [...2933e+00,  9.7358e-01,  ...,  8.6408e-01,
            3.2614e-01, -3.4367e-01]]]], device='cuda:0', requires_grad=True)
v = tensor([[[[ 0.4369,  0.6566,  0.8166,  ..., -1.5579,  0.7524,  0.3433],
          [ 0.0236, -0.4983, -0.0526,  ..., -1...
          [-0.1469, -0.2827, -0.7257,  ..., -0.4071,  1.3447,  0.4864]]]],
       device='cuda:0', requires_grad=True)
cache = LogStructuredKVCache(), scale = 0.3535533905932738, train_block = 8, second_order_scale = 0.0
k_raw = tensor([[[[-1.3318e+00, -4.5381e-01, -2.1309e-01,  ..., -2.2963e-01,
            5.8557e-01,  9.5605e-01],
          [...2933e+00,  9.7358e-01,  ...,  8.6408e-01,
            3.2614e-01, -3.4367e-01]]]], device='cuda:0', requires_grad=True)

    @staticmethod
    def forward(ctx, q, k, v, cache, scale, train_block, second_order_scale, k_raw=None):
        commit_k_raw = k if k_raw is None else k_raw
    
        T = q.size(2)
        train_block = int(train_block)
        if train_block < 2:
            raise ValueError(f"logKV train_block must be >= 2, got {train_block}")
        if train_block > cache.recent_size:
>           raise ValueError(
                f"logKV train_block ({train_block}) must be <= recent_size ({cache.recent_size})"
            )
E           ValueError: logKV train_block (8) must be <= recent_size (4)

litgpt/log_kv_cache.py:4726: ValueError
_______________________________________________ test_fused_route_replay_output_and_gradients[dtype1] _______________________________________________

dtype = torch.bfloat16

    @CUDA
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_fused_route_replay_output_and_gradients(dtype):
        assert kv._triton_updates() is not None
        torch.manual_seed(53)
        q = torch.randn(2, 4, 96, 8, device="cuda", dtype=dtype, requires_grad=True)
        k = torch.randn(2, 2, 128, 8, device="cuda", dtype=dtype)[:, :, 16:112].requires_grad_()
        v = torch.randn(2, 128, 2, 8, device="cuda", dtype=dtype).transpose(1, 2)[:, :, 16:112].requires_grad_()
        upstream = torch.randn_like(q)
    
        def run():
            cache = cache_for("cuda", dtype)
            out = kv.LogKVStreamTrainingAttention.apply(q, k, v, cache, 8 ** -.5, 8, 0., k)
            grad = torch.autograd.grad(out, (q, k, v), upstream)
            return (out, *grad), {name: t.clone() for name, t in cache.named_buffers()}
    
        with patch.object(kv, "_triton_updates", return_value=None):
>           reference, buffers = run()

tests/test_log_kv_updates.py:106: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_log_kv_updates.py:101: in run
    out = kv.LogKVStreamTrainingAttention.apply(q, k, v, cache, 8 ** -.5, 8, 0., k)
/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/torch/autograd/function.py:596: in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

ctx = <torch.autograd.function.LogKVStreamTrainingAttentionBackward object at 0x7f53a1591ad0>
q = tensor([[[[-7.5000e-01,  4.7461e-01,  5.0000e-01,  ...,  2.1875e-01,
            5.4688e-01,  7.5391e-01],
          [...-1.2031e+00,
           -1.0938e+00,  6.4062e-01]]]], device='cuda:0', dtype=torch.bfloat16,
       requires_grad=True)
k = tensor([[[[-1.3281e+00, -4.5312e-01, -2.1289e-01,  ..., -2.2949e-01,
            5.8594e-01,  9.5703e-01],
          [... 8.6328e-01,
            3.2617e-01, -3.4375e-01]]]], device='cuda:0', dtype=torch.bfloat16,
       requires_grad=True)
v = tensor([[[[ 0.4375,  0.6562,  0.8164,  ..., -1.5547,  0.7539,  0.3438],
          [ 0.0236, -0.4980, -0.0527,  ..., -1...0.2832, -0.7266,  ..., -0.4062,  1.3438,  0.4863]]]],
       device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
cache = LogStructuredKVCache(), scale = 0.3535533905932738, train_block = 8, second_order_scale = 0.0
k_raw = tensor([[[[-1.3281e+00, -4.5312e-01, -2.1289e-01,  ..., -2.2949e-01,
            5.8594e-01,  9.5703e-01],
          [... 8.6328e-01,
            3.2617e-01, -3.4375e-01]]]], device='cuda:0', dtype=torch.bfloat16,
       requires_grad=True)

    @staticmethod
    def forward(ctx, q, k, v, cache, scale, train_block, second_order_scale, k_raw=None):
        commit_k_raw = k if k_raw is None else k_raw
    
        T = q.size(2)
        train_block = int(train_block)
        if train_block < 2:
            raise ValueError(f"logKV train_block must be >= 2, got {train_block}")
        if train_block > cache.recent_size:
>           raise ValueError(
                f"logKV train_block ({train_block}) must be <= recent_size ({cache.recent_size})"
            )
E           ValueError: logKV train_block (8) must be <= recent_size (4)

litgpt/log_kv_cache.py:4726: ValueError
============================================================= short test summary info ==============================================================
FAILED tests/test_log_kv_updates.py::test_fused_route_replay_output_and_gradients[dtype0] - ValueError: logKV train_block (8) must be <= recent_size (4)
FAILED tests/test_log_kv_updates.py::test_fused_route_replay_output_and_gradients[dtype1] - ValueError: logKV train_block (8) must be <= recent_size (4)
2 failed, 24 passed, 1 skipped, 15 warnings in 70.24s (0:01:10)
