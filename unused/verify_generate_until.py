#!/usr/bin/env python3
"""
最小化调用 eval.py 里 CustomResearchLM.generate_until，检查续写是否正常。

与 lm-eval 一致：每条 request 为 ``args = (prompt_str, gen_kwargs_dict)``。

用法（在仓库根目录执行，以便 ``from eval import ...`` 与 ``utils`` 可解析）::

    python unused/verify_generate_until.py \\
        --checkpoint_dir checkpoints/Qwen/Qwen3-0.6B-Base \\
        --prompt "从前有座山，山上有座庙，庙里有个老和尚在讲故事。他说："

可选：``--sample`` 开启采样，``--max_gen_toks`` 控制最大新生成 token 数。
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

# 保证从仓库根目录运行时也能找到根目录下的 eval / utils
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> None:
    p = argparse.ArgumentParser(description="验证 eval.CustomResearchLM.generate_until")
    p.add_argument(
        "--checkpoint_dir",
        default="checkpoints/Qwen/Qwen3-0.6B-Base",
        help="含 lit_model.pth 与 tokenizer 的 checkpoint 目录",
    )
    p.add_argument(
        "--prompt",
        default="请用一句话介绍什么是大语言模型：",
        help="续写用的前文",
    )
    p.add_argument("--max_gen_toks", type=int, default=128, help="最多新生成 token 数（传给 lm-eval 风格 gen_kwargs）")
    p.add_argument("--device", default=None, help="例如 cuda:0 或 cpu；默认自动选 cuda 否则 cpu")
    p.add_argument("--map_branch", action="store_true", help="与 eval.py main 一致，映射 research 分支权重")
    p.add_argument("--sample", action="store_true", help="采样生成；默认贪心（与 generate_until 中 do_sample=False 一致）")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    args = p.parse_args()

    import torch

    from eval import CustomResearchLM

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    gen_kwargs: dict = {"max_gen_toks": args.max_gen_toks}
    if args.sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p
        gen_kwargs["top_k"] = args.top_k

    req = SimpleNamespace(args=(args.prompt, gen_kwargs))
    lm = CustomResearchLM(
        args.checkpoint_dir,
        device=device,
        map_branch=args.map_branch,
    )

    outputs = lm.generate_until([req])
    text = outputs[0] if outputs else ""

    print("--- prompt ---")
    print(args.prompt)
    print("--- generate_until 返回的续写片段（不含 prompt）---")
    print(text)
    print("--- 拼接预览 ---")
    print(args.prompt + text)


if __name__ == "__main__":
    main()
