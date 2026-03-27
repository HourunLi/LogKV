import os
from jsonargparse import CLI
from pathlib import Path

# 🌟 魔法在这里：直接导入 LitGPT 最底层的评测引擎
from litgpt.eval.evaluate import convert_and_evaluate 

def main(
    checkpoint_dir: str = "checkpoints/Qwen/Qwen3-0.6B-Base",
    benchmark: str = "debug",  
    custom_tasks: str = "",          
    batch_size: int = 1,
    limit: int | None = None,               
    out_dir: str = "./eval_results"
):
    TASK_SUITES = {
        "commonsense": "hellaswag,piqa,winogrande,arc_easy,arc_challenge",
        "longbench": "longbench_qasper,longbench_multifieldqa_en,longbench_samsum",
        "debug": "piqa" 
    }

    tasks = custom_tasks if benchmark == "custom" else TASK_SUITES.get(benchmark)
    if not tasks:
        raise ValueError(f"未知的 benchmark: {benchmark}")

    if "longbench" in benchmark:
        print("⚠️ 检测到 LongBench，自动锁定 batch_size=1 防爆显存")
        batch_size = 1

    result_dir = os.path.join(out_dir, benchmark)
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"🚀 启动原生 Python 评测管线...")
    print(f"📋 任务: {tasks} | ⏱️ Limit: {limit}")

    # ==========================================
    # 🌟 直接在当前进程调用 evaluate，完美支持断点调试和代码热更新！
    # ==========================================
    convert_and_evaluate(
        checkpoint_dir=Path(checkpoint_dir),
        out_dir=Path(result_dir),
        tasks=tasks,
        batch_size=batch_size,
        limit=limit
    )
    
    print("✅ 评测圆满完成！")

if __name__ == "__main__":
    CLI(main)