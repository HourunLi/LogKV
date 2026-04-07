# 首先安装 python 3.10
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
or -f https://mirrors.aliyun.com/pytorch-wheels/cu128/ 
pip install -e '.[all]' --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/
pip install hf_transfer --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/
pip install lm-eval --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/
pip install flash-attn --no-build-isolation

litgpt download Qwen/Qwen3-0.6B-Base