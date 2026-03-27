1. 我username是user
2. 我在litgpt环境下工作
3. 项目绝对路径：/home/user/test/litgpt/

# 项目目标
先从一个pretrained LLM出发，初始化prefill分支和decode分支，prefill分支强制SWA和IdentityOp交替，decode分支不变；数据上uniform的选断点，设定一个schedule。
