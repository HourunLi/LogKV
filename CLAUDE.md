1. 我现在在本地看代码，代码没法实际运行，也没有python环境，你就看代码文本来判断

# 项目目标
先从一个pretrained LLM出发，初始化prefill分支和decode分支，prefill分支强制SWA和IdentityOp交替，decode分支不变；数据上uniform的选断点，设定一个schedule。
