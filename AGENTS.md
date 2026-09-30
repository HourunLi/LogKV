# 项目工作约定

当前算法入口是 [README](README.md)、[算法规格](docs/algorithm-spec.md) 和
[统一路由](docs/semantic-unified-routing.md)。以源码和实际加载的 YAML 为准，
不要把旧分支、旧实验的参数或分数当成当前默认。

- 默认中文，先给结论。只询问会改变实现选择的信息，常规可逆工作直接推进。
- 统一路由配置显式开启 `log_kv_semantic_unified_route`；原 fast 路由仍用于对照，不能混淆。
- 推理 KV 预算必须保持 O(log N)。区分持久缓存、打包工作区与路由临时张量的显存口径。
- 保持真实 token 的唯一归属、质量和位置统计；checkpoint 重算与反向按记录回放，不重新路由。
- 优先已有工具和小检查。训练用 `majob.sh`、评测用 `eval.sh`，SWA 用专门对照脚本；
  具体命令集中在 README。不要默认建议全量扫参或重新训练。
- 文档只保留当前实现、必要公式、运行入口和明确限制；历史推导通过 Git 查找。
  不把运行得到的 profile/评测产物当作过时文档删除。
