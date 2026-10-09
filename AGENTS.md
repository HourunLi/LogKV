# 项目工作约定

当前算法入口是 [README](README.md)、[共享算法规格](docs/algorithm-spec.md) 和
[Alpha 精确片段算法](docs/alpha-logkv.md)。以当前源码和实际加载的 YAML 为准，
不要把旧分支、旧实验的参数或分数当成当前默认。

- 默认中文，先给结论。只询问会改变实现选择的信息，常规可逆工作直接推进。
- 当前默认路线是 attach 路由 + Alpha + Beta + Gamma（入口参数与 `arc_semantic_fast.yaml` 默认开启）；
  unified 路由保留为对照，其中 `merge_passes=1` 为增量路由、`4` 为冻结中心近似配对。
- 改默认值时，给依赖旧行为的实验 YAML 显式写出原开关，避免复评旧 checkpoint 时被悄悄改变。
- 推理 KV 预算必须保持 O(log N)。精确槽、未闭合片段和打包工作区按文档口径计入预算。
- 保持真实 token 的唯一归属、质量和位置统计；checkpoint 重算与反向不得重新路由或重选精确片段。
- 优先已有工具和小检查。训练使用 `majob.sh`，评测使用 `eval.sh`；不要默认建议全量扫参或重训。
- 文档只保留当前实现、必要公式、运行入口和明确限制；历史推导通过 Git 查找。
  不把运行得到的 profile/评测产物当作过时文档删除。
