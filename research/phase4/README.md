# Phase 4 — Continuous-time Motion Token bridge feasibility

本目录承接 PR #3；**只增加 phase4，不修改 Plan-R1/阶段三代码，不合并旧 PR**。本轮完成受约束合成可行性检验、实际小型 Head 训练与独立场景测试。没有完整模型/RL训练，没有大型数据、GPU或随机权重冒充预训练输出。

建议依次阅读：

1. [Go/no-go 与未完成事项](go_no_go.md)
2. [实际实验结果](experiment_results.md)
3. [官方 checkpoint、访问障碍与具体输入清单](checkpoint_and_inputs.md)
4. [实验设计/公平性/证书条件](experiment_design.md)
5. [方法重叠与 CorrectionPlanner 比较](method_scope.md)

代码：`bridge_dynamics.py`（初态/端点/C2桥、导数证书）、`geometry.py`（连续最小化标签与独立区间器）、`build_data.py`、`train_heads.py`、`evaluate.py`、`audit_labels.py`、`test_constraints.py`、`verify_saved.py`、`analyze_results.py`、`source_audit.py`、`real_inference.py`。

结果：

- `results/*_scenes.json`：768/192/512个训练/验证/测试场景，独立交通场景种子。
- `results/*_data.npz`：特征、双方共享的18候选标签/裕量、系数网格；`bases.json`：原词表ID、token、全部初态/多项式/获证幅值。
- `results/heads/*.npz`：12个真正训练过的小模型状态（可严格重载），不是 Plan-R1 checkpoint。
- `training_curves.csv`、`training_summary.json`：所有 epoch 和真实更新。
- `test_predictions.csv`、`test_metrics.csv`：所有方法/种子/场景原始结果、accepted/reject、unknown、动力学/端点、进度代理/实际CPU时延/参数/调用数。
- `feasibility.csv`、`failure_cases.json`、可用时 `existence_witness.json`：可行性与失败/存在性证据。
- `summary.json`、`constraint_tests.json`、`label_audit.json`、`verification.json`：统计和交叉验证。
- `*_ledger.json`、`*.log`、`environment.json`、`source_audit.json`、`checkpoint_preflight.json`：成本、复现日志、依赖、源码与官方文件核验。
- `figures/*.png`：由真实保存数据生成的图，不手工填数。
- `results/diagnostic_v1_sampled_labels/`：被独立检查发现200Hz标签混叠的完整旧版代码/数据/模型/日志。它不是主结果；保留以审计数值修正。若要复现旧版，请在另一个 checkout 将其中 `source/` 内容复制回 `research/phase4/`，不能直接在嵌套归档目录运行，亦不要覆盖本目录已交付结果。
- `artifact_manifest.json`：交付文件路径/大小/SHA-256。NPZ/PNG 是必要小型实验产物；无环境、缓存、官方权重或大型数据。

CPU复现（不训练完整 Plan-R1）：先使用 `requirements-cpu.txt` 的 Python3.9 CPU依赖，再在仓库根运行：

```bash
bash research/phase4/reproduce.sh /path/to/NEW_phase4_reproduction
```

脚本建立全新 worktree，顺序生成主协议 v2 数据、训练12个小 Head、检查/评价/作图；不会重做前三阶段。固定种子可复现数值，云CPU耗时不要求逐位一致。保存结果的快速核对（不训练）：

```bash
python research/phase4/verify_saved.py
```

`real_inference.py` 是需要真实 checkpoint+processed 场景的后续入口；本轮只运行其 meta 架构计数，严格加载推理路径未实际执行。更多前提见输入清单，不能用该脚本的存在替代真实模型证据。
