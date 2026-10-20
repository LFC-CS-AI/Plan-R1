# Continuous-Time Safety-Aligned Motion Token Planning：第一轮

**源码与机制已确认，独立方法贡献尚待验证。** 原训练碰撞奖励检查0.5秒端点，已有refinement在0.1秒评分；原始token合成测试发现稳定的区间漏检，但10Hz廉价筛选已解决当前候选选择测试。本轮没有运行真实预训练推理或完整模型训练。

## 报告入口

- [源码核查](source_analysis.md)：文件、函数、行号、原奖励几何与各条路径的时间语义。
- [文献比较](literature_comparison.md)：CorrectionPlanner、CCD、SMART、sampled-data CBF、安全过滤、reachability、SafeDiffuser、shielding及覆盖边界。
- [实验设计与真实模型条件](experiment_design.md)：原词表、运动桥、统计口径、参考器限制与输入格式。
- [实际结果与CPU账本](experiment_results.md)：所有检测器、候选基线、稳定性检查、图表与原型结果。
- [方法方案](method_proposal.md)：保留原端点词表的token/bridge联合生成与区间学习目标；与最近邻的实质差异和反驳。
- [继续/终止报告](go_no_go.md)：下一轮只进行有退出条件的小规模验证，不承诺CCF B/A创新已成立。
- [文件清单与SHA256](artifact_manifest.json)、[交付校验](delivery_verification.json)。

## 源码与数据

| 文件 | 功能 |
|---|---|
| `source_audit.py` | 原始源码行号/hash、真实词表、原transition AST执行对齐、资源检查 |
| `native_probe.py` | 直接运行未修改AgentCollisionReward/TTCReward |
| `motion_geometry.py` | 原token解码、线性/Hermite桥、非对称车框SAT、自适应区间检查、加速度外包络 |
| `run_diagnostics.py` | 按预先保存协议运行3072区间和2176候选，记录原始数据与实际耗时 |
| `test_geometry.py`, `validate_results.py` | 独立几何验证、witness核对、±10cm和曲率敏感性 |
| `interval_learning_prototype.py` | token概率和端点保留bridge的autograd连通检查，不训练模型 |
| `audit_saved_outputs.py` | 导入真实输出的NPZ/JSON接口；目前仅验通synthetic示例 |
| `analyze_results.py`, `build_report.py` | 从保存数据生成汇总、三张图和数值报告 |
| `protocol.json` | 在主运行前保存的设计与退出标准 |
| `results/diagnostics.csv` | 每例token ID、起终位姿、box、全部检测器与参考结果 |
| `results/vocabulary_kinematics.csv` | 全部Vehicle/Pedestrian词表与可用子集的解析运动界 |
| `results/candidate_bank.csv`, `selections.csv`, `selection_summary.csv` | 共享候选、偏好、筛选、重排、拒判和成本 |
| `results/detection_summary.csv`, `seed_summary.csv`, `leak_robustness.csv` | 事件分解、种子分层、事后敏感性 |
| `results/compute_ledger.csv`, `compute_summary.csv`, `metadata.json` | 实际CPU-only elapsed时间与已插桩几何查询；无神经模型调用 |
| `results/source_audit.json`, `geometry_tests.json`, `validation.json`, `learning_probe.json` | 可核查证据，明确没有真实模型安全结论 |
| `results/*.log`, `figures/*.png` | 复现日志与图 |

原词表不复制到本目录；直接读取仓库中的38682字节文件。没有提交checkpoint、虚拟环境、缓存、论文PDF或大型数据。本阶段不重跑phase1/phase2，不修改它们或原论文代码。

## 复现

需要已安装的CPU环境。实际版本见 `results/environment.json`：Python3.9.25，NumPy1.24.4、SciPy1.10.1、PyTorch2.1.0+cpu、torch-scatter2.1.2、PyG2.3.1、Shapely2.0.7、Matplotlib3.9.4；原奖励导入所需其他仓库依赖沿用现有环境。无需GPU或外部数据。

```bash
# 在本目录，已激活上述CPU环境。创建新的detached worktree，避免覆盖交付数据。
bash reproduce.sh /tmp/plan-r1-phase3-reproduction
```

这会重跑**本阶段**源码/机制/主诊断/敏感性/autograd和出图，不会执行前两阶段。实际时间随机器变化；metadata和源码清单会记录复现时版本，不要求包含wall time的文件字节相同。完整矩阵的二次复现结果见 `results/reproducibility.json`。在新worktree运行后，原压缩包/manifest仍是首轮交付快照，不会被脚本重新打包。

只从保存数据重新生成图表和报告、不调用仿真：

```bash
python analyze_results.py
python build_report.py
```

真实输出导入示例（所需字段、来源与缺失条件见实验设计）：

```bash
python audit_saved_outputs.py outputs.npz --provenance outputs.json --out /tmp/real-output-audit
```

区间证书始终依赖显式运动桥与有效误差界；未知不能当作安全或实际碰撞。单区间检查不等于全道路闭环安全，原型的非零梯度不等于学习有效。

## 审核与下载

分支 `research/phase3-continuous-safety` 独立基于main提交，PR只增加 `research/phase3/`，不依赖也不合并PR #1/#2。压缩包为 `archives/plan_r1_phase3_research.zip`，包含除自身和交付回执外的本阶段文件；hash/CRC及条目校验见交付回执。当前云运行环境没有原生聊天文件附件出口，可通过PR中的GitHub ZIP入口下载。
