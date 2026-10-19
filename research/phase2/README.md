# Plan-R1 第二阶段：反应性交通的风险排序审计

**当前结论：不直接立项新算法；保留小规模真实模型验证。** 机制在CPU toy中存在，普通ensemble已解决大部分失败，剩余危险接受更接近绝对风险低估。近邻已覆盖很多排序/选择/交互校准动机，尚无独立方法创新证据。详见 [go_no_go.md](go_no_go.md)。

## 阅读入口

- [文献比较](literature_comparison.md)：六篇最近邻的问题、算法、目标、实验、保证与覆盖边界；原文来源及版本见 [literature_sources.json](literature_sources.json)。
- [实验结果](experiment_results.md)和[完整高预算表](full_budget_table.md)：实际CSV/JSON生成的结果、图表、负对照和限制。
- [环境与指标](environment_and_metrics.md)：参考器与预测器的独立实现、统计量、预算与额外校准成本。
- [主实验方案](protocol.json)和[事后控制说明](baseline_extension.json)：清楚区分主方案与看结果后新增对照。
- [执行账本](results/execution_accounting.json)、[一致性/重放验证](results/verification.json)、[保全检查](results/preservation_audit.json)、[文件清单与SHA256](artifact_manifest.json)。

## 代码与数据

| 文件 | 用途 |
|---|---|
| `reference_simulator.py` | 有感知延迟、jerk限制和yield记忆的2D独立参考器 |
| `world_model.py` | 拟合解析到达时间代理及4成员bootstrap ensemble |
| `baselines.py` | 9种原始预测/筛选方法的共享计算 |
| `run_experiment.py` | 划分、训练小型统计代理、主比较、bootstrap和ledger |
| `extend_baselines.py` | 事后加入按动作残差/分位数控制，保留原始数据 |
| `validate_numerics.py` | dt与参考样本数的有限敏感性检查 |
| `test_mechanisms.py`, `verify_results.py` | 机制检查、所有记录的一致性和一个块的精确重放 |
| `analyze_results.py`, `build_report.py` | 从保存数据生成诊断、图表和数值报告 |
| `results/decisions.csv`, `extension_decisions.csv` | 所有方法每个场景的选择、预测、参考风险和成本 |
| `results/reference.csv` | 每候选风险计数/99%区间、进度、反应频率、独立部署结果 |
| `results/summary.csv`, `extension_summary.csv`, `all_methods_summary.csv` | 全部预算与条件的聚合指标和95%bootstrap区间 |
| `results/compute_ledger.csv`, `extension_ledger.csv` | 实际在线、离线、参考调用及wall time |
| `results/ranking_decomposition.csv`, `selection_failure_modes.csv` | 严格逆序/平局和选择关联分解 |
| `results/models.json`, `calibration.json`, `extension_parameters.json` | 可重用的小型拟合参数和校准量 |
| `results/*metadata.json`, `numerical_validation.*`, `verification.*` | 配置、实际耗时、源代码hash及数值/重放记录 |
| `results/*.log`, `figures/*.png` | 执行日志和四张图 |

没有提交虚拟环境、缓存、论文PDF、模型权重或大数据集。原始Plan-R1与第一阶段文件保持原样。

## 复现

需要已安装的Python3.9+ CPU环境，版本见 `requirements.txt`。本次使用3.9.25；依赖NumPy1.24.4、SciPy1.10.1、Matplotlib3.9.4。无需torch、GPU、外部数据或网络。

```bash
# 在本目录；目标必须是不存在的新目录，脚本不会覆盖交付数据。
bash reproduce.sh /tmp/plan-r1-phase2-reproduction
```

脚本复制代码/配置后执行主比较、补充控制、敏感性检查、验证和图表/报告生成。主数据和选择应精确重现，wall time随机器变化；新metadata会记录复现时的全部源码hash，不要求与首轮metadata字节相同。若只需重新出图/报告、**不运行仿真**：

```bash
python analyze_results.py
python build_report.py
```

快速机制检查：`python test_mechanisms.py`；快速端到端：`python run_experiment.py --quick --out /tmp/phase2-smoke-new`。后者是缩小配置，不替代交付的正式结果。

复现的对照只是完整论文的轻量思想实例化，非CRAFT/AWM/RISE/CPC等系统的成绩复现。当前选一次8秒宏动作，没有训练完整规划器，也没有证明真实世界泛化。所有科学限制与go/no-go以报告正文为准。

## 分支与审核

新分支 `research/phase2-risk-ranking` 基于未合并的第一阶段分支，PR目标为 `research/phase1-audit`，以便差异只含本目录。待PR #1合并后可再调整目标；本阶段不合并任何PR，不改main。压缩交付包位于 `archives/plan_r1_phase2_research.zip`；内部包含除自身和压缩验证回执外的本阶段文件。完整性回执见 `delivery_verification.json`，清单hash见manifest。
