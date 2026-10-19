# Plan-R1 第一阶段研究交付

研究日期：2026-10-20。推荐先读 [执行报告](execution_report.md)，再读 [研究建议](research_proposal.md)。

五份指定文件：

- [源码分析与数学解释](plan_r1_analysis.md)
- [相关研究查新](literature_review.md)
- [可运行 CPU 实验](toy_experiment.py)
- [实验数据解释与图表](experiment_results.md)
- [研究决定与下一步方案](research_proposal.md)

附加证据：`source_manifest.json` 固定作者代码版本和文件哈希；`results/` 包含 CSV、JSON、图表、执行日志及两次完整运行的一致性结果。`requirements-toy.txt` 固定独立 toy 依赖；源码审计使用云端已有 Plan-R1 CPU 环境。

```bash
source /workspace/.planr1/activate.sh
python /workspace/research/toy_experiment.py \
  --out /workspace/research/results \
  --source-root /workspace/Plan-R1
```

可移植 toy（不执行 Plan-R1 源码审计）：

```bash
python -m pip install -r requirements-toy.txt
python toy_experiment.py --out results
```

后一条命令会在 summary 中标记源码审计 skipped，不能用来更新“源码审计已通过”的证据。建议使用新的输出目录保留原始交付结果。

本交付没有运行完整模型训练、下载模型权重/大型数据集、执行真实 nuPlan 闭环基准或提交到远程仓库。机制实验不构成 Plan-R1 真实基准缺陷的直接证据。
