# 第一阶段研究执行报告

日期：2026-10-20（Asia/Shanghai）。状态：用户要求的五份研究文件、真实 CPU 实验、数据图表、查新及 A/B/C 决策均已完成。没有训练完整模型，没有下载模型权重或大型数据集。

## 1. 交付位置与清单

云端可直接访问 ` /workspace/research/ `，其实体目录为 `/workspace/output/research/`。本执行器的根文件系统只读，平台预设的 `/codex/browser/projectless/.../output` 不存在且不能创建，因此使用可写的 workspace 输出目录；这没有阻止研究和实验完成。

| 文件 | 内容 |
|---|---|
| `plan_r1_analysis.md` | 固定 SHA 的文件/函数/行号、GRPO/VD 实际差别、奖励和掩码公式、loss 与 world-model 机制、假设客观检验 |
| `literature_review.md` | 截至2026-10-20的定向查新；区分全文/摘要/搜索核实；列出直接重叠与未确证的新颖性 |
| `toy_experiment.py` | 可运行的连续运动CPU实验、真实源码函数审计、解析梯度检验、图表输出 |
| `experiment_results.md` | 参数、命令、统计表、图表、解释、额外计算成本和外推边界 |
| `research_proposal.md` | A/B/C明确决策、推荐问题、最近邻差别、否证门槛与下一步资源安排 |
| `results/` | 原始CSV/JSON、3组PNG、2组SVG、执行日志、复现一致性结果 |
| `source_manifest.json` | 作者main与本地HEAD核对结果、关键文件SHA256 |
| `README.md`、`requirements-toy.txt` | 导航与复现说明 |

输出目录上一级还提供 `plan_r1_phase1_research.zip` 完整压缩包。没有在GitHub写文件、发消息、推送或创建PR；“提交成果”以本地研究交付完成。

## 2. 已执行的研究与验证

核实 ICLR 正式记录、作者仓库 main SHA 和现有 checkout 一致，核心源码无本地修改。审计版本为 `23bc54433599002f275c18b88cf261ab5e87d775`。固定版本链接、函数行号、公式和文献入口均在正文给出。

实际运行五种估计量：逐步 reward、RTG、VD 风格累计优势、RTG+RLOO、配对反事实。测量8个seeds×4096组×4条事实轨迹，运行8个seeds×5种方法×120次轻量策略更新。解析期望目标用于评价；没有伪造测试集或依赖模拟器分数充当Plan-R1结果。

另外执行原始优势函数、reward_fn和training_step的受控测试；直接执行真实progress/speed reward类；枚举有限Bernoulli组验证自包含基线效应。最初新增的直线路线fixture曾因polygon类型使用Long而失败，已改为与源数据约定一致的uint8；这是测试fixture修正，不是发现Plan-R1 dtype缺陷。最终完整运行全部通过。

最终版本完整运行两遍，7份数值文件（6份CSV+1份audit JSON）逐字节相同，summary除运行时字段外一致；详见 `results/reproducibility.json`。脚本哈希在summary中固定。图表经实际打开检查。

## 3. 关键发现

1. **原始缺陷假设不成立。** Plan-R1已有累计未来中心化奖励；首次碰撞步参与训练，风险会回传到早期动作。不能以“缺少return-to-go”立项。
2. **累计优势不是完整解法。** 可具体研究自包含基线的时间缩放、随机终止的目标语义、outcome广播的时间权重，以及周车模型风险覆盖。源码还显示，首次违规后的屏蔽状态可通过全轨迹progress影响有效前缀reward，受控执行已复现。
3. **toy对反事实主线的支持有限。** 在已知模型下反事实降方差，但RTG、VD、RLOO和反事实最终可控风险均约1%，没有显示反事实决定性的最终优势；主比较未匹配额外分支总计算，不能宣称更高计算效率。
4. **相关研究高度拥挤。** COCOA、Shapley credit、驾驶CRAFT、AWM、RAD-2、AD-WM、RISE等已覆盖多项直觉。完整替代方向的新颖性尚未确证。

## 4. A/B/C 决定

**A：不建议沿“Plan-R1早期动作收不到后续风险”继续。** 保留短期目标一致性诊断；基础估计修复不自动成为方法贡献。

**B：有更合理的候选问题。** 优先研究反应性交通中的分支风险判断可靠性；次选mixed reward/终止一致性和连续时间token安全。对抗world model、CVaR、自纠错等泛化版本与既有工作直接重叠。

**C：最推荐的研究起点是“固定计算预算下，对ego干预引发的周车响应风险差进行可靠估计与选择后误差控制”。** 先验证其相对CRAFT/AWM/AD-WM/RISE及robust conformal planning的新内容。它仍是待否证候选，不能承诺已经达到CCF B/A独立创新要求。

## 5. 仍然没有被验证的事项

没有验证真实Plan-R1模型的nuPlan/interPlan表现、训练收敛、真实碰撞率或闭环风险排序；没有实现提案中的校准算法或证明其保证；没有穷尽未索引/付费文献；部分2026文献只核实一手摘要而非全文证明；未核实未来会议CFP与投稿等级。上述局限均在相关文档显式标注。

这些事项不属于本次CPU第一阶段已取得的成果。当前交付足以支持收缩错误命题、建立强基线并决定下一轮便宜的否证实验。
