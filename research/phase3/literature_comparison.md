# 最近邻比较与创新边界

检查日期：2026-10-20（Asia/Shanghai）。这是针对本题的查新，不是穷尽所有安全规划文献。以下优先使用作者论文/正式会议全文；阅读层级与无法验证部分明确区分，来源版本见 `literature_sources.json`。没有复现任何完整论文成绩。

## 1. 对照总表

| 工作 | 问题与方法 | 优化/保证对象 | 实验与本题重叠 |
|---|---|---|---|
| [FCL，ICRA 2012](https://gamma.cs.unc.edu/FCL/fcl_docs/webpage/pdfs/fcl_icra2012.pdf) | 层次包围体、距离与运动界、continuous collision queries/conservative advancement | 给定连续路径的碰撞与接近查询，不是学习模型 | 机器人及合成几何；“端点之间可能漏检”和CCD本身是成熟问题 |
| [Redon等，CGF 2002](https://doi.org/10.1111/1467-8659.t01-1-00587) | 计算刚体首次接触，处理高速/薄物体 | 指定刚体运动，不是未知动力学或策略学习 | 仅核实出版信息与摘要；全文入口失败，不能比较具体数值容差/复杂度 |
| [SMART，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cef5c8dec67597b854f0162ad76d92d2-Abstract.html) | 0.5秒运动段、k-disks词表、滚动匹配、自回归预测 | next-token CE和多车运动分布；没有从token端点到连续安全的保证 | WOMD/nuPlan运动生成；与Plan-R1表示直接相关 |
| [CorrectionPlanner，ICML 2026](https://proceedings.mlr.press/v306/guo26ae.html) | propose–evaluate–correct；碰撞critic；将被拒token作为trace条件重新生成 | IL correction CE + world-model RL，REINFORCE/KL；短期碰撞二分类 | Waymax/nuPlan；已经是生成内部的token安全修正，不能称其只做外部过滤 |
| [Sampled-data CBF，L-CSS 2021](https://arxiv.org/html/2103.03677v2) | 基于局部可达/导数界收紧离散CBF条件 | 分段常数控制下连续时间前向不变性；需动力学/界/可行输入 | unicycle、航天器；“离散安全不推出连续安全”及采样裕量均已覆盖 |
| [Predictive safety filter，Automatica 2021](https://arxiv.org/pdf/1812.05506) | MPC寻找通往安全终端集的备份，最小干预原输入 | 给定模型置信集合、初始可行性、稳定/正则条件下的概率安全 | 摆与四旋翼；不是只检查下一时刻，不能用一步过滤替代它的保证 |
| [REDEFINED](https://arxiv.org/pdf/2403.12280) | zonotope连续区间占据外包络、精确神经隐式距离、在线优化 | 已知保守ego/障碍包络下的not-at-fault规划 | 汽车复杂场景；扫掠体、距离梯度、神经安全约束本身已有工作 |
| [SafeDiffuser，ICLR 2025](https://openreview.net/pdf?id=ig2wk7kK9J) | 把CBF/QP引入去噪生成过程 | diffusion-time不变性；论文明确区分规划离散时间与连续执行 | 连续状态轨迹生成；“把安全引入生成而非仅后处理”不是空白 |
| [Shielding，AAAI 2018](https://arxiv.org/pdf/1708.08611) | 生成前删除不安全动作或生成后替换；安全游戏/抽象 | 形式规约，依赖正确抽象和shield状态 | 离散RL；动作mask、拒判惩罚、用安全修正反馈训练不是新方法 |

## 2. CorrectionPlanner：必须跨过的最近邻

精读正式PDF §3–4.3及算法/实验相关部分，另核对arXiv v1。其token同为0.5秒、1024个端点位移/转角。critic预测短窗口内碰撞，拒绝的token进入注意力trace，后续proposal分布因此改变；IL训练修正trace，RL用冻结reactive world model、轨迹奖励和KL更新执行token。

因此“增加collision head、发现危险后重采token、用RL训练修正”已经被直接覆盖。本文读到的保证不是区间占据的解析安全证书，但**没有其碰撞后端源码，不能断言它也只查0.5秒端点或存在同样漏检**。候选新方法必须对比“CorrectionPlanner+同一个区间标签”，否则收益可能完全来自标签修复。

## 3. 连续时间安全：现成工具不能包装为AI贡献

FCL §II.C解释固定采样遗漏、BVH区间包络和CA的距离/运动界；本轮自适应SAT区间器只是2D诊断实现，不复现FCL，也不声称新CCD算法。Redon原论文仅摘要可核查，故不引用其未读的定理。

Sampled-data CBF §III Theorem1/3用局部Lipschitz/二阶界补偿采样间演化；初始安全、模型条件及输入可行性不可省略。它处理的是ZOH控制，不直接等价于Plan-R1端点位姿token，但已经否定“第一次研究inter-sample safety”的说法。本轮`A T²/8`端点桥包络也是常规插值误差界，不是新定理。

## 4. Reachability、安全过滤与生成内部安全

REDEFINED §IV Assumptions9/10要求每个区间的ego/障碍占据被zonotope包含；Theorem12给出区间距离的保守关系，再把约束用于优化。这里的网络是特定几何函数的精确表示，不是随意训练的风险分类器。我们的宽松加速度tube还没有控制器误差、备份停止和递归可行性，不能对等宣称reachability安全规划。

PSF §4 Theorem4.6依赖安全终端集、收缩/置信集合正则和初始可行性；一步拒判不能继承它的未来安全。SafeDiffuser §3明确以离散规划状态为对象，连续规划时间可由下层CBF处理；它已把约束放入生成动力学。Shielding §5也同时覆盖生成前动作集合与生成后替换。只把CCD的0/1标签送回policy，不自动成为独立AI方法。

## 5. 本题还可能研究什么

已确认的工程问题是训练标签在端点和连续桥之间不一致；已有10Hz refinement也使部署动机弱于“全部系统漏检”。本轮候选选择的10Hz检查已与CCD一致，必须承认它是有效廉价基线。

可保留的**待验证方法差异**是：保留原端点token词表，同时由生成器选择并学习一个与起/终状态一致的区间运动桥，以区间margin约束联合更新token概率和桥参数。CorrectionPlanner改变的是token proposal/trace；这份草案改变可执行token的区间实现与训练安全目标。这个差异是算法定义上的区别，**尚未证明是文献空白或能带来性能提升**；混合动作、运动primitive细化、层次控制和shield蒸馏均存在大量邻近研究。

如果同样的区间标签加在原Plan-R1/CorrectionPlanner上就达到相同效果，或10Hz后处理已经达到同样安全—进度—计算前沿，应停止把“新桥+新loss”作为主贡献。完整方法方案与退出标准见 `method_proposal.md`、`go_no_go.md`。
