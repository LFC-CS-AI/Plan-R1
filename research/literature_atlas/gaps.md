# 待验证的研究空缺

这些是证据尚不足的研究问题，不是“无人做过”的宣言。每项先列已有结果，再列能推翻空缺判断的证据。实现和训练均未开始。

| ID | 问题与证据 | 尚缺的证据 | 什么会推翻/弱化该机会 |
|---|---|---|---|
| G01 | Root式5改变组权重；PlannerRFT §4.4处理同奖反馈；NoRD已有驾驶域Dr.GRPO应用 | 区分方差失真、全同reward与支持集缺失；按场景/训练阶段统计相对贡献 | 若控制c/KL及采样后全部收益可由Dr.GRPO或survival解释，则新归一化组合无独立贡献 |
| G02 | MotionLM §3脚注拒绝声称真实因果；Root表3显示学习WM有益 | ego离开日志分布后的反应有效性、校准和策略对WM的利用情况 | 若更强WM只改善realism而无跨背景planner收益，不足构成规划方法贡献 |
| G03 | PlanTF §III-C已有RL Adapter；Diffusion Planner附E与PlannerRFT附A讨论执行差距 | Root失败究竟来自token几何、行为选择、WM还是LQR；需计划/执行配对诊断 | 若固定适配现有Adapter已消除差距，则仅重新命名修正头没有新意 |
| G04 | CAT-K §4.2依赖GT模态；Root明确修正不安全示范；SMART-R1已有SFT–RFT–SFT | 安全偏离日志之后，恢复目标何时仍有效；保留人类行为与原则优化是否冲突 | 若普通交替CAT-K/RFT已稳定改进且无需新机制，则组合本身不足 |
| G05 | CarPlanner固定mode；PLUTO横纵query；PlannerRFT学习横纵探索 | Root是否存在可测量的模式摇摆或候选覆盖问题，而不是单纯奖励/执行误差 | 若top-1 Root没有明显模式不一致，或已有mode方案在同预算下足够，则不做额外架构 |

## 最值得继续扩展的五个方向

1. **驾驶RL中的难度偏差与有效样本支持。**先完整读 [NoRD](https://arxiv.org/html/2602.21172v2)（本轮L2，不计15篇），再沿Dr.GRPO与PlannerRFT参考文献检查已有优化/采样修正。NoRD已用Dr.GRPO，不能再主张首次将去std用于驾驶。检索负结果不等于未有论文。
2. **对ego干预有效的交通模型。**沿MotionLM、Trajeglish、SMART-R1的Related Work扩展CAT-K、其他闭环模拟后训练；需要从simulation realism走到planner transfer的证据链。
3. **后训练遗忘与安全恢复监督。**以CAT-K和SMART-R1为起点，补读R2SE及其残差specialist/扩展机制，优先排除“已有框架简单组合”。[R2SE的IEEE正式记录](https://ieeexplore.ieee.org/document/11353028/)已核到TPAMI 2026；本轮未精读，不引用其结果数字。
4. **执行一致性与可行性。**先复核PlanTF RL Adapter与PlannerRFT执行训练，再补轨迹修正/动力学约束工作；本轮不把“安全修正”自动缩成曾失败的一维Bridge Head。
5. **tokenizer与多模态覆盖。**SMART自身指出词表/粒度缺少充分消融；沿SMART-R1参考文献的TrajTok线索扩展。应先证明确有表示瓶颈，再谈新tokenizer；本轮不把低量化误差等同闭环收益。

## 选题进入下一阶段的门槛

先核实同方法组合的最近前人及官方可运行性；在固定场景、背景模型、后处理和预算下，写出能产生负结果的最小对照。评价需同时看安全、进度、舒适和每类场景；仅提升聚合CLS或只展示成功case不能支持独立方法贡献。

不使用CCF级别作为方法价值证据，也未根据会议名称推定该选题可达CCF B/A。第一阶段的作用是缩小可信问题范围。
