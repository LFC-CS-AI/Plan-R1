# 相关工作查新与创新性评估

检索截止：**2026-10-20，Asia/Shanghai**。不包含当月未来日期，不声称覆盖截至 10 月 31 日的论文。以下链接均在本次通过联网检索得到；结论优先依据论文/会议正式页面或作者仓库，未把自动论文解读作为方法证据。

## 1. 检索与核实范围

使用公开网页搜索发现文献，再打开 arXiv、NeurIPS、PMLR、ICLR、ACL、RLJ 等一手页面。查询涉及：`Plan-R1 ICLR 2026 github VD-GRPO`、`reinforcement learning temporal credit assignment counterfactual COCOA Hindsight Credit Assignment RUDDER`、`autonomous driving trajectory planning counterfactual credit assignment reinforcement learning 2025 2026`、`Plan-R1 credit assignment`、`counterfactual GRPO driving`、`adaptive rollout allocation variance budget`、`world model risk ranking calibration driving 2026`、`motion token safety adaptive planning 2026`，并对新发现题名精确检索。

核实级别：**F**=阅读所涉方法/补充章节或源码；**A**=阅读一手摘要/正式记录，未逐式核查方法证明；**S**=只核实一手搜索记录。A/S 足以识别研究主题的重叠，不能据此断言两个完整算法等价。没有访问付费 Scopus/WoS，也没有保证穷尽所有未被索引的 2026 预印本。

## 2. 信用分配与无 critic 优化

| 论文与可核实记录 | 年份/状态；级别 | 对本项目的直接限制 |
|---|---|---|
| [RUDDER: Return Decomposition for Delayed Rewards](https://papers.neurips.cc/paper_files/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html) | NeurIPS 2019；A | 将延迟回报重新分配到贡献动作已有先例；“把晚期风险移到早期”不是新的方法概念 |
| [Hindsight Credit Assignment](https://proceedings.neurips.cc/paper/2019/hash/195f15384c2a79cedf293e4a847ce85c-Abstract.html) | NeurIPS 2019；A | 用未来结果判断过去动作相关性，覆盖 hindsight 归因的基本思路 |
| [Counterfactual Multi-Agent Policy Gradients（COMA）](https://ojs.aaai.org/index.php/AAAI/article/view/11794) | AAAI 2018；A | 固定其他 agent、比较替代动作的反事实 baseline 已存在；其多 agent 归因不等同于 ego 时序归因，但不能忽略 |
| [Counterfactual Credit Assignment in Model-Free Reinforcement Learning](https://proceedings.mlr.press/v139/mesnard21a.html) | ICML 2021；A | 用未来条件 baseline 区分 skill/luck；明确要求控制未来信息中的动作泄漏，提醒“反事实”并不自动无偏 |
| [Would I have gotten that reward? Long-term credit assignment by counterfactual contribution analysis（COCOA）](https://papers.neurips.cc/paper_files/paper/2023/hash/d8bd445c2abe1343cce0e14b361b2fb3-Abstract-Conference.html) | NeurIPS 2023；A | 测量动作对未来 reward 的贡献已经是直接研究对象；本项目最重要的历史对照之一 |
| [Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs](https://aclanthology.org/2024.acl-long.662/) | ACL 2024；A | RLOO 是现成、便宜的无 critic 基线；只把组中心换成 leave-one-out 不够独立创新 |
| [Understanding R1-Zero-Like Training: A Critical Perspective（Dr. GRPO）](https://arxiv.org/abs/2503.20783) | 2025 预印本记录已核实；A | 移除组标准差、分析归一化与长度偏好已有基础；不能把 VD 的尺度修正再次当新贡献 |
| [Counterfactual Shapley Credit Assignment](https://rlj.cs.umass.edu/2026/papers/Paper45.html) | RLJ/RLC 2026，一手页仍标 pre-proceedings、页码 TBD；A | 已直接研究随机性、稀疏因果与延迟回报的归因；“Shapley+信用分配”属于高度覆盖方向 |
| [Improving Reward-Based Hindsight Credit Assignment（R-HCA）](https://rlj.cs.umass.edu/2026/papers/Paper133.html) | RLJ/RLC 2026，一手页仍标 pre-proceedings；A | 针对 COCOA 在带噪即时奖励下的方差问题；不可把 COCOA 当绝对最强或已解决全部问题 |
| [CRAFT: Counterfactual Credit Assignment from Free Sibling Rollouts for Self-Distilled Agentic Reinforcement Learning](https://arxiv.org/abs/2606.29476) | 2026-06-28 预印本；A | sibling rollouts 的细粒度反事实信用已出现；与下面驾驶 CRAFT 是两篇不同论文，不能混用 |
| [Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic](https://research.google/pubs/q-prop-sample-efficient-policy-gradient-with-an-off-policy-critic/) | ICLR 2017；A | 以模型/critic 作为 control variate 降方差是成熟路线；新论文须说明超出已有控制变量估计之处 |
| [Adaptive Rollout Allocation for Online Reinforcement Learning with Verifiable Rewards（VIP）](https://arxiv.org/abs/2602.01601) | ICLR 2026，正式记录与 arXiv 均已核实；A | 固定预算下按预计梯度方差分配 rollout 已有方法；单独做自适应采样预算也不新 |

本次 toy 的 single-action replacement 是明确模型内的 difference-return baseline，不等于实现了 COCOA、COMA、Shapley 或 R-HCA。不能把这些论文的理论保证直接套到 toy 或 Plan-R1。

## 3. 驾驶规划和世界模型：2026 年的重叠更直接

| 论文与一手入口 | 日期/状态；级别 | 重叠判断 |
|---|---|---|
| [Plan-R1](https://arxiv.org/html/2505.17659v5) / [固定代码](https://github.com/XiaolongTang23/Plan-R1/tree/23bc54433599002f275c18b88cf261ab5e87d775) | ICLR 2026，v5=2026-06-02；F | 本来就有累计优势、TTC、reactive frozen predictor；原假设需要收缩 |
| [CarPlanner: Consistent Auto-regressive Trajectory Planning for Large-Scale Reinforcement Learning in Autonomous Driving](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_CarPlanner_Consistent_Auto-regressive_Trajectory_Planning_for_Large-Scale_Reinforcement_Learning_in_CVPR_2025_paper.html) | CVPR 2025；A | 自回归轨迹 RL 与一致性已有直接驾驶基线 |
| [CaRL: Learning Scalable Planning Policies with Simple Rewards](https://proceedings.mlr.press/v305/jaeger25a.html) | CoRL 2025；A | 大规模 RL 规划和简单 reward 的效果应纳入基线选择，不能只与 IL 比较 |
| [AD-R1: Closed-Loop Reinforcement Learning for End-to-End Autonomous Driving with Impartial World Models](https://arxiv.org/abs/2511.20325) | 2025-11 预印本；A | 已提出用更适合替代 ego 行为的世界模型支持驾驶 RL；不把其会议接收状态当作已核实事实 |
| [Counterfactual VLA: Self-Reflective Vision-Language-Action Model with Adaptive Reasoning](https://openaccess.thecvf.com/content/CVPR2026/html/Peng_Counterfactual_VLA_Self-Reflective_Vision-Language-Action_Model_with_Adaptive_Reasoning_CVPR_2026_paper.html) | CVPR 2026；A | 反事实驾驶推理和按需启用推理已有工作；不能以“驾驶+反事实”概括创新 |
| [CorrectionPlanner: Self-Correction Planner with Reinforcement Learning in Autonomous Driving](https://arxiv.org/abs/2603.15771) | 2026-03-16 预印本记录；A | motion-token propose/evaluate/correct 与 model-based RL 已被直接研究；第三方标 ICML 接收，本次未独立核实该接收，故只按预印本引用 |
| [RAD-2: Scaling Reinforcement Learning in a Generator-Discriminator Framework](https://arxiv.org/abs/2604.15308) | 2026-04-16 预印本；A | 摘要明确提出 Temporally Consistent GRPO 缓解时序信用分配；未逐式核查，不能断言与 VD/本项目相同，但必须精读比较 |
| [CRAFT: Counterfactual-to-Interactive Reinforcement Fine-Tuning for Driving Policies](https://arxiv.org/abs/2605.04470) / [作者代码](https://github.com/CurryChen77/CraftPolicy) | 2026-05-06 预印本；作者仓库称 NeurIPS 2026 接收，本次未在大会录单独核实；A | **高度相关**：反事实 proxy advantage 加真实闭环 residual correction；简单“反事实+校正模型偏差”已经不是空白 |
| [World Models as Adversaries: Multi-Agent Self-Play Fine-Tuning for Robust Motion Planning（AWM）](https://arxiv.org/html/2607.10630v1) | 2026-07-12 预印本；F（方法目录、归因和基准相关段落） | **直接沿 Plan-R1 参考规划器扩展**；角色条件对抗周车、稀疏联盟、反事实贡献、尾风险和 nominal retention。不能再推荐泛化的“Plan-R1+对抗 world model+CVaR”作为新问题 |
| [How Can Driving World Models Do Counterfactual Prediction?](https://arxiv.org/abs/2608.11601) | 2026-08-12 预印本；A | 已指出 action-conditioned prediction 与同一 factual episode 的反事实不等价；其研究设定为短时、替代 ego 不改变周车演化，不能外推所有 reactive 长时情形已解决 |
| [RISE: Adaptive Imagination for World Action Models](https://arxiv.org/abs/2608.20430) | 2026-08-20 预印本；A | 风险/规划收益驱动的 imagination 长度分配已有驾驶工作；“危险时多想一会”不足成创新 |
| [AD-WM: Action-Discriminative World Models for Counterfactual Model Predictive Control](https://arxiv.org/abs/2609.30264) | 2026-09-24，v2=09-25 预印本；A | 事实预测误差与动作选择能力的区分已有实证和方法；不能声称首次提出“低 ADE 不等于好决策” |
| [ExceptionDrive: A Planning-Oriented Counterfactual Corner-Case Benchmark for Autonomous Driving](https://arxiv.org/abs/2609.37871) | 2026-09-29 预印本；A | 反事实风险场景与参考无关安全指标已有资源；不能只造 corner-case 数据就宣称独立方法贡献 |
| [Risk-Aware World Model Predictive Control for Generalizable End-to-End Autonomous Driving](https://arxiv.org/abs/2602.23259) | 2026-02-26 预印本；A | 风险动作的世界模型学习与低风险选择已有工作；单纯增加 risk head 不够 |

AWM 的周车联盟归因与 ego token 的时序归因不是完全同一个问题；CRAFT 的 closed-loop residual 也不等同于本项目的 token baseline。但是它们已覆盖所需的多个基本组件。必须展示无法被这些组件直接组合得到的新机制或新保证，不能靠改名和迁移数据集证明创新。

## 4. 安全与不确定性的历史及近期边界

| 论文 | 核实级别 | 对潜在替代方向的限制 |
|---|---|---|
| [Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a.html)，ICML 2017 | A | 奖励外显式安全约束及策略更新保证早已有基础 |
| [Policy Gradient for Coherent Risk Measures](https://papers.neurips.cc/paper_files/paper/2015/hash/024d7f84fff11dd7e8d9c510137a2381-Abstract.html)，NeurIPS 2015 | A | coherent risk/CVaR 的 PG 不是新技术 |
| [Risk-Constrained Reinforcement Learning with Percentile Risk Criteria](https://jmlr.csail.mit.edu/papers/v18/15-636.html)，JMLR 2018 正式页面 | A | chance/CVaR 约束和 Lagrange 更新已有体系 |
| [PETS: Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://papers.nips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html)，NeurIPS 2018 | A | ensemble 和不确定性传播不能独立算新 |
| [MOPO: Model-based Offline Policy Optimization](https://arxiv.org/abs/2005.13239)，2020 | A | 用模型 uncertainty 惩罚 reward 已有理论与方法 |
| [Safe Planning in Interactive Environments via Iterative Policy Updates and Adversarially Robust Conformal Prediction](https://proceedings.mlr.press/v331/mirzaeedodangeh26a.html)，L4DC 2026 | A | 反应性环境中 policy-induced shift 与 robust conformal prediction 已被明确研究；“reactive+conformal”不能直接算创新 |
| [When Environments Shift: Safe Planning with Generative Priors and Robust Conformal Prediction](https://proceedings.mlr.press/v331/rahaman26a.html)，L4DC 2026 | A | distribution shift 下生成模型与风险覆盖已有方法 |
| [Conformal Policy Control](https://proceedings.mlr.press/v306/prinster26a.html)，ICML 2026 | S（一手正式搜索记录） | 在可接受风险下探索新 policy 的校准控制已有研究；需精读才能比较其保证与拟议方法 |

## 5. 哪些创新主张已经不能成立

| 潜在主张 | 查新结论 |
|---|---|
| 首次给早期 motion token 传播后续风险 | **不成立**：Plan-R1 本身就累计未来中心化奖励 |
| 首次用反事实区分责任动作和环境随机性 | **不成立**：COMA、Mesnard、COCOA、2026 Shapley 等 |
| 首次去除组方差归一化以保留安全信号 | **不成立**：Dr. GRPO/VD-GRPO 已覆盖 |
| 首次引入驾驶反事实优势和闭环纠偏 | **不成立**：驾驶 CRAFT 已直接涉及 |
| 首次将 Plan-R1 世界模型改成对抗者并优化尾部风险 | **不成立**：AWM 已直接涉及 |
| 首次按风险分配 rollout/推理预算 | **不成立**：VIP、RISE 等已涉及相关机制 |
| 首次用决策相关世界模型替代平均预测误差 | **不成立**：AD-WM 等已涉及 |
| 终止一致的 mixed outcome/process 无 critic 估计在所有相关领域均未被研究 | **未验证**：本次有具体源码证据，但修复可能只是标准 RL 正确实现；不应先行宣称新颖 |
| 固定预算下，对 planner 选中、会引发周车反应的动作分支，校准风险差并控制错误接受概率 | **有条件的候选问题**：尚未找到完整同构方案；相关部件高度拥挤，仍需对最近邻全文做逐式排除，不能写成“已证实空白” |

## 6. 查新判定

**原始信用分配命题不适合作为 CCF B/A 论文的立项依据。** 目前最可靠的贡献是源码审计、机制反例和可复现研究资产；它们还不是完整 AI 方法论文。

推荐保留两条短期工作：第一，把混合奖励和终止语义理清，建立有竞争力且目标一致的 RTG/RLOO 基线；第二，把主研究问题改为“反应性交通中的、受预算约束的风险差可靠性”。后者应以预测分支的可辨识性、选择后误差和错误安全排序为目标，而不只报 ADE/FDE 或笼统加 uncertainty loss。其创新资格取决于能否明确超出 CRAFT、AWM、AD-WM、RISE 和 robust conformal planning。

没有证实一个现成方案已经具备 CCF B/A 级独立创新。若最近邻的方法组合即可达到同样效果，应放弃对应主张，而非把证据缺口当创新空间。
