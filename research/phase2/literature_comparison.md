# 第二阶段：最近邻精读与覆盖边界

日期与检索截止：2026-10-20（Asia/Shanghai）。沿用第一阶段的文献入口，不重做广泛查新。下面六篇已读方法与相关理论章节；CRAFT/AWM/AD-WM/RISE 使用所列版本的 arXiv 全文 HTML，两篇 conformal 工作下载 PMLR 正式 PDF 并阅读方法、假设、证明和实验。PDF 仅放工作缓存，不提交副本；来源与哈希见 `literature_sources.json`。

## 1. 问题、算法、目标、实验、保证：总览

| 工作 | 实际研究对象与核心算法 | 优化目标/数据 | 实验设置 | 理论覆盖及限制 |
|---|---|---|---|---|
| [CRAFT](https://arxiv.org/html/2605.04470v1) | 在真实访问状态上用候选轨迹 proxy advantage，加实际执行事件的 residual correction，再用 EMA teacher 约束 policy | 真实闭环期望回报；候选 counterfactual world 的组相对分数与执行后的碰撞/规则事件 | CARLA/Bench2Drive 的 HiP-AD、MindDrive、SparseDriveV2；正文及附录说明协议修订 | 梯度精确分解是恒等式；方差改善要求 proxy 与真 advantage 有足够协方差；实现使用近似 residual，误差界取决于未知 residual 误差；没有逐候选风险排序置信保证 |
| [AWM](https://arxiv.org/html/2607.10630v1) | Plan-R1 式 motion-token 自博弈；学习周车攻击角色/稀疏联盟；冻结对手后训练 ego | 参考相对 regret 的尾部风险、normal retention 和安全 dual surrogate | nuPlan、InterPlan、long-tail 模板；planner-world 交叉测试 | 有局部 frozen-opponent 近似误差、贡献梯度方向和条件方差分析；保证显式保留未知 support coverage gap；其 pair calibrator 是攻击增益概率，不是碰撞风险校准 |
| [AD-WM](https://arxiv.org/html/2609.30264v2) | residual latent dynamics + 预测转移上的逆动作恢复/normalized recovery；测试时仍用 CEM MPC | factual latent prediction、表征正则和 action recovery；offline 图像/动作数据 | OGBench Cube/Scene、Reacher、TwoRoom、PushT；V-JEPA2+DROID 后 Franka 迁移 | 恢复目标有 MI lower-bound 动机；没有稀有碰撞概率的有限样本保证；**已经研究全候选排名与被选中 elites 的不一致** |
| [RISE](https://arxiv.org/html/2608.20430v1) | learned Risk Profile、Future Planning Gain、逐前缀 Roll/Stop gate；CounterDrive 风险配对监督 | 预测误差、轨迹监督、风险排序 margin、规划收益和计算成本 | nuScenes、NAVSIM v1/v2；多个 WAM 架构的迁移 | 是已实现的预算/风险/决策收益方法；没有在已读正文和算法中给出有限样本错误安全接受率控制定理；其 risk calibration 表述不能替代统计覆盖证明 |
| [Interactive robust conformal planning](https://proceedings.mlr.press/v331/mirzaeedodangeh26a.html) | episode 内校准环境轨迹 tube；跨策略改变用响应敏感性扩张 tube；隐式或闭式半径更新 | chance-constrained、episodic open-loop planning；当前策略下新收集的环境轨迹 | 2-D car–pedestrian、quadcopter | **直接解决反应性环境破坏普通 CP exchangeability 的问题的一类情形**；依赖动力学 Lipschitz、初始安全半径、可行 robust planner；显式收敛还需 planner regularity/收缩 |
| [Conformal Policy Control（CPC）](https://proceedings.mlr.press/v306/prinster26a.html) | clipped likelihood-ratio policy family、带最坏测试点权重的校准、采样控制 | 用已知安全 policy 的有损失标签数据，控制新 policy 的 population expected bounded loss | 医学问答的 gCRC、受约束 active learning、序列/生物分子优化；不是驾驶排名基准 | **已经处理校准数据依赖的 policy 选择与反馈 shift**；定理含初始安全、互相绝对连续、准确权重和 relative replace-one stability 等条件，非任意场景/任意未知 OOD 保证 |

## 2. CRAFT：不是已经测准所有分支，而是用实际反馈修正代理

精读位置：§3.1–3.4、Appendix A、C、D.3–D.5。式(3)将真实 policy gradient 写为 proxy Φ 与 Δ=A_real−Φ 两项。在相同真实访问状态分布上成立，不需要假设 proxy 无偏；但是实现以稀疏纠正事件近似 Δ，故恒等分解不代表实际训练是精确无偏的。

Proposition 3.2 的 residual variance 等于 `Var(Y)+a²Var(C)−2aCov(Y,C)`；a=1 的改善要满足 `Cov(Y,C)>.5Var(C)`。Theorem 3.3 用 score 二阶矩与 residual approximation 的 L2 误差乘积界定梯度误差，不是从有限 rollout 数直接得到的安全概率证书。最终 loss（式14）包含 proxy、dual-clipped correction、双向非对称 teacher KL。

附录 D.3 的虚拟 ego 用 PID tracker/kinematic bicycle；周车是短窗保留当前动作后衰减 throttle、增加 braking 的模型。这给“proxy 可能错、需要 grounded correction”一个具体机制。实验中的近邻结论是：不能把我们的小型全局 residual offset 当作 CRAFT 复现，更不能把其效果不好说成 CRAFT 已失败；我们只借它检验最便宜的已知 proxy-correction 思路。

**覆盖判断：**已覆盖“反事实分支代理有误差、真实反馈纠偏有价值”；未直接给固定调用预算、对被接受动作的碰撞风险上界。但研究新 residual 修正前必须超出该框架。

## 3. AWM：模型内对抗风险与真实风险校准是不同对象

精读位置：§2–§4、Appendix C.1–C.7。内层角色条件世界模型改变周车行为，LOO/team contributions 分配联盟训练信用；外层优化参考相对 regret 的两分支 CVaR，并约束 nominal behavior。不是对每条普通 ego 候选输出统计可信的碰撞概率。

Proposition C.2 的局部误差界包含 `ε_A + δ_cov + 2Λ δ_ε`；Corollary C.3 还加外层优化误差 ε_B。`δ_cov` 正是结构化对手没覆盖完整最坏对手的差距，定理没有把它自动变成已知小数。C.6 将 CVaR 写成 tail-mass τ 的变分表达式；这不意味着分支的经验权重已经等于真实交通的发生概率。

§3.2/C.3 的 calibrator 预测“加第二个攻击者是否增加攻击收益”；其最优阈值还涉及正/负增益幅度条件。把它叫作 calibration 不能据此宣称物理碰撞风险或 selection 后错误率已经校准。C.4 的归因保证也要求 counterfactual baseline 对事实 attacker sample 的动作独立性。

**覆盖判断：**已覆盖 Plan-R1 的对抗响应、branch regret 与尾风险优化，故“加对抗周车+CVaR”不是新方向。真实风险排序的统计可靠性不等于此目标，但任何提出的新保证须明确是否依赖同样的 support gap。

## 4. AD-WM：第一阶段低估了它与“选择后诊断”的重叠

精读位置：§III.A–E、IV.A–E、V。预测器估计 latent increment，动作恢复头作用于预测转移端点；MI 目标实际是归一化动作重建加 Gaussian KL 正则。训练后去掉辅助头，不改 MPC。

它不仅提出“低 factual error 不够”。§III.E 已定义共享候选库上的 Spearman CAD，以及 predicted elite 在实际执行后的 best-in-elite/elite-mean normalized regret（式6–8）。CEM 的 300 candidates、30 elites、30 iterations 等受控设置，使该比较直接面对 selection。不能再声称首次把排名度量从全库移到被选中动作。

论文的主要控制比较是 OGBench/机器人，不是反应性交通的碰撞概率；作者也限定诊断在 Cube 的证据范围。其工作没有宣称每个选中动作都满足 chance constraint。我们采用它的 top-1 elite-regret 定义作轻量诊断，**没有训练它的视觉模型，未复现其控制成绩**。

**覆盖判断：**selection-facing 评估动机已经直接覆盖。若只有新 toy 上的相同观察，不足以成独立论文。

## 5. RISE：风险配对与计算分配都已经是算法的一部分

精读位置：§4.1–4.4、§5、Appendix 10/14。RISE 的预算单位是 imagined prefix depth。Evaluator 同时预测已有风险 R_h 与继续 rollout 的规划收益 B_h；Gate 根据收益、累计成本和额外成本决定 Roll/Stop。

式13–14 用 factual/counterfactual 风险排序 margin 与 incident-localization 监督；式18 的 B_h 目标是相对当前深度的实际 planning-score 改善；式19–20 用 `max_j[(q_j−q_h)−λ(c_j−c_h)]>0` 监督 gate。因此“按风险/未来收益多想几步”和“配对风险排序 loss”均不可当新贡献。

它的深度 gate 不等于在固定动作库上对随机风险差作 sequential testing。我们本轮固定全 horizon、改变 Monte Carlo 分支数，没有把一个按样本数停止的 heuristic 冒充 RISE 复现；也没有训练/下载其大型 encoder、diffusion planner 或 CounterDrive 数据。

**覆盖判断：**系统级预算路由与风险排序监督高度覆盖；未在所读理论部分找到可直接用于本研究的 finite-sample false-safe certificate。缺少同名定理不证明存在新颖空间。

## 6. Interactive robust conformal planning：已解决一类真正反应性漂移

精读正式 PDF：p.3–4 系统与 CP tube，p.6–8 Lemma1/2、式3.3–3.5、Theorem1–3，p.9–11 实验，Appendix A/B/C 证明与算法。

系统允许 `y_(t+1)=fY(y_t,x_t,u_t,ν_t)`，明确依赖 ego。用全轨迹 nonconformity `max_t ||y_t−ŷ_t||` 校准 tube。策略更新使周车分布改变，不能直接沿用旧 CP 半径；Lemma2 以响应增益 β_T 控制 trajectory drift。关键要求为：

```text
r_(j+1) >= q_j + β_T ||π*(r_(j+1)) − π_j||∞
```

在 planner 半径敏感性 L_U 已知且 κ=β_T L_U<1 时，有显式 shrink/expand 更新。Theorem1 是校准集外层概率至少1−δ_j下，下一 episode 的覆盖/安全概率至少1−α，且安全部分要求 robust planning 可行；Theorem3 的收敛还依赖分位函数收缩及误差控制。

不能再把“策略改变对手行为而破坏 exchangeability”当作未被处理的问题。另一方面，其 global sensitivity、feasibility、持续重采样条件具有成本；本轮 reference 有离散 yield latch，尚未提供全局 Lipschitz 证明，不满足条件的失败不反驳该论文。后续真正的新方法至少要在相近假设下改进保守性/样本预算，而非忽略已有保证。

## 7. CPC：不能把“选择后可靠性”一概列为空白

精读正式 PDF：§2、4.1/4.2、Theorem4.2/4.6、Appendix C.1/C.2。候选 policy family（式5）为：

\[
\pi_t^{(\beta)}(a|x)\propto\min(\pi_t(a|x),\beta\pi_0(a|x)).
\]

式6 从安全侧向激进侧搜索 β，要求直到该点的所有候选都满足带测试点最坏权重的 empirical risk 条件。实际实现以过往 policy mixture 和经验权重上界作近似，须与 exact permutation weights 下的理论区分。

Theorem4.6 的结果是：

\[
E[L_t]\le\alpha+\max(\alpha,B-\alpha)\,\epsilon^{(1)}.
\]

条件包括初始安全期望、bounded loss、policy 间互相绝对连续、校准解存在，以及 relative replace-one stability。**不是无附加项的任意 α 保证**。Theorem4.2 的非单调 loss 结果同样含 `Kε` 项。

在驾驶里可把 collision 定义为 bounded loss，ego 选择引发周车响应可纳入 action 的条件损失，只要必要分布/likelihood/数据条件成立。纯 emergency-stop reference 没有所有动作的支持，不能直接满足绝对连续条件；加入有界概率的全支持微混合可能解决支持问题，但还需证明安全期望与稳定性。隐藏驾驶人分布任意改变则不能不加假设地继承定理。

**覆盖判断：**CPC 已直接覆盖数据依赖的 policy 选择和部分反馈 shift 的风险控制。我们的“conditional on accepted”错误安全比例与它控制的 population expected collision 不是同一 estimand，这种不同必须先数学定义，不能只改术语称作新问题。

## 8. 对本阶段的具体约束

本轮不声称复现六篇完整系统。轻量实验实现普通 ensemble、exact-binomial 筛选、paired risk difference、全局 grounded residual、marginal/max-over-actions split calibration、worst-member 筛选；后二者分别是已有 proxy correction/robust aggregation 思路的最低成本对照，并非 CRAFT/AWM 原算法。AD-WM 的 elite-regret 公式在当前 action bank 上直接实例化。RISE、CPC 与 interactive robust CP 没有满足原论文数据/动力学/策略假设的完整实现，不能用本轮结果宣称击败它们。

固定5个候选的 selection effect 已可用全向量 simultaneous coverage 常规处理，不需要新方法。需要更多分支才能区分风险差本来就是 Monte Carlo/statistical decision 的问题。未知行为漂移下所有模型一致犯错也不构成新算法自动可解的问题。是否值得继续，应结合实验中简单方法的安全-进度折中和这些覆盖边界判断。
