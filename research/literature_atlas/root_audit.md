# Root Paper：机制、版本与代码核对

本页补充[R00卡片](paper_notes/R00.md)。分析对象是[ICLR proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/28a80012e6f564c0cc8e7661e1db83fe-Paper-Conference.pdf)，印刷页码比PDF零起始page index大1。早期arXiv v1/v2不可替代最终VD-GRPO版本；[arXiv版本页](https://arxiv.org/abs/2505.17659)列出不同修订。未声称所有版本逐字一致。

## 核心主张的证据边界

| 命题 | 原文锚点 | 解释边界 |
|---|---|---|
| 去std保留跨组尺度 | §3.3 式5；§4.3表2；附E | 是奖励梯度的相对权重机制；不能仅由高方差推出高风险 |
| 交互世界模型有用 | §3.3 Dual-Model；表3 | 冻结日志预测器有效的经验消融，不是干预正确性证书 |
| 原则对齐提高规划质量 | §4表1–2 | 无后处理与*后处理分别成立于对应protocol，不泛化为全场景胜出 |
| 长尾鲁棒性 | 附B表6 interPlan | 属扰动评测；无后处理不是该表最强方法，见comparison |
| 理论分析 | 附E.2–E.4 | 分析隐含重加权和奖励混合方差；不是碰撞约束满足证明 |

**分析判断：**这里的“safe/feasible”主要由任务奖励和经验评估定义。乘法安全项置零不会强制策略只在安全集内采样；一组轨迹全同奖时，中心化优势仍可为零。R00的改动解决尺度失真，不自动解决支持集缺失或无差异反馈。

## 论文与当前公开代码

只读核对仓库main `23bc54433599002f275c18b88cf261ab5e87d775`，无模型执行或代码改动。

| 位置 | 实际核到的行为 | 应如何理解 |
|---|---|---|
| [PlanR1.py L346](https://github.com/LFC-CS-AI/Plan-R1/blob/23bc54433599002f275c18b88cf261ab5e87d775/model/PlanR1.py#L346) | rollout调用process supervision优势函数 | 不能把文件内另一个outcome/std函数误认作当前训练路径 |
| [PlanR1.py L361](https://github.com/LFC-CS-AI/Plan-R1/blob/23bc54433599002f275c18b88cf261ab5e87d775/model/PlanR1.py#L361) | 按scenario合并sample与time有效项求均值；除固定`scaling_factor`；反向累计有效奖励 | 复现时必须记录均值归约轴与valid mask，不能仅写“去掉std” |
| [plan.yaml L48](https://github.com/LFC-CS-AI/Plan-R1/blob/23bc54433599002f275c18b88cf261ab5e87d775/config/train/plan.yaml#L48) | num_samples=4；beta=0.1；scaling_factor=0.1 | 与论文超参数对应；不意味着训练资源已验证 |
| [PlanR1.py L162](https://github.com/LFC-CS-AI/Plan-R1/blob/23bc54433599002f275c18b88cf261ab5e87d775/model/PlanR1.py#L162) | ratio来自`plan_logps - plan_logps.detach()` | 前向ratio=1而梯度非零；单凭静态表达式不能判为“没有policy gradient”，也不能当成多轮old-policy clipped PPO |

这段ratio形式及未clip surrogate与Root式4接近。若后续采用“标准GRPO”对照，需要明确rollout策略、旧策略快照、更新次数、mask归约和KL估计器，而不是仅按算法名称比较。

## 有待复核的原文问题

1. **固定缩放方向。**正文式5与代码均除以c；因此固定其他量时，增大c应减小RL项梯度尺度。附B印刷p16附近却有“larger c amplifies”方向表述。以公式/代码解释当前实现，保留文字不一致，未擅改原代码。
2. **归约轴。**附E式9文本提取为沿F时间的均值/方差，而正文描述为组内归一化、代码为有效sample×time归约。理论符号可能有简化或排版差异；本环境PDF截图工具不可用，须人工看原PDF及询问作者后再作勘误结论。
3. **高方差与危险的对应。**附E.3显式考虑非平凡unsafe混合；全零组没有该区分信号。可以提出组方差诊断，但不能把“越危险必定方差越高”当定理。
4. **统计与控制器。**本批未核到足以比较所有方法种子方差的统一报告。训练WM、评测IDM和执行LQR各有差异，若后续有收益，需分别归因。

以上是后续审计事项，不是新算法或已被实验验证的缺陷结论。本文不重启连续时间安全或一维Bridge Head路线。
