# Plan-R1 源码审计与信用分配假设检验

审计日：2026-10-20（Asia/Shanghai）。审计对象是作者仓库 `XiaolongTang23/Plan-R1` 的 main：`23bc54433599002f275c18b88cf261ab5e87d775`，提交时间 2026-07-26。现有 `/workspace/Plan-R1` 的 remote 指向一个 fork，但 HEAD 与经 GitHub API 核实的作者 main 一致，核心文件无本地修改。下文行号均固定到此 SHA，不能套用到任意后续版本。

论文入口：[ICLR 正式记录](https://proceedings.iclr.cc/paper_files/paper/2026/hash/28a80012e6f564c0cc8e7661e1db83fe-Abstract-Conference.html)、[arXiv v5](https://arxiv.org/html/2505.17659v5)。OpenReview PDF 直读遇到浏览器验证，方法和补充材料通过 arXiv v5 核对；不声称逐字比对了两个版本。

## 1. 结论先行与证据分级

**不支持“Plan-R1 没有把后续风险反馈给早期动作”。** 当前代码使用未折扣的累计中心化奖励，首次碰撞那一步仍参与训练。早期动作可因后续碰撞获得负优势，TTC 项还提供碰撞前的局部风险信号。

**也不能说累计优势已充分解决时序信用分配。** 它解决的是回传路径，不保证低方差、因果辨别、目标一致性、反事实有效性或真实闭环安全。最具体的待研究问题是：有限样本组均值对梯度的时间依赖影响；随机终止下的中心化与有效 token 归一化；全轨迹 outcome 广播和终止掩码的组合；确定性周车模型的风险覆盖。

| 证据层次 | 本次确认的内容 | 不应推出的结论 |
|---|---|---|
| 源码事实 | 奖励、掩码、优势、采样、损失的实际运算 | 真实 nuPlan 性能会下降多少 |
| 真实方法的受控执行 | AST 提取原函数；执行真实 progress/speed 类；断言及精确枚举通过 | 完整模型 rollout、碰撞几何和训练均已复现 |
| 连续运动 toy | 不同估计量的反馈、方差及模型误差敏感性 | Plan-R1 在真实基准存在已证实缺陷 |
| 研究推断 | 值得做有控制的目标一致性与世界模型检验 | 已找到独立且可发表的新方法 |

## 2. 从配置到训练的调用链

固定版本的 [model/PlanR1.py](https://github.com/XiaolongTang23/Plan-R1/blob/23bc54433599002f275c18b88cf261ab5e87d775/model/PlanR1.py) 是核心文件，以下简称 `M`。

| 文件、函数、行号 | 实现含义 |
|---|---|
| `train.py:18,25–31,37–47` | 固定 seed=1024；按配置加载预训练 checkpoint；Lightning 启动训练 |
| `config/train/plan.yaml:3–7,50–63` | 1024 motion tokens；80 个 0.1s 未来采样点按 interval=5 分组，得到 16 个 0.5s token；G=4，ego top-k=50，β=0.1，c=0.1；奖励权重 2/5/2/1 |
| `M:50–57,282–305`, `transition` | token 查表得到局部 Δx、Δy、Δheading，再转换到全局坐标，追加到联合历史；离散采样和几何转移，无可微环境反传 |
| `M:275–280`, `on_train_start` | 将 predictor 参数复制到 planner，并冻结 predictor |
| `M:307–348`, `rollout` | 同场景复制 G 次；逐 token 生成所有 agent；计算奖励、优势并返回 ego log probability |
| `M:145–175`, `training_step` | policy gradient 加 KL 正则 |

这里的“语言建模”是 motion token 自回归建模，不是调用外部语言大模型。没有在本次运行中下载 checkpoint、启动完整训练或读取大型数据集。

## 3. 实际 GRPO / VD-GRPO，而不是通用公式

### 3.1 组和时间如何聚合

`M:309–314` 的 batch 顺序是先全部场景，再复制整批 G−1 次。`compute_ae_process_supervision`（`M:361–388`）将 `[G×场景数,T]` reshape 成 `[场景数,G,T]`。

设场景 s 的第 g 条 rollout 在时间 t 的奖励为 r，m 为有效掩码，则实际中心为：

\[
\mu_s=\frac{\sum_{g=1}^{G}\sum_{t=1}^{T}m_{sgt}r_{sgt}}
                   {\sum_{g=1}^{G}\sum_{t=1}^{T}m_{sgt}}.
\]

**它在同场景的全部有效 rollout×时间点上求一个均值，不是每个时间点各求一个均值，也不是先求每条轨迹总回报再组归一化。** 不同前缀下的 rollout 共享场景基线，因此它不是状态条件的 V(s_t)。

### 3.2 当前启用的是 VD-GRPO

`M:376`：

\[
z_{sgt}=m_{sgt}(r_{sgt}-\mu_s)/c,\qquad c=0.1.
\]

普通过程监督 GRPO 的方差计算在 `M:371–373` **被注释**：

\[
\sigma_s=\sqrt{\max\left(\frac{\sum mr^2}{\sum m}-\mu_s^2,10^{-4}\right)},
\qquad z^{GRPO}=m(r-\mu_s)/(\sigma_s+10^{-4}).
\]

恢复该代码才是同一过程监督实现下的 GRPO 对照；当前没有配置开关直接切换。`compute_ae_outcome_supervision`（`M:350–359`）是另一个未被 rollout 调用的方法：它对标量 outcome 做组均值/标准差归一化，再广播优势到全部未来 token。不能把这个未启用函数当作主训练算法。

VD 的贡献是保留不同场景奖励差异的绝对尺度，**不意味着它新增了时序回传，也不意味着安全具有硬约束保证**。更换 c 同时改变 RL 与 β·KL 的相对强度；对含 KL 的整体训练，不能简单说 c 不影响优化结果。

### 3.3 Return-to-go、优势与终止

`M:380–386` 反向递推：

\[
\hat A_{sgt}=z_{sgt}+m_{sg,t+1}\hat A_{sg,t+1}.
\]

连续有效前缀下，令 τ 为第一次 done 的时间（没有 done 时 τ=T），则

\[
\hat A_{sgt}=\frac{1}{c}\sum_{u=t}^{\tau}(r_{sgu}-\mu_s).
\]

这里 γ=1，没有 critic、TD bootstrap、λ-return 或 GAE。称作“优势”是作者用语，具体是累计中心化奖励，不是已证明精确的 Q−V。

### 3.4 实际损失不是完整 PPO 多轮更新

`M:162–168`：

\[
\rho_t=\exp(\ell_t-\operatorname{stopgrad}(\ell_t)),\quad
L_{PG}=-\frac{\sum m_t\rho_t\hat A_t}{\sum m_t}.
\]

ρ 的前向值总是 1，但导数为 ∇ℓ，因此 policy gradient 不为零。这里没有另存旧策略、PPO clip、独立 rollout buffer 或在同一数据上多轮 importance-ratio 更新。它是单次 on-policy 风格的 score-function 更新。真实 `training_step` 受控执行已验证梯度非零。

KL 使用

\[
K_t=e^{\ell^{ref}_t-\ell_t}-(\ell^{ref}_t-\ell_t)-1,\quad L=L_{PG}+\beta\frac{\sum m_tK_t}{\sum m_t}.
\]

在动作确实按完整 π 采样的条件下，该样本表达式的期望为 KL(π‖π_ref)。但本代码 ego 按 top-50 截断分布采样（`M:331`），记录的却是完整 categorical 分布的 log probability（`M:333–340`）。`utils/process_data.py:147–159` 先将其他 logits 置 −∞ 再重归一化；没有看到截断采样校正。因而不能直接声称这里得到完整 π 下无偏 PG/KL。这是另一个独立估计问题，影响大小本次没有实测。

## 4. Motion Token 的奖励到底是什么

核心 `reward_fn`（`M:390–415`）使用：

\[
r_{gt}=I^{road}_{gt}I^{obs}_{gt}I^{agent}_{gt}
\frac{2C_{gt}+5TTC_{gt}+2S_g+P_g}{10}.
\]

权重取实际 train YAML；构造函数默认 speed/progress 权重是 4/2，不能与本配置混用。

| 文件、函数、行号（均为同一固定 SHA） | 粒度与数学解释 |
|---|---|
| `rewards/agent_collision_reward.py`, `forward:18–92` | 每未来 token 的 ego/agent 盒子角点几何；85–90 生成 0/1 reward 和 collision done；不是显式 −大常数碰撞罚分 |
| `rewards/obstacle_collision_reward.py`, `forward:20–72` | 每步用 ego 角点与静态边界关系得到 0/1 门控和 done |
| `rewards/on_road_reward.py`, `forward:20–72` | 可行驶区边界门控；不满足时 done |
| `rewards/comfort_reward.py`, `forward:17–53` | 对位置差分得到加速度/角速度/角加速度；超过阈值即该 token 的 comfort=0 |
| `rewards/ttc_reward.py`, `forward:19–92` | 当前差分速度外推 0.95s 后检查盒子碰撞，得到每步 0/1 风险信号；不是严格求连续时间最小 TTC |
| `rewards/speed_limit_reward.py`, `forward:13–61` | 每步速度超限量沿整条轨迹累加，归一化后得到一个标量 S_g∈[0,1] |
| `rewards/progress_reward.py`, `forward:16–73` | 将位移投影到路线、全时间累加；与 expert progress 的阈值归一化得到 P_g，expert 用于归一化而非轨迹模仿损失 |
| `M:404–407` | S_g、P_g 原样复制到每一个未来 token |

这些检查主要发生在离散 0.5s token 端点/外推位置上，不能将“full bounding box”理解成已验证了连续 swept-volume 安全。是否造成真实错误仍需独立测试。

## 5. 早期动作如何接收后续风险：能接收，但不等于能区分原因

done 为三种安全违规的 OR（`M:395`）；`M:397–398` 将未终止掩码的累乘右移一位：

\[
m_{gt}=\prod_{u<t}(1-d_{gu}).
\]

所以第一次违规 t=τ 的 reward=0 **仍参与基线、优势和 policy loss**；τ 后的 token 被屏蔽。在 μ_s>0 的组里，该步贡献 −μ_s/c，累加到此前动作。不存在“因为 done，碰撞当步整个被掩掉”的问题。

受控原函数例：G=4,T=4，第一条奖励 `[1,1,0,0]`、掩码 `[1,1,1,0]`，其余三条全 1。μ=14/15，c=.1，失败轨迹优势约为 `[-8,-8.6667,-9.3333,0]`，安全轨迹第一步为 +2.6667。详见 `results/source_audit.json`。

但是某次采样中的 A<0 不是“该动作造成事故”的因果判定。它混合了后续 ego 决策、周车模型的响应、初始场景难度以及 group 估计噪声。标准 return-to-go 对可加因果奖励的 PG 可以在期望上正确，无须每条事故只惩罚某个唯一责任动作；因此“所有前序动作都收到风险”本身不是算法错误。

若同组全部奖励都为零，则 reward-gradient 为零（KL 仍可作用）；VD 无法凭空生成更优对照。若四条轨迹时间奖励曲线完全相同 `[1,1,1,0]`，第一步累计中心化优势为零，而后续步仍可能有时间中心化信号；不能笼统称作整个组的所有优势都零。

## 6. 累计优势仍留下哪些真实、可检验的问题

### 6.1 包含自身 rollout 的组中心并非动作独立基线

在固定 T、全有效、因果可加奖励、独立 rollout、按完整 π 采样的简化条件下，令 S_gt=∇logπ(a_gt|h_gt)，n_t=T−t+1、G_gt=Σ_{u≥t}r_gu，有：

\[
E[S_{gt}\hat A_{gt}]
=\frac1c\left(1-\frac{n_t}{GT}\right)E[S_{gt}G_{gt}].
\]

推导：同一 rollout 的总奖励进入 μ 的系数是 1/(GT)，与 S_t 的协方差等于其未来回报协方差；其他 rollout 的 score 期望为零。故减去 n_t μ 会移除一部分本来的梯度。G=4 时，第一步系数 .75，最后一步约 .9844；一般不能只调一个全局学习率抵消。

对 H=16、仅第一步 Bernoulli 动作决定终点奖励的例子，枚举全部 2^4 组样本，用原优势函数精确得到期望梯度 .1875，真实梯度 .25。**这是受限条件下的估计差异证明，不是 Plan-R1 真训练梯度偏差大小的估计。** RLOO、独立样本基线是必须加入的便宜对照；基线去偏不是新概念。

随机终止时，μ 分母和每条剩余有效长度也依赖动作；`M:163` 又按 batch 的有效 token 总数归一化。上述固定系数不再成立，不能把固定时长推导机械外推。即便 μ 是外部固定均值，减去每个有效步的 living reward 也可能改变不同生存长度间的偏好。这些效应需要先明确要优化“每场景总回报”还是“每有效 token 平均回报”。

### 6.2 轨迹 outcome 广播改变时间权重

全有效且只考虑标量 q_g 广播时，原实现该项的优势为：

\[
A_{gt}^{out}=(T-t+1)(q_g-\bar q)/c.
\]

若 outcome 只作为终点一次奖励，权重则是 1，而不是 T−t+1。前者是一个有时间权重的训练代理；不能假设与优化一次性 outcome 等价。反过来，广播让“即时 token 奖励完全看不到未来信息”的说法也不成立，因为即时 reward 本身已包含未来整轨迹指标。

### 6.3 终止后的轨迹仍影响有效前缀的 outcome 分数

`rollout` 在生成全部 T 步后才算 done；`ProgressReward` 和 `SpeedLimitReward` 对整条生成轨迹计算，未接收上述 done mask，再广播到有效前缀。这意味着 τ 后已屏蔽 token 的位置，仍可能通过 P_g/S_g 改变 τ 前的有效奖励；后缀动作自己的 PG 却被屏蔽。

本次实际执行这两个原始 reward 类，在一条直线路线 fixture 上只改首次失败之后的状态，确认 progress 和有效前缀 soft reward 随之改变（具体值见 audit JSON）。这是明确的计算性质。它是否违背作者预期取决于终止语义：若训练目标是“第一次违规即结束的驾驶 episode”，这应修正；若 outcome 有意评价完整候选轨迹，则不应再默认上述终止式 PG 对完整 outcome 目标一致。不能在未声明目标时简单给它贴“bug”标签。

### 6.4 安全奖励与世界模型是另外两个瓶颈

安全门控将单步奖励清零，不等于概率约束 `P(collision)≤δ`，也不区分违规严重度。VD 保留尺度不能证明低概率灾难必然优先于所有效率目标。将其改成约束优化或 CVaR 也已有大量先例，不能直接当创新。

全局奖励差或模型预测“每个分支都不安全”时，换估计公式不会自动增加可用安全动作或修复世界模型。因此应把奖励目标、估计方差、采样覆盖和模型误差分别消融。

## 7. 世界模型实际怎样参与训练

`freeze_pred_model`（`M:262–273`）将 predictor 设 eval 并冻结参数。它有两个角色：为非 ego agent 预测动作；为 ego 选中的动作提供参考 log probability。Planner 和 predictor 先看到同一个联合历史，再各自输出该步预测；ego 动作覆盖 predictor 的 ego 输出（`M:342`），`transition` 更新联合历史。其他 agent 可在后续步骤响应 ego 已生成的历史，**不是当前同一 token 内先看 ego 新动作再响应**。

周车在 `M:330` 使用 top-k=1，在通常无并列 logits 时是确定性 argmax；ego 在 top-50 内随机采样。世界模型没有在线参数学习、ensemble、不确定性校准或随机多分支周车响应；也没有外部物理 simulator 参与此训练循环。优化靠 ego log probability 乘 scalar advantage，不反传穿过周车采样和几何规则。

这允许“对预测模型响应的优化”，不自动等价于识别真实人的干预响应。论文 v5 已报告 ego-state 扰动下预测误差测试，所以不能声称作者完全没研究 world-model robustness；但这类测试不是同场景真实干预下的风险排序校准证明。

## 8. 下一步真实验证的最小清单

先不训练：从小量合法场景缓存记录每个 token 的 reward 分量、done、mask、μ、A，以及同前缀候选动作在独立 reactive simulator 中的结果。分别测 prefix feedback、梯度方差、action-ranking error、分支风险召回。

然后以原 VD、恢复普通过程 GRPO、RTG+RLOO、仅修正 outcome/termination 的版本为基线，匹配 KL 强度、token 预算、模型调用数、场景分布和有效 rollout 数。全轨迹 outcome 应明确只记一次或作有证明的重分配。若只修改这些基础估计约定就消除差异，应作为复现修正，不应包装成新的因果规划方法。

所有真实基准验证均属未来工作。本次未得出 Plan-R1 的 nuPlan、interPlan 或真实车辆性能缺陷结论。
