# 问题、方法与协议比较矩阵

## 方法矩阵

| 工作 | 表达/动作 | 主要学习信号 | 他车/世界建模 | 安全/纠错入口 | 对Root的关键边界 |
|---|---|---|---|---|---|
| MotionLM | 均匀位移token AR | NTP | 时间因果联合预测 | 预测一致性 | 时间因果结构≠可识别干预 |
| Trajeglish | k-disks token AR | NTP | 同刻顺序条件化 | 模拟分布拟合 | sequential agent代价；不优化ego安全 |
| SMART | 道路/动作token | 双NTP | 分解注意力模拟 | 扰动增强 | realism及泛化≠规划安全 |
| PDM | 中心线/IDM proposals | 规则score；Hybrid远期IL | 恒速预测 | 仿真打分选择 | 不换道；需地图/精确状态 |
| PlanTF | 向量轨迹回归 | IL；Adapter另用PPO | 辅助预测 | dropout、增强、执行adapter | 已研究计划—执行差距 |
| PLUTO | 横纵query多模态 | IL+辅助+对比 | 单模态他车 | 代价图loss与规则选择 | 全候选失败不能靠选择补救 |
| Diffusion Planner | 连续联合轨迹 | 去噪目标 | 联合未来生成 | 可微能量、可选refinement | 引导为软偏置；需可微代价 |
| CarPlanner | 连续Gaussian AR | PPO+专家DE | 一次性非反应预测 | 质量奖励+规则选择 | 不是离散token；专家依赖强 |
| Gen-Drive | 多agent动作扩散 | 偏好模型+DDPO | 联合场景 | 规则筛选/GPT标签/评估器 | 奖励利用；非反应测试 |
| GameFormer | GMM联合轨迹 | IL+interaction loss | level-k未来交互 | 参考路投影/可选优化 | 层级推理不是均衡证明 |
| Think2Drive | steer/throttle/brake | latent actor-critic | RSSM | 想象rollout奖励 | CARLA privileged控制协议不同 |
| Dr.GRPO | 文本token | 可验证奖励RL | 无驾驶WM | 修正std/长度权重 | 去std已有；可变长与固定horizon不同 |
| CAT-K | SMART token / 单独GMM ego | recovery CE/NLL | 策略top-K rollout | 接近GT的闭环恢复 | GT不安全或模态不可达时需核查 |
| SMART-R1 | SMART token | MPO realism+SFT | 多agent模拟 | 阈值优势+交替训练 | SFT–RFT–SFT已有；不等于安全planner |
| PlannerRFT | DiT连续轨迹 | 探索PPO+生成GRPO | nuMax执行仿真 | guidance+survival奖励 | 全同reward与权重问题需分别消融 |
| **Plan-R1** | **离散motion AR** | **规则奖励+VD-GRPO** | **冻结NTP他车模型** | **安全门控与可选refinement** | **经验安全、模型反应与执行误差仍需分解** |

每行的原文位置和资源条件在对应paper_notes，机器可读版见methods.csv。此矩阵主要是机制归纳，不声称已统一复现。

## 可以/不可以怎样比较

| 维度 | 必须固定/分列的条件 | 本批发现 |
|---|---|---|
| 测试集 | Val14、Reduced-Val14、Test14-random、hard具体场景ID和过滤 | “Val14”名称相同不保证场景相同；PLUTO原表与Root表数值不同 |
| 反应方式 | replay NR、IDM R、学习型WM、全agent可控 | Root训练WM不是nuPlan R评测背景模型 |
| 后处理 | 无后处理、候选选择、轨迹优化修正分别列 | Root表1 *、PLUTO原文†和Diffusion表1 *语义不同，不能跨文照抄符号 |
| 执行 | LQR/iLQR、重规划频率、车辆动力学、计划/执行轨迹 | Gen-Drive为1Hz+iLQR；Root评测10Hz循环；不能仅凭score作严格结论 |
| 成本 | 训练数据、额外偏好、环境步、候选数、采样器、GPU和推理延迟 | PlannerRFT v1 8×H100/40M环境步；Root附D 8×4090；未做等算力比较 |
| WOSAC | 原始2023、后续2023修订、2024、2025 | Trajeglish旧表0.5339、SMART 2024和SMART-R1 2025不是一条连续可比较数轴 |
| 指标目标 | forecasting mAP、simulation realism、ego CLS、NAVSIM PDMS | 预测更准、模拟更真、planner更安全是不同命题 |
| 统计 | 种子、置信区间、场景配对与多目标拆分 | 本批不把未报告方差的单次差值解释为统计显著 |

## Root视角的直接数值锚点

仅用于定位证据；不重新拼接一个跨文“排行榜”。

| 原文位置 | 同一来源内可核比较 | 可支持 / 不能支持 |
|---|---|---|
| Root 表2 | GRPO NR/R=88.65/88.35；VD=91.23/90.04 | 作者设置下VD有收益；不能证明任意高方差组都更危险 |
| Root 表3 | GT replay R=87.44；Reactive WM R=90.04 | 该消融中WM有益；不能证明干预因果正确 |
| Root 附B表6 | interPlan不带后处理56.64，PLUTO57.74；Root*72.33 | 避免把带后处理SOTA推广到所有无后处理设置 |
| CAT-K 表3 | GMM ego collision 0.0568→CAT-3 0.0422 | 监督闭环恢复已有安全收益；不是nuPlan上的VD组合证据 |
| PlannerRFT v1 表2 | 固定DDIM下R 68.18→72.21 | 采样/探索收益有专门消融；不是相对Root的胜出证据 |
| SMART-R1 表3/4 | 分阶段消融与MPO/GRPO比较 | 支持其设置下后训练组合；不是通用RL算法优劣排序 |

PlannerRFT v1表1没有Plan-R1一行，且其hard R=72.21低于Root表1报告的77.20。**只能说跨文报告值如此，不能说完成了公平对照，更不能照搬PlannerRFT的SOTA主张。**
