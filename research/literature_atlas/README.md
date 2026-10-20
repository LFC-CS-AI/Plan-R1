# Plan-R1 Literature Atlas · 第一阶段

以 [Plan-R1 ICLR 2026 正式论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/28a80012e6f564c0cc8e7661e1db83fe-Abstract-Conference.html) 为唯一研究根，建立可扩展、可复核的选题证据库。核查截止 **2026-10-20（Asia/Shanghai）**。本阶段不提出已经成立的新算法，不运行此前连续时间安全或一维 Bridge Head 实验。

## 本批状态

- Root `R00` 单列，另有 `P01–P15` **15篇核心章节精读卡片**；均超出摘要阅读，核对方法、实验及关键表格，附录选择性阅读。不是逐行审计全部附录，也未复现作者数值。
- 按当前已核书目：**14篇正式会议论文、1篇仅确认预印本（PLUTO），比例14:1（93.3%:6.7%）**。Root另计，为正式论文。正式发表状态与实际所读全文版本分列；很多正式论文读的是作者公开的固定版本预印本。
- **42条原文引用关系**，其中12条Root backward、3条指向Root的forward；其余为选中文献间扩展。P14/P15引用的是Plan-R1 **2025预印本**；补充L2线索NoRD以ICLR2026引用Root。不能由书目断言引用者读过哪个最终PDF。该集合不是完整citation index。
- **70个有类型节点、237条关系**。`EXTENDS` 6条注明作者直接说明的组件/方法继承；`COMPETES_WITH` 7条区分已报告比较与机制/协议重叠推断；`POTENTIALLY_COMPLEMENTS` 5条均为假说。
- 已完成逐卡字段检查及CSV/图谱一致性检查；版本、资源和统计不确定性见 [quality_report.md](quality_report.md)。没有因未知字段强行推断会议、代码、GPU需求或结果。

## 阅读入口

| ID | 工作 | 发表状态 | 阅读重点 |
|---|---|---|---|
| [R00](paper_notes/R00.md) | Plan-R1 | ICLR 2026 | VD-GRPO、冻结WM、奖励及控制器边界 |
| [P01](paper_notes/P01.md) | MotionLM | ICCV 2023 | 时间因果联合预测 |
| [P02](paper_notes/P02.md) | Trajeglish | ICLR 2024 | k-disks、同刻agent依赖 |
| [P03](paper_notes/P03.md) | SMART | NeurIPS 2024 | 地图token、分解注意力、泛化 |
| [P04](paper_notes/P04.md) | PDM | CoRL 2023 | OLS/CLS不一致、规则强基线 |
| [P05](paper_notes/P05.md) | PlanTF | ICRA 2024 | 捷径、增强、执行差距与RL Adapter |
| [P06](paper_notes/P06.md) | PLUTO | 预印本 | 对比IL、辅助安全loss、候选选择 |
| [P07](paper_notes/P07.md) | Diffusion Planner | ICLR 2025 | 联合去噪、可微引导、refinement |
| [P08](paper_notes/P08.md) | CarPlanner | CVPR 2025 | 一致mode、连续AR、专家奖励 |
| [P09](paper_notes/P09.md) | Gen-Drive | ICRA 2025 | AI辅助偏好、DDPO、奖励利用 |
| [P10](paper_notes/P10.md) | GameFormer | ICCV 2023 | level-k交互及优化后处理 |
| [P11](paper_notes/P11.md) | Think2Drive | ECCV 2024 | 潜空间世界模型与控制RL |
| [P12](paper_notes/P12.md) | Dr.GRPO | COLM 2025 | std/长度偏差及已有贡献边界 |
| [P13](paper_notes/P13.md) | CAT-K / CLSFT | CVPR 2025 | 闭环监督恢复和GT模态有效性 |
| [P14](paper_notes/P14.md) | SMART-R1 | ICLR 2026 | MPO及SFT–RFT–SFT |
| [P15](paper_notes/P15.md) | PlannerRFT | CVPR 2026 | 场景适配探索、survival奖励 |

## 文件与语义

| 文件 | 用途 |
|---|---|
| [papers.csv](papers.csv) | 16条完整主记录，`quota=0` 为Root，12项内容同步到卡片 |
| [leads.csv](leads.csv) | X01 NoRD部分阅读线索；不计15篇，不计正式/预印本核心比例 |
| [paper_notes/](paper_notes/) | 元信息、研究问题到复现条件的逐篇卡片 |
| [references.csv](references.csv) | 仅原文核实的 `CITES`，方向为引用者→被引者，附版本/位置 |
| [nodes.csv](nodes.csv) | Paper / Problem / Method / Assumption / Limitation / Benchmark节点注册表 |
| [methods.csv](methods.csv) | 多标签问题、方法、假设、局限、benchmark映射及奖励/推理接口 |
| [relations.csv](relations.csv) | 全部技术边；证据状态、URL、位置和理由必填 |
| [taxonomy.md](taxonomy.md) | 按问题组织的分类树和关系图 |
| [timeline.md](timeline.md) | 首次公开与正式发表分别记录的时间线 |
| [comparison.md](comparison.md) | 方法/问题/安全机制比较矩阵及评测协议边界 |
| [root_audit.md](root_audit.md) | 正式Root与当前代码的静态证据审计 |
| [resource_audit.md](resource_audit.md) | README blob与代码/权重/数据发布状态分开核对 |
| [gaps.md](gaps.md) | 待验证空缺、反证与扩展优先级 |
| [combinations.md](combinations.md) | 每个组合的A/B功能、现有工作、机制、强基线及可证伪设计 |
| [sources.md](sources.md) | 原始文献、书目证据、版本冲突、检索记录和待扩展线索 |
| [quality_report.md](quality_report.md) | 逐篇核查矩阵、完成数量、缺失与不确定性 |
| [validate.py](validate.py) | 仅检查知识库结构，不执行Plan-R1或任何训练 |

### 证据规则

`L0`=仅书目；`L1`=摘要；`L2`=部分正文；`L3`=核心章节与关键实验表格逐节核对、附录选读；`L4`=另有复现核验。本批最高L3，没有L4。待扩展线索不冒充L3，不占配额。

`verified_citation` 是原文有引用；`author_explicit` 是作者明确说使用/改造某方法；`verified_comparison` 是有具体比较表；`inferred` 是分析判断；`hypothesis` 是待验证互补。不把时间先后等同继承。`USES` 指向Assumption时表示依赖该假设，指向Benchmark时表示采用该协议；并非使用了另一个算法。

`LIMITED_BY` 边默认是分析归纳，不能一律归给作者。作者明确限制保存在每张卡的单独字段中。未定位Limitations节也会明说。分类标签表达机制/问题重叠，不表达已获得的普遍优势。

### 扩展与维护

从Root出发，最多两层。先读正式Root的References；再读被选关键工作的Related Work/References。CAT-K的已核两层路径是 `R00 ← P14 → P13`，也可从SMART的后续工作发现。竞争工作按问题/方法/协议重叠纳入，不能补造引用边。

新增文献先进入sources待办，核对主标题、作者和版本；完成正文/关键表检查后再分配正式Paper ID。保留旧版本的表号和数值，不能无痕替换。新增断言要同时更新卡片、CSV和关系证据，再运行 `python research/literature_atlas/validate.py`。

## 仓库边界

分支：`research/planr1-literature-atlas`；base为main `23bc54433599002f275c18b88cf261ab5e87d775`，不合并此前PR。本次仅新增本目录；原训练、模型及仿真代码不改。没有训练、下载nuPlan数据或权重，也不提交版权PDF。代码复用判断是静态接口分析，不能解释成可运行性认证。
