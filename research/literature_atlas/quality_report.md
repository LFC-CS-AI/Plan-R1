# 逐项质量核查与完成报告

## 数量与范围

核心精读 **15篇 + Root 1篇**，均为L3核心章节与关键实验核查；额外NoRD仅L2线索，不计配额。核心15篇按当前书目为 **14正式 : 1预印本**；Root正式。使用公开预印本作技术阅读来源不改变该工作已经正式发表的书目状态，也不表示预印本等同最终版。

引用图共有 **42条已核原文CITES**：12条Root backward、3条forward（P14/P15引用2025preprint，L2线索X01以ICLR2026引用Root）、其余27条选中文献间引用。不是42篇论文，也不是全部forward文献。技术图70节点、237边；包括6条author-explicit EXTENDS、7条COMPETES_WITH（其中P15→Root为推断）、5条假说互补边。分类和LIMITED_BY不伪装成作者承认的事实。

## 每篇质量门

“核心段”表示核对Introduction/Related Work、Method及Experiments的相关原文段落；不是宣称逐字读完每行附录。“限制”区分明确限制、正文限定与没有独立限制节。所有论文12项字段均已填；未知用文字标明，不用猜测补齐。

| ID | 核心段与关键证据 | 限制核查 | 正式/技术版本差异或仍缺内容 |
|---|---|---|---|
| [R00](paper_notes/R00.md) | §1–5;References;Ethics;Reproducibility;附B/C/D/E（选读）;表1–6 | Ethics Statement（印刷p10）：仅仿真评估。正文未设置独立Limitations节；不得据此声称无局限。 | ICLR proceedings PDF; arXiv v3 only as historical cross-check |
| [P01](paper_notes/P01.md) | §1–5核心段;表1–4;Fig6;附A/B/D仅索引定位 | §3脚注明确不声称识别真实因果；§3.1承认误差累积、隐藏混杂；§5把规划应用与低延迟蒸馏留作未来。 | arXiv v1 (2023-09-28); venue verified separately |
| [P02](paper_notes/P02.md) | §1–4;§3.1–3.2;表1;Fig6/8–12;附录索引核查 | §4.3/Fig11指出WOMD数据受限、超参数尚待系统研究；Fig12中LoRA迁移表达能力不足。 | ICLR-hosted PDF (watermark Preprint); old metric in Table1 |
| [P03](paper_notes/P03.md) | §1–5核心段;Limitations;表1–4;附A仅索引定位 | Limitations：简单tokenizer、数据规模有限（约100M模型）；粒度/词表消融及规划迁移未充分验证。 | arXiv v3; proceedings metadata title uses Motion Generation variant |
| [P04](paper_notes/P04.md) | §1–4;Limitations;表1;Fig2;References | 印刷p8 Limitations：不做换道；单nuPlan仿真、未系统测分布转移；replay过激、IDM reactive过于被动。 | PMLR proceedings PDF |
| [P05](paper_notes/P05.md) | §I–V;表I–VI;项目Limitations;Fig3/5 | 官方项目Limitations：精细操作和主动换道仍弱，开环/闭环差距；§III-C把Adapter与planner联合训练留作未来。 | arXiv v1 PDF; title singular Planner; published/project title plural Planners |
| [P06](paper_notes/P06.md) | §I–VI;Limitations;表I/II;§V-C消融 | §VI Limitations：单模态他车预测；全体候选不可用时后处理无能为力。 | arXiv v1; no primary formal venue verified |
| [P07](paper_notes/P07.md) | §1–6核心段;附E;表1/2;§5.2消融;附B/C仅索引定位 | 附E：结构化输入信息损失；大横移不足；planner与controller差距。 | arXiv v1; ICLR publication separately verified |
| [P08](paper_notes/P08.md) | §1–5核心段;§3.2–3.4;表1–4;附录仅索引定位 | §4.2指出reactive弱于PDM，归因未与IDM交互训练；§3.2明确用非reactive转移模型作效率取舍。 | arXiv v2 (2025-03-05); CVF proceedings verified |
| [P09](paper_notes/P09.md) | §I–V;§III-D/E;§IV-A/C;表I–III | §IV-C承认长时间RL可能利用奖励导致退化；§IV-B承认PDM总体分数更高。 | arXiv v1 (2024-10-08); ICRA verified by author institution |
| [P10](paper_notes/P10.md) | §1–5核心段;表2–5;附A.4/A.5仅索引定位 | §4.2.3明确非反应测试；§4.2.4因测试服务器资源缩小网络。未定位独立作者Limitations节。 | arXiv v2 (2023-08-11) |
| [P11](paper_notes/P11.md) | §1–4;§3.1–3.3;表2/3;§4.4 | §4.4承认仅raw sensor的TCP学生与privileged expert有明显差距；未定位独立Limitations节。 | ECVA proceedings PDF |
| [P12](paper_notes/P12.md) | v1/v2 §1–4核心段;v2 §3.2;表2/3;Fig5–7;附A仅索引定位;COLM venue | §3明确β=0分析范围；没有驾驶实验。未定位独立Limitations节，不把应用范围当成作者承认的所有局限。 | arXiv v2 (2025-10-06); COLM accepted list and final PDF header verified |
| [P13](paper_notes/P13.md) | §1–6核心段;算法1;表1–3;附B/C仅索引定位 | §4.2明确依赖GT模态覆盖；§6把top-p、软化winner-takes-all及其他policy类列为未来。 | arXiv v2 (2025-03-14); CVF metadata verified |
| [P14](paper_notes/P14.md) | §1–5;References;附A.1;表1–6 | §4.3：全验证评估昂贵，消融用2%子集；§3.4明确RFT分布漂移风险；未见独立Limitations节。 | arXiv v2 (2026-02-28); official final PDF access challenge; formal status verified |
| [P15](paper_notes/P15.md) | §1–6核心段;附A讨论;表1–5;References[48];附B/F仅索引定位 | §6：仅结构化输入验证，视觉运动规划尚未验证；§5.3承认只用失败场景会遗忘常规行为。 | arXiv v1 (2026-01-19); CVF final metadata indexed; final PDF 403 |

## 本批主要技术分支

motion token与联合AR；IL的捷径/增强/闭环恢复；规则与偏好RL对齐；交互世界模型；扩散引导与有效探索；执行一致性。按问题的树见taxonomy，方法/假设矩阵见comparison及methods.csv。

优先扩展：①NoRD等驾驶难度偏差/支持集；②干预有效的交通模型；③后训练遗忘与恢复监督；④计划执行差距；⑤tokenizer及多模态覆盖。每个方向的反证门槛在gaps.md，五个组合记录在combinations.md；都未进入实现。

## 信息缺失与判断保留

- 正式PDF未全部取得：P14 OpenReview challenge，P15 CVF403；P15作者名单与v1不同。其他固定arXiv阅读版本也未逐字比对最终排版。
- Gen-Drive的ICRA状态据作者机构记录；PLUTO尚未核实正式发表。不能用arXiv DOI当作会议录用证据。
- Root附E归约符号、附B的c方向文字，以及Think2Drive训练路线数差异保留；PDF截图不可用，不能把抽取歧义宣布为作者错误。
- 资源按已见原文报告；未核实代码/权重的论文明确留空。对“有checkpoint入口”的工作也没有下载、校验或运行。
- 没有统一场景清单、种子/方差/算力对齐，因此不重新计算跨论文胜率，不声称统计显著或复现SOTA。
- Gen-Drive微调数据与评测split的独立性需要专门核对。当前只记录原文措辞，不直接宣判泄漏。
- Forward检索有召回和时间滞后限制。2026后续线索不代表穷尽最新工作；未核论文不填虚假卡片。
- 未系统覆盖形式安全约束的第二层文献；本批safe reward、guidance与refinement不等同CBF/可达性等保证。

## 已执行的质量检查

运行 `python research/literature_atlas/validate.py` 检查：核心数量/Root独立性；ID和关系端点；CSV列完整性；所需节点/关系类型；所有CITES与references双向对应；EXTENDS证据状态；无悬空链接；每卡12字段；变更限定在research/literature_atlas。另用 `git diff --check` 检查文本格式。

这些检查验证知识库结构与证据可追溯性，不替代人的论文批判阅读，也不冒充模型测试。没有训练或复跑论文实验；没有修改Plan-R1原代码。
