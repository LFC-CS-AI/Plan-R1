# 技术发展时间线

首次公开年和正式发表年分开记录；年内顺序不代表继承。以方法关系为结论时必须回到relations.csv的证据。此表不是完整领域历史。

| 首次公开 | 正式发表/当前状态 | 工作 | 在Root问题空间中的位置 |
|---|---|---|---|
| 2023 | ICCV 2023 (formal) | [MotionLM](paper_notes/P01.md) | motion_token / autoregressive / joint_prediction / temporal_causality |
| 2023 | CoRL 2023; PMLR 229:1268–1281 (formal) | [PDM](paper_notes/P04.md) | rule_based / proposal_selection / benchmark / open_closed_gap |
| 2023 | ICCV 2023 (formal) | [GameFormer](paper_notes/P10.md) | interactive_prediction / level_k / IL / refinement |
| 2023 | ICLR 2024 (formal) | [Trajeglish](paper_notes/P02.md) | motion_token / k_disks / autoregressive / intra_step / traffic_simulation |
| 2023 | ICRA 2024 (formal) | [PlanTF](paper_notes/P05.md) | IL / state_dropout / augmentation / controller_gap / correction |
| 2024 | NeurIPS 2024 (formal) | [SMART](paper_notes/P03.md) | motion_token / map_token / factorized_attention / traffic_simulation / generalization |
| 2024 | arXiv:2404.14327 (preprint) | [PLUTO](paper_notes/P06.md) | IL / contrastive_learning / auxiliary_safety_loss / proposal_selection |
| 2024 | ECCV 2024 (formal) | [Think2Drive](paper_notes/P11.md) | world_model / latent_imagination / RL / control |
| 2024 | ICRA 2025 (formal) | [Gen-Drive](paper_notes/P09.md) | diffusion / preference_reward / RL / alignment / joint_generation |
| 2024 | CVPR 2025 (formal) | [CAT-K / CLSFT](paper_notes/P13.md) | motion_token / IL / closed_loop_SFT / recovery / covariate_shift |
| 2025 | ICLR 2025 (formal) | [Diffusion Planner](paper_notes/P07.md) | diffusion / IL / joint_generation / guidance / safety_alignment |
| 2025 | CVPR 2025 (formal) | [CarPlanner](paper_notes/P08.md) | autoregressive / RL / consistent_mode / world_model / selection |
| 2025 | COLM 2025 (formal) | [Dr.GRPO](paper_notes/P12.md) | RL / GRPO / normalization / length_bias |
| 2025 | ICLR 2026 (formal) | [SMART-R1](paper_notes/P14.md) | motion_token / RL / MPO / closed_loop_SFT / distribution_alignment |
| 2025 | ICLR 2026 (formal) | [Plan-R1](paper_notes/R00.md) | motion_token / autoregressive / IL / RL / rule_reward / world_model / normalization |
| 2026 | CVPR 2026 (formal) | [PlannerRFT](paper_notes/P15.md) | diffusion / RL / adaptive_exploration / survival_reward / controller_gap |

## 三条有证据的演化支线

1. **从行为分布到原则对齐。**MotionLM/Trajeglish提供motion NTP与联合分解；SMART进一步强调结构化地图与模拟效率；Root将行为先验与ego规则RL分开。前两者不是安全planner，不能把任务变化描述为一个排行榜不断进步。
2. **从扰动增强到策略访问状态。**PlanTF/PLUTO提供IL增强与训练信号；CAT-K用policy top-K构造恢复监督；SMART-R1明确组合CAT-K与RFT。不同组件之间存在引用和使用证据，但不是所有早期工作都被逐一技术继承。
3. **从连续生成到可控探索。**Gen-Drive采用偏好评价与去噪RL；Diffusion Planner展示可微行为引导；PlannerRFT以Diffusion Planner为基础联结探索policy、生成器更新和survival奖励。Root与它们竞争的是安全/交互/评测问题，不是使用相同动作参数化。

Dr.GRPO在2025先提出去std等偏差修正，Root正式§2.2明确承认启发。2026 NoRD（L2线索）也将其用于驾驶。因此“从GRPO去std用于驾驶”已不能作为待主张的新贡献。

**时间先后不证明继承：**Trajeglish将MotionLM称为同期工作；SMART-R1和PlannerRFT引用Root2025preprint，均不自动意味着实现最终VD-GRPO。X01的书目写ICLR2026，也不能证明其作者读过哪个最终PDF。
