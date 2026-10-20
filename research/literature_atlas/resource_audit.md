# 复现资源入口静态核查

日期2026-10-20；只读README及源代码，没有执行安装/训练、下载checkpoint或数据，没有联系作者。“代码存在”“作者公布权重入口”“取得并运行权重”是不同状态，本批未达到最后一项。

以下SHA是GitHub返回的README **blob SHA**，不是仓库commit SHA；可用仓库Git blob API追溯当时README，不能拿它当整仓库版本号。

| 工作 | README / blob SHA | 已核到的资源状态 |
|---|---|---|
| SMART | [README](https://github.com/rainmaker22/SMART/blob/main/README.md) / `e172d9b54ffd01c8e069df6fe46068adf80ce5cd` | 训练/评测/预处理代码与配置；预训练参数节为将来发布非Waymo模型的承诺，不能声称现成WOMD权重已公开 |
| PDM | [README](https://github.com/autonomousvision/tuplan_garage/blob/main/README.md) / `732d6e0a5c3def0321d7c3ff0ffad9bc84d2d9d1` | 规则planner、训练代码与模型Drive链接；未验证下载与各个模型文件 |
| PlanTF | [README](https://github.com/jchengai/planTF/blob/main/README.md) / `a92714ae38ed0a928c7a77d73fb5ec17e38535c1` | 预处理/训练/评测及Trained models表；README报告batch32约4–6GB/卡，只是作者设置，不是本轮实测 |
| PLUTO | [README](https://github.com/jchengai/pluto/blob/main/README.md) / `5aa6d88aa448ea4a73f0f9cf1b03763a08e44794` | training code与checkpoint勾选完成；未验证checkpoint内容 |
| Diffusion Planner | [README](https://github.com/ZhengYinan-AIR/Diffusion-Planner/blob/main/README.md) / `1978dbd9355cffd848b207261da0bf0050488943` | 训练/评测/guidance教程及Hugging Face模型入口；配送车数据项仍未完成，注明审批中 |
| CAT-K | [README](https://github.com/NVlabs/catk/blob/main/README.md) / `d3450f86ae4eeaa0cfd6ec06fd9e804bb985a2d6` | 完整训练/评测配置；checkpoint要求WOMD资格截图并邮件申请；不能把代码公开等同权重匿名可取 |

其余论文资源条件逐卡注明：MotionLM、CarPlanner、SMART-R1与PlannerRFT的官方可运行训练发布未核实；Trajeglish项目标TBA，Gen-Drive项目标Coming Soon；Think2Drive项目主要是论文/视频，未核实完整训练发布。Dr.GRPO有官方代码/模型入口，但本轮没有重现其LLM训练。

Root资源与当前代码路径见root_audit。所有数据可用性还要按nuPlan/WOMD等各自许可及版本核对。本页是可复现性排程依据，不宣称作者资产实际缺失或无法申请。
