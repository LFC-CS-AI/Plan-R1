# 官方权重与真实输入核验

核验日：2026-10-20。用户已确认没有现成 checkpoint/真实特征，本轮继续受约束组件验证。

## 访问、大小与许可边界

[作者仓库](https://github.com/XiaolongTang23/Plan-R1) README 明确给出[官方 Drive 文件夹](https://drive.google.com/drive/folders/1I8wPrpLAeKFS7x7fpQOwunE7qtDJN9hm)。通过 **Google Drive 插件**列目录、查元数据并实际请求 fine-tuning 文件；不是猜测下载地址。

| 文件 | 字节数 | 本轮实际状态 |
|---|---:|---|
| fine-tuning.ckpt | 81,364,059 | 连接器完整取回；转存执行器失败：`file exceeds the executor transfer limit of 32 MiB` |
| pre-training.ckpt | 81,310,520 | 元数据已核实，未下载 |

[fine-tuning 官方入口](https://drive.google.com/file/d/1Imnrju-5vwz1kzUMENv5s38iHMIMOH4W/view)，[pre-training 官方入口](https://drive.google.com/file/d/10ceLBFefi5C-2h11Fh0_imZ4Yw6zzpyR/view)。没有把临时签名链接或权重提交到 Git。连接器 `can_download=true`；这证明本次连接器能获取，不能证明每个匿名客户端都可直连。代码仓库 Apache-2.0；README 邀请研究者下载权重。文件夹只返回上述两个文件，未找到独立权重许可，不能从代码许可证推导权重再分发权利。此处是来源/访问核验，不是法律意见或授权扩张。

**没有加载本地 checkpoint，没有任何真实预训练输出，也未生成随机权重输出。** 执行器未取得文件，因此文件内部 tensor/hparams、哈希和严格加载兼容性尚未核实；元数据没有提供可核验的哈希。`results/checkpoint_preflight.json` 保留已核实项目和未完成项。

## 代码可推理性与资源

本环境实际成功导入原始 `PlanR1`，在 meta device 仅构建结构：默认配置 **10,072,576 参数**，FP32 参数存储下界 **40,290,304 字节（38.42 MiB）**。见 `results/architecture_probe.log`。原代码预测/规划两个 backbone 均包含在此计数，checkpoint 含何种训练器状态未检查。

`PlanR1.plan_inference` 本身没有强制 CUDA；CPU Python3.9、PyTorch2.1.0+cpu、PyG2.3.1、torch-scatter2.1.2 的依赖导入成功。meta 探针原始测量含导入 RSS 281,572 KiB；它不是完整推理峰值。小 batch CPU RAM 4–8 GiB 只是安排资源的估计；**无法报告已测得的最小推理内存、延迟或真实成功率**。图规模（agent/map 数量）会显著影响内存。无需先下载全部 nuPlan 才能做一条预处理特征推理。

## 还需要的具体文件

1. 可由执行器读取的完整官方 `fine-tuning.ckpt`（81,364,059 字节），或可信本地文件位置；本轮不需要两个权重都提供。
2. 至少一个原始 `NuplanDataset.get` 风格 `.pt` processed dict / `HeteroData`，包含下面的真实历史与地图。**仅 token 序列、截图、终点坐标或 agent 状态表不够。** 文件须附场景标识、来源/处理版本、时间步和坐标系说明。未来 ground-truth 不是最小开放环生成的必要输入；独立真实风险验证需要另提供未来轨迹/仿真入口。
3. 为审查实际执行桥，还需要控制器/轨迹重建设置与参考点定义；若特征中无可靠初始加速度，可由真实 10Hz 历史估计并记录误差。一个场景只能打通推理，不能建立独立测试集结论。

原始输入（精确源码范围/哈希见 `results/source_audit.json`）：

- `agent`：`position[N,>=21,2]`、`heading[N,>=21]`、`velocity[N,>=21,2]`、`visible_mask[N,>=21]`、`type[N]`、`identity[N]`、`box[N,4]`、`num_nodes`。历史 2 秒、10Hz，含索引20当前帧；ego 排首位。ego 用后轴参考点，box 按前/后/左/右距离排列；周车通常中心参考点。
- `polygon`：`position`、`heading`、`type`、`speed_limit`、`speed_limit_valid_mask`、`heading_valid_mask`、`traffic_light`、`on_route_mask`、`num_nodes`。
- `polyline`：`position`、`heading`、`length`、`num_nodes`。
- `('polyline','polygon')`：`polyline_to_polygon_edge_index`；`('polygon','polygon')`：`left_edge_index`、`right_edge_index`、`incoming_edge_index`、`outgoing_edge_index`。Batch 构造生成 `batch/ptr`；请优先给未经手工删字段的官方 processed 文件。

`TokenBuilder` 从历史重建四个 0.5s token，原 `plan_inference` 返回16个未来端点（0.5–8s），并不包含当前点。`real_inference.py` 显式拼回当前点，保存17个位姿、所有 agent token/box 和来源哈希。它使用严格 checkpoint 加载，缺任一输入则报错，不降级为随机权重：

```bash
python research/phase4/real_inference.py --checkpoint /path/fine-tuning.ckpt \
  --features /path/real_processed.pt --out /path/NEW_real_outputs
```

该真实推理执行路径本轮**尚未端到端执行**。仅 `--architecture-only` 已运行，不能将脚本存在写成真实推理成功。
