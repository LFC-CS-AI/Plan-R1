# Plan-R1 连续时间安全：源码核查

日期：2026-10-20（Asia/Shanghai）。基线提交：`23bc54433599002f275c18b88cf261ab5e87d775`。本阶段在独立worktree、独立分支基于main开展，不修改前两阶段或原始实现。机器可核查的函数行号与原文件SHA256见 `results/source_audit.json`。

**结论：训练车辆碰撞奖励确实只在0.5秒端点求值；部署的refinement路径已有0.1秒几何评分。token本身没有定义唯一的区间运动，不能把某种插值下的漏检直接说成真实闭环漏检。**

## 1. Token是什么，0.5秒来自哪里

| 源码/函数 | 行号 | 确认事实 |
|---|---:|---|
| `config/train/plan.yaml` | 3–7 | 1024词表、interval=5、历史20帧、未来80帧 |
| `transforms/token_builder.py:TokenBuilder.__call__` | 33–42 | 原始步长按`0.1*interval`使用，位置/朝向每5帧取一次 |
| 同函数 | 55–74 | 局部相对位移/朝向匹配平均角点距离最近token，然后从重建状态滚动匹配 |
| 同函数 | 78–86 | infer历史有4个端点，最后一个为当前时刻；未来从第0.5秒开始 |
| `model/PlanR1.py:transition` | 282–305 | 每次只增加一个位置、朝向和mask，没有速度/控制量/中间轨迹 |
| `utils/process_data.py:transform_point_to_global_coordinate` | 87–91 | 使用起点朝向把局部位移旋转到世界系 |

词表是字典：Vehicle/Pedestrian/Bicycle各`[1024,3]`，每行是`(Δx,Δy,Δψ)`。原文件38682字节，SHA256为`72ad21c4c7530c48c6a7fd47355487e1b65b217ceda2db3bb0ee133466934213`。原始定义是端点转移：

\[
p_{k+1}=p_k+R(\psi_k)[\Delta x_z,\Delta y_z]^\top,\quad
\psi_{k+1}=\operatorname{wrap}(\psi_k+\Delta\psi_z),\quad T=0.5s.
\]

本次从AST提取**未改动的原`transition`函数体**，以真实词表、HeteroData直接执行，再与诊断解码比较，通过2e-6容差检查。这只验证解码，**没有加载完整模型或生成真实策略输出**。

## 2. 车辆碰撞奖励的时间与几何判据

`rewards/agent_collision_reward.py:AgentCollisionReward.forward` 的20–28行取`infer_position[:,4:]`并逐时刻生成角点；36–61行只在同一个T索引匹配ego和agent；64–81行求角点进入另一矩形的深度；85–90行返回每个端点的0/1奖励与done。没有跨`k−1,k`边、插值、扫掠体或区间最小距离。

忽略阈值为0的严格接触边界，该实现大致是：

\[
D_k=1\{\exists v\in\operatorname{corners}(B^e_k):v\in\operatorname{int}B^o_k
\ \text{or reverse}\}.
\]

它既不同于完整矩形相交`B^e_k∩B^o_k≠∅`，也不同于区间事件：

\[
C_k=1\{\exists\tau\in[0,T]:B^e(kT+\tau)\cap B^o(kT+\tau)\ne\varnothing\}.
\]

本次原奖励单测确认：两辆4.4m×1.8m矩形在中心相同、朝向相差90°时形成十字交叉，SAT/Shapely相交，原角点判据返回无碰撞。**这是空间判据缺口，不应算作时间漏检。** 主实验用SAT端点对照隔离二者，并且排除起点已碰撞样本后再统计时间漏检。

## 3. 静态障碍、TTC及组合奖励没有自动补上区间

`rewards/obstacle_collision_reward.py:forward`（18–32、45–70行）同样从infer端点生成ego角点，再按静态障碍边界最近边/节点的损失判断；没有区间扫掠。本次未建立带nuPlan静态边界拓扑的完整回放，**不能给出该类真实漏检率**；跨越小障碍的机制推断还需单独验证。

`rewards/ttc_reward.py:TTCReward.forward`（21–24、68–90行）计算：

\[
\bar v_k=(p_k-p_{k-1})/0.5,\quad p^{ttc}_k=p_k+0.95\bar v_k.
\]

然后在这个外推时刻以当前朝向执行同样角点判据。它不是对`[0,0.95]`求最小碰撞时间，不能保证捕获已经在`(k−1,k)`发生又消失的重叠。主诊断的83个纯时间漏检，原始TTC函数也均未惩罚；这是这些构造的一步函数结果，不代表完整场景TTC永远无效。

`model/PlanR1.py:reward_fn`（390–415行）将agent/obstacle/on-road三个端点安全奖励相乘，TTC只在加权软奖励内，done来自前三项。安全标签未检测到的事件不会仅因为后续学习而自动变为正确标签。本阶段不重做优势/信用分配研究。

## 4. 重建、部署、refinement和外部评估必须分开

| 路径 | 时间/几何语义 | 对结论的限制 |
|---|---|---|
| `simulation/planr1_planner.py:compute_planner_trajectory` 28–29、63–79；`global_trajectory_to_states` 90–101 | token未来8秒、0.5秒端点，补当前ego后返回`InterpolatedTrajectory` | 轨迹对象支持任意时间查询，但这不是执行了CCD |
| nuPlan `interpolated_trajectory.py` 32–48、66–79；`geometry/compute.py:AngularInterpolator` 151起 | position等线性状态用`interp1d`；角度先unwrap后插值 | 本诊断线性SE(2)桥与这种**位姿插值语义**相符，但不等于车辆跟踪器实际运动 |
| `simulation/planr1_planner_with_refinement.py:_get_closed_loop_trajectory` 172–205 | 8秒重采81点，加入PDM/STR候选，模拟并评分 | 原系统已有廉价高频检查/候选重排序路径，必须作为强基线 |
| `config/planner/planr1_planner_with_refinement.yaml` 19–31 | 输出0.1秒、80姿态；评分proposal为0.1秒、40姿态（4秒） | 不能把这一路写成仅0.5秒检测 |
| `tuplan_garage/.../scoring/pdm_scorer.py:_calculate_no_at_fault_collision` 328–343 | 按proposal时间索引用polygon `intersects`；包含归责分类 | 完整多边形端点几何，非原角点判据；仍不是解析CCD |
| nuPlan `no_ego_at_fault_collisions.py` 254–259 | 遍历`history.data`逐时间戳调用`find_new_collisions` | 实际频率取决于scenario/time-controller配置；没有真实日志不能指定统一频率或推断其成绩有漏检 |

特别是 `utils/process_data.py:interpolate_traj_using_midpoints`（161–174行）确实存在，但搜索只有定义和export，没有在奖励/规划路径找到调用；不能因函数名存在就推断训练已经补了中点。

ego位置是**后轴参考点**，`datasets/nuplan_utils.py:get_features` 261–270行使用非对称front/rear box；周车用中心及对称半长（282–289行）。本次几何显式支持这些偏置，没有把ego后轴误当车体中心。初步合成ego取[3.95,1.05,1,1]，并非某一真实scenario的标定尺寸。

## 5. 确认与尚未确认

已确认：端点训练标签与所声明连续桥下事件存在可复现不一致；部分是独立的空间判据问题；既有refinement已部分覆盖简单修复。

尚未确认：真实预训练Plan-R1会以多大概率生成这些token组合；跟踪器和周车真实运动是否重现区间事件；nuPlan实际闭环得分是否受影响；高频标签训练后是否仍需要新增学习方法。以上都不能从合成几何频率推出。
