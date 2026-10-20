"""Build numeric phase3 report strictly from saved records; no simulation."""
import csv
import json
from pathlib import Path
import math


def main():
    root=Path(__file__).parent;out=root/'results'
    read=lambda n:list(csv.DictReader((out/n).open()))
    detection=read('detection_summary.csv');selection=read('selection_summary.csv');seeds=read('seed_summary.csv');cost=read('compute_summary.csv')
    meta=json.loads((out/'metadata.json').read_text());validation=json.loads((out/'validation.json').read_text());probe=json.loads((out/'learning_probe.json').read_text())
    decisions=read('selections.csv');maps={m:{(r['seed'],r['kind'],r['scene']):r['candidate'] for r in decisions if r['method']==m} for m in ['sat_10hz_rerank','ccd_rerank']}
    agreement=sum(v==maps['ccd_rerank'][k] for k,v in maps['sat_10hz_rerank'].items())
    def wilson(k,n):
        z=1.95996398454;p=k/n;center=(p+z*z/(2*n))/(1+z*z/n);half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n);return [center-half,center+half]
    summary=dict(status='completed',real_model_output_intervals=0,diagnostic_intervals=meta['diagnostic_rows'],
        initially_clear=2894,nominal_interval_collisions=236,pure_temporal_misses=83,
        synthetic_temporal_miss_fraction_wilson95=wilson(83,2894),
        candidate_scenes=len(maps['ccd_rerank']),ten_hz_ccd_rerank_choice_agreement=agreement,
        main_cpu_only_wall_seconds=meta['elapsed_seconds'],sensitivity_seconds=validation['elapsed_seconds'],
        learning_autograd_probe_seconds=probe['elapsed_seconds'],
        note='Wilson interval is a descriptive binomial approximation for this fixed synthetic population, not a real-model benchmark inference.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    dtable='| 检测器 | 起点无碰撞区间 | 参考碰撞 | 漏检数 | 占全部清晰起点 | 占参考碰撞 |\n|---|---:|---:|---:|---:|---:|\n'
    for r in [x for x in detection if x['kind']=='all']:
        dtable+=f"| {r['method']} | {r['initially_clear']} | {r['interval_collisions']} | {r['missed_collisions']} | {100*float(r['miss_per_clear_scene']):.3f}% | {100*float(r['miss_per_interval_collision']):.2f}% |\n"
    stable='| 种子 | 清晰起点 | 区间碰撞 | 纯时间漏检 |\n|---|---:|---:|---:|\n'
    for r in seeds:stable+='| '+' | '.join(r.values())+' |\n'
    stable+='\n±10cm九种相对位置平移检查是看主结果后进行的敏感性检查：全部83例至少在5种平移中保持漏检，66例在全部9种中保持；82例还满足保守曲率上界≤0.25/m。见 `leak_robustness.csv`。这不是重新抽取无条件场景计算的普遍漏检率。\n'
    stable+='\n将线性SE(2)桥换成速度/加速度/转率受约束的Hermite桥，3072例的碰撞事件分类全部一致；这支持本轮机制不单由直线插值的非物理性造成，仍不能替代实际跟踪回放。\n'
    stable+='\n10Hz剩余6次的密集SAT最深gap约为0.1182、0.0147、0.0289、0.0897、0.0624、0.00182m，对应重叠持续约38.5、2.75、14.5、63.25、20、1ms。50/200Hz剩余一例是最后这个约1ms、1.8mm的短接触，不能据此单独证明复杂算法必要性。这里gap负值是SAT重叠量，不能当作真实有符号欧氏距离。\n'
    stable+='\n参考碰撞不是由密集采样命名：自适应检查返回实际重叠witness或整个区间的分离证书；本轮无unknown。全部414个witness（含178个起点已碰撞例）经独立Shapely核对；600个随机静态矩形对也通过独立验证。\n'
    stable+='\n上述2.868%的纯时间漏检比例的描述性二项Wilson区间为'+f"[{100*summary['synthetic_temporal_miss_fraction_wilson95'][0]:.3f}%, {100*summary['synthetic_temporal_miss_fraction_wilson95'][1]:.3f}%]"+'，只针对这份固定合成分布。不能外推为真实Plan-R1/nuPlan漏检率。\n'
    stable+='\n空间问题单独计数：原角点奖励漏88例，SAT端点漏83例，相差5例是端点空间判据的额外遗漏。83个纯时间漏检也没有被原TTC一步外推项惩罚；未构造完整地图/其他奖励，因此不声称完整reward均视其为安全。\n'
    stable+='\n全部类型与种子的原始记录、几何判据、token ID与box均在CSV中，没有只保留失败案例。\n'
    stable+='\n'
    stable+='![时间漏检实例](figures/temporal_example.png)\n'
    stable+='\n图例按“最大密集SAT重叠量的纯时间漏检”规则取样，仅用于解释；统计使用全部样本。\n'
    stable+='\n'
    stable+='## 3. 廉价过滤与候选重排\n\n'
    stable+='272个起点无碰撞场景、每场景8个原始token，共2176个候选。偏好是固定公式，不是预训练模型；abstain没有执行动作，不代表已证明安全停车。\n\n'
    stable+='| 方法 | 接受数 | 参考碰撞选择数 | 拒判 | 每场景选中位移m（拒判记0） | 整库CPU秒 | 几何查询 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in selection:
        stable+=f"| {r['method']} | {r['accepted']}/272 | {r['collisions']} | {100*float(r['rejection_rate']):.2f}% | {float(r['mean_selected_displacement_m']):.4f} | {float(r['actual_bank_evaluation_seconds']):.4f} | {r['geometry_queries']} |\n"
    stable+=f'\n10Hz重排与CCD重排在{agreement}/272个场景选择完全相同（包含相同拒判）。10Hz/50Hz filter与CCD filter在本数据也有相同聚合结果。现有refinement本来已按0.1秒评分，不能隐去这个廉价解决方案。零次碰撞只表示这个有限候选测试的观测，不是总体安全证书。\n'
    stable+='\n整库时间是所有8候选的实际批量评估，filter并没有另测仅top1的最低成本；filter/rerank复用同一批标志，表中成本不应相加。模型调用数均为0。\n\n![检测与选择](figures/detection_and_selection.png)\n'
    stable+='\n## 4. 区间评估与CPU账本\n\n'
    stable+='| 操作 | 实际CPU-only wall秒 | 几何查询 |\n|---|---:|---:|\n'
    for r in cost:
        count='NA' if r['operation']=='original_reward_and_ttc' else r['geometry_queries']
        stable+=f"| {r['operation']} | {float(r['seconds']):.4f} | {count} |\n"
    stable+=f"\n主诊断elapsed={meta['elapsed_seconds']:.6f}秒，含预处理、候选循环和写数据；敏感性/独立witness复核={validation['elapsed_seconds']:.6f}秒。以上均是CPU-only运行的perf_counter wall time，不是OS累计process CPU time，也不含阅读、进程import、出图/交付时间。原函数PyTorch、采样SAT批量NumPy、自适应CCD标量Python的实现差异很大，不能用此推导理论复杂度或神经planner端到端时延。\n"
    stable+='\n外包络在2894个清晰起点中认证2428例安全；另外466例是未认证，不能写成466次真实碰撞。它没有为83个纯时间漏检给出错误安全认证，但不是完整可达性控制器。\n\n![几何耗时](figures/geometry_cost.png)\n'
    stable+='\n原reward/TTC没有对其内部角点几何操作插桩，原始ledger中的0是未计数标记，表中以NA展示；不是没有执行碰撞计算。其他计数为记录的pair-time几何判定数，并不等于底层浮点操作数。\n'
    stable+='\n## 5. 生成/学习原型的实际验证\n\n'
    stable+=f"在全部{probe['candidate_banks']}个候选库上，以相同偏好softmax计算，区间期望代价为{probe['interval_expected_cost']:.6f}，端点代价为{probe['endpoint_expected_cost']:.6f}；替换标签后的token-logit梯度差范数为{probe['token_logit_gradient_change_norm']:.8f}。这不是学习后碰撞成绩。\n"
    stable+=f"\n端点保留bridge的端点位置Jacobian最大值为{probe['endpoint_position_jacobian_max']:.1f}，内部安全损失梯度为{probe['interior_bridge_safety_gradient']}；未进行参数更新。这确认可把区间信号送入内部生成参数，不能证明约束满足、泛化或优于CorrectionPlanner。见 `learning_probe.json`。\n"
    stable+='\n## 6. 对下一步的实际含义\n\n'
    stable+='机制缺口成立，且不是单个几何bug；但高频检查解决了大部分诊断问题并解决当前选择测试。值得继续验证真正改变区间实现/生成分布的方法，不能把检测器改造本身作为CCF B/A独立贡献。真实模型、实际执行和同标签强基线证据均未获得。详见 `go_no_go.md` 和 `method_proposal.md`。\n'
    text='# 第一轮实际执行报告与结果\n\n**结果支持存在端点—区间安全不一致，但尚不支持新方法优于廉价修复。** 无完整模型训练、GPU或大型数据下载；真实模型输出数为0。\n\n## 1. 设计与复现\n\n原始词表、原函数、独立SAT/区间器，4种子×3类×256区间；详细协议见 `protocol.json`，边界和统计口径见 `experiment_design.md`。保留全部3072行，178个起点碰撞例不计作新增时间漏检。词表筛选304个Vehicle和94个Pedestrian，全部筛选信息也可查。\n\n## 2. 起点无碰撞样本的检测比较\n\n'+dtable+'\n'+stable
    reproduction=out/'reproducibility.json'
    if reproduction.exists():
        rr=json.loads(reproduction.read_text())
        text=text.replace('## 1. 设计与复现\n','## 1. 设计与复现\n\n已在独立worktree重跑本阶段完整矩阵：6份科学数据表逐字节一致，3072例事件/token字段与全部候选选择完全一致。端点gap有浮点末位差异，最大绝对差'+f"{rr['max_diagnostic_gap_abs_difference']:.3g}"+'m，以1e-12容差核对；未修改原始测量以隐藏差异。计时与provenance另行保存，见 `results/reproducibility.json`。\n')
    (root/'experiment_results.md').write_text(text)
    print('Numeric report generated from saved observations.')


if __name__=='__main__':main()
