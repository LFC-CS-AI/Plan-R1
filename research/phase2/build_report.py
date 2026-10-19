"""Build the numerical report from saved CSV/JSON; does not run simulation."""
import csv
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent;out=root/'results'
    summaries=list(csv.DictReader((out/'all_methods_summary.csv').open()))
    lookup={(r['domain'],float(r['strength']),int(float(r['budget'])),r['method']):r for r in summaries}
    meta=json.loads((out/'metadata.json').read_text());ext=json.loads((out/'extension_metadata.json').read_text())
    num=json.loads((out/'numerical_validation.json').read_text());check=json.loads((out/'verification.json').read_text())
    accounting={
        'main':dict(elapsed_seconds=meta['elapsed_seconds'],online_model_calls=meta['total_online_model_calls'],offline_model_calls=meta['total_offline_model_calls'],reference_calls=meta['total_reference_calls']),
        'post_hoc_controls':dict(elapsed_seconds=ext['elapsed_seconds'],online_model_calls=int(ext['additional_online_model_calls']),offline_model_calls=ext['additional_offline_model_calls'],reference_calls=ext['additional_reference_calls']),
        'numerical_validation':dict(elapsed_seconds=num['timing_seconds'],online_model_calls=0,offline_model_calls=0,reference_calls=num['reference_calls']),
        'one_block_replay':dict(elapsed_seconds=check['elapsed_seconds'],online_model_calls=check['additional_online_model_calls'],offline_model_calls=0,reference_calls=check['additional_reference_calls'])}
    accounting['total_recorded_components']={k:sum(v[k] for v in accounting.values()) for k in accounting['main']}
    accounting['timing_scope']='CPU-only elapsed wall times, not summed OS process CPU time; excludes smoke/unit tests, literature reading, plotting/reporting and delivery I/O.'
    (out/'execution_accounting.json').write_text(json.dumps(accounting,indent=2))
    methods=list(dict.fromkeys(r['method'] for r in summaries))
    # Use the same readable ordering as the protocol, with post-hoc controls last.
    methods=meta['config']['methods']+['action_residual','actionwise_cp']
    def pct(v):
        x=float(v);return 'NA' if x!=x else f'{100*x:.2f}%'
    def fmt(v):return f'{float(v):.4f}'
    def table(domain):
        text='| 方法 | 排序错率* | 错误接受/已接受 | 部署碰撞 | 期望碰撞 | 部署进度m | 进度/参考规则 | 停车 | 调用数 | CPU秒 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        for m in methods:
            r=lookup[(domain,1.3,1280,m)]
            text+='| '+m+' | '+' | '.join(pct(r[k]) for k in ['ranking_error','false_safe_conditional','collision_rate','expected_collision'])+' | '+f"{float(r['progress_m']):.2f}"+' | '+pct(r['progress_relative_to_oracle'])+' | '+pct(r['rejection_rate'])+' | '+str(int(float(r['model_calls'])))+' | '+fmt(r['cpu_seconds'])+' |\n'
        return text
    budgettable='| 域 | B | ensemble 排序错率 | 错误接受/已接受 | 期望碰撞 | 进度比 | 停车 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for d in ['id','driver_shift']:
        for b in [80,320,1280]:
            r=lookup[(d,1.3,b,'ensemble')]
            budgettable+='| '+d+' | '+str(b)+' | '+' | '.join(pct(r[k]) for k in ['ranking_error','false_safe_conditional','expected_collision','progress_relative_to_oracle','rejection_rate'])+' |\n'
    compute='| 阶段 | CPU-only elapsed秒 | 在线模型调用 | 离线模型调用 | 参考调用 |\n|---|---:|---:|---:|---:|\n'
    for name,v in accounting.items():
        if not isinstance(v,dict):continue
        compute+=f"| {name} | {v['elapsed_seconds']:.3f} | {v['online_model_calls']:,} | {v['offline_model_calls']:,} | {v['reference_calls']:,} |\n"
    text='''# 第二阶段实际实验结果

结论：toy 中存在预测排序与参考风险不一致，但**目前不支持以新的风险排序算法直接立项**。普通多样本/ensemble 消除大部分确定性基线的危险选择；剩余问题需分开检查绝对风险低估、保守性与隐藏行为漂移。没有运行 Plan-R1 真实模型，不能据此评价其基准安全性。

## 设计与数据完整性

主方案见 `protocol.json`，在主结果产生前固定；四种子、三档反应、两域、三个预算、9个原始方法，31104条决策。随后保留原始文件，增加2个按动作校准的普通控制（6912条决策），见 `baseline_extension.json`。补充是在看到全局 CP 全停车后决定的，不是预注册验证；没有用 test 标签拟合校准参数或调超参。完整数据共38016条决策、5760条参考候选记录，CSV均可追溯场景与种子。

机制测试通过；对全部38016条记录检查风险、接受、进度、部署结果和预算一致性；额外重放一个48场景参考块及27个原始方法/预算组合，计数、进度、预测和选择**逐项精确一致**，见 `results/verification.json`。这是一个块的复现，不冒充全矩阵二次复现。

## 1. 强反应、B=1280：所有方法

每表192场景。排序错率*为严格逆序+半个平局，必须结合下文的分解；NA=全部停车、条件错误率未定义。调用数和CPU秒为该表行的192场景合计在线成本，离线成本另列。所有强度、两域、全部11方法的B=1280表见 [full_budget_table.md](full_budget_table.md)，其他预算及95%区间见 [all_methods_summary.csv](results/all_methods_summary.csv)。

### ID

'''+table('id')+'''
### 隐藏驾驶人漂移

'''+table('driver_shift')+'''
## 2. 固定预算比较

'''+budgettable+'''
确定性预测每场景只用5次有用调用；其他方法用满B，不以无效重复填充预算。高预算并非每个指标都单调改善；例如ID强反应ensemble在B=320时条件错误接受1.98%，B=1280为3.06%，且进度略下降，不能只选最有利档位。

无反应负对照中，B=1280、ID的MC/ensemble排序错率为0、错误接受0、进度比91.45%；中反应ensemble为4.30%错误接受、82.38%进度比。结构错设与交互会增加困难，但不是所有互动强度都需要新算法。负对照未被隐去。

## 3. 排序、打平与危险选择不是同一个指标

ID强反应700个可区分动作对：deterministic有567平局、17严格逆序，tie-aware错率42.93%；ensemble有3平局、65严格逆序，错率9.50%。**ensemble的严格逆序占比更高，但碰撞更少**：只报告42.93%→9.50%会夸大“纠正逆序”证据。

同条件deterministic有100次错误接受，65次存在安全非停车替代，只有3次所选动作与安全替代严格反排。ensemble只有3次错误接受，全部无安全非停车替代；shift下ensemble的9次也是如此。见 [selection_failure_modes.csv](results/selection_failure_modes.csv)。这些残余失败更直接是没有识别“应当停车”的绝对风险阈值错误。排序可能仍有问题，但没有证明它是强基线剩余碰撞的主要原因。

[example.json](results/example.json)保存按明确规则取到的首个deterministic实例：seed202/scene7，强反应ID。assert预测风险0，参考0.15088；slow预测1，参考0.03564，后者99%参考区间上限0.04757、前者下限0.13110。选择assert且独立部署发生碰撞，说明这种机制**可以**出现；它是说明例，不是挑例后估计频率。

![预测与参考风险示例](figures/ranking_example.png)

## 4. 便宜方法已经做到什么，还不能断言什么

ID强反应期望碰撞：deterministic15.91%（95%bootstrap区间13.35–18.45%），ensemble0.81%（0.32–1.80%）。ensemble条件错误接受为3/98=3.06%，bootstrap区间0–9.28%，进度比75.41%。这是明显改善，但不能以点估计断言达到严格5%保证。

新增action_residual同条件错误接受6/125=4.8%（0.81–10.25%），进度比86.71%，达到预先经验门槛的**点估计**。它使用普通校准，说明“安全与进度之间一定存在新方法缺口”尚未成立；其区间和事后设计又不足以宣布问题完全解决。需要新场景验证，不能从当前测试集选出最佳方法后包装泛化。

max-over-actions/pooled/actionwise CP本轮全部停车，0碰撞不能视为有效解决；对应进度只有参考规则的约10–12%。这不推翻CP理论：我们尚未做上下文条件化校准，也未满足 interactive robust CP/CPC 的完整假设。模型内binomial筛选更保守，在高预算下仍有约78%停车；paired difference的模型风险差上界也不能修正绝对偏差，shift下其错误接受可达33.85%。这些都不是击败同名完整论文的证据。

隐藏shift下ensemble条件错误接受9.18%、期望碰撞1.28%，预测与ID完全相同。这说明同一个错设模型的自信度不能保证隐藏参数漂移后的安全；不说明存在一个无需任何额外可观测信息/真实反馈就能解决它的方法。

![安全与进度](figures/safety_progress.png)

![排序随预算变化](figures/ranking_by_budget.png)

![选择可靠性与拒判](figures/selection_reliability.png)

## 5. 数值精度与CPU成本

每条件4个未用于主测试的新场景，dt从0.08减到0.04的平均/最大风险差为0.00102/0.02148；独立8192样本参考与2048样本参考平均/最大差为0.00638/0.03394。见 `numerical_validation.csv/json`。这是有限敏感性检查，不证明精确连续动力学或所有阈值标签正确。

'''+compute+'''
时间为真实CPU-only执行的wall time，非OS累计process CPU time；主阶段包含20.432秒bootstrap汇总、日志和checkpoint I/O。上表合计不含早期smoke/单元测试、阅读、图表分析及交付。在线“调用”是候选随机完整结果，不是神经网络forward。可共享的校准/参考也保留原始重复实际调用数，不隐去事后控制的额外成本。

完整定义、校准条件与公平性限制见 [environment_and_metrics.md](environment_and_metrics.md)。最终立项判断及尚缺证据见 [go_no_go.md](go_no_go.md)。本轮没有提供新算法或新定理，也不能支持Plan-R1真实基准存在同类缺陷的结论。
'''
    (root/'experiment_results.md').write_text(text)
    print('Report generated from saved observations; no simulation executed.')


if __name__=='__main__':main()
