"""Reports/figures derived only from actually saved experiment outputs."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bridge_dynamics import pose,path
from geometry import gaps,other_pose

HERE=Path(__file__).resolve().parent


def read(name):return list(csv.DictReader((HERE/'results'/name).open()))


def run():
    r=HERE/'results';fig=HERE/'figures';fig.mkdir(exist_ok=True);metrics=read('test_metrics.csv');pred=read('test_predictions.csv');feas=read('feasibility.csv');bases=json.loads((r/'bases.json').read_text())['bases'];training=json.loads((r/'training_summary.json').read_text());ledger=json.loads((r/'evaluation_ledger.json').read_text());data=json.loads((r/'data_ledger.json').read_text());summary=[]
    methods=list(dict.fromkeys(z['method'] for z in metrics))
    for method in methods:
        rows=[z for z in metrics if z['method']==method and z['population']=='all'];a={k:float(np.mean([float(z[k]) for z in rows])) for k in ['collision_rate_all_proposals','accepted_collision_rate','collision_per_scene','unknown_rate','rejection_rate','mean_progress_m','accepted_progress_proxy_m','dynamics_feasible_rate','token_compatible_rate','mean_endpoint_error_m','parameters','head_calls','online_pose_queries','warm_p50_ms','warm_p95_ms','scene_inference_p50_ms']};a['method']=method;a['collision_seed_min']=min(float(z['collision_rate_all_proposals']) for z in rows);a['collision_seed_max']=max(float(z['collision_rate_all_proposals']) for z in rows);summary.append(a)
    displacement=np.array([b['max_bridge_displacement_m'] for b in bases]);rescues={str(buf):sum(int(z['rescue_'+str(buf)]) for z in feas) for buf in [0,.005,.02,.05]};broadrescues=sum(int(z['rescue_0']) for z in feas if z['population']=='broad');boundrescues=sum(int(z['rescue_0']) for z in feas if z['population']=='boundary')
    paired=[]
    for seed in [17,29,43,71]:
        b=[z for z in pred if z['method']=='bridge_head' and int(z['seed'])==seed];f=[z for z in pred if z['method']=='reward_selector' and int(z['seed'])==seed];p=[z for z in pred if z['method']=='projected_residual' and int(z['seed'])==seed]
        assert [z['scene_id'] for z in b]==[z['scene_id'] for z in f]
        assert all(z['coefficient']==q['coefficient'] and z['collision']==q['collision'] for z,q in zip(b,p))
        d=np.array([int(z['collision'])-int(q['collision']) for z,q in zip(b,f)]);rng=np.random.default_rng(seed+91000);boot=d[rng.integers(len(d),size=(3000,len(d)))].mean(1)
        paired.append(dict(seed=seed,bridge_minus_selector=float(d.mean()),paired_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),better_scenes=int((d<0).sum()),worse_scenes=int((d>0).sum())))
    record=dict(metrics_seed_means=summary,displacement_m=dict(min=float(displacement.min()),median=float(np.median(displacement)),max=float(displacement.max())),oracle_rescues=rescues,zero_buffer_rescues_by_population=dict(broad=broadrescues,boundary=boundrescues),paired=paired,total_optimizer_steps=sum(z['optimizer_updates'] for z in training),training_seconds=sum(z['training_seconds'] for z in training),evaluation_seconds=ledger['seconds'],data_seconds=data['total_seconds'],go_gpu=False,scope='Constrained synthetic component evidence only; no real policy outputs')
    (r/'summary.json').write_text(json.dumps(record,indent=2))
    # Raw generation metrics, not acceptance rates after an external filter.
    shown=[z for z in summary if z['method'] in ['fixed_bridge','reward_selector','residual_head','bridge_head','projected_residual']]
    colors=['#718096','#ed8936','#a855f7','#2563eb','#16a34a'];f,ax=plt.subplots(1,2,figsize=(11,4));x=np.arange(len(shown));ax[0].bar(x,[z['collision_rate_all_proposals']*100 for z in shown],color=colors);ax[0].set_ylabel('Raw overlap proposals (%)');ax[0].set_xticks(x,[z['method'].replace('_','\n') for z in shown],fontsize=8);ax[1].bar(x,[z['token_compatible_rate']*100 for z in shown],color=colors);ax[1].set_ylabel('Exact token-compatible (%)');ax[1].set_xticks(x,[z['method'].replace('_','\n') for z in shown],fontsize=8);f.suptitle('Synthetic held-out scenes; mean of four training seeds');f.tight_layout();f.savefig(fig/'raw_generation.png',dpi=160);plt.close(f)
    f,ax=plt.subplots(1,2,figsize=(10,4));ax[0].bar([b['token_id'] for b in bases],displacement*1000,width=10);ax[0].set_xlabel('Original Vehicle token ID');ax[0].set_ylabel('Max certified-family displacement (mm)');ax[1].bar(list(rescues),list(rescues.values()),color='#2563eb');ax[1].set_xlabel('Required SAT separation margin (m)');ax[1].set_ylabel('Fixed bridge unsafe -> grid bridge certified safe');f.tight_layout();f.savefig(fig/'feasibility.png',dpi=160);plt.close(f)
    curves=read('training_curves.csv');f,ax=plt.subplots(figsize=(8,4))
    for method in ['reward_selector','residual_head','bridge_head']:
        epochs=np.arange(1,201);loss=[np.mean([float(z['validation_loss']) for z in curves if z['method']==method and int(z['epoch'])==e]) for e in epochs];ax.plot(epochs,loss,label=method)
    ax.set(xlabel='Epoch (six real optimizer updates per epoch)',ylabel='Shared-bank validation loss');ax.legend();f.tight_layout();f.savefig(fig/'training.png',dpi=160);plt.close(f)
    # Save a reproducible existence witness, if present, never invent one.
    resc=[z for z in feas if int(z['rescue_0'])]
    if resc:
        case=resc[0];scenes=json.loads((r/'test_scenes.json').read_text());s=next(z for z in scenes if z['scene_id']==case['scene_id']);b=bases[s['base_index']];coeff=np.array(b['coeff']);c=float(case['grid_best_c']);t=np.linspace(0,1,10001);g0=gaps(coeff,0.,s,t);g1=gaps(coeff,c,s,t)
        f,ax=plt.subplots(1,2,figsize=(11,4));ax[0].plot(t,g0*1000,label='fixed bridge');ax[0].plot(t,g1*1000,label='same-token grid bridge');ax[0].axhline(0,color='k',lw=.8);tm=t[np.argmin(g0)];ax[0].set(xlabel='Time (s); zoom near closest approach',ylabel='SAT gap (mm)',ylim=(-2,30),xlim=(max(0,tm-.005),min(1,tm+.005)));ax[0].legend();ax[1].plot(t,np.linalg.norm(path(coeff,c,t)-path(coeff,0.,t),axis=-1)*1000);ax[1].set(xlabel='Time (s)',ylabel='Bridge displacement from fixed path (mm)');ax[1].set_title(s['scene_id']+'; boundary diagnostic');f.tight_layout();f.savefig(fig/'existence_witness.png',dpi=160);plt.close(f)
        (r/'existence_witness.json').write_text(json.dumps(dict(scene=s,base=b,oracle_case=case,note='Feasible grid oracle, NOT learned-head success; conditional synthetic existence only.'),indent=2))
    header='| 方法 | 原始碰撞% | 接受碰撞% | unknown% | 拒判% | 位移/接受位移m | 端点兼容% | p50/p95 ms | 参数 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    table=''.join(f"| {z['method']} | {100*z['collision_rate_all_proposals']:.3f} | {100*z['accepted_collision_rate']:.3f} | {100*z['unknown_rate']:.3f} | {100*z['rejection_rate']:.3f} | {z['mean_progress_m']:.3f}/{z['accepted_progress_proxy_m']:.3f} | {100*z['token_compatible_rate']:.1f} | {z['warm_p50_ms']:.3f}/{z['warm_p95_ms']:.3f} | {int(z['parameters'])} |\n" for z in summary)
    text=f'''# 第四阶段实际执行结果（主结果：协议 v2）

结论：**当前一维端点保持 Bridge Head 不满足继续 GPU/论文方法开发门槛（NO-GO）**。这是受约束合成组件结论，不是 Plan-R1 真实模型结论，也不是所有可能桥族不可行的证明。

## 真实性与训练

使用原 Vehicle 词表，得到 {len(bases)} 个通过本轮充分动力学证书的基桥。训练/验证/测试 768/192/512 个分离交通场景，测试 broad/boundary 各256；没有真实 checkpoint 输出。12个独立训练模型，每个1,769参数/1,200次 Adam 更新，合计 **{record['total_optimizer_steps']:,} 次真实参数更新**，CPU训练合计 {record['training_seconds']:.2f}s。参数变化、选中 epoch、初始/最终损失逐项见 `results/training_summary.json`；四个 seed 全部保留，没有挑最好 seed。

## 受约束桥存在性

认证系数族的最大空间变化，按21个 token 汇总：min/median/max={1000*displacement.min():.3f}/{1000*np.median(displacement):.3f}/{1000*displacement.max():.3f}mm。它是当前充分证书下的族容量，不是所有轨迹的理论上限。共享初态、三个位姿和全程 C2/jerk 约束均检查。

9候选同 token oracle 在512场景中，固定桥 unsafe 而所选桥获得 safe 证书的数量：零 margin **{rescues['0']}**（broad {broadrescues} / boundary {boundrescues}）；5mm **{rescues['0.005']}**；2cm **{rescues['0.02']}**；5cm **{rescues['0.05']}**。这些数是**各自 margin 下从不满足到满足**的数量，分母/事件不同，不能解读成“零碰撞救回后还有2cm余量”。`feasibility.csv` 提供每个场景的基桥/最优桥 status。找不到 grid bridge 不证明无解。

## 同标签比较

学习行是四种子均值，固定基线单次。所有 raw Head **零在线碰撞几何查询、零拒判**；因此其碰撞结果来自生成轨迹本身。`projected_residual` 与 Bridge 函数族相同，复用同一权重作恒等性检验，不算额外训练模型。

{header}{table}

动力学获证率所有行100%，由 decoder 系数盒的解析充分证书给出，另有数值交叉核对；不是实车可执行性认证。普通残差平均端点误差约 {next(z for z in summary if z['method']=='residual_head')['mean_endpoint_error_m']:.3f}m，端点兼容0%，它较低的碰撞不能当作保持原 token 的改善。位移是1秒末 x 位置，非 closed-loop 行驶进度；拒判时不执行任何声明安全的停车轨迹。

Bridge 四 seed 原始碰撞率范围 {100*next(z for z in summary if z['method']=='bridge_head')['collision_seed_min']:.3f}–{100*next(z for z in summary if z['method']=='bridge_head')['collision_seed_max']:.3f}%。与同标签 selector 的逐场景差和 bootstrap 区间见 `results/summary.json`（负数才是 Bridge 好）；bootstrap 仅针对这512个合成场景，不能外推真实分布。与投影残差逐场景系数/碰撞完全相同。

10/50Hz 对照对当前指定桥做过滤/重排；boundary 人为富集近切触，不能将其接受碰撞率当真实系统漏检率，也没有重新运行阶段三大样本漏检统计。所有 split/population/seed、unknown、参数、调用数与逐场景时延在 `test_metrics.csv`/`test_predictions.csv`。重排9候选查询更多，不称严格相同在线计算。

## 成本、数值修正与失败

主数据生成 {data['total_seconds']:.2f}s；训练 {record['training_seconds']:.2f}s；评价 {ledger['seconds']:.2f}s。CPU单线程，具体型号/依赖见 environment.json；云主机延迟会抖动。数据集标签轨迹共26,496条（含测试）；所有方法共享同一库。世界模型调用0。在线几何查询按行记录，离线局部最小化/场景构造查询未逐次计数，墙钟包含其代价。

v1使用200Hz标签，被独立区间器发现时间混叠；所有 v1源码/数据/训练日志/指标保留于 `results/diagnostic_v1_sampled_labels/`，**不是主结果**。v2对每个局部极小值 bracket 做连续最小化，并据此构造固定桥附近的边界，网络/限值/种子/训练预算不变。这是发现问题后的数值修正，不是预注册盲测；训练/测试场景仍相互分离，未用测试训练或选 checkpoint。主标签又通过独立区间器抽查，见 label_audit.json；这仍不是所有标签的全局解析证明。

失败案例 `failure_cases.json` 保存 seed17 所有 Bridge 碰撞/unknown；若存在 oracle 可救回场景，`existence_witness.json`/图保存的是**oracle存在性**，不混称学到该能力。第一次JSON导出失败和第二次vectorization广播失败均留日志，修正不涉及原始 Plan-R1。

尚未完成：真实 checkpoint 严格加载/真实输出/闭环评估、完整 Plan-R1仅改奖励基线、完整 CorrectionPlanner 同标签比较、任意 token 组合/高维桥和跟踪误差验证。具体文件清单见 checkpoint_and_inputs.md。
'''
    (HERE/'experiment_results.md').write_text(text);print(json.dumps({k:v for k,v in record.items() if k not in ['metrics_seed_means']},indent=2))

if __name__=='__main__':run()
