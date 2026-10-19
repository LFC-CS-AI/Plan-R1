#!/usr/bin/env python3
"""CPU mechanism experiments; no Plan-R1 checkpoint or nuPlan data required.

Python >=3.9, numpy, matplotlib. Optional source audits require the configured
Plan-R1 CPU environment (torch, torch_geometric, torch_scatter and source deps).
Run: python toy_experiment.py --out results --source-root /workspace/Plan-R1
This is NOT a reproduction of Plan-R1's nuPlan benchmark.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('MPLCONFIGDIR', '/tmp/planr1-research-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H, DT, V0, SIGMA = 16, 0.5, 8.0, 0.8
W = DT ** 2 * (H - np.arange(H) - 0.5)
BARRIER, PENALTY, EXO_P = 66.0, 20.0, 0.15
PROGRESS, EFFORT = 0.05, 0.01
METHODS = ['step', 'rtg', 'vd', 'rloo', 'counterfactual']


def reverse_sum(r):
    return np.cumsum(r[..., ::-1], axis=-1)[..., ::-1]


def simulate(a, exo, barrier=BARRIER):
    """Continuous longitudinal point-mass motion, fixed horizon, delayed wall cost.

    A terminal wall event is used to isolate delayed credit; it does not terminate
    intermediate states. Exogenous terminal loss is independent of all actions.
    Negative velocities are allowed (unconstrained linear mechanism model).
    """
    vel_before = V0 + DT * (np.cumsum(a, axis=-1) - a)
    dx = DT * vel_before + 0.5 * DT ** 2 * a
    x = np.cumsum(dx, axis=-1)
    rewards = PROGRESS * dx - EFFORT * a ** 2
    crash = x[..., -1] >= barrier
    rewards[..., -1] -= PENALTY * (crash.astype(float) + exo)
    return rewards, crash, x


def true_objective_gradient(mu, barrier=BARRIER):
    mean_x = V0 * H * DT + W @ mu
    std_x = SIGMA * np.linalg.norm(W)
    z = (barrier - mean_x) / std_x
    prob = 0.5 * math.erfc(z / math.sqrt(2))
    density = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    value = PROGRESS * mean_x - EFFORT * np.sum(mu ** 2 + SIGMA ** 2)
    value -= PENALTY * (prob + EXO_P)
    grad = PROGRESS * W - 2 * EFFORT * mu - PENALTY * density * W / std_x
    return value, grad, prob


def weights(a, rewards, exo, mu, method, rng, k=8, model_shift=0.0):
    if method == 'step':
        return rewards
    if method == 'rtg':
        return reverse_sum(rewards)
    if method == 'vd':
        # Source-equivalent pooling over all G*T entries, all-valid mask.
        # c=1 aligns units with raw returns. c=.1 would multiply these by 10.
        return reverse_sum(rewards - rewards.mean(axis=(-2, -1), keepdims=True))
    if method == 'rloo':
        rtg = reverse_sum(rewards)
        return rtg - (rtg.sum(axis=-2, keepdims=True) - rtg) / (a.shape[-2] - 1)
    if method != 'counterfactual':
        raise ValueError(method)
    # Replace a single action with K independent samples from its policy.
    # Hold all other actions and exogenous draw fixed. This baseline is action
    # independent ONLY because this toy policy is open-loop factorized.
    alternatives = mu.reshape((1,) * (a.ndim - 1) + (H, 1)) + SIGMA * rng.normal(size=a.shape + (k,))
    change = alternatives - a[..., None]
    terminal_x = V0 * H * DT + a @ W
    cf_x = terminal_x[..., None, None] + W[None, None, :, None] * change
    # Exact full-return change under linear dynamics; exogenous loss cancels.
    cf_total = rewards.sum(-1)[..., None, None]
    cf_total = cf_total + PROGRESS * W[None, None, :, None] * change
    cf_total = cf_total - EFFORT * (alternatives ** 2 - a[..., None] ** 2)
    factual_model_crash = terminal_x >= BARRIER + model_shift
    cf_crash = cf_x >= BARRIER + model_shift
    if model_shift:
        # Both factual and alternative risk evaluated by misspecified model.
        factual_total = rewards.sum(-1) + PENALTY * ((terminal_x >= BARRIER) - factual_model_crash.astype(float))
        cf_total = cf_total + PENALTY * ((terminal_x >= BARRIER) - factual_model_crash.astype(float))[..., None, None]
    else:
        factual_total = rewards.sum(-1)
    cf_total = cf_total - PENALTY * (cf_crash.astype(float) - factual_model_crash[..., None, None])
    return factual_total[..., None] - cf_total.mean(-1)


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def estimator_experiment(out, seeds, groups):
    rows, profiles, bias_rows = [], [], []
    mu = np.zeros(H)
    value, truth, prob = true_objective_gradient(mu)
    for seed in range(seeds):
        rng = np.random.default_rng(4100 + seed)
        a = mu + SIGMA * rng.normal(size=(groups, 4, H))
        exo = (rng.random((groups, 1)) < EXO_P).astype(float)
        exo = np.broadcast_to(exo, (groups, 4))  # same scene disturbance in each group
        rewards, _, _ = simulate(a, exo)
        score = (a - mu) / SIGMA ** 2
        for method in METHODS:
            adv = weights(a, rewards, exo, mu, method, np.random.default_rng(8700 + seed))
            estimates = (score * adv).mean(axis=1)
            mean = estimates.mean(axis=0)
            for t in range(H):
                profiles.append(dict(seed=seed, method=method, t=t, mean=mean[t], truth=truth[t],
                                     variance=estimates[:, t].var(ddof=1)))
            rows.append(dict(seed=seed, method=method, early_mean=mean[:4].mean(),
                             early_truth=truth[:4].mean(),
                             variance=estimates.var(axis=0, ddof=1).mean(),
                             mse=np.mean((estimates - truth) ** 2),
                             mean_gradient_rmse=np.sqrt(np.mean((mean - truth) ** 2)),
                             cosine=np.dot(mean, truth) / (np.linalg.norm(mean) * np.linalg.norm(truth))))
        for shift in [0., 2., 6., 12.]:
            adv = weights(a, rewards, exo, mu, 'counterfactual', np.random.default_rng(8700 + seed), model_shift=shift)
            estimates = (score * adv).mean(axis=1)
            bias_rows.append(dict(seed=seed, shift=shift, early_mean=estimates.mean(0)[:4].mean(),
                                  early_truth=truth[:4].mean(),
                                  mean_gradient_rmse=np.sqrt(np.mean((estimates.mean(0) - truth) ** 2))))
    write_csv(out / 'gradient_metrics.csv', rows)
    write_csv(out / 'gradient_profiles.csv', profiles)
    write_csv(out / 'model_error.csv', bias_rows)
    return rows, profiles, bias_rows, dict(value=value, crash_probability=prob, gradient=truth.tolist())


def learning_experiment(out, seeds, iterations):
    rows = []
    for seed in range(seeds):
        for method in METHODS:
            mu = np.zeros(H)
            rng = np.random.default_rng(19000 + seed)
            cf_rng = np.random.default_rng(29000 + seed)
            calls = 0
            for iteration in range(iterations + 1):
                val, _, prob = true_objective_gradient(mu)
                if iteration % 10 == 0 or iteration == iterations:
                    rows.append(dict(seed=seed, method=method, iteration=iteration,
                                     expected_return=val, crash_probability=prob,
                                     expected_terminal_x=V0 * H * DT + W @ mu,
                                     rollout_equivalents=calls))
                if iteration == iterations:
                    break
                a = mu + SIGMA * rng.normal(size=(64, 4, H))
                exo = np.broadcast_to((rng.random((64, 1)) < EXO_P).astype(float), (64, 4))
                rewards, _, _ = simulate(a, exo)
                adv = weights(a, rewards, exo, mu, method, cf_rng)
                grad = ((a - mu) / SIGMA ** 2 * adv).mean(axis=(0, 1))
                # Same effective step-size and bounds for all methods, no tuning.
                mu = np.clip(mu + 0.01 * grad, -1.5, 1.5)
                calls += 256 * (1 + H * 8 if method == 'counterfactual' else 1)
    write_csv(out / 'learning.csv', rows)
    return rows


def source_audit(out, root):
    if root is None:
        return {'status': 'skipped: pass --source-root to execute actual source methods'}
    import torch
    root = Path(root)
    path = root / 'model/PlanR1.py'
    src = path.read_text()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PlanR1')
    selected = ['compute_ae_process_supervision', 'reward_fn', 'training_step']
    ns = {'torch': torch, 'Batch': object}
    for name in selected:
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    obj = SimpleNamespace(num_samples=4, scaling_factor=.1)
    r = torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.bool)
    adv = ns['compute_ae_process_supervision'](obj, r, mask)
    np_r, np_mask = r.numpy(), mask.numpy()
    centered = (np_r - (np_r * np_mask).sum() / np_mask.sum()) * np_mask / .1
    expected = reverse_sum(centered)
    np.testing.assert_allclose(adv.numpy(), expected, atol=2e-6)
    # No-reward-difference group: all same time profile, first token sum = 0.
    identical = torch.tensor([[1., 1., 1., 0.]] * 4)
    id_adv = ns['compute_ae_process_supervision'](obj, identical, torch.ones_like(identical, dtype=torch.bool))
    assert torch.allclose(id_adv[:, 0], torch.zeros(4))
    # Execute real reward_fn with controlled component fixtures. Collision at t=2.
    done = torch.tensor([[False, False, True, False]])
    gate = (~done).float()
    no_done = torch.zeros_like(done)
    reward_obj = SimpleNamespace(num_future_intervals=4,
        agent_collision_reward=lambda _: (done, gate),
        obstacle_collision_reward=lambda _: (no_done, torch.ones_like(gate)),
        on_road_reward=lambda _: (no_done, torch.ones_like(gate)),
        comfort_reward=lambda _: torch.ones_like(gate),
        ttc_reward=lambda _: torch.ones_like(gate),
        progress_reward=lambda _: torch.ones(1),
        speed_limit_reward=lambda _: torch.ones(1),
        comfort_reward_weight=2, ttc_reward_weight=5,
        speed_limit_reward_weight=2, progress_reward_weight=1)
    real_r, real_done, real_mask = ns['reward_fn'](reward_obj, None)
    assert real_mask.tolist() == [[True, True, True, False]]
    assert real_r.tolist() == [[1., 1., 0., 1.]]
    # Execute actual training_step; verify ratio's forward=1 and nonzero gradient.
    p = torch.tensor([[0.2, -.3]], requires_grad=True)
    p_ref = torch.tensor([[-.1, -.2]])
    a_fixture = torch.tensor([[2., -1.]])
    log = lambda *args, **kwargs: None
    train_obj = SimpleNamespace(mode='plan', beta=.1, log=log,
        rollout=lambda _: (a_fixture, p_ref, p, torch.ones_like(p), torch.ones_like(p)))
    loss = ns['training_step'](train_obj, None)
    loss.backward()
    assert (p.grad.abs() > 0).all()
    # Exact enumeration: G independent Bernoulli actions with one terminal reward.
    # E[gradient at t=0] / true gradient = 1 - 1/G for c=1, fixed H.
    enum_obj = SimpleNamespace(num_samples=4, scaling_factor=1.)
    enum_grad = 0.
    for b in range(16):
        acts = torch.tensor([(b >> g) & 1 for g in range(4)], dtype=torch.float32)
        enum_r = torch.zeros(4, H)
        enum_r[:, -1] = acts
        enum_a = ns['compute_ae_process_supervision'](enum_obj, enum_r, torch.ones_like(enum_r, dtype=torch.bool))
        enum_grad += ((acts - .5) * enum_a[:, 0]).mean().item() / 16
    assert abs(enum_grad - .1875) < 1e-7
    # Real trajectory-level reward classes: change ONLY states after first failure.
    # Prefix reward can still change because outcome metrics use the full rollout.
    sys.path.insert(0, str(root))
    from rewards.progress_reward import ProgressReward
    from rewards.speed_limit_reward import SpeedLimitReward
    infer_pos = torch.zeros(1, H+4, 2)
    infer_pos[0, :, 0] = torch.arange(H+4) * 4.
    expert_pos = torch.zeros(1, 100, 2)
    expert_pos[0, :, 0] = torch.arange(100) * .8
    fixture = {'agent': {'ptr': torch.tensor([0, 1]), 'batch': torch.tensor([0]),
                'infer_position': infer_pos.clone(), 'infer_heading': torch.zeros(1,H+4),
                'position': expert_pos},
        'polyline': {'batch': torch.tensor([0]), 'position': torch.tensor([[100.,0.]]), 'heading': torch.zeros(1)},
        ('polyline', 'polygon'): {'polyline_to_polygon_edge_index':torch.tensor([[0],[0]])},
        'polygon': {'on_route_mask':torch.tensor([True]), 'type':torch.tensor([0], dtype=torch.uint8),
                    'speed_limit':torch.tensor([10.]), 'speed_limit_valid_mask':torch.tensor([True])}}
    prog_fn, speed_fn = ProgressReward(), SpeedLimitReward()
    base_prog, base_speed = prog_fn(fixture), speed_fn(fixture)
    fixture['agent']['infer_position'][:,7:] = infer_pos[:,6:7]
    tail_prog, tail_speed = prog_fn(fixture), speed_fn(fixture)
    assert abs(base_prog.item()-tail_prog.item()) > .1
    # First failure is future-token index 2; changed positions start at index 3.
    prefix_soft_before = (9.+base_prog.item())/10.
    prefix_soft_after = (9.+tail_prog.item())/10.
    tail_leakage = dict(failure_token=2, changed_first_token=3,
        progress_before=base_prog.item(), progress_after=tail_prog.item(),
        speed_before=base_speed.item(), speed_after=tail_speed.item(),
        valid_prefix_reward_before=prefix_soft_before,
        valid_prefix_reward_after=prefix_soft_after,
        scope='real ProgressReward and SpeedLimitReward on one straight-route fixture; other components fixed at 1')
    result = dict(status='passed', source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        commit=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        source_line_ranges={'compute_ae_process_supervision':[361,388], 'reward_fn':[390,415], 'training_step':[145,175]},
        input_reward=r.tolist(), mask=mask.tolist(), advantages=adv.tolist(),
        collision_mask=real_mask.tolist(), raw_reward_after_collision=real_r.tolist(),
        identical_group_first_advantage=id_adv[:, 0].tolist(),
        exact_enumerated_gradient=enum_grad, true_bernoulli_gradient=.25,
        post_failure_outcome_leakage=tail_leakage,
        source_policy_loss_gradient=p.grad.tolist(),
        scope='AST-extracted real methods, controlled tensors; NOT full rollout or real geometry validation')
    (out / 'source_audit.json').write_text(json.dumps(result, indent=2))
    return result


def termination_and_broadcast(out):
    # Exact source-equivalent constant positive reward, stop includes failure step.
    rows = []
    for tau in [1, 4, 8, 12, 16]:
        r = np.ones((4, H)); mask = np.ones_like(r)
        r[0, tau - 1] = 0; mask[0, tau:] = 0
        mean = (r * mask).sum() / mask.sum()
        adv = reverse_sum((r - mean) * mask / .1)
        rows.append(dict(failure_step=tau, raw_return=float((r[0]*mask[0]).sum()),
                         baseline=mean, first_advantage=adv[0, 0],
                         survivor_first_advantage=adv[1, 0]))
    write_csv(out / 'termination.csv', rows)
    # Marginal sensitivity of RTG to a globally broadcast trajectory scalar q.
    bcast = [dict(t=t, broadcast_weight=H-t, terminal_only_weight=1) for t in range(H)]
    write_csv(out / 'broadcast_weights.csv', bcast)
    return rows


def plots(out, metrics, profiles, error, learning, termination):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for method in METHODS:
        p = [r for r in profiles if r['method'] == method]
        means = np.array([[r['mean'] for r in p if r['seed'] == s] for s in sorted(set(r['seed'] for r in p))])
        axes[0].plot(range(H), means.mean(0), label=method)
    axes[0].plot(range(H), [p['truth'] for p in profiles[:H]], 'k--', label='analytic truth')
    axes[0].set(xlabel='Action time', ylabel='Mean policy gradient', title='Future risk reaches early actions')
    axes[0].legend(fontsize=8)
    axes[1].bar(METHODS, [np.mean([r['variance'] for r in metrics if r['method']==m]) for m in METHODS])
    axes[1].set(yscale='log', ylabel='Mean per-coordinate variance (G=4)', title='Variance; extra CF simulation cost')
    axes[1].tick_params(axis='x', rotation=25)
    shifts = sorted(set(r['shift'] for r in error))
    axes[2].plot(shifts, [np.mean([r['early_mean'] for r in error if r['shift']==s]) for s in shifts], 'o-')
    axes[2].axhline(error[0]['early_truth'], ls='--', color='k')
    axes[2].set(xlabel='Model wall-position error (m)', ylabel='Early-action gradient', title='Wrong model biases counterfactuals')
    fig.savefig(out / 'credit_mechanisms.png', dpi=170)
    fig.savefig(out / 'credit_mechanisms.svg')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for method in METHODS:
        subset = [r for r in learning if r['method']==method]
        times = sorted(set(r['iteration'] for r in subset))
        for ax, key in zip(axes, ['expected_return', 'crash_probability']):
            vals = np.array([[r[key] for r in subset if r['iteration']==t] for t in times])
            ax.plot(times, vals.mean(1), label=method)
            ax.fill_between(times, vals.mean(1)-vals.std(1,ddof=1), vals.mean(1)+vals.std(1,ddof=1), alpha=.12)
    axes[0].set(xlabel='Updates (equal factual trajectories)', ylabel='Analytic expected return')
    axes[1].set(xlabel='Updates (CF has extra rollouts)', ylabel='Analytic controllable crash probability')
    axes[1].legend(fontsize=8)
    fig.savefig(out / 'learning.png', dpi=170); fig.savefig(out / 'learning.svg'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot([r['failure_step'] for r in termination], [r['first_advantage'] for r in termination], 'o-')
    axes[0].set(xlabel='First failure step', ylabel='Failed trajectory first-token advantage', title='Feedback exists with shifted termination mask')
    axes[1].plot(range(H), H-np.arange(H), label='Broadcast + RTG')
    axes[1].plot(range(H), np.ones(H), label='Terminal-only scalar')
    axes[1].set(xlabel='Token time', ylabel='Weight on a trajectory-level scalar', title='Different temporal weights'); axes[1].legend()
    fig.savefig(out / 'source_mechanisms.png', dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'results')
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--groups', type=int, default=4096)
    parser.add_argument('--iterations', type=int, default=120)
    args = parser.parse_args()
    if args.seeds < 2 or args.groups < 2 or args.iterations < 1:
        parser.error('seeds/groups >= 2, iterations >= 1 required')
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    audit = source_audit(args.out, args.source_root)
    metrics, profiles, errors, truth = estimator_experiment(args.out, args.seeds, args.groups)
    learning = learning_experiment(args.out, args.seeds, args.iterations)
    termination = termination_and_broadcast(args.out)
    plots(args.out, metrics, profiles, errors, learning, termination)
    # Reference finite difference against exact analytic expected objective.
    eps = 1e-5; mu = np.zeros(H)
    fd = np.array([(true_objective_gradient(mu+eps*np.eye(H)[t])[0]-true_objective_gradient(mu-eps*np.eye(H)[t])[0])/(2*eps) for t in range(H)])
    np.testing.assert_allclose(fd, truth['gradient'], atol=1e-7)
    summary = dict(parameters=dict(H=H, dt=DT, v0=V0, sigma=SIGMA, wall=BARRIER,
        penalty=PENALTY, exogenous_p=EXO_P, progress_weight=PROGRESS, effort_weight=EFFORT,
        group_size=4, cf_alternatives=8, vd_scale=1., learning_rate=.01,
        seeds=args.seeds, groups_per_seed=args.groups, iterations=args.iterations),
        truth=truth, source_audit=audit['status'], finite_difference_max_error=float(np.max(np.abs(fd-truth['gradient']))),
        versions=dict(python=sys.version, numpy=np.__version__, matplotlib=matplotlib.__version__, platform=platform.platform()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seconds=time.monotonic()-start,
        aggregated={m:{k:float(np.mean([r[k] for r in metrics if r['method']==m])) for k in ['early_mean','variance','mse','mean_gradient_rmse','cosine']} for m in METHODS},
        final_learning={m:{k:float(np.mean([r[k] for r in learning if r['method']==m and r['iteration']==args.iterations])) for k in ['expected_return','crash_probability','rollout_equivalents']} for m in METHODS})
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
