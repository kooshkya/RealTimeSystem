#!/usr/bin/env python3
"""
Phase 2 Simulator — Adaptive Power Management for Real-Time Multi-Core Systems

Algorithms:
  1. bayesian_slack — Bayesian Slack-Based Alpha Control (Algorithm 2)
  2. static_high   — alpha = 1.0 always
  3. static_base   — alpha = alpha_base always
"""

import argparse
import json
import math
import os
import glob
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Power model
# ---------------------------------------------------------------------------
def energy_rate(alpha: float) -> float:
    return alpha + 2.0 * alpha ** 3


DEFAULT_ALPHA_LEVELS = [round(0.1 * i, 1) for i in range(2, 11)]


# ---------------------------------------------------------------------------
# Bayesian estimator (Gamma–Poisson)
# ---------------------------------------------------------------------------
class GroupLambdaEstimator:
    def __init__(self, a: float = 1.0, b: float = 10.0):
        self.a = a
        self.b = b
        self.wcet_sum = 0.0
        self.wcet_count = 0

    def observe_arrival(self, wcet: float):
        self.wcet_sum += wcet
        self.wcet_count += 1

    def update(self, count: int, window: float):
        if window > 0:
            self.a += count
            self.b += window

    @property
    def lambda_est(self) -> float:
        return self.a / self.b

    @property
    def wcet_mean(self) -> float:
        return self.wcet_sum / self.wcet_count if self.wcet_count > 0 else 5.0


# ---------------------------------------------------------------------------
# Slack-based alpha control (Algorithm 2)
# ---------------------------------------------------------------------------
def periodic_util(core) -> float:
    return sum(t['wcet'] / t['period'] for t in core['tasks'])


def update_alpha_slack_based(core, group_estimators, group_to_core):
    EPS = 0.05
    DELTA_UP = 0.05
    DELTA_DOWN = 0.02
    EMA_BETA = 0.9

    U_p = periodic_util(core)

    U_a = 0.0
    for g, cid in group_to_core.items():
        if cid == core['id']:
            est = group_estimators[g]
            U_a += est.lambda_est * est.wcet_mean

    slack = 1.0 - U_p - U_a

    if 'slack_ema' not in core:
        core['slack_ema'] = slack
    else:
        core['slack_ema'] = EMA_BETA * core['slack_ema'] + (1 - EMA_BETA) * slack

    if core['slack_ema'] < EPS:
        core['alpha'] = min(1.0, core['alpha'] + DELTA_UP)
    elif core['slack_ema'] > 2 * EPS:
        core['alpha'] = max(core['alpha_base'], core['alpha'] - DELTA_DOWN)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate(run_data: Dict[str, Any], algo: str, update_interval: float = 1.0):
    params = run_data['parameters']
    M = params['M']
    T_sim = params['sim_duration']
    true_lambda = params['aperiodic_lambda']

    periodic_by_id = {t['id']: t for t in run_data['periodic_tasks']}

    cores = []
    for cid, cdata in run_data['core_mapping'].items():
        tasks = []
        for tid in cdata['tasks']:
            t = periodic_by_id[tid]
            tasks.append({
                'id': tid,
                'period': float(t['period']),
                'wcet': float(t['wcet']),
                'next_release': 0.0,
                'abs_deadline': float(t['deadline']),
            })
        cores.append({
            'id': int(cid),
            'tasks': tasks,
            'periodic_queue': [],
            'aperiodic_queue': [],
            'current_periodic': None,
            'current_aperiodic': None,
            'remaining_periodic': 0.0,
            'remaining_aperiodic': 0.0,
            'alpha_base': cdata['alpha_base'],
            'alpha': cdata['alpha_base'],
            'energy': 0.0,
        })

    cores.sort(key=lambda c: c['id'])

    ap_pool = sorted([
        {
            'group': t['group'],
            'arrival': float(t['arrival_time']),
            'wcet': float(t['wcet']),
            'abs_deadline': float(t['abs_deadline']),
            'completed': False,
            'missed': False,
        }
        for t in run_data['aperiodic_tasks']
        if float(t['arrival_time']) < T_sim
    ], key=lambda x: x['arrival'])

    group_estimators = {}
    group_to_core = {}

    window_arrivals = {}
    window_start = 0.0
    next_update = update_interval

    def allocate_group(group):
        best = min(cores, key=periodic_util)
        group_to_core[group] = best['id']

    dt = 0.1
    t = 0.0
    ap_index = 0

    ap_total = len(ap_pool)
    ap_arrived = 0
    ap_completed = 0
    ap_missed = 0

    periodic_deadlines_total = 0
    periodic_deadlines_missed = 0

    total_energy = 0.0
    alpha_samples = []
    util_samples = []
    lambda_errors = []

    if algo == 'static_high':
        for c in cores:
            c['alpha'] = 1.0
    elif algo == 'static_base':
        for c in cores:
            c['alpha'] = c['alpha_base']

    while t < T_sim - 1e-9:
        while ap_index < len(ap_pool) and ap_pool[ap_index]['arrival'] <= t + 1e-9:
            ap = ap_pool[ap_index]
            ap_arrived += 1
            g = ap['group']
            if g not in group_estimators:
                group_estimators[g] = GroupLambdaEstimator()
                window_arrivals[g] = 0
                allocate_group(g)
            group_estimators[g].observe_arrival(ap['wcet'])
            window_arrivals[g] += 1
            core = cores[group_to_core[g]]
            core['aperiodic_queue'].append(ap)
            ap_index += 1

        if algo == 'bayesian_slack' and t >= next_update - 1e-9:
            window_len = t - window_start
            for g, est in group_estimators.items():
                est.update(window_arrivals[g], window_len)
                lambda_errors.append(abs(est.lambda_est - true_lambda))
                window_arrivals[g] = 0
            window_start = t
            next_update += update_interval

        core_utils = []
        for c in cores:
            if algo == 'bayesian_slack':
                update_alpha_slack_based(c, group_estimators, group_to_core)

            alpha = c['alpha']
            remaining_dt = dt
            time_used = 0.0

            for pt in c['tasks']:
                if abs(pt['next_release'] - t) < dt / 2:
                    c['periodic_queue'].append(pt)
                    pt['next_release'] += pt['period']
                    pt['abs_deadline'] = pt['next_release']
                    periodic_deadlines_total += 1

            c['periodic_queue'].sort(key=lambda x: x['abs_deadline'])

            for pt in c['periodic_queue'][:]:
                if pt['abs_deadline'] < t:
                    periodic_deadlines_missed += 1
                    c['periodic_queue'].remove(pt)

            while c['periodic_queue'] and remaining_dt > 1e-9:
                pt = c['periodic_queue'][0]
                if c['current_periodic'] is not pt:
                    c['current_periodic'] = pt
                    c['remaining_periodic'] = pt['wcet']
                    c['periodic_queue'].pop(0)

                exec_work = min(c['remaining_periodic'], alpha * remaining_dt)
                c['remaining_periodic'] -= exec_work
                exec_time = exec_work / alpha
                remaining_dt -= exec_time
                time_used += exec_time

                if c['remaining_periodic'] <= 1e-9:
                    c['current_periodic'] = None
                    break

            c['aperiodic_queue'].sort(key=lambda x: x['abs_deadline'])

            for ap in c['aperiodic_queue'][:]:
                if ap['abs_deadline'] < t and not ap['completed']:
                    ap['missed'] = True
                    ap_missed += 1
                    c['aperiodic_queue'].remove(ap)

            if remaining_dt > 1e-9 and c['aperiodic_queue']:
                ap = c['aperiodic_queue'][0]
                if c['current_aperiodic'] is not ap:
                    c['current_aperiodic'] = ap
                    c['remaining_aperiodic'] = ap['wcet']
                    c['aperiodic_queue'].pop(0)

                exec_work = min(c['remaining_aperiodic'], alpha * remaining_dt)
                c['remaining_aperiodic'] -= exec_work
                exec_time = exec_work / alpha
                remaining_dt -= exec_time
                time_used += exec_time

                if c['remaining_aperiodic'] <= 1e-9:
                    c['current_aperiodic']['completed'] = True
                    ap_completed += 1
                    c['current_aperiodic'] = None

            energy = energy_rate(alpha) * dt
            c['energy'] += energy
            total_energy += energy
            alpha_samples.append(alpha)
            core_utils.append(time_used / dt)

        util_samples.append(sum(core_utils) / len(core_utils))
        t = round(t + dt, 6)

    miss_rate = ap_missed / ap_total if ap_total > 0 else 0.0
    periodic_miss_rate = (
        periodic_deadlines_missed / periodic_deadlines_total
        if periodic_deadlines_total > 0 else 0.0
    )

    lambda_est_final = None
    if group_estimators:
        lambda_est_final = sum(e.lambda_est for e in group_estimators.values()) / len(group_estimators)

    return {
        'energy_total': total_energy,
        'ap_miss_rate': miss_rate,
        'periodic_miss_rate': periodic_miss_rate,
        'alpha_mean': sum(alpha_samples) / len(alpha_samples),
        'util_mean': sum(util_samples) / len(util_samples),
        'lambda_est_final': lambda_est_final,
        'lambda_est_error_mean': sum(lambda_errors) / len(lambda_errors) if lambda_errors else None,
    }


# ---------------------------------------------------------------------------
# Directory runner + CLI (unchanged behavior)
# ---------------------------------------------------------------------------
def run_directory(directory, algo, update_interval=1.0):
    files = sorted(glob.glob(os.path.join(directory, 'run_*.json')))
    results = []
    for f in files:
        with open(f) as fh:
            run_data = json.load(fh)
        results.append(simulate(run_data, algo, update_interval))

    def avg(k):
        vals = [r[k] for r in results if r[k] is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        'algorithm': algo,
        'num_runs': len(results),
        'energy_total_mean': avg('energy_total'),
        'ap_miss_rate_mean': avg('ap_miss_rate'),
        'periodic_miss_rate_mean': avg('periodic_miss_rate'),
        'alpha_mean': avg('alpha_mean'),
        'util_mean': avg('util_mean'),
        'lambda_est_final_mean': avg('lambda_est_final'),
        'lambda_est_error_mean': avg('lambda_est_error_mean'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument('--algo', default='all',
                        choices=['bayesian_slack', 'static_high', 'static_base', 'all'])
    parser.add_argument('--update_interval', type=float, default=1.0)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    algos = ['bayesian_slack', 'static_high', 'static_base'] if args.algo == 'all' else [args.algo]
    output = {}

    for a in algos:
        output[a] = run_directory(args.dir, a, args.update_interval)

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(output, f, indent=2)
    else:
        print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()