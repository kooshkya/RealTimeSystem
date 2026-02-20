#!/usr/bin/env python3
"""
Phase 2 Simulator — Adaptive Power Management for Real-Time Multi-Core Systems

Three scheduling algorithms:
  1. bayesian    — Gamma-Poisson online inference of lambda; adjusts alpha per core
  2. static_high — alpha = 1.0 always
  3. static_base — alpha = alpha_base always

Usage:
  python3 phase2_simulator.py --dir tasksets/M2/pm2_am1_u0_25 --algo all --out results.json
"""

import argparse
import json
import math
import os
import glob
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Power model
# ---------------------------------------------------------------------------
def energy_rate(alpha: float) -> float:
    """P(alpha) = alpha + 2*alpha^3"""
    return alpha + 2.0 * alpha ** 3


# ---------------------------------------------------------------------------
# Alpha levels (from JSON parameters, fallback to 0.2..1.0)
# ---------------------------------------------------------------------------
DEFAULT_ALPHA_LEVELS = [round(0.1 * i, 1) for i in range(2, 11)]


def select_alpha(alpha_base: float, lambda_est: float,
                 ap_wcet_mean: float, alpha_levels: List[float]) -> float:
    """Pick lowest alpha >= alpha_base + lambda_est * ap_wcet_mean."""
    required = alpha_base + lambda_est * ap_wcet_mean
    for a in sorted(alpha_levels):
        if a >= required - 1e-9:
            return a
    return max(alpha_levels)


# ---------------------------------------------------------------------------
# Bayesian Gamma-Poisson estimator
# ---------------------------------------------------------------------------
class BayesianLambdaEstimator:
    def __init__(self, a: float = 1.0, b: float = 10.0):
        self.a = a   # shape
        self.b = b   # rate

    def update(self, count: int, window: float):
        if window > 0:
            self.a += count
            self.b += window

    @property
    def mean(self) -> float:
        return self.a / self.b


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate(run_data: Dict[str, Any], algo: str,
             update_interval: float = 1.0) -> Dict[str, Any]:

    params = run_data['parameters']
    M = params['M']
    T_sim = params['sim_duration']
    true_lambda = params['aperiodic_lambda']
    alpha_levels = params.get('alpha_levels', DEFAULT_ALPHA_LEVELS)

    # Build task lookup by id
    periodic_by_id = {t['id']: t for t in run_data['periodic_tasks']}

    # Build core structures
    # core_mapping keys are strings "0", "1", ...
    core_mapping = run_data['core_mapping']
    cores = []
    for core_id_str, cdata in core_mapping.items():
        core_id = int(core_id_str)
        alpha_base = cdata['alpha_base']
        tasks = []
        for tid in cdata['tasks']:
            t = periodic_by_id[tid]
            tasks.append({
                'id': tid,
                'period': float(t['period']),
                'wcet': float(t['wcet']),
                'deadline': float(t['deadline']),
                # simulation state
                'next_release': 0.0,
                'abs_deadline': float(t['deadline']),  # first deadline
                'remaining': 0.0,
                'in_queue': False,
            })
        cores.append({
            'id': core_id,
            'alpha_base': alpha_base,
            'alpha': alpha_base,
            'tasks': tasks,
            'periodic_queue': [],   # list of task dicts ready to run
            'current_periodic': None,
            'remaining_periodic': 0.0,
            'aperiodic_queue': [],
            'current_aperiodic': None,
            'remaining_aperiodic': 0.0,
            'energy': 0.0,
        })

    # Sort cores by id
    cores.sort(key=lambda c: c['id'])

    # Aperiodic tasks pool (pre-generated, sorted by arrival)
    ap_pool = sorted([
        {
            'id': t['id'],
            'arrival': float(t['arrival_time']),
            'wcet': float(t['wcet']),
            'rel_deadline': float(t['deadline']),
            'abs_deadline': float(t['abs_deadline']),
            'completed': False,
            'missed': False,
            'assigned_core': None,
        }
        for t in run_data['aperiodic_tasks']
        if float(t['arrival_time']) < T_sim
    ], key=lambda x: x['arrival'])

    ap_wcet_mean = (sum(t['wcet'] for t in ap_pool) / len(ap_pool)) if ap_pool else 1.0

    # Set initial alpha
    def set_alphas(lambda_est=None):
        for c in cores:
            if algo == 'static_high':
                c['alpha'] = 1.0
            elif algo == 'static_base':
                c['alpha'] = c['alpha_base']
            elif algo == 'bayesian' and lambda_est is not None:
                c['alpha'] = select_alpha(c['alpha_base'], lambda_est, ap_wcet_mean, alpha_levels)
            else:
                c['alpha'] = c['alpha_base']

    set_alphas()

    # Route aperiodic to least-loaded core (by periodic utilization)
    def route_aperiodic(ap):
        best = min(cores, key=lambda c: sum(
            t['wcet'] / t['period'] for t in c['tasks']
        ))
        ap['assigned_core'] = best['id']
        best['aperiodic_queue'].append(ap)

    # Metrics
    total_energy = 0.0
    alpha_samples = []
    util_samples = []

    # Periodic miss tracking
    periodic_deadlines_total = 0
    periodic_deadlines_missed = 0

    # Aperiodic tracking
    ap_total = len(ap_pool)
    ap_arrived = 0  # how many have actually arrived
    ap_completed = 0
    ap_missed = 0

    # Bayesian
    bayes = BayesianLambdaEstimator(a=1.0, b=10.0)
    lambda_errors = []
    window_start = 0.0
    window_arrivals = 0
    next_update = update_interval

    ap_index = 0
    dt = 0.1
    t = 0.0

    while t < T_sim - 1e-9:

        # --- Arrive aperiodic tasks ---
        while ap_index < len(ap_pool) and ap_pool[ap_index]['arrival'] <= t + 1e-9:
            ap = ap_pool[ap_index]
            ap_arrived += 1
            route_aperiodic(ap)
            window_arrivals += 1
            ap_index += 1

        # --- Bayesian update ---
        if algo == 'bayesian' and t >= next_update - 1e-9:
            window_len = t - window_start
            bayes.update(window_arrivals, window_len)
            lambda_est = bayes.mean
            lambda_errors.append(abs(lambda_est - true_lambda))
            set_alphas(lambda_est)
            window_arrivals = 0
            window_start = t
            next_update += update_interval

        # --- Per-core tick ---
        core_utils = []
        for c in cores:
            alpha = c['alpha']
            remaining_dt = dt
            time_used = 0.0

            # Release periodic tasks
            for pt in c['tasks']:
                if abs(pt['next_release'] - t) < dt / 2.0:
                    # Check if previous instance missed deadline
                    # (if it was never completed — tracked via remaining)
                    c['periodic_queue'].append(pt)
                    pt['next_release'] += pt['period']
                    pt['abs_deadline'] = pt['next_release']  # deadline = next release
                    periodic_deadlines_total += 1

            # EDF sort
            c['periodic_queue'].sort(key=lambda x: x['abs_deadline'])

            # Check periodic deadline misses
            for pt in c['periodic_queue'][:]:
                if pt['abs_deadline'] < t - 1e-6:
                    periodic_deadlines_missed += 1
                    c['periodic_queue'].remove(pt)

            # Execute periodic (EDF)
            while c['periodic_queue'] and remaining_dt > 1e-9:
                pt = c['periodic_queue'][0]
                if c['current_periodic'] is not pt:
                    c['current_periodic'] = pt
                    c['remaining_periodic'] = pt['wcet'] / alpha
                    c['periodic_queue'].pop(0)

                exec_t = min(c['remaining_periodic'], remaining_dt)
                c['remaining_periodic'] -= exec_t
                remaining_dt -= exec_t
                time_used += exec_t

                if c['remaining_periodic'] <= 1e-9:
                    c['current_periodic'] = None
                    break

            # Execute aperiodic in slack
            c['aperiodic_queue'].sort(key=lambda x: x['abs_deadline'])

            # Check aperiodic deadline misses in queue
            for ap in c['aperiodic_queue'][:]:
                if ap['abs_deadline'] < t - 1e-6 and not ap['completed'] and not ap['missed']:
                    ap['missed'] = True
                    ap_missed += 1
                    c['aperiodic_queue'].remove(ap)

            if remaining_dt > 1e-9 and c['aperiodic_queue']:
                ap = c['aperiodic_queue'][0]
                if c['current_aperiodic'] is not ap:
                    c['current_aperiodic'] = ap
                    c['remaining_aperiodic'] = ap['wcet'] / alpha
                    c['aperiodic_queue'].pop(0)

                exec_t = min(c['remaining_aperiodic'], remaining_dt)
                c['remaining_aperiodic'] -= exec_t
                remaining_dt -= exec_t
                time_used += exec_t

                if c['remaining_aperiodic'] <= 1e-9:
                    c['current_aperiodic']['completed'] = True
                    ap_completed += 1
                    c['current_aperiodic'] = None

            # Check in-progress aperiodic deadline
            if (c['current_aperiodic'] is not None and
                    c['current_aperiodic']['abs_deadline'] < t - 1e-6 and
                    not c['current_aperiodic']['completed'] and
                    not c['current_aperiodic']['missed']):
                c['current_aperiodic']['missed'] = True
                ap_missed += 1
                c['current_aperiodic'] = None

            energy = energy_rate(alpha) * dt
            c['energy'] += energy
            total_energy += energy
            alpha_samples.append(alpha)
            core_utils.append(time_used / dt)

        util_samples.append(sum(core_utils) / len(core_utils) if core_utils else 0.0)
        t = round(t + dt, 6)

    # Final pass: any remaining aperiodics that never completed
    for c in cores:
        for ap in c['aperiodic_queue']:
            if not ap['completed'] and not ap['missed']:
                ap['missed'] = True
                ap_missed += 1
        if c['current_aperiodic'] and not c['current_aperiodic']['completed'] and not c['current_aperiodic']['missed']:
            c['current_aperiodic']['missed'] = True
            ap_missed += 1

    miss_rate = ap_missed / ap_total if ap_total > 0 else 0.0
    periodic_miss_rate = periodic_deadlines_missed / periodic_deadlines_total if periodic_deadlines_total > 0 else 0.0

    return {
        'energy_total': total_energy,
        'energy_per_core': [c['energy'] for c in cores],
        # Aperiodic metrics
        'ap_total': ap_total,
        'ap_arrived': ap_arrived,
        'ap_completed': ap_completed,
        'ap_missed': ap_missed,
        'ap_miss_rate': miss_rate,
        # Periodic metrics
        'periodic_deadlines_total': periodic_deadlines_total,
        'periodic_deadlines_missed': periodic_deadlines_missed,
        'periodic_miss_rate': periodic_miss_rate,
        # Alpha / util
        'alpha_mean': sum(alpha_samples) / len(alpha_samples) if alpha_samples else 0.0,
        'util_mean': sum(util_samples) / len(util_samples) if util_samples else 0.0,
        # Bayesian
        'lambda_true': true_lambda,
        'lambda_est_final': bayes.mean if algo == 'bayesian' else None,
        'lambda_est_error_mean': (sum(lambda_errors) / len(lambda_errors)) if lambda_errors else None,
    }


# ---------------------------------------------------------------------------
# Aggregate over a directory of JSON files
# ---------------------------------------------------------------------------
def run_directory(directory: str, algo: str,
                  update_interval: float = 1.0) -> Dict[str, Any]:
    files = sorted(glob.glob(os.path.join(directory, 'run_*.json')))
    if not files:
        raise FileNotFoundError(f"No run_*.json files in {directory}")

    results = []
    for fpath in files:
        with open(fpath) as f:
            run_data = json.load(f)
        r = simulate(run_data, algo, update_interval=update_interval)
        results.append(r)

    def avg(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def total(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(vals) if vals else None

    # Pull config from first file
    with open(files[0]) as f:
        sample = json.load(f)
    p = sample['parameters']

    return {
        'algorithm': algo,
        'directory': directory,
        'num_runs': len(results),
        # Config
        'M': p['M'],
        'n_periodic': p['n_periodic'],
        'n_aperiodic_groups': p['n_aperiodic_groups'],
        'n_aperiodic_instances': p['n_aperiodic_instances'],
        'U_total': p['U_total'],
        'sim_duration': p['sim_duration'],
        'lambda_true': p['aperiodic_lambda'],
        # Averaged metrics
        'energy_total_mean': avg('energy_total'),
        'ap_total_mean': avg('ap_total'),
        'ap_arrived_mean': avg('ap_arrived'),
        'ap_completed_mean': avg('ap_completed'),
        'ap_missed_mean': avg('ap_missed'),
        'ap_miss_rate_mean': avg('ap_miss_rate'),
        'periodic_deadlines_total_mean': avg('periodic_deadlines_total'),
        'periodic_deadlines_missed_mean': avg('periodic_deadlines_missed'),
        'periodic_miss_rate_mean': avg('periodic_miss_rate'),
        'alpha_mean': avg('alpha_mean'),
        'util_mean': avg('util_mean'),
        'lambda_est_final_mean': avg('lambda_est_final'),
        'lambda_est_error_mean': avg('lambda_est_error_mean'),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Phase 2 Simulator')
    parser.add_argument('--dir', required=True,
                        help='Directory with run_*.json files')
    parser.add_argument('--algo', default='all',
                        choices=['bayesian', 'static_high', 'static_base', 'all'])
    parser.add_argument('--update_interval', type=float, default=1,
                        help='Bayesian update window (default: 1)')
    parser.add_argument('--out', default=None,
                        help='Output JSON path (default: stdout)')
    args = parser.parse_args()

    algos = ['bayesian', 'static_high', 'static_base'] if args.algo == 'all' else [args.algo]

    output = {}
    for algo in algos:
        print(f"Running {algo} on {args.dir} ...")
        result = run_directory(args.dir, algo, args.update_interval)
        output[algo] = result
        print(f"  energy={result['energy_total_mean']:.2f}  "
              f"ap_miss={result['ap_miss_rate_mean']:.4f}  "
              f"periodic_miss={result['periodic_miss_rate_mean']:.4f}  "
              f"alpha={result['alpha_mean']:.3f}  "
              f"util={result['util_mean']:.3f}")
        if result['lambda_est_error_mean'] is not None:
            print(f"  lambda_error={result['lambda_est_error_mean']:.5f}  "
                  f"lambda_est_final={result['lambda_est_final_mean']:.5f}")

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {args.out}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
