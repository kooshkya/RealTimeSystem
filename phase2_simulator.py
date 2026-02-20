#!/usr/bin/env python3
"""
Phase 2 Simulator — Adaptive Power Management for Real-Time Multi-Core Systems

Three scheduling algorithms:
  1. bayesian    — Per-group Gamma-Poisson lambda; allocates entire group to one core
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


# ---------------------------------------------------------------------------
# Per-group Bayesian estimator
# ---------------------------------------------------------------------------
class GroupLambdaEstimator:
    """Gamma-Poisson conjugate for one aperiodic group."""
    def __init__(self, a: float = 1.0, b: float = 10.0):
        self.a = a   # shape
        self.b = b   # rate
        self.wcet_sum = 0.0
        self.wcet_count = 0

    def observe_arrival(self, wcet: float):
        """Called when a task from this group arrives."""
        self.wcet_sum += wcet
        self.wcet_count += 1

    def update(self, count: int, window: float):
        """Bayesian update at end of window."""
        if window > 0:
            self.a += count
            self.b += window

    @property
    def lambda_est(self) -> float:
        return self.a / self.b

    @property
    def wcet_mean(self) -> float:
        if self.wcet_count == 0:
            return 5.0  # prior: midpoint of [1,10]
        return self.wcet_sum / self.wcet_count


# ---------------------------------------------------------------------------
# Lightweight EDF schedulability check
# ---------------------------------------------------------------------------
def compute_min_alpha_edf(tasks: List[Dict], alpha_levels: List[float]) -> float:
    """
    Find minimum alpha such that sum(C_i / (alpha * T_i)) <= 1.
    tasks = [{'wcet': ..., 'period': ...}, ...]
    Returns: lowest alpha in alpha_levels satisfying the constraint.
    """
    if not tasks:
        return min(alpha_levels)
    
    # U_total = sum(C_i / T_i)
    U_total = sum(t['wcet'] / t['period'] for t in tasks)
    # We need: U_total / alpha <= 1  =>  alpha >= U_total
    required_alpha = U_total
    
    for a in sorted(alpha_levels):
        if a >= required_alpha - 1e-9:
            return a
    return max(alpha_levels)


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
                'next_release': 0.0,
                'abs_deadline': float(t['deadline']),
                'remaining': 0.0,
                'in_queue': False,
            })
        cores.append({
            'id': core_id,
            'alpha_base': alpha_base,
            'alpha': alpha_base,
            'tasks': tasks,
            # Virtual periodic tasks for aperiodic groups
            'virtual_tasks': [],  # [{'group': 'A0', 'period': 1/lambda, 'wcet': C_mean}]
            'periodic_queue': [],
            'current_periodic': None,
            'remaining_periodic': 0.0,
            'aperiodic_queue': [],
            'current_aperiodic': None,
            'remaining_aperiodic': 0.0,
            'energy': 0.0,
        })

    cores.sort(key=lambda c: c['id'])

    # Aperiodic tasks pool
    ap_pool = sorted([
        {
            'id': t['id'],
            'group': t['group'],
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

    # Per-group estimators (shared across cores)
    group_estimators = {}  # {group_id: GroupLambdaEstimator}
    group_to_core = {}     # {group_id: core_id}

    # Bayesian update tracking
    window_start = 0.0
    window_arrivals = {}  # {group_id: count}
    next_update = update_interval

    def allocate_group_to_core(group_id: str):
        """Allocate entire group to the emptiest core (by total utilization including virtual tasks)."""
        def core_utilization(c):
            U_periodic = sum(t['wcet'] / t['period'] for t in c['tasks'])
            U_virtual = sum(vt['wcet'] / vt['period'] for vt in c['virtual_tasks'])
            return U_periodic + U_virtual
        
        best = min(cores, key=core_utilization)
        group_to_core[group_id] = best['id']
        
        # Create virtual periodic task for this group
        estimator = group_estimators[group_id]
        period = 1.0 / estimator.lambda_est if estimator.lambda_est > 1e-9 else 1e6
        wcet = estimator.wcet_mean
        
        best['virtual_tasks'].append({
            'group': group_id,
            'period': period,
            'wcet': wcet,
        })
        
        # Recompute alpha for this core
        if algo == 'bayesian':
            all_tasks = [
                {'wcet': t['wcet'], 'period': t['period']} for t in best['tasks']
            ] + [
                {'wcet': vt['wcet'], 'period': vt['period']} for vt in best['virtual_tasks']
            ]
            best['alpha'] = compute_min_alpha_edf(all_tasks, alpha_levels)

    def update_group_virtual_task(group_id: str):
        """Update the virtual task parameters for a group and recompute core alpha."""
        core_id = group_to_core.get(group_id)
        if core_id is None:
            return
        
        core = next(c for c in cores if c['id'] == core_id)
        estimator = group_estimators[group_id]
        
        # Find and update the virtual task
        for vt in core['virtual_tasks']:
            if vt['group'] == group_id:
                vt['period'] = 1.0 / estimator.lambda_est if estimator.lambda_est > 1e-9 else 1e6
                vt['wcet'] = estimator.wcet_mean
                break
        
        # Recompute alpha
        if algo == 'bayesian':
            all_tasks = [
                {'wcet': t['wcet'], 'period': t['period']} for t in core['tasks']
            ] + [
                {'wcet': vt['wcet'], 'period': vt['period']} for vt in core['virtual_tasks']
            ]
            core['alpha'] = compute_min_alpha_edf(all_tasks, alpha_levels)

    def route_aperiodic(ap):
        """Route aperiodic to its group's assigned core."""
        group_id = ap['group']
        
        # Initialize estimator if first task from this group
        if group_id not in group_estimators:
            group_estimators[group_id] = GroupLambdaEstimator(a=1.0, b=10.0)
            window_arrivals[group_id] = 0
        
        estimator = group_estimators[group_id]
        estimator.observe_arrival(ap['wcet'])
        window_arrivals[group_id] += 1
        
        # Allocate group to core if first arrival
        if group_id not in group_to_core:
            allocate_group_to_core(group_id)
        
        # Route to assigned core
        core_id = group_to_core[group_id]
        core = cores[core_id]  # Direct indexing since core['id'] == list index
        ap['assigned_core'] = core_id
        core['aperiodic_queue'].append(ap)

    # Set initial alpha
    for c in cores:
        if algo == 'static_high':
            c['alpha'] = 1.0
        elif algo == 'static_base':
            c['alpha'] = c['alpha_base']
        else:
            c['alpha'] = c['alpha_base']

    # Metrics
    total_energy = 0.0
    alpha_samples = []
    util_samples = []
    periodic_deadlines_total = 0
    periodic_deadlines_missed = 0
    ap_total = len(ap_pool)
    ap_arrived = 0
    ap_completed = 0
    ap_missed = 0
    lambda_errors = []

    ap_index = 0
    dt = 0.1
    t = 0.0

    while t < T_sim - 1e-9:

        # --- Arrive aperiodic tasks ---
        while ap_index < len(ap_pool) and ap_pool[ap_index]['arrival'] <= t + 1e-9:
            ap = ap_pool[ap_index]
            ap_arrived += 1
            route_aperiodic(ap)
            ap_index += 1

        # --- Bayesian update (per group) ---
        if algo == 'bayesian' and t >= next_update - 1e-9:
            window_len = t - window_start
            for group_id, estimator in group_estimators.items():
                count = window_arrivals.get(group_id, 0)
                estimator.update(count, window_len)
                window_arrivals[group_id] = 0
                
                # Update virtual task and recompute alpha
                update_group_virtual_task(group_id)
                
                # Track lambda error
                lambda_errors.append(abs(estimator.lambda_est - true_lambda))
            
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
                    c['periodic_queue'].append(pt)
                    pt['next_release'] += pt['period']
                    pt['abs_deadline'] = pt['next_release']
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

    # Aggregate lambda estimate
    lambda_est_final = None
    if group_estimators:
        lambda_est_final = sum(e.lambda_est for e in group_estimators.values()) / len(group_estimators)

    return {
        'energy_total': total_energy,
        'energy_per_core': [c['energy'] for c in cores],
        'ap_total': ap_total,
        'ap_arrived': ap_arrived,
        'ap_completed': ap_completed,
        'ap_missed': ap_missed,
        'ap_miss_rate': miss_rate,
        'periodic_deadlines_total': periodic_deadlines_total,
        'periodic_deadlines_missed': periodic_deadlines_missed,
        'periodic_miss_rate': periodic_miss_rate,
        'alpha_mean': sum(alpha_samples) / len(alpha_samples) if alpha_samples else 0.0,
        'util_mean': sum(util_samples) / len(util_samples) if util_samples else 0.0,
        'lambda_true': true_lambda,
        'lambda_est_final': lambda_est_final,
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
        'M': p['M'],
        'n_periodic': p['n_periodic'],
        'n_aperiodic_groups': p['n_aperiodic_groups'],
        'n_aperiodic_instances': p['n_aperiodic_instances'],
        'U_total': p['U_total'],
        'sim_duration': p['sim_duration'],
        'lambda_true': p['aperiodic_lambda'],
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
