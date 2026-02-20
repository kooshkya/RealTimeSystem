"""
Phase 1: Offline Task Set Generation and Base Frequency Calculation
=================================================================
Generates periodic tasks (UUnifast-Discard) and aperiodic tasks (Poisson),
maps periodic tasks to cores using WFD or BFD, computes base alpha (frequency
scaling factor) per core via EDF schedulability analysis, and saves everything
to a JSON file.

Usage:
    python phase1_generator.py --M 4 --periodic_mult 3 --aperiodic_mult 2
                               --util_coeff 0.75 --mapping wfd --out taskset.json

Arguments:
    M               : number of cores
    periodic_mult   : multiplier for periodic task count  (n_p = periodic_mult * M)
    aperiodic_mult  : multiplier for aperiodic task count (n_a = aperiodic_mult * M)
    util_coeff      : total utilization coefficient (e.g. 0.75 → U_total = 0.75 * M)
    mapping         : 'wfd' (Worst Fit Decreasing) or 'bfd' (Best Fit Decreasing)
    out             : output JSON file path
"""

import argparse
import json
import math
import random
from math import gcd
from functools import reduce

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PERIODS = [10, 20, 30, 40, 50]            # Discrete period set
ALPHA_LEVELS = [round(a * 0.1, 1) for a in range(2, 11)]  # {0.2, 0.3, ..., 1.0}
APERIODIC_LAMBDA = 0.05   # default Poisson arrival rate (tasks per time unit)
                           # sensible: ~1 task per 20 time units

# ---------------------------------------------------------------------------
# 1. UUnifast-Discard – periodic task generation
# ---------------------------------------------------------------------------

def uunifast_discard(n: int, U_total: float, max_u: float = 1.0) -> list[float]:
    """
    Generate n utilisation values summing to U_total using UUnifast.
    Discards any sample where any individual utilisation > max_u.
    max_u defaults to 1.0 (EDF per-task feasibility on a single core).
    """
    while True:
        utils = []
        cumsum = U_total
        for i in range(1, n):
            next_sum = cumsum * (random.random() ** (1.0 / (n - i)))
            utils.append(cumsum - next_sum)
            cumsum = next_sum
        utils.append(cumsum)
        if all(u <= max_u for u in utils):
            return utils


def generate_periodic_tasks(n: int, U_total: float) -> list[dict]:
    """
    Generate n periodic tasks with total utilisation U_total.
    Period is chosen uniformly from PERIODS; WCET is derived from U * T.
    Implicit-deadline assumed: D_i = T_i.
    """
    utils = uunifast_discard(n, U_total)
    tasks = []
    for i, u in enumerate(utils):
        T = random.choice(PERIODS)
        C = max(1, round(u * T))          # WCET ≥ 1
        actual_u = C / T
        tasks.append({
            "id": f"P{i}",
            "type": "periodic",
            "period": T,
            "wcet": C,
            "deadline": T,
            "utilization": actual_u,
        })
    return tasks


# ---------------------------------------------------------------------------
# 2. Aperiodic task generation (Poisson arrivals)
# ---------------------------------------------------------------------------
def generate_aperiodic_tasks(n_groups: int, sim_duration: float,
                              lam: float = APERIODIC_LAMBDA) -> list[dict]:
    """
    Generate aperiodic task instances from n_groups independent Poisson processes,
    each with arrival rate `lam`, running over [0, sim_duration].
    Each instance is tagged with its group ID (e.g., 'A0', 'A1', ...).
    Returns all instances sorted by arrival time, with unique instance IDs.
    """
    all_tasks = []
    for g in range(n_groups):
        arrival = 0.0
        instance = 0
        while True:
            inter = random.expovariate(lam)
            arrival += inter
            if arrival > sim_duration:
                break
            C = random.randint(1, 10)
            D = random.randint(C, 3 * C)
            all_tasks.append({
                "id": f"A{g}_{instance}",   # group_instance
                "group": f"A{g}",            # which aperiodic task type
                "type": "aperiodic",
                "arrival_time": round(arrival, 4),
                "wcet": C,
                "deadline": D,
                "abs_deadline": round(arrival + D, 4),
            })
            instance += 1

    # Sort all instances by arrival time
    all_tasks.sort(key=lambda t: t["arrival_time"])
    return all_tasks


# ---------------------------------------------------------------------------
# 3. Mapping algorithms – WFD and BFD
# ---------------------------------------------------------------------------

def wfd_mapping(tasks: list[dict], M: int) -> dict[int, list[dict]]:
    """
    Worst Fit Decreasing: sort tasks by utilisation descending;
    assign each task to the core with the HIGHEST remaining capacity
    (least loaded → most remaining).
    """
    sorted_tasks = sorted(tasks, key=lambda t: t["utilization"], reverse=True)
    core_util = [0.0] * M
    cores: dict[int, list[dict]] = {i: [] for i in range(M)}

    for task in sorted_tasks:
        # Pick core with lowest current utilisation (worst fit = most room)
        chosen = int(np.argmin(core_util))
        cores[chosen].append(task)
        core_util[chosen] += task["utilization"]

    return cores


def bfd_mapping(tasks: list[dict], M: int) -> dict[int, list[dict]]:
    """
    Best Fit Decreasing: sort tasks by utilisation descending;
    assign each task to the core with the LEAST remaining capacity that
    still fits (tightest fit), falling back to the least loaded core.
    """
    sorted_tasks = sorted(tasks, key=lambda t: t["utilization"], reverse=True)
    core_util = [0.0] * M
    cores: dict[int, list[dict]] = {i: [] for i in range(M)}

    for task in sorted_tasks:
        u = task["utilization"]
        # Find cores that can accommodate the task (util + u ≤ 1)
        feasible = [(core_util[i], i) for i in range(M) if core_util[i] + u <= 1.0]
        if feasible:
            # Best fit: core with highest current utilisation among feasible
            chosen = max(feasible, key=lambda x: x[0])[1]
        else:
            # Overflow: put on least loaded core
            chosen = int(np.argmin(core_util))
        cores[chosen].append(task)
        core_util[chosen] += u

    return cores


# ---------------------------------------------------------------------------
# 4. Base frequency (alpha) calculation using EDF schedulability test
# ---------------------------------------------------------------------------

def lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def hyperperiod(tasks: list[dict]) -> int:
    periods = [t["period"] for t in tasks]
    return reduce(lcm, periods)


def edf_schedulable(tasks: list[dict], alpha: float) -> bool:
    """
    EDF schedulability test for a set of periodic tasks running at
    frequency scaling factor alpha.

    At frequency alpha:
      - Effective WCET of task i becomes C_i / alpha
        (lower frequency → more real-time to execute C_i units of work).
      - Total utilisation = sum(C_i / (alpha * T_i)) = (1/alpha) * sum(u_i).

    For implicit-deadline periodic tasks under EDF:
        feasible  iff  sum(C_i_eff / T_i) ≤ 1
                  iff  sum(C_i / (alpha * T_i)) ≤ 1
                  iff  U_core / alpha ≤ 1
                  iff  alpha ≥ U_core

    So the minimum alpha is ceil(U_core * 10) / 10 mapped to ALPHA_LEVELS.
    We keep the full demand-bound check for generality (handles constrained
    deadlines if needed).
    """
    if not tasks:
        return True

    # Compute effective utilisation at this alpha
    eff_util = sum(t["wcet"] / (alpha * t["period"]) for t in tasks)
    if eff_util > 1.0 + 1e-9:
        return False

    # Demand-bound function check over hyper-period
    H = hyperperiod(tasks)
    # Cap hyper-period to avoid combinatorial explosion
    H = min(H, 10_000)

    for t_check in range(1, H + 1):
        dbf = sum(
            max(0, (math.floor((t_check - t["deadline"]) / t["period"]) + 1)
                * (t["wcet"] / alpha))
            for t in tasks
        )
        if dbf > t_check + 1e-9:
            return False
    return True


def compute_base_alpha(tasks: list[dict]) -> float:
    """
    Find the lowest alpha in ALPHA_LEVELS such that the task set is
    EDF-schedulable on a single core running at that frequency level.
    """
    if not tasks:
        return ALPHA_LEVELS[0]

    # Fast lower bound from utilisation
    U_core = sum(t["utilization"] for t in tasks)

    for alpha in ALPHA_LEVELS:
        if alpha >= U_core - 1e-9:   # utilisation necessary condition
            if edf_schedulable(tasks, alpha):
                return alpha

    # If nothing works, return maximum
    return ALPHA_LEVELS[-1]


# ---------------------------------------------------------------------------
# 5. Main orchestration
# ---------------------------------------------------------------------------

def build_taskset(M: int, periodic_mult: int, aperiodic_mult: int,
                  util_coeff: float, mapping: str = "wfd",
                  lam: float = APERIODIC_LAMBDA) -> dict:
    """
    Build and return a complete task-set dictionary ready for JSON serialisation.
    """
    n_p = periodic_mult * M
    n_a = aperiodic_mult * M
    U_total = util_coeff * M             # aggregate utilisation across all cores

    # --- Generate tasks -------------------------------------------------------
    periodic_tasks = generate_periodic_tasks(n_p, U_total)

    # Estimate simulation duration for aperiodic generation
    # Use 20 × largest hyper-period as reference (computed after task gen)
    H = hyperperiod(periodic_tasks) if periodic_tasks else 100
    sim_duration = 20 * H
    aperiodic_tasks = generate_aperiodic_tasks(n_a, sim_duration, lam)

    if mapping.lower() == "bfd":
        core_map = bfd_mapping(periodic_tasks, M)
    else:
        core_map = wfd_mapping(periodic_tasks, M)

    # --- Compute base alpha per core ------------------------------------------
    core_info = {}
    for core_id, tasks in core_map.items():
        alpha_base = compute_base_alpha(tasks)
        U_core = sum(t["utilization"] for t in tasks)
        core_info[core_id] = {
            "tasks": [t["id"] for t in tasks],
            "utilization": round(U_core, 6),
            "alpha_base": alpha_base,
        }

    # --- Assemble output ------------------------------------------------------
    return {
        "parameters": {
            "M": M,
            "periodic_mult": periodic_mult,
            "aperiodic_mult": aperiodic_mult,
            "n_periodic": n_p,
            "n_aperiodic_groups": n_a,         
            "n_aperiodic_instances": len(aperiodic_tasks), 
            "util_coeff": util_coeff,
            "U_total": round(U_total, 6),
            "sim_duration": sim_duration,
            "mapping_algorithm": mapping.upper(),
            "aperiodic_lambda": lam,
            "alpha_levels": ALPHA_LEVELS,
        },
        "periodic_tasks": periodic_tasks,
        "aperiodic_tasks": aperiodic_tasks,
        "core_mapping": core_info,
    }


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Generate task set and compute base alpha per core."
    )
    parser.add_argument("--M", type=int, default=4,
                        help="Number of cores (default: 4)")
    parser.add_argument("--periodic_mult", type=int, default=3,
                        choices=[2, 3, 4],
                        help="Periodic task count = mult × M (default: 3)")
    parser.add_argument("--aperiodic_mult", type=int, default=1,
                        choices=[1, 2],
                        help="Aperiodic task count = mult × M (default: 1)")
    parser.add_argument("--util_coeff", type=float, default=0.5,
                        help="Total utilisation = coeff × M (default: 0.5)")
    parser.add_argument("--mapping", type=str, default="wfd",
                        choices=["wfd", "bfd"],
                        help="Mapping algorithm: wfd or bfd (default: wfd)")
    parser.add_argument("--lam", type=float, default=APERIODIC_LAMBDA,
                        help=f"Poisson arrival rate λ (default: {APERIODIC_LAMBDA})")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--out", type=str, default="taskset.json",
                        help="Output JSON file path (default: taskset.json)")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    print(f"Generating task set: M={args.M}, n_p={args.periodic_mult}×M, "
          f"n_a={args.aperiodic_mult}×M, U={args.util_coeff}×M, "
          f"mapping={args.mapping.upper()}")

    data = build_taskset(
        M=args.M,
        periodic_mult=args.periodic_mult,
        aperiodic_mult=args.aperiodic_mult,
        util_coeff=args.util_coeff,
        mapping=args.mapping,
        lam=args.lam,
    )

    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Task set saved to '{args.out}'")
    print(f"\n--- Core Summary ---")
    for core_id, info in data["core_mapping"].items():
        print(f"  Core {core_id}: tasks={info['tasks']}, "
              f"U={info['utilization']:.4f}, α_base={info['alpha_base']}")

    print(f"\n--- Task Summary ---")
    print(f"  Periodic  tasks : {data['parameters']['n_periodic']}")
    print(f"  Aperiodic tasks : {data['parameters']['n_aperiodic']}")
    print(f"  Total U         : {data['parameters']['U_total']:.4f}")
    print(f"  Hyper-period    : {data['periodic_tasks'][0]['period'] if data['periodic_tasks'] else 'N/A'} "
          f"(see JSON for full set)")
    print(f"  Sim duration    : {data['parameters']['sim_duration']} time units")


if __name__ == "__main__":
    main()
