
import csv
import os
import sys
import time
import traceback
from datetime import datetime, timezone

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'core'))

from sage.all import GF, vector, matrix, set_random_seed

from instances_generator import (
    generate_random_monomial,
    generate_bit_channel_hint,
)
from LEP_prediction_and_repair_v2 import (
    compute_posterior_table,
    monomial_approximation,
    support,
    active_error_set,
    set_verbose,
    pyrandom_sample,
)
from posterior_repair_radius import (
    active_row_error_probabilities,
    posterior_repair_radius,
)

set_verbose(False)  # this script prints its own one-line-per-instance summary instead

CSV_COLUMNS = [
    'n', 'k', 'q', 'alpha', 'beta', 'weight_factor', 'seed', 'config_source',
    'target_weight', 'v_weight', 't_active', 'rows_qhat_correct',
    'r_v_true', 'mean_est_r',
    'rmax_90', 'coverage_90',
    'rmax_95', 'coverage_95',
    'rmax_99', 'coverage_99',
    'elapsed_posterior_s', 'elapsed_hungarian_s', 'elapsed_estimate_s',
    'error',
]

DELTAS = {'90': 0.10, '95': 0.05, '99': 0.01}


# ---------------------------------------------------------------------------
# Instance construction with a planted low-weight codeword.
# ---------------------------------------------------------------------------

def gilbert_varshamov_bound(n, k, q):
    """
    Computes the Gilbert-Varshamov bound: the largest d such that

        sum_{i=0}^{d-2} C(n-1, i) * (q-1)^i  <  q^(n-k)

    Copied from run_lep_experiments.py (same formula used throughout
    rmax_approximation.md as the stand-in for the enumeration module's
    target weight theta_enum).

    :param n: code length
    :param k: code dimension
    :param q: field size
    :return: the Gilbert-Varshamov bound, an integer >= 1
    """
    from math import comb
    capacity = q ** (n - k)
    cumulative = 0
    d = 1
    while True:
        if cumulative >= capacity:
            return d - 1
        cumulative += comb(n - 1, d - 1) * (q - 1) ** (d - 1)
        d += 1


def random_weight_vector(F, n, weight):
    """
    Samples a uniformly random vector in F^n of Hamming weight exactly
    `weight` (identical to planted_codeword_repair_test.py's function of
    the same name).

    :param F: the base field, e.g. GF(q)
    :param n: the vector length
    :param weight: the desired exact Hamming weight, 0 <= weight <= n
    :return: a vector in F^n of Hamming weight exactly `weight`
    """
    v = vector(F, n)
    positions = pyrandom_sample(range(n), weight)
    for pos in positions:
        a = F.random_element()
        while a == 0:
            a = F.random_element()
        v[pos] = a
    return v


def build_generator_with_planted_low_weight_row(n, k, q, target_weight):
    """
    Builds a k x n generator matrix G over GF(q) whose row space is
    guaranteed to contain a codeword of Hamming weight exactly
    target_weight, planted as row 0 (identical to
    planted_codeword_repair_test.py's function of the same name).

    :param n: code length
    :param k: code dimension
    :param q: field size
    :param target_weight: the exact Hamming weight of the planted row
    :return: a tuple (G, v0)
    """
    F = GF(q)
    v0 = random_weight_vector(F, n, target_weight)
    rows = [v0]

    while len(rows) < k:
        candidate = vector(F, [F.random_element() for _ in range(n)])
        trial = matrix(F, rows + [candidate])
        if trial.rank() == len(rows) + 1:
            rows.append(candidate)

    G = matrix(F, rows)
    return G, v0


def generate_noisy_LCE_instance_from_G1(G1, q, alpha, beta, is_monomial=True):
    """
    Builds (G2, Q, Q_noisy) from an already-constructed G1 (identical to
    planted_codeword_repair_test.py's function of the same name).

    :param G1: a k x n generator matrix of C, over GF(q)
    :param q: field size
    :param alpha: Pr[bit 1 -> 0]
    :param beta: Pr[bit 0 -> 1]
    :param is_monomial: True for LEP (random monomial secret)
    :return: a tuple (G2, Q, Q_noisy)
    """
    F = G1.base_ring()
    n = G1.ncols()

    Q = generate_random_monomial(n, q, is_permutation=not is_monomial)
    G2 = (G1 * Q).rref()

    Q_noisy = generate_bit_channel_hint(Q, q, alpha, beta)
    Q_noisy = matrix(F, Q_noisy)

    return G2, Q, Q_noisy


# ---------------------------------------------------------------------------
# One instance: build it, compute Q_hat, compare true r(v) against the
# estimated repair radius at each confidence level.
# ---------------------------------------------------------------------------

def run_one_instance(n, k, q, alpha, beta, weight_factor, seed):
    """
    Builds one planted-codeword instance and computes both the true active
    error dimension r(v0) and the estimated PosteriorRepairRadius at 90%,
    95% and 99% confidence.

    :param n, k, q: code and field parameters
    :param alpha, beta: bit-flip channel parameters
    :param weight_factor: multiplies the Gilbert-Varshamov bound to get
        the planted codeword's exact target weight (matches
        planted_codeword_repair_test.py's convention)
    :param seed: Sage random seed, for reproducibility
    :return: a dict with every column in CSV_COLUMNS filled in ('error' is
        the empty string on success, or a short message on failure - in
        which case every other numeric field is left as None)
    """
    row = {col: None for col in CSV_COLUMNS}
    row.update(n=n, k=k, q=q, alpha=alpha, beta=beta,
               weight_factor=weight_factor, seed=seed, error='')

    try:
        set_random_seed(seed)
        F = GF(q)

        gv_bound = gilbert_varshamov_bound(n, k, q)
        target_weight = max(1, round(gv_bound * weight_factor))
        row['target_weight'] = target_weight

        G1, v0 = build_generator_with_planted_low_weight_row(n, k, q, target_weight)
        row['v_weight'] = v0.hamming_weight()

        G2, Q, Q_noisy = generate_noisy_LCE_instance_from_G1(G1, q, alpha, beta, is_monomial=True)

        t0 = time.time()
        posterior_table = compute_posterior_table(Q_noisy, alpha, beta, is_permutation=False)
        elapsed_posterior = time.time() - t0

        t0 = time.time()
        Q_hat, S, D_loc, pi = monomial_approximation(posterior_table, F)
        elapsed_hungarian = time.time() - t0

        rows_correct = sum(1 for i in range(n) if list(Q_hat[i]) == list(Q[i]))

        A = support(v0)
        r_v_true = len(active_error_set(v0, Q, Q_hat))

        t0 = time.time()
        p_dict = active_row_error_probabilities(posterior_table, pi, A)
        p_list = [p_dict[i] for i in A]
        mean_est_r = sum(p_list)

        results = {}
        for label, delta in DELTAS.items():
            r_max, _D = posterior_repair_radius(p_list, delta=delta)
            results[label] = r_max
        elapsed_estimate = time.time() - t0

        row.update(
            t_active=len(A),
            rows_qhat_correct=rows_correct,
            r_v_true=r_v_true,
            mean_est_r=mean_est_r,
            rmax_90=results['90'], coverage_90=int(r_v_true <= results['90']),
            rmax_95=results['95'], coverage_95=int(r_v_true <= results['95']),
            rmax_99=results['99'], coverage_99=int(r_v_true <= results['99']),
            elapsed_posterior_s=elapsed_posterior,
            elapsed_hungarian_s=elapsed_hungarian,
            elapsed_estimate_s=elapsed_estimate,
        )

    except Exception as exc:  
        row['error'] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    return row



N_K_PAIRS = [(32, 16), (64, 32), (128, 64)]
Q_VALUES = [7, 29, 127]
ALPHA_BETA_PAIRS = [(0.01, 0.03), (0.01, 0.05), (0.02, 0.08)]
WEIGHT_FACTORS = [1.1]
SEEDS_PER_COMBO = 5
BASE_SEED = 20260831  # offset so seeds don't collide across combos

NAMED_INSTANCES = [
    (128, 64, 127),
    (252, 126, 127),
    (548, 274, 127),
]
NAMED_SEEDS_PER_COMBO = 3


def build_sweep_configs():
    """
    Expands N_K_PAIRS x Q_VALUES x ALPHA_BETA_PAIRS x WEIGHT_FACTORS x
    SEEDS_PER_COMBO into a flat list of instance configs (the general grid),
    then appends NAMED_INSTANCES x ALPHA_BETA_PAIRS x WEIGHT_FACTORS x
    NAMED_SEEDS_PER_COMBO on top (the specific test points requested above).

    :return: a list of (n, k, q, alpha, beta, weight_factor, seed, source)
        tuples, source in {'grid', 'named'}
    """
    configs = []
    seed_counter = 0

    for (n, k) in N_K_PAIRS:
        for q in Q_VALUES:
            for (alpha, beta) in ALPHA_BETA_PAIRS:
                for weight_factor in WEIGHT_FACTORS:
                    for _ in range(SEEDS_PER_COMBO):
                        seed = BASE_SEED + seed_counter
                        seed_counter += 1
                        configs.append((n, k, q, alpha, beta, weight_factor, seed, 'grid'))

    for (n, k, q) in NAMED_INSTANCES:
        for (alpha, beta) in ALPHA_BETA_PAIRS:
            for weight_factor in WEIGHT_FACTORS:
                for _ in range(NAMED_SEEDS_PER_COMBO):
                    seed = BASE_SEED + seed_counter
                    seed_counter += 1
                    configs.append((n, k, q, alpha, beta, weight_factor, seed, 'named'))

    return configs


def main(output_csv='csv_results/rmax_posterior_results.csv'):
    """
    Runs every instance in build_sweep_configs(), writing one CSV row per
    instance to output_csv as soon as it completes (so a long sweep can be
    interrupted without losing earlier results).

    :param output_csv: path to write the results CSV to (relative to this
        script's directory unless an absolute path is given)
    """
    configs = build_sweep_configs()
    out_path = output_csv if os.path.isabs(output_csv) else os.path.join(_THIS_DIR, output_csv)

    print(f"Running {len(configs)} instances; writing to {out_path}")
    t_start = time.time()

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for idx, (n, k, q, alpha, beta, weight_factor, seed, source) in enumerate(configs):
            t0 = time.time()
            row = run_one_instance(n, k, q, alpha, beta, weight_factor, seed)
            row['config_source'] = source
            elapsed = time.time() - t0

            writer.writerow(row)
            f.flush()

            if row['error']:
                print(f"[{idx + 1}/{len(configs)}] ({source}) n={n} k={k} q={q} alpha={alpha} beta={beta} "
                      f"seed={seed} -> ERROR: {row['error']} ({elapsed:.2f}s)")
            else:
                print(f"[{idx + 1}/{len(configs)}] ({source}) n={n} k={k} q={q} alpha={alpha} beta={beta} "
                      f"seed={seed} -> t={row['t_active']} r_v={row['r_v_true']} "
                      f"E[r]={row['mean_est_r']:.2f} "
                      f"Rmax(90/95/99)={row['rmax_90']}/{row['rmax_95']}/{row['rmax_99']} "
                      f"cov(90/95/99)={row['coverage_90']}/{row['coverage_95']}/{row['coverage_99']} "
                      f"({elapsed:.2f}s)")

    total_elapsed = time.time() - t_start
    print(f"\nDone: {len(configs)} instances in {total_elapsed:.1f}s "
          f"({total_elapsed / max(1, len(configs)):.2f}s/instance average). "
          f"Results written to {out_path}")


if __name__ == "__main__":
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    main(output_csv=f'rmax_posterior_results_{timestamp}.csv')
