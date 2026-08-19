"""
test_planted_codeword_repair.py

Tests the prediction-and-repair stage of the LEP pipeline in isolation
from the low-weight-codeword search problem, by *planting* a codeword of
the target weight directly into the generator matrix instead of searching
a random code for one (which is exactly the hard problem the enumerator
debugging in test_real_isd_enumerator.py kept running into - near the
Gilbert-Varshamov bound, at n=128, q=127, that search can legitimately
fail or take a long time, which says nothing about whether the repair
stage itself works).

Construction, per instance:
    1. target_weight = round(gv_bound(n, k, q) * weight_factor)
    2. v0 = a uniformly random vector in GF(q)^n of Hamming weight
       exactly target_weight (target_weight random positions, each an
       independent uniform nonzero field element)
    3. G1 = a k x n generator matrix with v0 as its first row, filled out
       with k - 1 further uniformly random rows, each accepted only if it
       keeps the matrix full rank k
    4. G2, Q, Q_noisy = the usual LEP instance construction (random
       monomial secret Q, G2 = rref(G1 * Q), bit-flip-channel hint)
    5. v = v0 is used directly as "the" low-weight codeword fed into the
       prediction-and-repair stage (predict_image, then each of the three
       REPAIR_ALGORITHMS in turn)

v0 is guaranteed to be a codeword of C = rowspace(G1) by construction (it
literally is row 0), so no enumerator, ISD or otherwise, is involved at
all - this isolates the question "does prediction-and-repair work, given
that a codeword of this weight exists and was found" from "can an
enumerator find one in the first place".

Using SageMath. Run with:
    sage test_planted_codeword_repair.py
(from the Code/ directory, so the sibling modules import correctly).
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

# instances_generator.py / LEP_prediction_and_repair_v2.py now live in
# Code/core/, and run_lep_experiments.py lives in
# Code/strategies/05_full_parameter_sweep/ (this file was relocated into
# strategies/06_planted_codeword_repair_isolation/ during the Aug 2026
# folder reorganization), so both need to be added to sys.path explicitly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'core'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '05_full_parameter_sweep'))

from sage.all import GF, vector, matrix, set_random_seed, save as sage_save

from instances_generator import (
    generate_random_monomial,
    generate_bit_channel_hint,
    obtain_parity_check_matrix,
)
from LEP_prediction_and_repair_v2 import (
    compute_posterior_table,
    monomial_approximation,
    build_active_row_lists,
    structured_sd_repair,
    induced_sd_repair,
    posterior_aware_prange_repair,
    predict_image,
    active_error_set,
    set_verbose,
    pyrandom_sample,
)
from run_lep_experiments import gilbert_varshamov_bound, REPAIR_ALGORITHMS

set_verbose(True)

# If True, every run_instance() call saves its full instance and outcome
# to disk under RUNS_DIR - see save_run below for exactly what is saved.
# Set to False to skip saving (e.g. while iterating quickly on parameters).
SAVE_RUN = True
RUNS_DIR = 'runs'


# ---------------------------------------------------------------------------
# Persistence: saves everything needed to inspect or replay one run later -
# the problem instance, the planted min-weight codeword, Q_hat, the
# predicted word handed to the repair stage, and the repair outcome.
#
# Two files are written per run, sharing the same base name:
#   - <run_id>.sobj  : the full dict below, via Sage's save()/load() (so
#     every Sage matrix/vector round-trips exactly, including field info)
#   - <run_id>.json  : a small, plain-Python summary (dimensions, weights,
#     timings, success) for quick grepping without starting Sage
# ---------------------------------------------------------------------------

def save_run(params, G1, G2, Q, Q_noisy, v, Q_hat, w_tilde, w, r_v, rmax,
             tau, elapsed, outcome):
    """
    Saves one run's problem instance and outcome to RUNS_DIR.

    :param params: dict of the run's input parameters (n, k, q, alpha,
        beta, weight_factor, budgets, seed, target_weight)
    :param G1: generator matrix of C (with the planted row as row 0)
    :param G2: generator matrix of C' (= rref(G1 * Q))
    :param Q: the true secret monomial matrix (only available because this
        is a controlled test, not a real attack)
    :param Q_noisy: the noisy hint for Q
    :param v: the planted min-weight codeword (v0), in C
    :param Q_hat: the monomial approximation built from Q_noisy
    :param w_tilde: predict_image(v, Q_hat) - the predicted word handed to
        the repair stage
    :param w: the repaired word returned by posterior_aware_prange_repair,
        or None if repair failed
    :param r_v: the true active error dimension of v (Definition 3)
    :param rmax: the repair dimension used (derived or requested)
    :param tau: the Prange decoding radius used (2 * rmax)
    :param elapsed: seconds spent in the repair call
    :param outcome: one of 'correct', 'wrong', 'failed'
    :return: the base path (without extension) the run was saved under
    """
    os.makedirs(RUNS_DIR, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    run_id = (
        f"n{params['n']}_k{params['k']}_q{params['q']}_"
        f"seed{params['seed']}_{timestamp}"
    )
    base_path = os.path.join(RUNS_DIR, run_id)

    data = {
        'params': params,
        'G1': G1,
        'G2': G2,
        'Q': Q,
        'Q_noisy': Q_noisy,
        'v': v,
        'Q_hat': Q_hat,
        'w_tilde': w_tilde,
        'w': w,
        'r_v': r_v,
        'rmax': rmax,
        'tau': tau,
        'elapsed_s': elapsed,
        'outcome': outcome,
    }
    # Sage's save()/load() pickle every Sage object (matrices, vectors,
    # field elements) exactly, including their base ring - unlike plain
    # Python pickle/json, which don't know how to serialize them directly.
    sage_save(data, base_path)

    summary = {
        **params,
        'v_weight': v.hamming_weight(),
        'r_v': r_v,
        'rmax': rmax,
        'tau': tau,
        'elapsed_s': elapsed,
        'outcome': outcome,
        'repaired': w is not None,
    }
    with open(base_path + '.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  [save_run] saved instance + outcome to {base_path}.sobj "
          f"(+ {base_path}.json summary)")
    return base_path


# ---------------------------------------------------------------------------
# Steps 1-3: build a generator matrix with a planted low-weight first row.
# ---------------------------------------------------------------------------

def random_weight_vector(F, n, weight):
    """
    Samples a uniformly random vector in F^n of Hamming weight exactly
    `weight`: picks `weight` distinct coordinate positions uniformly at
    random, and fills each with an independent uniformly random nonzero
    field element, leaving every other coordinate at zero.

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
    target_weight, by planting it directly as the first row, then filling
    in k - 1 further rows (uniformly random over F^n) subject to keeping
    the matrix full rank k.

    Unlike get_random_generator_matrix in instances_generator.py (which
    regenerates the *entire* k x n matrix from scratch on every rank
    failure), rows are accepted incrementally here, since the first row
    (v0) must stay fixed - regenerating everything would risk losing it.
    In practice, for q=127 and n, k of a few hundred, a uniformly random
    row is linearly dependent on the rows so far with very low
    probability, so this loop essentially always succeeds within a
    handful of attempts per row.

    :param n: code length
    :param k: code dimension
    :param q: field size
    :param target_weight: the exact Hamming weight of the planted row,
        0 < target_weight <= n
    :return: a tuple (G, v0): G is a k x n Sage matrix of full rank k
        (NOT necessarily in reduced row-echelon form), and v0 is its
        first row - the planted codeword of Hamming weight target_weight
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


# ---------------------------------------------------------------------------
# Step 4: build the rest of the LEP instance from a *given* G1 (instead of
# instances_generator.generate_noisy_LCE_instance_CBA_bit_flip_version,
# which always generates a fresh random G1 internally and so cannot be
# handed a matrix with a planted row).
# ---------------------------------------------------------------------------

def generate_noisy_LCE_instance_from_G1(G1, q, alpha, beta, is_monomial=True):
    """
    Same construction as
    instances_generator.generate_noisy_LCE_instance_CBA_bit_flip_version,
    but takes an already-built generator matrix G1 as input instead of
    generating a fresh random one internally, so
    build_generator_with_planted_low_weight_row's output can be used
    directly.

    :param G1: a k x n generator matrix of C, over GF(q)
    :param q: field size
    :param alpha: Pr[bit 1 -> 0] in the bitwise leakage channel
    :param beta: Pr[bit 0 -> 1] in the bitwise leakage channel
    :param is_monomial: True for LEP (random monomial secret), False for
        PEP (random permutation secret)
    :return: a tuple (G2, Q, Q_noisy), matching the (G2, QP, QP_noisy)
        outputs of generate_noisy_LCE_instance_CBA_bit_flip_version
    """
    F = G1.base_ring()
    n = G1.ncols()

    Q = generate_random_monomial(n, q, is_permutation=not is_monomial)
    G2 = (G1 * Q).rref()

    Q_noisy = generate_bit_channel_hint(Q, q, alpha, beta)
    Q_noisy = matrix(F, Q_noisy)

    return G2, Q, Q_noisy


# ---------------------------------------------------------------------------
# Step 5 + repair stage: use v0 directly as the low-weight codeword, and
# test every REPAIR_ALGORITHMS entry against it.
# ---------------------------------------------------------------------------

def run_instance(n, k, q, alpha, beta, weight_factor=1.1, rmax=None, budgets=3,
                  seed=20260813, label=""):
    """
    Runs one full instance: plants a low-weight codeword v0 into a fresh
    generator matrix G1, builds the rest of the LEP instance around it,
    computes Q_hat, then attempts to repair predict_image(v0, Q_hat) into
    a codeword of C' with posterior_aware_prange_repair (the other two
    REPAIR_ALGORITHMS are commented out below the r(v)/rmax computation),
    reporting whether the recovered pair is genuinely equivalent
    (w == v0 * Q, checked against the true secret - only possible in this
    controlled test, not in a real attack).

    :param n, k, q: code and field parameters
    :param alpha, beta: bit-flip channel parameters
    :param weight_factor: multiplies the Gilbert-Varshamov bound to get
        the planted codeword's exact target weight
    :param rmax: repair dimension used by structured_sd_repair; also used
        to derive tau = 2 * rmax for the other two repair algorithms
        (Corollary 1 in the paper)
    :param budgets: per-active-row candidate budget (structured_sd_repair)
        / K_B (posterior_aware_prange_repair)
    :param seed: fixed Sage random seed, for reproducibility across runs
        with the same parameters
    :param label: a short string printed before this run's output
    :return: a dict of per-repair-algorithm outcomes, or None if instance
        construction failed
    """
    set_random_seed(seed)
    F = GF(q)
    print(f"\n{'=' * 70}\n{label} n={n} k={k} q={q} alpha={alpha} beta={beta} seed={seed}\n{'=' * 70}")

    gv_bound = gilbert_varshamov_bound(n, k, q)
    target_weight = max(1, round(gv_bound * weight_factor))
    print(f"  Gilbert-Varshamov bound: {gv_bound}; planted target_weight = {target_weight}")

    t0 = time.time()
    G1, v0 = build_generator_with_planted_low_weight_row(n, k, q, target_weight)
    print(f"  planted-row generator matrix built ({time.time() - t0:.2f}s); "
          f"v0 weight = {v0.hamming_weight()}")

    t0 = time.time()
    G2, Q, Q_noisy = generate_noisy_LCE_instance_from_G1(G1, q, alpha, beta, is_monomial=True)
    print(f"  instance generated ({time.time() - t0:.2f}s)")

    t0 = time.time()
    posterior_table = compute_posterior_table(Q_noisy, alpha, beta, is_permutation=False)
    Q_hat, S, D_loc, pi = monomial_approximation(posterior_table, F)
    rows_full = sum(1 for i in range(n) if list(Q_hat[i]) == list(Q[i]))
    print(f"  Q_hat built ({time.time() - t0:.2f}s); {rows_full}/{n} rows exactly correct")

    H2 = obtain_parity_check_matrix(G2)

    v = v0
    r_v = len(active_error_set(v, Q, Q_hat))

    # posterior_aware_prange_repair (unlike structured_sd_repair) has no
    # combinatorial blowup in rmax - its cost is bounded by max_trials
    # regardless of rmax, since it only uses rmax to set the Prange
    # decoding radius tau = 2 * rmax. So the "logical" choice of rmax here
    # isn't about staying small for tractability, it's about making sure
    # tau actually covers r(v): the induced syndrome-decoding subproblem
    # has an *exact* error weight of r(v) (the true active error
    # dimension), so if tau < r(v), no number of trials can ever find the
    # right error vector - it's not there to be found at that radius. If
    # an explicit rmax was passed in, it is used as-is (and a warning is
    # printed if it looks too small); otherwise rmax is derived directly
    # from the measured r(v) of this instance, with a safety margin (this
    # is exactly what "logical for this algorithm" means: tau >= r(v) is
    # a requirement, not a tuning knob).
    margin = 4
    derived_rmax = -(-(r_v + margin) // 2)  # ceil((r_v + margin) / 2)
    if rmax is None:
        rmax = derived_rmax
        print(f"  true active error dimension r(v) = {r_v}; "
              f"derived rmax = {rmax} (tau = {2 * rmax}, "
              f"margin = {margin} above r(v))")
    else:
        print(f"  true active error dimension r(v) = {r_v}; using requested "
              f"rmax = {rmax} (tau = {2 * rmax})")
        if 2 * rmax < r_v:
            print(f"  WARNING: tau = {2 * rmax} is below r(v) = {r_v} - "
                  f"the true error vector cannot be found at this radius, "
                  f"repair will fail deterministically; consider rmax >= "
                  f"{derived_rmax}")

    w_tilde = predict_image(v, Q_hat)
    tau = 2 * rmax

    results = {}

    # Only posterior_aware_prange_repair is tested here (induced_sd_repair
    # and structured_sd_repair are commented out below). To re-enable
    # them: induced_sd_repair(w_tilde, v, H2, tau); for
    # structured_sd_repair, remember it needs its own much smaller rmax
    # cap (its cost is combinatorial: C(|A|, rmax) * budgets**rmax), see
    # the discussion in the earlier version of this script / conversation.
    repair_algorithm = 'posterior_aware_prange_repair'
    t0 = time.time()
    w = posterior_aware_prange_repair(w_tilde, v, H2, Q_hat, S, k, tau, K_B=budgets, max_trials=100000)
    # w = induced_sd_repair(w_tilde, v, H2, tau)
    # A, L = build_active_row_lists(v, S, D_loc, F, budgets=budgets)
    # w = structured_sd_repair(w_tilde, v, H2, Q_hat, A, L, rmax)
    elapsed = time.time() - t0

    if w is None:
        print(f"  [{repair_algorithm}] FAILED to repair ({elapsed:.2f}s)")
        results[repair_algorithm] = ('failed', elapsed)
        outcome_tag = 'failed'
    else:
        is_correct = (w == v * Q)
        outcome = 'CORRECT equivalent pair' if is_correct else 'weight/syndrome-consistent but WRONG pair'
        print(f"  [{repair_algorithm}] repaired in {elapsed:.2f}s -> {outcome}")
        outcome_tag = 'correct' if is_correct else 'wrong'
        results[repair_algorithm] = (outcome_tag, elapsed)

    if SAVE_RUN:
        params = {
            'n': n, 'k': k, 'q': q, 'alpha': alpha, 'beta': beta,
            'weight_factor': weight_factor, 'budgets': budgets, 'seed': seed,
            'target_weight': target_weight,
        }
        save_run(
            params, G1, G2, Q, Q_noisy, v, Q_hat, w_tilde, w,
            r_v, rmax, tau, elapsed, outcome_tag,
        )

    return {'v': v, 'r_v': r_v, 'results': results}


def main():
    # run_instance(
    #     n=16, k=8, q=127, alpha=0.01, beta=0.08,
    #     weight_factor=1.1, rmax=None, budgets=5,
    #     label="[small instance]", seed=20260814
    # )

    seeds = [i for i in range(26, 36)]
    for seed in seeds:
        run_instance(
            n=128, k=64, q=127, alpha=0.01, beta=0.05,
            weight_factor=1.1, rmax=None, budgets=5,
            label="[requested instance]", seed=seed
        )


if __name__ == "__main__":
    main()
