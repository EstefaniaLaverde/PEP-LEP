"""
Parameter sweep for the LEP prediction-and-repair pipeline under the BBLM
(bitwise leakage channel) model. Runs the full pipeline (instance generation
-> posterior -> Q_hat -> prediction-and-repair) for multiple combinations of
(n, k, q, alpha, beta, rmax, weight_factor, budgets, enum_version,
repair_algorithm), saving a checkpoint (one CSV row) after every individual
run, with a fixed, reproducible seed per run.

Compared to the previous version of this script:
  - estimate_min_distance / brute_force_min_weight_codewords (which either
    ran a full O(q**k) search or fell back to the Singleton bound) are
    replaced by the Gilbert-Varshamov bound, a cheap closed-form estimate
    that works at any (n, k, q), including the larger sizes this sweep now
    targets (n up to a few hundred). Since the GV bound depends only on
    (n, k, q) and not on the specific code, it is the same for C and C'.
  - target_weight is no longer fixed to "the" minimum distance: it is swept
    as target_weight = round(gv_bound * weight_factor), for a list of
    weight_factor values (this is the "weight of v" testing variable).
  - budgets (the per-active-row candidate list size used by
    build_active_row_lists) is now a swept variable instead of a constant.
  - three enumeration modules ("Enum" in the paper's Algorithm 1) are
    available and swept via enum_version:
      'random'             - plain rejection sampling (the original
                              placeholder: sample a random message, check
                              if the resulting codeword is low-weight)
      'linear_combination' - build a pool of base codewords, then search
                              scalar linear combinations of small subsets
                              of that pool for a low-weight result
      'local_search'       - a greedy stochastic local search over message
                              vectors (an "RL-lite" heuristic: not a
                              trained policy, but an iterative,
                              reward-driven search rather than pure/
                              structured random sampling)
  - three repair modules ("Repair" in the paper's Algorithm 1) are available
    and swept via repair_algorithm (see REPAIR_ALGORITHMS below):
      'induced_sd_repair'               - Algorithm 6, generic Prange-based
                                           induced syndrome decoding
      'structured_sd_repair'            - Algorithm 8, exhaustive hypothesis
                                           search over active-row
                                           replacements
      'posterior_aware_prange_repair'   - Algorithm 12, posterior-biased
                                           Prange trials

Requires (adjust the two module names below if your local filenames differ):
    - instances_generator.py
      (generate_noisy_LCE_instance_CBA_bit_flip_version, obtain_parity_check_matrix)
    - the prediction-and-repair module (imported here as
      LEP_prediction_and_repair_v2; change this import if your local file is
      named differently, e.g. prediction_and_repair)

Using SageMath.
"""
import copy
import csv
import hashlib
import os
import sys
import time
import itertools
from datetime import datetime, timezone
from math import comb

# instances_generator.py / LEP_prediction_and_repair_v2.py now live in
# Code/core/ (this file was relocated into strategies/05_full_parameter_sweep/
# during the Aug 2026 folder reorganization), so core/ needs to be added to
# sys.path explicitly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'core'))

from sage.all import GF, vector, randint, random, set_random_seed

from instances_generator import (
    generate_noisy_LCE_instance_CBA_bit_flip_version,
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
    enum_low_weight_codeword,  # the "random" enumerator, version 1
    set_verbose,
)

# ---------------------------------------------------------------------------
# Repair algorithm dispatch: the three repair modules pluggable into
# Algorithm 1 (PredictionAndRepairFramework), swept via `repair_algorithm`.
#   - 'induced_sd_repair'          : Algorithm 6, generic Prange-based
#                                     induced syndrome decoding.
#   - 'structured_sd_repair'       : Algorithm 8, exhaustive hypothesis
#                                     search over active-row replacements
#                                     (combinatorial in |A| and rmax - see
#                                     REPAIR_ALGORITHMS docstring below).
#   - 'posterior_aware_prange_repair': Algorithm 12, posterior-biased Prange
#                                     trials; cost is a fixed number of
#                                     trials regardless of |A| or rmax, so it
#                                     does not suffer the combinatorial
#                                     blow-up of structured_sd_repair.
#
# Both induced_sd_repair and posterior_aware_prange_repair take a decoding
# radius tau rather than rmax directly; per Corollary 1 in the paper,
# tau = 2 * rmax, so the existing rmax sweep axis is reused instead of adding
# a separate tau axis.
# ---------------------------------------------------------------------------

REPAIR_ALGORITHMS = ['induced_sd_repair', 'structured_sd_repair', 'posterior_aware_prange_repair']

# ---------------------------------------------------------------------------
# Verbosity control: gates the per-stage / heartbeat prints below (instance
# generation, posterior/Q_hat timing, prediction-and-repair progress), so a
# slow run's current stage is visible instead of the sweep looking hung. Set
# to False for quieter output (only the one line per combo from main() will
# print). Also propagated to LEP_prediction_and_repair_v2.py so both files'
# progress prints turn on/off together.
# ---------------------------------------------------------------------------
VERBOSE = True
set_verbose(VERBOSE)


def _vprint(*args, **kwargs):
    """
    Prints only when VERBOSE is enabled, mirroring _vprint in
    LEP_prediction_and_repair_v2.py.
    """
    if VERBOSE:
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Reproducibility: a deterministic seed per run, independent of execution
# order (so a sweep can be interrupted and resumed without losing
# reproducibility for the runs already done or the ones still pending).
# ---------------------------------------------------------------------------

def deterministic_seed(n, k, q, alpha, beta, rmax, weight_factor, budgets,
                        enum_version, repair_algorithm, trial):
    """
    Computes a deterministic seed from the run's parameters, so that every
    combination always gets the same seed regardless of the order the
    sweep executes in (important for resuming an interrupted sweep).

    :return: an integer in [0, 2**31 - 1], suitable for set_random_seed
    """
    key = (
        f"{n}_{k}_{q}_{alpha}_{beta}_{rmax}_{weight_factor}_{budgets}_"
        f"{enum_version}_{repair_algorithm}_{trial}"
    )
    digest = hashlib.sha256(key.encode()).hexdigest()
    return int(digest, 16) % (2**31 - 1)


# ---------------------------------------------------------------------------
# Gilbert-Varshamov bound: a cheap, closed-form estimate of an achievable
# minimum distance for an [n, k]_q code, used instead of an expensive exact
# (brute-force) or loose (Singleton) estimate. Replaces estimate_min_distance
# / brute_force_min_weight_codewords entirely.
# ---------------------------------------------------------------------------

def gilbert_varshamov_bound(n, k, q):
    """
    Computes the Gilbert-Varshamov bound: the largest d such that

        sum_{i=0}^{d-2} C(n-1, i) * (q-1)^i  <  q^(n-k)

    which guarantees the existence of an [n, k]_q code with minimum
    distance at least d, and in practice tracks fairly closely the minimum
    distance of a "typical" random code of these parameters (see the
    n=7,k=3,q=7 case: GV bound 4 matches the empirically observed minimum
    distance from earlier brute-force testing at that size).

    This depends only on (n, k, q), not on the specific generator matrix,
    so the same value applies to both C and C' in this pipeline (they share
    the same n, k, q).

    :param n: code length
    :param k: code dimension
    :param q: field size
    :return: the Gilbert-Varshamov bound, an integer >= 1
    """
    capacity = q ** (n - k)
    cumulative = 0
    d = 1
    while True:
        if cumulative >= capacity:
            return d - 1
        term = comb(n - 1, d - 1) * (q - 1) ** (d - 1)
        cumulative += term
        d += 1
        if d > n + 1:
            # Safety fallback; should not trigger for valid [n, k]_q parameters.
            return n


# ---------------------------------------------------------------------------
# Small local helpers shared by the enumerators below (kept local instead of
# imported, so this script does not depend on private helpers of the
# library module staying stable).
# ---------------------------------------------------------------------------

def _pyrandom_sample(population, size):
    """
    Uniform-without-replacement sample of `size` elements from a sequence,
    implemented with Sage's randint so the whole script's randomness stays
    tied to a single, seedable source.

    :param population: the sequence to sample from
    :param size: number of distinct elements to draw
    :return: a list of `size` elements drawn from population
    """
    pool = list(population)
    sample = []
    for _ in range(min(size, len(pool))):
        idx = randint(0, len(pool) - 1)
        sample.append(pool.pop(idx))
    return sample


def _random_message(F, k, sparse=False):
    """
    Samples a random message vector in F^k. When sparse is True, only a
    random subset of coordinates (from 1 up to all k) are set to a random
    nonzero field element, leaving the rest at zero; otherwise every
    coordinate is drawn independently (may include zeros).

    :param F: the base field GF(q)
    :param k: message length
    :param sparse: whether to zero out a random subset of coordinates
    :return: a random message vector in F^k
    """
    if not sparse:
        return vector(F, [F.random_element() for _ in range(k)])

    m = vector(F, k)
    num_nonzero = randint(1, k)
    for idx in _pyrandom_sample(range(k), num_nonzero):
        a = F.random_element()
        while a == 0:
            a = F.random_element()
        m[idx] = a
    return m


# ---------------------------------------------------------------------------
# Enumerator version 2: linear-combination search.
#
# Builds a pool of "base" codewords (random, not necessarily low-weight
# themselves), then searches scalar linear combinations of small subsets of
# that pool. Since C is a vector space, any such combination is itself a
# valid codeword; the point is that combining a handful of base vectors
# explores a structured neighborhood of the code (their sum/difference
# patterns can partially cancel in shared support positions) rather than
# resampling a fresh independent message every single trial.
# ---------------------------------------------------------------------------

def enum_low_weight_codeword_linear_combination(G, target_weight, max_trials=1000,
                                                  num_base_words=10, combo_size=2):
    """
    Enumerator version 2: builds a pool of num_base_words random codewords, then repeatedly samples
    combo_size of them, combines them with random nonzero field scalars,
    and checks whether the resulting codeword has weight <= target_weight.

    :param G: a generator matrix of C
    :param target_weight: accept any nonzero codeword of weight <= this value
    :param max_trials: number of random combinations to try
    :param num_base_words: size of the base pool of codewords to draw combinations from
    :param combo_size: number of base codewords combined per trial
    :return: a codeword v in C of weight <= target_weight, or None on Failure
    """
    F = G.base_ring()
    k = G.nrows()

    # Build the base pool. Codewords are stored (not their messages), since
    # combinations of codewords are themselves codewords directly.
    base_words = []
    for _ in range(num_base_words):
        m = _random_message(F, k, sparse=False)
        if m.is_zero():
            continue
        base_words.append(m * G)

    if len(base_words) < combo_size:
        return None

    nonzero_elements = [a for a in F if a != F(0)]

    for _ in range(max_trials):
        idxs = _pyrandom_sample(range(len(base_words)), combo_size)
        coeffs = [nonzero_elements[randint(0, len(nonzero_elements) - 1)] for _ in idxs]

        v = vector(F, G.ncols())
        for idx, c in zip(idxs, coeffs):
            v = v + c * base_words[idx]

        if 0 < v.hamming_weight() <= target_weight:
            return v

    return None


# ---------------------------------------------------------------------------
# Enumerator version 3: local search ("RL-lite").
#
# Not a trained reinforcement-learning policy - there is no value function
# and nothing is learned across instances - but it plays the analogous
# structural role: the message vector is treated as a state, -weight(v) as
# a reward, and the search iteratively proposes small perturbations and
# greedily accepts non-worsening moves, with periodic restarts to escape
# local optima. This is a stochastic hill-climb, explicitly documented as a
# heuristic rather than literal RL.
# ---------------------------------------------------------------------------

def enum_low_weight_codeword_local_search(G, target_weight, max_trials=2000, restarts=10):
    """
    Enumerator version 3 ("something like reinforcement learning ... finds
    the codeword that optimizes / searches for that minimum weight"): a
    greedy stochastic local search over message vectors. Each restart
    starts from a fresh random message and repeatedly proposes changing one
    random coordinate to a new random field value, accepting the move
    whenever it does not increase the resulting codeword's weight.

    :param G: a generator matrix of C
    :param target_weight: stop early and return as soon as a codeword of
        weight <= target_weight is found
    :param max_trials: total perturbation steps, divided across restarts
    :param restarts: number of independent hill-climbing runs
    :return: a codeword v in C of weight <= target_weight, or None on Failure
    """
    F = G.base_ring()
    k = G.nrows()
    nonzero_elements = [a for a in F if a != F(0)]
    steps_per_restart = max(1, max_trials // max(1, restarts))

    for _ in range(restarts):
        m = _random_message(F, k, sparse=False)
        v = m * G
        w = v.hamming_weight()

        for _ in range(steps_per_restart):
            if 0 < w <= target_weight:
                return v

            # Propose changing one random coordinate of the message to a
            # new random value (occasionally zeroing it out, to also allow
            # the search to reduce the message's own sparsity).
            idx = randint(0, k - 1)
            if random() < 0.9:
                new_val = nonzero_elements[randint(0, len(nonzero_elements) - 1)]
            else:
                new_val = F(0)

            m_candidate = copy.copy(m)
            m_candidate[idx] = new_val
            v_candidate = m_candidate * G
            w_candidate = v_candidate.hamming_weight()

            # Greedy acceptance: keep the move if it does not make the
            # weight worse (and does not collapse to the zero codeword).
            if 0 < w_candidate <= w:
                m, v, w = m_candidate, v_candidate, w_candidate

        if 0 < w <= target_weight:
            return v

    return None


# ---------------------------------------------------------------------------
# Dispatch table: uniform (G, target_weight, max_trials) -> v or None
# interface over all three enumerator versions, so the instrumented run
# loop does not need to know which one is selected.
# ---------------------------------------------------------------------------

ENUMERATORS = {
    'random': lambda G, target_weight, max_trials: enum_low_weight_codeword(
        G, target_weight, max_trials=max_trials,
    ),
    'linear_combination': lambda G, target_weight, max_trials: enum_low_weight_codeword_linear_combination(
        G, target_weight, max_trials=max_trials, num_base_words=10, combo_size=2,
    ),
    'local_search': lambda G, target_weight, max_trials: enum_low_weight_codeword_local_search(
        G, target_weight, max_trials=max_trials, restarts=10,
    ),
}


# ---------------------------------------------------------------------------
# Instrumented prediction-and-repair run: reimplements the Algorithm 1 loop
# instead of calling prediction_and_repair_framework directly, so we can
# record extra per-attempt metrics (active error dimension r(v), enumeration
# failures, etc.) that the library function does not expose, and so we can
# select among the three enumerator versions above.
# ---------------------------------------------------------------------------

def run_prediction_and_repair_instrumented(G1, Q, Q_hat, S, D_loc, F, H2,
                                            rmax, m_pair, n_iter,
                                            target_weight, max_trials_enum,
                                            budgets, enum_version,
                                            repair_algorithm, k,
                                            heartbeat_every_s=5.0):
    """
    Same as prediction_and_repair_framework (Algorithm 1), but additionally
    records, for every codeword v that was actually generated, the true
    active error dimension r(v) = |E(v)| (Definition 3), computed using the
    true secret Q (only possible in a controlled experiment, not in a real
    attack), and how many times Enum failed to find a suitably-weighted v.

    Prints a heartbeat (iteration index, pairs found so far, enum failures,
    elapsed time) at most once every `heartbeat_every_s` seconds, so a slow
    iteration (e.g. large n / large rmax) is visible instead of looking hung.

    :param enum_version: one of 'random', 'linear_combination', 'local_search'
    :param repair_algorithm: one of REPAIR_ALGORITHMS - selects which repair
        module (Algorithm 6, 8, or 12) is used for every repair attempt.
        induced_sd_repair and posterior_aware_prange_repair use
        tau = 2 * rmax as their decoding radius (Corollary 1); only
        structured_sd_repair uses rmax directly, together with the
        per-active-row candidate lists built from budgets.
    :param k: dimension of C' (== C, since both share (n, k, q)), needed by
        posterior_aware_prange_repair as the information-set size
    :return: a tuple (pairs, r_values, iters_used, enum_failures)
    """
    enumerator = ENUMERATORS[enum_version]
    tau = 2 * rmax  # Corollary 1: the decoding radius used by the two
                     # non-structured repair modules, derived from rmax so
                     # no separate tau sweep axis is needed.

    pairs = []
    r_values = []
    enum_failures = 0
    iters_used = 0

    t_start = time.time()
    t_last_heartbeat = t_start

    for it in range(n_iter):
        iters_used = it + 1
        if len(pairs) >= m_pair:
            break

        t_iter0 = time.time()
        v = enumerator(G1, target_weight, max_trials_enum)
        enum_time = time.time() - t_iter0
        repair_time = None
        if v is None:
            enum_failures += 1
        else:
            # True active error dimension for this v (diagnostic only; uses
            # the true secret Q, which a real attacker would not have).
            r_v = len(active_error_set(v, Q, Q_hat))
            r_values.append(r_v)

            w_tilde = predict_image(v, Q_hat)
            t_repair0 = time.time()

            if repair_algorithm == 'induced_sd_repair':
                w = induced_sd_repair(w_tilde, v, H2, tau)
            elif repair_algorithm == 'posterior_aware_prange_repair':
                # Reuses `budgets` as K_B, the per-row coordinate budget for
                # the marginal error region - same role budgets plays as the
                # per-row candidate-list size in structured_sd_repair.
                w = posterior_aware_prange_repair(
                    w_tilde, v, H2, Q_hat, S, k, tau, K_B=budgets,
                )
            elif repair_algorithm == 'structured_sd_repair':
                A, L = build_active_row_lists(v, S, D_loc, F, budgets=budgets)
                w = structured_sd_repair(w_tilde, v, H2, Q_hat, A, L, rmax)
            else:
                raise ValueError(f"Unknown repair_algorithm: {repair_algorithm!r}")

            repair_time = time.time() - t_repair0

            if w is not None:
                pairs.append((v, w))

        now = time.time()
        if now - t_last_heartbeat >= heartbeat_every_s or it == 0:
            repair_time_str = f"{repair_time:.2f}s" if repair_time is not None else "n/a"
            _vprint(
                f"\n      [repair] iter {it + 1}/{n_iter} "
                f"pairs={len(pairs)}/{m_pair} enum_failures={enum_failures} "
                f"last_enum_time={enum_time:.2f}s last_repair_time={repair_time_str} "
                f"elapsed={now - t_start:.1f}s",
                end='', flush=True,
            )
            t_last_heartbeat = now

    return pairs, r_values, iters_used, enum_failures


# ---------------------------------------------------------------------------
# One full run: generates the instance, builds Q_hat, runs
# prediction-and-repair, and assembles a dict with every metric. Any
# exception is caught and recorded as a failed run, so a long sweep does not
# crash entirely because of a single problematic case.
# ---------------------------------------------------------------------------

def run_single_experiment(n, k, q, alpha, beta, rmax, weight_factor, budgets,
                           enum_version, repair_algorithm, trial, m_pair=2,
                           n_iter=500, max_trials_enum=500):
    """
    Runs one full instance of the LEP pipeline for a given parameter
    combination, with a deterministic and reproducible seed.

    :param n, k, q: code and field parameters
    :param alpha: Pr[bit 1 -> 0] in the bitwise channel
    :param beta: Pr[bit 0 -> 1] in the bitwise channel
    :param rmax: maximum repair dimension explored by structured_sd_repair;
        for induced_sd_repair / posterior_aware_prange_repair this is instead
        used to derive the decoding radius tau = 2 * rmax (Corollary 1)
    :param weight_factor: multiplier applied to the Gilbert-Varshamov bound
        to obtain target_weight = round(gv_bound * weight_factor)
    :param budgets: candidate-list size per active row (build_active_row_lists),
        reused as K_B (per-row coordinate budget) when
        repair_algorithm == 'posterior_aware_prange_repair'
    :param enum_version: one of 'random', 'linear_combination', 'local_search'
    :param repair_algorithm: one of REPAIR_ALGORITHMS - 'induced_sd_repair',
        'structured_sd_repair', or 'posterior_aware_prange_repair' - selects
        which repair module (Algorithm 6, 8, or 12) is used
    :param trial: repetition index (to average/vary runs under the same configuration)
    :param m_pair: target number of candidate pairs to recover
    :param n_iter: maximum number of framework iterations
    :param max_trials_enum: maximum trials per call to the enumerator
    :return: a dict with every parameter, metric, and the outcome of the run
    """
    seed = deterministic_seed(n, k, q, alpha, beta, rmax, weight_factor,
                               budgets, enum_version, repair_algorithm, trial)
    set_random_seed(seed)
    F = GF(q)

    result = {
        'n': n, 'k': k, 'q': q, 'alpha': alpha, 'beta': beta, 'rmax': rmax,
        'weight_factor': weight_factor, 'budgets': budgets,
        'enum_version': enum_version, 'repair_algorithm': repair_algorithm,
        'trial': trial, 'seed': seed,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'm_pair_target': m_pair, 'n_iter': n_iter,
        'max_trials_enum': max_trials_enum,
        'error': '',
    }

    t_total0 = time.time()
    try:
        # --- 1) Generate the instance (bitwise channel) --------------------
        _vprint("\n    [stage 1/5] generating instance...", end='', flush=True)
        t0 = time.time()
        G1, G2, Q, Q_noisy = generate_noisy_LCE_instance_CBA_bit_flip_version(
            n, k, q, alpha, beta, is_monomial=True,
        )
        result['gen_time_s'] = time.time() - t0
        _vprint(f" done ({result['gen_time_s']:.2f}s)", end='', flush=True)

        # --- 2) BBLM posterior table ----------------------------------------
        _vprint("\n    [stage 2/5] computing posterior table...", end='', flush=True)
        t0 = time.time()
        posterior_table = compute_posterior_table(Q_noisy, alpha, beta, is_permutation=False)
        result['posterior_time_s'] = time.time() - t0
        _vprint(f" done ({result['posterior_time_s']:.2f}s)", end='', flush=True)

        # --- 3) Monomial approximation Q_hat ---------------------------------
        _vprint("\n    [stage 3/5] computing monomial approximation (Q_hat)...", end='', flush=True)
        t0 = time.time()
        Q_hat, S, D_loc, pi = monomial_approximation(posterior_table, F)
        result['qhat_time_s'] = time.time() - t0
        _vprint(f" done ({result['qhat_time_s']:.2f}s)", end='', flush=True)

        # Rows that are fully correct (both column and scalar match).
        rows_full = sum(1 for i in range(n) if list(Q_hat[i]) == list(Q[i]))
        # Rows with the correct column (support) but the wrong scalar:
        # partial recovery, useful for separating the effect of alpha/beta
        # from the effect of rmax.
        rows_support_only = 0
        for i in range(n):
            col_hat = Q_hat[i].nonzero_positions()
            col_true = Q[i].nonzero_positions()
            if col_hat and col_true and col_hat[0] == col_true[0] and list(Q_hat[i]) != list(Q[i]):
                rows_support_only += 1

        result['rows_correct_full'] = rows_full
        result['rows_correct_support_only'] = rows_support_only
        result['n_rows_total'] = n

        # --- 4) Gilbert-Varshamov-based target weight ------------------------
        H2 = obtain_parity_check_matrix(G2)
        gv_bound = gilbert_varshamov_bound(n, k, q)
        target_weight = max(1, round(gv_bound * weight_factor))
        result['gv_bound'] = gv_bound
        result['target_weight_used'] = target_weight
        # Diagnostic only (Corollary 2 in the paper): the repair is unique
        # within the explored radius if d(C') > 4 * rmax. Uses the GV bound
        # as a stand-in for the true (generally unknown, at this scale)
        # minimum distance of C'.
        result['uniqueness_guaranteed_gv'] = bool(gv_bound > 4 * rmax)

        # --- 5) Instrumented prediction-and-repair ---------------------------
        _vprint(
            f"\n    [stage 4/5] gv_bound={gv_bound} target_weight={target_weight}",
            end='', flush=True,
        )
        _vprint(
            f"\n    [stage 5/5] prediction-and-repair (up to {n_iter} iters, "
            f"target m_pair={m_pair})...", end='', flush=True,
        )
        t0 = time.time()
        pairs, r_values, iters_used, enum_failures = run_prediction_and_repair_instrumented(
            G1, Q, Q_hat, S, D_loc, F, H2, rmax, m_pair, n_iter,
            target_weight, max_trials_enum, budgets, enum_version,
            repair_algorithm, k,
        )
        result['repair_time_s'] = time.time() - t0
        _vprint(
            f"\n    [stage 5/5] done ({result['repair_time_s']:.2f}s), "
            f"pairs={len(pairs)}/{m_pair}, iters_used={iters_used}, "
            f"enum_failures={enum_failures}", end='', flush=True,
        )

        pairs_correct = sum(1 for v, w in pairs if w == v * Q)

        result['pairs_recovered'] = len(pairs)
        result['pairs_correct'] = pairs_correct
        result['iters_used'] = iters_used
        result['enum_failures'] = enum_failures

        result['avg_active_error_dim'] = (sum(r_values) / len(r_values)) if r_values else None
        result['max_active_error_dim'] = max(r_values) if r_values else None
        # Fraction of sampled codewords whose true active error dimension
        # falls within what rmax can correct (Theorem 1): a direct measure
        # of how well the chosen rmax "covers" the actual instance.
        result['frac_r_le_rmax'] = (
            sum(1 for r in r_values if r <= rmax) / len(r_values)
        ) if r_values else None

        result['avg_v_weight'] = (
            sum(v.hamming_weight() for v, _w in pairs) / len(pairs)
        ) if pairs else None

        # Success: the requested number of pairs was reached AND all of
        # them are genuinely true (not merely syndrome/weight-consistent).
        result['success'] = bool(
            len(pairs) >= m_pair and pairs_correct == len(pairs) and len(pairs) > 0
        )

    except Exception as exc:
        result['error'] = f"{type(exc).__name__}: {exc}"
        result['success'] = False

    result['total_time_s'] = time.time() - t_total0
    return result


# ---------------------------------------------------------------------------
# Checkpointing: every row is written to (and flushed on) disk as soon as
# its run finishes, and on startup the already-present combinations are
# read from the CSV so a resumed sweep does not repeat them.
# ---------------------------------------------------------------------------

FIELDNAMES = [
    'n', 'k', 'q', 'alpha', 'beta', 'rmax', 'weight_factor', 'budgets',
    'enum_version', 'repair_algorithm', 'trial', 'seed', 'timestamp',
    'm_pair_target', 'n_iter', 'max_trials_enum',
    'gen_time_s', 'posterior_time_s', 'qhat_time_s', 'repair_time_s', 'total_time_s',
    'rows_correct_full', 'rows_correct_support_only', 'n_rows_total',
    'gv_bound', 'target_weight_used', 'uniqueness_guaranteed_gv',
    'iters_used', 'enum_failures',
    'pairs_recovered', 'pairs_correct',
    'avg_active_error_dim', 'max_active_error_dim', 'frac_r_le_rmax',
    'avg_v_weight', 'success', 'error',
]

# The key fields that identify a unique run, in the order used both to build
# deterministic_seed and to detect already-completed runs on resume.
KEY_FIELDS = ['n', 'k', 'q', 'alpha', 'beta', 'rmax', 'weight_factor',
              'budgets', 'enum_version', 'repair_algorithm', 'trial']


def load_completed_keys(csv_path):
    """
    Reads an existing results CSV (if any) and returns the set of
    KEY_FIELDS combinations already present, so they can be skipped when
    resuming an interrupted sweep.

    If the existing CSV predates the current schema (e.g. it is missing
    KEY_FIELDS columns such as weight_factor/budgets/enum_version, or was
    generated before some other change to this script), it is renamed to a
    timestamped backup file instead of being parsed, and the sweep starts
    a fresh checkpoint. This also matters if the CSV predates the MSB-first
    bit-order fix in instances_generator.py: that data is not comparable to
    new runs, so it should not silently be treated as "already completed".

    :param csv_path: path to the checkpoint CSV
    :return: a set of tuples matching KEY_FIELDS
    """
    if not os.path.exists(csv_path):
        return set()

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []

        missing = [field for field in KEY_FIELDS if field not in header]
        if missing:
            f.close()
            backup_path = csv_path + f".bak-{int(time.time())}"
            os.rename(csv_path, backup_path)
            print(
                f"Existing checkpoint at {csv_path} is missing columns "
                f"{missing} (schema mismatch, likely from an older version "
                f"of this script). Moved it to {backup_path} and starting "
                f"a fresh checkpoint."
            )
            return set()

        done = set()
        for row in reader:
            key = (
                int(row['n']), int(row['k']), int(row['q']),
                float(row['alpha']), float(row['beta']), int(row['rmax']),
                float(row['weight_factor']), int(row['budgets']),
                row['enum_version'], row['repair_algorithm'], int(row['trial']),
            )
            done.add(key)
    return done


def append_result(csv_path, result):
    """
    Appends one result row to the checkpoint CSV, writing the header if the
    file does not exist yet, and flushing immediately so the row is safely
    on disk even if the sweep is interrupted right after.

    :param csv_path: path to the checkpoint CSV
    :param result: the dict returned by run_single_experiment
    """
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Main sweep. Adjust the parameter lists to whatever you want to cover; the
# grid below is a reasonable, not-too-expensive starting point. Note that
# adding weight_factor, budgets, enum_version, and repair_algorithm as full
# sweep axes multiplies the combo count fast - trim the other axes
# accordingly. repair_algorithm in particular is worth trimming first for
# large n: 'structured_sd_repair' is combinatorial in |A| and rmax (see
# REPAIR_ALGORITHMS above) and can make a single repair attempt take hours
# to days at n>=128, while 'induced_sd_repair' and
# 'posterior_aware_prange_repair' cost a fixed number of trials regardless
# of n or rmax.
# ---------------------------------------------------------------------------

def main():
    # Set this to False to run the full grid defined below. With True, only
    # a handful of small, fast combinations run, useful for checking that
    # the whole pipeline (imports, module names, etc.) works before
    # launching the full sweep.
    QUICK_TEST = False

    # __file__-relative (rather than cwd-relative) so this still finds/
    # creates results/ next to this script regardless of what directory
    # `sage run_lep_experiments.py` is invoked from.
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    csv_path = os.path.join(results_dir, 'lep_experiment_results_large_combinations.csv')
    os.makedirs(results_dir, exist_ok=True)

    if QUICK_TEST:
        nk_values = [(7, 3)]
        q_values = [7]
        alpha_values = [0.01]
        beta_values = [0.1]
        rmax_values = [1, 2]
        weight_factor_values = [1.5]
        budgets_values = [3]
        enum_version_values = ['random', 'linear_combination', 'local_search']
        repair_algorithm_values = REPAIR_ALGORITHMS
        num_trials = 1

        combos = list(itertools.product(
            nk_values, q_values, alpha_values, beta_values, rmax_values,
            weight_factor_values, budgets_values, enum_version_values,
            repair_algorithm_values, range(num_trials),
        ))
    else:
        # (n, k) pairs with k < n, spanning toy sizes up to the actual LESS
        # NIST Category 1 parameters: (n, k, q) = (252, 126, 127). That is
        # the largest official LESS parameter set with n <= 256 - NIST-3
        # uses n=400 and NIST-5 uses n=548, both above the requested cap.
        # LESS keeps rate k/n = 1/2 exactly at every category
        # (126/252 = 200/400 = 274/548 = 0.5); that ratio is kept here for
        # the larger (n, k) pairs too. q=127 is LESS's actual field choice,
        # picked so every field element fits in one byte.
        small_nk_values = [(7, 3), (16, 8), (32, 16)]
        large_nk_values = [(64, 32), (128, 64), (252, 126)]

        q_values = [7, 17, 31, 127]      # 127 = LESS's actual field size

        alpha_values = [0.01, 0.02, 0.05]
        beta_values = [0.05, 0.1, 0.2]
        rmax_values = [1, 2, 3]
        weight_factor_values = [1.1, 1.2, 1.5]   # multiplies the GV bound
        budgets_values = [2, 3, 5, 10]
        enum_version_values = ['random', 'linear_combination', 'local_search']
        repair_algorithm_values = REPAIR_ALGORITHMS
        num_trials = 5

        # IMPORTANT: per-run cost grows fast with n (Q_hat construction
        # alone is O(n^3), from the Hungarian assignment), so running the
        # full hyperparameter grid at n=252 would take a very long time.
        # More fundamentally, finding a low-weight codeword at n=252 with
        # these heuristic enumerators (none of them is a state-of-the-art
        # ISD implementation) is essentially the same hard problem LESS's
        # 128-bit security bound rests on - expect very low or zero
        # success there with a reasonable trial budget. That is the
        # expected, interesting result at this scale, not a bug.
        #
        # So: the full hyperparameter grid runs only at the small sizes;
        # the large sizes get a much thinner grid (fixed q=127, only two
        # alpha/beta/rmax/budgets values, weight_factor fixed higher since
        # the true minimum distance is more likely to require a larger
        # multiplier to be reachable at all, one trial), just enough to
        # observe scaling behavior and compare the three enumerators.
        small_combos = list(itertools.product(
            small_nk_values, q_values, alpha_values, beta_values, rmax_values,
            weight_factor_values, budgets_values, enum_version_values,
            repair_algorithm_values, range(num_trials),
        ))
        large_combos = list(itertools.product(
            large_nk_values, [127],
            [0.01, 0.05], [0.05, 0.2], [1, 2, 3,],
            [1.1, 1.2, 1.3], [3], enum_version_values,
            repair_algorithm_values, range(2),
        ))
        # combos = small_combos + large_combos
        combos = large_combos

    m_pair = 1 # experiment also with 1 for LESS parameters
    n_iter = 1000
    max_trials_enum = 10000

    done = load_completed_keys(csv_path)
    print(f"Runs already completed found in checkpoint: {len(done)}")

    total = len(combos)
    print(f"Total runs under consideration: {total}")

    for idx, ((n, k), q, alpha, beta, rmax, weight_factor, budgets, enum_version,
              repair_algorithm, trial) in enumerate(combos, start=1):
        key = (n, k, q, alpha, beta, rmax, weight_factor, budgets, enum_version,
               repair_algorithm, trial)
        if key in done:
            continue

        print(
            f"[{idx}/{total}] n={n} k={k} q={q} alpha={alpha} beta={beta} "
            f"rmax={rmax} weight_factor={weight_factor} budgets={budgets} "
            f"enum={enum_version} repair={repair_algorithm} trial={trial} ...",
            end=' ', flush=True,
        )

        result = run_single_experiment(
            n, k, q, alpha, beta, rmax, weight_factor, budgets, enum_version,
            repair_algorithm, trial,
            m_pair=m_pair, n_iter=n_iter, max_trials_enum=max_trials_enum,
        )
        append_result(csv_path, result)

        status = 'OK' if result['success'] else ('ERROR' if result['error'] else 'no success')
        prefix = '\n  -> ' if VERBOSE else ''
        print(f"{prefix}{status}  (total_time={result['total_time_s']:.2f}s)")

    print(f"\nSweep finished. Results in: {csv_path}")


if __name__ == "__main__":
    main()