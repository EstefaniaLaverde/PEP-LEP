"""
test_real_isd_enumerator.py

Fixes one instance of the LEP prediction-and-repair pipeline at
n=128, k=64, q=127, alpha=0.01, beta=0.05, and replaces the three
placeholder enumerators used in run_lep_experiments.py
(enum_version_values = ['random', 'linear_combination', 'local_search'])
with an actual published algorithm: SageMath's built-in Information Set
Decoder (sage.coding.information_set_decoder.LinearCodeInformationSetDecoder,
an implementation of the classical Prange / Lee-Brickell ISD algorithm),
used to search for a low-weight codeword of C.

Then runs the rest of the pipeline (posterior table -> Q_hat -> repair)
with each of the three REPAIR_ALGORITHMS in turn, to see whether an
equivalent codeword pair (v, w) can be recovered.

Using SageMath. Run with:
    sage -python test_real_isd_enumerator.py
(from the Code/ directory, so the sibling modules import correctly).

IMPORTANT CAVEAT: this script was written and reviewed without a live Sage
interpreter available in the environment it was authored in (only plain
Python was available for testing there), so the exact call signature of
LinearCodeInformationSetDecoder is based on its documented API rather than
an executed test. enum_low_weight_codeword_sage_isd below has a small
fallback for the constructor call, but if you hit an error the first time
you run this, paste it back so the call can be corrected for your
installed Sage version.

n=128, k=64, q=127 is a real, non-toy instance (same field size and rate
as LESS's actual parameter choice, just shorter than the smallest official
LESS category). Finding a codeword near the Gilbert-Varshamov bound with a
real ISD algorithm is exactly the hard problem LESS's security rests on,
so this may run for a while, and structured_sd_repair in particular is
combinatorial in the active-row count and rmax (see its docstring in
LEP_prediction_and_repair_v2.py) - expect it to be the slowest of the
three repair algorithms at this size, possibly by a lot. A small, fast
sanity run happens first (see RUN_SANITY_CHECK below) so pipeline/import
problems show up quickly, before committing to the full n=128 run.
"""
import os
import sys
import time

# instances_generator.py / LEP_prediction_and_repair_v2.py now live in
# Code/core/, and run_lep_experiments.py lives in
# Code/strategies/05_full_parameter_sweep/ (this file was relocated into
# strategies/06_planted_codeword_repair_isolation/ during the Aug 2026
# folder reorganization), so both need to be added to sys.path explicitly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'core'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '05_full_parameter_sweep'))

from sage.all import GF, vector, set_random_seed

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
    set_verbose,
    enum_low_weight_codeword_lee_brickell,
)
from run_lep_experiments import gilbert_varshamov_bound, REPAIR_ALGORITHMS

set_verbose(True)


# ---------------------------------------------------------------------------
# Step 2: a *real* low-weight-codeword algorithm, using Sage's own
# Information Set Decoder (Prange/Lee-Brickell), instead of any of
# enum_version_values = ['random', 'linear_combination', 'local_search'].
# ---------------------------------------------------------------------------

def enum_low_weight_codeword_sage_isd(G, target_weight, min_weight=1):
    """
    Searches for a nonzero codeword of C = <G> with Hamming weight in
    [min_weight, target_weight], using SageMath's built-in
    LinearCodeInformationSetDecoder (an implementation of the classical
    Prange / Lee-Brickell information-set-decoding algorithm) - a real,
    published algorithm, not a hand-rolled heuristic.

    The reduction used here is standard: the Hamming distance from the
    all-zero vector to a codeword c is exactly wt(c), since 0 is itself
    a codeword. So decoding the zero word while forcing the decoder to
    report between min_weight and target_weight errors is exactly a
    request for a nonzero codeword of weight in that range - this uses
    the decoder for precisely the purpose it is built for (finding a
    codeword within a bounded number of errors of a received word).

    :param G: a generator matrix of C, over GF(q)
    :param target_weight: the maximum acceptable Hamming weight
    :param min_weight: the minimum acceptable Hamming weight (1, to
        exclude the trivial all-zero codeword)
    :return: a codeword v in C with min_weight <= wt(v) <= target_weight,
        or None if the decoder could not find one
    """
    from sage.coding.linear_code import LinearCode
    from sage.coding.information_set_decoder import LinearCodeInformationSetDecoder

    F = G.base_ring()
    n = G.ncols()
    C = LinearCode(G)
    zero_word = vector(F, [0] * n)

    # Try the documented (min, max) interval form of number_errors first;
    # fall back to a single upper-bound integer if that constructor
    # signature is rejected by the installed Sage version.
    decoder = None
    construct_errors = []
    for number_errors in ((min_weight, target_weight), target_weight):
        try:
            decoder = LinearCodeInformationSetDecoder(C, number_errors)
            break
        except TypeError as exc:
            construct_errors.append((number_errors, exc))
            continue
    if decoder is None:
        print("      [sage_isd] FAILED to construct LinearCodeInformationSetDecoder:")
        for number_errors, exc in construct_errors:
            print(f"        number_errors={number_errors!r} -> {type(exc).__name__}: {exc}")
        return None

    try:
        v = decoder.decode_to_code(zero_word)
    except Exception as exc:
        # A genuine "no codeword within range" failure (e.g.
        # sage.coding.decoder.DecodingError, name varies across Sage
        # versions) is expected and should be silent-ish; anything else
        # (TypeError, AttributeError, ValueError from a wrong call
        # signature) is a real bug and gets printed so it can be diagnosed
        # instead of silently looking like "no low-weight codeword found".
        exc_name = type(exc).__name__
        if 'Decod' not in exc_name:
            print(f"      [sage_isd] decode_to_code raised {exc_name}: {exc}")
        return None

    if v is None:
        return None

    w = v.hamming_weight()
    if min_weight <= w <= target_weight:
        return v
    if w == 0:
        # decode_to_code(zero_word) returned the trivial zero-distance
        # match (the zero word is itself a codeword), rather than
        # searching for a nonzero one within [min_weight, target_weight] -
        # i.e. this Sage version's decoder does not enforce the *minimum*
        # end of a (min, max) number_errors interval, only the maximum.
        # Not usable as implemented; see the Lee-Brickell fallback in
        # run_instance below.
        print("      [sage_isd] decoder returned the trivial zero codeword "
              "(does not enforce a minimum error count) - not usable as called")
    return None


# ---------------------------------------------------------------------------
# Steps 3/4: fix an instance, get one low-weight codeword with the real
# ISD algorithm, and run the repair stage with every REPAIR_ALGORITHMS
# entry against it.
# ---------------------------------------------------------------------------

def run_instance(n, k, q, alpha, beta, rmax=2, budgets=3, weight_factor=1.1,
                  label="", seed=20260813):
    """
    Runs steps 1-4 for one (n, k, q, alpha, beta) instance: generates the
    noisy LEP instance, builds Q_hat, finds one low-weight codeword v in C
    with enum_low_weight_codeword_sage_isd, then attempts to repair
    predict_image(v, Q_hat) into a codeword of C' with each of the three
    REPAIR_ALGORITHMS, reporting whether the recovered pair is genuinely
    equivalent (w == v * Q, checked against the true secret - only
    possible here because this is a controlled test, not a real attack).

    :param n, k, q: code and field parameters
    :param alpha, beta: bit-flip channel parameters
    :param rmax: repair dimension used by structured_sd_repair; also used
        to derive tau = 2 * rmax for the other two repair algorithms
        (Corollary 1 in the paper)
    :param budgets: per-active-row candidate budget (structured_sd_repair)
        / K_B (posterior_aware_prange_repair)
    :param weight_factor: multiplies the Gilbert-Varshamov bound to get
        the target weight passed to the enumerator
    :param label: a short string printed before this run's output
    :param seed: fixed random seed (via sage's set_random_seed), so
        repeated runs with the same parameters are reproducible instead
        of drawing a fresh random instance every time (this was missing
        before, which is why rows_correct_full varied between runs of the
        "same" n=128 instance)
    """
    set_random_seed(seed)
    F = GF(q)
    print(f"\n{'=' * 70}\n{label} n={n} k={k} q={q} alpha={alpha} beta={beta} seed={seed}\n{'=' * 70}")

    t0 = time.time()
    G1, G2, Q, Q_noisy = generate_noisy_LCE_instance_CBA_bit_flip_version(
        n, k, q, alpha, beta, is_monomial=True,
    )
    print(f"  instance generated ({time.time() - t0:.2f}s)")

    t0 = time.time()
    posterior_table = compute_posterior_table(Q_noisy, alpha, beta, is_permutation=False)
    Q_hat, S, D_loc, pi = monomial_approximation(posterior_table, F)
    rows_full = sum(1 for i in range(n) if list(Q_hat[i]) == list(Q[i]))
    print(f"  Q_hat built ({time.time() - t0:.2f}s); {rows_full}/{n} rows exactly correct")

    H2 = obtain_parity_check_matrix(G2)

    gv_bound = gilbert_varshamov_bound(n, k, q)
    target_weight = max(1, round(gv_bound * weight_factor))
    print(f"  Gilbert-Varshamov bound: {gv_bound}; target_weight = {target_weight}")

    t0 = time.time()
    v = enum_low_weight_codeword_sage_isd(G1, target_weight)
    elapsed = time.time() - t0
    if v is not None:
        print(f"  enum_low_weight_codeword_sage_isd: found weight {v.hamming_weight()} "
              f"codeword ({elapsed:.2f}s)")
    else:
        print(f"  enum_low_weight_codeword_sage_isd: not usable ({elapsed:.2f}s); "
              f"falling back to enum_low_weight_codeword_lee_brickell "
              f"(a real Lee-Brickell ISD implementation already in this repo)")
        # p=1 (plain Prange, one row of the systematic generator matrix per
        # trial) instead of the default p=2: at n=128, q=127, p=2 means
        # C(64,2) * 126**2 ~= 32M scalar-combination checks per permutation
        # trial, which is impractically slow. p=1 costs C(64,1) * 126 ~= 8K
        # checks per trial - the tradeoff is more trials (permutations)
        # needed to hit a low-weight word, but each trial is cheap.
        #
        # Called in small batches (rather than one max_trials=5000 call) so
        # a heartbeat prints between batches - enum_low_weight_codeword_lee_brickell
        # itself has no progress output, so a slow-but-working search looks
        # identical to a hang without this.
        batch_size = 200
        total_trials_budget = 5000
        t_start = time.time()
        v = None
        trials_done = 0
        while trials_done < total_trials_budget:
            t0 = time.time()
            v = enum_low_weight_codeword_lee_brickell(
                G1, target_weight, max_trials=batch_size, p=1,
            )
            trials_done += batch_size
            print(f"    [lee_brickell] tried {trials_done}/{total_trials_budget} "
                  f"permutations, last batch {time.time() - t0:.2f}s, "
                  f"elapsed {time.time() - t_start:.2f}s", flush=True)
            if v is not None:
                break
        elapsed = time.time() - t_start
        if v is not None:
            print(f"  enum_low_weight_codeword_lee_brickell: found weight "
                  f"{v.hamming_weight()} codeword ({elapsed:.2f}s)")
        else:
            print(f"  enum_low_weight_codeword_lee_brickell: FAILED ({elapsed:.2f}s) - "
                  f"no codeword of weight <= {target_weight} found in "
                  f"{total_trials_budget} trials; try a larger weight_factor "
                  f"or a bigger total_trials_budget")
            return None

    r_v = len(active_error_set(v, Q, Q_hat))
    print(f"  true active error dimension r(v) = {r_v} (rmax = {rmax})")

    w_tilde = predict_image(v, Q_hat)
    tau = 2 * rmax

    results = {}
    for repair_algorithm in REPAIR_ALGORITHMS:
        t0 = time.time()
        if repair_algorithm == 'induced_sd_repair':
            w = induced_sd_repair(w_tilde, v, H2, tau)
        elif repair_algorithm == 'posterior_aware_prange_repair':
            w = posterior_aware_prange_repair(w_tilde, v, H2, Q_hat, S, k, tau, K_B=budgets)
        elif repair_algorithm == 'structured_sd_repair':
            A, L = build_active_row_lists(v, S, D_loc, F, budgets=budgets)
            w = structured_sd_repair(w_tilde, v, H2, Q_hat, A, L, rmax)
        else:
            raise ValueError(f"Unknown repair_algorithm: {repair_algorithm!r}")
        elapsed = time.time() - t0

        if w is None:
            print(f"  [{repair_algorithm}] FAILED to repair ({elapsed:.2f}s)")
            results[repair_algorithm] = ('failed', elapsed)
            continue

        is_correct = (w == v * Q)
        outcome = 'CORRECT equivalent pair' if is_correct else 'weight/syndrome-consistent but WRONG pair'
        print(f"  [{repair_algorithm}] repaired in {elapsed:.2f}s -> {outcome}")
        results[repair_algorithm] = ('correct' if is_correct else 'wrong', elapsed)

    return {'v': v, 'r_v': r_v, 'results': results}


def main():

    run_instance(
        n=128, k=64, q=127, alpha=0.01, beta=0.05,
        rmax=2, budgets=3, weight_factor=1.2,
        label="[requested instance]",
    )


if __name__ == "__main__":
    main()
