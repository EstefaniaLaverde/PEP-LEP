import json
import math
import os
import sys
import time
from datetime import datetime, timezone

# select the parameter sweep, the planted codeword repair and the list based prediction and repair
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'core'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '05_full_parameter_sweep'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '06_planted_codeword_repair_isolation'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '07_list_based_prediction_and_repair'))

from sage.all import GF, set_random_seed, save as sage_save

from instances_generator import obtain_parity_check_matrix
from LEP_prediction_and_repair_v2 import (
    compute_posterior_table,
    compute_posterior_table_exact,
    compute_posterior_table_list_based,
    monomial_approximation,
    predict_image,
    support,
    active_error_set,
    set_verbose,
    row_hypothesis_universe,
    _safe_log,
)
from structured_repair_search_tree import (
    build_row_domains,
    build_word_from_choice,
    RowEnumerationIterator,
)
from list_based_row_domains import (
    compute_channel_posteriors_list_based,
    build_row_domains_list_based,
)
from instances_generator_monomial_as_list import build_leaked_monomial_matrix
from posterior_repair_radius import active_row_error_probabilities, posterior_repair_radius
from run_lep_experiments import gilbert_varshamov_bound


from planted_codeword_repair_test import build_generator_with_planted_low_weight_row
from planted_codeword_repair_test_list_based import generate_noisy_LCE_instance_from_G1_list_based
from planted_codeword_repair_test import generate_noisy_LCE_instance_from_G1

set_verbose(True)

SAVE_RUN = True
RUNS_DIR = 'runs'
COMPLETED_MANIFEST = os.path.join(RUNS_DIR, 'completed_combos.json')

# LESS's actual NIST Category 1 parameters
LESS_N, LESS_K, LESS_Q = 252, 126, 127

# Full alpha x beta grid 
ALPHAS = [0.005, 0.01, 0.015, 0.02]
BETAS = [0.05, 0.1, 0.15, 0.2]
ALPHA_BETA_COMBOS = [(a, b) for a in ALPHAS for b in BETAS]

# posterior_repair_radius's target failure probability
DELTAS = [0.05, 0.15, 0.3]

REPRESENTATIONS = ['monomial_based', 'list_based', 'list_based_naive']
SEEDS = list(range(1, 6))  # 5 seeds per (alpha, beta, delta, representation) combo in main()

WEIGHT_FACTOR = 1.1   # multiplies the GV bound, matching every other script here

BUDGETS_L = 100
DELTA = 0.3
V_MAX_POPS = 30000 # hard pop cap per run

# L_MIN / L_SCALE: non-keep-candidate budget L, now a function of (n, q)
# instead of one fixed constant for every tier. row_hypothesis_universe
# ranks the full n * (q - 1) (destination, coefficient) candidate universe
# per active row, so L is grown with sqrt(n * (q - 1)) and floored at
# L_MIN so small instances keep a reasonable safety margin.
#
# L_SCALE = 0.8 is anchored so budgets_for reproduces this script's
# original fixed default, BUDGETS_L = 100, at medium_test()'s tier
# (n=124, q=127): 0.8 * sqrt(124 * 126) ~= 100.0. At small_test()'s tier
# (n=32, q=127): ~51. At main()'s LESS Category 1 tier (n=252, q=127): ~143.
L_MIN = 20
L_SCALE = 0.8


def budgets_for(n, q):
    """
    Size-dependent per-active-row candidate budget L(n, q) - see the
    L_MIN / L_SCALE comment above for the formula and its anchoring.
    """
    return max(L_MIN, math.ceil(L_SCALE * math.sqrt(n * (q - 1))))

def _fmt_float_for_filename(x):
    """'0.001' -> '0p001', so filenames stay filesystem- and shell-safe."""
    return str(x).replace('.', 'p')


def save_run(params, G1, G2, Q, Q_noisy_or_lists, v, Q_hat, w_tilde, w,
             r_v, r, visited, elapsed, outcome):
    """
    Saves one run's problem instance and outcome to RUNS_DIR, following the
    same .sobj + .json convention as save_run in
    planted_codeword_repair_test.py, extended with the fields this sweep
    needs (representation, r/visited/V_MAX_POPS instead of rmax/tau, and a
    richer run_id since alpha/beta/representation vary across the sweep
    instead of being fixed per script).

    :param params: dict of the run's input parameters (n, k, q, alpha,
        beta, representation, weight_factor, budgets, delta, v_max,
        seed, target_weight)
    :param G1: generator matrix of C (with the planted row as row 0)
    :param G2: generator matrix of C'
    :param Q: the true secret monomial matrix (ground truth, for this
        controlled test only)
    :param Q_noisy_or_lists: the noisy hint - a matrix for 'monomial_based',
        a {'noisy_permutation':.., 'noisy_values':..} dict for 'list_based'
        (mirroring planted_codeword_repair_test_list_based.py's own
        save_run call), or a {'noisy_permutation':.., 'noisy_values':..,
        'naive_hint':..} dict for 'list_based_naive' (the same two lists,
        plus the scattered matrix build_leaked_monomial_matrix produced
        from them and that was actually fed to compute_posterior_table)
    :param v: the planted min-weight codeword (v0), in C
    :param Q_hat: the monomial approximation built from the noisy hint
    :param w_tilde: predict_image(v, Q_hat)
    :param w: the repaired word, or None if repair did not produce one
    :param r_v: the TRUE active error dimension of v (Definition 3),
        measured against the real secret Q - a ground-truth diagnostic
        only, saved for post-hoc comparison against r; never used to
        derive r itself (see the module docstring)
    :param r: the departure radius actually used by the search tree -
        posterior_repair_radius's output, not r_v
    :param visited: RowEnumerationIterator.visited after the run - the
        number of states popped from the queue
    :param elapsed: seconds spent in next_candidate()
    :param outcome: one of 'correct', 'wrong', 'failed_exhausted',
        'failed_limit'
    :return: the base path (without extension) the run was saved under
    """
    os.makedirs(RUNS_DIR, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    run_id = (
        f"n{params['n']}_k{params['k']}_q{params['q']}_"
        f"a{_fmt_float_for_filename(params['alpha'])}_"
        f"b{_fmt_float_for_filename(params['beta'])}_"
        f"{params['representation']}_seed{params['seed']}_{timestamp}"
    )
    base_path = os.path.join(RUNS_DIR, run_id)

    data = {
        'params': params,
        'G1': G1,
        'G2': G2,
        'Q': Q,
        'Q_noisy_or_lists': Q_noisy_or_lists,
        'v': v,
        'Q_hat': Q_hat,
        'w_tilde': w_tilde,
        'w': w,
        'r_v': r_v,
        'r': r,
        'visited': visited,
        'elapsed_s': elapsed,
        'outcome': outcome,
    }
    sage_save(data, base_path)

    summary = {
        **params,
        'v_weight': v.hamming_weight(),
        'r_v': r_v,
        'r': r,
        'visited': visited,
        'elapsed_s': elapsed,
        'outcome': outcome,
        'repaired': w is not None,
    }
    with open(base_path + '.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  [save_run] saved instance + outcome to {base_path}.sobj "
          f"(+ {base_path}.json summary)")
    return base_path


def _load_completed():
    if not os.path.exists(COMPLETED_MANIFEST):
        return set()
    with open(COMPLETED_MANIFEST) as f:
        rows = json.load(f)
    return {tuple(row) for row in rows}


def _mark_completed(key):
    os.makedirs(RUNS_DIR, exist_ok=True)
    completed = _load_completed()
    completed.add(tuple(key))
    with open(COMPLETED_MANIFEST, 'w') as f:
        json.dump(sorted(completed), f, indent=2)


# ---------------------------------------------------------------------------
# One run: plant a low-weight codeword, leak the secret monomial (either
# representation), build Q_hat, then repair predict_image(v, Q_hat) with
# the Section 5 best-first search tree.
# ---------------------------------------------------------------------------

def run_instance(n, k, q, alpha, beta, representation, weight_factor=WEIGHT_FACTOR,
                  budgets=None, delta=DELTA, v_max=V_MAX_POPS,
                  seed=1, label=""):
    """
    Plants a low-weight codeword v0 into C, leaks the secret monomial under
    the requested representation, builds Q_hat, then repairs
    predict_image(v0, Q_hat) with the Section 5 best-first search tree
    (RowEnumerationIterator), reporting whether the recovered word is
    genuinely v0 * Q (checked against the true secret - only possible in
    this controlled test).

    :param n, k, q: code and field parameters
    :param alpha, beta: bit-flip channel parameters
    :param representation: 'monomial_based' or 'list_based' - see the
        module docstring for exactly what each one builds Q_hat from
    :param weight_factor: multiplies the Gilbert-Varshamov bound to get
        the planted codeword's exact target weight
    :param budgets: non-keep options retained per active row (passed to
        build_row_domains as L). None (the default) computes the
        size-dependent budgets_for(n, q) instead of a fixed constant - see
        that function for the formula.
    :param delta: the search tree's departure radius r is derived from the
        posterior ALONE (no ground truth) via
        posterior_repair_radius.active_row_error_probabilities +
        posterior_repair_radius, as the smallest r with
        Pr[r(v) <= r] >= 1 - delta under the posterior's own per-row error
        estimates - see the module docstring for why this replaced the
        earlier ground-truth-based r_v + margin derivation. The TRUE active
        error dimension r_v is still measured and saved on every run, but
        purely as a diagnostic - it never feeds into r
    :param v_max: pop cap passed to RowEnumerationIterator
    :param seed: fixed Sage random seed
    :param label: a short string printed before this run's output
    :return: a dict summarizing the run (also see save_run for what is
        persisted to disk)
    """
    if budgets is None:
        budgets = budgets_for(n, q)
    set_random_seed(seed)
    F = GF(q)
    print(f"\n{'=' * 70}\n{label} n={n} k={k} q={q} alpha={alpha} beta={beta} "
          f"[{representation}] seed={seed} delta={delta} L={budgets}\n{'=' * 70}")

    gv_bound = gilbert_varshamov_bound(n, k, q)
    target_weight = max(1, round(gv_bound * weight_factor))
    print(f"  GV bound: {gv_bound}; planted target_weight = {target_weight}")

    t0 = time.time()
    G1, v0 = build_generator_with_planted_low_weight_row(n, k, q, target_weight)
    print(f"  planted-row generator matrix built ({time.time() - t0:.2f}s); "
          f"v0 weight = {v0.hamming_weight()}")

    t0 = time.time()
    perm_posteriors, scale_posteriors = None, None
    if representation == 'monomial_based':
        G2, Q, Q_noisy = generate_noisy_LCE_instance_from_G1(G1, q, alpha, beta, is_monomial=True)
        Q_noisy_or_lists = Q_noisy
        # Q_noisy is the RAW (unreduced) leaked hint - use the thesis-exact
        # scorer, not compute_posterior_table (which assumes an
        # already-GF(q)-reduced hint; see build_bit_channel_matrix_exact's
        # docstring in core/LEP_prediction_and_repair_v2.py).
        posterior_table = compute_posterior_table_exact(Q_noisy, n, q, alpha, beta, is_permutation=False)
    elif representation == 'list_based':
        G2, Q, permutation, values, noisy_permutation, noisy_values = \
            generate_noisy_LCE_instance_from_G1_list_based(G1, q, alpha, beta, is_monomial=True)
        Q_noisy_or_lists = {'noisy_permutation': noisy_permutation, 'noisy_values': noisy_values}
        # Folded table - still needed for Q_hat/S/D_loc/pi via
        # monomial_approximation below, and for active_row_error_probabilities
        # (which only needs the folded per-(i,j) posterior, not the split
        # channels).
        posterior_table = compute_posterior_table_list_based(
            noisy_permutation, noisy_values, n, q, alpha, beta, is_permutation=False
        )
        # The two channels SEPARATELY (not folded) - this is what
        # build_row_domains_list_based needs to avoid collapsing to a
        # single coefficient per row. See list_based_row_domains.py's
        # module docstring / list_based_leakage_problem_and_fix.md.
        perm_posteriors, scale_posteriors = compute_channel_posteriors_list_based(
            noisy_permutation, noisy_values, n, q, alpha, beta, is_permutation=False
        )
    elif representation == 'list_based_naive':
        # SAME two-list leakage as 'list_based' (same generator call, same
        # noisy_permutation/noisy_values) - the difference is entirely in
        # how it's turned into a monomial approximation from here.
        G2, Q, permutation, values, noisy_permutation, noisy_values = \
            generate_noisy_LCE_instance_from_G1_list_based(G1, q, alpha, beta, is_monomial=True)
        # Scatter the two lists into a single n x n matrix: row i's leaked
        # value goes into column (noisy_permutation[i] mod n). This is NOT
        # a valid monomial matrix - two rows can collide on the same
        # column while another gets none - see build_leaked_monomial_matrix's
        # own docstring and algorithm5_demo.ipynb's "naive hint" comparison
        # cell for the full explanation.
        naive_hint = build_leaked_monomial_matrix(n, q, noisy_permutation, noisy_values)
        Q_noisy_or_lists = {
            'noisy_permutation': noisy_permutation,
            'noisy_values': noisy_values,
            'naive_hint': naive_hint,
        }
        # From here on this is IDENTICAL to 'monomial_based': feed the
        # (naive, scatter-noise) matrix through the original whole-matrix
        # pipeline. compute_posterior_table assumes per-entry bit-flip
        # noise, which is not exactly what naive_hint has (collision/gap
        # noise instead) - that mismatch is the whole point of this
        # representation, i.e. whether the pipeline is robust to it anyway.
        posterior_table = compute_posterior_table(naive_hint, alpha, beta, is_permutation=False)
    else:
        raise ValueError(f"unknown representation: {representation!r}")
    print(f"  [{representation}] instance generated ({time.time() - t0:.2f}s)")

    t0 = time.time()
    Q_hat, S, D_loc, pi = monomial_approximation(posterior_table, F)
    rows_full = sum(1 for i in range(n) if list(Q_hat[i]) == list(Q[i]))
    print(f"  Q_hat built ({time.time() - t0:.2f}s); {rows_full}/{n} rows exactly correct")

    H2 = obtain_parity_check_matrix(G2)

    v = v0
    A = support(v)

    # Ground-truth diagnostic ONLY - never fed into r. Ground truth (Q) is
    # available here purely because this is a controlled, planted-codeword
    # test; a real attacker could not compute this.
    wrong_rows = active_error_set(v, Q, Q_hat)
    r_v = len(wrong_rows)

    # --- Diagnostic: for each truly-wrong active row, is its TRUE
    # (destination, coefficient) pair even reachable by the domain the
    # tree will actually search? Now that build_row_domains (monomial_based
    # / list_based_naive) and build_row_domains_list_based (list_based)
    # BOTH rank the full (j, a) candidate universe and keep only the
    # overall top-`budgets`, the reachability check is the same shape for
    # every representation: where does (j_true, a_true) rank in that same
    # universe? See row_hypothesis_universe's docstring
    # (core/structured_repair_search_tree.py) and
    # list_based_leakage_problem_and_fix.md for the bug this replaced (a
    # single coefficient locked per column, independently of a). This is
    # ground-truth diagnostics only (uses the real secret Q), same caveat
    # as r_v/wrong_rows above - never used to drive the search.
    unreachable_rows = []
    for i in sorted(wrong_rows):
        j_true = next(j for j in range(n) if Q[i, j] != 0)
        a_true = Q[i, j_true]
        pi_tilde_i = next(j for j in range(n) if Q_hat[i, j] != 0)
        d_tilde_i = Q_hat[i, pi_tilde_i]

        if (j_true, a_true) == (pi_tilde_i, d_tilde_i):
            # Can't happen: i is in wrong_rows exactly because
            # Q_hat[i] != Q[i], i.e. the keep choice is NOT the true row.
            continue

        if representation == 'list_based':
            col_post = perm_posteriors[i]
            val_post = scale_posteriors[i]
            universe = sorted(
                ((j, a, _safe_log(col_post[j]) + _safe_log(p_a))
                 for j in range(n) for a, p_a in val_post.items()
                 if (j, a) != (pi_tilde_i, d_tilde_i)),
                key=lambda t: t[2], reverse=True,
            )
        else:
            universe = row_hypothesis_universe(i, posterior_table[i], F, n,
                                                 exclude=(pi_tilde_i, d_tilde_i))

        rank = next((idx for idx, (j, a, _s) in enumerate(universe)
                     if (j, a) == (j_true, a_true)), None)
        kept = rank is not None and rank < budgets
        if not kept:
            unreachable_rows.append((i, j_true, a_true, rank))

    if unreachable_rows:
        print(f"  DIAGNOSTIC: {len(unreachable_rows)}/{len(wrong_rows)} truly-wrong active "
              f"row(s) have their TRUE (destination, coefficient) excluded from the domain "
              f"the tree will search (budgets=L={budgets}) - the search tree CANNOT reach "
              f"the correct repair for these rows regardless of r or v_max:")
        for i, j_true, a_true, rank in unreachable_rows:
            rank_desc = f"ranks #{rank}" if rank is not None else "has zero/degenerate posterior mass"
            print(f"    row {i}: true (dest={j_true}, coeff={a_true!r}) {rank_desc} in the "
                  f"full candidate universe (only top {budgets} are retained)")
        print(f"  -> raise budgets (L) so these true hypotheses get retained, or accept that "
              f"{representation} at this alpha/beta won't self-correct these rows; raising "
              f"v_max cannot fix this, since the correct leaf isn't in the tree at all")
    elif wrong_rows:
        print(f"  DIAGNOSTIC: all {len(wrong_rows)} truly-wrong active row(s) have their true "
              f"(destination, coefficient) retained in the domain the tree will search - the "
              f"tree SHOULD be able to reach the correct repair structurally; a failure would "
              f"point elsewhere (r too small to admit it, or v_max too small to reach it in time)")


    # The actual radius driving the search: posterior-only, via
    # posterior_repair_radius.py's PosteriorRepairRadius (Poisson-binomial
    # DP over per-active-row error estimates read off the posterior table
    # alone - see the module docstring).
    p_err = active_row_error_probabilities(posterior_table, pi, A)
    r, pmf = posterior_repair_radius(list(p_err.values()), delta=delta)
    mean_p_err = sum(p_err.values()) / len(p_err) if p_err else 0.0
    max_p_err = max(p_err.values()) if p_err else 0.0

    print(f"  posterior_repair_radius: r = {r} (delta = {delta}, mean p_err = {mean_p_err:.4f}, "
          f"max p_err = {max_p_err:.4f}) | ground-truth r_v = {r_v} (diagnostic only) "
          f"| |Supp(v)| = {len(A)}, L = {budgets}, v_max = {v_max}")
    if r < r_v:
        print(f"  NOTE: posterior-approximate r ({r}) is BELOW the true r_v ({r_v}) - "
              f"the tree cannot reach the true row assignment at this radius; a failure "
              f"here would reflect the posterior underestimating its own error, not a "
              f"search-tree bug")

    t0 = time.time()
    w_tilde = predict_image(v, Q_hat)
    if representation == 'list_based':
        domains = build_row_domains_list_based(
            v, A, perm_posteriors, scale_posteriors, Q_hat, H2, F, budgets
        )
    else:
        domains = build_row_domains(v, A, posterior_table, Q_hat, H2, F, budgets)
    active_order = sorted(A)
    s = H2 * w_tilde

    it = RowEnumerationIterator(active_order, domains, s, r, v_max=v_max)
    result = it.next_candidate(build_word=lambda choice: build_word_from_choice(v, choice, n, F))
    elapsed = time.time() - t0

    w = None
    if result == 'Limit':
        outcome_tag = 'failed_limit'
        print(f"  [structured_sd_repair_bestfirst] hit v_max={v_max} pops before finding a "
              f"candidate ({elapsed:.2f}s, visited={it.visited})")
    elif result == 'Exhausted':
        outcome_tag = 'failed_exhausted'
        print(f"  [structured_sd_repair_bestfirst] tree exhausted, no accepting leaf within "
              f"r={r} ({elapsed:.2f}s, visited={it.visited})")
    else:
        _choice, w, g = result
        is_correct = (w == v * Q)
        outcome = 'CORRECT equivalent pair' if is_correct else 'weight/syndrome-consistent but WRONG pair'
        print(f"  [structured_sd_repair_bestfirst] repaired in {elapsed:.2f}s "
              f"(visited={it.visited}, g={g:.3f}) -> {outcome}")
        outcome_tag = 'correct' if is_correct else 'wrong'

    if SAVE_RUN:
        params = {
            'n': n, 'k': k, 'q': q, 'alpha': alpha, 'beta': beta,
            'representation': representation, 'weight_factor': weight_factor,
            'budgets': budgets, 'delta': delta, 'v_max': v_max, 'seed': seed,
            'target_weight': target_weight, 'gv_bound': gv_bound,
            'rows_correct_full': rows_full, 'repair_algorithm': 'structured_sd_repair_bestfirst',
            'mean_p_err': mean_p_err, 'max_p_err': max_p_err,
            'r_approx_ge_r_v': r >= r_v,
        }
        save_run(params, G1, G2, Q, Q_noisy_or_lists, v, Q_hat, w_tilde, w,
                 r_v, r, it.visited, elapsed, outcome_tag)

    return {'v': v, 'r_v': r_v, 'r': r, 'visited': it.visited, 'outcome': outcome_tag}


def main():
    """
    Full LESS Category 1 sweep (n=252, k=126, q=127) over the full grid:
    16 (alpha, beta) combos x 3 delta values x 3 representations x 5 seeds
    = 720 runs total. L is fixed per this tier via budgets_for(LESS_N, LESS_Q)
    (~143 - see that function's docstring).
    """
    budgets = budgets_for(LESS_N, LESS_Q)
    combos = [
        (alpha, beta, delta, representation, seed)
        for (alpha, beta) in ALPHA_BETA_COMBOS
        for delta in DELTAS
        for representation in REPRESENTATIONS
        for seed in SEEDS
    ]
    completed = _load_completed()
    print(f"Total combinations: {len(combos)}; already completed: {len(completed)}; L={budgets}")

    for idx, (alpha, beta, delta, representation, seed) in enumerate(combos, start=1):
        key = (LESS_N, LESS_K, LESS_Q, alpha, beta, delta, representation, seed)
        if key in completed:
            continue

        run_instance(
            n=LESS_N, k=LESS_K, q=LESS_Q, alpha=alpha, beta=beta,
            representation=representation, delta=delta, budgets=budgets, seed=seed,
            label=f"[{idx}/{len(combos)}, LESS Category 1, search-tree repair]",
        )
        _mark_completed(key)

    print(f"\nSweep finished. Results in: {RUNS_DIR}/")


def small_test():
    """
    A fast, small-scale sanity check (n=32, so target_weight and the tree
    size stay small) - run this FIRST, before main()'s full 720-run LESS
    Category 1 sweep, to confirm the pipeline works end to end and to get
    a feel for actual wall-clock time before committing to the full sweep.

    Runs the SAME full grid as main() (16 alpha/beta combos x 3 delta
    values x 3 representations), but at a single fixed seed=1, giving
    16 * 3 * 3 = 144 runs. L is fixed per this tier via
    budgets_for(32, 127) (~51 - see that function's docstring).
    """
    n, k, q = 32, 16, 127
    budgets = budgets_for(n, q)
    combos = [
        (alpha, beta, delta, representation)
        for (alpha, beta) in ALPHA_BETA_COMBOS
        for delta in DELTAS
        for representation in REPRESENTATIONS
    ]
    print(f"[small_test] n={n} k={k} q={q}: {len(combos)} runs (seed=1), L={budgets}")
    for idx, (alpha, beta, delta, representation) in enumerate(combos, start=1):
        run_instance(
            n=n, k=k, q=q, alpha=alpha, beta=beta,
            representation=representation, delta=delta, budgets=budgets, seed=1,
            label=f"[small_test {idx}/{len(combos)}]"
        )


def medium_test():
    """
    An intermediate-scale sanity check between small_test() (n=32) and
    main()'s full LESS Category 1 sweep (n=252) - n=124 (k=62, keeping
    LESS's k/n = 1/2 rate), so wt(v)/active-row count and r land somewhere
    between the two, giving a feel for how the defaults actually scale
    with n before committing to the full 252-scale sweep. Run this AFTER
    small_test() and BEFORE main().

    Runs the SAME full grid as main() and small_test() (16 alpha/beta
    combos x 3 delta values x 3 representations), at a single fixed
    seed=1, giving 16 * 3 * 3 = 144 runs. L is fixed per this tier via
    budgets_for(124, 127) (~100, the script's original fixed default -
    see that function's docstring for why this tier was chosen as the
    anchor).
    """
    n, k, q = 124, 62, 127
    budgets = budgets_for(n, q)
    combos = [
        (alpha, beta, delta, representation)
        for (alpha, beta) in ALPHA_BETA_COMBOS
        for delta in DELTAS
        for representation in REPRESENTATIONS
    ]
    print(f"[medium_test] n={n} k={k} q={q}: {len(combos)} runs (seed=1), L={budgets}")
    for idx, (alpha, beta, delta, representation) in enumerate(combos, start=1):
        run_instance(
            n=n, k=k, q=q, alpha=alpha, beta=beta,
            representation=representation, delta=delta, budgets=budgets, seed=1,
            label=f"[medium_test {idx}/{len(combos)}]"
        )


if __name__ == "__main__":
    # Run `sage planted_codeword_repair_test_search_tree.py small` for the
    # small sanity check, `... medium` for the n=124 intermediate check;
    # any other/no argument runs the full sweep.
    if len(sys.argv) > 1 and sys.argv[1] == 'small':
        small_test()
    elif len(sys.argv) > 1 and sys.argv[1] == 'medium':
        medium_test()
    else:
        main()
