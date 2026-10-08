"""
list_based_row_domains.py  (Code/strategies/07_list_based_prediction_and_repair/)

Fix for the list-based leakage / search-tree incompatibility documented in
list_based_leakage_problem_and_fix.md (write-up shared with the user,
2026-09-22, before this file was implemented). Summary of the problem:

compute_posterior_table_list_based (LEP_prediction_and_repair_v2.py) folds
the two INDEPENDENT list-based leakage channels - the permutation channel
p_i^perm(j) and the value channel p_i^scale(a) - into a single
table[i][j][a] = p_i^perm(j) * p_i^scale(a) before build_row_domains ever
sees it. Because p_i^scale does not depend on j, monomial_approximation's
D_loc[i][j] = argmax_a table[i][j][a] collapses to the SAME argmax_a
p_i^scale(a) for every column j in a given row - exactly what Algorithm 5
(MonomialApproximationVector, manuscript Section 5.5) computes as d_hat_i,
one scalar per row, independent of column. This is not a bug: under the
two-vector leakage model the value channel genuinely carries no
column-specific information (Section 5.5: "no cross-terms between pi(i)
and d_i"). But collapsing it to a single argmax BEFORE the search tree
runs throws away every other coefficient candidate the value channel
still had non-trivial mass on. When that one argmax happens to be wrong
(observed on a real run: two active rows whose true coefficient was 51,
with D_loc retaining 116 and 55 respectively), the correct (column,
coefficient) pair is never constructed as a RowOption at all - at any L
or v_max, regardless of r, since raising L only retains more COLUMNS,
never a different coefficient for a column that is already retained.

This module provides the two pieces needed to fix that, without touching
compute_posterior_table_list_based (left completely unchanged, so any
existing caller keeps working exactly as before):

  compute_channel_posteriors_list_based  - returns the two per-row
      posterior channels (permutation, value) SEPARATELY, instead of
      immediately folding them into table[i][j][a]. This is almost the
      same loop body as compute_posterior_table_list_based's, just
      stopping one step earlier, before the fold.

  build_row_domains_list_based           - build_row_domains's
      counterpart for list-based leakage: instead of one frozen
      coefficient D_loc[i][j] per column (which is necessarily identical
      across every column in a row, per the above), builds the FULL
      candidate universe {(j, a) : j in [n], a in F_q^*} - all n*(q-1)
      row hypotheses for the row - scored individually with the
      manuscript's own two-vector joint decomposition (Section 4.2.1's
      Definition 9, two-vector case):

          gamma_i(j, a) = log p_i^perm(j) + log p_i^scale(a)

      and keeps the overall top-L. This mirrors exactly what
      core/structured_repair_search_tree.py's build_row_domains now does
      for the matrix representation via row_hypothesis_universe (see that
      function's docstring) - both representations now rank the SAME kind
      of object (individually-scored (j, a) pairs from the full universe),
      just with a different score formula, per Algorithm 8's own split
      between the matrix and two-vector cases. This replaces the earlier
      "top-L columns x top-M coefficients, then trim" heuristic entirely:
      that shortlist-then-cross approach is provably sufficient only when
      M >= L (a consequence of gamma_i(j, a) being additively separable in
      j and a - the top-L sums of two sorted lists are always found within
      the top-L of each list), so at M < L it could clip genuine top-L
      hypotheses; building the full universe directly removes that
      precondition and matches Algorithm 8 exactly, at negligible extra
      cost (n*(q-1) candidates per active row - a few tens of thousands
      of scored tuples even at LESS Category 1 scale, sorted once).

NOT RUN WITH `sage` IN THIS SESSION: like every other Sage-dependent file
written in this sandbox (see e.g. the validation note at the top of
structured_repair_search_tree.py), this module was written with no
SageMath installation available and has not itself been executed here.
planted_codeword_repair_test_search_tree.py's small_test() is what
exercises this module against the real pipeline and needs an actual
`sage` run to confirm it before relying on it further.
"""
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'core'))

from LEP_prediction_and_repair_v2 import build_bit_channel_matrix_exact, _safe_log, _vprint
from structured_repair_search_tree import RowOption


def compute_channel_posteriors_list_based(noisy_permutation, noisy_values, n, q, alpha, beta,
                                           is_permutation=False):
    """
    List-based counterpart of compute_posterior_table_list_based that
    returns the two per-row leakage channels SEPARATELY instead of
    folding them into a single table[i][j][a]. See the module docstring
    for why the fold throws away information build_row_domains_list_based
    needs back.

    Reuses the thesis-exact channel construction,
    build_bit_channel_matrix_exact, which scores every candidate directly
    against the RAW observed bit string with no modulo folding (see that
    function's docstring in LEP_prediction_and_repair_v2.py) - the same
    construction compute_posterior_table_list_based and
    compute_vector_posteriors_list_based use. This function only stops one
    step earlier, before multiplying the two posteriors together.

    :param noisy_permutation: list of n leaked permutation indices, RAW
        bit-flip observations in [0, 2**ceil(log2(n)) - 1] - used directly,
        with NO reduction modulo n
    :param noisy_values: list of n leaked diagonal values, RAW bit-flip
        observations in [0, 2**ceil(log2(q)) - 1] - used directly, with NO
        reduction modulo q
    :param n: degree of the monomial (and length of both lists)
    :param q: size of the finite field
    :param alpha: probability that a 1 bit flips to 0
    :param beta: probability that a 0 bit flips to 1
    :param is_permutation: whether the secret is a PEP permutation (values
        fixed to 1) rather than an LEP monomial - matches
        compute_posterior_table_list_based's own parameter
    :return: (perm_posteriors, scale_posteriors)
        perm_posteriors[i]  - a length-n list, perm_posteriors[i][j] =
            Pr[pi(i) = j | leakage] (same shape as compute_posterior_table_list_based's
            internal col_post)
        scale_posteriors[i] - a dict {a: Pr[d_i = a | leakage]} over
            nonzero field elements a in F = GF(q) (same shape as that
            function's internal val_post)
    """
    from sage.all import GF
    F = GF(q)

    _vprint(f"      [posterior_channels_list] building n={n} per-row perm/value "
            f"posteriors (q={q}) from lists...", end='', flush=True)
    t0 = time.time()

    channel_n = build_bit_channel_matrix_exact(n, alpha, beta)
    channel_q = build_bit_channel_matrix_exact(q, alpha, beta)

    perm_posteriors = [None] * n
    scale_posteriors = [None] * n

    for i in range(n):
        h_obs = int(noisy_permutation[i])
        v_obs = int(noisy_values[i])

        col_raw = [channel_n[j][h_obs] for j in range(n)]
        col_total = sum(col_raw)
        perm_posteriors[i] = [c / col_total for c in col_raw] if col_total > 0 else [1 / n] * n

        if is_permutation:
            scale_posteriors[i] = {F(1): 1.0}
        else:
            val_raw = {a: channel_q[int(a)][v_obs] for a in F if a != F(0)}
            val_total = sum(val_raw.values())
            if val_total > 0:
                scale_posteriors[i] = {a: val_raw[a] / val_total for a in val_raw}
            else:
                scale_posteriors[i] = {a: 1 / (q - 1) for a in val_raw}

        if n >= 32 and (i + 1) % max(1, n // 10) == 0:
            _vprint(f" row {i + 1}/{n}", end='', flush=True)

    _vprint(f" done ({time.time() - t0:.2f}s)", flush=True)
    return perm_posteriors, scale_posteriors


def build_row_domains_list_based(v, A, perm_posteriors, scale_posteriors, Q_hat, H_prime, F, L):
    """
    build_row_domains's counterpart for list-based leakage. Builds, for
    every active row i in A = Supp(v), the finite retained-option domain
    D_i, ranking the FULL candidate universe {(j, a) : j in [n], a in
    F_q^*} - not one coefficient locked per column (see the module
    docstring for why that was wrong, and for how this now mirrors
    core/structured_repair_search_tree.py's build_row_domains exactly,
    modulo the score formula).

    The keep option K_i = (pi_tilde(i), d_tilde_i), read off Q_hat exactly
    as in build_row_domains, is always included at cost 0.

    Unlike build_row_domains (matrix representation), there is no
    zero-entry term here: the two-vector model's own joint score
    (manuscript Definition 9, two-vector case) is simply
    log p_i^perm(j) + log p_i^scale(a), with no "all other entries in the
    row are zero" probability to add in, since the two-vector model never
    represents that probability in the first place.

    :param v: a row vector v in C
    :param A: the active support Supp(v) (any order; not assumed sorted)
    :param perm_posteriors: perm_posteriors[i][j] = Pr[pi(i) = j | leakage],
        as returned by compute_channel_posteriors_list_based
    :param scale_posteriors: scale_posteriors[i] = {a: Pr[d_i = a | leakage]},
        as returned by compute_channel_posteriors_list_based
    :param Q_hat: the monomial approximation of the secret Q (read for the
        keep choice only, exactly as in build_row_domains)
    :param H_prime: a parity-check matrix of C' (used only for its
        columns, to precompute sigma_i(j, a) exactly as in
        build_row_domains)
    :param F: the base field GF(q)
    :param L: the number of non-keep options retained per active row in
        the domain (same meaning as build_row_domains's L)
    :return: {i: [RowOption, ...]} for i in A, each list sorted by
        nonincreasing log_score with the same tie rule as build_row_domains
    """
    n = Q_hat.ncols()
    columns = [H_prime.column(j) for j in range(n)]

    domains = {}
    for i in A:
        pi_tilde_i = next(j for j in range(n) if Q_hat[i, j] != 0)
        d_tilde_i = Q_hat[i, pi_tilde_i]

        col_post = perm_posteriors[i]
        val_post = scale_posteriors[i]

        # Full candidate universe: every (j, a) pair, scored individually
        # via gamma_i(j, a) = log p_i^perm(j) + log p_i^scale(a) - not a
        # per-column argmax, and not a top-L x top-M shortlist cross. The
        # exact (pi_tilde_i, d_tilde_i) pair is excluded here since it is
        # the keep choice, added back below at cost 0.
        candidates = []
        for j in range(n):
            log_pj = _safe_log(col_post[j])
            for a, p_a in val_post.items():
                if j == pi_tilde_i and a == d_tilde_i:
                    continue
                score = log_pj + _safe_log(p_a)
                # sigma_i(j, a) (5.4) - unchanged formula from build_row_domains;
                # it only depends on the hypothesis (j, a), not on how that
                # hypothesis was scored or selected.
                sigma = v[i] * (d_tilde_i * columns[pi_tilde_i] - a * columns[j])
                candidates.append(RowOption(j=j, a=a, cost=1, log_score=score, sigma=sigma))

        candidates.sort(key=lambda opt: opt.log_score, reverse=True)
        candidates = candidates[:L]

        keep_score = _safe_log(col_post[pi_tilde_i]) + _safe_log(val_post.get(d_tilde_i, 0.0))
        keep_sigma = v[i] * (d_tilde_i * columns[pi_tilde_i] - d_tilde_i * columns[pi_tilde_i])
        keep_option = RowOption(j=pi_tilde_i, a=d_tilde_i, cost=0, log_score=keep_score,
                                 sigma=keep_sigma)

        options = [keep_option] + candidates
        options.sort(key=lambda opt: (-opt.log_score, opt.j, opt.a))
        domains[i] = options

    return domains
