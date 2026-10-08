"""
posterior_repair_radius.py  (Code/core/ - shared library, meant to sit next
to instances_generator.py and LEP_prediction_and_repair_v2.py)

Implements the "PosteriorRepairRadius" idea (from the conversation with
Ricardo / ChatGPT, https://chatgpt.com/share/6a8f6b7c-fb08-83e9-a98b-c14ddf7246d6)
as a concrete, testable addition to the r_max approximation work: instead of
deriving a single population-level per-row error probability from the
channel parameters (alpha, beta) alone (as in rmax_approximation.md), this
module reads the row-error probability directly off the entrywise BBLM
posterior table that compute_posterior_table() already builds for one
*specific* observed leakage instance, and aggregates the (now
row-dependent, non-identical) per-row probabilities via a Poisson-binomial
distribution instead of a plain Binomial.

This is deliberately Sage-free: every function here works on plain Python
floats, dicts and lists, so it can be unit-tested (see the __main__ block)
without a Sage installation. The only thing it expects from the caller is
the *shape* of the objects LEP_prediction_and_repair_v2.py already
produces:

    posterior_table  - n x n list of dicts, posterior_table[i][j] = {a: p}
                        (exactly compute_posterior_table's return value)
    pi               - list of length n, pi[i] = the column Hungarian-
                        assigned to row i (exactly monomial_approximation's
                        4th return value)

Two independent pieces are provided:

1. active_row_error_probabilities(...) - the "local approximation" from the
   ChatGPT conversation: for each active row i, estimates
       Pr[Q_hat_i != Q_i | L]
   directly from posterior_table[i], without needing the K-best-assignment
   machinery (computing that exactly would need Murty's algorithm for
   K-best perfect matchings, which is NOT implemented here - see the note
   in the module docstring below the functions). This is the "if even that
   is too costly" fallback ChatGPT gave, and it is exactly computable from
   quantities compute_row_scores() already derives (see the derivation in
   _column_log_terms below).

2. poisson_binomial_pmf(...) / posterior_repair_radius(...) - the
   PosteriorRepairRadius algorithm exactly as specified in the
   conversation: a Poisson-binomial DP (O(t^2) time, O(t) memory) that
   aggregates a list of (possibly all-different) per-row error
   probabilities into a repair-radius quantile.

Relationship to rmax_approximation.md: that document computes a single
p_err from (n, q, alpha, beta) alone, valid *before* any leakage is
observed, and aggregates t i.i.d. active rows via a plain Binomial. This
module instead computes one p_i per active row *after* a specific leakage
L has been observed (using the actual posterior table for that instance),
and aggregates via the more general Poisson-binomial. The two approaches
answer different questions - see the accompanying experiment script and
the LaTeX section this was written to support.
"""
import math


# ---------------------------------------------------------------------------
# Part 1: per-row error probability from the posterior table
#   ("local approximation" from the ChatGPT conversation)
# ---------------------------------------------------------------------------

def _logsumexp(values):
    """
    Numerically stable log(sum(exp(v) for v in values)).

    :param values: an iterable of real numbers (may be empty)
    :return: log(sum(exp(v) for v in values)); -inf if values is empty
    """
    values = list(values)
    if not values:
        return float('-inf')
    m = max(values)
    if m == float('-inf'):
        return float('-inf')
    return m + math.log(sum(math.exp(v - m) for v in values))


def _column_log_terms(col_dist, floor=1e-12):
    """
    From one entry's posterior distribution col_dist = posterior_table[i][k]
    = {a: p_i,k(a)}, computes the two log-quantities needed by
    active_row_error_probabilities, both expressed *relative to* the row
    constant Z_i = sum_l log p_i,l(0) (Section 4.3.3 of the paper / the
    "row_score" derivation in LEP_prediction_and_repair_v2.py), so that Z_i
    itself never needs to be computed - it cancels in the final ratio.

    Recall (Ricardo's ChatGPT thread): R_i(k, a) = p_i,k(a) * prod_{l!=k}
    p_i,l(0) = p_i,k(a) * exp(Z_i) / p_i,k(0), and W_i,k = sum_{a!=0}
    R_i(k, a) = exp(Z_i) * (1 - p_i,k(0)) / p_i,k(0). So:

        log_num = log p_i,k(a_best) - log p_i,k(0)   [ = log R_i(k,a_best) - Z_i ]
        log_den = log(1 - p_i,k(0)) - log p_i,k(0)   [ = log W_i,k - Z_i ]

    where a_best = argmax_{a != 0} p_i,k(a) (matching D_loc[i][k] in
    LEP_prediction_and_repair_v2.compute_row_scores).

    :param col_dist: dict {a: p(a)}, keys may be plain ints/floats or Sage
        field elements (only compared with `== 0`, never hashed against 0
        directly, so either works)
    :param floor: numerical floor used in every log(), matching the
        LOG_ZERO_FLOOR convention in LEP_prediction_and_repair_v2.py
    :return: a tuple (log_num, log_den)
    """
    p0 = 0.0
    best_p = 0.0
    nonzero_mass = 0.0
    for a, p in col_dist.items():
        p = float(p)
        if a == 0:
            p0 = p
        else:
            nonzero_mass += p
            if p > best_p:
                best_p = p

    log_p0 = math.log(max(p0, floor))
    log_num = math.log(max(best_p, floor)) - log_p0
    log_den = math.log(max(nonzero_mass, floor)) - log_p0
    return log_num, log_den


def active_row_error_probabilities(posterior_table, pi, active_rows, floor=1e-12):
    """
    The "local approximation" column/row-error probability from the
    ChatGPT conversation:

        Pr[Q_hat_i != Q_i | L] ~= 1 - R_i(pi[i], a_hat) / sum_k W_i,k

    computed independently for every active row (i.e. ignoring competition
    between different rows for the same column - the caveat ChatGPT itself
    flagged: "it ignores competition between different rows for the same
    column"). This is the cheap, always-available estimate; the more
    accurate K-best-assignment / weighted-quantile version described in
    the same conversation is NOT implemented here (see the module
    docstring) since it needs a K-best perfect-matching routine (e.g.
    Murty's algorithm) that does not currently exist in this codebase.

    :param posterior_table: the n x n posterior table from
        compute_posterior_table (LEP_prediction_and_repair_v2.py)
    :param pi: the length-n assignment from monomial_approximation
        (pi[i] = the column Hungarian-assigned to row i)
    :param active_rows: the active support A = Supp(v) (e.g. from
        LEP_prediction_and_repair_v2.support(v)) - only these rows are
        scored, since only they matter for r(v)
    :param floor: numerical floor forwarded to _column_log_terms
    :return: a dict {i: p_i for i in active_rows}, each p_i in [0, 1]
    """
    n = len(posterior_table)
    p_err = {}

    for i in active_rows:
        row = posterior_table[i]
        j_star = pi[i]

        log_dens = []
        log_num_at_j = None
        for k in range(n):
            log_num_k, log_den_k = _column_log_terms(row[k], floor)
            log_dens.append(log_den_k)
            if k == j_star:
                log_num_at_j = log_num_k

        if log_num_at_j is None:
            raise ValueError(f"pi[{i}] = {j_star} is not a valid column index (0..{n - 1})")

        log_Z = _logsumexp(log_dens)
        log_p_correct = log_num_at_j - log_Z
        # Clip at 0 in log-space (i.e. p_correct <= 1): floating-point
        # slack in the floor/argmax computation above can very occasionally
        # push this a hair above 0 even though it never should be in exact
        # arithmetic.
        p_correct = math.exp(min(log_p_correct, 0.0))
        p_err[i] = min(max(1.0 - p_correct, 0.0), 1.0)

    return p_err


# ---------------------------------------------------------------------------
# Part 2: PosteriorRepairRadius (the Poisson-binomial DP, verbatim from the
#   ChatGPT conversation)
# ---------------------------------------------------------------------------

def poisson_binomial_pmf(p_list):
    """
    Computes the exact probability mass function of

        r = sum_i X_i,   X_i ~ Bernoulli(p_i) independent,

    via the standard O(t^2)-time, O(t)-memory dynamic program (t = len(p_list)):
    D[r] is updated in place, iterating r from the current count down to 0
    so that D[r-1] (needed for D[r]'s update) hasn't been overwritten yet
    on this pass - this is exactly the algorithm given in the ChatGPT
    conversation.

    :param p_list: a list of per-row error probabilities p_i in [0, 1]
        (order doesn't matter - the sum is symmetric in the p_i)
    :return: a list D of length len(p_list) + 1, D[r] = Pr[r(v) = r]
    """
    t = len(p_list)
    D = [0.0] * (t + 1)
    D[0] = 1.0

    m = 0
    for p in p_list:
        m += 1
        for r in range(m, -1, -1):
            if r == 0:
                D[0] = (1.0 - p) * D[0]
            else:
                D[r] = (1.0 - p) * D[r] + p * D[r - 1]

    return D


def repair_radius_from_pmf(D, delta=0.05):
    """
    Reads the smallest r such that Pr[r(v) <= r] >= 1 - delta off an
    already-computed Poisson-binomial PMF.

    :param D: a PMF as returned by poisson_binomial_pmf
    :param delta: target failure probability (e.g. 0.05 for 95% confidence)
    :return: the smallest r in {0, ..., len(D) - 1} with cumulative mass
        >= 1 - delta (returns len(D) - 1, i.e. t, if even the full sum
        falls short by floating-point slack)
    """
    cumulative = 0.0
    target = 1.0 - delta
    for r, d in enumerate(D):
        cumulative += d
        if cumulative >= target:
            return r
    return len(D) - 1


def posterior_repair_radius(p_list, delta=0.05):
    """
    Algorithm PosteriorRepairRadius (the full pipeline: DP + quantile
    read-off), matching the ChatGPT conversation's pseudocode 1:1.

    :param p_list: per-active-row error probabilities (e.g. the values of
        active_row_error_probabilities(...), in any order)
    :param delta: target failure probability
    :return: a tuple (R_max, D) - the repair radius and the full PMF, so
        callers can also read off e.g. the mean (sum(p_list)) or other
        quantiles without recomputing the DP
    """
    D = poisson_binomial_pmf(p_list)
    R_max = repair_radius_from_pmf(D, delta)
    return R_max, D


# ---------------------------------------------------------------------------
# Self-test: run "python3 posterior_repair_radius.py" (no Sage needed) to
# sanity-check the Poisson-binomial DP and active_row_error_probabilities
# in isolation, before trusting them inside the full Sage-based pipeline.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random as _random

    print("=== Test 1: Poisson-binomial reduces to Binomial when all p_i are equal ===")
    try:
        from scipy.stats import binom as _scipy_binom
        have_scipy = True
    except ImportError:
        have_scipy = False
        print("  (scipy not available, will only check internal consistency)")

    for t, p in [(5, 0.2), (10, 0.5), (30, 0.05), (49, 0.203656)]:
        D = poisson_binomial_pmf([p] * t)
        assert abs(sum(D) - 1.0) < 1e-9, f"PMF does not sum to 1 for t={t}, p={p}: sum={sum(D)}"
        if have_scipy:
            for r in range(t + 1):
                expected = _scipy_binom.pmf(r, t, p)
                assert abs(D[r] - expected) < 1e-9, (
                    f"mismatch at t={t}, p={p}, r={r}: got {D[r]}, expected {expected}"
                )
        print(f"  t={t:3d} p={p:.6f}  sum(D)={sum(D):.10f}  mean(D)={sum(r * d for r, d in enumerate(D)):.4f}"
              f"  (expected mean = t*p = {t * p:.4f})  OK")

    print("\n=== Test 2: heterogeneous p_i, mean and quantile sanity checks ===")
    _random.seed(0)
    for trial in range(5):
        t = _random.randint(5, 40)
        p_list = [_random.uniform(0.01, 0.4) for _ in range(t)]
        D = poisson_binomial_pmf(p_list)
        assert abs(sum(D) - 1.0) < 1e-9
        mean_D = sum(r * d for r, d in enumerate(D))
        mean_expected = sum(p_list)
        assert abs(mean_D - mean_expected) < 1e-6, (mean_D, mean_expected)

        for delta in (0.10, 0.05, 0.01):
            R_max, D2 = posterior_repair_radius(p_list, delta=delta)
            assert D2 is D or D2 == D
            cum = sum(D[:R_max + 1])
            assert cum >= 1 - delta - 1e-9, f"quantile violated: cum={cum} < 1-delta={1 - delta}"
            if R_max > 0:
                cum_prev = sum(D[:R_max])
                assert cum_prev < 1 - delta + 1e-9, "R_max is not the *smallest* r satisfying the quantile"
        print(f"  trial {trial}: t={t:3d} mean(p)={mean_expected / t:.3f} "
              f"E[r]={mean_expected:.2f} "
              f"R_max(90/95/99%)="
              f"{posterior_repair_radius(p_list, 0.10)[0]}/"
              f"{posterior_repair_radius(p_list, 0.05)[0]}/"
              f"{posterior_repair_radius(p_list, 0.01)[0]}  OK")

    print("\n=== Test 3: active_row_error_probabilities on a small synthetic posterior table ===")
    # Build a tiny fake posterior_table by hand (n=4, no Sage needed - keys
    # are plain ints, exactly like Sage GF(q) elements compare with ==).
    n = 4
    # Row 0: very confident, correctly points at column 0 with value 3.
    row0 = [
        {0: 0.02, 1: 0.02, 2: 0.02, 3: 0.94},  # column 0: true entry, high confidence
        {0: 0.9, 1: 0.05, 2: 0.03, 3: 0.02},
        {0: 0.9, 1: 0.05, 2: 0.03, 3: 0.02},
        {0: 0.9, 1: 0.05, 2: 0.03, 3: 0.02},
    ]
    # Row 1: ambiguous between column 1 and column 2, both plausible.
    row1 = [
        {0: 0.9, 1: 0.05, 2: 0.03, 3: 0.02},
        {0: 0.55, 1: 0.4, 2: 0.03, 3: 0.02},  # column 1: assigned here
        {0: 0.5, 1: 0.05, 2: 0.4, 3: 0.05},   # column 2: close competitor
        {0: 0.9, 1: 0.05, 2: 0.03, 3: 0.02},
    ]
    posterior_table = [row0, row1, row0, row0]
    pi = [0, 1, 2, 3]
    p_err = active_row_error_probabilities(posterior_table, pi, active_rows=[0, 1])
    print(f"  p_err[0] (confident row) = {p_err[0]:.4f}  (expect small)")
    print(f"  p_err[1] (ambiguous row) = {p_err[1]:.4f}  (expect noticeably larger)")
    assert p_err[0] < p_err[1], "the ambiguous row should get a higher error probability than the confident one"
    assert 0.0 <= p_err[0] <= 1.0 and 0.0 <= p_err[1] <= 1.0

    print("\nAll self-tests passed.")
