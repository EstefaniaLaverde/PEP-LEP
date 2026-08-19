"""
LEP_prediction_and_repair_v2.py  (Code/core/ - shared library, imported by most strategies/)

File implementing the posterior-guided prediction-and-repair framework for
the Linear Code Equivalence Problem (LEP), under the Bitwise Bayesian
Leakage Model (BBLM). This is the current, actively-used implementation -
archive/lep_prediction_and_repair_v1_superseded.py is an earlier version
kept only for history.

This module builds directly on top of instances_generator.py: it consumes
the noisy hint Q_noisy produced by
generate_noisy_LCE_instance_CBA_bit_flip_version (where alpha is the
probability that a 1-bit flips to 0 and beta is the probability that a
0-bit flips to 1, applied to the binary representation of every entry) and
turns it into (a) an entrywise posterior table, (b) a monomial approximation
Q_hat of the secret Q, and (c) prediction-and-repair procedures that use
Q_hat to recover equivalent codeword pairs (v, w) with v in C, w in C'.

Implements Algorithms 1-12:
    - Algorithm 1  : PredictionAndRepairFramework
    - Algorithm 2  : RowScore
    - Algorithm 3  : ScoreToCost
    - Algorithm 4  : HungarianAssignment
    - Algorithm 5  : MonomialApproximation
    - Algorithm 6  : InducedSDRepair
    - Algorithm 7  : BuildActiveRowLists
    - Algorithm 8  : StructuredSDRepair
    - Algorithm 9  : MarginalErrorRegion
    - Algorithm 10 : AssignmentConsistentErrorRegion (simplified)
    - Algorithm 11 : MinimumOverlapSampler
    - Algorithm 12 : PosteriorAwarePrangeRepair

Using SageMath.
"""
from sage.all import GF, matrix, vector, log as sage_log, RR, random, randint
from scipy.optimize import linear_sum_assignment
import itertools
import copy
import time

from instances_generator import obtain_parity_check_matrix

# A very small floor used instead of 0 inside logarithms, so that a zero
# posterior probability becomes a large negative (but finite) log-score
# rather than -infinity. This keeps every score matrix numerically usable.
LOG_ZERO_FLOOR = 1e-12

# ---------------------------------------------------------------------------
# Verbosity control: module-level switch for the progress prints scattered
# through this file. Off by default (these functions are also used as a
# library); call set_verbose(True) - e.g. from a driver script like
# run_lep_experiments.py - to see per-stage progress while a run is in
# flight, which is useful to tell a slow step apart from a hung one.
# ---------------------------------------------------------------------------
VERBOSE = False


def set_verbose(value):
    """
    Enables or disables the progress prints (_vprint calls) in this module.

    :param value: True to enable prints, False to disable them
    """
    global VERBOSE
    VERBOSE = value


def _vprint(*args, **kwargs):
    """
    Prints only when VERBOSE is enabled (see set_verbose). Forwards all
    arguments to the builtin print, including flush.

    """
    if VERBOSE:
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Section 2.3 / Section 4.1: bitwise posterior construction (BBLM)
# ---------------------------------------------------------------------------
def _bits_msb_first(x, r):
    """
    Writes the integer x using r bits, most-significant bit first, matching
    the encoding used by generate_bit_channel_hint.

    :param x: a nonnegative integer, 0 <= x < 2**r
    :param r: the number of bits to use
    :return: a list of r bits (0/1), MSB first
    """
    # the code >> (r - 1 - t) & 1 extracts the bit at position t from the left
    # (most significant bit first)
    return [(x >> (r - 1 - t)) & 1 for t in range(r)]


def _bit_transition_prob(bx, by, alpha, beta):
    """
    Computes the probability that a single true bit bx is observed as by,
    under the asymmetric channel used by generate_bit_channel_hint: a 1-bit
    flips to 0 with probability alpha, and a 0-bit flips to 1 with
    probability beta.

    :param bx: the true bit (0 or 1)
    :param by: the observed bit (0 or 1)
    :param alpha: probability that a 1 bit flips to 0
    :param beta: probability that a 0 bit flips to 1
    :return: Pr[by | bx]
    """
    if bx == 1:
        return alpha if by == 0 else (1 - alpha)
    return beta if by == 1 else (1 - beta)


def build_bit_channel_matrix(q, alpha, beta):
    """
    Precomputes the exact field-level channel Pr[h | x] for every true value
    x and observed hint value h in [0, q - 1], induced by the bitwise
    channel of generate_bit_channel_hint followed by reduction modulo q.
    Since alpha and beta are the same at every matrix position, this table
    only needs to be built once per instance and reused for every entry.

    :param q: size of the (prime) finite field
    :param alpha: probability that a 1 bit flips to 0
    :param beta: probability that a 0 bit flips to 1
    :return: a q x q list of lists, channel[x][h] = Pr[h | x]
    """
    # compute the number of bits needed to represent q-1
    bits_needed = 0
    while (1 << bits_needed) < q:
        bits_needed += 1
    l = 1 << bits_needed

    # save the channel probabilities in a q x q matrix, where channel[x][h] = Pr[h | x]
    # this is basically a table of the probabilities of x being the real value when h is observed
    channel = [[0.0 for _ in range(q)] for _ in range(q)]
    for x in range(q):
        bx = _bits_msb_first(x, bits_needed)
        for y in range(l):
            by = _bits_msb_first(y, bits_needed)

            # Probability that the bitwise channel turns x's bits into y's.
            p_y_given_x = 1.0
            for bxi, byi in zip(bx, by):
                p_y_given_x *= _bit_transition_prob(bxi, byi, alpha, beta)

            h = y % q
            channel[x][h] += p_y_given_x

    return channel


def _row_prior(F, n, is_permutation):
    """
    Computes the entrywise prior Pr[Q_ij = a] induced by the row structure
    of a random monomial matrix: exactly one of the n entries in a row is
    nonzero, chosen uniformly among the n columns, and, for LEP, its value
    is uniform over the field. This is the prior used to turn the leakage
    channel into an entrywise posterior (Section 2.3).

    :param F: the base field GF(q)
    :param n: the size of the monomial matrix
    :param is_permutation: if True, the nonzero value is always 1 (PEP)
    :return: a dict {a: Pr[Q_ij = a]}
    """
    # save the prior probabilities in a dict, where prior[a] = Pr[Q_ij = a]
    p_nonzero_here = 1 / n # this is the probability that the non-zero entry is in this column
    p_zero_here = 1 - p_nonzero_here # the probability that the non-zero entry is not in this column

    # if the matrix is a permutation matrix, then the non-zero entry is always 1, 
    # so we can return a dict with only two entries: 0 and 1
    if is_permutation:
        return {F(0): p_zero_here, F(1): p_nonzero_here}

    q = F.cardinality()
    prior = {F(0): p_zero_here}
    for a in F:
        if a != F(0):
            # the prior probability of a non-zero entry is the probability that the non-zero entry 
            # is in this column, divided by the number of non-zero elements in the field
            prior[a] = p_nonzero_here / (q - 1)
    return prior


def compute_posterior_table(hint, alpha, beta, is_permutation=False):
    """
    Builds the entrywise BBLM posterior table p_ij(a) = Pr[Q_ij = a | hint]
    from a hint matrix produced by generate_noisy_LCE_instance_CBA_bit_flip_version,
    using Bayes' rule with the exact bitwise channel of build_bit_channel_matrix
    and the row prior of _row_prior.

    :param hint: the noisy hint matrix, over GF(q) (output of
        generate_noisy_LCE_instance_CBA_bit_flip_version)
    :param alpha: probability that a 1 bit flips to 0
    :param beta: probability that a 0 bit flips to 1
    :param is_permutation: whether hint encodes a PEP secret (values in {0,1})
    :return: an n x n list of dicts, table[i][j] = {a: p_ij(a) for a in Fq}
    """
    F = hint.base_ring()
    q = F.cardinality()
    n = hint.nrows()

    _vprint(f"      [posterior] building n={n} x n={n} table (q={q})...", end='', flush=True)
    t0 = time.time()

    # compute the prior probabilitites for each entry of the matrix
    prior = _row_prior(F, n, is_permutation)
    channel = build_bit_channel_matrix(q, alpha, beta)

    table = [[None for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            h_obs = int(hint[i, j])
            unnormalized = {}
            for a in F:
                x = int(a)
                unnormalized[a] = prior[a] * channel[x][h_obs]

            total = sum(unnormalized.values())
            if total == 0:
                # Degenerate case (should not happen for alpha, beta < 1):
                # fall back to the prior.
                table[i][j] = prior
            else:
                table[i][j] = {a: unnormalized[a] / total for a in F}
        if VERBOSE and n >= 32 and (i + 1) % max(1, n // 10) == 0:
            _vprint(f" row {i + 1}/{n}", end='', flush=True)

    _vprint(f" done ({time.time() - t0:.2f}s)", flush=True)
    return table


# ---------------------------------------------------------------------------
# Section 4.3: local row optimization  (Algorithm 2: RowScore)
# ---------------------------------------------------------------------------

def _safe_log(p):
    """
    Computes log(p), clipping p away from 0 so the result stays finite.

    :param p: a probability in [0, 1]
    :return: log(max(p, LOG_ZERO_FLOOR)) as a real number
    """
    return sage_log(RR(max(p, LOG_ZERO_FLOOR)))


def row_score(i, j, posterior_row, F):
    """
    Algorithm 2 (RowScore). Computes the optimal nonzero value d_hat_ij and
    the local posterior score s_ij for the hypothesis "row i is assigned to
    column j", following Definition 1 / Section 4.3.2.

    :param i: row index (unused beyond bookkeeping, kept for signature clarity)
    :param j: candidate column index
    :param posterior_row: list of n dicts, posterior_row[l] = p_i,l(.), the
        posterior distributions for every entry of row i
    :param F: the base field GF(q)
    :return: a tuple (d_hat_ij, s_ij)
    """
    # Optimal nonzero value at position (i, j): maximize p_ij(a) over a != 0.
    p_ij = posterior_row[j]
    d_hat = max((a for a in F if a != F(0)), key=lambda a: p_ij.get(a, 0))
    best = _safe_log(p_ij.get(d_hat, 0))

    # Zero-entry contribution: every other column l != j must be zero.
    z_ij = sum(_safe_log(posterior_row[l].get(F(0), 0)) for l in range(len(posterior_row)) if l != j)

    s_ij = best + z_ij
    return d_hat, s_ij


def compute_row_scores(posterior_table, F):
    """
    Computes the full local-score matrix S and value matrix D_loc, using the
    efficient form s_ij = log p_ij(d_hat_ij) + Z_i - log p_ij(0), where
    Z_i = sum_l log p_i,l(0) is precomputed once per row (Section 4.3.3).

    :param posterior_table: the n x n posterior table from compute_posterior_table
    :param F: the base field GF(q)
    :return: a tuple (S, D_loc), each an n x n list of lists
    """
    n = len(posterior_table)
    S = [[0 for _ in range(n)] for _ in range(n)]
    D_loc = [[None for _ in range(n)] for _ in range(n)]

    t0 = time.time()
    _vprint(f"      [row_scores] scoring n={n} rows...", end='', flush=True)

    for i in range(n):
        # Precompute the row's total zero-score once (Section 4.3.3).
        Z_i = sum(_safe_log(posterior_table[i][l].get(F(0), 0)) for l in range(n))

        for j in range(n):
            p_ij = posterior_table[i][j]
            d_hat = max((a for a in F if a != F(0)), key=lambda a: p_ij.get(a, 0))
            best = _safe_log(p_ij.get(d_hat, 0))
            log_zero_ij = _safe_log(p_ij.get(F(0), 0))

            D_loc[i][j] = d_hat
            S[i][j] = best + Z_i - log_zero_ij

        if VERBOSE and n >= 32 and (i + 1) % max(1, n // 10) == 0:
            _vprint(f" row {i + 1}/{n}", end='', flush=True)

    _vprint(f" done ({time.time() - t0:.2f}s)", flush=True)
    return S, D_loc


# ---------------------------------------------------------------------------
# Section 4.4: global monomial optimization
# (Algorithm 3: ScoreToCost, Algorithm 4: HungarianAssignment)
# ---------------------------------------------------------------------------

def score_to_cost(S):
    """
    Algorithm 3 (ScoreToCost). Converts a score matrix (to be maximized)
    into a nonnegative cost matrix (to be minimized), by subtracting every
    entry from the global maximum score.

    :param S: an n x n score matrix (list of lists)
    :return: an n x n cost matrix (list of lists), entries >= 0
    """
    n = len(S)
    M = max(S[i][j] for i in range(n) for j in range(n))
    return [[float(M - S[i][j]) for j in range(n)] for i in range(n)]


def hungarian_assignment(S):
    """
    Algorithm 4 (HungarianAssignment). Finds the permutation pi maximizing
    sum_i s_i,pi(i) by reducing to a minimum-cost assignment problem and
    solving it with the Hungarian algorithm (here, scipy's implementation).

    :param S: an n x n score matrix (list of lists)
    :return: a list pi of length n, where pi[i] is the column assigned to row i
    """
    t0 = time.time()
    _vprint(f"      [hungarian] solving {len(S)}x{len(S)} assignment...", end='', flush=True)
    C = score_to_cost(S)
    row_ind, col_ind = linear_sum_assignment(C)
    # linear_sum_assignment already returns row_ind sorted 0..n-1.
    pi = [None] * len(S)
    for i, j in zip(row_ind, col_ind):
        pi[i] = j
    _vprint(f" done ({time.time() - t0:.2f}s)", flush=True)
    return pi


# ---------------------------------------------------------------------------
# Section 4.5: construction of the monomial approximation
# (Algorithm 5: MonomialApproximation)
# ---------------------------------------------------------------------------

def monomial_approximation(posterior_table, F):
    """
    Algorithm 5 (MonomialApproximation). Combines the local row optimization
    and the global assignment step to build the monomial MAP estimate
    Q_hat of the secret monomial matrix Q, under the entrywise posterior
    approximation of Section 4.1.

    :param posterior_table: the n x n posterior table from compute_posterior_table
    :param F: the base field GF(q)
    :return: a tuple (Q_hat, S, D_loc, pi), where Q_hat is the n x n monomial
        matrix, S/D_loc are the local score/value matrices (reused later by
        the repair modules), and pi is the selected permutation
    """
    n = len(posterior_table)
    _vprint(f"    [monomial_approximation] n={n}")

    # Local stage: score every row-column hypothesis.
    S, D_loc = compute_row_scores(posterior_table, F)

    # Global stage: pick the best perfect matching of rows to columns.
    pi = hungarian_assignment(S)

    # Assemble the monomial matrix from the selected assignment.
    Q_hat = matrix(F, n, n)
    for i in range(n):
        Q_hat[i, pi[i]] = D_loc[i][pi[i]]

    return Q_hat, S, D_loc, pi


# ---------------------------------------------------------------------------
# Section 3 / 5.1: prediction step and active-row error analysis
# ---------------------------------------------------------------------------

def predict_image(v, Q_hat):
    """
    Computes the predicted image w_tilde = v * Q_hat of a codeword v under
    the monomial approximation (the fixed prediction step of Algorithm 1).

    :param v: a row vector v in C
    :param Q_hat: the monomial approximation of the secret Q
    :return: the predicted image w_tilde
    """
    return v * Q_hat


def support(v):
    """
    Computes the support A = Supp(v) = { i : v_i != 0 } of a row vector v.

    :param v: a row vector
    :return: a list of the nonzero coordinate indices of v
    """
    return [i for i in range(len(v)) if v[i] != 0]


def active_error_set(v, Q_true, Q_hat):
    """
    Computes the active error set E(v) = { i in Supp(v) : Q_hat_i != Q_i }
    (Definition 2). This uses the true secret Q and is intended for testing
    and analysis only; it is not available to the actual attacker.

    :param v: a row vector v in C
    :param Q_true: the true secret monomial matrix Q
    :param Q_hat: the monomial approximation Q_hat
    :return: the list E(v) of active, incorrectly approximated row indices
    """
    A = support(v)
    return [i for i in A if list(Q_hat[i]) != list(Q_true[i])]


# ---------------------------------------------------------------------------
# Section 5.2: induced syndrome decoding  (Algorithm 6: InducedSDRepair)
# ---------------------------------------------------------------------------

def prange_isd(H, s, tau, max_trials, F):
    """
    A generic Prange information-set-decoding routine, used as the plain
    SDDecode black box inside InducedSDRepair. Repeatedly samples an
    information set I, and whenever the complementary columns J = [n] \\ I
    give an invertible submatrix H_J, solves H_J e_J^T = s and accepts the
    solution if it has weight at most tau. (The Hamming-weight-of-w
    consistency check described in induced_sd_repair is applied by the
    caller after this returns, since this routine only sees e, not w_tilde.)

    :param H: an (n-k) x n parity-check matrix
    :param s: the target syndrome (as a vector over F)
    :param tau: the maximum tolerated Hamming weight of the error
    :param max_trials: the maximum number of information sets to try
    :param F: the base field GF(q)
    :return: an error vector e of weight <= tau with H*e^T = s, or None on failure
    """
    n = H.ncols()
    r = H.nrows()

    for _ in range(max_trials):
        # Sample a candidate set J of r columns to solve on.
        J = list(range(n))
        # Fisher-Yates-style partial shuffle to pick r columns uniformly.
        for idx in range(r):
            swap = randint(idx, n - 1)
            J[idx], J[swap] = J[swap], J[idx]
        J = J[:r]

        H_J = H.matrix_from_columns(J)
        if H_J.rank() < r:
            continue

        e_J = H_J.solve_right(s)

        e = vector(F, n)
        for pos, col in enumerate(J):
            e[col] = e_J[pos]

        if e.hamming_weight() <= tau:
            return e

    return None


def induced_sd_repair(w_tilde, v, H_prime, tau, max_trials=2000, decoder=None):
    """
    Algorithm 6 (InducedSDRepair). Attempts to repair a predicted word
    w_tilde into a codeword of C' by solving the induced syndrome decoding
    instance H' * e^T = H' * w_tilde^T for an error e of weight <= tau.

    Accepts a candidate repair only if it also preserves the Hamming weight
    of v: monomial transformations satisfy wt(vQ) = wt(v) exactly
    (Section 2.1), so a repaired word with a different weight cannot be the
    true image of v, regardless of whether its syndrome happens to match.
    Generic syndrome decoding (unlike structured_sd_repair) has no way to
    retry a different error of the same weight class, so on a weight
    mismatch this simply reports Failure rather than returning a
    provably-wrong word.

    :param w_tilde: the predicted word, w_tilde = v * Q_hat
    :param v: the codeword v in C that produced w_tilde, used only to check
        the weight-preservation property above
    :param H_prime: a parity-check matrix of C'
    :param tau: the target decoding radius (Corollary 1 suggests tau = 2*rmax)
    :param max_trials: the trial budget passed to the underlying decoder
    :param decoder: an optional SDDecode callable H, s, tau, max_trials, F -> e;
        defaults to prange_isd
    :return: the repaired word w in C', or None on Failure
    """
    F = H_prime.base_ring()
    target_weight = v.hamming_weight()
    s = H_prime * w_tilde

    if s.is_zero():
        return w_tilde if w_tilde.hamming_weight() == target_weight else None

    if decoder is None:
        decoder = prange_isd

    e = decoder(H_prime, s, tau, max_trials, F)
    if e is None:
        return None

    w_candidate = w_tilde - e
    if w_candidate.hamming_weight() != target_weight:
        return None

    return w_candidate


# ---------------------------------------------------------------------------
# Section 5.3: structured syndrome decoding from active rows
# (Algorithm 7: BuildActiveRowLists, Algorithm 8: StructuredSDRepair)
# ---------------------------------------------------------------------------

def build_active_row_lists(v, S, D_loc, F, budgets):
    """
    Algorithm 7 (BuildActiveRowLists). For every active row i in Supp(v),
    builds a candidate list L_i of the highest-scoring monomial-row
    replacement hypotheses R_i(j, a), reusing the local score matrix S and
    value matrix D_loc already computed by monomial_approximation.

    :param v: a row vector v in C
    :param S: the local score matrix from monomial_approximation
    :param D_loc: the local value matrix from monomial_approximation
    :param F: the base field GF(q)
    :param budgets: a dict {i: N_i} giving the candidate budget per active row
        (a single int may be passed instead to use the same budget everywhere)
    :return: a tuple (A, L), where A = Supp(v) and L is a dict
        {i: [(row_vector, score), ...]} sorted by decreasing score
    """
    n = len(S)
    A = support(v)

    # Accept either a per-row dict {i: N_i} or a single scalar budget applied
    # to every active row. IMPORTANT: this must check "not a dict" (i.e.
    # "is it a scalar?"), not "is a dict" - flipping this condition either
    # re-wraps an already-correct dict, or (worse) leaves a scalar budget
    # unconverted, and the .get(...) call below then crashes because a bare
    # number (e.g. Sage's Integer(3)) has no .get method.
    if not isinstance(budgets, dict):
        budgets = {i: budgets for i in A}

    L = {}
    for i in A:
        # Rank every column hypothesis j for row i by its local score s_ij.
        candidates = sorted(range(n), key=lambda j: S[i][j], reverse=True)
        N_i = budgets.get(i, n)

        L_i = []
        for j in candidates[:N_i]:
            row = vector(F, n)
            row[j] = D_loc[i][j]
            L_i.append((row, S[i][j]))
        L[i] = L_i

    _vprint(f"        [active_rows] |A|={len(A)} budgets={[len(L[i]) for i in A]}", flush=True)
    return A, L


def _hypothesis_generator(A, L, rmax):
    """
    A complete generator over H_{<=rmax}: enumerates hypotheses (T, (R_i))
    by increasing repair dimension r = 0, 1, ..., rmax, all subsets T of A
    with |T| = r, and all choices of replacement rows from the lists L_i
    (Section 5.3.3). r = 0 is skipped here, since the calling algorithm
    handles the empty-repair case separately via the s = 0 test.

    :param A: the active support Supp(v)
    :param L: the candidate row lists, as returned by build_active_row_lists
    :param rmax: the maximum repair dimension to explore
    :return: a generator yielding (T, {i: row_vector}) hypotheses
    """
    for r in range(1, rmax + 1):
        for T in itertools.combinations(A, r):
            # All choices of one candidate row per index in T.
            choice_lists = [L[i] for i in T]
            for combo in itertools.product(*choice_lists):
                rows = {i: combo[k][0] for k, i in enumerate(T)}
                yield T, rows


def structured_sd_repair(w_tilde, v, H_prime, Q_hat, A, L, rmax):
    """
    Algorithm 8 (StructuredSDRepair). Repairs a predicted word by searching
    over hypotheses for the erroneous active rows of Q_hat, rather than
    treating the prediction error as an arbitrary sparse vector. Accepts
    the first hypothesis whose induced syndrome matches the true syndrome
    AND whose resulting word has the same Hamming weight as v.

    The weight check is not a heuristic: monomial transformations preserve
    Hamming weight exactly (wt(vQ) = wt(v), Section 2.1), so any candidate
    w with wt(w) != wt(v) is provably not the true image of v, no matter
    how its syndrome happens to compare. Without this check, a hypothesis
    can still be accepted purely by coincidence whenever the true active
    error dimension exceeds rmax (or the true replacement row is missing
    from the candidate lists), since the syndrome space H' lives in can be
    small enough for accidental collisions.

    :param w_tilde: the predicted word, w_tilde = v * Q_hat
    :param v: the codeword v in C that produced w_tilde
    :param H_prime: a parity-check matrix of C'
    :param Q_hat: the monomial approximation of the secret Q
    :param A: the active support Supp(v), from build_active_row_lists
    :param L: the candidate row lists, from build_active_row_lists
    :param rmax: the maximum repair dimension to explore
    :return: the repaired word w in C', or None on Failure
    """
    target_weight = v.hamming_weight()

    s = H_prime * w_tilde
    if s.is_zero():
        # w_tilde already lies in C'; this can only be the true image,
        # since it requires no correction at all (E(v) is empty).
        return w_tilde if w_tilde.hamming_weight() == target_weight else None

    # Precompute, for every active row i and candidate R, the syndrome
    # contribution sigma(i, R) = H' * (v_i * (Q_hat_i - R))^T.
    sigma = {}
    for i in A:
        for R, _score in L[i]:
            diff = v[i] * (Q_hat[i] - R)
            sigma[(i, tuple(R))] = H_prime * diff

    # The search space size grows fast with rmax and the per-row budgets
    # (sum_{r=1..rmax} C(|A|, r) * budget^r), so this is the most likely
    # place a run looks "stuck" - report the size and a periodic heartbeat.
    if VERBOSE:
        budget_sizes = [len(L[i]) for i in A]
        avg_budget = sum(budget_sizes) / len(budget_sizes) if budget_sizes else 0
        est_total = sum(
            len(list(itertools.combinations(A, r))) * (avg_budget ** r)
            for r in range(1, rmax + 1)
        )
        _vprint(
            f"        [structured_sd_repair] |A|={len(A)} rmax={rmax} "
            f"est_hypotheses~={est_total:.3g}", flush=True,
        )

    t0 = time.time()
    t_last = t0
    tried = 0
    for T, rows in _hypothesis_generator(A, L, rmax):
        tried += 1
        # Sum the syndrome contributions of the hypothesis' repaired rows.
        s_h = sum((sigma[(i, tuple(rows[i]))] for i in T))

        if s_h == s:
            e_h = sum((v[i] * (Q_hat[i] - rows[i]) for i in T))
            w_candidate = w_tilde - e_h

            # Reject syndrome-only coincidences: a genuine equivalent pair
            # must preserve the Hamming weight of v exactly.
            if w_candidate.hamming_weight() == target_weight:
                _vprint(
                    f"        [structured_sd_repair] found match after "
                    f"{tried} hypotheses ({time.time() - t0:.2f}s)", flush=True,
                )
                return w_candidate
            # Otherwise, keep searching other hypotheses instead of
            # returning a provably-wrong repair.

        now = time.time()
        if VERBOSE and now - t_last >= 5.0:
            _vprint(
                f"        [structured_sd_repair] tried {tried} hypotheses, "
                f"elapsed={now - t0:.1f}s", flush=True,
            )
            t_last = now

    _vprint(
        f"        [structured_sd_repair] exhausted {tried} hypotheses, "
        f"no match ({time.time() - t0:.2f}s)", flush=True,
    )
    return None


# ---------------------------------------------------------------------------
# Section 5.4: posterior-aware Prange repair
# (Algorithm 9: MarginalErrorRegion, Algorithm 11: MinimumOverlapSampler,
#  Algorithm 12: PosteriorAwarePrangeRepair)
# ---------------------------------------------------------------------------

def marginal_error_region(A, Q_hat, S, K_B):
    """
    Algorithm 9 (MarginalErrorRegion). Builds a coordinate-level estimate
    B of the likely support of the prediction error, by taking, for every
    active row, its predicted nonzero column together with its K_B
    highest-scoring alternative columns.

    :param A: the active support Supp(v)
    :param Q_hat: the monomial approximation of the secret Q
    :param S: the local score matrix from monomial_approximation
    :param K_B: the per-row coordinate budget
    :return: a set B of column indices, the estimated error region
    """
    n = Q_hat.ncols()
    B = set()

    for i in A:
        # The coordinate currently predicted for row i is always included,
        # since every erroneous active row contributes its Q_hat entry.
        pi_hat_i = next(j for j in range(n) if Q_hat[i, j] != 0)
        B.add(pi_hat_i)

        top_j = sorted(range(n), key=lambda j: S[i][j], reverse=True)[:K_B]
        B.update(top_j)

    return B


def minimum_overlap_sampler(n, k, B):
    """
    Algorithm 11 (MinimumOverlapSampler). Samples an information set I of
    size k while minimizing its overlap with the estimated error region B,
    by filling I with coordinates outside B first and only reaching into B
    when unavoidable.

    :param n: the code length
    :param k: the information-set size (code dimension)
    :param B: the estimated error region, a set of column indices
    :return: an information set I, as a list of k distinct indices
    """
    G = [j for j in range(n) if j not in B]
    B_list = list(B)

    h_min = max(0, k - len(G))

    I_B = pyrandom_sample(B_list, h_min)
    I_G = pyrandom_sample(G, k - h_min)

    return I_B + I_G


def pyrandom_sample(population, size):
    """
    Small helper wrapping a uniform-without-replacement sample, used by
    minimum_overlap_sampler (kept local so the module only depends on
    SageMath's random(), matching the style of instances_generator.py).

    :param population: the list to sample from
    :param size: the number of elements to draw
    :return: a list of `size` distinct elements drawn from population
    """
    pool = list(population)
    sample = []
    for _ in range(size):
        idx = randint(0, len(pool) - 1)
        sample.append(pool.pop(idx))
    return sample


def posterior_aware_prange_repair(w_tilde, v, H_prime, Q_hat, S, k, tau,
                                   K_B=2, max_trials=2000):
    """
    Algorithm 12 (PosteriorAwarePrangeRepair). A lighter repair strategy
    than structured_sd_repair: posterior information is used only to bias
    the choice of Prange information sets away from the estimated error
    region B, after which each trial is an ordinary Prange decoding step.

    :param w_tilde: the predicted word, w_tilde = v * Q_hat
    :param v: the codeword v in C that produced w_tilde
    :param H_prime: a parity-check matrix of C'
    :param Q_hat: the monomial approximation of the secret Q
    :param S: the local score matrix from monomial_approximation
    :param k: the dimension of C' (size of the information set)
    :param tau: the target decoding radius
    :param K_B: the per-row coordinate budget for the error-region selector
    :param max_trials: the maximum number of Prange trials
    :return: the repaired word w in C', or None on Failure
    """
    F = H_prime.base_ring()
    n = H_prime.ncols()
    target_weight = v.hamming_weight()

    s = H_prime * w_tilde
    if s.is_zero():
        return w_tilde if w_tilde.hamming_weight() == target_weight else None

    A = support(v)
    B = marginal_error_region(A, Q_hat, S, K_B)

    for _ in range(max_trials):
        I = minimum_overlap_sampler(n, k, B)
        J = [j for j in range(n) if j not in I]

        H_J = H_prime.matrix_from_columns(J)
        if H_J.rank() < len(J):
            continue

        e_J = H_J.solve_right(s)

        e = vector(F, n)
        for pos, col in enumerate(J):
            e[col] = e_J[pos]

        if e.hamming_weight() <= tau:
            w_candidate = w_tilde - e
            # Reject syndrome-only coincidences: a genuine equivalent pair
            # must preserve the Hamming weight of v exactly (Section 2.1).
            if w_candidate.hamming_weight() == target_weight:
                return w_candidate
            # Otherwise, keep trying other information sets.

    return None


# ---------------------------------------------------------------------------
# Section 3.3 / Algorithm 1: enumeration module and top-level framework
# ---------------------------------------------------------------------------

def enum_low_weight_codeword(G, target_weight, max_trials=500):
    """
    A simple black-box enumeration module Enum(C, theta_enum): samples
    random messages until a codeword of Hamming weight at most the
    prescribed target is produced. This is a placeholder implementation;
    any better low-weight codeword generator can be substituted without
    changing the rest of the framework (Section 3.3).

    Note: for a random [n, k] code, the minimum distance is generically
    close to the Singleton bound n - k + 1, so requesting a target_weight
    well below that bound may never succeed. Use
    brute_force_min_weight_codewords (for small q**k) to find an achievable
    target before calling this.

    :param G: a generator matrix of C, in row-reduced echelon form
    :param target_weight: the maximum acceptable Hamming weight of the
        returned codeword (the search accepts weight <= target_weight)
    :param max_trials: the maximum number of sampled messages to try
    :return: a codeword v in C of weight <= target_weight, or None on Failure
    """
    F = G.base_ring()
    k = G.nrows()

    t0 = time.time()
    t_last = t0
    for trial in range(max_trials):
        # Sample a random message with a random number of nonzero entries
        # (from 1 up to all k coordinates), so that every linear combination
        # of rows of G is reachable, not just single scaled rows. Capping
        # num_nonzero at k // 2 (as an earlier version of this function did)
        # restricts v to a scalar multiple of a single row whenever k <= 2,
        # which usually cannot reach the code's actual minimum weight.
        message = vector(F, k)
        num_nonzero = randint(1, k)
        for idx in pyrandom_sample(range(k), num_nonzero):
            a = F.random_element()
            while a == 0:
                a = F.random_element()
            message[idx] = a

        v = message * G
        if 0 < v.hamming_weight() <= target_weight:
            return v

        now = time.time()
        if VERBOSE and now - t_last >= 5.0:
            _vprint(
                f"        [enum_random] trial {trial + 1}/{max_trials} "
                f"target_weight={target_weight} elapsed={now - t0:.1f}s", flush=True,
            )
            t_last = now

    return None

import random

def enum_low_weight_codeword_lee_brickell(G, target_weight, max_trials=1000, p=2):
    """
    Encuentra una palabra de código de peso <= target_weight usando el algoritmo
    de Information Set Decoding (ISD) de Lee-Brickell.

    :param G: Matriz generadora del código (Sage Matrix).
    :param target_weight: Peso Hamming máximo deseado (0 < peso <= target_weight).
    :param max_trials: Número máximo de permutaciones de columnas (Information Sets) a probar.
    :param p: Número de filas a combinar por iteración (típicamente p=1 o p=2).
    :return: Un vector v en el código con 0 < v.hamming_weight() <= target_weight, o None si no se halló.
    """
    F = G.base_ring()
    k = G.nrows()
    n = G.ncols()
    
    # Validar parámetro p
    if p > k:
        p = k

    # Obtener los elementos no nulos del cuerpo (para combinaciones escalares en F_q)
    non_zero_elements = [x for x in F if x != 0]

    for trial in range(max_trials):
        # 1. Seleccionar una permutación aleatoria de las columnas
        perm = list(range(n))
        random.shuffle(perm)
        P = matrix(F, n, n)
        for i, j in enumerate(perm):
            P[i, j] = 1

        G_perm = G * P

        # 2. Intentar poner las primeras k columnas en forma sistemática [I_k | A]
        try:
            # Extraer submatriz de las primeras k columnas
            G_left = G_perm.submatrix(0, 0, k, k)
            if G_left.rank() < k:
                continue # No forma un Information Set válido, probar otra permutación
            
            # Formar [I_k | A] mediante eliminación gaussiana
            G_sys = G_left.inverse() * G_perm
        except:
            continue

        # Extraer la submatriz derecha A de tamaño k x (n - k)
        A = G_sys.submatrix(0, k, k, n - k)

        # 3. Probar combinaciones de p filas de G_sys
        from itertools import combinations, product

        for row_indices in combinations(range(k), p):
            # Probar escalares no nulos para las filas seleccionadas
            for coeffs in product(non_zero_elements, repeat=p):
                
                # Calcular el peso en las últimas (n - k) posiciones
                # v_right = sum(coeffs[i] * A[row_indices[i]])
                v_right = vector(F, n - k)
                for idx, c in zip(row_indices, coeffs):
                    v_right += c * A[idx]

                w_right = v_right.hamming_weight()

                # El peso total en la matriz permutada es p (en las k col) + w_right (en las n-k col)
                if 0 < (p + w_right) <= target_weight:
                    # Reconstruir el mensaje en la base permutada
                    msg_perm = vector(F, k)
                    for idx, c in zip(row_indices, coeffs):
                        msg_perm[idx] = c
                    
                    # Generar la palabra permutada completa
                    v_perm = msg_perm * G_sys
                    
                    # Deshacer la permutación (multiplicar por P^-1, que es P^T)
                    v = v_perm * P.transpose()
                    
                    # Verificación de seguridad
                    if 0 < v.hamming_weight() <= target_weight:
                        return v

    return None


def brute_force_min_weight_codewords(G, cap=200000):
    """
    Exhaustively searches C = <G> for the minimum nonzero Hamming weight and
    the codewords achieving it, by trying every nonzero message in F^k. Only
    intended for small instances where q**k <= cap; useful for picking a
    target_weight for enum_low_weight_codeword that is actually achievable
    (see the Singleton-bound remark in enum_low_weight_codeword).

    :param G: a generator matrix of C
    :param cap: refuse to run (raise ValueError) if q**k exceeds this many
        messages, to avoid an accidental full search over a huge code
    :return: a tuple (min_weight, examples), where examples is a list of up
        to 5 codewords of that minimum weight
    """
    F = G.base_ring()
    q = F.cardinality()
    k = G.nrows()

    if q ** k > cap:
        raise ValueError(
            f"q**k = {q ** k} exceeds cap = {cap}; brute force is not "
            f"intended for instances this large."
        )

    min_weight = None
    examples = []
    for message in itertools.product(F, repeat=k):
        if all(a == 0 for a in message):
            continue
        v = vector(F, message) * G
        w = v.hamming_weight()
        if min_weight is None or w < min_weight:
            min_weight = w
            examples = [v]
        elif w == min_weight and len(examples) < 5:
            examples.append(v)

    return min_weight, examples


def prediction_and_repair_framework(C_gen, C_prime_gen, Q_hat, m_pair, n_iter,
                                     repair_fn, enum_params=None):
    """
    Algorithm 1 (PredictionAndRepairFramework). Repeatedly generates a
    low-weight codeword v in C, computes the fixed prediction
    w_tilde = v * Q_hat, and attempts to repair w_tilde into a codeword of
    C' using the supplied repair module. Successful repairs are stored as
    candidate equivalent codeword pairs.

    :param C_gen: a generator matrix of C
    :param C_prime_gen: a generator matrix of C'
    :param Q_hat: the monomial approximation of the secret Q
    :param m_pair: the target number of candidate pairs to recover
    :param n_iter: the maximum number of iterations (enumeration attempts)
    :param repair_fn: a callable (w_tilde, v) -> w or None, wrapping one of
        induced_sd_repair / structured_sd_repair / posterior_aware_prange_repair
        with its problem-specific arguments already bound
    :param enum_params: optional dict of parameters forwarded to
        enum_low_weight_codeword (e.g. {'target_weight': t, 'max_trials': m})
    :return: the list P of recovered candidate pairs (v, w)
    """
    enum_params = enum_params or {'target_weight': C_gen.ncols() // 4}

    P = []
    for _ in range(n_iter):
        if len(P) >= m_pair:
            break

        v = enum_low_weight_codeword_lee_brickell(C_gen, **enum_params)
        if v is None:
            continue

        w_tilde = predict_image(v, Q_hat)
        w = repair_fn(w_tilde, v)

        if w is not None:
            P.append((v, w))

    return P


if __name__ == "__main__":
    from instances_generator import generate_noisy_LCE_instance_CBA_bit_flip_version

    # --- Example parameters (mirrors instances_generator.py's __main__) ---
    n, k, q = 7, 3, 7
    alpha, beta = 0.01, 0.2  # alpha: Pr[1 -> 0], beta: Pr[0 -> 1]
    F = GF(q)

    # --- Step 1: generate a noisy LEP instance (bitwise leakage channel) ---
    G1, G2, Q, Q_noisy = generate_noisy_LCE_instance_CBA_bit_flip_version(
        n, k, q, alpha, beta, is_monomial=True,
    )

    # --- Step 2: turn the noisy hint into a BBLM posterior table -----------
    posterior_table = compute_posterior_table(Q_noisy, alpha, beta, is_permutation=False)

    # --- Step 3: build the monomial approximation Q_hat --------------------
    Q_hat, S, D_loc, pi = monomial_approximation(posterior_table, F)

    print("Secret monomial matrix Q:")
    print(Q)
    print("\nMonomial approximation Q_hat:")
    print(Q_hat)
    print("\nNumber of correctly approximated rows:",
          sum(1 for i in range(n) if list(Q_hat[i]) == list(Q[i])), "/", n)

    # --- Step 4: run the prediction-and-repair framework --------------------
    H2 = obtain_parity_check_matrix(G2)
    rmax = 2

    # For this toy size (q**k = 343), find the actual minimum distance of C
    # so that the enumeration module is given a target weight that is
    # achievable. For a random [n, k] code this is generically close to the
    # Singleton bound n - k + 1, so a fixed small constant (e.g. 3) may not
    # correspond to any codeword at all.
    min_weight, _ = brute_force_min_weight_codewords(G1)
    print("\nMinimum distance of C (brute force):", min_weight)

    def repair_fn(w_tilde, v):
        A, L = build_active_row_lists(v, S, D_loc, F, budgets=3)
        return structured_sd_repair(w_tilde, v, H2, Q_hat, A, L, rmax)

    pairs = prediction_and_repair_framework(
        G1, G2, Q_hat, m_pair=2, n_iter=200, repair_fn=repair_fn,
        enum_params={'target_weight': min_weight, 'max_trials': 500},
    )

    print("\nRecovered candidate equivalent codeword pairs:")
    for v, w in pairs:
        print("v =", v, " w =", w)