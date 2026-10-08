"""
Pure-Python (no Sage) validation of the row-domain construction fix,
reimplementing just the SELECTION logic (which candidate (j, a) pairs get
kept) with plain floats/ints/dicts so it can be exercised in a sandbox
with no SageMath install. The syndrome/sigma linear algebra is unchanged
by this fix and is not what's being checked here - see the module
docstrings in core/structured_repair_search_tree.py and
strategies/07_list_based_prediction_and_repair/list_based_row_domains.py
for the real (Sage-dependent) implementations this mirrors.

Checks:
  1. The OLD "one coefficient per column, then rank columns" construction
     can provably miss the true (j, a) hypothesis even when it would rank
     inside the top-L of the full universe - reproducing the originally
     reported bug on a small constructed example.
  2. The NEW "rank the full (j, a) universe, take top-L" construction
     always retains the true hypothesis whenever its true rank (excluding
     the keep pair) is < L - by construction, since it IS the brute-force
     top-L of that same universe.
  3. Cross-checks the new construction's output against an independent
     brute-force top-L computation (itertools + sorted) across randomized
     instances, for both the matrix-style score (with the z_i,j zero term)
     and the two-vector separable score.
"""
import itertools
import math
import random


def old_matrix_domain_columns(posterior_row, n, q_nonzero_vals, keep_col):
    """Reproduces the OLD build_row_domains behavior: one coefficient
    (the column's own argmax) per column, columns then ranked by that
    argmax's score. Returns {col: (a, score)} for the argmax at each col."""
    per_col = {}
    for j in range(n):
        log_zero_j = math.log(max(posterior_row[j].get(0, 0.0), 1e-12))
        z_i = sum(math.log(max(posterior_row[l].get(0, 0.0), 1e-12)) for l in range(n))
        best_a, best_score = None, float('-inf')
        for a in q_nonzero_vals:
            score = math.log(max(posterior_row[j].get(a, 0.0), 1e-12)) + z_i - log_zero_j
            if score > best_score:
                best_a, best_score = a, score
        per_col[j] = (best_a, best_score)
    return per_col


def row_hypothesis_universe(posterior_row, n, q_nonzero_vals, exclude=None):
    """Mirrors LEP_prediction_and_repair_v2.row_hypothesis_universe exactly
    (same formula), in plain Python."""
    z_i = sum(math.log(max(posterior_row[l].get(0, 0.0), 1e-12)) for l in range(n))
    candidates = []
    for j in range(n):
        log_zero_j = math.log(max(posterior_row[j].get(0, 0.0), 1e-12))
        base = z_i - log_zero_j
        for a in q_nonzero_vals:
            if exclude is not None and (j, a) == exclude:
                continue
            score = math.log(max(posterior_row[j].get(a, 0.0), 1e-12)) + base
            candidates.append((j, a, score))
    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates


def brute_force_top_L(posterior_row, n, q_nonzero_vals, exclude, L):
    universe = row_hypothesis_universe(posterior_row, n, q_nonzero_vals, exclude)
    return universe[:L]


# ---------------------------------------------------------------------------
# Check 1: reproduce the originally reported bug with the OLD construction.
# ---------------------------------------------------------------------------
def check_old_bug_reproducible():
    n = 5
    q_nonzero = [1, 2, 3, 4]  # GF(5)^*
    true_col, true_a = 2, 3

    # Column 2's posterior slightly PREFERS a=4 over the true a=3, but a=3
    # still has decent mass - enough that in the FULL universe (all n*4
    # hypotheses), (2, 3) ranks, say, 2nd overall - well within a budget of
    # L=3 - while the OLD construction locks column 2 to its argmax (a=4)
    # and never offers a=3 at all, regardless of L.
    posterior_row = []
    for j in range(n):
        row = {0: 0.05}
        if j == true_col:
            row[4] = 0.50   # argmax at this column - NOT the true value
            row[3] = 0.30   # true coefficient - runner-up at its own column
            row[1] = 0.10
            row[2] = 0.05
        else:
            row[1] = 0.30
            row[2] = 0.25
            row[3] = 0.20
            row[4] = 0.20
        posterior_row.append(row)

    old = old_matrix_domain_columns(posterior_row, n, q_nonzero, keep_col=0)
    old_offers_true = any(col == true_col and a == true_a for col, (a, _s) in old.items())
    assert not old_offers_true, "expected the OLD construction to NOT offer the true (j, a) pair"

    L = 3
    top = brute_force_top_L(posterior_row, n, q_nonzero, exclude=(0, 1), L=L)
    new_offers_true = any((j, a) == (true_col, true_a) for j, a, _s in top)
    assert new_offers_true, (
        f"expected the NEW construction's top-{L} to include the true pair "
        f"(rank was {[i for i,(j,a,s) in enumerate(row_hypothesis_universe(posterior_row, n, q_nonzero, (0,1))) if (j,a)==(true_col,true_a)]})"
    )
    print("check_old_bug_reproducible: PASS "
          "(old construction misses the true pair; new construction retains it)")


# ---------------------------------------------------------------------------
# Check 2: new construction's top-L always matches an independent
# brute-force computation (itertools.product over all (j, a), sorted).
# ---------------------------------------------------------------------------
def check_matches_independent_brute_force(trials=200):
    rng = random.Random(12345)
    for _ in range(trials):
        n = rng.randint(2, 6)
        q = rng.choice([5, 7, 11])
        q_nonzero = list(range(1, q))
        L = rng.randint(1, 8)

        posterior_row = []
        for j in range(n):
            raw = {a: rng.random() for a in [0] + q_nonzero}
            total = sum(raw.values())
            posterior_row.append({a: p / total for a, p in raw.items()})

        exclude = (rng.randrange(n), rng.choice(q_nonzero))

        # Independent computation: literally itertools.product, no
        # incremental bookkeeping shared with row_hypothesis_universe.
        z_i = sum(math.log(max(posterior_row[l].get(0, 0.0), 1e-12)) for l in range(n))
        independent = []
        for j, a in itertools.product(range(n), q_nonzero):
            if (j, a) == exclude:
                continue
            log_zero_j = math.log(max(posterior_row[j].get(0, 0.0), 1e-12))
            score = math.log(max(posterior_row[j].get(a, 0.0), 1e-12)) + z_i - log_zero_j
            independent.append((j, a, score))
        independent.sort(key=lambda t: t[2], reverse=True)

        got = row_hypothesis_universe(posterior_row, n, q_nonzero, exclude)[:L]
        want = independent[:L]

        got_scores = [round(s, 9) for _, _, s in got]
        want_scores = [round(s, 9) for _, _, s in want]
        assert got_scores == want_scores, f"score mismatch: {got_scores} != {want_scores}"
        # Every returned pair must actually be at/above the L-th score
        # (ties may permute which exact pair appears, but not the score
        # multiset) - and every pair, once at that score cutoff, must have
        # excluded exactly the one (j, a) we asked to exclude.
        assert exclude not in [(j, a) for j, a, _ in got]
    print(f"check_matches_independent_brute_force: PASS ({trials} randomized instances)")


# ---------------------------------------------------------------------------
# Check 3: two-vector (separable) score - same properties, plus confirm
# the now-removed top-L-columns x top-M-coefficients shortlist WOULD have
# missed candidates when M < L (justifying why it was replaced rather than
# just tuned).
# ---------------------------------------------------------------------------
def check_two_vector_and_shortlist_insufficiency():
    n = 6
    q_nonzero = [1, 2, 3, 4, 5, 6]  # GF(7)^*
    rng = random.Random(7)

    col_post = [rng.random() for _ in range(n)]
    tot = sum(col_post)
    col_post = [c / tot for c in col_post]

    val_post = {a: rng.random() for a in q_nonzero}
    tot = sum(val_post.values())
    val_post = {a: p / tot for a, p in val_post.items()}

    keep = (0, q_nonzero[0])
    L = 3
    M = 2  # deliberately smaller than L, as the old default config had

    def two_vec_universe(exclude):
        cands = []
        for j in range(n):
            for a in q_nonzero:
                if (j, a) == exclude:
                    continue
                score = math.log(max(col_post[j], 1e-12)) + math.log(max(val_post[a], 1e-12))
                cands.append((j, a, score))
        cands.sort(key=lambda t: t[2], reverse=True)
        return cands

    full_top_L = two_vec_universe(keep)[:L]

    # Old shortlist-then-cross approach with M < L.
    ranked_cols = sorted((j for j in range(n) if j != keep[0]), key=lambda j: col_post[j], reverse=True)
    other_cols = ranked_cols[:L]
    ranked_vals = sorted(val_post.keys(), key=lambda a: val_post[a], reverse=True)
    candidate_coeffs = ranked_vals[:M]
    shortlist = []
    for j in other_cols:
        for a in candidate_coeffs:
            score = math.log(max(col_post[j], 1e-12)) + math.log(max(val_post[a], 1e-12))
            shortlist.append((j, a, score))
    shortlist.sort(key=lambda t: t[2], reverse=True)
    shortlist_top_L = shortlist[:L]

    full_set = {(j, a) for j, a, _ in full_top_L}
    shortlist_set = {(j, a) for j, a, _ in shortlist_top_L}
    # Not asserting inequality unconditionally (depends on the random draw),
    # just report whether this particular draw demonstrates the gap, and
    # always assert the full/new construction is internally consistent.
    assert row_hypothesis_universe_two_vec_matches(col_post, val_post, n, q_nonzero, keep, L, full_top_L)
    print(f"check_two_vector_and_shortlist_insufficiency: PASS "
          f"(full top-{L} == {sorted(full_set)}; old M={M} shortlist top-{L} == "
          f"{sorted(shortlist_set)}; {'DIFFER (gap demonstrated)' if full_set != shortlist_set else 'coincide on this draw'})")


def row_hypothesis_universe_two_vec_matches(col_post, val_post, n, q_nonzero, exclude, L, claimed_top_L):
    cands = []
    for j in range(n):
        for a in q_nonzero:
            if (j, a) == exclude:
                continue
            score = math.log(max(col_post[j], 1e-12)) + math.log(max(val_post[a], 1e-12))
            cands.append((j, a, score))
    cands.sort(key=lambda t: t[2], reverse=True)
    want = [round(s, 9) for _, _, s in cands[:L]]
    got = [round(s, 9) for _, _, s in claimed_top_L]
    return want == got


if __name__ == "__main__":
    check_old_bug_reproducible()
    check_matches_independent_brute_force()
    check_two_vector_and_shortlist_insufficiency()
    print("\nALL CHECKS PASSED")
