"""
test_structured_repair.py  (Code/tests/)

Unit tests for the Section 4.2/4.3 pieces touched by the 2026-09-29 fix
(row_hypothesis_universe / build_row_domains / build_row_domains_list_based
now rank the FULL (column, coefficient) candidate universe, instead of
locking one coefficient per column before ranking columns - see
list_based_leakage_problem_and_fix.md and the module docstrings of
core/structured_repair_search_tree.py and
strategies/07_list_based_prediction_and_repair/list_based_row_domains.py
for the bug this replaced, confirmed independently by the thesis
director's remark of 2026-09-29 that L_i must contain the real row
somewhere in its score order).

Uses small, hand-checkable instances (n <= 6, q = 5 or 7) so every
expected value in these tests can be verified by hand from the posterior
tables constructed here, rather than only by re-running the same code.

Run with (from Code/, so the core/ modules import):
    sage -python -m unittest tests/test_structured_repair.py -v
or equivalently:
    sage tests/test_structured_repair.py

NOT RUN WITH `sage` IN THIS SESSION: like every Sage-dependent file in
this codebase (see e.g. the validation note at the top of
core/structured_repair_search_tree.py), this file was written with no
SageMath installation available and has not itself been executed here.
The plain-Python logic it exercises (full-universe ranking, exclusion of
the keep pair, nonincreasing score order) was independently validated
against a Sage-free reimplementation before this file was written - see
validate_row_domains_logic.py, shared alongside this file - but that is
not a substitute for actually running THIS file with `sage`.
"""
import os
import sys
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', 'core'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', 'strategies',
                                 '07_list_based_prediction_and_repair'))

from sage.all import GF, matrix, vector

from LEP_prediction_and_repair_v2 import (
    row_hypothesis_universe,
    compute_posterior_table,
    compute_posterior_table_list_based,
    monomial_approximation,
    _safe_log,
)
from structured_repair_search_tree import (
    build_row_domains,
    completion_bound,
    RowEnumerationIterator,
    RowState,
    build_word_from_choice,
)
from list_based_row_domains import (
    compute_channel_posteriors_list_based,
    build_row_domains_list_based,
)


def _uniform_row(F, n, mass):
    """A posterior_table row where every column/value pair in `mass`
    ({(j, a): p, ...}) has that probability and everything else shares
    whatever mass remains, split evenly - just enough structure to hand
    -verify a few specific entries without writing out all n*q of them."""
    row = [dict() for _ in range(n)]
    for j in range(n):
        for a in F:
            row[j][a] = mass.get((j, a), 0.0)
    for j in range(n):
        total = sum(row[j].values())
        assert 0.0 < total <= 1.0 + 1e-9, (j, total)
        # Whatever probability wasn't explicitly assigned for this column
        # is spread over its unassigned entries so every row sums to 1.
        unassigned = [a for a in F if (j, a) not in mass]
        if unassigned:
            leftover = (1.0 - total) / len(unassigned)
            for a in unassigned:
                row[j][a] += leftover
    return row


class TestRowHypothesisUniverse(unittest.TestCase):
    """Section 4.3.3 / Definition 9 (matrix representation)."""

    def test_full_universe_size_and_ordering(self):
        F = GF(5)
        n = 3
        # Row 1, column 1 clearly favors a=2; everything else is close to
        # uniform, so the ranking is dominated by that one entry.
        row = _uniform_row(F, n, {(1, F(2)): 0.6, (1, F(0)): 0.1})
        universe = row_hypothesis_universe(1, row, F, n)

        # n=3 columns * (q-1)=4 nonzero values = 12 hypotheses total.
        self.assertEqual(len(universe), n * 4)
        # Sorted by nonincreasing score.
        scores = [s for _, _, s in universe]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # The dominant hypothesis must be (column 1, value 2).
        self.assertEqual((universe[0][0], universe[0][1]), (1, F(2)))

    def test_exclude_drops_exactly_one_pair(self):
        F = GF(5)
        n = 3
        row = _uniform_row(F, n, {(1, F(2)): 0.6, (1, F(0)): 0.1})
        full = row_hypothesis_universe(1, row, F, n)
        without_keep = row_hypothesis_universe(1, row, F, n, exclude=(1, F(2)))
        self.assertEqual(len(without_keep), len(full) - 1)
        self.assertNotIn((1, F(2)), [(j, a) for j, a, _ in without_keep])


class TestBuildRowDomains(unittest.TestCase):
    """core/structured_repair_search_tree.py's build_row_domains
    (matrix representation) - the tree's own domain builder."""

    def _tiny_instance(self):
        # n=4, q=5. Q_hat is the identity-ish monomial matrix
        # (pi_tilde(i) = i, d_tilde_i = 1 for every row) so the keep
        # choice is easy to state by hand.
        F = GF(5)
        n = 4
        Q_hat = matrix(F, n, n)
        for i in range(n):
            Q_hat[i, i] = F(1)
        H_prime = matrix(F, 2, n, [[1, 0, 1, 0], [0, 1, 0, 1]])
        v = vector(F, [1, 1, 1, 1])
        return F, n, Q_hat, H_prime, v

    def test_keep_choice_present_at_cost_zero(self):
        F, n, Q_hat, H_prime, v = self._tiny_instance()
        posterior_table = [_uniform_row(F, n, {}) for _ in range(n)]
        domains = build_row_domains(v, [0, 1], posterior_table, Q_hat, H_prime, F, L=2)

        for i in (0, 1):
            keeps = [opt for opt in domains[i] if opt.cost == 0]
            self.assertEqual(len(keeps), 1, f"row {i} should have exactly one keep option")
            self.assertEqual((keeps[0].j, keeps[0].a), (i, F(1)))
            self.assertTrue(keeps[0].sigma.is_zero(), "keep option's sigma must be the zero vector")

    def test_domain_size_is_one_plus_L_when_universe_is_large_enough(self):
        F, n, Q_hat, H_prime, v = self._tiny_instance()
        posterior_table = [_uniform_row(F, n, {}) for _ in range(n)]
        L = 3
        domains = build_row_domains(v, [0], posterior_table, Q_hat, H_prime, F, L=L)
        self.assertEqual(len(domains[0]), 1 + L)

    def test_recovers_hypothesis_the_old_per_column_argmax_construction_missed(self):
        """
        Reproduces the originally reported failure mode: column 1's own
        argmax coefficient is F(4), NOT the true coefficient F(2), which
        is only that column's runner-up. The OLD build_row_domains locked
        D_loc[i][1] = F(4) and could never offer F(2) at column 1, no
        matter how large L was. The NEW construction ranks the full
        (column, coefficient) universe, so (1, F(2)) is offered whenever
        its overall rank is < L.
        """
        F, n, Q_hat, H_prime, v = self._tiny_instance()
        row_0 = _uniform_row(F, n, {(1, F(4)): 0.5, (1, F(2)): 0.3, (1, F(0)): 0.05})
        posterior_table = [row_0] + [_uniform_row(F, n, {}) for _ in range(n - 1)]

        domains = build_row_domains(v, [0], posterior_table, Q_hat, H_prime, F, L=5)
        offered = [(opt.j, opt.a) for opt in domains[0]]
        self.assertIn((1, F(2)), offered, "the true (column, coefficient) pair must be reachable")
        # And, as a sanity check on the scenario itself, F(4) (that
        # column's own argmax) should ALSO be offered as a *separate*
        # option at the SAME column - something the old per-column
        # construction could never represent (one slot per column).
        self.assertIn((1, F(4)), offered)


class TestBuildRowDomainsListBased(unittest.TestCase):
    """strategies/07's build_row_domains_list_based (two-vector repr)."""

    def test_full_cross_product_ranked_correctly(self):
        F = GF(5)
        n = 3
        Q_hat = matrix(F, n, n)
        for i in range(n):
            Q_hat[i, i] = F(1)
        H_prime = matrix(F, 2, n, [[1, 0, 1], [0, 1, 1]])
        v = vector(F, [1, 1, 1])

        perm_posteriors = [[0.05, 0.90, 0.05] for _ in range(n)]
        scale_posteriors = [{F(1): 0.1, F(2): 0.6, F(3): 0.2, F(4): 0.1} for _ in range(n)]

        domains = build_row_domains_list_based(
            v, [0], perm_posteriors, scale_posteriors, Q_hat, H_prime, F, L=3
        )
        offered = [(opt.j, opt.a, opt.cost) for opt in domains[0]]
        # Keep is (0, 1) at cost 0.
        self.assertIn((0, F(1), 0), offered)
        # The dominant non-keep hypothesis should be (column 1, value 2) -
        # column 1 dominates the permutation channel and value 2 dominates
        # the scale channel, and the score is additive in the two.
        non_keep = [opt for opt in domains[0] if opt.cost == 1]
        best = max(non_keep, key=lambda opt: opt.log_score)
        self.assertEqual((best.j, best.a), (1, F(2)))


class TestRowEnumerationIterator(unittest.TestCase):
    """Algorithms 9-11 (CompletionBound / InitializeRowEnumeration /
    NextCandidate) via core/structured_repair_search_tree.py."""

    def test_nonincreasing_order_and_exhaustion(self):
        F = GF(5)
        n = 3
        Q_hat = matrix(F, n, n)
        for i in range(n):
            Q_hat[i, i] = F(1)
        H_prime = matrix(F, 2, n, [[1, 0, 1], [0, 1, 1]])
        v = vector(F, [1, 1, 1])
        posterior_table = [_uniform_row(F, n, {}) for _ in range(n)]

        domains = build_row_domains(v, [0, 1], posterior_table, Q_hat, H_prime, F, L=2)
        active_order = [0, 1]
        s = vector(F, [0, 0])  # residual for an all-keep (accepting) leaf

        it = RowEnumerationIterator(active_order, domains, s, r=2, v_max=None)
        build_word = lambda choice: build_word_from_choice(v, choice, n, F)

        seen_scores = []
        result = it.next_candidate(build_word=build_word)
        while result != 'Exhausted':
            self.assertNotEqual(result, 'Limit', "v_max=None should never hit a limit")
            _choice, _w, g = result
            seen_scores.append(float(g))
            result = it.next_candidate(build_word=build_word)

        self.assertGreater(len(seen_scores), 0)
        self.assertEqual(seen_scores, sorted(seen_scores, reverse=True))

    def test_v_max_returns_limit_and_resumes(self):
        F = GF(5)
        n = 3
        Q_hat = matrix(F, n, n)
        for i in range(n):
            Q_hat[i, i] = F(1)
        H_prime = matrix(F, 2, n, [[1, 0, 1], [0, 1, 1]])
        v = vector(F, [1, 1, 1])
        posterior_table = [_uniform_row(F, n, {}) for _ in range(n)]
        domains = build_row_domains(v, [0, 1], posterior_table, Q_hat, H_prime, F, L=2)

        it = RowEnumerationIterator([0, 1], domains, vector(F, [0, 0]), r=2, v_max=1)
        result = it.next_candidate()
        self.assertEqual(result, 'Limit')
        self.assertEqual(it.visited, 1)

        # Raising v_max and calling again resumes from the same queue
        # state rather than restarting - visited keeps counting up.
        it.v_max = 1000
        result = it.next_candidate()
        self.assertNotEqual(result, 'Limit')


if __name__ == "__main__":
    unittest.main()
