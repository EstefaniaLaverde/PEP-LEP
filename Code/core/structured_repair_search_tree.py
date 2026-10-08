"""
structured_repair_search_tree.py  (Code/core/ - shared library)

Implements Section 5 of the manuscript ("Monomial Approximation, Prediction
and Repair", integrated revision, 13 September 2026): "Structured repair by
constrained best-first enumeration". Specifically:

    Algorithm 2  CompletionBound          -> completion_bound
    Algorithm 3  InitializeRowEnumeration -> RowEnumerationIterator(...)
    Algorithm 4  NextCandidate            -> RowEnumerationIterator.next_candidate

plus the supporting constructions of Sections 5.1-5.3 (the retained option
domains D_i, the keep choice K_i, departure costs c_i, log scores lambda_i,
and the syndrome contributions sigma_i - build_row_domains), the word
assembly of (5.2) (build_word_from_choice), and the depth-first comparison
policy of Section 5.9 (depth_first_repair).

Like the rest of Code/core/, this module is written throughout in terms of
Sage vectors and matrices (Q_hat, H_prime, v, the residual syndrome, and
the per-option sigma_i(j, a) contributions are all Sage objects), reusing
the local score/value matrices S and D_loc that monomial_approximation
already produces and the `support` helper from LEP_prediction_and_repair_v2.py,
exactly as build_active_row_lists / structured_sd_repair do.

RELATIONSHIP TO structured_sd_repair (Algorithm 7/8, older section
numbering, in LEP_prediction_and_repair_v2.py): that function is an
unranked brute-force search over subsets of active rows and their
replacement options, tried in itertools.product order until one hypothesis
happens to match the syndrome. It has no priority queue and no notion of
an optimistic bound. This module implements the CURRENT manuscript's
actual Section 5 instead: a best-first search tree, keyed by an admissible
bound on the best possible completion of a partial assignment, so
successive candidates come out in nonincreasing total log score
(Proposition 5.1) - the FIRST candidate returned is provably the
maximum-log-score element of the feasible set Omega, not just "some"
element of it. structured_sd_repair itself is left untouched; this module
is a new sibling, the same way posterior_repair_radius.py sits next to
LEP_prediction_and_repair_v2.py rather than inside it.

Using SageMath. Depends on LEP_prediction_and_repair_v2.py (for `support`)
being an importable sibling in Code/core/.

NOT RUN WITH `sage` IN THIS SESSION: like several existing files in this
codebase (see e.g. the validation note at the top of
Code/strategies/07_list_based_prediction_and_repair/
planted_codeword_repair_test_list_based_algo5.py), this module was written
in a sandbox with no SageMath installation, so it has not itself been
executed here. Its control flow (the queue, the bound, the legality and
acceptance tests) was cross-checked against independent brute-force
enumeration across thousands of randomized instances using a
field-agnostic prototype during development - see the validation note
above RowEnumerationIterator for exactly what that did and did not cover.
Code/strategies/08_structured_repair_search_tree/structured_repair_search_tree_demo.py
runs this module itself against the existing pipeline and needs a real
`sage` run to confirm before relying on it.
"""
import heapq
from collections import namedtuple

from sage.all import vector

import LEP_prediction_and_repair_v2 as _lep
from LEP_prediction_and_repair_v2 import support, row_hypothesis_universe, _safe_log


def _vprint(*args, **kwargs):
    """
    Prints only when LEP_prediction_and_repair_v2's VERBOSE switch is on
    (see set_verbose there), so one set_verbose(True) call traces both
    modules. Read at call time, not import time.
    """
    if _lep.VERBOSE:
        print(*args, **kwargs)


def _fmt_choice(choice):
    """Compact {i: RowOption} -> '{i->(j,a), ...}' string for traces."""
    return "{" + ", ".join(f"{i}->({opt.j},{opt.a})" for i, opt in sorted(choice.items())) + "}"


def _fmt_opt(opt):
    """Compact RowOption string for traces."""
    tag = "keep" if opt.cost == 0 else "move"
    return f"({opt.j},{opt.a})[{tag}, lambda={float(opt.log_score):.3f}]"

RowOption = namedtuple("RowOption", ["j", "a", "cost", "log_score", "sigma"])
# One retained hypothesis (j, a) in the domain D_i of some active row i
# (Section 5.1):
#   j          - the candidate destination column
#   a          - the coefficient (nonzero field element) attached to j,
#                i.e. a_{i,j} from Section 3.3 / D_loc[i][j]
#   cost       - the departure cost c_i(j, a) in {0, 1}: 0 exactly for the
#                keep choice K_i = (pi_tilde(i), d_tilde_i), 1 otherwise
#   log_score  - lambda_i(j, a) = log w_i(j, a); this is exactly S[i][j]
#                from compute_row_scores/monomial_approximation, so no
#                logarithm is recomputed here
#   sigma      - the precomputed syndrome contribution sigma_i(j, a) from
#                (5.4): a Sage vector over F, of length m

RowState = namedtuple("RowState", ["ell", "b", "used", "rho", "choice", "g"])
# A node of the search tree - the six state fields of Section 5.3, minus
# the active order i_1 < ... < i_t and the domains, which do not change
# during a run and live on the iterator instead:
#   ell    - next active position: active_order[0 .. ell - 2] are assigned;
#            the root has ell = 1
#   b      - remaining departure budget (starts at r)
#   used   - frozenset of destinations already claimed by assigned rows
#   rho    - remaining syndrome (a Sage vector): s minus the contributions
#            charged so far
#   choice - dict {i: RowOption} of the options assigned so far
#   g      - Lambda of the partial assignment: the sum of assigned log scores


def build_row_domains(v, A, posterior_table, Q_hat, H_prime, F, L):
    """
    Section 4.3.3 (BuildActiveRowLists, Algorithm 8) / Section 5.1 of this
    module. Builds, for every active row i in A = Supp(v), the finite
    retained-option domain D_i, with departure costs c_i(j, a), fixed log
    scores lambda_i(j, a), and precomputed syndrome contributions
    sigma_i(j, a) (5.4).

    D_i = {K_i} u L_i, where K_i = (pi_tilde(i), d_tilde_i) is the keep
    choice (read off Q_hat's own row i, cost 0) and L_i is the L
    highest-scoring OTHER row hypotheses R_i(j, a) from the FULL candidate
    universe {(j, a) : j in [n], a in F_q^*} (row_hypothesis_universe),
    scored individually via gamma_i(j, a) - NOT one coefficient locked per
    column. That distinction matters: a column can appear in L_i more than
    once, with different coefficients, if its second-best coefficient
    still outscores other columns' best; and the keep COLUMN itself can
    appear again with a different (non-keep) coefficient, representing
    "row i's true destination is where it already is, but with a
    different value" - a case the old per-column-argmax construction could
    never express, since it discarded every non-argmax coefficient before
    this function ever saw it. See row_hypothesis_universe's docstring
    and list_based_leakage_problem_and_fix.md for the concrete failure
    this fixes (confirmed independently by the thesis director, 2026-09-29:
    L_i must contain the real row somewhere in its score order, not be
    pre-filtered down to a single coefficient per column).

    The combined list is then sorted by nonincreasing log score, with ties
    broken by (destination, coefficient) for a fixed, deterministic order
    - so keep ends up wherever its own score places it, not always first.

    :param v: a row vector v in C
    :param A: the active support Supp(v) (any order; not assumed sorted)
    :param posterior_table: the n x n posterior table from
        compute_posterior_table / compute_posterior_table_list_based -
        posterior_table[i][j] = {a: p_i,j(a) for a in F_q}
    :param Q_hat: the monomial approximation of the secret Q (its row i
        gives the keep choice: pi_tilde(i) is the unique j with
        Q_hat[i, j] != 0, and d_tilde_i = Q_hat[i, pi_tilde(i)])
    :param H_prime: a parity-check matrix of C' (used only for its columns,
        to precompute sigma_i(j, a))
    :param F: the base field GF(q)
    :param L: the number of non-keep options retained per active row
    :return: {i: [RowOption, ...]} for i in A, each list sorted by
        nonincreasing log_score with the fixed tie rule
    """
    n = Q_hat.ncols()
    columns = [H_prime.column(j) for j in range(n)]

    domains = {}
    for i in A:
        pi_tilde_i = next(j for j in range(n) if Q_hat[i, j] != 0)
        d_tilde_i = Q_hat[i, pi_tilde_i]

        universe = row_hypothesis_universe(i, posterior_table[i], F, n,
                                            exclude=(pi_tilde_i, d_tilde_i))
        top = universe[:L]

        # Keep choice K_i = (pi_tilde(i), d_tilde_i), cost 0. Its score is
        # gamma_i(pi_tilde(i), d_tilde_i) - the same formula
        # row_hypothesis_universe uses internally for every other (j, a) -
        # and its syndrome contribution sigma_i is algebraically the zero
        # vector (5.4), same as before this fix.
        keep_score = (_safe_log(posterior_table[i][pi_tilde_i].get(d_tilde_i, 0))
                      + sum(_safe_log(posterior_table[i][l].get(F(0), 0))
                            for l in range(n) if l != pi_tilde_i))
        keep_sigma = v[i] * (d_tilde_i * columns[pi_tilde_i] - d_tilde_i * columns[pi_tilde_i])

        options = [RowOption(j=pi_tilde_i, a=d_tilde_i, cost=0, log_score=keep_score,
                              sigma=keep_sigma)]
        for j, a, score in top:
            sigma = v[i] * (d_tilde_i * columns[pi_tilde_i] - a * columns[j])
            options.append(RowOption(j=j, a=a, cost=1, log_score=score, sigma=sigma))

        options.sort(key=lambda opt: (-opt.log_score, opt.j, opt.a))
        domains[i] = options

        _vprint(f"  [domains] row {i} (v_i={v[i]}): keep K_i=({pi_tilde_i},{d_tilde_i}), "
                f"|D_i|={len(options)}")
        for rank, opt in enumerate(options):
            tag = "KEEP" if opt.cost == 0 else "    "
            _vprint(f"      #{rank} {tag} (j={opt.j}, a={opt.a})  cost={opt.cost}  "
                    f"lambda={float(opt.log_score):.3f}  sigma={opt.sigma}")

    return domains


def completion_bound(state, active_order, domains):
    """
    Algorithm 2 (CompletionBound). Computes an optimistic upper bound UB(X)
    on the total log score Lambda(R) of any completion of the partial
    assignment X, or reports that X has no completion at all.

    For every remaining active row (positions ell..t), only options whose
    destination is not already in Used and whose departure cost does not
    exceed the remaining budget b are eligible (5.6's legality test,
    applied one row ahead); if some remaining row has no eligible option,
    X cannot be completed. Otherwise the bound adds, independently for
    each remaining row, its single best eligible log score (5.7) - a
    relaxation, since the maximizers picked for different rows may
    collectively reuse a destination or spend more than b departures in
    total; those relaxations make the bound optimistic (never smaller than
    any actual completion's score) and cheap to compute, at the cost of
    not being tight.

    :param state: a RowState
    :param active_order: the fixed list [i_1, ..., i_t] of active rows
    :param domains: {i: [RowOption, ...]} - i's finite retained-option list
    :return: (False, None) if some remaining row has no eligible option in
        its domain; otherwise (True, UB) with UB(X) >= Lambda(R) for every
        completion R of X
    """
    u = state.g
    t = len(active_order)
    for h in range(state.ell, t + 1):
        i = active_order[h - 1]
        best = None
        for opt in domains[i]:
            if opt.j in state.used or opt.cost > state.b:
                continue
            if best is None or opt.log_score > best:
                best = opt.log_score
        if best is None:
            return False, None
        u += best
    return True, u


def _child_state(state, i, opt):
    """
    Builds the child state Y obtained by legally assigning option opt to
    active row i in parent state X, following (5.6). Uses fresh copies
    throughout (a new frozenset, a new dict, a residual produced by Sage's
    vector subtraction, never mutated in place), so the parent state X
    remains valid and reusable after this call, as required for a
    persistent search tree where siblings are expanded from the same
    parent.

    Does not check legality itself (opt.j not in state.used and
    opt.cost <= state.b) - callers (next_candidate, depth_first_repair)
    are expected to filter for that first, exactly as Algorithm 4 does.

    :param state: the parent RowState X
    :param i: the active row being assigned (active_order[state.ell - 1])
    :param opt: the RowOption assigned to row i
    :return: the child RowState Y
    """
    new_choice = dict(state.choice)
    new_choice[i] = opt
    return RowState(
        ell=state.ell + 1,
        b=state.b - opt.cost,
        used=state.used | {opt.j},
        rho=state.rho - opt.sigma,
        choice=new_choice,
        g=state.g + opt.log_score,
    )


def build_word_from_choice(v, choice, n, F):
    """
    (5.2). Assembles the candidate word w(R) = sum_{i in A} v_i R(j_i, a_i)
    from a completed Choice dict - the accepted active-row assignment
    returned by RowEnumerationIterator.next_candidate or
    depth_first_repair.

    :param v: the row vector v in C that produced the prediction
    :param choice: {i: RowOption} for every i in Supp(v)
    :param n: the code length
    :param F: the base field GF(q)
    :return: the Sage vector w(R), of weight exactly len(choice)
    """
    w = vector(F, n)
    for i, opt in choice.items():
        w[opt.j] += v[i] * opt.a
    return w


class RowEnumerationIterator:
    """
    Algorithms 3 (InitializeRowEnumeration) and 4 (NextCandidate).

    A persistent best-first iterator over the feasible active-row
    assignments Omega (Section 5.1): repeated calls to next_candidate()
    emit its elements in nonincreasing total log score Lambda
    (Proposition 5.1 - soundness, ordering, and completeness), until
    Omega is exhausted or a pop cap is reached.

    Internals, matching the paper's Algorithm 3/4 exactly:
      - A max-priority queue keyed by (UB, -serial): the larger key is
        removed first, so equal-bound ties resolve in favor of the
        earlier insertion. Implemented with heapq (a min-heap) by pushing
        (-UB, serial, state) and popping the minimum - serial is unique,
        so Python never needs to compare two `state` objects (which hold
        Sage vectors with no total order) directly.
      - `visited` counts REMOVALS from the queue (the root, every
        expanded internal state, every accepted leaf, and every rejected
        leaf) - never insertions, and never a state discarded purely by a
        failed CompletionBound check before it was ever queued.
      - A pop-cap `v_max` (called V_max in the paper) that, once reached,
        returns 'Limit' while leaving the queue untouched, so a later
        call (after raising v_max) resumes exactly where the run left
        off. An empty queue takes precedence over the cap: finishing
        exactly at the cap still reports 'Exhausted' on the next call,
        not 'Limit'.

    VALIDATION NOTE: the control flow implemented here (queue ordering,
    the legality/bound checks in next_candidate, the residual bookkeeping
    in _child_state) is a direct, mechanical translation of a field-agnostic
    prototype that WAS run and cross-checked against independent
    brute-force enumeration across 2000+ randomized instances (varying the
    number of active rows, options per row, residual dimension, and
    departure radius r), plus a dedicated resume test for the v_max/Limit
    behavior, during development of this module. That prototype used plain
    Python tuples with a `-` operator and an equality-with-zero test in
    place of Sage vectors and field elements - the only two operations
    this class ever performs on `state.rho` and `opt.sigma` - so the
    swap to real Sage vectors here changes no control-flow decision, only
    which concrete objects flow through it. It does NOT substitute for
    actually running this module (and the Sage-specific parts introduced
    here - build_row_domains's sigma construction, build_word_from_choice)
    with `sage`; see this module's own docstring.
    """

    def __init__(self, active_order, domains, s, r, v_max=None):
        """
        Builds the root state (1, r, empty-set, s, empty-dict, 0),
        evaluates CompletionBound on it, and inserts it into an initially
        empty queue if it has any completion at all.

        :param active_order: the fixed list [i_1, ..., i_t] of active rows,
            in the order the tree assigns them (any fixed order is valid;
            it need not be numeric row order, though that is the natural
            default)
        :param domains: {i: [RowOption, ...]} for every i in active_order -
            each list already sorted and truncated per the fixed list
            policy of Section 5.1 (this class does not sort or truncate;
            it only respects whatever order and content build_row_domains
            gave it)
        :param s: the target residual, e.g. H_prime * w_tilde (a Sage
            vector over F)
        :param r: the departure radius, 0 <= r <= t = len(active_order)
        :param v_max: an optional cap on the number of pops across the
            iterator's whole lifetime; None means unlimited
        """
        self.active_order = list(active_order)
        self.domains = domains
        self.v_max = v_max
        self.visited = 0
        self._serial = 0
        self._heap = []

        root = RowState(ell=1, b=r, used=frozenset(), rho=s, choice={}, g=0.0)
        ok, u = completion_bound(root, self.active_order, domains)
        _vprint(f"  [bestfirst] init: active_order={self.active_order}, r={r}, s={s}, v_max={v_max}")
        if ok:
            heapq.heappush(self._heap, (-u, self._serial, root))
            self._serial += 1
            _vprint(f"  [bestfirst] root UB={float(u):.3f} -> queued")
        else:
            _vprint("  [bestfirst] root has no completion -> Omega is empty")

    def next_candidate(self, build_word=None, trace=None):
        """
        Algorithm 4 (NextCandidate). Repeatedly removes the maximum-key
        state from the queue until an accepting leaf (ell = t + 1 and
        rho = 0) is found, the queue empties, or v_max pops have been
        spent since construction.

        :param build_word: an optional callable choice -> w, used to turn
            an accepted leaf's Choice dict into a concrete word via (5.2);
            structured_sd_repair_bestfirst passes build_word_from_choice
            here. If omitted, w is always None (useful for inspecting the
            search itself without any field arithmetic).
        :param trace: an optional list; if given, one dict is appended per
            pop, with keys 'ell', 'b', 'used', 'choice' (as a
            {i: (j, a)} snapshot), 'rho', 'ub' (the popped state's own
            bound), and 'outcome' in {'expand', 'reject', 'return'}. This
            is the "complete queue trace" of Section 5.5/5.6, kept here for
            debugging and for testing this module against worked examples.
        :return: (choice, w, g) for the next-best assignment in Omega, in
            nonincreasing g order across successive calls; the string
            'Exhausted' once Omega has been fully emitted; or the string
            'Limit' if v_max is reached first (raise v_max and call again
            to resume from exactly this point)
        """
        t = len(self.active_order)
        while self._heap:
            if self.v_max is not None and self.visited >= self.v_max:
                _vprint(f"  [bestfirst] LIMIT: v_max={self.v_max} pops spent "
                        f"({len(self._heap)} states still queued)")
                return 'Limit'

            neg_u, _serial, state = heapq.heappop(self._heap)
            self.visited += 1
            ub = -neg_u
            _vprint(f"  [bestfirst] pop #{self.visited}: UB={float(ub):.3f} g={float(state.g):.3f} "
                    f"ell={state.ell}/{t + 1} b={state.b} used={sorted(state.used)} "
                    f"choice={_fmt_choice(state.choice)} rho={state.rho} "
                    f"(queue left={len(self._heap)})")

            if state.ell == t + 1:
                if state.rho.is_zero():
                    _vprint(f"  [bestfirst]   -> ACCEPT leaf (rho = 0), Lambda={float(state.g):.3f}")
                    if trace is not None:
                        trace.append(self._trace_row(state, ub, 'return'))
                    w = build_word(state.choice) if build_word is not None else None
                    return state.choice, w, state.g
                _vprint("  [bestfirst]   -> REJECT leaf (rho != 0)")
                if trace is not None:
                    trace.append(self._trace_row(state, ub, 'reject'))
                continue

            if trace is not None:
                trace.append(self._trace_row(state, ub, 'expand'))

            i = self.active_order[state.ell - 1]
            _vprint(f"  [bestfirst]   -> EXPAND row {i}")
            for opt in self.domains[i]:
                if opt.j in state.used:
                    _vprint(f"  [bestfirst]      skip {_fmt_opt(opt)}: destination {opt.j} already used")
                    continue
                if opt.cost > state.b:
                    _vprint(f"  [bestfirst]      skip {_fmt_opt(opt)}: no departure budget left")
                    continue
                child = _child_state(state, i, opt)
                ok, u = completion_bound(child, self.active_order, self.domains)
                if ok:
                    heapq.heappush(self._heap, (-u, self._serial, child))
                    self._serial += 1
                    _vprint(f"  [bestfirst]      push {_fmt_opt(opt)}: UB={float(u):.3f}")
                else:
                    _vprint(f"  [bestfirst]      prune {_fmt_opt(opt)}: some later row has no eligible option")

        _vprint(f"  [bestfirst] EXHAUSTED after {self.visited} pops")
        return 'Exhausted'

    @staticmethod
    def _trace_row(state, ub, outcome):
        """Builds one Section-5.5-style trace row for a popped state."""
        return {
            'ell': state.ell,
            'b': state.b,
            'used': frozenset(state.used),
            'choice': {i: (opt.j, opt.a) for i, opt in state.choice.items()},
            'rho': state.rho,
            'ub': ub,
            'outcome': outcome,
        }


def depth_first_repair(active_order, domains, s, r, build_word=None, keep_first=True,
                        zero_residual_shortcut=True):
    """
    Section 5.9 (depth-first repair as a comparison policy). Traverses the
    same feasible tree and residual invariant as RowEnumerationIterator,
    recursively, keeping only the current path (O(t) states) instead of a
    queue that can hold the tree's whole frontier. It returns the FIRST
    accepting leaf reached, not the highest-scoring one: unlike the
    best-first iterator, this gives no global score-order guarantee.

    :param active_order: the fixed list [i_1, ..., i_t] of active rows
    :param domains: {i: [RowOption, ...]}, as built by build_row_domains
    :param s: the target residual (e.g. H_prime * w_tilde)
    :param r: the departure radius, 0 <= r <= t
    :param build_word: optional callable choice -> w, as in
        RowEnumerationIterator.next_candidate
    :param keep_first: if True (the default), each row's keep option
        (cost 0) is tried before its other options, reproducing the
        earlier backtracking routine described in Section 5.9. If False,
        each row's options are tried in domains[i]'s own order (typically
        nonincreasing log score), matching the best-first iterator's
        per-row order, for a like-for-like comparison of traversal work.
    :param zero_residual_shortcut: if True (the default) and s is already
        zero, returns the all-keep assignment immediately with no search
        at all. Section 5.9 notes this shortcut must be disabled when
        comparing traversal work against the ranked iterator's own
        convention (the iterator has no such shortcut - it still builds
        and evaluates the root via CompletionBound).
    :return: (choice, w) for the first accepting leaf found, or None if
        the tree has no accepting leaf
    """
    if zero_residual_shortcut and s.is_zero():
        _vprint("  [depthfirst] s = 0 -> all-keep shortcut, no search")
        choice = {i: next(opt for opt in domains[i] if opt.cost == 0) for i in active_order}
        w = build_word(choice) if build_word is not None else None
        return choice, w

    t = len(active_order)
    order_cache = {}
    for i in active_order:
        opts = domains[i]
        if keep_first:
            keep = [opt for opt in opts if opt.cost == 0]
            rest = [opt for opt in opts if opt.cost != 0]
            order_cache[i] = keep + rest
        else:
            order_cache[i] = list(opts)

    n_leaves = [0]

    def _recurse(ell, b, used, rho, choice):
        pad = "  [depthfirst] " + "  " * ell
        if ell == t + 1:
            n_leaves[0] += 1
            if rho.is_zero():
                _vprint(f"{pad}leaf {_fmt_choice(choice)} -> ACCEPT (rho = 0)")
                return dict(choice)
            _vprint(f"{pad}leaf {_fmt_choice(choice)} -> reject (rho={rho})")
            return None
        i = active_order[ell - 1]
        for opt in order_cache[i]:
            if opt.j in used:
                _vprint(f"{pad}row {i}: skip {_fmt_opt(opt)} (destination used)")
                continue
            if opt.cost > b:
                _vprint(f"{pad}row {i}: skip {_fmt_opt(opt)} (no budget, b={b})")
                continue
            _vprint(f"{pad}row {i}: try {_fmt_opt(opt)}  b->{b - opt.cost}")
            choice[i] = opt
            found = _recurse(ell + 1, b - opt.cost, used | {opt.j}, rho - opt.sigma, choice)
            if found is not None:
                return found
            del choice[i]
        _vprint(f"{pad}row {i}: options exhausted -> backtrack")
        return None

    _vprint(f"  [depthfirst] start: active_order={list(active_order)}, r={r}, s={s}, keep_first={keep_first}")
    result = _recurse(1, r, frozenset(), s, {})
    _vprint(f"  [depthfirst] {'found' if result is not None else 'no'} accepting leaf "
            f"({n_leaves[0]} leaves reached)")
    if result is None:
        return None
    w = build_word(result) if build_word is not None else None
    return result, w


def structured_sd_repair_bestfirst(w_tilde, v, H_prime, Q_hat, posterior_table, F, r, L, v_max=None):
    """
    Best-first analogue of structured_sd_repair: repairs a predicted word
    by best-first search over active-row hypotheses (Algorithms 2-4)
    instead of an unranked enumeration over all size-<=r subsets of active
    rows. Because RowEnumerationIterator emits the feasible set Omega in
    nonincreasing log score (Proposition 5.1), the single candidate
    returned here is the maximum-log-score repair reachable within the
    departure radius r and the retained-option lists D_i - not merely the
    first one some fixed combinatorial order happened to reach.

    As with structured_sd_repair, every candidate this can return already
    has weight equal to wt(v) by construction (an accepted assignment is
    injective over exactly |Supp(v)| active rows, each contributing one
    nonzero coordinate at a distinct destination with a nonzero
    coefficient - Section 5.1), so no separate weight check is needed
    here; the caller should still run Verify/reconstruction downstream, as
    for any other candidate pair (Section 1.3).

    :param w_tilde: the predicted word, w_tilde = v * Q_hat
    :param v: the codeword v in C that produced w_tilde
    :param H_prime: a parity-check matrix of C'
    :param Q_hat: the monomial approximation of the secret Q
    :param posterior_table: the n x n posterior table from
        compute_posterior_table / compute_posterior_table_list_based
    :param F: the base field GF(q)
    :param r: the departure radius (called rmax for structured_sd_repair)
    :param L: the number of non-keep options retained per active row
        (passed to build_row_domains)
    :param v_max: an optional cap on the number of pops; None means
        unlimited. If the cap is hit before any accepting leaf is found,
        this returns None just like exhaustion - use iter_ranked_repairs
        directly if the Limit/Exhausted distinction matters to the caller
    :return: the repaired word w in C', or None if no accepting assignment
        was found (Omega is empty, or v_max was reached first)
    """
    target_weight = v.hamming_weight()
    n = Q_hat.ncols()

    s = H_prime * w_tilde
    _vprint(f"  syndrome s = H' * w_tilde = {s}")
    if s.is_zero():
        _vprint("  s = 0: w_tilde already in C' -> no repair needed "
                f"(weight {'ok' if w_tilde.hamming_weight() == target_weight else 'WRONG'})")
        return w_tilde if w_tilde.hamming_weight() == target_weight else None

    A = support(v)
    domains = build_row_domains(v, A, posterior_table, Q_hat, H_prime, F, L)
    active_order = sorted(A)

    it = RowEnumerationIterator(active_order, domains, s, r, v_max=v_max)
    result = it.next_candidate(build_word=lambda choice: build_word_from_choice(v, choice, n, F))
    if result in ('Exhausted', 'Limit'):
        _vprint(f"  [bestfirst] no repair ({result}) after {it.visited} pops")
        return None
    _choice, w, _g = result
    _vprint(f"  [bestfirst] repair after {it.visited} pops: choice={_fmt_choice(_choice)} "
            f"Lambda={float(_g):.3f} w={w}")
    return w


def iter_ranked_repairs(w_tilde, v, H_prime, Q_hat, posterior_table, F, r, L, v_max=None):
    """
    Generator form of structured_sd_repair_bestfirst, for a caller (e.g.
    the Section 12 coordinator) that wants to try several ranked
    candidates - in nonincreasing log score - against a downstream
    Verify/reconstruction step, rather than committing to only the single
    top-ranked one. Mirrors the "candidate consumption" pattern Section
    12.2 describes for a specified source of codewords, but at the level
    of one v's own repair candidates instead of across different v's.

    :param w_tilde: the predicted word, w_tilde = v * Q_hat
    :param v: the codeword v in C that produced w_tilde
    :param H_prime: a parity-check matrix of C'
    :param Q_hat: the monomial approximation of the secret Q
    :param posterior_table: the n x n posterior table from
        compute_posterior_table / compute_posterior_table_list_based
    :param F: the base field GF(q)
    :param r: the departure radius
    :param L: the number of non-keep options retained per active row
    :param v_max: an optional cap on the number of pops across the WHOLE
        generator's lifetime (shared by every yielded candidate, matching
        RowEnumerationIterator's own v_max semantics)
    :yield: (w, g) pairs in nonincreasing g order; stops (StopIteration)
        once Omega is exhausted or v_max is reached - callers that need to
        tell these two cases apart should drive a RowEnumerationIterator
        directly instead of using this generator
    """
    target_weight = v.hamming_weight()
    n = Q_hat.ncols()

    s = H_prime * w_tilde
    _vprint(f"  syndrome s = H' * w_tilde = {s}")
    if s.is_zero():
        _vprint("  s = 0: w_tilde already in C' -> yielding it as the only candidate")
        if w_tilde.hamming_weight() == target_weight:
            yield w_tilde, 0.0
        return

    A = support(v)
    domains = build_row_domains(v, A, posterior_table, Q_hat, H_prime, F, L)
    active_order = sorted(A)

    it = RowEnumerationIterator(active_order, domains, s, r, v_max=v_max)
    build_word = lambda choice: build_word_from_choice(v, choice, n, F)
    while True:
        result = it.next_candidate(build_word=build_word)
        if result in ('Exhausted', 'Limit'):
            return
        _choice, w, g = result
        _vprint(f"  [ranked] candidate: choice={_fmt_choice(_choice)} Lambda={float(g):.3f} w={w}")
        yield w, g


def structured_sd_repair_depth_first(w_tilde, v, H_prime, Q_hat, posterior_table, F, r, L,
                                      keep_first=True):
    """
    Depth-first analogue of structured_sd_repair_bestfirst (Section 5.9):
    same domains and departure-budget invariant, but traversed
    recursively with O(t) working memory and no score-order guarantee -
    useful as a low-memory comparison policy, not as a drop-in replacement
    for the ranked search.

    :param w_tilde: the predicted word, w_tilde = v * Q_hat
    :param v: the codeword v in C that produced w_tilde
    :param H_prime: a parity-check matrix of C'
    :param Q_hat: the monomial approximation of the secret Q
    :param posterior_table: the n x n posterior table from
        compute_posterior_table / compute_posterior_table_list_based
    :param F: the base field GF(q)
    :param r: the departure radius
    :param L: the number of non-keep options retained per active row
    :param keep_first: see depth_first_repair
    :return: the repaired word w in C', or None if the tree has no
        accepting leaf
    """
    target_weight = v.hamming_weight()
    n = Q_hat.ncols()

    s = H_prime * w_tilde
    _vprint(f"  syndrome s = H' * w_tilde = {s}")
    if s.is_zero():
        _vprint("  s = 0: w_tilde already in C' -> no repair needed "
                f"(weight {'ok' if w_tilde.hamming_weight() == target_weight else 'WRONG'})")
        return w_tilde if w_tilde.hamming_weight() == target_weight else None

    A = support(v)
    domains = build_row_domains(v, A, posterior_table, Q_hat, H_prime, F, L)
    active_order = sorted(A)

    result = depth_first_repair(active_order, domains, s, r,
                                 build_word=lambda choice: build_word_from_choice(v, choice, n, F),
                                 keep_first=keep_first)
    if result is None:
        return None
    _choice, w = result
    return w
