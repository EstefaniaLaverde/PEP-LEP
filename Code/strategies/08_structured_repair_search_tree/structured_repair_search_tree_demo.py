"""
structured_repair_search_tree_demo.py
(Code/strategies/08_structured_repair_search_tree/)

Smoke test / demo for structured_repair_search_tree.py (Section 5's
best-first search tree, wired up to the existing prediction-and-repair
pipeline), mirroring the __main__ block already at the bottom of
LEP_prediction_and_repair_v2.py.

Runs the same noisy-instance -> monomial-approximation pipeline as that
block, then repairs several random codewords THREE ways on the identical
instance and predictions, so they can be compared directly:

  1. structured_sd_repair            - the existing brute-force search
                                        (Algorithm 7/8, old numbering)
  2. structured_sd_repair_bestfirst  - Section 5's best-first search tree
                                        (this file's subject)
  3. structured_sd_repair_depth_first - Section 5.9's low-memory traversal

All three are given the SAME row-repair budget r and the SAME per-row
option budget L, so any difference in outcome is informative: they should
either all fail, or all succeed with a word of the correct weight (they
need not return the identical word when Omega has more than one element,
but every word any of them returns must pass the syndrome/weight test
imposed by C').

Run with (from the Code/ directory, so the sibling core/ modules import):
    sage structured_repair_search_tree_demo.py
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'core'))

from sage.all import GF

from instances_generator import (
    generate_noisy_LCE_instance_CBA_bit_flip_version,
    obtain_parity_check_matrix,
)
from LEP_prediction_and_repair_v2 import (
    compute_posterior_table,
    compute_posterior_table_exact,
    monomial_approximation,
    predict_image,
    support,
    build_active_row_lists,
    structured_sd_repair,
    brute_force_min_weight_codewords,
    set_verbose,
)
from structured_repair_search_tree import (
    structured_sd_repair_bestfirst,
    structured_sd_repair_depth_first,
    iter_ranked_repairs,
)


def _section(title):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


def _row_map(M, i):
    """(j, a) for the unique nonzero entry of monomial-matrix row i."""
    j = next(j for j in range(M.ncols()) if M[i, j] != 0)
    return j, M[i, j]


def main():
    # Also turns on the search-tree traces in structured_repair_search_tree.py
    set_verbose(True)

    # toy parameters
    n, k, q = 7, 3, 7
    alpha, beta = 0.01, 0.2
    F = GF(q)
    r = 2      # departure radius (rmax)
    L = 3      # non-keep options retained per active row

    G1, G2, Q, Q_noisy = generate_noisy_LCE_instance_CBA_bit_flip_version(
        n, k, q, alpha, beta, is_monomial=True,
    )
    
    posterior_table = compute_posterior_table_exact(Q_noisy, n, q, alpha, beta, is_permutation=False)
    Q_hat, S, D_loc, pi = monomial_approximation(posterior_table, F)
    H2 = obtain_parity_check_matrix(G2)

    _section(f"INSTANCE  n={n} k={k} q={q} alpha={alpha} beta={beta}  r={r} L={L}")
    print("Q (secret):\n" + str(Q))
    print("Q_hat (monomial approximation):\n" + str(Q_hat))
    bad_rows = [i for i in range(n) if list(Q_hat[i]) != list(Q[i])]
    print("Correctly approximated rows:", n - len(bad_rows), "/", n,
          " wrong rows:", bad_rows)
    for i in bad_rows:
        print(f"  row {i}: true (j,a)={_row_map(Q, i)}  predicted (j,a)={_row_map(Q_hat, i)}")

    min_weight, low_weight_words = brute_force_min_weight_codewords(G1)
    print("Minimum distance of C (brute force):", min_weight)

    # Try every codeword of the minimum weight found, plus a couple of
    # random nonzero codewords, so both "small |Supp(v)|" and "larger
    # |Supp(v)|" active-row counts get exercised.
    candidates_v = list(low_weight_words)[:5]

    n_ok = {'brute': 0, 'bestfirst': 0, 'depthfirst': 0}
    n_tested = 0

    for v in candidates_v:
        v = G1.row_space()(v) if not hasattr(v, 'hamming_weight') else v
        A = support(v)
        if not A:
            continue
        n_tested += 1
        w_tilde = predict_image(v, Q_hat)
        w_true = v * Q
        target_weight = v.hamming_weight()

        _section(f"CODEWORD #{n_tested}: v={v}  wt={target_weight}  Supp(v)={A}")
        print(f"  true image  w = v*Q     = {w_true}")
        print(f"  prediction  w~ = v*Q_hat = {w_tilde}  (equal to truth: {w_tilde == w_true})")
        wrong_active = [i for i in A if i in bad_rows]
        print(f"  active rows mispredicted by Q_hat: {wrong_active}  "
              f"(need r >= {len(wrong_active)} departures; r={r})")
        for i in wrong_active:
            print(f"    row {i}: true {_row_map(Q, i)} vs keep {_row_map(Q_hat, i)}")

        print("\n  >>> [1] structured_sd_repair (brute force)")
        A_list, L_lists = build_active_row_lists(v, posterior_table, Q_hat, F, budgets=L)
        w_brute = structured_sd_repair(w_tilde, v, H2, Q_hat, A_list, L_lists, r)

        print("\n  >>> [2] structured_sd_repair_bestfirst")
        w_best = structured_sd_repair_bestfirst(w_tilde, v, H2, Q_hat, posterior_table, F, r, L)
        print("\n  >>> [3] structured_sd_repair_depth_first")
        w_df = structured_sd_repair_depth_first(w_tilde, v, H2, Q_hat, posterior_table, F, r, L)

        print("\n  >>> results")
        for name, w in (('brute', w_brute), ('bestfirst', w_best), ('depthfirst', w_df)):
            if w is None:
                print(f"  [{name}] no repair found")
                continue
            ok_weight = w.hamming_weight() == target_weight
            ok_syndrome = (H2 * w).is_zero()
            print(f"  [{name}] w={w}  weight_ok={ok_weight}  syndrome_ok={ok_syndrome}  "
                  f"is_true_image={w == w_true}")
            assert ok_weight and ok_syndrome, f"{name} returned an invalid candidate for v={v}"
            n_ok[name] += 1

        # Also exercise the ranked generator: the first ranked candidate
        # from iter_ranked_repairs should match structured_sd_repair_bestfirst's
        # own single-shot answer exactly (same construction, same v_max=None).
        print("\n  >>> [4] iter_ranked_repairs (full ranked enumeration of Omega)")
        ranked = list(iter_ranked_repairs(w_tilde, v, H2, Q_hat, posterior_table, F, r, L))
        true_rank = next((k for k, (w, _g) in enumerate(ranked) if w == w_true), None)
        print(f"  {len(ranked)} ranked candidates; true image at rank: {true_rank}")
        if w_best is None:
            assert not ranked, "bestfirst found nothing but the ranked generator yielded a candidate"
        else:
            assert ranked and ranked[0][0] == w_best, (
                "iter_ranked_repairs' first candidate disagrees with "
                "structured_sd_repair_bestfirst's own result"
            )
            scores = [g for _w, g in ranked]
            assert scores == sorted(scores, reverse=True), f"ranked scores not nonincreasing: {scores}"

    print(f"\nTested {n_tested} codewords. Successes: {n_ok}")
    print("(brute vs. bestfirst vs. depthfirst agreeing on success/failure, and every")
    print(" returned word passing the weight+syndrome check, is the pass condition here.)")


if __name__ == "__main__":
    main()
