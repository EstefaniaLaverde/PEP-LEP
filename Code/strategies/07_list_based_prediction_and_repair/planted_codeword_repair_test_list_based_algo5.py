"""
planted_codeword_repair_test_list_based_algo5.py

Second list-based counterpart of
strategies/06_planted_codeword_repair_isolation/planted_codeword_repair_test.py -
sibling to planted_codeword_repair_test_list_based.py, which already exists in this
folder. Both scripts plant the same low-weight codeword and leak the secret monomial
the same way (as two independent lists, see instances_generator_monomial_as_list.py);
they differ ONLY in how the monomial approximation Q_hat is reconstructed from the two
leaked lists:

    - planted_codeword_repair_test_list_based.py (existing) folds the two list
      posteriors into a single entrywise table (compute_posterior_table_list_based)
      and reuses monomial_approximation - the full-matrix MonomialApproximation,
      Algorithm 6 in the paper's current numbering.
    - This script instead calls compute_vector_posteriors_list_based +
      monomial_approximation_vector directly - the paper's actual Algorithm 5
      (MonomialApproximationVector, Section 5.5), which keeps the permutation and
      scale posteriors decoupled instead of recombining them into an entrywise table.
      See the comment above monomial_approximation_vector in
      core/LEP_prediction_and_repair_v2.py for why these two reconstructions are NOT
      mathematically equivalent (the entrywise recombination reintroduces a
      column-dependent zero-entry background term that Algorithm 5 has no counterpart
      for).

Kept as a SEPARATE file rather than editing planted_codeword_repair_test_list_based.py
in place, so the two reconstruction paths can be run and compared side by side, and so
the existing script's already-validated (numpy-mirrored) results stay untouched.

Everything else - planting the low-weight codeword into G1, leaking the two lists, and
the repair stage itself (posterior_aware_prange_repair) - is identical in structure to
planted_codeword_repair_test_list_based.py, and is exactly the point: Algorithm 5's
S_perm output (the permutation log-score matrix) has the same shape/role as the S
matrix Algorithm 6 produces, so posterior_aware_prange_repair / marginal_error_region
plug in completely unchanged. This is "so that the rest of the framework can be used
as it is."

IMPORTANT - validation note: like planted_codeword_repair_test_list_based.py, this
script requires SageMath and has NOT been run with `sage` in this session (Sage is
unavailable in the sandbox this file was written from). compute_vector_posteriors_list_based
and monomial_approximation_vector reuse build_bit_channel_matrix and hungarian_assignment
unchanged from the already-validated matrix-model code path (only the row-scoring and
matrix-assembly steps are new - see the module-level comment above
monomial_approximation_vector), so the delta needing fresh validation is small, but this
script itself still needs an actual `sage` run to confirm before relying on its output.

Using SageMath. Run with:
    sage planted_codeword_repair_test_list_based_algo5.py
(from the Code/ directory, so the sibling modules import correctly).
"""
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'core'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '05_full_parameter_sweep'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '06_planted_codeword_repair_isolation'))

from sage.all import GF, matrix, set_random_seed

from instances_generator_monomial_as_list import (
    generate_random_monomial_as_lists,
    generate_bit_channel_hint_for_list,
    monomial_lists_to_matrix,
)
from instances_generator import get_random_generator_matrix, obtain_parity_check_matrix
from LEP_prediction_and_repair_v2 import (
    compute_vector_posteriors_list_based,
    monomial_approximation_vector,
    posterior_aware_prange_repair,
    predict_image,
    predict_image_from_lists,
    active_error_set,
    set_verbose,
)
from run_lep_experiments import gilbert_varshamov_bound

# Reused unchanged from the whole-matrix version: planting a low-weight codeword into
# G1, and the run-persistence format, have nothing to do with how the secret monomial
# is later leaked or reconstructed.
from planted_codeword_repair_test import (
    build_generator_with_planted_low_weight_row,
    save_run,
)

# Reused unchanged from the OTHER list-based script: building the rest of the instance
# from a given G1 with list-based leakage is identical regardless of which
# reconstruction algorithm is used downstream - only the posterior/approximation step
# (below, in run_instance) differs.
from planted_codeword_repair_test_list_based import generate_noisy_LCE_instance_from_G1_list_based

set_verbose(True)

SAVE_RUN = True
RUNS_DIR = 'runs'


# ---------------------------------------------------------------------------
# Step 5 + repair stage: identical in structure to run_instance in
# planted_codeword_repair_test_list_based.py, except Q_hat and the score matrix fed to
# posterior_aware_prange_repair come from Algorithm 5 (monomial_approximation_vector)
# instead of Algorithm 6 (monomial_approximation) via the entrywise-table detour.
# ---------------------------------------------------------------------------

def run_instance(n, k, q, alpha, beta, weight_factor=1.1, rmax=None, budgets=3,
                  seed=20260813, label=""):
    """
    Same as planted_codeword_repair_test_list_based.run_instance, except the monomial
    approximation is built with compute_vector_posteriors_list_based +
    monomial_approximation_vector (Algorithm 5, the paper's actual two-vector
    reconstruction) instead of compute_posterior_table_list_based + monomial_approximation
    (Algorithm 6 via the entrywise-table detour). See that function's docstring for
    parameters not listed here.
    """
    set_random_seed(seed)
    F = GF(q)
    print(f"\n{'=' * 70}\n{label} n={n} k={k} q={q} alpha={alpha} beta={beta} seed={seed} "
          f"[list-based, Algorithm 5]\n{'=' * 70}")

    gv_bound = gilbert_varshamov_bound(n, k, q)
    target_weight = max(1, round(gv_bound * weight_factor))
    print(f"  Gilbert-Varshamov bound: {gv_bound}; planted target_weight = {target_weight}")

    t0 = time.time()
    G1, v0 = build_generator_with_planted_low_weight_row(n, k, q, target_weight)
    print(f"  planted-row generator matrix built ({time.time() - t0:.2f}s); "
          f"v0 weight = {v0.hamming_weight()}")

    t0 = time.time()
    G2, Q, permutation, values, noisy_permutation, noisy_values = \
        generate_noisy_LCE_instance_from_G1_list_based(G1, q, alpha, beta, is_monomial=True)
    print(f"  list-based instance generated ({time.time() - t0:.2f}s)")

    t0 = time.time()
    p_perm, p_scale = compute_vector_posteriors_list_based(
        noisy_permutation, noisy_values, n, q, alpha, beta, is_permutation=False
    )
    Q_hat, S_perm, d_hat, pi_hat = monomial_approximation_vector(p_perm, p_scale, F)
    rows_full = sum(1 for i in range(n) if list(Q_hat[i]) == list(Q[i]))
    print(f"  Q_hat built via Algorithm 5 ({time.time() - t0:.2f}s); "
          f"{rows_full}/{n} rows exactly correct")

    H2 = obtain_parity_check_matrix(G2)

    v = v0
    r_v = len(active_error_set(v, Q, Q_hat))

    margin = 4
    derived_rmax = -(-(r_v + margin) // 2)  # ceil((r_v + margin) / 2)
    if rmax is None:
        rmax = derived_rmax
        print(f"  true active error dimension r(v) = {r_v}; "
              f"derived rmax = {rmax} (tau = {2 * rmax}, margin = {margin} above r(v))")
    else:
        print(f"  true active error dimension r(v) = {r_v}; using requested "
              f"rmax = {rmax} (tau = {2 * rmax})")
        if 2 * rmax < r_v:
            print(f"  WARNING: tau = {2 * rmax} is below r(v) = {r_v} - "
                  f"the true error vector cannot be found at this radius, "
                  f"repair will fail deterministically; consider rmax >= "
                  f"{derived_rmax}")

    # predict_image_from_lists(v, pi_hat, d_hat, F) is Remark 1's action-based shortcut -
    # equivalent to predict_image(v, Q_hat), computed here instead to demonstrate the
    # full list-based path end to end without ever materializing Q_hat for this step
    # (Q_hat is still built above for the ground-truth row comparison and
    # active_error_set, which are only available in this controlled test).
    w_tilde = predict_image_from_lists(v, pi_hat, d_hat, F)
    assert w_tilde == predict_image(v, Q_hat), \
        "predict_image_from_lists must agree with predict_image(v, Q_hat)"
    tau = 2 * rmax

    # posterior_aware_prange_repair takes Q_hat and a score matrix S used only by
    # marginal_error_region to rank each active row's alternative columns
    # (Algorithm 11 / MarginalErrorRegion) - S_perm[i][j] = log p_i^perm(j) plays
    # exactly that role, unchanged, since it is scored over the same n candidate
    # columns per active row as the S matrix Algorithm 6 would have produced.
    repair_algorithm = 'posterior_aware_prange_repair (Algorithm 5 reconstruction)'
    t0 = time.time()
    w = posterior_aware_prange_repair(w_tilde, v, H2, Q_hat, S_perm, k, tau, K_B=budgets,
                                       max_trials=1000000)
    elapsed = time.time() - t0

    if w is None:
        print(f"  [{repair_algorithm}] FAILED to repair ({elapsed:.2f}s)")
        outcome_tag = 'failed'
    else:
        is_correct = (w == v * Q)
        outcome = 'CORRECT equivalent pair' if is_correct else 'weight/syndrome-consistent but WRONG pair'
        print(f"  [{repair_algorithm}] repaired in {elapsed:.2f}s -> {outcome}")
        outcome_tag = 'correct' if is_correct else 'wrong'

    if SAVE_RUN:
        params = {
            'n': n, 'k': k, 'q': q, 'alpha': alpha, 'beta': beta,
            'weight_factor': weight_factor, 'budgets': budgets, 'seed': seed,
            'target_weight': target_weight, 'representation': 'list_based_algorithm5',
        }
        # save_run's 5th positional argument is labeled 'Q_noisy' in the saved dict, but
        # it is just saved via Sage's save() as whatever picklable object is passed in -
        # there is no whole-matrix noisy hint in the list-based representation, so the
        # two leaked lists (plus the decoupled posteriors' argmax outputs) are passed
        # through as a dict instead of being reshaped into a (semantically wrong) matrix.
        noisy_lists = {
            'noisy_permutation': noisy_permutation,
            'noisy_values': noisy_values,
            'pi_hat': pi_hat,
            'd_hat': d_hat,
        }
        save_run(
            params, G1, G2, Q, noisy_lists, v, Q_hat, w_tilde, w,
            r_v, rmax, tau, elapsed, outcome_tag,
        )

    return {'v': v, 'r_v': r_v, 'outcome': outcome_tag}


def main():
    seeds = [i for i in range(5, 15)]
    for seed in seeds:
        run_instance(
            n=252, k=126, q=127, alpha=0.01, beta=0.03,
            weight_factor=1.1, rmax=20, budgets=10,
            label="[requested instance, list-based, Algorithm 5]", seed=seed
        )


if __name__ == "__main__":
    main()
