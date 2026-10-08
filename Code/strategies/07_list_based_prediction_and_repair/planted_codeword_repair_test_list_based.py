"""
planted_codeword_repair_test_list_based.py

List-based counterpart of
strategies/06_planted_codeword_repair_isolation/planted_codeword_repair_test.py:
same planted-low-weight-codeword construction and the same repair stage
(posterior_aware_prange_repair), but the secret monomial is generated and
leaked with the list-based representation
(instances_generator_monomial_as_list.generate_random_monomial_as_lists /
generate_bit_channel_hint_for_list) instead of a whole matrix, and the
posterior is built with the new compute_posterior_table_list_based instead
of compute_posterior_table. See
strategies/07_list_based_prediction_and_repair/list_based_posterior_pseudocode.md
for the pseudocode relating the two posterior constructions.

Steps 1-3 (planting the low-weight codeword into G1) and the repair stage
are IDENTICAL to the whole-matrix version - build_generator_with_planted_low_weight_row,
random_weight_vector and save_run are imported directly from
planted_codeword_repair_test.py rather than duplicated, since none of that
logic depends on how the secret monomial ends up leaked. Only step 4 (build
the rest of the instance from G1) and the posterior-table call change.

IMPORTANT - validation note: this script requires SageMath (as does the
whole-matrix version it mirrors) and has NOT been run with `sage` in this
session - Sage is unavailable in the sandbox this file was written from
(confirmed via `import sage.all` failing both locally and on the machine
this repo lives on). Before relying on this script, the pipeline it
implements was instead validated end to end with an independent numpy/scipy
reimplementation of every function used here (GF(q) linear algebra,
compute_posterior_table_list_based, monomial_approximation,
posterior_aware_prange_repair, and this same planted-codeword construction),
run over 10 random seeds at n=16, k=8, q=29 (matching the small instance
the whole-matrix script itself keeps as a commented-out sanity check) plus
one instance at n=64, k=32, q=127 - see the pseudocode .md file's Section 6
for a summary and where that validation code lives. This script itself
still needs an actual `sage` run (e.g. `sage planted_codeword_repair_test_list_based.py`)
to confirm the real Sage objects behave identically to their numpy mirror
at the parameter scale used by main() below.

Using SageMath. Run with:
    sage planted_codeword_repair_test_list_based.py
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
    compute_posterior_table_list_based,
    monomial_approximation,
    posterior_aware_prange_repair,
    predict_image,
    active_error_set,
    set_verbose,
)
from run_lep_experiments import gilbert_varshamov_bound

# Reused unchanged from the whole-matrix version: planting a low-weight
# codeword into G1 has nothing to do with how the secret monomial is later
# leaked, so this logic (and save_run's persistence format) is imported
# directly rather than duplicated.
from planted_codeword_repair_test import (
    build_generator_with_planted_low_weight_row,
    save_run,
)

set_verbose(True)

SAVE_RUN = True
RUNS_DIR = 'runs'


# ---------------------------------------------------------------------------
# Step 4 (list-based): build the rest of the LEP instance from a *given* G1,
# using the list-based monomial representation and leakage instead of the
# whole-matrix hint - the list-based analog of
# planted_codeword_repair_test.generate_noisy_LCE_instance_from_G1, which is
# itself instances_generator.generate_noisy_LCE_instance_CBA_bit_flip_version's
# body with the "generate G1" step removed. This is
# instances_generator_monomial_as_list.generate_noisy_LCE_instance_CBA_bit_flip_lists_version's
# body with the same step removed, so build_generator_with_planted_low_weight_row's
# output can be used directly.
# ---------------------------------------------------------------------------

def generate_noisy_LCE_instance_from_G1_list_based(G1, q, alpha, beta, is_monomial=True,
                                                     permutation_bit_width=None, values_bit_width=None):
    """
    List-based counterpart of generate_noisy_LCE_instance_from_G1.

    :param G1: a k x n generator matrix of C, over GF(q)
    :param q: field size
    :param alpha: Pr[bit 1 -> 0] in the bitwise leakage channel
    :param beta: Pr[bit 0 -> 1] in the bitwise leakage channel
    :param is_monomial: True for LEP (random monomial secret), False for PEP (random
        permutation secret)
    :param permutation_bit_width: (Optional) bits used to leak each permutation index,
        defaults to ceil(log2(n))
    :param values_bit_width: (Optional) bits used to leak each diagonal value, defaults to
        ceil(log2(q))
    :return: a tuple (G2, Q, permutation, values, noisy_permutation, noisy_values):
        - G2: the second generator matrix, as in the other instance generators
        - Q: the true secret monomial matrix (built only for ground-truth comparisons in this
          controlled test - the real attacker never sees this)
        - permutation, values: the true secret lists
        - noisy_permutation, noisy_values: the leaked lists, as produced by
          generate_bit_channel_hint_for_list
    """
    import math

    F = G1.base_ring()
    n = G1.ncols()

    permutation, values = generate_random_monomial_as_lists(n, q, is_permutation=not is_monomial)
    Q = monomial_lists_to_matrix(n, q, permutation, values)
    G2 = (G1 * Q).rref()

    if permutation_bit_width is None:
        permutation_bit_width = math.ceil(math.log2(n))
    if values_bit_width is None:
        values_bit_width = math.ceil(math.log2(q))

    noisy_permutation = generate_bit_channel_hint_for_list(permutation, alpha, beta, bit_width=permutation_bit_width)
    noisy_values = generate_bit_channel_hint_for_list(values, alpha, beta, bit_width=values_bit_width)

    return G2, Q, permutation, values, noisy_permutation, noisy_values


# ---------------------------------------------------------------------------
# Step 5 + repair stage: identical in structure to run_instance in the
# whole-matrix script, but building the instance and the posterior the
# list-based way.
# ---------------------------------------------------------------------------

def run_instance(n, k, q, alpha, beta, weight_factor=1.1, rmax=None, budgets=3,
                  seed=20260813, label=""):
    """
    Same as planted_codeword_repair_test.run_instance, except the secret
    monomial is generated and leaked as two lists, and the posterior is
    built with compute_posterior_table_list_based instead of
    compute_posterior_table. See that function's docstring for parameters
    not listed here.
    """
    set_random_seed(seed)
    F = GF(q)
    print(f"\n{'=' * 70}\n{label} n={n} k={k} q={q} alpha={alpha} beta={beta} seed={seed} [list-based]\n{'=' * 70}")

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
    posterior_table = compute_posterior_table_list_based(
        noisy_permutation, noisy_values, n, q, alpha, beta, is_permutation=False
    )
    Q_hat, S, D_loc, pi = monomial_approximation(posterior_table, F)
    rows_full = sum(1 for i in range(n) if list(Q_hat[i]) == list(Q[i]))
    print(f"  Q_hat built ({time.time() - t0:.2f}s); {rows_full}/{n} rows exactly correct")

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

    w_tilde = predict_image(v, Q_hat)
    tau = 2 * rmax

    repair_algorithm = 'posterior_aware_prange_repair'
    t0 = time.time()
    w = posterior_aware_prange_repair(w_tilde, v, H2, Q_hat, S, k, tau, K_B=budgets, max_trials=1000000)
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
            'target_weight': target_weight, 'representation': 'list_based',
        }
        # save_run's 5th positional argument is labeled 'Q_noisy' in the saved dict,
        # but it is just saved via Sage's save() as whatever picklable object is
        # passed in - there is no whole-matrix noisy hint in the list-based
        # representation, so the two leaked lists are passed through as a dict
        # instead of being reshaped into a (semantically wrong) matrix.
        noisy_lists = {'noisy_permutation': noisy_permutation, 'noisy_values': noisy_values}
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
            label="[requested instance, list-based]", seed=seed
        )


if __name__ == "__main__":
    main()
