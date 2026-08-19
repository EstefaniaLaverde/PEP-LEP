"""
lee_brickell_batch_experiments.py

Strategy 02 (ISD syndrome-decoding prototypes): batch-experiment driver
for the standalone Lee-Brickell Information Set Decoder implemented in
this file (lee_brickel_ISD) - runs it over many random PEP/LEP instances
for a fixed (n, k) and reports success/failure statistics. This predates
the main prediction-and-repair framework (Code/core/) and its own
enumerators - it tests raw ISD decoding performance in isolation, not the
posterior-guided repair pipeline.

Used by run_batch_experiment_cli.py (local CLI) and
run_batch_experiments_on_modal.py (remote/parallel execution on Modal).

Using SageMath.
"""
import os
import sys

# instances_generator.py now lives in Code/core/ (this file was relocated
# into strategies/02_isd_syndrome_decoding_prototypes/ during the Aug 2026
# folder reorganization), so it needs to be added to sys.path explicitly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'core'))

from sage.all import *
from sage.coding.information_set_decoder import LinearCodeInformationSetDecoder
from instances_generator import *

import signal
import time


def lee_brickel_ISD(H, c, w, F, timeout=None):
    """
    Lee-Brickell Information Set Decoder using SageMath.
    """

    def timeout_handler(signum, frame):
        raise TimeoutError("The decoding process timed out.")

    c_vector = vector(F, c)

    C = codes.from_parity_check_matrix(H)

    chosen_p = min(2, w)

    D = C.decoder(
        "InformationSet",
        w,
        algorithm="Lee-Brickell",
        search_size=chosen_p,
    )

    signal.signal(signal.SIGALRM, timeout_handler)

    if timeout is not None:
        signal.alarm(int(timeout))

    try:
        estimated_codeword = D.decode_to_code(c_vector)
        e = c_vector - estimated_codeword
        return e

    except TimeoutError:
        return None

    finally:
        signal.alarm(0)


###############################################################################
# One decoding trial
###############################################################################

def run_trial(n, k, q=7, alpha=0.01, beta=0.2,):
    """
    Runs one randomly generated instance.
    Returns a dictionary containing all statistics.
    """

    G1, G2, P, P_noisy = generate_noisy_LCE_instance_CBA(
        n,
        k,
        q,
        alpha,
        beta,
        is_monomial=False,
    )

    H_tilde, vectorP_noisy = transform_problem_to_syndrome_decoding(
        G1,
        G2,
        P_noisy,
    )

    H_tilde_1, vectorP_noisy_1 = transform_problem_to_syndrome_decoding_H1(
        G1,
        G2,
        P_noisy,
    )

    vectorP = transform_secret_to_single_vector(P)

    F = GF(q)

    vectorP = vector(F, vectorP)
    vectorP_noisy = vector(F, vectorP_noisy)

    vectorE = vectorP_noisy - vectorP

    w = sum(1 for x in vectorE if x != 0)

    # Decode using H1 only
    start = time.time()

    e2 = lee_brickel_ISD(
        H_tilde_1,
        vectorP_noisy_1,
        w,
        F,
    )

    end2 = time.time()

    # Decode using full H
    e1 = lee_brickel_ISD(
        H_tilde,
        vectorP_noisy,
        w,
        F,
    )

    end1 = time.time()

    return {
        "n": n,
        "k": k,
        "q": q,
        "alpha": alpha,
        "beta": beta,
        "w": w,
        "full_H_tilde_size": (
            H_tilde.nrows(),
            H_tilde.ncols(),
        ),
        "H1_tilde_size": (
            H_tilde_1.nrows(),
            H_tilde_1.ncols(),
        ),
        "success_e1": (
            vectorE == e1
            if e1 is not None
            else False
        ),
        "success_e2": (
            vectorE == e2
            if e2 is not None
            else False
        ),
        "timeout_occurred_e1": e1 is None,
        "timeout_occurred_e2": e2 is None,
        "decoding_time_seconds_e1": end1 - start,
        "decoding_time_seconds_e2": end2 - start,
    }


# Several trials for ONE parameter set
def run_parameter_set(n, k, num_trials=10, q=7, alpha=0.01, beta=0.2,):
    """
    Runs several random instances for fixed (n,k).
    """

    results = []

    for i in range(num_trials):

        print(
            f"Trial {i+1}/{num_trials} "
            f"(n={n}, k={k})"
        )

        r = run_trial(
            n=n,
            k=k,
            q=q,
            alpha=alpha,
            beta=beta,
        )

        r["trial"] = i

        results.append(r)

    return results


# Parameter sweep
def run_parameter_sweep(n_list, k_list, num_trials=10, q=7, alpha=0.01, beta=0.2,):
    """
    Runs all parameter sets.
    """

    all_results = []

    for n, k in zip(n_list, k_list):

        all_results.extend(
            run_parameter_set(
                n=n,
                k=k,
                num_trials=num_trials,
                q=q,
                alpha=alpha,
                beta=beta,
            )
        )

    return all_results

if __name__ == "__main__":

    import pandas as pd

    n_list = [50, 100]
    k_list = [25, 50]

    results = run_parameter_sweep(
        n_list=n_list,
        k_list=k_list,
        num_trials=1,
    )

    df = pd.DataFrame(results)

    df.to_csv(
        "decoding_results_PEP_leebrickel.csv",
        index=False,
    )

    print(df)