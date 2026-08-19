# %% [markdown]
# # PEP transformation to syndrome-decoding tests with the Lee-Brickell Information Set decoder
# (note: reworded from the original "...syndrome decoding: tests..." - that
# exact substring is misread by Python's PEP 263 encoding-cookie detector as
# a file-encoding declaration ("coding: tests"), which raised a SyntaxError
# any time this script was run directly with `python`/`sage` rather than
# through Jupyter - a real, pre-existing bug unrelated to the folder move)

# %%
import os
import sys

# instances_generator.py now lives in Code/core/ (this file was relocated
# into strategies/02_isd_syndrome_decoding_prototypes/ during the Aug 2026
# folder reorganization), so it needs to be added to sys.path explicitly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'core'))

from sage.coding.information_set_decoder import LinearCodeInformationSetDecoder
from instances_generator import *
import random
import signal
import time
import pandas as pd

# %% [markdown]
# ## Test PEP transformation to syndrome decoding with Lee-Brickell Information Set decoder

# %%
n = 7  # length of the code
k = 3   # dimension of the code
q = 7   # size of the finite field
alpha = 0.01  # probability of flipping 0 to a random element
beta = 0.2   # probability of flipping a random element to 0

# Generate a noisy PEP instance
G1, G2, P, P_noisy = generate_noisy_LCE_instance_CBA(n, k, q, alpha, beta, is_monomial=False)

# Transform the problem to a syndrome decoding problem
H_tilde, vectorP_noisy = transform_problem_to_syndrome_decoding(G1, G2, P_noisy)

vectorP = transform_secret_to_single_vector(P)

print("Generator matrix G1:")
print(G1)
print("\nGenerator matrix G2:")
print(G2)
print("\nSecret permutation matrix P:")
print(P)
print("\nNoisy hint for P:")
print(P_noisy)

print("\nParity-check matrix H_tilde:")
print(H_tilde)
print("\nNoisy hint vector for P:")
print(vectorP_noisy)

# %% [markdown]
# ## ISD Lee-Brickell Information Set decoder using SageMath

# %%
def lee_brickel_ISD(H, c, w, F, timeout=None):
    """
    Lee-Brickell Information Set Decoder (ISD) implementation using SageMath.

    Parameters:
    H : Matrix
        Parity check matrix of the linear code.
    c : Vector
        Received vector (noisy codeword).
    w : int
        Expected Hamming weight of the error.
    F : FiniteField
        Finite field over which the code is defined.
    timeout : int, optional
        Maximum time allowed for the decoding process in seconds (default is 600).

    Returns:
    e : Vector or None
        Estimated error vector, or None if a timeout occurs.
    """
    # Define the internal handler that raises an exception when the alarm rings
    def timeout_handler(signum, frame):
        raise TimeoutError("The decoding process timed out.")

    # transform the input vector to a Sage vector over the finite field
    c_vector = vector(F, c)

    # create a linear code object from the parity check matrix
    C = codes.from_parity_check_matrix(H)

    # initialize the Lee-Brickell decoder with the code and the expected weight
    chosen_p = min(2, w)
    D2 = C.decoder('InformationSet', w, algorithm='Lee-Brickell', search_size=chosen_p)

    # Register the timeout signal handler
    signal.signal(signal.SIGALRM, timeout_handler)
    # Start the countdown timer (expects seconds as an integer)
    if timeout is not None:
        signal.alarm(int(timeout))

    try:
        # decode the received vector to get the estimated codeword
        estimated_codeword = D2.decode_to_code(c_vector)
        
        # extract the estimated error vector
        e = c_vector - estimated_codeword
        return e

    except TimeoutError:
        return None

    finally:
        # disable the alarm so it doesn't interrupt subsequent lines of code
        signal.alarm(0)

# %% [markdown]
# ### Single Instance Test

# %%
n = 7  # length of the code
k = 3   # dimension of the code
q = 7   # size of the finite field
alpha = 0.01  # probability of flipping 0 to a random element
beta = 0.2   # probability of flipping a random element to 0

# Generate a noisy PEP instance
G1, G2, P, P_noisy = generate_noisy_LCE_instance_CBA(n, k, q, alpha, beta, is_monomial=False)

# Transform the problem to a syndrome decoding problem
H_tilde, vectorP_noisy = transform_problem_to_syndrome_decoding(G1, G2, P_noisy)
vectorP = transform_secret_to_single_vector(P)

# Set up the finite field and the received vector for decoding
F = GF(q)
vectorP = vector(F, vectorP)
vectorP_noisy = vector(F, vectorP_noisy)

# TODO: Aproximate the weight of the error vector
vectorE = vectorP_noisy - vectorP
w = sum(1 for x in vectorE if x != 0)

# Decode using the Lee-Brickell ISD algorithm
e = lee_brickel_ISD(H_tilde, vectorP_noisy, w, F)

print("Success:", vectorE == e)

# %% [markdown]
# ### Test with multiple n and k values

# %%
n_list = [7, 20, 30, 50, 100]
k_list = [3, 10, 15, 25, 50]
q = 7   # size of the finite field
alpha = 0.01  # probability of flipping 0 to a random element
beta = 0.2   # probability of flipping a random element to 0

# %%
all_results = []
for n, k in zip(n_list, k_list):
    print(f"\nTesting with n={n}, k={k}")
    # Generate a noisy PEP instance
    G1, G2, P, P_noisy = generate_noisy_LCE_instance_CBA(n, k, q, alpha, beta, is_monomial=False)

    # Transform the problem to a syndrome decoding problem
    H_tilde, vectorP_noisy = transform_problem_to_syndrome_decoding(G1, G2, P_noisy)
    vectorP = transform_secret_to_single_vector(P)

    # Set up the finite field and the received vector for decoding
    F = GF(q)
    vectorP = vector(F, vectorP)
    vectorP_noisy = vector(F, vectorP_noisy)

    # Decode using the Lee-Brickell ISD algorithm
    start_time = time.time()
    e = lee_brickel_ISD(H_tilde, vectorP_noisy, w, F)
    end_time = time.time()
    if e is not None:
        print("Success:", vectorP_noisy - vectorP == e)
    else:
        print("Decoding failed due to timeout.")
    print(f"Decoding time: {end_time - start_time:.2f} seconds")

    # write results to a .csv file
    results = {
        "n": n,
        "k": k,
        "q": q,
        "w": w,
        "alpha": alpha,
        "beta": beta,
        "H_tilde_size": (H_tilde.nrows(), H_tilde.ncols()),
        "success": vectorP_noisy - vectorP == e if e is not None else False,
        "decoding_time_seconds": end_time - start_time,
        "timeout_occurred": e is None
    }
    all_results.append(results)

    # Save results to a CSV file
    df = pd.DataFrame(all_results)
    df.to_csv("decoding_results_PEP_leebrickel.csv", index=False)
    
    