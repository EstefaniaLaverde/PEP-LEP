"""
lep_prediction_and_repair_v1_superseded.py

ARCHIVED / SUPERSEDED - kept for history only, nothing else imports this.
Renamed from LEP_prediction_and_repair.py (the original, non-"_v2" file).
core/LEP_prediction_and_repair_v2.py is the current, actively-used
implementation of the prediction-and-repair pipeline; this v1 predates the
BBLM posterior formalization and the three-repair-algorithm split
(induced_sd_repair / structured_sd_repair / posterior_aware_prange_repair)
that v2 introduced.

Original docstring:
    This file contains the functions for the prediction and repair of the
    LEP problem using noisy hints.
"""

from sage.all import GF, random_matrix, random, shuffle, matrix
import math
from random import randint
import copy

# PHASE 1: build the monomial approximation Q_hat
def compute_channel_likelihood(true_value, observed_value, alpha, beta, q, bit_width = None):
    """
    Computes the likelihood of observing a value given the true value and the channel parameters.
    That is,
        P(observed_value | true_value)
    """
    if bit_width is None:
        # compute the bit_width
        bit_width = math.ceil(math.log2(q))

    probability = 1.0

    for k in range(bit_width):

        # extract the k-th bit of the true value and the observed value
        true_bit = (true_value >> k) & 1
        observed_bit = (observed_value >> k) & 1

        # the probability of the bit being 1 and the observed bit being 1 is (1 - alpha)
        if true_bit == 1 and observed_bit == 1:
            probability *= (1 - alpha)

        # the probability of a bit flip from 1 -> 0 (alpha)
        elif true_bit == 1 and observed_bit == 0:
            probability *= alpha

        # the probability of the bit being 0 and the observed bit being 0 is (1 - beta)
        elif true_bit == 0 and observed_bit == 0:
            probability *= (1 - beta)

        # the probability of a bit flip from 0 -> 1 (beta)
        else:
            probability *= beta

    return probability
    
def build_entry_posteriors_from_hint(Q_noisy, alpha, beta, q, bit_width = None):
    """
    Computes the posterior probabilities

        P[i][j][a] = Pr(Q_ij = a | Q_noisy_ij)

    for every matrix position (i,j) and every field element a in GF(q),

    The output for each position would be something like:
    Pr(Q[i,j]=0 | obs)
    Pr(Q[i,j]=1 | obs)
    ...
    Pr(Q[i,j]=q-1 | obs)

    The output is a 3D list of size n_rows x n_cols x q, where n_rows 
    and n_cols are the dimensions of Q_noisy and q is the size of the field GF(q).
    """
    
    if bit_width is None:
        # compute the bit_width
        bit_width = math.ceil(math.log2(q))

    n_rows = Q_noisy.nrows()
    n_cols = Q_noisy.ncols()

    posteriors = []

    for i in range(n_rows):

        row_posteriors = []

        for j in range(n_cols):

            obs = int(Q_noisy[i, j])

            posterior = []

            normalization_constant = 0.0

            # Compute unnormalized posteriors
            for a in range(q):

                likelihood = compute_channel_likelihood(
                    true_value=a,
                    observed_value=obs,
                    alpha=alpha,
                    beta=beta,
                    q=q,
                    bit_width=bit_width
                )

                posterior.append(likelihood)
                normalization_constant += likelihood

            # Normalize
            posterior = [
                p / normalization_constant
                for p in posterior
            ]

            row_posteriors.append(posterior)

        posteriors.append(row_posteriors)

    return posteriors

def compute_zero_scores(posteriors):
    """
    Computes the quantity:
        z_i = sum_{j=1} log(pij(0))
    """
    Z_scores = []

    # iterate over the rows of the posterior probabilities. The rows correspond to the rows of the matrix Q_noisy.
    for i in range(len(posteriors)):

        score = 0.0
        # for each column j in the row i, we compute the log of the posterior probability of the entry being 0 and sum them up.
        for j in range(len(posteriors[i])):

            score += math.log(posteriors[i][j][0])

        Z_scores.append(score)

    return Z_scores

def compute_best_nonzero_values(posteriors, q):
    """
    Computes for every possible position (i,j) in the noisy matrix, the value:
        Dlocal[i,j] = argmax_{a \neq 0} p_{ij}(a)

    This is, if (i,j) were the nonzero entry of the monomial matrix, which field
    element would be the most likely value.
    """

    n_rows = len(posteriors)
    n_cols = len(posteriors[0])

    D_local = []

    for i in range(n_rows):

        row = []

        for j in range(n_cols):

            best_value = 1
            best_probability = posteriors[i][j][1]

            for a in range(2, q):

                if posteriors[i][j][a] > best_probability:
                    best_probability = posteriors[i][j][a]
                    best_value = a

            row.append(best_value)

        D_local.append(row)

    return D_local

def compute_local_score_matrix(posteriors, D_local, Z_scores):
    """
    Computes the local score matrix S, where each entry S[i,j] is given by:
        S[i,j] = log(p_{ij}(D_local[i,j])) - log(p_{ij}(0)) + Z_scores[i]

    This score represents the log-likelihood that row i has its unique nonzero entry in column j.
    """

    n_rows = len(posteriors)
    n_cols = len(posteriors[0])

    S = []

    for i in range(n_rows):

        row_scores = []

        for j in range(n_cols):

            nonzero_value = D_local[i][j]

            score = (
                math.log(posteriors[i][j][nonzero_value])
                + Z_scores[i]
                - math.log(posteriors[i][j][0])
            )

            row_scores.append(score)

        S.append(row_scores)

    return S


