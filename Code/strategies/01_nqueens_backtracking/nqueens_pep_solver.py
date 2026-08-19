"""
nqueens_pep_solver.py  (Strategy 01: N-Queens backtracking)

Solve Permutation Equivalence Problem using an algorithm inspired on the n-queens problem.

Status: early exploratory approach, standalone (does not depend on
Code/core/) - a backtracking search rather than the posterior-guided
prediction-and-repair framework used by the later strategies. Its
parameter sweep lives in nqueens_parameter_sweep.ipynb in this same
folder.

Main Idea:
- An nxn permutation matrix has exactly one 1 in each row and column, and 0s eslewhere. It is similar to the n-queens problem, except the diagonals are not a problem.
- Use backtracking to place all the 1s in the matrix, ensuring that no two 1s are in the same row or column.
- The function free(i,j) checks if the column i in G1 and the column j in G2 are the same and if the position is not already assigned in the current solution.
- Hints act as pre-assigned positions for the 1s.
"""
import numpy as np
import time

def generate_test_matrices(k: int, n: int, q: int, num_hints: int, seed: int = None):
    """
    Generate random test matrices G1 and G2, permutation mapping and hints from permutation.

    Parameters:
    - k: number of rows
    - n: number of columns
    - q: size of the finite field (e.g., 2 for binary)
    - num_hints: number of hints to provide (columns of G1 with known mapping in G2)
    - seed: random seed for reproducibility

    Returns:
    - G1: random k x n matrix over F_q
    - G2: permuted version of G1 according to a random permutation
    - true_p: the actual permutation used to create G2 from G1
    - hints: an array of length n where hints[i] is the column in G2 that corresponds to column i in G1, or -1 if no hint is given for that column
    """
    if seed is not None:
        np.random.seed(seed)
    
    if num_hints > n:
        num_hints = n

    # generate G1 matrix with random values in F_q
    G1 = np.random.randint(0, q, size=(k, n))
    
    # generate a random permutation of columns for G2
    true_p = np.random.permutation(n)
    
    # use the permutation to create G2 from G1
    G2 = np.zeros((k, n), dtype=int)
    for col_g1, col_g2 in enumerate(true_p):
        G2[:, col_g2] = G1[:, col_g1]
        
    # select num_hints random columns to reveal from the true permutation
    # if the column is revealed, hints[i] = true_p[i], otherwise hints[i] = -1
    hints = np.full(n, -1)
    if num_hints > 0:
        indices_to_reveal = np.random.choice(n, num_hints, replace=False)
        for idx in indices_to_reveal:
            hints[idx] = true_p[idx]
            
    return G1, G2, true_p, hints

def solve_pep(G1: np.ndarray, G2: np.ndarray, hints: np.ndarray, max_time: int = 60):
    """
    Solve the Permutation Equivalence Problem using a backtracking approach with hints, inspired by the n-queens problem.

    Parameters:
    - G1: k x n matrix over F_q
    - G2: k x n matrix over F_q, permuted version of G1
    - hints: array of length n with known column mappings (-1 if unknown)
    - max_time: maximum time allowed for backtracking (optional, can be used to limit search)

    Returns:
    - solution: array of length n with the column mapping from G1 to G2, or None if no solution is found
    """
    # select dimension of the problem and intialize solution and tracking structures
    n = G1.shape[1] # number of columns
    solution = np.full(n, -1, dtype=int) # solution[i] = j means column i of G1 maps to column j of G2
    assigned_in_g2 = [False] * n # array of size n to track which columns in G2 have been assigned in the current solution

    # precomute candidates for each column in G1. This is a list of lists, where candidates[i] contains the indices of columns in G2 that are identical to column i in G1.
    candidates = []
    for i in range(n):
        matches = [j for j in range(n) if np.array_equal(G1[:, i], G2[:, j])]
        candidates.append(matches)

    def backtrack(col_g1):
        # if every column in G1 has been assigned, we found a solution
        if col_g1 == n:
            return True
        
        # if max_time is specified and we have reached it, stop the search
        if max_time is not None and (time.time() - start_time) >= max_time:
            return False
        
        # determine possible columns in G2 for the current column in G1, based on hints and candidates
        # if there is a hint for this column, we only consider that specific column in G2. Otherwise, we consider all candidates that match the column in G1
        possible_cols = [hints[col_g1]] if hints[col_g1] != -1 else candidates[col_g1]
        

        for target_g2 in possible_cols:
            # for every possible target column in G2, check if it is not already assigned in the current solution. If it is free, assign it and continue to the next column in G1. If we find a solution, return True. Otherwise, backtrack by unassigning the target column in G2 and trying the next candidate.
            if not assigned_in_g2[target_g2] and target_g2 in candidates[col_g1]:
                solution[col_g1] = target_g2
                assigned_in_g2[target_g2] = True
                
                if backtrack(col_g1 + 1):
                    return True
                
                assigned_in_g2[target_g2] = False
                solution[col_g1] = -1
        return False

    start_time = time.time()
    return solution if backtrack(0) else None

if __name__ == "__main__":
    # test
    K, N = 10, 400 # generator matrices dimension
    Q = 3 # field size
    N_HINTS = 10 # number of hints to provide to the algorithm
    SEED = 10 # seed for reproducibility

    # generate matrices
    G1, G2, real_p, hints_vector = generate_test_matrices(K, N, Q, N_HINTS, SEED)
    print(G1)
    print(hints_vector)

    start = time.time()
    found_p = solve_pep(G1, G2, hints_vector)
    end = time.time()

    if found_p is not None:
        unique_columns, counts = np.unique(G1, axis=1, return_counts=True)
        n_unique = unique_columns.shape[1]
        all_unique = (n_unique == N)
        print("All columns unique?", all_unique)
        print("Valid solution:", np.all(real_p == found_p))
        print(f"Time taken: {end - start:.4f} seconds")