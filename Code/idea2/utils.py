#!/usr/bin/env sage
from copy import deepcopy
from sage.all import zero_matrix, randint, GF, FiniteField, rank, floor, random_matrix
from random import random

# Utility checks and sampling functions remain unchanged...
def check_solution(Q, G1, G2):
    if not is_monomial(Q): return 0
    if G2.rref() != (G1 * Q).rref(): return 0
    return 1

def is_monomial(Q):
    if Q is None: return False
    for i in range(Q.nrows()):
        if sum(1 for j in range(Q.ncols()) if Q[i, j] != 0) != 1: return False
    for j in range(Q.ncols()):
        if sum(1 for i in range(Q.nrows()) if Q[i, j] != 0) != 1: return False
    return True

def sample_low_weight(n, q, w):
    a = [i for i in range(0, n)]
    for i in range(n - 1, n - 1 - w, -1):
        j = randint(0, i)
        a[i], a[j] = a[j], a[i]
    a = a[n - w:n]
    lw = zero_matrix(GF(q), 1, n)
    for x in range(w): lw[:, a[x]] = randint(1, q - 1)
    return lw

def sample_pair_low_weight(n, q, w, ell):
    values = [i for i in range(0, n)]
    a, b = [0]*w, [0]*w
    for i in range(ell):
        tmp = values.pop(randint(0, len(values)-1))
        a[i], b[i] = tmp, tmp
    for i in range(ell, w):
        a[i] = values.pop(randint(0, len(values)-1))
        b[i] = values.pop(randint(0, len(values)-1))
    lw1, lw2 = zero_matrix(GF(q), 1, n), zero_matrix(GF(q), 1, n)
    for x in range(w):
        lw1[:, a[x]] = randint(1, q - 1)
        lw2[:, b[x]] = randint(1, q - 1)
    return [lw1, lw2]

def sample_monomial_matrix(n, q):
    P = zero_matrix(FiniteField(q), n, n)
    a = [i for i in range(0, n)]
    for i in range(n-1, 0, -1):
        j = randint(0, i)
        a[i], a[j] = a[j], a[i]
    for i in range(0, n): P[i, a[i]] = randint(1, q-1)
    return P

def support_intersection(n, lw1, lw2):
    return sum(1 for i in range(n) if lw1[0, i] != 0 and lw2[0, i] != 0)

def sample_instance(n, k, q, w, N=2, ell=-1):
    if ell == -1:
        lw = [sample_low_weight(n, q, w) for _ in range(N)]
        ell = support_intersection(n, lw[0], lw[1])
        while ell < floor(w*w/n):
            lw = [sample_low_weight(n, q, w) for _ in range(N)]
            ell = support_intersection(n, lw[0], lw[1])
    else:
        lw = sample_pair_low_weight(n, q, w, ell)

    while True:
        G = lw[0]
        for i in range(1, N): G = G.stack(lw[i])
        G = G.stack(random_matrix(GF(q), k - N, n))
        if rank(G) == k: break
            
    Q = sample_monomial_matrix(n, q)
    G = G.rref()
    G_ = (G * Q).rref()
    pairs = [[lw[i], lw[i] * Q] for i in range(N)]
    return G, G_, Q, pairs

def sample_noisy_monomial(Q, q, alpha, beta):
    Q_noisy = deepcopy(Q)
    n = Q_noisy.nrows()
    F = GF(q)
    for i in range(n):
        for j in range(n):
            current_val = Q_noisy[i, j]
            if current_val != F(0):
                if random() < alpha:
                    all_elements = list(F)
                    all_elements.remove(current_val)
                    Q_noisy[i, j] = all_elements[randint(0, len(all_elements) - 1)]
            else:
                if random() < beta:
                    Q_noisy[i, j] = F(randint(1, q - 1))
    return Q_noisy

def compute_column_distance(c1, c2):
    """ Computes coordinate-level Hamming distance between two vectors """
    return sum(1 for x, y in zip(c1, c2) if x != y)

def solve_lce_from_noise(Q_noisy, q, pairs, verbose=False):
    # obtain dimension (n) and field information
    n = Q_noisy.ncols()
    F = GF(q)

    # instance reconstruction matrix initialized to zero
    Q_reconstructed = zero_matrix(F, n, n)
    
    # get the codeword pairs (a_i, b_i) for i=1,2
    a1, b1 = pairs[0][0], pairs[0][1]
    a2, b2 = pairs[1][0], pairs[1][1]
    
    if verbose:
        print("\n" + "="*60)
        print("STARTING MATRIX RECONSTRUCTION")
        print("="*60)
        
    for j in range(n):
        # extract the j-th column from the noisy hint matrix
        observed_noisy_col = [Q_noisy[i, j] for i in range(n)]
        
        # track the non zero positions in the observed noisy column
        noisy_nz_rows = [idx for idx, val in enumerate(observed_noisy_col) if val != 0]
        
        possible_candidates = []
        
        # Deterministic check: Is the column deterministic because b1(j) != 0 or b2(j) != 0?
        is_deterministic = (b1[0, j] != 0) or (b2[0, j] != 0)
        
        for i in range(n):
            for v_val in range(1, q):
                v = F(v_val)
                
                # Check algebraic validation
                algebraic_match = False
                if is_deterministic:
                    # Clean codeword validation constraint
                    if (a1[0, i] * v == b1[0, j]) and (a2[0, i] * v == b2[0, j]):
                        algebraic_match = True
                else:
                    # Fallback solution: If b1(j) == 0 and b2(j) == 0, then if the real template candidate
                    # is valid, its coordinates must map to zero (a1[0,i] == 0 and a2[0,i] == 0).
                    if (a1[0, i] == 0) and (a2[0, i] == 0):
                        algebraic_match = True
                
                if algebraic_match:
                    # Create the candidate template column vector
                    candidate_col = [F(0)] * n
                    candidate_col[i] = v
                    
                    # Step 2: Extract granular structural metrics
                    hamming_dist = sum(1 for x, y in zip(observed_noisy_col, candidate_col) if x != y)
                    
                    # Structural Penalty 1: Positional Mismatch
                    position_mismatch_penalty = 0 if (i in noisy_nz_rows) else 1
                    
                    # Structural Penalty 2: Scalar Mutation Value
                    if position_mismatch_penalty == 0:
                        value_mismatch_penalty = 0 if (observed_noisy_col[i] == v) else 1
                    else:
                        value_mismatch_penalty = 1
                    
                    # Append sorting dictionary
                    possible_candidates.append({
                        'row': i,
                        'value': v,
                        'pos_penalty': position_mismatch_penalty,
                        'val_penalty': value_mismatch_penalty,
                        'h_dist': hamming_dist
                    })
        
        if verbose:
            print(f"\n---> Column {j}: Real Targets (b1={b1[0,j]}, b2={b2[0,j]})")
            print(f"  Deterministic Column? : {is_deterministic}")
            print(f"  Noisy Column State: {observed_noisy_col}")
            print(f"  Found {len(possible_candidates)} mathematically compatible candidate structures.")

        if possible_candidates:
            # Sort hierarchically:
            # 1. First by positional correctness (keep structural permutation intact)
            # 2. Second by exact value match
            # 3. Third by total Hamming distance to break remaining ties
            possible_candidates.sort(key=lambda x: (x['pos_penalty'], x['val_penalty'], x['h_dist']))
            
            best_candidate = possible_candidates[0]
            Q_reconstructed[best_candidate['row'], j] = best_candidate['value']
            
            if verbose:
                print(f"  [RESOLVED] Selected Row {best_candidate['row']} with Value {best_candidate['value']} "
                      f"(Pos Penalty={best_candidate['pos_penalty']}, Val Penalty={best_candidate['val_penalty']}, Hamming Dist={best_candidate['h_dist']})")
        else:
            if verbose:
                print(f"  [Warning] No candidates satisfied algebraic verification at column {j}.")
                
    return Q_reconstructed

def solver_v2(Q_noisy, G1, G2, pairs, verbose=False):
    """
    Recover the monomial matrix Q or return None on Failure.
    Following the CBA Column Minimum-Distance Correction and Verification routine.
    """
    n = Q_noisy.ncols()
    F = Q_noisy.base_ring()
    # F*_q: Monomial scales must be non-zero elements
    F_nonzero = [x for x in F if x != 0]

    # Initialize Qrec as a copy of Q_noisy
    Q_rec = deepcopy(Q_noisy)

    # Initialize tracking sets
    unverified_columns = set(range(n))

    # Process pair vectors
    for idx, (a_i, b_i) in enumerate(pairs):
        # The verification check must map how the current projection aligns 
        # with the experimental target sequence
        w = a_i * Q_rec

        for j in range(n):
            # 1. Check for Exact Match
            if w[0, j] == b_i[0, j]:
                # Verify that the matching entry is a reliable non-zero signal
                if w[0, j] != 0:
                    unverified_columns.discard(j)
                continue  # Stable position under this pair; move to next column

            # 2. Check for Mismatch -> Trigger Correction Routine
            candidate_scores = []
            
            # Build algebraic candidate template set S_j for column j
            Sj = [(i, alpha) for i in range(n) for alpha in F_nonzero 
                  if a_i[0, i] * alpha == b_i[0, j]]
            
            if not Sj:
                continue

            for (i, v) in Sj:
                # Generate clean monomial vector template e_i(v)
                candidate_col = [F(0)] * n
                candidate_col[i] = v
                
                # Fetch observed noisy column state
                current_col_entries = [Q_rec[r, j] for r in range(n)]
                
                # Calculate coordinate-level distance
                d = sum(1 for x, y in zip(candidate_col, current_col_entries) if x != y)
                candidate_scores.append((i, v, d))
            
            if not candidate_scores:
                continue

            # Isolate minimum distance candidates
            M = min(score[2] for score in candidate_scores)
            min_candidates = [(i, v) for (i, v, d) in candidate_scores if d == M]

            # 3. Resolve structural ties using auxiliary pairs if multiple mins exist
            if len(min_candidates) > 1:
                for a_aux, b_aux in pairs:
                    filtered_candidates = [
                        (i, v) for (i, v) in min_candidates 
                        if v * a_aux[0, i] == b_aux[0, j]
                    ]
                    if len(filtered_candidates) == 1:
                        min_candidates = filtered_candidates
                        break

            # 4. Extract the unique optimal candidate and structurally update Q_rec
            if len(min_candidates) == 1:
                i_opt, v_opt = min_candidates[0]
                
                # Critical structural fix: clear old column entries to maintain a clean monomial structure
                for r in range(n):
                    Q_rec[r, j] = F(0)
                
                # Assign corrected target parameters
                Q_rec[i_opt, j] = v_opt
                unverified_columns.discard(j)

    # Global Linear Code Equivalence validation check
    if (G1 * Q_rec).rref() == G2.rref():
        if verbose:
            print(f"Success: Reconstructed matrix Q satisfies G1 * Q == G2.")
        return Q_rec
    else:
        if verbose:
            print("Failure: Reconstructed matrix Q does not satisfy code equivalence.")
        return None

if __name__ == "__main__":
    n_length, k_dim, q_mod, w_weight = 8, 4, 11, 3
    G1, G2, Q_true, codeword_pairs = sample_instance(n_length, k_dim, q_mod, w_weight)
    print("G1 (RREF):\n", G1)
    print("G2 (RREF):\n", G2)
    print("True Q (Monomial Matrix):\n", Q_true)
    print("Codeword Pairs (a_i, b_i):")
    for idx, (a_i, b_i) in enumerate(codeword_pairs):
        print(f"  Pair {idx+1}: a_i = {a_i}, b_i = {b_i}")
    
    alpha_rate, beta_rate = 0.20, 0.05
    Q_noisy_hint = sample_noisy_monomial(Q_true, q_mod, alpha_rate, beta_rate)
    print("\nNoisy Hint Matrix (Q_noisy_hint):\n", Q_noisy_hint)
    
    Q_recovered = solver_v2(Q_noisy_hint, G1, G2, codeword_pairs, verbose=True)
    print("\nRecovered Q (Q_recovered):\n", Q_recovered)
    
    print("\n" + "="*47)
    print(f"Recovered matrix exactly matches True Q?   : {Q_recovered == Q_true}")
    print(f"Recovered matrix is valid LCE solution?    : {bool(check_solution(Q_recovered, G1, G2))}")
    print("="*47)