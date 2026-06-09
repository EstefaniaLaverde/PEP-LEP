from copy import deepcopy
import random
from sage.all import shuffle, zero_matrix, randint, binomial, log, GF, FiniteField, rank, floor, ceil, random_matrix
import itertools
from math import sqrt


def check_solution(Q, G1, G2):
    """
    Check if Q is a solution to the LCE instance defined by G1 and G2, i.e., if G1*Q and G2 generate the same code.

    Args:
        Q: monomial matrix
        G1: generator matrix of the first code
        G2: generator matrix of the second code

    Returns:
        1 if Q is a solution, 0 otherwise
    """

    # first check if Q is a monomial matrix, if not, return 0 immediately
    if not(is_monomial(Q)):
        return 0

    # compute RREF(G1*Q) and RREF(G2) and check if they are the same
    if G2.rref() != (G1 * Q).rref():
        return 0

    return 1

def is_monomial(Q):
    """
    Check if a matrix Q is a monomial matrix.

    Args:
        Q: matrix to check

    Returns:
        True if Q is monomial, False otherwise
    """

    # if Q is None, return False
    if Q == None:
        return False

    # check if Q has exactly one non-zero entry in each row and each column
    for i in range(Q.nrows()):
        count = 0
        for j in range(Q.ncols()):
            if Q[i,j] != 0:
                count += 1
        if count != 1:
            return False

    for i in range(Q.ncols()):
        count = 0
        for j in range(Q.nrows()):
            if Q[i,j] != 0:
                count += 1
        if count != 1:
            return False

    return True


def sample_low_weight(n, q, w):
    '''
    Sample a codeword of length n and weight w by selecting w random positions and assigning them random non zero values in Fq
    :param n: length of the codeword
    :param q: modulo
    :param w: Hamming weight
    :return: low weight codeword
    '''
    a = [i for i in range(0, n)]
    for i in range( n -1, n- 1 - w, -1):
        j = randint(0, i)
        tmp = a[i]
        a[i] = a[j]
        a[j] = tmp

    a = a[n - w:n]

    lw = zero_matrix(GF(q), 1, n)
    for x in range(w):
        lw[:, a[x]] = randint(1, q - 1)

    return lw


# sample a pair with a fixed overlap ell
def sample_pair_low_weight(n, q, w, ell):
    '''
    Sample a pair of codewords of length n, weight w, and overlap ell
    :param n: length
    :param q: modulo
    :param w: hamming weight
    :param ell: overlap
    :return: list containing the two codewords
    '''
    values = [i for i in range(0, n)]
    a = [0]*w
    b = [0]*w
    for i in range(ell):
        tmp = values.pop(randint(0,len(values)-1))
        a[i] = tmp
        b[i] = tmp
    for i in range(ell, w):
        tmp = values.pop(randint(0,len(values)-1))
        a[i] = tmp
        tmp = values.pop(randint(0,len(values)-1))
        b[i] = tmp

    lw1 = zero_matrix(GF(q), 1, n)
    lw2 = zero_matrix(GF(q), 1, n)
    for x in range(w):
        lw1[:, a[x]] = randint(1, q - 1)
        lw2[:, b[x]] = randint(1, q - 1)

    return [lw1, lw2]

def sample_monomial_matrix(n, q):
    '''
    Sample random monomial matrix
    :param n: length
    :param q: modulo
    :return: monomial matrix
    '''
    P = zero_matrix(FiniteField(q), n, n)

    a = [i for i in range(0, n)]
    for i in range(n-1, 0, -1):
        j = randint(0, i)
        tmp = a[i]
        a[i] = a[j]
        a[j] = tmp

    for i in range(0, n):
        P[i,a[i]] = randint(1,q-1)

    return P

def minimal_w(n, k, q):
    '''
    Return the minimum weight of a (n,k) code over Fq, according to the Gilbert-Varshamov bound. 
    :param n: length
    :param k: dimension
    :param q: modulo
    :return: w, minimum Hamming weight
    '''
    w = 1
    s = 0
    pow = q ** (n - k)
    while True:
        s += binomial(n,w)*(q-1)**(w-1)
        if s > pow:
            return w
        w = w+1

def H_q(x, q):
    """
    q-ary entropy function (base q).
    """
    if x <= 0 or x >= 1:
        return 0.0
    return (x * log(q-1, q)
            - x * log(x, q)
            - (1-x) * log(1-x, q))

def minimal_w_entropy(n, k, q, tol=1e-6):
    """
    Estimate minimum distance of a random [n,k] code over F_q
    using the q-ary entropy function.

    Returns:
        w (int): estimated minimum weight
    """
    rate = k / n
    target = 1 - rate  # GV bound threshold

    # Binary search for delta in [0,1]
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if H_q(mid, q) < target:
            lo = mid
        else:
            hi = mid

    delta = (lo + hi) / 2
    return ceil(delta * n)


def support_intersection(n, lw1, lw2):
    '''
    Compute the intersection of the support of two codewords, i.e.,
    the number of entries that are non-zero for both codewords
    Args:
        n: length of the codewords
        lw1: first codeword
        lw2: second codeword

    Returns: ell, the number of entries that are non-zero for both codewords
    '''
    ell = 0
    for i in range(n):
        if lw1[0][i] != 0 and lw2[0][i] != 0:
            ell += 1
    return ell

def sample_instance(n, k, q, w, N=2, ell=-1, verbose=False):
    """
    Sample a LCE instance together with N low-weight codewords pairs in the codes

    Parameters:
        n: code length
        k: code dimension
        q: modulo
        w: hamming weight of low-weight pairs
        N: number of low-weight codewords, default to 2 in this project
        ell: intersection length. Set to -1 for enforcing ell>=E[ell], otherwise give a specific value
        verbose: True or False for printing details or not
    Returns:
        G: generator of first code
        G_: generator of second code
        Q: monomial matrix
        pairs: low-weight codewords pairs
    """

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
        for i in range(1, N):
            G = G.stack(lw[i])
        G = G.stack(random_matrix(GF(q), k - N, n))
        if rank(G) == k:
            break
    Q = sample_monomial_matrix(n, q)
    G = G.rref()
    G_ = (G * Q).rref()
    pairs = [[lw[i], lw[i] * Q] for i in range(N)]

    if verbose:
        print("Generator matrix 1")
        print(G)
        print("Generator matrix 2")
        print(G_)
        print("Monomial matrix")
        print(Q)
        print("Low-weight pairs")
        for pair in pairs:
            print(pair)

    return G, G_, Q, pairs


def map_supports_11(pairs, support1, support2):
    '''
    Args:
        pairs: low weight codeword pairs from the two codes
        support1: indices in the support intersection from the first code
        support2: indices in the support intersection from the second code

    Returns: list of indices mapped to each other from the support intersection
    '''

    c1 = pairs[0][0][0]
    c2 = pairs[1][0][0]
    d1 = pairs[0][1][0]
    d2 = pairs[1][1][0]

    maps = {}

    skip_r = []
    skip_l = []
    L_nonzero = []
    for i1 in support1:
        if i1 in skip_l:
            continue
        L_nonzero_r_tmp = []
        for i2 in support2:
            if i2 in skip_r:
                continue
            if c2[i1] / d2[i2] == c1[i1] / d1[i2]:  # the entry might be non-zero
                L_nonzero_r_tmp += [i2]
        skip_r += L_nonzero_r_tmp
        if len(L_nonzero_r_tmp) == 1:  # there is a 1o1 match
            L_nonzero.append({
                    'left'  : [i1],
                    'right' : [L_nonzero_r_tmp[0]]
                })
            skip_l += [i1]
        elif len(L_nonzero_r_tmp) > 1: # there is not a 1o1 match, so search for the others on the left
            L_nonzero_l_tmp = []
            for i1_tmp in support1:
                if i1_tmp in skip_l:
                    continue
                if c2[i1_tmp] / c1[i1_tmp] == c2[i1] / c1[i1]:
                    L_nonzero_l_tmp += [i1_tmp]
            L_nonzero.append({
                'left': L_nonzero_l_tmp,
                'right': L_nonzero_r_tmp
            })
            skip_l += L_nonzero_l_tmp
        else:
            raise ValueError("It should not go here. The supports are not equivalent.")

    return L_nonzero


def shuffle_D(D):
    '''
    Args:
        D: input list

    Returns: A random shuffle of D
    '''
    n = len(D)
    for i in range(n - 1, 0, -1):
        j = randint(0, i)
        D[i], D[j] = D[j], D[i]

def generate_all_pairs(codeword_list):
    """
    Generates all unique combinations of size 2 from a list of codewords.
    """
    return list(itertools.combinations(codeword_list, 2))


def compute_compatibility_signature(codeword_pair, length_n):
    """
    Computes the structural compatibility signature (v_comp or w_comp) 
    for a given pair of codewords, converting finite field quotients 
    to primitive integers to ensure they are hashable.
    """
    c1, c2 = codeword_pair[0], codeword_pair[1]
    
    # Split into zero/non-zero support categories (adapted from determine_four_set_elements)
    set_11 = []
    for i in range(length_n):
        # Coordinates are checked for simultaneous non-zero presence
        if c1[:, i] != 0 and c2[:, i] != 0:
            set_11.append(i)
            
    ell = len(set_11)
    
    # Calculate element-wise quotients over the support intersection
    quotients = []
    for idx in set_11:
        # Extract the scalar cell explicitly out of the 1xn matrix
        val1 = c1[0, idx]
        val2 = c2[0, idx]
        
        # Calculate field quotient
        q_val = val1 / val2
        
        # Convert to a standard Python integer so it can be hashed as a dictionary key
        quotients.append(int(q_val))
        
    # Sorting converts the list to a position-invariant multiset signature
    # Converting elements to simple types allows seamless tuple hashing
    quotient_multiset = tuple(sorted(quotients))
    
    return (ell, quotient_multiset)


def find_compatible_pairs(L1, L2, n):
    """
    Main pipeline function that matches compatible codeword pairs 
    between Code 1 (from list L1) and Code 2 (from list L2).
    """
    # 1. Generate all pairs from both input lists
    pairs_L1 = generate_all_pairs(L1)
    pairs_L2 = generate_all_pairs(L2)
    
    # 2. Group Code 2 pairs into a hash-map by their signature (w_comp) for fast lookup
    signature_map_L2 = {}
    for pair_w in pairs_L2:
        w_comp = compute_compatibility_signature(pair_w, n)
        if w_comp not in signature_map_L2:
            signature_map_L2[w_comp] = []
        signature_map_L2[w_comp].append(pair_w)
        
    compatible_matches = []
    
    # 3. Stream through Code 1 pairs (v_comp) and check for collisions against Code 2 pairs
    for pair_v in pairs_L1:
        v_comp = compute_compatibility_signature(pair_v, n)
        
        if v_comp in signature_map_L2:
            for matching_pair_w in signature_map_L2[v_comp]:
                compatible_matches.append({
                    "pair_C1": pair_v,          # (v1, v2)
                    "pair_C2": matching_pair_w, # (w1, w2)
                    "signature": v_comp         # (ell, sorted_quotients)
                })
                
    return compatible_matches

def calculate_expected_nw(n, k, q, w):
    """
    Computes N_w: The expected number of codewords of weight w 
    in a random [n, k] code over Fq.
    """
    # Number of possible support positions * combinations of non-zero field scalars
    total_vectors_of_weight_w = binomial(n, w) * (q - 1)**w
    
    # Probability that a random vector falls inside a k-dimensional subspace
    subspace_probability = q**(k - n)
    
    N_w = total_vectors_of_weight_w * subspace_probability
    return float(N_w)

def calculate_article_list_size(n, k, q, w):
    """
    Computes N: The optimal size of lists L1 and L2 as defined in the article.
    Formula: ceil( sqrt( (2 * N_w) / (q - 1) ) )
    """
    N_w = calculate_expected_nw(n, k, q, w)
    
    # Avoid math errors if code parameters yield an expected Nw close to 0
    if N_w <= 0:
        return 2 
        
    target_N = ceil(sqrt((2 * N_w) / (q - 1)))
    
    # The algorithm fundamentally requires at least 2 items to form combinations
    return max(2, int(target_N))

def sample_noisy_monomial(Q, q, alpha, beta):
    """
    Generates a noisy version of a monomial matrix Q based on error rates alpha and beta.
    
    Args:
        Q: The original clean monomial matrix (Sage matrix over GF(q))
        q: The finite field modulus
        alpha: Probability of a non-zero entry changing to 0
        beta: Probability of a zero entry changing to a random non-zero element
        
    Returns:
        Q_noisy: A new matrix with applied noise
    """
    # create a deep copy to avoid modifying the original matrix
    Q_noisy = deepcopy(Q)
    n_rows = Q_noisy.nrows()
    n_cols = Q_noisy.ncols()
    F = GF(q)
    
    for i in range(n_rows):
        for j in range(n_cols):
            current_val = Q_noisy[i, j]
            
            if current_val != 0:
                # Non-zero entry drops to 0 with probability alpha
                if random.random() < alpha:
                    Q_noisy[i, j] = F(0)
            else:
                # Zero entry flips to a random field element with probability beta
                if random.random() < beta:
                    # Pick a random element from 1 to q-1
                    random_nonzero = F(randint(1, q - 1))
                    Q_noisy[i, j] = random_nonzero
                    
    return Q_noisy

def analyze_noisy_codewords(a1, a2, Q, Q_noisy, q):
    """
    Simulates the generation of clean and noisy codeword pairs and performs
    the filtering check based on matching indices.
    
    Args:
        a1, a2: Clean low-weight codewords from Code 1 (1 x n matrices)
        Q: The true, clean monomial matrix
        Q_noisy: The degraded hint matrix
        q: Modulo of the field
        
    Returns:
        A dictionary containing the noisy vectors and a strategy array for each column.
    """
    # 1. Obtain clean counterparts based on the real Q
    # Note: Depending on your convention, if vectors are 1 x n rows, 
    # multiplying by an n x n matrix is done as: b = a * Q
    b1 = a1 * Q
    b2 = a2 * Q
    
    # 2. Obtain noisy counterparts based on Q_noisy
    b1_noisy = a1 * Q_noisy
    b2_noisy = a2 * Q_noisy
    
    n = Q.ncols()
    column_strategies = {}
    
    # 3. Check consistency index by index
    for i in range(n):
        # Check if both vectors match their noisy versions at index i
        match_b1 = (b1[0, i] == b1_noisy[0, i])
        match_b2 = (b2[0, i] == b2_noisy[0, i])
        
        if match_b1 and match_b2:
            column_strategies[i] = {
                "action": "trust_hint",
                "trusted_column": Q_noisy[:, i] # Keep the column intact
            }
        else:
            column_strategies[i] = {
                "action": "full_search",
                "reason": "Mismatched codeword entry detected due to channel noise."
            }
            
    return {
        "b1_noisy": b1_noisy,
        "b2_noisy": b2_noisy,
        "strategies": column_strategies
    }

def reconstruct_Q_from_noise(Q_noisy, q, pairs):
    """
    Reconstructs the true monomial matrix Q using a noisy matrix hint 
    and two known equivalent codeword pairs.
    
    Args:
        Q_noisy: The noisy matrix hint (matrix over GF(q))
        q: Modulo of the finite field
        pairs: List of pairs [[a1, b1], [a2, b2]] where b_i = a_i * Q
        
    Returns:
        Q_reconstructed: The reconstructed matrix, or None if it fails.
    """
    n = Q_noisy.ncols()
    F = GF(q)
    
    # Initialize an empty matrix to build the reconstruction
    Q_reconstructed = zero_matrix(F, n, n)
    
    # Extract the test pairs
    a1, b1 = pairs[0][0], pairs[0][1]
    a2, b2 = pairs[1][0], pairs[1][1]
    
    # Precompute the noisy products to evaluate the filtering rule
    b1_noisy = a1 * Q_noisy
    b2_noisy = a2 * Q_noisy
    
    for j in range(n):
        # 1. Check if the noisy column produces the correct codeword elements
        if (b1[0, j] == b1_noisy[0, j]) and (b2[0, j] == b2_noisy[0, j]):
            # Noise-free column detected: trust and copy the entire column from Q_noisy
            for i in range(n):
                Q_reconstructed[i, j] = Q_noisy[i, j]
                
        # 2. Otherwise, noise occurred. We must exhaustively resolve this column.
        else:
            possible_candidates = []
            
            # Test every possible row 'i' where the single non-zero entry could live
            for i in range(n):
                # Test every possible non-zero scalar value 'v' in the field
                for v_int in range(1, q):
                    v = F(v_int)
                    
                    # Verify if placing 'v' at row 'i' satisfies both equations
                    cond1 = (a1[0, i] * v == b1[0, j])
                    cond2 = (a2[0, i] * v == b2[0, j])
                    
                    if cond1 and cond2:
                        possible_candidates.append((i, v))
            
            # If our two codewords successfully isolated a unique entry arrangement:
            if len(possible_candidates) == 1:
                target_i, target_v = possible_candidates[0]
                Q_reconstructed[target_i, j] = target_v
            elif len(possible_candidates) > 1:
                # Ambiguity occurs if the low-weight codewords have zeroes at index i.
                # In your full framework, this triggers your tree-backtracking search!
                print(f"Ambiguity at column {j}: multiple candidates match. Selection required.")
                # For this baseline script, we select the first one found
                target_i, target_v = possible_candidates[0]
                Q_reconstructed[target_i, j] = target_v
            else:
                print(f"Warning: No valid candidate found for column {j}.")
                
    return Q_reconstructed