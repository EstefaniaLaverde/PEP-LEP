from copy import deepcopy
from sage.all import shuffle, zero_matrix, randint, binomial, log, GF, FiniteField, rank, floor, ceil, random_matrix
import itertools
from math import sqrt

# No need to import 'sorted' from sage.all because Python handles it globally natively!

def check_solution(Q, G1, G2):

    if not(is_monomial(Q)):
        return 0

    if G2.rref() != (G1 * Q).rref():
        return 0

    return 1

def is_monomial(Q):

    if Q == None:
        return False

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
    Sample a codeword of length n and weight w
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
    Return the minimum weight of a (n,k) code over Fq
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
    """q-ary entropy function (base q)."""
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
