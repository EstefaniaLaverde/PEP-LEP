"""
instances_generator.py  (Code/core/ - shared library, imported by most strategies/)

File to generate instances of LEP and PEP with noisy hints.
There are two versions of the noisy hints:
1. The first version is a matrix over GF(q) where each element is flipped with probability alpha and beta on the same field
2. The second version is a bit-level representation of the matrix where each bit is flipped independently with probability alpha and beta.

This is core, actively-used, shared code - it lives in Code/core/ rather
than under any single strategies/ folder because most strategies import
from it. If you're adding a new strategy that needs a fresh LEP/PEP
instance, this is almost certainly the module to use rather than writing
a new generator from scratch.
"""
from sage.all import GF, random_matrix, random, shuffle, matrix
import math
from random import randint
import copy

def get_random_generator_matrix(n, k, q):
    """
    Generates a random k x n generator matrix over GF(q) with full rank k.
    
    :param n: length of the code (number of columns)
    :param k: dimension of the code (number of rows)
    :param q: size of the finite field
    :return: A k x n SageMath matrix over GF(q) with rank k
    """
    F = GF(q)
    while True:
        # Generate a random k x n matrix over the finite field
        G = random_matrix(F, k, n)
        
        # Ensure the matrix has full row rank k so it forms a valid basis
        if G.rank() == k:
            return G
        
def generate_noisy_hint(Q, alpha, beta, is_permutation=False):
    """
    Generates a noisy hint for the given matrix Q.
    
    :param Q: The original matrix (generator matrix)
    :param alpha: Probability of 0 flipping to a random element in the field, different from 0
    :param beta: Probability of a random element flipping to 0
    :return: A noisy hint matrix
    """
    # Obtain the field and dimensions of Q
    F = Q.base_ring()
    n, m = Q.nrows(), Q.ncols()
    
    # Create a copy of Q to modify as the hint
    hint = copy.copy(Q)
    
    # Flip bits with probability alpha
    for i in range(n):
        for j in range(m):
            if random() < alpha:
                # Flip to a random element in the field, different from the current element
                if is_permutation:
                    new_value = F(1) if hint[i, j] == F(0) else F(0)
                else:
                    new_value = F.random_element()
                    while new_value == hint[i, j]:
                        new_value = F.random_element()
                hint[i, j] = new_value
    
    # Add noise with probability beta
    for i in range(n):
        for j in range(m):
            if random() < beta:
                hint[i, j] = 0  # Flip to 0 with probability beta
    
    return hint

def generate_bit_channel_hint(Q, q, alpha, beta, bit_width=None):
    """
    Generates a bit-level noisy hint for a monomial or permutation matrix.
    Each bit used to store the field elements is flipped independently based on its value.
    
    :param Q: The original matrix (Monomial or Permutation)
    :param q: The size of the finite field
    :param alpha: Probability of a '1' bit flipping to '0'
    :param beta: Probability of a '0' bit flipping to '1'
    :param bit_width: (Optional) Exact number of bits used for storage. 
                      Defaults to the minimum bits needed for the field size q.
    :return: A nested list of integers representing the noisy matrix
    """
    # convert Q to a standard matrix or nested format if it's a structural object
    # to ensure we can seamlessly iterate over rows and columns.
    try:
        n, m = Q.nrows(), Q.ncols()
        matrix_source = Q
    except AttributeError:
        # If Q is a Permutation object or structural Monomial representation
        matrix_source = Q.matrix()
        n, m = matrix_source.nrows(), matrix_source.ncols()
    
    # determine bit width based on the explicit field size q
    if bit_width is None:
        bit_width = math.ceil(math.log2(q))
    
    # Initialize the noisy matrix structure
    hint = [[0 for _ in range(m)] for _ in range(n)]
    
    for i in range(n):
        for j in range(m):
            # Extract the element as a raw integer
            val = int(matrix_source[i, j])
            new_val = 0
            
            # Evaluate each bit position independently, MSB first: loop
            # index b=0 corresponds to the most significant bit, i.e. shift
            # amount (bit_width - 1 - b). This must match the bit order
            # assumed by build_bit_channel_matrix in the prediction-and-repair
            # module (_bits_msb_first), or the posterior computed from this
            # hint will be built against the wrong channel.
            for b in range(bit_width):
                shift = bit_width - 1 - b
                # Extract the bit at this (MSB-first) position (0 or 1)
                bit = (val >> shift) & 1
                
                if bit == 1:
                    # 1 flips to 0 with probability alpha
                    new_bit = 0 if random() < alpha else 1
                else:
                    # 0 flips to 1 with probability beta
                    new_bit = 1 if random() < beta else 0
                
                # Reconstruct the integer bit by bit, MSB first
                new_val |= (new_bit << shift)
                
            hint[i][j] = new_val
            
    return hint

def generate_random_monomial(n, q, is_permutation=False):
    """
    Generates a random monomial matrix of degree n over GF(q).
    Monomial matrices have one single non-zero entry in each row and column, which is a non-zero element of the field.
    
    :param n: degree of the monomial
    :param q: size of the finite field
    :return: A random monomial matrix represented as a list of coefficients
    """

    F = GF(q)
    # Generate a random permutation of the indices
    indices = list(range(n))
    shuffle(indices)

    # Generate random non-zero coefficients for the monomial
    if is_permutation:
        coefficients = [F(1) for _ in range(n)]
    else:
        coefficients = [F.random_element() for _ in range(n)]
        for i in range(n):
            while coefficients[i] == 0:
                coefficients[i] = F.random_element()

    # Create the monomial matrix as a list of coefficients corresponding to the permutation
    monomial_matrix = matrix(F, n, n)
    for i in range(n):
        monomial_matrix[i, indices[i]] = coefficients[i]
    return monomial_matrix

def generate_noisy_LCE_instance_CBA(n, k, q, alpha, beta, is_monomial=False):
    """
    Generates a noisy LCE instance for the CBA problem. 
    If is_monomial is False, the secret is a random permutation matrix (PEP). Otherwise, the secret 
    is a random monomial (LEP).
    
    :param n: length of the code (number of columns)
    :param k: dimension of the code (number of rows)
    :param q: size of the finite field
    :param alpha: Probability of 0 flipping to a random element in the field, different from 0
    :param beta: Probability of a random element flipping to 0
    :return: A tuple (Q, hint) where Q is the original generator matrix and hint is the noisy hint matrix
    """
    # Generate a random generator matrix G1
    G1 = get_random_generator_matrix(n, k, q).rref()

    # Create a random secret matrix QP
    if is_monomial:
        QP = generate_random_monomial(n, q, is_permutation=False)
    else:
        QP = generate_random_monomial(n, q, is_permutation=True)

    G2 = (G1*QP).rref()

    # Obtain noisy hint for G2
    QP_noisy = generate_noisy_hint(QP, alpha, beta, is_permutation=not is_monomial)

    return G1, G2, QP, QP_noisy

def generate_noisy_LCE_instance_CBA_bit_flip_version(n, k, q, alpha, beta, is_monomial=False):
    """
    Generates a noisy LCE instance for the CBA problem. 
    If is_monomial is False, the secret is a random permutation matrix (PEP). Otherwise, the secret 
    is a random monomial (LEP).
    
    :param n: length of the code (number of columns)
    :param k: dimension of the code (number of rows)
    :param q: size of the finite field
    :param alpha: Probability of 0 flipping to a random element in the field, different from 0
    :param beta: Probability of a random element flipping to 0
    :return: A tuple (Q, hint) where Q is the original generator matrix and hint is the noisy hint matrix
    """
    # Generate a random generator matrix G1
    G1 = get_random_generator_matrix(n, k, q).rref()

    # Create a random secret matrix QP
    if is_monomial:
        QP = generate_random_monomial(n, q, is_permutation=False)
    else:
        QP = generate_random_monomial(n, q, is_permutation=True)

    G2 = (G1*QP).rref()

    # Obtain noisy hint for G2
    QP_noisy = generate_bit_channel_hint(QP, q, alpha, beta)

    # Pass QP to a matrix on the field l, where l is the maximum value the bits for the field can create
    bits_needed = math.ceil(math.log2(q))
    l = 2**bits_needed
    QP_noisy = matrix(GF(q), QP_noisy)

    return G1, G2, QP, QP_noisy

def compute_kronecker_product(A, B):
    """
    Computes the Kronecker product of two matrices A and B.
    
    :param A: First matrix
    :param B: Second matrix
    :return: The Kronecker product of A and B
    """
    return A.tensor_product(B)

def transform_secret_to_single_vector(Q):
    """
    Transforms a matrix Q into a single vector by concatenating its columns.
    
    :param Q: The input matrix
    :return: A single vector obtained by concatenating the columns of Q
    """
    n, m = Q.nrows(), Q.ncols()
    single_vector = []
    for j in range(m):
        for i in range(n):
            single_vector.append(Q[i, j])
    return single_vector

def obtain_parity_check_matrix(G):
    """
    Computes the parity-check matrix H for a given generator matrix G.
    
    :param G: The generator matrix
    :return: The parity-check matrix H such that G * H^T = 0
    """
    n, k = G.ncols(), G.nrows()
    # Compute the null space of G to get the parity-check matrix
    H = G.right_kernel().matrix()
    return H

def transform_problem_to_syndrome_decoding(G1, G2, P_noisy):
    """
    Transforms the problem of finding the secret permutation matrix P into a syndrome decoding problem using
    the parity-check matrix of G1 and G2 and the Kronecker product.
    
    :param G1: The first generator matrix
    :param G2: The second generator matrix rref(G1 * P)
    :param P_noisy: The noisy hint for the secret permutation matrix P
    :return: A tuple (H_tilde, vectorP_noisy)
    """
    # Compute the parity-check matrix for G1 and G2
    H1 = obtain_parity_check_matrix(G1)
    H2 = obtain_parity_check_matrix(G2)

    # Compute the kronecker product
    H1_tilde = compute_kronecker_product(H2, G1)
    H2_tilde = compute_kronecker_product(G2, H1)

    # Create a single matrix stacking (vertically) H1_tilde and H2_tilde
    H_tilde = H1_tilde.stack(H2_tilde)

    # Transform the noisy hint P_noisy into a single vector
    vectorP_noisy = transform_secret_to_single_vector(P_noisy)

    return H_tilde, vectorP_noisy

def transform_problem_to_syndrome_decoding_H1(G1, G2, P_noisy):
    """
    Transforms the problem of finding the secret permutation matrix P into a syndrome decoding problem using
    the parity-check matrix of G1, G2 and the kronecker product. H_tilde is only based on H1.
    
    :param G1: The first generator matrix
    :param G2: The second generator matrix rref(G1 * P)
    :param P_noisy: The noisy hint for the secret permutation matrix P
    :return: A tuple (H_tilde, vectorP_noisy)
    """
    # Compute the parity-check matrix for G1 and G2
    H1 = obtain_parity_check_matrix(G1)
    H2 = obtain_parity_check_matrix(G2)

    # Compute the kronecker product
    H_tilde = compute_kronecker_product(H2, G1)

    # Transform the noisy hint P_noisy into a single vector
    vectorP_noisy = transform_secret_to_single_vector(P_noisy)

    return H_tilde, vectorP_noisy


if __name__ == "__main__":
    # Set parameters for example instance generation
    n = 7  # length of the code
    k = 3   # dimension of the code
    q = 7   # size of the finite field
    alpha = 0.01  # probability of flipping 0 to a random element
    beta = 0.2   # probability of flipping a random element to 0

    # Generate a noisy PEP instance
    G1, G2, P, P_noisy = generate_noisy_LCE_instance_CBA(n, k, q, alpha, beta, is_monomial=False)

    # Generate a noisy LEP instance
    G1, G2, Q, Q_noisy = generate_noisy_LCE_instance_CBA(n, k, q, alpha, beta, is_monomial=True)

    # # Transform the problem to a syndrome decoding problem
    # H_tilde, vectorP_noisy = transform_problem_to_syndrome_decoding(G1, G2, P_noisy)

    print("Generator matrix G1:")
    print(G1)
    print("\nGenerator matrix G2:")
    print(G2)
    print("\nSecret monomial matrix Q:")
    print(Q)
    print("\nNoisy hint for Q:")
    print(Q_noisy)

    # print("\nParity-check matrix H_tilde:")
    # print(H_tilde)
    # print("\nNoisy hint vector for P:")
    # print(vectorP_noisy)

    # # Generate a noisy LEP instance
    # G1, G2, Q, Q_noisy = generate_noisy_LCE_instance_CBA(n, k, q, alpha, beta, is_monomial=True)
    # print("\nGenerator matrix G1:")
    # print(G1)
    # print("\nGenerator matrix G2:")
    # print(G2)
    # print("\nSecret monomial matrix Q:")
    # print(Q)
    # print("\nNoisy hint for Q:")
    # print(Q_noisy)