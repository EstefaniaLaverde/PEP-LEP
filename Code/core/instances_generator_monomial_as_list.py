"""
instances_generator.py  (Code/core/ - shared library, imported by most strategies/)

File to generate instances of LEP and PEP with noisy hints.
There are three versions of the noisy hints:
1. The first version is a matrix over GF(q) where each element is flipped with probability alpha and beta on the same field
2. The second version is a bit-level representation of the matrix where each bit is flipped independently with probability alpha and beta.
3. The third version stores the monomial directly as a permutation list and a list of diagonal
   values (instead of as a full n x n matrix). The leakage is produced by converting each
   entry of BOTH lists into its minimal bit-width value representation - ceil(log2(n)) bits
   per permutation index, ceil(log2(q)) bits per diagonal value - and flipping those bits
   independently with probability alpha and beta, exactly as in version 2, just applied per
   list entry instead of per matrix entry. Both the real and the leaked secret are returned
   as (permutation, values) pairs of lists, rather than as a full matrix.

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

def _flip_value_bits(val, bit_width, alpha, beta):
    """
    Flips the bits of a single integer value independently, MSB first: a '1' bit flips to
    '0' with probability alpha, a '0' bit flips to '1' with probability beta. Shares the
    exact bit-flip logic used per-entry in generate_bit_channel_hint, factored out so it can
    be reused for a flat list of values instead of a full matrix.

    :param val: the raw integer value to leak
    :param bit_width: number of bits used to store val (e.g. 8 for a byte)
    :param alpha: probability of a '1' bit flipping to '0'
    :param beta: probability of a '0' bit flipping to '1'
    :return: the noisy integer obtained after flipping each of the bit_width bits
    """
    new_val = 0
    for b in range(bit_width):
        shift = bit_width - 1 - b
        bit = (val >> shift) & 1

        if bit == 1:
            new_bit = 0 if random() < alpha else 1
        else:
            new_bit = 1 if random() < beta else 0

        new_val |= (new_bit << shift)

    return new_val

def generate_bit_channel_hint_for_list(entries, alpha, beta, bit_width=8):
    """
    Generates a byte-level noisy hint for a flat list of monomial entries (the list-based
    counterpart of generate_bit_channel_hint). This is generic over what the list holds - it
    is used for both the permutation list (indices) and the diagonal values list (field
    elements) of the list-based monomial representation (see generate_random_monomial_as_lists),
    leaking each one the same way.

    Each entry in the list is first transformed into its byte value (an integer in
    [0, 2**bit_width - 1]) and the leakage is then produced on those bits with alpha and beta,
    exactly like generate_bit_channel_hint does per matrix entry: a '1' bit flips to '0' with
    probability alpha, a '0' bit flips to '1' with probability beta.

    :param entries: list of values to leak - either the permutation's indices or the
                    monomial's diagonal coefficients (field elements, or anything int() can
                    be applied to)
    :param alpha: probability of a '1' bit flipping to '0'
    :param beta: probability of a '0' bit flipping to '1'
    :param bit_width: number of bits used to store each entry as a "byte". Defaults to 8
                      (a real byte), independent of the field size q - i.e. each entry is
                      modeled as leaking from an 8-bit register/memory cell, the same way a
                      side-channel would observe it. Pass bit_width=math.ceil(math.log2(q))
                      instead if you want the tighter, field-sized bit width used in
                      generate_bit_channel_hint.
    :return: list of noisy integers, one per input entry, each in [0, 2**bit_width - 1]
    """
    noisy_entries = []
    for entry in entries:
        # Transform the entry into its byte value
        byte_value = int(entry) & (2 ** bit_width - 1)
        noisy_entries.append(_flip_value_bits(byte_value, bit_width, alpha, beta))

    return noisy_entries

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

def generate_random_monomial_as_lists(n, q, is_permutation=False):
    """
    Generates a random monomial of degree n over GF(q), represented directly as a
    permutation list and a list of diagonal (nonzero) coefficient values, instead of as an
    n x n matrix (compare with generate_random_monomial). Row i of the monomial has its
    single nonzero entry at column permutation[i], with value values[i].

    :param n: degree of the monomial
    :param q: size of the finite field
    :param is_permutation: if True, every coefficient is fixed to 1 (a pure permutation);
                           otherwise each coefficient is a random nonzero element of GF(q)
    :return: A tuple (permutation, values) where permutation is a list of n indices (a
             permutation of range(n)) and values is a list of n nonzero GF(q) elements
    """
    F = GF(q)

    # Generate a random permutation of the indices
    permutation = list(range(n))
    shuffle(permutation)

    # Generate the diagonal values
    if is_permutation:
        values = [F(1) for _ in range(n)]
    else:
        values = [F.random_element() for _ in range(n)]
        for i in range(n):
            while values[i] == 0:
                values[i] = F.random_element()

    return permutation, values

def monomial_lists_to_matrix(n, q, permutation, values):
    """
    Builds the n x n monomial matrix corresponding to a (permutation, values) pair, as
    produced by generate_random_monomial_as_lists. Row i has its single nonzero entry at
    column permutation[i], with value values[i].

    :param n: degree of the monomial
    :param q: size of the finite field
    :param permutation: list of n column indices (a permutation of range(n))
    :param values: list of n nonzero GF(q) elements
    :return: The corresponding n x n SageMath matrix over GF(q)
    """
    F = GF(q)
    monomial_matrix = matrix(F, n, n)
    for i in range(n):
        monomial_matrix[i, permutation[i]] = values[i]
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

    # Obtain noisy hint for G2 - kept RAW (NOT coerced into GF(q)); see
    # the identical fix and explanation in instances_generator.py's copy
    # of this function.
    QP_noisy = generate_bit_channel_hint(QP, q, alpha, beta)

    return G1, G2, QP, QP_noisy

def generate_noisy_LCE_instance_CBA_bit_flip_lists_version(n, k, q, alpha, beta, is_monomial=False,
                                                             permutation_bit_width=None, values_bit_width=None):
    """
    Generates a noisy LCE instance for the CBA problem, using the list-based monomial
    representation instead of a full matrix: the secret is stored (and leaked) as a
    permutation list and a list of diagonal values (see generate_random_monomial_as_lists).

    If is_monomial is False, the secret is a random permutation (PEP). Otherwise, the secret
    is a random monomial (LEP).

    The leakage is produced exactly like generate_noisy_LCE_instance_CBA_bit_flip_version's
    bit channel, but applied per list entry instead of per matrix entry: each entry of BOTH
    the permutation list and the diagonal values list is transformed into a byte value, and
    the leakage is generated on those bits with alpha and beta
    (generate_bit_channel_hint_for_list), independently for each list. Each list uses its own
    bit width, sized to the minimum number of bits its entries actually need: permutation
    indices range over [0, n-1], so they need ceil(log2(n)) bits; diagonal values range over
    GF(q), so they need ceil(log2(q)) bits.

    :param n: length of the code (number of columns)
    :param k: dimension of the code (number of rows)
    :param q: size of the finite field
    :param alpha: Probability of a '1' bit flipping to '0'
    :param beta: Probability of a '0' bit flipping to '1'
    :param is_monomial: if True the secret is a random monomial (LEP); if False it is a
                        random permutation (PEP)
    :param permutation_bit_width: (Optional) exact number of bits used to leak each
                                  permutation index. Defaults to the minimum needed to
                                  represent indices up to n, i.e. ceil(log2(n)).
    :param values_bit_width: (Optional) exact number of bits used to leak each diagonal
                             value. Defaults to the minimum needed to represent the field
                             size q, i.e. ceil(log2(q)).
    :return: A tuple (G1, G2, permutation, values, noisy_permutation, noisy_values):
        - G1, G2: the generator matrices, as in the other instance generators
        - permutation: list of n indices - the real secret permutation
        - values: list of n GF(q) elements - the real secret diagonal coefficients
        - noisy_permutation: list of n integers - the leaked permutation indices, each the
          result of flipping the bits of the corresponding value in `permutation`
        - noisy_values: list of n integers - the leaked diagonal coefficients, each the
          result of flipping the bits of the corresponding value in `values`
    """
    # Generate a random generator matrix G1
    G1 = get_random_generator_matrix(n, k, q).rref()

    # Create the random secret monomial directly as a (permutation, values) pair
    permutation, values = generate_random_monomial_as_lists(n, q, is_permutation=not is_monomial)

    # Rebuild the matrix form only to compute G2 = rref(G1 * QP)
    QP = monomial_lists_to_matrix(n, q, permutation, values)
    G2 = (G1 * QP).rref()

    # Size each list's leakage to the minimum bits its entries need: n for the permutation,
    # q for the diagonal values
    if permutation_bit_width is None:
        permutation_bit_width = math.ceil(math.log2(n))
    if values_bit_width is None:
        values_bit_width = math.ceil(math.log2(q))

    # Leak both lists independently, bit by bit, each at its own minimal bit width
    noisy_permutation = generate_bit_channel_hint_for_list(permutation, alpha, beta, bit_width=permutation_bit_width)
    noisy_values = generate_bit_channel_hint_for_list(values, alpha, beta, bit_width=values_bit_width)

    return G1, G2, permutation, values, noisy_permutation, noisy_values

def build_leaked_monomial_matrix(n, q, noisy_permutation, noisy_values):
    """
    Reconstructs the full n x n leaked monomial matrix from the two noisy/leaked lists
    (noisy_permutation, noisy_values), such as the ones returned by
    generate_noisy_LCE_instance_CBA_bit_flip_lists_version - i.e. the list-based counterpart
    of the whole-matrix hint produced directly by generate_noisy_LCE_instance_CBA_bit_flip_version.

    Handling out-of-range permutation indices: bit-flipping noisy_permutation's entries can
    (and does) push some of them outside the valid column range [0, n-1]. For example n=7
    needs permutation_bit_width=ceil(log2(7))=3 bits to store an index, but 3 bits can
    represent up to 7, one past the largest valid column index 6. Rather than discard that
    row's observation or clip it to n-1 (which would bias every overflowing index towards the
    same column), each noisy index is reduced modulo n, wrapping it back into [0, n-1]. This
    keeps every bit pattern equally likely to land on any column and mirrors exactly how
    generate_noisy_LCE_instance_CBA_bit_flip_version already treats the analogous overflow on
    the diagonal values (an out-of-range integer is coerced into GF(q), which reduces it
    modulo q for a prime field) - so both lists get the same "wrap the noisy observation back
    into its valid domain" treatment instead of only one of them overflowing silently.

    Note the resulting matrix is a *noisy observation*, not a valid monomial matrix: because
    each row's index is wrapped independently, two different rows can land on the same
    column (giving it two nonzero entries) while another column gets none. That's expected -
    the same loss of monomial structure already happens in the whole-matrix bit-flip hint
    (version 2), where any entry can flip on or off independently of the rest of the matrix.

    :param n: degree of the monomial (and length of the permutation/values lists)
    :param q: size of the finite field
    :param noisy_permutation: list of n leaked column indices - possibly containing values
                              outside [0, n-1], which are wrapped modulo n
    :param noisy_values: list of n leaked diagonal values - possibly containing values
                         outside [0, q-1], which are reduced into GF(q) the same way
                         generate_noisy_LCE_instance_CBA_bit_flip_version already does
    :return: The n x n SageMath matrix over GF(q) built by placing, for each row i,
             noisy_values[i] (reduced into GF(q)) at column (noisy_permutation[i] mod n)
    """
    F = GF(q)
    leaked_matrix = matrix(F, n, n)
    for i in range(n):
        col = int(noisy_permutation[i]) % n
        leaked_matrix[i, col] = F(int(noisy_values[i]))
    return leaked_matrix

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
    q = 127   # size of the finite field
    alpha = 0.01  # probability of flipping 0 to a random element
    beta = 0.1   # probability of flipping a random element to 0

    # # Generate a noisy PEP instance
    # G1, G2, P, P_noisy = generate_noisy_LCE_instance_CBA(n, k, q, alpha, beta, is_monomial=False)

    # # Generate a noisy LEP instance
    # G1, G2, Q, Q_noisy = generate_noisy_LCE_instance_CBA(n, k, q, alpha, beta, is_monomial=True)

    # # # Transform the problem to a syndrome decoding problem
    # # H_tilde, vectorP_noisy = transform_problem_to_syndrome_decoding(G1, G2, P_noisy)

    # print("Generator matrix G1:")
    # print(G1)
    # print("\nGenerator matrix G2:")
    # print(G2)
    # print("\nSecret monomial matrix Q:")
    # print(Q)
    # print("\nNoisy hint for Q:")
    # print(Q_noisy)

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

    # Generate a noisy LEP instance using the list-based (permutation, diagonal values)
    # representation, with byte-level leakage on both lists.
    G1, G2, permutation, values, noisy_permutation, noisy_values = generate_noisy_LCE_instance_CBA_bit_flip_lists_version(
        n, k, q, alpha, beta, is_monomial=True
    )

    print("\nSecret permutation:")
    print(permutation)
    print("\nLeaked (noisy) permutation:")
    print(noisy_permutation)
    print("\nSecret diagonal values:")
    print(values)

    print("\nLeaked (noisy) diagonal values:")
    print(noisy_values)

    # build the leaked monomial matrix from the lists
    real_monomial = monomial_lists_to_matrix(n, q, permutation, values)
    print('\nReal monomial matrix')
    print(real_monomial)

    # Rebuild the full leaked monomial matrix from the two noisy lists (note noisy_permutation
    # can contain indices >= n, wrapped modulo n by build_leaked_monomial_matrix)
    leaked_matrix = build_leaked_monomial_matrix(n, q, noisy_permutation, noisy_values)
    print("\nLeaked monomial matrix (rebuilt from the two noisy lists):")
    print(leaked_matrix)