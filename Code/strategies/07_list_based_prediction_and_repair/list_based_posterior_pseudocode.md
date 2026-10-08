# List-based posterior construction: pseudocode and relation to the existing framework

This note explains, in pseudocode, how `compute_posterior_table_list_based` (the new function added to `core/LEP_prediction_and_repair_v2.py`) relates to the existing `compute_posterior_table`, and exactly what changes and what stays the same in the prediction-and-repair framework when the secret monomial is leaked as two lists (`instances_generator_monomial_as_list.py`) instead of as a whole matrix (`instances_generator.py`).

## 1. Where this fits in the framework

The framework (`LEP_prediction_and_repair_v2.py`, Algorithms 1-12) has exactly one place where it touches the leakage itself: `compute_posterior_table` is called once, at the very start of `prediction_and_repair_framework` / `run_instance`, to turn the raw hint into an entrywise posterior table `table[i][j] = {a: Pr[Q_ij = a | observation]}`. Every algorithm after that point (`RowScore`, `HungarianAssignment`, `MonomialApproximation`, and the whole repair stage) only ever reads this abstract table, never the raw hint again. That makes the posterior construction a clean, isolated swap point: to support a different leakage model, only this one function needs to change, and the new function only needs to produce a table in the exact same shape.

```
              compute_posterior_table(hint, alpha, beta)
                              |
                              v
   raw hint  ---->  posterior_table[i][j] = {a: p_ij(a)}  ---->  RowScore / HungarianAssignment
                              ^                                  / MonomialApproximation
                              |                                  / repair stage  (UNCHANGED)
   noisy_permutation,          compute_posterior_table_list_based(
   noisy_values      ---->        noisy_permutation, noisy_values, n, q, alpha, beta)
```

## 2. Original: `compute_posterior_table` (whole-matrix hint)

```
function compute_posterior_table(hint, alpha, beta, is_permutation):
    n <- hint.nrows()
    prior <- _row_prior(F, n, is_permutation)          # Pr[Q_ij = a], row-uniform prior
    channel <- build_bit_channel_matrix(q, alpha, beta) # Pr[observed h | true x], q x q

    for i in 0..n-1:
        for j in 0..n-1:
            h_obs <- hint[i, j]                         # ONE q-ary observation per cell
            for a in F:
                unnorm[a] <- prior[a] * channel[a][h_obs]
            table[i][j] <- normalize(unnorm)
    return table
```

Here every one of the `n^2` cells of the monomial matrix is observed **directly**, through the *same* `q`-ary channel: the hint already tells you, for cell `(i, j)`, a noisy version of the field element that sits there (0 if the true nonzero entry of row `i` is not in column `j`).

## 3. New: `compute_posterior_table_list_based` (two-list hint)

With the list-based representation, cell `(i, j)` is never observed directly. Row `i`'s single nonzero entry is described by two *separate* numbers - `permutation[i]` (which column it's in) and `values[i]` (what the nonzero value is) - and each of those two numbers is leaked through its **own** bitwise channel, at its own bit width (`ceil(log2(n))` for the permutation, `ceil(log2(q))` for the values). So cell `(i, j)`'s posterior has to be derived from two independent observations, not read off one of them.

```
function compute_posterior_table_list_based(noisy_permutation, noisy_values, n, q, alpha, beta, is_permutation):
    channel_n <- build_bit_channel_matrix(n, alpha, beta)   # Pr[observed h | true column x], n x n  -- NEW: same function, called with modulus = n
    channel_q <- build_bit_channel_matrix(q, alpha, beta)   # Pr[observed v | true value a],  q x q  -- SAME as before, modulus = q

    for i in 0..n-1:
        h_obs <- noisy_permutation[i] mod n                 # NEW: which column did we (noisily) see row i's entry in?
        v_obs <- noisy_values[i] mod q                       # NEW: what value did we (noisily) see it take?

        # NEW: column posterior - Pr[permutation[i] = j | h_obs], for every candidate j
        for j in 0..n-1:
            col_raw[j] <- channel_n[j][h_obs]
        col_post <- normalize(col_raw)                       # length-n distribution over columns

        # NEW: value posterior - Pr[values[i] = a | v_obs], for every nonzero a  (SAME channel_q as compute_posterior_table used, just applied to v_obs instead of hint[i,j])
        if is_permutation:
            val_post <- {1: 1.0}
        else:
            for a in F, a != 0:
                val_raw[a] <- channel_q[a][v_obs]
            val_post <- normalize(val_raw)

        # NEW: recombine the two independent posteriors into the SAME table[i][j] shape compute_posterior_table produces
        for j in 0..n-1:
            table[i][j][0] <- 1 - col_post[j]
            for a in val_post:
                table[i][j][a] <- col_post[j] * val_post[a]
    return table
```

## 4. What is reused unchanged, and what is genuinely new

**Reused unchanged:**
- `build_bit_channel_matrix(modulus, alpha, beta)` itself - it was already written generically in terms of a `modulus` argument (despite being called with `q` everywhere in the original code), so it needs zero modification to serve as the permutation channel (`modulus = n`) as well as the values channel (`modulus = q`).
- Everything downstream: `compute_row_scores` / `RowScore`, `score_to_cost` / `ScoreToCost`, `hungarian_assignment` / `HungarianAssignment`, `monomial_approximation` / `MonomialApproximation`, and the entire repair stage (`InducedSDRepair`, `StructuredSDRepair`, `MarginalErrorRegion`, `MinimumOverlapSampler`, `PosteriorAwarePrangeRepair`). None of it is touched, because all of it consumes only `posterior_table[i][j] = {a: p_ij(a)}`, which has exactly the same shape and semantics regardless of which function built it.
- The output format: dict keys are field elements of `F = GF(q)`, exactly like `compute_posterior_table`, so `table[i][j].get(F(0), 0)` and `[a for a in F if a != F(0)]` continue to work unmodified in `compute_row_scores`.

**Genuinely new:**
- Two channel builds instead of one (`channel_n` and `channel_q`, both via the same `build_bit_channel_matrix`).
- The per-row factorization step: instead of reading `hint[i, j]` and looking up one `channel[a][h_obs]` per candidate value, each row now computes a length-`n` column posterior from `noisy_permutation[i]` and a length-`(q-1)` value posterior from `noisy_values[i]`, and combines them as a product, `p_ij(a) = col_post[j] * val_post[a]` for `a != 0` (and `p_ij(0) = 1 - col_post[j]`).
- The out-of-range wraparound on the permutation observation (`noisy_permutation[i] mod n`), needed because bit-flipping a `ceil(log2(n))`-bit index can land outside `[0, n-1]` when `n` is not a power of two - handled the same way `build_leaked_monomial_matrix` already handles it for matrix reconstruction, so the two code paths treat overflow consistently.

## 5. Why the factorization is exact (not a heuristic)

Given the same row-independence approximation the existing framework already makes (each row's nonzero position and value are treated independently of every other row), `permutation[i]` and `values[i]` are independent *a priori* (the row prior already factorizes as "which column" times "which value") **and** are leaked through statistically independent channels by construction (two separate `generate_bit_channel_hint_for_list` calls, on two separate lists, with independent randomness). Bayes' rule on two independent observations of two independent variables factorizes exactly into the product used above - this was checked directly by brute-force Bayes enumeration against the closed-form formula (200 random toy cases, ~1e-15 max deviation), not assumed.

This also motivates why the raw lists have to be fed into the posterior directly, rather than first reconstructed into a full matrix via `build_leaked_monomial_matrix` and then run through the *original* `compute_posterior_table`: whenever a row's noisy value happens to reduce to exactly 0, that row's reconstructed matrix cell looks identical to "no observation at all", and the permutation channel's information about which column it came from is silently lost. Measured across a range of `(q, alpha, beta)`, this affects roughly 2-8.5% of rows - `compute_posterior_table_list_based` never discards that information, since it never reconstructs the matrix.

## 6. End-to-end test performed

`compute_posterior_table_list_based` was validated by running the full planted-low-weight-codeword pipeline end to end (mirroring `strategies/06_planted_codeword_repair_isolation/planted_codeword_repair_test.py`, but with the list-based leakage and this new posterior function in place of the whole-matrix hint and `compute_posterior_table`) - see `strategies/07_list_based_prediction_and_repair/planted_codeword_repair_test_list_based.py`, and the note at the top of that file on the numpy/Sage validation gap.
