# Column Reference of Results CSV

One row = one run of `run_single_experiment` for a specific
`(n, k, q, alpha, beta, rmax, weight_factor, budgets, enum_version, repair_algorithm, trial)`
combination.

## Run parameters (inputs)

| Column | Meaning |
|---|---|
| `n` | Code length (number of columns of the generator matrices). |
| `k` | Code dimension (number of rows of the generator matrices). |
| `q` | Field size, `GF(q)`. Assumed prime (see the bitwise-channel caveat). |
| `alpha` | Bitwise channel parameter: `Pr[bit 1 -> 0]`. |
| `beta` | Bitwise channel parameter: `Pr[bit 0 -> 1]`. |
| `rmax` | Meaning depends on `repair_algorithm`. For `'structured_sd_repair'`: the maximum repair dimension explored — the cap on how many active rows of `Q_hat` it's willing to hypothesize as wrong and try to correct in one repair attempt. For `'induced_sd_repair'` and `'posterior_aware_prange_repair'`: not used directly — instead it sets the decoding radius `tau = 2 * rmax` (Corollary 1). |
| `weight_factor` | Multiplier applied to the Gilbert-Varshamov bound to get the enumerator's target weight: `target_weight = round(gv_bound * weight_factor)`. |
| `budgets` | Candidate-list size per active row. For `'structured_sd_repair'`: passed to `build_active_row_lists` (how many replacement-row hypotheses are kept per row). For `'posterior_aware_prange_repair'`: reused as `K_B`, the per-row coordinate budget for the marginal error region. Unused by `'induced_sd_repair'`. |
| `enum_version` | Which low-weight-codeword enumerator was used: `'random'` (rejection sampling), `'linear_combination'`, or `'local_search'`. |
| `repair_algorithm` | Which repair module (from the paper's Algorithm 1 "Repair" step) was used: `'induced_sd_repair'` (Algorithm 6, generic Prange-based induced syndrome decoding), `'structured_sd_repair'` (Algorithm 8, exhaustive hypothesis search over active-row replacements — combinatorial in the active support size and `rmax`, can become very slow at large `n`), or `'posterior_aware_prange_repair'` (Algorithm 12, posterior-biased Prange trials — a fixed trial budget regardless of `n` or `rmax`). |
| `trial` | Repetition index within this parameter combination (0-indexed). Used to average/vary runs under otherwise identical settings. |
| `seed` | The deterministic seed (derived from all the parameters above) that `set_random_seed` was called with — makes the run reproducible. |
| `timestamp` | UTC timestamp (ISO 8601) of when the run finished. |
| `m_pair_target` | Target number of candidate equivalent codeword pairs the framework tries to recover (`m_pair` in Algorithm 1). |
| `n_iter` | Maximum number of framework iterations (enumeration + repair attempts) before giving up. |
| `max_trials_enum` | Maximum trials allowed per call to the enumerator. |

## Timing (all in seconds)

| Column | Meaning |
|---|---|
| `gen_time_s` | Time to generate the noisy LCE instance (`G1`, `G2`, `Q`, `Q_noisy`). |
| `posterior_time_s` | Time to build the BBLM posterior table from the noisy hint. |
| `qhat_time_s` | Time to build the monomial approximation `Q_hat` (local row scoring + Hungarian assignment). Dominates at large `n` (`O(n^3)`). |
| `repair_time_s` | Time spent in the instrumented prediction-and-repair loop (enumeration + repair attempts combined). |
| `total_time_s` | Total wall-clock time for the whole run, including error handling overhead. |

## `Q_hat` recovery quality

| Column | Meaning |
|---|---|
| `rows_correct_full` | Number of rows where `Q_hat` exactly matches the true secret `Q` (correct column **and** correct scalar). |
| `rows_correct_support_only` | Number of rows where `Q_hat` has the correct column (support) but the wrong scalar value — partial recovery. |
| `n_rows_total` | Total number of rows (`= n`), included for convenience when normalizing the two counts above. |

## Target-weight / minimum-distance diagnostics

| Column | Meaning |
|---|---|
| `gv_bound` | The Gilbert-Varshamov bound for this `(n, k, q)` — a cheap, closed-form estimate of an achievable minimum distance (same value for `C` and `C'`, since it depends only on `n, k, q`). |
| `target_weight_used` | The actual target weight passed to the enumerator: `round(gv_bound * weight_factor)`. |
| `uniqueness_guaranteed_gv` | Boolean. Whether `gv_bound > 4 * rmax` — the (GV-bound-approximated) condition from Corollary 2 in the paper under which a repair, once found, is guaranteed unique/correct. A diagnostic only; the pipeline still runs and may still succeed even when this is `False`. |

## Enumeration outcome

| Column | Meaning |
|---|---|
| `iters_used` | Number of framework iterations actually used before stopping (success or exhaustion). |
| `enum_failures` | Number of times the enumerator failed to find a codeword of the required weight within `max_trials_enum`. |

## Repair outcome

| Column | Meaning |
|---|---|
| `pairs_recovered` | Number of candidate pairs `(v, w)` returned by the repair step (i.e. the repair module selected by `repair_algorithm` didn't return `None`). |
| `pairs_correct` | Of those, how many are genuinely correct — verified as `w == v * Q` against the true secret (only possible in this controlled experiment). |
| `avg_active_error_dim` | Mean true active error dimension `r(v) = \|E(v)\|` (Definition 3 in the paper) over every codeword `v` the enumerator actually produced during this run. Diagnostic only — computed using the true secret `Q`, and independent of `repair_algorithm` (it characterizes the instance, not the repair attempt). |
| `max_active_error_dim` | Maximum `r(v)` observed among those same codewords. |
| `frac_r_le_rmax` | Fraction of those sampled codewords whose `r(v) <= rmax` — i.e., how much of the sampled traffic actually falls within what `rmax` can theoretically repair (Theorem 1's completeness condition). Most directly meaningful for `repair_algorithm == 'structured_sd_repair'`, where `rmax` is literally the explored repair dimension; for the other two algorithms it's still a useful reference point (`rmax` there sets `tau = 2 * rmax`) but not a strict completeness guarantee. |
| `avg_v_weight` | Mean Hamming weight of `v` across the **recovered** pairs only (not all sampled codewords). |

## Overall result

| Column | Meaning |
|---|---|
| `success` | Boolean. `True` only if `pairs_recovered >= m_pair_target` **and** every recovered pair is genuinely correct (`pairs_correct == pairs_recovered`) **and** at least one pair was found. This is a strict criterion — syndrome-consistent-but-wrong repairs don't count (all three repair modules under `repair_algorithm` include a weight-consistency filter that rejects those). |
| `error` | Empty string on a normal run. If any exception was raised during the run, its type and message are recorded here and `success` is forced to `False`, instead of the whole sweep crashing. |
