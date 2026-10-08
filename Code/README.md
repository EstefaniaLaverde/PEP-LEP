# Code/ — map of strategies

This folder holds every strategy/approach tried for the LEP/PEP (Linear/Permutation
Code Equivalence Problem) thesis work, reorganized (Aug 2026) so it's clear what
each piece is, where it came from, and whether it's still active.

```
Code/
  core/                                        shared library, imported by most strategies
  strategies/
    01_nqueens_backtracking/                   early backtracking approach for PEP
    02_isd_syndrome_decoding_prototypes/       standalone Prange/Lee-Brickell ISD exploration
    03_gv_bound_instance_sizing/               GV-bound / target-weight sanity checks
    04_error_distribution_analysis/            noise/error distribution analysis
    05_full_parameter_sweep/                   the big 3-enumerator x 3-repair-algorithm sweep
    06_planted_codeword_repair_isolation/      isolates the repair stage from the search problem
    07_list_based_prediction_and_repair/       Q as two lists (permutation, values) + Algorithm 5
  archive/                                     superseded code, kept for history only
```

## core/

The shared, actively-maintained library that most strategies import from.

- **`instances_generator.py`** — generates random LEP/PEP instances (generator matrices,
  random monomial/permutation secrets, noisy hints under two channel models).
- **`LEP_prediction_and_repair_v2.py`** — the current prediction-and-repair framework
  (BBLM posterior table, monomial approximation `Q_hat`, and the three repair
  algorithms: `induced_sd_repair`, `structured_sd_repair`, `posterior_aware_prange_repair`).
  Implements Algorithms 1-12 from the paper. The "_v2" is historical (a v1 exists,
  archived) — this is simply the current version.

If you're starting a new strategy that needs an LEP/PEP instance or the
prediction-and-repair pipeline, import from here rather than writing a new version.

## strategies/

Each folder is one distinct approach or question explored. **When you start testing
something new, add a new numbered folder here** (`07_your_strategy_name/`) rather than
dropping files into `Code/` directly — that's the whole point of this reorganization.
Give it a short, descriptive name (not just "test3"), and add a one-line entry to this
README plus a summary docstring/markdown cell at the top of each script/notebook
inside it, the same way every existing strategy folder does.

### 01_nqueens_backtracking/
Solves PEP via backtracking, inspired by the n-queens problem (treats permutation-matrix
placement like placing non-attacking queens, without the diagonal constraint).
Standalone — does not depend on `core/`. Early exploratory approach, superseded in
spirit by the posterior-guided framework in `core/`, but functionally independent.
- `nqueens_pep_solver.py` — the solver itself.
- `nqueens_parameter_sweep.ipynb` — parameter sweep over the solver.

### 02_isd_syndrome_decoding_prototypes/
Standalone exploration of turning PEP/LEP into a syndrome-decoding problem and solving
it with classical Information Set Decoding (Prange, then Lee-Brickell), before this
got folded into the posterior-guided framework. Predates `core/`.
- `prange_isd_exploration.ipynb` — plain Prange ISD, first pass.
- `lee_brickell_isd_exploration.ipynb` / `lee_brickell_isd_exploration_script.py` —
  Lee-Brickell ISD (a generalization of Prange), notebook + equivalent script.
- `lee_brickell_batch_experiments.py` — batch-runs Lee-Brickell ISD trials for a
  given `(n, k)` and reports success statistics.
- `run_batch_experiment_cli.py` — CLI wrapper around the above (used as a subprocess
  entry point).
- `run_batch_experiments_on_modal.py` — same experiments, fanned out as parallel
  jobs on Modal (modal.com). **Not verified since the Aug 2026 reorg** — see the
  caveat in its docstring before relying on it.

### 03_gv_bound_instance_sizing/
Sanity-checks the Gilbert-Varshamov bound and Hamming-ball-volume formulas against
the framework's actual behavior, to pick reasonable target weights before running a
bigger sweep.
- `gv_bound_sanity_checks.ipynb`

### 04_error_distribution_analysis/
Analyzes the empirical distribution of the bit-flip channel's induced errors
(`alpha`/`beta`) across different code sizes, and how well the empirical rates match
the requested channel parameters.
- `error_distribution_analysis.ipynb`
- `error_distribution_analysis_rref.ipynb` — same analysis, RREF-normalized variant.

### 05_full_parameter_sweep/
The main experiment sweep: runs the full pipeline (instance generation → posterior →
`Q_hat` → prediction-and-repair) across combinations of `(n, k, q, alpha, beta, rmax,
weight_factor, budgets, enum_version, repair_algorithm)`, checkpointing every run to
CSV.
- `run_lep_experiments.py` — the sweep driver.
- `visualize_sweep_results.py` — generates plots + a summary table from a results CSV.
- `results/` — every results CSV produced by a sweep, plus `results_guide.md`
  (a full column-by-column reference for the CSV schema).
- `plots_large_tests/` — plots from the large-`(n,k)` sweep (was `plots_large_combinations/`).
- `plots_small_tests_v1/` — plots from an earlier small-test run (was `result_plots/`).
- `plots_small_tests_v2/` — plots from a later small-test run, same day (was `plots/`).
  (`v1`/`v2` here just reflect which was generated first — check `results/` for the
  exact CSV each corresponds to if you need to trace it further.)

### 06_planted_codeword_repair_isolation/
Isolates the repair stage from the low-weight-codeword *search* problem, by planting
a codeword of the target weight directly into the generator matrix (guaranteed to
exist by construction) instead of searching a random code for one — useful because
that search is itself a hard problem (near the Gilbert-Varshamov bound, ISD-based
search can legitimately fail or take a long time, independent of whether the repair
stage works). The most recent/active strategy folder as of this reorg.
- `search_low_weight_codeword_sage_isd.py` — an attempt at using Sage's built-in
  `LinearCodeInformationSetDecoder` as a real (non-heuristic) enumerator; falls back
  to `enum_low_weight_codeword_lee_brickell` (from `core/`) since the direct Sage-ISD
  call turned out not to enforce a minimum error count (see its docstring).
- `planted_codeword_repair_test.py` — the planted-codeword construction + repair test
  itself; saves every run's full instance and outcome to `runs/` when `SAVE_RUN = True`.
- `visualize_runs.ipynb` — reads `runs/*.json` for an aggregate view across runs, and
  `runs/*.sobj` (via Sage's `load()`) for a deep dive into one run's actual matrices/vectors.
- `lincodeweightinv_analysis.ipynb` — loads a saved run's `G1` and runs it through
  **LinCodeWeightInv** (ACM TOMS Algorithm 1059, external library in `../../../1059/`,
  not part of this repo) via the small CLI bridge `1059/Src/testProgram/LinCodeBridge.cpp`:
  either the full weight spectrum (bar-plotted) or a count/search for codewords below the
  run's `target_weight`. See the notebook's own setup cell for the one-time build steps,
  and `LinCodeBridge.cpp`'s header comment for why it doesn't use LinCodeWeightInv's own
  file format (a digit-width bug there corrupts `q=127` matrices).
- `runs/` — saved run data (`.sobj` + `.json` pairs), one per `run_instance()` call.

### 07_list_based_prediction_and_repair/
Explores representing the secret monomial `Q` as two lists (a permutation list and a
diagonal-values list) instead of the full n x n matrix, per the thesis advisor's suggestion,
and reconstructs it with the paper's dedicated Algorithm 5 (`MonomialApproximationVector`,
Section 5.5) rather than routing it through the full-matrix Algorithm 6.
- `list_based_posterior_pseudocode.md` - design notes for the entrywise-table detour
  (`compute_posterior_table_list_based`), including the modulus-wraparound treatment of
  out-of-range leaked observations and its validation (numpy-mirrored, not yet run with Sage).
- `planted_codeword_repair_test_list_based.py` - list-based counterpart of
  `06_planted_codeword_repair_isolation/planted_codeword_repair_test.py`, using
  `compute_posterior_table_list_based` + `monomial_approximation` (the entrywise-table route,
  i.e. Algorithm 6 fed a table built from the two lists).
- `planted_codeword_repair_test_list_based_algo5.py` - sibling to the above, using
  `compute_vector_posteriors_list_based` + `monomial_approximation_vector`
  (`core/LEP_prediction_and_repair_v2.py`) instead - the paper's actual decoupled Algorithm 5,
  kept separate so the two reconstruction paths can be compared side by side on the same
  planted-codeword construction and repair stage.
- `algorithm5_demo.ipynb` - small interactive notebook (SageMath kernel) for testing Algorithm
  5 on its own at a size you can actually look at: builds a small instance, leaks it, runs
  `monomial_approximation_vector`, and visualizes `Q` vs. `Q_hat` and the permutation
  posteriors; also compares against the naive "matrix straight from the two vectors"
  construction fed through the original Algorithm 6 pipeline.

## archive/

Superseded code, kept only for history — nothing in `core/` or `strategies/`
imports from here.

- `lep_prediction_and_repair_v1_superseded.py` — the original prediction-and-repair
  implementation, before `core/LEP_prediction_and_repair_v2.py`.
- `early_prototypes/` — four generations of exploratory utility modules
  (`prototype_utils_v1.py` through `v4.py`, originally `idea/utils.py` and
  `idea2/utils.py`/`utilsv2.py`/`utilsv3.py`) plus `find_compatible_pairs_exploration.ipynb`,
  from before `core/instances_generator.py` existed.

## A note on imports

Files under `strategies/*/` that import from `core/` add a small `sys.path` bootstrap
near the top (look for the "folder reorganization" comment) so the import resolves
regardless of what directory Sage/Python is invoked from. If you add a new script that
needs `core/`, copy that pattern rather than a bare `from instances_generator import ...`.
