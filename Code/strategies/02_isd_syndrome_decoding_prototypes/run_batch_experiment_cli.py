"""
run_batch_experiment_cli.py

Strategy 02 (ISD syndrome-decoding prototypes): thin CLI wrapper around
lee_brickell_batch_experiments.run_parameter_set - takes (n, k,
num_trials) as command-line arguments, runs that many Lee-Brickell ISD
trials, and prints the result as a single JSON line on stdout (all other
logging goes to stderr, so stdout stays parseable). Used as the
subprocess entry point by run_batch_experiments_on_modal.py for remote
execution.

Usage: sage --python run_batch_experiment_cli.py <n> <k> <num_trials>

(Renamed from runner.py, and lee_brickell_batch_experiments.py was
renamed from experiments_lee_brickell.py, during the Aug 2026 folder
reorganization - the import below was updated to match.)
"""
import json
import sys

print("Starting runner...", file=sys.stderr, flush=True)

from lee_brickell_batch_experiments import run_parameter_set

print("Imported experiment module", file=sys.stderr, flush=True)

n = int(sys.argv[1])
k = int(sys.argv[2])
num_trials = int(sys.argv[3])

print(
    f"Running n={n}, k={k}, trials={num_trials}",
    file=sys.stderr,
    flush=True,
)

results = run_parameter_set(
    n=n,
    k=k,
    num_trials=num_trials,
)

print("Finished experiment", file=sys.stderr, flush=True)

# ONLY JSON goes to stdout
print(json.dumps(results))