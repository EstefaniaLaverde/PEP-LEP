"""
run_batch_experiments_on_modal.py

Strategy 02 (ISD syndrome-decoding prototypes): remote/parallel runner
for the same Lee-Brickell ISD batch experiments as
run_batch_experiment_cli.py, using Modal (modal.com) to fan out multiple
(n, k) parameter sets as concurrent cloud jobs, each shelling out to
`sage --python run_batch_experiment_cli.py`. Run with `modal run
run_batch_experiments_on_modal.py`.

CAVEAT (Aug 2026 folder reorganization): this file used to sit flat in
Code/ alongside instances_generator.py, so `add_local_dir(".", ...)`
uploaded everything ISD-related in one shot. Now that instances_generator.py
lives in Code/core/ (a sibling of this file's strategies/ subfolder), the
upload path below was changed to `../..` (Code/'s root) so both core/ and
this strategy folder end up on the remote box, and the subprocess path was
updated to match. This has NOT been re-run against a live Modal deployment
since the move - if `modal run` fails on the image build or the subprocess
call, the most likely culprit is one of these two paths.
"""
import json
import subprocess

import modal

app = modal.App("pep-leebrickell")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "sagemath",
        "python3-sage",
    )
    .pip_install("pandas")
    # Uploads the whole Code/ directory (core/ + every strategies/ folder),
    # not just this strategy's subfolder, since lee_brickell_batch_experiments.py
    # (imported by run_batch_experiment_cli.py) now depends on
    # Code/core/instances_generator.py living outside this directory.
    .add_local_dir("../..", remote_path="/root/project")
)

@app.function(image=image, cpu=8, memory=16384, timeout=60 * 60 * 24)
def run_parameter_set_remote(n, k, num_trials):

    import subprocess

    result = subprocess.run(
        [
            "sage",
            "--python",
            "/root/project/strategies/02_isd_syndrome_decoding_prototypes/run_batch_experiment_cli.py",
            str(n),
            str(k),
            str(num_trials),
        ],
        text=True,
        capture_output=True,
    )

    print("========== RETURN CODE ==========")
    print(result.returncode)

    print("========== STDOUT ==========")
    print(repr(result.stdout))

    print("========== STDERR ==========")
    print(repr(result.stderr))

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout

@app.local_entrypoint()
def main():

    parameter_sets = [
        (7, 5),
        (50, 25),
        (100, 50),
    ]

    jobs = []

    for n, k in parameter_sets:
        jobs.append(
            run_parameter_set_remote.spawn(
                n,
                k,
                1,
            )
        )

    all_results = []

    for job in jobs:
        all_results.extend(job.get())

    import pandas as pd

    df = pd.DataFrame(all_results)

    print(df)

    df.to_csv(
        "results.csv",
        index=False,
    )