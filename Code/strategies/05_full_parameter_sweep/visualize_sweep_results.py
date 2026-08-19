"""
Visualization script for the LEP prediction-and-repair experiment sweep
results (the CSV produced by run_lep_experiments.py). Generates a set of
PNG plots plus a numeric summary table, covering timing and success rate
first (the two most important things), the specific axes this sweep is
built to study (rmax, weight_factor, budgets, and the three enumerator
versions), and a dedicated set of plots examining whether rmax behaves as
a function of the noise level (alpha, beta): a heatmap of the true active
error dimension r(v) over (alpha, beta), a heatmap of the empirically
required rmax per (alpha, beta), a scatter with a fitted line, and printed
Pearson correlation coefficients.

This script only needs pandas and matplotlib - it does NOT need SageMath,
since it just reads/aggregates the CSV. Run it with plain `python3`:

    python3 visualize_lep_results.py
    python3 visualize_lep_results.py --csv results/lep_experiment_results.csv --outdir plots

Every plotting function degrades gracefully (skips itself with a printed
note) if the columns/axis it needs are missing or have fewer than 2
distinct values in the loaded data, so this also works fine against
partial/quick-test runs, not just a full sweep.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # no display needed; just save PNGs
import matplotlib.pyplot as plt


NUMERIC_COLUMNS = [
    'n', 'k', 'q', 'alpha', 'beta', 'rmax', 'weight_factor', 'budgets',
    'trial', 'seed', 'm_pair_target', 'n_iter', 'max_trials_enum',
    'gen_time_s', 'posterior_time_s', 'qhat_time_s', 'repair_time_s', 'total_time_s',
    'rows_correct_full', 'rows_correct_support_only', 'n_rows_total',
    'gv_bound', 'target_weight_used',
    'iters_used', 'enum_failures',
    'pairs_recovered', 'pairs_correct',
    'avg_active_error_dim', 'max_active_error_dim', 'frac_r_le_rmax',
    'avg_v_weight',
]
BOOL_COLUMNS = ['success', 'uniqueness_guaranteed_gv']

TIME_COLUMNS = ['gen_time_s', 'posterior_time_s', 'qhat_time_s', 'repair_time_s', 'total_time_s']


# ---------------------------------------------------------------------------
# Loading and typing
# ---------------------------------------------------------------------------

def load_results(csv_path):
    """
    Loads the experiment results CSV and coerces columns to their proper
    types: numeric columns to floats (pd.to_numeric with errors='coerce',
    so blanks/None become NaN rather than crashing), boolean columns from
    their 'True'/'False' string form to actual booleans.

    :param csv_path: path to the CSV produced by run_lep_experiments.py
    :return: a pandas DataFrame with typed columns
    """
    df = pd.read_csv(csv_path)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in BOOL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower() == 'true'

    # Derived convenience column: fraction of Q_hat rows fully correct.
    if 'rows_correct_full' in df.columns and 'n_rows_total' in df.columns:
        df['frac_rows_correct_full'] = df['rows_correct_full'] / df['n_rows_total']

    return df


def _has_enough_data(df, cols, min_unique=1):
    """
    Checks that every column in `cols` exists in df and has at least one
    non-null value, and (for the first column) at least `min_unique`
    distinct values - used to skip a plot cleanly instead of crashing on
    sparse/partial result sets.
    """
    for col in cols:
        if col not in df.columns or df[col].dropna().empty:
            return False
    if min_unique > 1 and df[cols[0]].dropna().nunique() < min_unique:
        return False
    return True


# ---------------------------------------------------------------------------
# Generic grouped-bar helper, reused by several of the success-rate plots
# below (x-axis = group_col, one bar per distinct hue_col value).
# ---------------------------------------------------------------------------

def _grouped_bar(df, group_col, hue_col, value_col, outpath, title, ylabel,
                  ylim=None, agg='mean'):
    grouped = df.groupby([hue_col, group_col])[value_col].agg(agg).reset_index()
    hue_values = sorted(grouped[hue_col].dropna().unique(), key=str)
    group_values = sorted(grouped[group_col].dropna().unique(), key=str)
    if not hue_values or not group_values:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / max(len(hue_values), 1)
    x_base = range(len(group_values))

    for i, hv in enumerate(hue_values):
        sub = grouped[grouped[hue_col] == hv].set_index(group_col).reindex(group_values)
        offsets = [x + i * width for x in x_base]
        ax.bar(offsets, sub[value_col].values, width=width, label=f'{hue_col}={hv}')

    ax.set_xticks([x + width * (len(hue_values) - 1) / 2 for x in x_base])
    ax.set_xticklabels([str(g) for g in group_values], rotation=20, ha='right')
    ax.set_xlabel(group_col)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Success-rate plots
# ---------------------------------------------------------------------------

def plot_success_rate_by_rmax(df, outdir):
    """Success rate vs rmax, one bar group per distinct n."""
    if not _has_enough_data(df, ['rmax', 'n', 'success']):
        return
    _grouped_bar(
        df, group_col='rmax', hue_col='n', value_col='success',
        outpath=os.path.join(outdir, 'success_rate_by_rmax.png'),
        title='Success rate by rmax (grouped by n)', ylabel='Success rate', ylim=(0, 1.05),
    )


def plot_success_rate_by_weight_factor(df, outdir):
    """Success rate vs weight_factor (target_weight = round(gv_bound * factor))."""
    if not _has_enough_data(df, ['weight_factor', 'rmax', 'success']):
        return
    _grouped_bar(
        df, group_col='weight_factor', hue_col='rmax', value_col='success',
        outpath=os.path.join(outdir, 'success_rate_by_weight_factor.png'),
        title='Success rate by weight_factor (grouped by rmax)',
        ylabel='Success rate', ylim=(0, 1.05),
    )


def plot_success_rate_by_budgets(df, outdir):
    """Success rate vs budgets (candidate-list size per active row)."""
    if not _has_enough_data(df, ['budgets', 'rmax', 'success']):
        return
    _grouped_bar(
        df, group_col='budgets', hue_col='rmax', value_col='success',
        outpath=os.path.join(outdir, 'success_rate_by_budgets.png'),
        title='Success rate by budgets (grouped by rmax)',
        ylabel='Success rate', ylim=(0, 1.05),
    )


def plot_success_rate_by_enum_version(df, outdir):
    """Success rate vs enumerator version, one bar group per distinct n."""
    if not _has_enough_data(df, ['enum_version', 'n', 'success']):
        return
    _grouped_bar(
        df, group_col='enum_version', hue_col='n', value_col='success',
        outpath=os.path.join(outdir, 'success_rate_by_enum_version.png'),
        title='Success rate by enumerator version (grouped by n)',
        ylabel='Success rate', ylim=(0, 1.05),
    )


def plot_success_rate_heatmap_alpha_beta(df, outdir):
    """Heatmap of mean success rate over (alpha, beta), averaged over everything else."""
    if not _has_enough_data(df, ['alpha', 'beta', 'success']):
        return
    pivot = df.pivot_table(index='beta', columns='alpha', values='success', aggfunc='mean')
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, aspect='auto', origin='lower', vmin=0, vmax=1, cmap='viridis')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(a) for a in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(b) for b in pivot.index])
    ax.set_xlabel('alpha (Pr[1 -> 0])')
    ax.set_ylabel('beta (Pr[0 -> 1])')
    ax.set_title('Success rate by (alpha, beta)')
    fig.colorbar(im, ax=ax, label='Success rate')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'success_rate_alpha_beta_heatmap.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Timing plots
# ---------------------------------------------------------------------------

def plot_timing_by_n(df, outdir):
    """Mean time per phase (generation, posterior, Q_hat, repair, total) vs n, log-scale y."""
    present_time_cols = [c for c in TIME_COLUMNS if c in df.columns]
    if not present_time_cols or not _has_enough_data(df, ['n'] + present_time_cols):
        return

    grouped = df.groupby('n')[present_time_cols].mean().reset_index().sort_values('n')

    fig, ax = plt.subplots(figsize=(8, 5))
    for col in present_time_cols:
        ax.plot(grouped['n'], grouped[col], marker='o', label=col)

    ax.set_yscale('log')
    ax.set_xlabel('n')
    ax.set_ylabel('Time (s, log scale)')
    ax.set_title('Mean time per phase vs n')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'timing_by_n.png'), dpi=150)
    plt.close(fig)


def plot_repair_time_by_rmax(df, outdir):
    """Boxplot of repair_time_s grouped by rmax."""
    if not _has_enough_data(df, ['rmax', 'repair_time_s']):
        return
    rmax_values = sorted(df['rmax'].dropna().unique())
    data = [df.loc[df['rmax'] == r, 'repair_time_s'].dropna().values for r in rmax_values]
    if not any(len(d) for d in data):
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        ax.boxplot(data, tick_labels=[str(int(r)) for r in rmax_values])
    except TypeError:
        # Older matplotlib (<3.9) does not have tick_labels yet.
        ax.boxplot(data, labels=[str(int(r)) for r in rmax_values])
    ax.set_xlabel('rmax')
    ax.set_ylabel('repair_time_s')
    ax.set_title('Repair time distribution by rmax')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'repair_time_by_rmax.png'), dpi=150)
    plt.close(fig)


def plot_total_time_by_enum_version(df, outdir):
    """Boxplot of total_time_s grouped by enum_version - cost comparison across enumerators."""
    if not _has_enough_data(df, ['enum_version', 'total_time_s']):
        return
    versions = sorted(df['enum_version'].dropna().unique())
    data = [df.loc[df['enum_version'] == v, 'total_time_s'].dropna().values for v in versions]
    if not any(len(d) for d in data):
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        ax.boxplot(data, tick_labels=versions)
    except TypeError:
        ax.boxplot(data, labels=versions)
    ax.set_xlabel('enum_version')
    ax.set_ylabel('total_time_s')
    ax.set_title('Total run time distribution by enumerator version')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'total_time_by_enum_version.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Enumerator-efficiency plots
# ---------------------------------------------------------------------------

def plot_enum_failures_by_enum_version(df, outdir):
    """Mean enum_failures vs enumerator version, grouped by n - how often each enumerator gives up."""
    if not _has_enough_data(df, ['enum_version', 'n', 'enum_failures']):
        return
    _grouped_bar(
        df, group_col='enum_version', hue_col='n', value_col='enum_failures',
        outpath=os.path.join(outdir, 'enum_failures_by_enum_version.png'),
        title='Mean enumeration failures by enumerator version (grouped by n)',
        ylabel='Mean enum_failures',
    )


# ---------------------------------------------------------------------------
# rmax / active-error-dimension diagnostics: the core "effect of rmax" plot
# ---------------------------------------------------------------------------

def plot_active_error_dim_vs_rmax(df, outdir):
    """
    Mean active error dimension r(v) and the fraction of sampled v with
    r(v) <= rmax, both plotted against rmax. This is the direct picture of
    whether rmax is actually large enough to cover the errors Q_hat makes
    (Theorem 1's completeness condition).
    """
    if not _has_enough_data(df, ['rmax', 'avg_active_error_dim', 'frac_r_le_rmax']):
        return

    grouped = df.groupby('rmax')[['avg_active_error_dim', 'frac_r_le_rmax']].mean().reset_index()
    grouped = grouped.sort_values('rmax')

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(grouped['rmax'], grouped['avg_active_error_dim'], marker='o', color='tab:blue',
              label='avg active error dim r(v)')
    ax1.set_xlabel('rmax')
    ax1.set_ylabel('avg active error dim r(v)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(grouped['rmax'], grouped['frac_r_le_rmax'], marker='s', color='tab:red',
              label='fraction of v with r(v) <= rmax')
    ax2.set_ylabel('fraction of v with r(v) <= rmax', color='tab:red')
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.suptitle('Active error dimension vs rmax')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'active_error_dim_vs_rmax.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# rmax vs (alpha, beta): does rmax behave like a function of the noise
# level? Three complementary views of the same question.
# ---------------------------------------------------------------------------

def plot_active_error_dim_alpha_beta_heatmap(df, outdir):
    """
    Heatmap of the mean active error dimension r(v) over (alpha, beta).
    r(v) does not depend on which rmax was used to attempt repair (Q_hat
    is built once, before rmax matters) - it is purely a property of Q_hat's
    quality, which is driven by the noise level. By Theorem 1, rmax needs
    to be at least this large (for the relevant codewords) for
    structured_sd_repair to be guaranteed to find the true repair. This is
    the mechanistic link: if rmax should scale with (alpha, beta), it is
    because r(v) does, and this plot shows whether that premise holds.
    """
    if not _has_enough_data(df, ['alpha', 'beta', 'avg_active_error_dim']):
        return
    pivot = df.pivot_table(index='beta', columns='alpha', values='avg_active_error_dim', aggfunc='mean')
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(a) for a in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(b) for b in pivot.index])
    ax.set_xlabel('alpha (Pr[1 -> 0])')
    ax.set_ylabel('beta (Pr[0 -> 1])')
    ax.set_title('Mean active error dimension r(v) by (alpha, beta)')
    fig.colorbar(im, ax=ax, label='Mean r(v)')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'active_error_dim_alpha_beta_heatmap.png'), dpi=150)
    plt.close(fig)


def plot_required_rmax_heatmap(df, outdir, success_threshold=0.5):
    """
    For every (alpha, beta) combination present in the data, finds the
    smallest tested rmax whose mean success rate is >= success_threshold,
    and plots that as a heatmap over (alpha, beta). This is the most
    direct empirical answer to "can rmax be read off from alpha and beta?"
    - it shows what rmax was actually needed at each noise level, using
    only the rmax values that were actually swept. Cells where no tested
    rmax reached the threshold are left blank.

    :param success_threshold: the minimum mean success rate to count as
        "reached" for a given rmax (default 50%)
    """
    if not _has_enough_data(df, ['alpha', 'beta', 'rmax', 'success']):
        return

    grouped = df.groupby(['alpha', 'beta', 'rmax'])['success'].mean().reset_index()

    def _min_rmax_reaching_threshold(sub):
        ok = sub[sub['success'] >= success_threshold]
        return ok['rmax'].min() if not ok.empty else np.nan

    required = (
        grouped.groupby(['alpha', 'beta'])
        .apply(_min_rmax_reaching_threshold, include_groups=False)
        .reset_index(name='required_rmax')
    )
    pivot = required.pivot_table(index='beta', columns='alpha', values='required_rmax')
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    masked = np.ma.masked_invalid(pivot.values)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='lightgray')
    im = ax.imshow(masked, aspect='auto', origin='lower', cmap=cmap)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(a) for a in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(b) for b in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            text = 'NA' if np.isnan(val) else f'{val:.0f}'
            ax.text(j, i, text, ha='center', va='center', color='white')
    ax.set_xlabel('alpha (Pr[1 -> 0])')
    ax.set_ylabel('beta (Pr[0 -> 1])')
    ax.set_title(f'Minimum tested rmax reaching success rate >= {success_threshold:.0%}\n(gray = no tested rmax reached it)')
    fig.colorbar(im, ax=ax, label='Minimum rmax')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'required_rmax_by_alpha_beta.png'), dpi=150)
    plt.close(fig)


def plot_active_error_dim_vs_noise_scatter(df, outdir):
    """
    Scatter of avg_active_error_dim (per run) against alpha + beta (total
    per-bit flip probability), with a least-squares line overlaid - a
    direct visual and numeric check of the relationship's shape (roughly
    linear? saturating? no relationship at all?).
    """
    if not _has_enough_data(df, ['avg_active_error_dim', 'alpha', 'beta']):
        return
    sub = df[['avg_active_error_dim', 'alpha', 'beta']].dropna().copy()
    if len(sub) < 2:
        return
    sub['noise_total'] = sub['alpha'] + sub['beta']

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sub['noise_total'], sub['avg_active_error_dim'], alpha=0.4, s=15)

    coeffs = np.polyfit(sub['noise_total'], sub['avg_active_error_dim'], deg=1)
    xs = np.linspace(sub['noise_total'].min(), sub['noise_total'].max(), 50)
    ax.plot(xs, coeffs[0] * xs + coeffs[1], color='red',
            label=f'fit: r(v) ~= {coeffs[0]:.2f}*(alpha+beta) + {coeffs[1]:.2f}')
    ax.legend(fontsize=8)

    ax.set_xlabel('alpha + beta (total per-bit flip probability)')
    ax.set_ylabel('avg active error dimension r(v)')
    ax.set_title('Active error dimension vs total noise level')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'active_error_dim_vs_noise_scatter.png'), dpi=150)
    plt.close(fig)


def print_alpha_beta_rmax_correlation(df):
    """
    Prints Pearson correlation coefficients between the active error
    dimension r(v) - the quantity rmax must cover, per Theorem 1 - and each
    of alpha, beta, and alpha+beta, as a quick numeric check of whether
    "rmax should scale with alpha/beta" holds up in the collected data.
    A coefficient near 0 means no linear relationship was found; near +1
    means r(v) grows roughly linearly with that quantity.
    """
    if not {'avg_active_error_dim', 'alpha', 'beta'}.issubset(df.columns):
        return
    sub = df[['avg_active_error_dim', 'alpha', 'beta']].dropna()
    if len(sub) < 3:
        return

    corr_alpha = sub['avg_active_error_dim'].corr(sub['alpha'])
    corr_beta = sub['avg_active_error_dim'].corr(sub['beta'])
    corr_sum = sub['avg_active_error_dim'].corr(sub['alpha'] + sub['beta'])

    print("\nCorrelation between mean active error dimension r(v) and noise parameters:")
    print(f"  corr(r(v), alpha)      = {corr_alpha:+.3f}")
    print(f"  corr(r(v), beta)       = {corr_beta:+.3f}")
    print(f"  corr(r(v), alpha+beta) = {corr_sum:+.3f}")


# ---------------------------------------------------------------------------
# Additional summary plots
# ---------------------------------------------------------------------------

def plot_success_rate_by_n(df, outdir):
    """Success rate vs n, one bar group per q - overall scaling behavior."""
    if not _has_enough_data(df, ['n', 'q', 'success']):
        return
    _grouped_bar(
        df, group_col='n', hue_col='q', value_col='success',
        outpath=os.path.join(outdir, 'success_rate_by_n.png'),
        title='Success rate by n (grouped by q)', ylabel='Success rate', ylim=(0, 1.05),
    )


def plot_timing_scaling_fit(df, outdir):
    """
    Log-log plot of qhat_time_s and total_time_s against n, with a fitted
    power-law exponent annotated for each (slope of the log-log line) -
    an empirical check of the O(n^3) claim for Q_hat construction.
    """
    cols = [c for c in ['qhat_time_s', 'total_time_s'] if c in df.columns]
    if not cols or not _has_enough_data(df, ['n'] + cols, min_unique=2):
        return

    grouped = df.groupby('n')[cols].mean().reset_index().sort_values('n')
    grouped = grouped[(grouped['n'] > 0)]
    for c in cols:
        grouped = grouped[grouped[c] > 0]
    if len(grouped) < 2:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    log_n = np.log(grouped['n'])
    for col in cols:
        log_t = np.log(grouped[col])
        slope, intercept = np.polyfit(log_n, log_t, deg=1)
        ax.plot(grouped['n'], grouped[col], marker='o', label=f'{col} (fit slope ~= {slope:.2f})')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('n (log scale)')
    ax.set_ylabel('Time (s, log scale)')
    ax.set_title('Timing scaling vs n (fitted log-log slope ~= polynomial degree)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'timing_scaling_fit.png'), dpi=150)
    plt.close(fig)


def plot_enum_efficiency_by_version(df, outdir):
    """
    Mean pairs_recovered / iters_used (a rough "pairs found per iteration
    spent" efficiency measure) by enumerator version, grouped by n.
    """
    needed = {'enum_version', 'n', 'pairs_recovered', 'iters_used'}
    if not needed.issubset(df.columns):
        return
    sub = df.copy()
    sub = sub[sub['iters_used'] > 0]
    if sub.empty:
        return
    sub['pairs_per_iter'] = sub['pairs_recovered'] / sub['iters_used']

    if not _has_enough_data(sub, ['enum_version', 'n', 'pairs_per_iter']):
        return
    _grouped_bar(
        sub, group_col='enum_version', hue_col='n', value_col='pairs_per_iter',
        outpath=os.path.join(outdir, 'enum_efficiency_by_version.png'),
        title='Pairs recovered per iteration by enumerator version (grouped by n)',
        ylabel='Mean pairs_recovered / iters_used',
    )


def plot_numeric_correlation_heatmap(df, outdir):
    """
    General-purpose Pearson correlation heatmap between success (as 0/1)
    and every swept numeric parameter / outcome metric, as an at-a-glance
    "what actually matters" view complementing the more targeted plots
    above.
    """
    candidate_cols = [
        'n', 'k', 'q', 'alpha', 'beta', 'rmax', 'weight_factor', 'budgets',
        'gv_bound', 'target_weight_used', 'avg_active_error_dim',
        'frac_r_le_rmax', 'total_time_s', 'success',
    ]
    cols = [c for c in candidate_cols if c in df.columns]
    if len(cols) < 2:
        return

    sub = df[cols].copy()
    if 'success' in sub.columns:
        sub['success'] = sub['success'].astype(float)
    sub = sub.dropna(how='all')
    if len(sub) < 3:
        return

    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center', fontsize=6)
    ax.set_title('Pearson correlation between parameters and outcomes')
    fig.colorbar(im, ax=ax, label='Correlation')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'numeric_correlation_heatmap.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Q_hat quality vs noise level: separates "did the posterior/Q_hat step do
# its job" from "did the repair step do its job".
# ---------------------------------------------------------------------------

def plot_qhat_quality_heatmap(df, outdir):
    """Heatmap of mean fraction of fully-correct Q_hat rows over (alpha, beta)."""
    if not _has_enough_data(df, ['alpha', 'beta', 'frac_rows_correct_full']):
        return
    pivot = df.pivot_table(index='beta', columns='alpha', values='frac_rows_correct_full', aggfunc='mean')
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, aspect='auto', origin='lower', vmin=0, vmax=1, cmap='viridis')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(a) for a in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(b) for b in pivot.index])
    ax.set_xlabel('alpha (Pr[1 -> 0])')
    ax.set_ylabel('beta (Pr[0 -> 1])')
    ax.set_title('Fraction of fully-correct Q_hat rows by (alpha, beta)')
    fig.colorbar(im, ax=ax, label='Fraction of rows fully correct')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'qhat_quality_alpha_beta_heatmap.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Numeric summary table (a compact complement to the plots)
# ---------------------------------------------------------------------------

def save_summary_table(df, outdir):
    """
    Saves a compact CSV summarizing, for every (n, k, q, alpha, beta, rmax,
    weight_factor, budgets, enum_version) combination present, the mean
    success rate, mean total time, mean active-error-dimension coverage,
    and how many trials that combination had.

    :param df: the results DataFrame
    :param outdir: directory to save the summary CSV into
    :return: the path to the saved summary CSV
    """
    group_cols = [c for c in [
        'n', 'k', 'q', 'alpha', 'beta', 'rmax', 'weight_factor', 'budgets', 'enum_version',
    ] if c in df.columns]

    agg_spec = {}
    if 'success' in df.columns:
        agg_spec['success'] = 'mean'
    if 'total_time_s' in df.columns:
        agg_spec['total_time_s'] = 'mean'
    if 'repair_time_s' in df.columns:
        agg_spec['repair_time_s'] = 'mean'
    if 'frac_r_le_rmax' in df.columns:
        agg_spec['frac_r_le_rmax'] = 'mean'
    if 'pairs_recovered' in df.columns:
        agg_spec['pairs_recovered'] = 'mean'
    if 'trial' in df.columns:
        agg_spec['trial'] = 'count'

    if not group_cols or not agg_spec:
        return None

    summary = df.groupby(group_cols).agg(agg_spec)
    summary = summary.rename(columns={'trial': 'num_trials'}).reset_index()
    summary_path = os.path.join(outdir, 'summary_by_combination.csv')
    summary.to_csv(summary_path, index=False)
    return summary_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # __file__-relative defaults so these resolve correctly regardless of
    # the invoking cwd; the folder reorganization (Aug 2026) also renamed
    # plots_large_combinations/ -> plots_large_tests/, result_plots/ ->
    # plots_small_tests_v1/, and plots/ -> plots_small_tests_v2/ (results/
    # kept its name). Override with --csv/--outdir for any other CSV/plot
    # batch, e.g. the small-test runs.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Visualize LEP experiment sweep results.')
    parser.add_argument('--csv', default=os.path.join(this_dir, 'results', 'lep_experiment_results_large_combinations.csv'),
                         help='Path to the results CSV produced by run_lep_experiments.py')
    parser.add_argument('--outdir', default=os.path.join(this_dir, 'plots_large_tests'),
                         help='Directory to save plots and summary into')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_results(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    if df.empty:
        print("No data to plot - the CSV is empty.")
        return

    plot_success_rate_by_rmax(df, args.outdir)
    plot_success_rate_by_weight_factor(df, args.outdir)
    plot_success_rate_by_budgets(df, args.outdir)
    plot_success_rate_by_enum_version(df, args.outdir)
    plot_success_rate_by_n(df, args.outdir)
    plot_success_rate_heatmap_alpha_beta(df, args.outdir)

    plot_timing_by_n(df, args.outdir)
    plot_timing_scaling_fit(df, args.outdir)
    plot_repair_time_by_rmax(df, args.outdir)
    plot_total_time_by_enum_version(df, args.outdir)

    plot_enum_failures_by_enum_version(df, args.outdir)
    plot_enum_efficiency_by_version(df, args.outdir)
    plot_active_error_dim_vs_rmax(df, args.outdir)
    plot_qhat_quality_heatmap(df, args.outdir)

    # rmax vs (alpha, beta): the "is rmax a function of the noise level?" plots
    plot_active_error_dim_alpha_beta_heatmap(df, args.outdir)
    plot_required_rmax_heatmap(df, args.outdir)
    plot_active_error_dim_vs_noise_scatter(df, args.outdir)

    plot_numeric_correlation_heatmap(df, args.outdir)

    summary_path = save_summary_table(df, args.outdir)

    if 'success' in df.columns:
        print(f"Overall success rate: {df['success'].mean():.2%}")
    if 'total_time_s' in df.columns:
        print(f"Mean total_time_s: {df['total_time_s'].mean():.3f}")
    print_alpha_beta_rmax_correlation(df)
    print(f"Plots saved to: {args.outdir}/")
    if summary_path:
        print(f"Summary table saved to: {summary_path}")


if __name__ == "__main__":
    main()