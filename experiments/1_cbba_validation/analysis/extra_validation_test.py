"""
Additional / revised benchmark plots for Chapter 3 (SC-CBBA validation).

Designed to drop into the existing generate_plots() script. They reuse the same
conventions: a `df` with an 'Algorithm' column (via label_condition), an
`algo_order` list, an `algo_palette` dict, and the save_plot(...) signature.

New figures
-----------
  Plot3.2.7  Requirement-satisfaction profile      (grouped bars + radar)
  Plot3.2.8  Revisit requirement                   (completeness + realized interval)
  Plot3.2.9  Paired benchmark advantage            (win-rate + per-scenario delta)

Revised figure
--------------
  Plot3.2.6  Response-time tradeoff                 (median + IQR instead of pooled std;
                                                     full-completion variant for 6b)

All benchmark figures here pool over the full operating envelope (all latency
levels), consistent with this subsection benchmarking planners and the
communications subsection owning the latency breakdown.
"""

from datetime import datetime
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ----------------------------------------------------------------------
#  SHARED (mirrors the main script)
# ----------------------------------------------------------------------
def label_condition(row):
    pre, rep = row['Preplanner'], row['Replanner']
    if pre == 'DP' and pd.isna(rep):     return 'DP'
    if pd.isna(pre) and rep == 'Greedy': return 'GR'
    if pre == 'DP'  and rep == 'Greedy': return 'DP-GR'
    if pd.isna(pre) and rep == 'CBBA':   return 'SC-CBBA'
    if pre == 'DP'  and rep == 'CBBA':   return 'DP-SC-CBBA'
    return 'Unknown'


def save_plot(base_dir, local_base_dir, save_dir, local_save_dir, plot_filename):
    save_path = os.path.join(save_dir, plot_filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if base_dir != local_base_dir:
        plt.savefig(os.path.join(local_save_dir, plot_filename),
                    dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


def make_boxplot(data, metric, ax, algo_order, algo_palette, ylabel=None):
    sns.boxplot(
        data=data, x='Algorithm', y=metric, order=algo_order,
        palette=algo_palette, hue='Algorithm', hue_order=algo_order,
        width=0.5, linewidth=0.8, fliersize=0, legend=False, ax=ax,
        showmeans=True, meanline=True,
        medianprops=dict(color='black', linewidth=0.8),
        meanprops=dict(color='red', linewidth=1.5, linestyle='--'),
    )
    sns.stripplot(
        data=data, x='Algorithm', y=metric, order=algo_order,
        palette=algo_palette, hue='Algorithm', hue_order=algo_order,
        size=4, alpha=0.5, jitter=True, dodge=False, legend=False, ax=ax,
        marker='o', edgecolor='auto', linewidth=0.5,
    )
    ax.set_xlabel('Algorithm Configuration')
    ax.set_ylabel(ylabel if ylabel else metric)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.4)

    ax.legend(
        handles=[Line2D([], [], color='red', linewidth=1.5, linestyle='--', label='Mean')],
        fontsize=7, loc='upper right',
    )

# Requirement-profile axes. Conditional rates are aggregated as ratio-of-sums
# from the underlying count columns (num, den) -- this avoids both the -1
# "undefined" sentinel in the per-scenario rate columns and the bias of
# averaging ratios across scenarios with very different denominators.
# Response quality has no count form: it is 1 - mean(normalized response time).
#   (display name, numerator col, denominator col or None, invert?)
# PROFILE_AXES = [
#     ('Coverage',              'Tasks Observed',           'Tasks Observable',           False),
#     ('Revisit\ncompleteness', 'Events Re-observed',       'Events Re-observable',       False),
#     ('Full\ncompletion',      'Events Fully Co-observed', 'Events Fully Co-observable',  False),
#     ('Response\nquality',     'Average Normalized Response Time to Task', None,          True),
# ]
PROFILE_AXES = [
    (r'$P(\mathrm{Task\ Observed}\mid$' + '\n' + r'$\mathrm{Observable})$',
     'Tasks Observed', 'Tasks Observable', False),
    (r'$P(\mathrm{Task\ Reobserved}\mid$' + '\n' + r'$\mathrm{Reobservable})$',
     'Events Re-observed', 'Events Re-observable', False),
    (r'$P(\mathrm{Task\ Fully\ Completed}\mid$' + '\n' + r'$\mathrm{Fully\ Completable})$',
     'Events Fully Co-observed', 'Events Fully Co-observable', False),
    ('Response Time Quality\n' + r'$1 - \bar{\tau}_{\mathrm{resp}}$',
     'Average Normalized Response Time to Task', None, True),
]



def _profile_scores(df, algo_order):
    """Achieved score per algorithm per axis (pooled over the full envelope)."""
    out = {}
    for algo in algo_order:
        g = df[df['Algorithm'] == algo]
        vals = []
        for _name, num, den, invert in PROFILE_AXES:
            if den is None:
                v = g[num].mean()
                v = (1.0 - v) if invert else v
            else:
                tot = g[den].sum()
                v = (g[num].sum() / tot) if tot else np.nan
            vals.append(v)
        out[algo] = vals
    return out


# ----------------------------------------------------------------------
#  PLOT 3.2.7 — Requirement-satisfaction profile (grouped bars)
# ----------------------------------------------------------------------
def plot_requirement_profile_bars(df, algo_order, algo_palette, save_args,
                                  filename_stem='Plot3.2.7-Requirement_Profile'):
    scores = _profile_scores(df, algo_order)
    names = [a[0] for a in PROFILE_AXES]
    rows = [{'Requirement': names[i], 'Algorithm': a, 'Score': scores[a][i]}
            for a in algo_order for i in range(len(names))]
    prof = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=prof, x='Requirement', y='Score', order=names,
                hue='Algorithm', hue_order=algo_order, palette=algo_palette,
                width=0.8, ax=ax)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('')
    ax.set_ylabel('')
    # ax.set_ylabel('Achieved score (higher is better)')
    ax.grid(True, axis='y', linestyle='--', linewidth=0.4)
    ax.legend(title='Algorithm', fontsize=8, title_fontsize=9, ncol=5,
              loc='lower center', bbox_to_anchor=(0.5, 1.02))
    ax.set_title('Performance Profile by Algorithm\n\n\n', fontsize=13)
    plt.tight_layout()
    save_plot(*save_args, f'{filename_stem}_bars.png')
    plt.close()


# ----------------------------------------------------------------------
#  PLOT 3.2.7 — Requirement-satisfaction profile (radar / strategy fingerprint)
# ----------------------------------------------------------------------
def plot_requirement_profile_radar(df, algo_order, algo_palette, save_args,
                                   filename_stem='Plot3.2.7-Requirement_Profile'):
    scores = _profile_scores(df, algo_order)
    labels = [a[0] for a in PROFILE_AXES]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 5), subplot_kw=dict(polar=True))
    for algo in algo_order:
        d = scores[algo] + scores[algo][:1]
        ax.plot(angles, d, color=algo_palette[algo], linewidth=1.8, label=algo)
        # ax.fill(angles, d, color=algo_palette[algo], alpha=0.07)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    for angle, label_text in zip(angles[:-1], labels):
        angle_deg = np.degrees(angle) % 360
        # tangent to the circle is perpendicular to the radius
        rotation = (angle_deg - 90) % 360
        # flip text that would otherwise read right-to-left / upside-down
        if 90 < rotation < 270:
            rotation = (rotation + 180) % 360
        ax.text(angle, 1.22, label_text,
                ha='center', va='center',
                rotation=rotation, rotation_mode='anchor',
                fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7)
    ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.10),
              fontsize=8, title='Algorithm')
    # ax.set_title('Performance Profile by Algorithm\n\n\n', fontsize=13)
    plt.tight_layout()
    save_plot(*save_args, f'{filename_stem}_radar.png')
    plt.close()


# ----------------------------------------------------------------------
#  PLOT 3.2.8 — Revisit requirement
# ----------------------------------------------------------------------
def _p_rev(v):
    """Revisit-time preference (Eq. pref_revisit_time): 0.1 floor below 10 s,
    steep rise to the peak at 60 s, gentle linear decay to 0 by 9000 s."""
    v = np.asarray(v, dtype=float)
    return np.where(v < 10, 0.1,
           np.where(v < 60, 0.1 + 0.9 * (v - 10) / 50.0,
           np.where(v < 9000, 1.0 - (v - 60) / 8940.0, 0.0)))

def plot_revisit(df, algo_order, algo_palette, save_args,
                 filename_stem='Plot3.2.8-Revisit'):
    comp_col = 'P(Task Reobserved | Task Reobservable)'
    int_col = 'Median Task Reobservation Time [s]'

    agg = (df.groupby(['Algorithm', 'Num Sats', 'Task Arrival Rate'])[[comp_col]]
             .mean().reset_index())

    fig, axes = plt.subplots(2, 1, figsize=(9, 10), sharex=True, layout='constrained')

    # (a) completeness: how many reobservable tasks were actually reobserved
    make_boxplot(agg, comp_col, axes[0], algo_order, algo_palette,
                 ylabel=r'$P(\mathrm{Task\ Reobserved\,|\,Reobservable})$')
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title('(a)')

    # (b) realized revisit interval; shade the preferred band (peak 60 s, 0 by 9000 s)
    # rev = df[(df[int_col].notna()) & (df[int_col] > 0)]
    # sns.boxplot(data=rev, x='Algorithm', y=int_col, order=algo_order,
    #             palette=algo_palette, hue='Algorithm', hue_order=algo_order,
    #             width=0.5, linewidth=0.8, fliersize=2, legend=False, ax=axes[1])
    # axes[1].set_yscale('log')
    # axes[1].axhspan(60, 9000, color='tab:green', alpha=0.07,
    #                 label='preferred band')
    # axes[1].axhline(60, color='tab:green', linestyle=':', linewidth=0.9)
    # axes[1].set_ylabel('Median revisit interval [s]')
    # axes[1].set_xlabel('Algorithm Configuration')
    # axes[1].tick_params(axis='x', rotation=15)
    # axes[1].grid(True, axis='y', which='both', linestyle='--', linewidth=0.4)
    # axes[1].set_title('(b) Realized revisit interval')
    # axes[1].legend(fontsize=7, loc='upper right')

    # (b) realized revisit interval; background gradient = revisit-time preference p_rev(v)
    rev = df[(df[int_col].notna()) & (df[int_col] > 0)]
    sns.boxplot(data=rev, x='Algorithm', y=int_col, order=algo_order,
                palette=algo_palette, hue='Algorithm', hue_order=algo_order,
                width=0.5, linewidth=0.8, fliersize=2, legend=False, ax=axes[1], zorder=3,
                showmeans=True, meanline=True,
                medianprops=dict(color='black', linewidth=0.8),
                meanprops=dict(color='red', linewidth=1.5, linestyle='--'),
                )
    axes[1].set_yscale('log')
    x0, x1 = axes[1].get_xlim()
    yg = np.logspace(0, np.log10(9000), 400)
    ymid = np.sqrt(yg[:-1] * yg[1:])
    pcm = axes[1].pcolormesh(np.array([x0, x1]), yg, _p_rev(ymid).reshape(-1, 1),
                             cmap=LinearSegmentedColormap.from_list('Greens_light', plt.cm.Greens(np.linspace(0, 0.65, 256))),
                             alpha=0.45, shading='flat', zorder=0)
    axes[1].set_xlim(x0, x1)
    axes[1].set_ylim(1, 9000)
    axes[1].axhline(60, color='darkgreen', linestyle=':', linewidth=1.0, zorder=1)
    axes[1].text(x0 + 0.06, 66, 'peak (60 s)', color='darkgreen', fontsize=7, va='bottom', zorder=4)
    cax = inset_axes(axes[1], width="2.8%", height="100%", loc='lower left',
                     bbox_to_anchor=(1.02, 0., 1, 1), bbox_transform=axes[1].transAxes, borderpad=0)
    cb = fig.colorbar(pcm, cax=cax)
    cb.set_label(r'Revisit-time preference $p_{\mathrm{rev}}(v)$', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    axes[1].set_ylabel('Median revisit interval [s]')
    axes[1].set_xlabel('Algorithm Configuration')
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].grid(True, axis='y', which='both', linestyle='--', linewidth=0.4, zorder=1)
    axes[1].set_title('(b)')

    plt.suptitle('Revisit Requirement Satisfaction by Algorithm', fontsize=13)
    save_plot(*save_args, f'{filename_stem}.png')
    plt.close()


# ----------------------------------------------------------------------
#  PLOT 3.2.9 — Paired benchmark advantage
# ----------------------------------------------------------------------
def plot_paired_advantage(df, algo_order, algo_palette, save_args,
                          ref='SC-CBBA', metric='Total Obtained Reward [norm]',
                          filename_stem='Plot3.2.9-Paired_Advantage'):
    keys = ['Num Sats', 'Task Arrival Rate', 'Latency', 'Scenario']
    piv = df.pivot_table(index=keys, columns='Algorithm', values=metric)
    baselines = [a for a in algo_order if a != ref]

    win, diff_rows = {}, []
    for b in baselines:
        m = piv[[ref, b]].notna().all(axis=1)
        win[b] = (piv.loc[m, ref] > piv.loc[m, b]).mean()
        for v in (piv.loc[m, ref] - piv.loc[m, b]):
            diff_rows.append({'Baseline': f'vs {b}', 'diff': v})
    diffdf = pd.DataFrame(diff_rows)
    blabels = [f'vs {b}' for b in baselines]
    bcolors = [sns.desaturate(algo_palette[b], 0.75) for b in baselines]

    fig, axes = plt.subplots(2, 1, figsize=(9, 5 * 2), sharex=False)

    # (a) per-scenario win rate
    axes[0].bar(blabels, [win[b] for b in baselines],
                color=bcolors, width=0.6, edgecolor='white')
    axes[0].axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
    for i, b in enumerate(baselines):
        axes[0].text(i, win[b] + 0.02, f'{win[b]:.0%}', ha='center', fontsize=10)
    axes[0].tick_params(axis='x', labelbottom=False)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel(f'Fraction of matched scenarios\n{ref} achieves higher reward')
    axes[0].set_title('(a)')
    axes[0].grid(True, axis='y', linestyle='--', linewidth=0.4)

    # (b) per-scenario reward advantage distribution
    sns.boxplot(data=diffdf, x='Baseline', y='diff', order=blabels,
                palette={f'vs {b}': algo_palette[b] for b in baselines},
                hue='Baseline', hue_order=blabels,
                width=0.5, linewidth=0.8, fliersize=2, legend=False, ax=axes[1])
    axes[1].axhline(0, color='black', linewidth=0.9)
    axes[1].set_ylabel(f'{ref} \u2212 baseline   (normalized reward)')
    axes[1].set_xlabel('')
    axes[1].set_title('(b)')
    axes[1].grid(True, axis='y', linestyle='--', linewidth=0.4)

    plt.suptitle(f'Paired Benchmark Advantage of {ref}', fontsize=13)
    plt.tight_layout()
    save_plot(*save_args, f'{filename_stem}.png')
    plt.close()


# ----------------------------------------------------------------------
#  PLOT 3.2.6 (revised) — Response-time tradeoff with median + IQR
# ----------------------------------------------------------------------
def plot_rt_tradeoff(df, algo_order, algo_palette, save_args,
                     y_specs, suptitle, filename_stem):
    """
    y_specs : list of (column, ylabel). One panel each.
    Cloud   = per-cell means (Algo, Num Sats, Arrival Rate, Latency).
    Marker  = median; error bars = inter-quartile range (robust, bounded).
    """
    rt_col = 'Average Normalized Response Time to Task'
    cell_cols = [rt_col] + [c for c, _ in y_specs]
    cell = (df.groupby(['Algorithm', 'Num Sats', 'Task Arrival Rate', 'Latency'])[cell_cols]
              .mean().reset_index())

    def med_iqr(s):
        return s.median(), s.median() - s.quantile(0.25), s.quantile(0.75) - s.median()

    fig, axes = plt.subplots(len(y_specs), 1, figsize=(8, 6 * len(y_specs)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    for i, (ax, (y_col, ylabel)) in enumerate(zip(axes, y_specs)):
        for algo in algo_order:
            sub = cell[cell['Algorithm'] == algo]
            ax.scatter(sub[rt_col], sub[y_col],
                       color=algo_palette[algo], alpha=0.12, s=18, zorder=1)
        ax.scatter(0, 1, marker='*', s=220, color='gold',
                   edgecolors='black', linewidths=0.8, zorder=5)
        ax.annotate('Utopia', xy=(0, 1), xytext=(0.04, 0.93), fontsize=7, ha='left')

        for algo in algo_order:
            sub = cell[cell['Algorithm'] == algo]
            xm, xlo, xhi = med_iqr(sub[rt_col])
            ym, ylo, yhi = med_iqr(sub[y_col])
            ax.errorbar(xm, ym, xerr=[[xlo], [xhi]], yerr=[[ylo], [yhi]],
                        fmt='none', color=algo_palette[algo], capsize=4,
                        capthick=1.2, elinewidth=1.2, alpha=0.7, zorder=2)
            ax.scatter(xm, ym, color=algo_palette[algo], s=120, marker='D',
                       zorder=3, edgecolors='white', linewidths=0.8)

        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle='--', linewidth=0.4)
        if len(axes) > 1:
            ax.set_title(f'({chr(ord("a") + i)})')

        if i < len(axes) - 1:
            ax.tick_params(axis='x', labelbottom=False)
        else:
            ax.set_xlabel('Avg. Normalized Response Time to Task ' + r'$\bar{\tau}_{\mathrm{resp}}$')

    handles = [Line2D([], [], color=algo_palette[a], marker='D', linestyle='',
                      markersize=8, label=a) for a in algo_order]
    axes[-1].legend(handles=handles, title='Algorithm (median \u00b1 IQR)',
                    loc='best', fontsize=8)
    plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    save_plot(*save_args, f'{filename_stem}.png')
    plt.close()


# ----------------------------------------------------------------------
#  RUNNER (standalone demo on the compiled CSV)
# ----------------------------------------------------------------------
def generate_extra_plots(trial_name, base_dir=None):
    local_base_dir = os.path.join('experiments', '1_cbba_validation', 'analysis')
    if base_dir is None:
        base_dir = local_base_dir
        compiled_results_path = os.path.join(
            base_dir, 'compiled', f'{trial_name}_compiled_results.csv')
    else:
        compiled_results_path = os.path.join(
            base_dir, f'{trial_name}_compiled_results.csv')

    if not os.path.exists(compiled_results_path):
        raise FileNotFoundError(
            f"Compiled results file not found at: `{compiled_results_path}`. "
            "Please run `compiler.py` first.")

    # define save directory and filename for plot
    # name the output file with the current date
    date_str = datetime.now().strftime("%Y-%m-%d")
    dirname = f"{trial_name}_P{date_str}"

    save_dir = os.path.join(base_dir, 'plots', 'rq2', dirname)
    os.makedirs(save_dir, exist_ok=True)

    # if saving to external directory, also save a copy to the local analysis directory 
    local_save_dir = os.path.join(local_base_dir, 'plots', 'rq2', dirname)
    os.makedirs(local_save_dir, exist_ok=True)

    # csv_path = os.path.join('experiments', '1_cbba_validation', 'analysis', 'compiled', trial_name)
    df = pd.read_csv(compiled_results_path)
    df['Algorithm'] = df.apply(label_condition, axis=1)

    # sentinel cleanup: the conditional full-completion rate stores -1 when a
    # scenario has no fully-co-observable events; treat as missing.
    _fc = 'P(Event Fully Co-observed | Fully Co-observable)'
    df.loc[df[_fc] < 0, _fc] = np.nan

    algo_order = ['DP', 'GR', 'DP-GR', 'SC-CBBA', 'DP-SC-CBBA']
    algo_palette = {
        'DP':         '#D55E00',   # vermillion (preplanner only)
        'DP-GR':      '#E69F00',   # amber
        'GR':         '#009E73',   # teal
        'SC-CBBA':    '#56B4E9',   # sky blue
        'DP-SC-CBBA': '#0072B2',   # deep blue
    }
    save_args = (save_dir, local_save_dir, save_dir, local_save_dir)

    plot_requirement_profile_bars(df, algo_order, algo_palette, save_args)
    plot_requirement_profile_radar(df, algo_order, algo_palette, save_args)
    plot_revisit(df, algo_order, algo_palette, save_args)
    plot_paired_advantage(df, algo_order, algo_palette, save_args)
    plot_rt_tradeoff(
        df, algo_order, algo_palette, save_args,
        y_specs=[
            ('P(Task Observed)', r'$P(\mathrm{Task\ Observed})$'),
            ('P(Task Observed | Task Observable)', r'$P(\mathrm{Task\ Observed\ |\ Observable})$'),
        ],
        suptitle='Response Time vs Task Observation Quality',
        filename_stem='Plot3.2.6a-Response_Time_vs_Task_Observation',
    )
    plot_rt_tradeoff(
        df, algo_order, algo_palette, save_args,
        y_specs=[
            ('P(Event Fully Co-observed | Fully Co-observable)',
             r'$P(\mathrm{Task\ Fully\ Completed\ |\ Fully\ Completable})$'),
        ],
        suptitle='Response Time vs Full Completion',
        filename_stem='Plot3.2.6b-Response_Time_vs_Full_Completion',
    )
    print('DONE')


if __name__ == '__main__':
    generate_extra_plots(
        'full_factorial_trials_2026-03-15'
    )