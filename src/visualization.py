"""
Visualization functions for the experiment.
All plots use publication-quality settings.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Publication style settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'explanation': '#E74C3C',
    'blind': '#3498DB',
    'measured': '#27AE60',
    'reported': '#F39C12',
    'shap': '#9B59B6',
    'lime': '#1ABC9C',
}


def plot_equivalence_curve(results, blind_attempts_list, title='', save_path=None):
    """
    THE KEY VISUALIZATION.
    Shows blind attempts vs. score reduction with explanation line.
    The crossing point is the equivalence metric.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Average explanation reduction
    exp_reductions = [r['explanation_reduction'] for r in results]
    mean_exp = np.mean(exp_reductions)

    # Average blind reduction for each N
    blind_means = []
    blind_stds = []
    for n in blind_attempts_list:
        reductions = [r['blind_reductions'][n] for r in results]
        blind_means.append(np.mean(reductions))
        blind_stds.append(np.std(reductions))

    # Plot blind curve
    ax.plot(blind_attempts_list, blind_means, 'b-o', linewidth=2,
            markersize=6, label='Blind Trial-and-Error', zorder=3,
            color=COLORS['blind'])
    ax.fill_between(blind_attempts_list,
                     np.array(blind_means) - np.array(blind_stds),
                     np.array(blind_means) + np.array(blind_stds),
                     alpha=0.2, color=COLORS['blind'])

    # Plot explanation line
    ax.axhline(y=mean_exp, color=COLORS['explanation'], linestyle='--',
               linewidth=2, label=f'Explanation (mean = {mean_exp:.4f})')

    # Find and mark crossing point
    for i in range(len(blind_means) - 1):
        if blind_means[i] <= mean_exp <= blind_means[i + 1]:
            frac = (mean_exp - blind_means[i]) / (blind_means[i + 1] - blind_means[i])
            crossing_n = blind_attempts_list[i] + frac * (
                blind_attempts_list[i + 1] - blind_attempts_list[i]
            )
            ax.axvline(x=crossing_n, color='green', linestyle=':', linewidth=2,
                       label=f'Equivalence: N*≈{crossing_n:.1f}')
            ax.plot(crossing_n, mean_exp, 'g*', markersize=15, zorder=5)
            break

    ax.set_xlabel('Number of Blind Attempts')
    ax.set_ylabel('Mean Score Reduction')
    ax.set_title(title or 'Explanation vs. Blind: Equivalence Curve')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_score_reductions(results, save_path=None):
    """Box plots comparing score reductions across strategies."""
    fig, ax = plt.subplots(figsize=(12, 6))

    exp_reds = [r['explanation_reduction'] for r in results]
    data_to_plot = [exp_reds]
    labels = ['Explanation']

    for n in [1, 5, 10, 20, 50, 100]:
        if n in results[0]['blind_reductions']:
            reds = [r['blind_reductions'][n] for r in results]
            data_to_plot.append(reds)
            labels.append(f'Blind (N={n})')

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = [COLORS['explanation']] + [COLORS['blind']] * (len(labels) - 1)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Score Reduction')
    ax.set_title('Score Reduction: Explanation vs. Blind Attempts')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"  Saved: {save_path}")
    plt.close()


def plot_model_comparison(all_results, save_path=None):
    """Bar chart comparing measured vs self-reported model vulnerability."""
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = []
    measured_vals = []
    reported_vals = []

    for suffix in ['rf_shap', 'rf_lime', 'lr_shap', 'lr_lime']:
        m_key = f'measured_{suffix}'
        r_key = f'reported_{suffix}'
        if m_key in all_results and r_key in all_results:
            configs.append(suffix.upper().replace('_', '+'))
            measured_vals.append(all_results[m_key]['summary']['median_equivalence'])
            reported_vals.append(all_results[r_key]['summary']['median_equivalence'])

    if not configs:
        return

    x = np.arange(len(configs))
    width = 0.35

    ax.bar(x - width / 2, measured_vals, width,
           label='Model A (Measured)', color=COLORS['measured'], alpha=0.85)
    ax.bar(x + width / 2, reported_vals, width,
           label='Model B (Self-Reported)', color=COLORS['reported'], alpha=0.85)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Median Equivalence (N*)')
    ax.set_title('Model A vs Model B: Does Input Verification Matter?')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"  Saved: {save_path}")
    plt.close()
