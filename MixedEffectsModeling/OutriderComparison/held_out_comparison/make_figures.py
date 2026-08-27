import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import MixedEffectsModeling.config as config

parent_dir = str(Path(__file__).resolve().parents[3])
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from viz_style import apply_style

apply_style()

DIR = Path(__file__).parent
(DIR / "Figures").mkdir(exist_ok=True)


def _p_to_asterisk(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _short_batch_label(batch):
    name, _, num = batch.replace(" et al.", "").partition("_Batch_")
    return f"{name}_{num}" if num else name


def plot_mmd_direction(df, raw, out_path, p_col="p_direction"):
    d = df.sort_values("mmd2", ascending=False).reset_index(drop=True)
    n_batches = len(d)

    fig, axes = plt.subplots(n_batches, 1, figsize=(7, 1.4 * n_batches), sharex=True)
    if n_batches == 1:
        axes = [axes]

    color_hc, color_sig, color_ns = "#A4AFB8", "#00C78B", "#D64545"

    for ax, (_, row) in zip(axes, d.iterrows()):
        b = row["batch"]
        r = raw[b]
        d_hc, d_dis = np.asarray(r["d_hc"]), np.asarray(r["d_dis"])

        x_min, x_max = min(d_hc.min(), d_dis.min()), max(d_hc.max(), d_dis.max())
        x_margin = (x_max - x_min) * 0.2
        x_grid = np.linspace(x_min - x_margin, x_max + x_margin, 300)
        kde_hc, kde_dis = gaussian_kde(d_hc)(x_grid), gaussian_kde(d_dis)(x_grid)

        mean_hc, mean_dis = d_hc.mean(), d_dis.mean()
        p_val = row.get(p_col, 1.0)
        color_dis = color_sig if p_val < 0.05 else color_ns

        ax.plot(x_grid, kde_hc, color=color_hc, lw=1.5)
        ax.fill_between(x_grid, kde_hc, color=color_hc, alpha=0.35)
        ax.plot(x_grid, kde_dis, color=color_dis, lw=1.5)
        ax.fill_between(x_grid, kde_dis, color=color_dis, alpha=0.35)

        max_y = max(kde_hc.max(), kde_dis.max())
        ax.axvline(mean_hc, color=color_hc, linestyle="--", lw=1.5, zorder=3)
        ax.axvline(mean_dis, color=color_dis, linestyle="--", lw=1.5, zorder=3)

        y_bar = max_y * 1.15
        ax.annotate("", xy=(mean_hc, y_bar), xytext=(mean_dis, y_bar),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))

        asterisk = _p_to_asterisk(p_val) or "n.s."
        delta = mean_dis - mean_hc
        delta_color = color_ns if delta < 0 else "black"
        mid_x = (mean_hc + mean_dis) / 2
        ax.text(mid_x, y_bar + max_y * 0.08, f"delta={delta:+.3f} ({asterisk})",
                ha="center", va="bottom", fontweight="bold", color=delta_color)

        ax.set_ylabel(_short_batch_label(b), rotation=0, ha="right", va="center")
        ax.set_ylim(0, max_y * 1.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    axes[-1].set_xlabel("Kernel embedding distance from HC reference")
    legend_elements = [
        Patch(facecolor=color_hc, edgecolor=color_hc, alpha=0.5, label="Held-out HC"),
        Patch(facecolor=color_sig, edgecolor=color_sig, alpha=0.5, label="Disease (sig.)"),
        Patch(facecolor=color_ns, edgecolor=color_ns, alpha=0.5, label="Disease (n.s.)"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper right", frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    return fig


if __name__ == "__main__":
    outr_mmd = pd.read_csv(DIR / "outrider_mmd_summary.csv")
    with open(DIR / "outrider_mmd_raw.pkl", "rb") as f:
        raw_outrider = pickle.load(f)
    plot_mmd_direction(outr_mmd, raw_outrider, DIR / "Figures" / "mmd_direction_outrider_held_out.png")

    eng_mmd_shash = pd.read_csv(config.LOBO_MIXED_DIR / "mmd_summary_shash.csv")
    with open(config.LOBO_MIXED_DIR / "mmd_raw_shash.pkl", "rb") as f:
        raw_ours_shash = pickle.load(f)
    plot_mmd_direction(eng_mmd_shash, raw_ours_shash, DIR / "Figures" / "mmd_direction_our_engine_shash.png")

    merged = eng_mmd_shash[["batch", "n_hc", "n_dis", "perm_p", "p_direction", "disease_farther"]].merge(
        outr_mmd[["batch", "perm_p", "p_direction", "disease_farther"]],
        on="batch", suffixes=("_ours", "_outrider_held_out"))
    merged = merged.sort_values("n_dis", ascending=False)
    merged.to_csv(DIR / "mmd_comparison_held_out.csv", index=False)
    print(f"direction-significant: ours={int((merged.p_direction_ours < 0.05).sum())}/6  "
          f"outrider_held_out={int((merged.p_direction_outrider_held_out < 0.05).sum())}/6")
    print(merged.to_string())
