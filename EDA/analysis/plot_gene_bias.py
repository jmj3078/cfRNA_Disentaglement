import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.plot_utils import _save


def plot_gene_wise_bias_summary(df_detail, group_name, save_path=None):
    """Joint R² histogram + contamination severity bar chart for one group."""
    joint_r2 = df_detail["Joint_R2_All_Biases"]
    bins_edges = [-np.inf, 0.05, 0.10, 0.30, np.inf]
    sev_labels = ["Minimal (< 5%)", "Moderate (5-10%)", "High (10-30%)", "Severe (> 30%)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    sns.histplot(joint_r2, bins=75, color="steelblue", ax=ax1, edgecolor="black")
    ax1.axvline(0.10, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
    ax1.set_xlim(0, 1.0)
    ax1.set_xlabel("Total Variance Explained by All Biases (Joint R²)")
    ax1.set_ylabel("Number of Genes")
    ax1.set_title(group_name)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    cats = pd.cut(joint_r2, bins=bins_edges, labels=sev_labels)
    counts = cats.value_counts(sort=False)
    n_total = len(joint_r2)
    pcts = counts / n_total * 100
    df_sev = pd.DataFrame({"Severity": sev_labels, "Pct": pcts.values, "N": counts.values})
    df_sev = df_sev.iloc[::-1].reset_index(drop=True)

    sns.barplot(data=df_sev, x="Pct", y="Severity", ax=ax2,
                palette="Reds_r", hue="Severity", legend=False, edgecolor="black")
    max_pct = df_sev["Pct"].max()
    for idx, row in df_sev.iterrows():
        ax2.text(row["Pct"] + max_pct * 0.02, idx,
                 f"{row['Pct']:.1f}% ({int(row['N']):,})", va="center", fontsize=11, color="#333333")
    ax2.set_xlim(0, max_pct * 1.35)
    ax2.set_xlabel("Proportion of Total Genes (%)")
    ax2.set_ylabel("Contamination Severity (Joint R²)")
    ax2.grid(axis="x", linestyle="--", alpha=0.3)

    _save(fig, save_path)
    plt.show()
