import matplotlib.pyplot as plt
import numpy as np

from analysis.plot_utils import _save


def plot_knee(series, title="Knee Plot", threshold=None, save_path=None):
    sorted_vals = np.sort(series.dropna())
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_vals, color="black", linewidth=2)
    plt.yscale("log")
    plt.ylabel(series.name or "Value")
    plt.xlabel("Samples (sorted)")
    plt.title(title)
    if threshold is not None:
        plt.axhline(threshold, color="red", linestyle="--", linewidth=1.8,
                    label=f"Threshold = {threshold}")
        plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    _save(plt, save_path)
    plt.show()
