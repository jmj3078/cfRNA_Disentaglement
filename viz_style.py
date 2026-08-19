import matplotlib
import matplotlib.pyplot as plt

# Named sizes for manual ax.text()/set_title(fontsize=...) annotations that
# rcParams can't reach. Two tiers only: pick the closer one, don't add a third.
FS_ANNOT_LG = 12
FS_ANNOT_SM = 8

def apply_style():
    matplotlib.rcParams["font.family"] = "Arial"
    plt.rcParams.update({
        'font.family':        'Arial',
        'figure.dpi':         100,
        'savefig.dpi':        300,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'lines.linewidth':    1.0,
        'axes.linewidth':     1.0,
        'grid.linewidth':     1.0,
        'axes.titlesize':     15,
        'axes.labelsize':     13,
        'xtick.labelsize':    13,
        'ytick.labelsize':    13,
        'legend.fontsize':    13,
    })
