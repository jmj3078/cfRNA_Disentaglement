import os

import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#0072B2", "#D55E00", "#009E73", "#FF0000", "#1F1F1F",
]


def _save(fig_or_plt, save_path, dpi=300):
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        obj = fig_or_plt if hasattr(fig_or_plt, "savefig") else plt
        obj.savefig(save_path, dpi=dpi, bbox_inches="tight")


def _build_classifiers():
    return {
        "LogReg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=1000, random_state=42)),
        "SVM": make_pipeline(StandardScaler(),
                             SVC(probability=True, kernel="rbf", random_state=42)),
        "RF": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "GBM": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
    }
