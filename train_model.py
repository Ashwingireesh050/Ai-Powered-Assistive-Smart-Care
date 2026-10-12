"""
train_model.py  -  train the gesture classifier from gesture_data.csv

Run:  python train_model.py
Output: gesture_model.pkl, confusion_matrix.png (for your report)

Validation is GROUPED BY ~10 s BLOCKS of recording, so test windows always come from
stretches the model never saw (limits leakage from overlapping windows).
"""
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from gesture_ai import DATA_PATH, FEATURE_NAMES, LABELS, MODEL_PATH, fidx

# thresholds from the original controller (for the baseline comparison)
MOUTH_OPEN_THRESHOLD = 0.42
MOUTH_CLOSE_THRESHOLD = 0.32
EYE_CLOSED_THRESHOLD = 0.20


def threshold_baseline(X):
    """Your original rule-based logic applied to the same windows."""
    i_ear_min = fidx("avg_ear", "min")
    i_m_max, i_m_last = fidx("mouth", "max"), fidx("mouth", "last")
    out = []
    for r in X:
        if r[i_m_max] >= MOUTH_OPEN_THRESHOLD and r[i_m_last] <= MOUTH_CLOSE_THRESHOLD:
            out.append("mouth_cycle")
        elif r[i_ear_min] < EYE_CLOSED_THRESHOLD:
            out.append("blink")
        else:
            out.append("rest")          # rules have no concept of 'noise'
    return np.array(out)


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"{len(df)} windows, {df['session'].nunique()} sessions")
    print(df.groupby("label")["session"].agg(["count", "nunique"])
            .rename(columns={"count": "windows", "nunique": "sessions"}), "\n")

    X = df[FEATURE_NAMES].values
    y = df["label"].values
    labels = [l for l in LABELS if l in set(y)]

    # Validation groups = consecutive blocks of 100 windows (~10 s) of each label.
    # Test blocks are therefore separate stretches of recording the model never saw,
    # and it does not matter how many times the record key was pressed.
    block = df.groupby("label").cumcount() // 100
    g = (df["label"] + "_" + block.astype(str)).values
    per_label = pd.Series(g).groupby(df["label"].values).nunique()
    if (per_label < 3).any():
        sys.exit("Not enough data. Need at least ~30 seconds of recording per label:\n"
                 + per_label.to_string())

    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                 random_state=0, n_jobs=-1)

    pred = cross_val_predict(clf, X, y, groups=g,
                             cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0))
    print("=== AI (Random Forest), cross-validation on unseen blocks ===")
    print(classification_report(y, pred, labels=labels, digits=3))
    ai_acc = accuracy_score(y, pred)

    base = threshold_baseline(X)
    print("=== Threshold baseline (original logic) ===")
    print(classification_report(y, base, labels=labels, digits=3, zero_division=0))
    base_acc = accuracy_score(y, base)

    print(f"ACCURACY  threshold: {base_acc:.3f}   AI: {ai_acc:.3f}\n")

    clf.fit(X, y)
    joblib.dump({"clf": clf, "classes": list(clf.classes_)}, MODEL_PATH)
    print(f"saved {MODEL_PATH}")

    imp = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda p: -p[1])[:8]
    print("top features:", ", ".join(f"{n} ({v:.2f})" for n, v in imp))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, p, title in [(axes[0], base, f"Threshold logic ({base_acc:.1%})"),
                             (axes[1], pred, f"Random Forest AI ({ai_acc:.1%})")]:
            cm = confusion_matrix(y, p, labels=labels)
            cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks(range(len(labels)), labels, rotation=30)
            ax.set_yticks(range(len(labels)), labels)
            ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(title)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                            color="white" if cm[i, j] > 0.5 else "black")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        print("saved confusion_matrix.png")
    except ImportError:
        print("(pip install matplotlib to get the confusion-matrix figure)")


if __name__ == "__main__":
    main()
