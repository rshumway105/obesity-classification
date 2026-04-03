"""
Evaluation utilities: metrics, confusion matrices, cross-validation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import cross_val_score
import sys
sys.path.append("..")
from config import TARGET_CLASSES, CV_FOLDS, FIGURES_DIR


def evaluate(model, X_test, y_test, model_name="Model"):
    """Print classification report and return a dict of key metrics."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  Macro F1-Score: {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=TARGET_CLASSES)}")

    return {"model": model_name, "accuracy": acc, "macro_f1": f1}


def plot_confusion_matrix(model, X_test, y_test, model_name="Model", save=False):
    """Plot a heatmap confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=TARGET_CLASSES,
        yticklabels=TARGET_CLASSES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / f"cm_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()


def cross_validate_models(models, X, y):
    """
    Run stratified k-fold cross-validation for each model.

    Parameters
    ----------
    models : dict of {name: estimator}
    X, y : features and target

    Returns
    -------
    DataFrame with mean and std accuracy per model.
    """
    results = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=CV_FOLDS, scoring="accuracy")
        results.append({
            "model": name,
            "mean_accuracy": scores.mean(),
            "std_accuracy": scores.std(),
        })
        print(f"{name:25s}  acc = {scores.mean():.4f} ± {scores.std():.4f}")
    return pd.DataFrame(results)
