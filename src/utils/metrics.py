"""Competition metrics for BirdCLEF+ 2026."""
import numpy as np
from sklearn.metrics import roc_auc_score

def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-averaged ROC-AUC, skipping classes with no positives.

    Matches the exact competition metric definition.

    Args:
        y_true: Binary labels, shape (N, C).
        y_pred: Predicted scores, shape (N, C).

    Returns:
        Macro-averaged AUC over classes that have at least one positive sample.
    """
    n_classes = y_true.shape[1]
    aucs = []
    for c in range(n_classes):
        n_pos = y_true[:, c].sum()
        n_neg = (1 - y_true[:, c]).sum()
        # skip classes with no positives or all positives
        if n_pos == 0 or n_neg == 0:
            continue
        col_pred = y_pred[:, c]
        if not np.all(np.isfinite(col_pred)):
            continue
        aucs.append(roc_auc_score(y_true[:, c], col_pred))
    if not aucs:
        return 0.0
    return float(np.mean(aucs))
