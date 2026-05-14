import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    balanced_accuracy_score
)


def get_evaluate(clf, X, y_true, threshold=None):
    y_score = None

    try:
        y_score = clf.predict_proba(X)[:, 1]
        default_threshold = 0.5
    except AttributeError:
        try:
            y_score = clf.decision_function(X)
            default_threshold = 0.0
        except AttributeError:
            default_threshold = None

    if threshold is None:
        y_pred = clf.predict(X)
    else:
        if y_score is None:
            raise ValueError("O classificador não fornece scores para aplicar threshold.")
        y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "y_pred": y_pred,
        "y_score": y_score
    }

    if y_score is not None:
        metrics["auc"] = roc_auc_score(y_true, y_score)
    else:
        metrics["auc"] = None

    return metrics


def find_best_threshold(y_true, y_score):
    thresholds = np.unique(y_score)

    best_threshold = thresholds[0]
    best_f1 = -1

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thr

    return best_threshold, best_f1
