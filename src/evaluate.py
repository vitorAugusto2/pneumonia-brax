from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)


def get_evaluate(clf, X, y_true):
    y_pred = clf.predict(X)

    try:
        y_score = clf.predict_proba(X)[:, 1]
    except AttributeError:
        y_score = clf.decision_function(X)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "cm": confusion_matrix(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_score)
    }

    return metrics