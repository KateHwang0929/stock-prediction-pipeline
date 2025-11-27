# classification_models.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def _to_labels(y_cont: np.ndarray) -> np.ndarray:
    """Label next-day return > 0 as 1, else 0."""
    return (y_cont > 0).astype(int)

def _eval_clf(name: str, clf, X_test, y_test_bin, y_proba=None) -> Dict[str, Any]:
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test_bin, y_pred)
    f1 = f1_score(y_test_bin, y_pred, zero_division=0)
    auc = None
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_test_bin, y_proba)
        except Exception:
            auc = None
    print(f"[{name}] Acc={acc:.4f}  F1={f1:.4f}" + (f"  AUC={auc:.4f}" if auc is not None else ""))
    return {"Model": name, "Accuracy": acc, "F1": f1, "AUC": auc}

def run_all_classifiers(X_train, X_test, y_train_cont, y_test_cont) -> pd.DataFrame:
    y_train = _to_labels(y_train_cont.values if hasattr(y_train_cont, "values") else y_train_cont)
    y_test  = _to_labels(y_test_cont.values  if hasattr(y_test_cont, "values")  else y_test_cont)

    models = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("SVM RBF",            SVC(kernel="rbf", probability=True, class_weight="balanced")),
        ("RandomForest",       RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
        ("GradientBoosting",   GradientBoostingClassifier(random_state=42)),
        ("AdaBoost",           AdaBoostClassifier(random_state=42)),
        ("KNN-15",             KNeighborsClassifier(n_neighbors=15)),
    ]

    rows = []
    for name, clf in models:
        clf.fit(X_train, y_train)
        y_proba = None
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            # Map scores to [0,1] via logistic-ish transform as fallback
            scores = clf.decision_function(X_test)
            y_proba = 1 / (1 + np.exp(-scores))
        rows.append(_eval_clf(name, clf, X_test, y_test, y_proba))
    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
