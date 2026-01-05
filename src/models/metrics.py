from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


@dataclass(frozen=True)
class Metrics:
    roc_auc: float
    precision: float
    recall: float
    f1: float


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Metrics:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    return Metrics(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )
