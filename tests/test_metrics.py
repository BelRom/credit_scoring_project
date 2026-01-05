import numpy as np
from src.models.metrics import compute_metrics


def test_compute_metrics_range_and_types():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])

    m = compute_metrics(y_true, y_prob, threshold=0.5)

    assert 0.0 <= m.roc_auc <= 1.0
    assert 0.0 <= m.precision <= 1.0
    assert 0.0 <= m.recall <= 1.0
    assert 0.0 <= m.f1 <= 1.0
    assert isinstance(m.roc_auc, float)


def test_compute_metrics_known_case():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.8, 0.2])  # идеально разделяет классы
    m = compute_metrics(y_true, y_prob)
    assert m.roc_auc == 1.0
