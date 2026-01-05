from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.models.pipeline import (
    SUPPORTED_MODELS,
    build_estimator,
    build_preprocessor,
    create_pipeline,
    fix_bool_dtypes,
)


def _make_tiny_X(n: int = 8) -> pd.DataFrame:
    """Маленький DataFrame c числами, категориальными и bool."""
    return pd.DataFrame(
        {
            "age": [25, 40, 33, 55, np.nan, 29, 60, 41][:n],
            "income": [50_000, 80_000, 60_000, 120_000, 70_000, np.nan, 90_000, 65_000][
                :n
            ],
            "city": ["A", "B", "A", "C", "B", None, "A", "C"][:n],
            "is_student": [True, False, False, True, False, True, False, False][:n],
        }
    )


def _make_tiny_y(n: int = 8) -> np.ndarray:
    return np.array([0, 1, 0, 1, 0, 0, 1, 0][:n])


def test_fix_bool_dtypes_casts_bool_to_int():
    X = _make_tiny_X()
    assert X["is_student"].dtype == bool
    X2 = fix_bool_dtypes(X)

    assert X["is_student"].dtype == bool
    assert str(X2["is_student"].dtype) in ("int64", "int32")
    assert set(X2["is_student"].unique().tolist()) <= {0, 1}


def test_build_preprocessor_returns_column_transformer_and_transforms():
    X = _make_tiny_X()
    pre = build_preprocessor(X)

    assert isinstance(pre, ColumnTransformer)

    Xt = pre.fit_transform(X)

    # главное: количество строк сохраняется
    assert Xt.shape[0] == X.shape[0]

    # и что после трансформации есть хотя бы 1 признак
    assert Xt.shape[1] > 0


@pytest.mark.parametrize("model_name", list(SUPPORTED_MODELS))
def test_build_estimator_supported_models(model_name: str):
    est = build_estimator(model_name)
    # проверяем, что модель создаётся (не падает) и имеет fit
    assert hasattr(est, "fit")


def test_build_estimator_raises_on_unknown():
    with pytest.raises(ValueError):
        build_estimator("unknown_model")


def test_create_pipeline_structure_and_quick_fit_logreg():
    X = _make_tiny_X()
    y = _make_tiny_y()

    pipe = create_pipeline("logreg", X)

    assert isinstance(pipe, Pipeline)
    assert "prep" in pipe.named_steps
    assert "model" in pipe.named_steps

    # smoke-test: пайплайн обучается на маленьких данных
    pipe.fit(X, y)

    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
