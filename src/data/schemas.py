from __future__ import annotations

from pandera import Column, Check, DataFrameSchema


def raw_train_schema() -> DataFrameSchema:
    return DataFrameSchema(
        columns={
            "LIMIT_BAL": Column(float, Check.ge(0), nullable=False),
            "SEX": Column(int, Check.isin([1, 2]), nullable=False),
            "EDUCATION": Column(int, Check.isin([0, 1, 2, 3, 4, 5, 6]), nullable=False),
            "MARRIAGE": Column(int, Check.isin([0, 1, 2, 3]), nullable=False),
            "AGE": Column(int, [Check.ge(18), Check.le(120)], nullable=False),
            "PAY_0": Column(int, [Check.ge(-2), Check.le(10)], nullable=False),
            "PAY_2": Column(int, [Check.ge(-2), Check.le(10)], nullable=False),
            "PAY_3": Column(int, [Check.ge(-2), Check.le(10)], nullable=False),
            "PAY_4": Column(int, [Check.ge(-2), Check.le(10)], nullable=False),
            "PAY_5": Column(int, [Check.ge(-2), Check.le(10)], nullable=False),
            "PAY_6": Column(int, [Check.ge(-2), Check.le(10)], nullable=False),
            "BILL_AMT1": Column(float, nullable=False),
            "BILL_AMT2": Column(float, nullable=False),
            "BILL_AMT3": Column(float, nullable=False),
            "BILL_AMT4": Column(float, nullable=False),
            "BILL_AMT5": Column(float, nullable=False),
            "BILL_AMT6": Column(float, nullable=False),
            "PAY_AMT1": Column(float, Check.ge(0), nullable=False),
            "PAY_AMT2": Column(float, Check.ge(0), nullable=False),
            "PAY_AMT3": Column(float, Check.ge(0), nullable=False),
            "PAY_AMT4": Column(float, Check.ge(0), nullable=False),
            "PAY_AMT5": Column(float, Check.ge(0), nullable=False),
            "PAY_AMT6": Column(float, Check.ge(0), nullable=False),
            "target": Column(int, Check.isin([0, 1]), nullable=False),
        },
        strict=False,
        coerce=True,
    )


def raw_predict_schema() -> DataFrameSchema:
    return DataFrameSchema(
        columns={
            "LIMIT_BAL": Column(float, Check.ge(0), nullable=False),
            "SEX": Column(int, Check.isin([1, 2]), nullable=False),
            "EDUCATION": Column(int, Check.isin([0, 1, 2, 3, 4, 5, 6]), nullable=False),
            "MARRIAGE": Column(int, Check.isin([0, 1, 2, 3]), nullable=False),
            "AGE": Column(int, [Check.ge(18), Check.le(120)], nullable=False),
            "PAY_0": Column(int, [Check.ge(-2), Check.le(10)], nullable=False),
            "PAY_2": Column(int, [Check.ge(-2), Check.le(10)], nullable=False),
            "PAY_3": Column(int, [Check.ge(-2), Check.le(10)], nullable=False),
            "PAY_4": Column(int, [Check.ge(-2), Check.le(10)], nullable=False),
            "PAY_5": Column(int, [Check.ge(-2), Check.le(10)], nullable=False),
            "PAY_6": Column(int, [Check.ge(-2), Check.le(10)], nullable=False),
            "BILL_AMT1": Column(float, nullable=False),
            "BILL_AMT2": Column(float, nullable=False),
            "BILL_AMT3": Column(float, nullable=False),
            "BILL_AMT4": Column(float, nullable=False),
            "BILL_AMT5": Column(float, nullable=False),
            "BILL_AMT6": Column(float, nullable=False),
            "PAY_AMT1": Column(float, Check.ge(0), nullable=False),
            "PAY_AMT2": Column(float, Check.ge(0), nullable=False),
            "PAY_AMT3": Column(float, Check.ge(0), nullable=False),
            "PAY_AMT4": Column(float, Check.ge(0), nullable=False),
            "PAY_AMT5": Column(float, Check.ge(0), nullable=False),
            "PAY_AMT6": Column(float, Check.ge(0), nullable=False),
        },
        strict=True,
        coerce=True,
    )
