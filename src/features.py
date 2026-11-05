from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _looks_datetime(series: pd.Series, threshold: float = 0.8) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
        return False
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return float(parsed.notna().mean()) >= threshold


def infer_column_types(
    df: pd.DataFrame,
    target_col: str | None = None,
    id_col: str | None = None,
    parse_datetime: bool = True,
) -> dict[str, list[str]]:
    excluded = {col for col in [target_col, id_col] if col}
    feature_df = df.drop(columns=[c for c in excluded if c in df.columns])

    datetime_cols = []
    if parse_datetime:
        datetime_cols = [col for col in feature_df.columns if _looks_datetime(feature_df[col])]

    remaining = feature_df.drop(columns=datetime_cols, errors="ignore")
    numeric_cols = [
        col for col in remaining.columns if pd.api.types.is_numeric_dtype(remaining[col])
    ]
    categorical_cols = [col for col in remaining.columns if col not in numeric_cols]
    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
    }


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Leakage-safe fold-local preprocessing for tabular baselines."""

    def __init__(
        self,
        target_col: str | None = None,
        id_col: str | None = None,
        parse_datetime: bool = True,
        add_missing_indicators: bool = True,
        datetime_cols: list[str] | None = None,
    ) -> None:
        self.target_col = target_col
        self.id_col = id_col
        self.parse_datetime = parse_datetime
        self.add_missing_indicators = add_missing_indicators
        self.datetime_cols = datetime_cols

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureBuilder":
        X_work = self._base_features(X, fit=True)
        self.datetime_cols_ = [
            col for col in (self.datetime_cols or []) if col in X_work.columns
        ] if self.parse_datetime else []
        X_prepared = self._prepare_datetime_features(X_work, fit=True)
        self.final_column_types_ = infer_column_types(
            X_prepared,
            target_col=None,
            id_col=None,
            parse_datetime=False,
        )

        numeric_cols = self.final_column_types_["numeric"]
        categorical_cols = self.final_column_types_["categorical"]
        if not numeric_cols and not categorical_cols:
            raise ValueError("No usable feature columns found after excluding target/id.")

        transformers = []
        if numeric_cols:
            transformers.append(
                (
                    "numeric",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=self.add_missing_indicators,
                    ),
                    numeric_cols,
                )
            )
        if categorical_cols:
            categorical_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", make_one_hot_encoder()),
                ]
            )
            transformers.append(("categorical", categorical_pipe, categorical_cols))

        self.preprocessor_ = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0.3,
        )
        self.preprocessor_.fit(X_prepared, y)
        self.feature_columns_ = list(X_prepared.columns)
        return self

    def transform(self, X: pd.DataFrame) -> Any:
        X_work = self._base_features(X, fit=False)
        X_prepared = self._prepare_datetime_features(X_work, fit=False)
        for col in self.feature_columns_:
            if col not in X_prepared.columns:
                X_prepared[col] = np.nan
        X_prepared = X_prepared[self.feature_columns_]
        return self.preprocessor_.transform(X_prepared)

    def _base_features(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureBuilder expects a pandas DataFrame.")
        drop_cols = [c for c in [self.target_col, self.id_col] if c and c in X.columns]
        X_work = X.drop(columns=drop_cols).copy()
        if fit:
            self.input_feature_columns_ = list(X_work.columns)
        else:
            for col in self.input_feature_columns_:
                if col not in X_work.columns:
                    X_work[col] = np.nan
            X_work = X_work[self.input_feature_columns_]
        return X_work

    def _prepare_datetime_features(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        X_work = X.copy()
        datetime_cols = getattr(self, "datetime_cols_", [])
        for col in datetime_cols:
            if col not in X_work.columns:
                continue
            parsed = pd.to_datetime(X_work[col], errors="coerce")
            X_work[f"{col}__year"] = parsed.dt.year
            X_work[f"{col}__month"] = parsed.dt.month
            X_work[f"{col}__day"] = parsed.dt.day
            X_work[f"{col}__dayofweek"] = parsed.dt.dayofweek
            X_work[f"{col}__hour"] = parsed.dt.hour
            X_work[f"{col}__is_missing"] = parsed.isna().astype(int)
            X_work = X_work.drop(columns=[col])
        return X_work


# Intentionally absent until the official target/time semantics are known:
# target encoding, target-based rolling features, and as-of joins.
