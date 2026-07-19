"""Walk-forward machine-learning and neural-network factor combinations."""

from __future__ import annotations

import math
import pickle
from datetime import timedelta
from typing import Any

import numpy as np
import polars as pl

from trend_trader.combinations.base import (
    KEYS,
    FactorCombinationRequest,
    FactorCombinationResult,
    FactorCombiner,
    assemble_combined_dataset,
    prepare_features,
    prepare_target,
)
from trend_trader.data.models import bar_minutes
from trend_trader.research import ResearchDataset


class WalkForwardModelCombiner(FactorCombiner):
    version = "1"

    def __init__(
        self,
        method: str = "machine_learning",
        *,
        forced_model: str | None = None,
    ) -> None:
        self.method = method
        self.forced_model = forced_model

    def combine(
        self, dataset: ResearchDataset, request: FactorCombinationRequest
    ) -> FactorCombinationResult:
        features = prepare_features(dataset, request.factor_names)
        targets = prepare_target(dataset, request.training_horizon)
        training = features.join(targets, on=KEYS, how="left")
        params = dict(request.params)
        model_name = self.forced_model or str(params.get("model", "ridge")).lower()
        model_params = params.get("model_params", {})
        if not isinstance(model_params, dict):
            raise TypeError("model_params must be a mapping")
        min_observations = int(params.get("min_train_observations", 500))
        min_periods = int(params.get("min_train_periods", 20))
        train_window = params.get("train_window_periods")
        train_window = int(train_window) if train_window is not None else None
        retrain_every = int(params.get("retrain_every", 24))
        embargo_bars = int(params.get("embargo_bars", 0))
        label_lag_bars = int(params.get("label_lag_bars", 1))
        min_present_factors = int(params.get("min_present_factors", 1))
        target_transform = str(params.get("target_transform", "none")).lower()
        if min_observations < 2 or min_periods < 2 or retrain_every <= 0:
            raise ValueError("invalid walk-forward training minimums")
        if train_window is not None and train_window < min_periods:
            raise ValueError("train_window_periods must be at least min_train_periods")
        if (
            embargo_bars < 0
            or label_lag_bars < 1
            or not 1 <= min_present_factors <= len(request.factor_names)
        ):
            raise ValueError("invalid embargo or min_present_factors")
        if target_transform not in {"none", "demean", "zscore"}:
            raise ValueError("target_transform must be none, demean, or zscore")

        bar_types = features["bar_type"].unique().to_list()
        if len(bar_types) != 1:
            raise ValueError("model combination requires exactly one bar_type")
        availability_lag = timedelta(
            minutes=bar_minutes(str(bar_types[0])) * (label_lag_bars + embargo_bars)
        )
        score_rows: list[dict[str, object]] = []
        weight_rows: list[dict[str, object]] = []
        current_model: Any | None = None
        last_fit_index: int | None = None
        fit_count = 0
        last_training_size = 0
        timestamps = features["timestamp"].unique(maintain_order=True).sort().to_list()
        features_by_timestamp = {
            frame["timestamp"][0]: frame
            for frame in features.partition_by("timestamp", maintain_order=True)
        }
        for index, timestamp in enumerate(timestamps):
            current = features_by_timestamp[timestamp]
            should_refit = last_fit_index is None or index - last_fit_index >= retrain_every
            if should_refit and index >= min_periods:
                eligible = training.filter(
                    pl.col("label_is_valid")
                    & pl.col("label_value").is_not_null()
                    & (pl.col("exit_time") + availability_lag <= timestamp)
                )
                if train_window is not None and not eligible.is_empty():
                    eligible_times = eligible["timestamp"].unique().sort().tail(train_window)
                    eligible = eligible.filter(pl.col("timestamp").is_in(eligible_times))
                period_count = eligible["timestamp"].n_unique()
                if eligible.height >= min_observations and period_count >= min_periods:
                    x_train = _matrix(eligible, request.factor_names)
                    y_train = _transform_target(
                        np.asarray(eligible["label_value"].to_list(), dtype=float),
                        target_transform,
                    )
                    if not np.isnan(x_train).all(axis=0).any():
                        candidate = _build_model(model_name, model_params)
                        candidate.fit(x_train, y_train)
                        current_model = candidate
                        last_fit_index = index
                        fit_count += 1
                        last_training_size = eligible.height
                        weight_rows.extend(
                            _model_weights(candidate, request.factor_names, timestamp, fit_count)
                        )
            if current_model is None:
                score_rows.extend(_invalid_rows(current, "MODEL_WARMUP_INCOMPLETE"))
                continue
            x_current = _matrix(current, request.factor_names)
            predictions = current_model.predict(x_current)
            for item, prediction, row_values in zip(
                current.iter_rows(named=True), predictions, x_current, strict=True
            ):
                present = int(np.isfinite(row_values).sum())
                valid = present >= min_present_factors and math.isfinite(float(prediction))
                score_rows.append(
                    {
                        **{key: item[key] for key in KEYS},
                        "raw_value": float(prediction) if valid else None,
                        "factor_is_valid": valid,
                        "factor_quality_flags": "" if valid else "MISSING_COMPONENT_FACTOR",
                    }
                )

        scores = pl.DataFrame(score_rows, infer_schema_length=None).with_columns(
            pl.col(key).cast(features.schema[key]).alias(key) for key in KEYS
        )
        weights = (
            pl.DataFrame(weight_rows, infer_schema_length=None).with_columns(
                pl.col("timestamp").cast(features.schema["timestamp"])
            )
            if weight_rows
            else pl.DataFrame()
        )
        diagnostics = {
            "method": self.method,
            "model": model_name,
            "model_params": model_params,
            "training_horizon": request.training_horizon,
            "fit_count": fit_count,
            "last_training_observations": last_training_size,
            "retrain_every": retrain_every,
            "train_window_periods": train_window,
            "embargo_bars": embargo_bars,
            "label_lag_bars": label_lag_bars,
            "target_transform": target_transform,
            "sklearn_version": _sklearn_version(),
            "leakage_guard": (
                "training labels satisfy exit_time + label lag + embargo <= prediction timestamp"
            ),
        }
        return FactorCombinationResult(
            assemble_combined_dataset(dataset, scores, request=request, version=self.version),
            weights=weights,
            diagnostics=diagnostics,
            model_bytes=pickle.dumps(current_model) if current_model is not None else None,
        )


def _build_model(model_name: str, params: dict[str, Any]) -> Any:
    from sklearn.base import RegressorMixin
    from sklearn.ensemble import (
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    options = dict(params)
    random_state = int(options.pop("random_state", 42))
    scaled = True
    estimator: RegressorMixin
    if model_name == "ridge":
        estimator = Ridge(**options)
    elif model_name == "elastic_net":
        options.setdefault("max_iter", 5_000)
        estimator = ElasticNet(random_state=random_state, **options)
    elif model_name == "random_forest":
        options.setdefault("n_estimators", 200)
        options.setdefault("n_jobs", 1)
        estimator = RandomForestRegressor(random_state=random_state, **options)
        scaled = False
    elif model_name == "gradient_boosting":
        estimator = GradientBoostingRegressor(random_state=random_state, **options)
        scaled = False
    elif model_name == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(random_state=random_state, **options)
        scaled = False
    elif model_name == "mlp":
        hidden = options.pop("hidden_layer_sizes", (64, 32))
        if isinstance(hidden, int):
            hidden = (hidden,)
        options["hidden_layer_sizes"] = tuple(int(value) for value in hidden)
        options.setdefault("max_iter", 500)
        options.setdefault("early_stopping", True)
        estimator = MLPRegressor(random_state=random_state, **options)
    else:
        raise ValueError(
            "model must be ridge, elastic_net, random_forest, gradient_boosting, "
            "hist_gradient_boosting, or mlp"
        )
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scaled:
        steps.append(("scaler", StandardScaler()))
    steps.append(("estimator", estimator))
    return Pipeline(steps)


def _matrix(frame: pl.DataFrame, factors: tuple[str, ...]) -> np.ndarray:
    values = frame.select(*factors).to_numpy()
    return np.asarray(values, dtype=float)


def _transform_target(values: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return values
    centered = values - float(np.mean(values))
    if method == "demean":
        return centered
    scale = float(np.std(centered))
    return centered / scale if scale > 0 else centered


def _model_weights(
    model: Any,
    factors: tuple[str, ...],
    timestamp: object,
    fit_number: int,
) -> list[dict[str, object]]:
    estimator = model.named_steps["estimator"]
    values = getattr(estimator, "coef_", None)
    kind = "coefficient"
    if values is None:
        values = getattr(estimator, "feature_importances_", None)
        kind = "feature_importance"
    if values is None and hasattr(estimator, "coefs_"):
        values = np.abs(estimator.coefs_[0]).mean(axis=1)
        kind = "mean_absolute_input_weight"
    if values is None:
        return []
    flattened = np.asarray(values).reshape(-1)
    if len(flattened) != len(factors):
        return []
    return [
        {
            "timestamp": timestamp,
            "fit_number": fit_number,
            "factor_name": factor,
            "weight": float(value),
            "weight_kind": kind,
        }
        for factor, value in zip(factors, flattened, strict=True)
    ]


def _sklearn_version() -> str:
    import sklearn

    return str(sklearn.__version__)


def _invalid_rows(frame: pl.DataFrame, flag: str) -> list[dict[str, object]]:
    return [
        {
            **{key: item[key] for key in KEYS},
            "raw_value": None,
            "factor_is_valid": False,
            "factor_quality_flags": flag,
        }
        for item in frame.iter_rows(named=True)
    ]
