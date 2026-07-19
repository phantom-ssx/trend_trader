"""Common statistical analysis for factor research datasets."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations_with_replacement
from typing import Literal

import polars as pl

from trend_trader.research.models import (
    FactorAnalysisReport,
    RedundancyAnalysisReport,
    ResearchDataset,
)

CorrelationMethod = Literal["pearson", "spearman"]
QuantileScope = Literal["cross_sectional", "time_series"]


class FactorAnalyzer:
    def __init__(self, dataset: ResearchDataset | pl.DataFrame) -> None:
        self.frame = dataset.frame if isinstance(dataset, ResearchDataset) else dataset
        required = {
            "venue",
            "instrument_id",
            "timestamp",
            "factor_name",
            "value",
            "label_name",
            "horizon_bars",
            "label_value",
            "factor_is_valid",
            "label_is_valid",
            "is_valid",
        }
        missing = sorted(required.difference(self.frame.columns))
        if missing:
            raise ValueError(f"research dataset is missing columns: {missing}")

    def summary(self) -> pl.DataFrame:
        """Coverage and descriptive statistics for every factor-label pair."""

        keys = ["factor_name", "label_name", "horizon_bars"]
        result = self.frame.group_by(keys).agg(
            pl.len().alias("observations"),
            pl.col("factor_is_valid").sum().alias("factor_valid_observations"),
            pl.col("label_is_valid").sum().alias("label_valid_observations"),
            pl.col("is_valid").sum().alias("valid_observations"),
            pl.col("value").filter(pl.col("is_valid")).mean().alias("factor_mean"),
            pl.col("value").filter(pl.col("is_valid")).std().alias("factor_std"),
            pl.col("value").filter(pl.col("is_valid")).median().alias("factor_median"),
            pl.col("value").filter(pl.col("is_valid")).quantile(0.01).alias("factor_q01"),
            pl.col("value").filter(pl.col("is_valid")).quantile(0.99).alias("factor_q99"),
            pl.col("label_value").filter(pl.col("is_valid")).mean().alias("label_mean"),
            pl.col("label_value").filter(pl.col("is_valid")).std().alias("label_std"),
            (pl.col("label_value").filter(pl.col("is_valid")) > 0)
            .mean()
            .alias("label_positive_rate"),
        )
        return result.with_columns(
            (pl.col("valid_observations") / pl.col("observations")).alias("coverage")
        ).sort(*keys)

    def overall_ic(
        self,
        *,
        method: CorrelationMethod = "spearman",
        min_observations: int = 5,
    ) -> pl.DataFrame:
        """Correlation over all valid observations."""

        _validate_minimum(min_observations)
        valid = self._valid()
        keys = ["factor_name", "label_name", "horizon_bars"]
        return (
            valid.group_by(keys)
            .agg(
                pl.len().alias("observations"),
                pl.corr("value", "label_value", method=method).alias("ic"),
            )
            .with_columns(pl.lit(method).alias("method"))
            .filter(pl.col("observations") >= min_observations)
            .sort(*keys)
        )

    def ic_series(
        self,
        *,
        method: CorrelationMethod = "spearman",
        min_observations: int = 5,
    ) -> pl.DataFrame:
        """Cross-sectional IC at each factor timestamp."""

        _validate_minimum(min_observations)
        keys = ["factor_name", "label_name", "horizon_bars", "timestamp"]
        return (
            self._valid()
            .group_by(keys)
            .agg(
                pl.len().alias("observations"),
                pl.corr("value", "label_value", method=method).alias("ic"),
            )
            .with_columns(pl.lit(method).alias("method"))
            .filter(
                (pl.col("observations") >= min_observations)
                & pl.col("ic").is_not_null()
                & pl.col("ic").is_finite()
            )
            .sort(*keys)
        )

    def ic_summary(self, ic_series: pl.DataFrame | None = None) -> pl.DataFrame:
        """Mean IC, ICIR, t-statistic, and positive-IC rate."""

        series = self.ic_series() if ic_series is None else ic_series
        keys = ["factor_name", "label_name", "horizon_bars", "method"]
        summary = series.group_by(keys).agg(
            pl.len().alias("periods"),
            pl.col("ic").mean().alias("mean_ic"),
            pl.col("ic").std().alias("std_ic"),
            pl.col("ic").median().alias("median_ic"),
            (pl.col("ic") > 0).mean().alias("positive_rate"),
        )
        return summary.with_columns(
            pl.when(pl.col("std_ic") > 0).then(pl.col("mean_ic") / pl.col("std_ic")).alias("icir"),
            pl.when((pl.col("std_ic") > 0) & (pl.col("periods") > 1))
            .then(pl.col("mean_ic") / (pl.col("std_ic") / pl.col("periods").sqrt()))
            .alias("t_stat"),
        ).sort(*keys)

    def quantile_returns(
        self,
        *,
        quantiles: int = 5,
        scope: QuantileScope = "cross_sectional",
    ) -> pl.DataFrame:
        """Mean future return for portfolios sorted by factor value."""

        if quantiles < 2:
            raise ValueError("quantiles must be at least 2")
        valid = self._valid().sort("factor_name", "label_name", "instrument_id", "timestamp")
        if scope == "cross_sectional":
            rank_keys = ["factor_name", "label_name", "timestamp"]
        elif scope == "time_series":
            rank_keys = ["factor_name", "label_name", "venue", "instrument_id"]
        else:
            raise ValueError("scope must be cross_sectional or time_series")
        ranked = valid.with_columns(
            pl.col("value").rank(method="average").over(rank_keys).alias("_rank"),
            pl.len().over(rank_keys).alias("_count"),
        ).filter(pl.col("_count") >= quantiles)
        ranked = ranked.with_columns(
            (((pl.col("_rank") - 1) * quantiles / pl.col("_count")).floor() + 1)
            .clip(1, quantiles)
            .cast(pl.Int16)
            .alias("quantile")
        )
        keys = ["factor_name", "label_name", "horizon_bars", "quantile"]
        return (
            ranked.group_by(keys)
            .agg(
                pl.len().alias("observations"),
                pl.col("label_value").mean().alias("mean_return"),
                pl.col("label_value").std().alias("std_return"),
                pl.col("label_value").median().alias("median_return"),
                (pl.col("label_value") > 0).mean().alias("win_rate"),
                pl.col("value").mean().alias("mean_factor"),
            )
            .with_columns(pl.lit(scope).alias("scope"))
            .sort(*keys)
        )

    def factor_returns(self, *, min_observations: int = 5) -> pl.DataFrame:
        """Cross-sectional univariate regression slope at each timestamp."""

        _validate_minimum(min_observations)
        keys = ["factor_name", "label_name", "horizon_bars", "timestamp"]
        result = (
            self._valid()
            .group_by(keys)
            .agg(
                pl.len().alias("observations"),
                pl.col("value").mean().alias("_factor_mean"),
                pl.col("label_value").mean().alias("_label_mean"),
                pl.cov("value", "label_value").alias("_covariance"),
                pl.col("value").var().alias("_factor_variance"),
            )
        )
        return (
            result.with_columns(
                pl.when(pl.col("_factor_variance") > 0)
                .then(pl.col("_covariance") / pl.col("_factor_variance"))
                .alias("factor_return")
            )
            .with_columns(
                (pl.col("_label_mean") - pl.col("factor_return") * pl.col("_factor_mean")).alias(
                    "intercept"
                )
            )
            .filter(
                (pl.col("observations") >= min_observations)
                & pl.col("factor_return").is_not_null()
                & pl.col("factor_return").is_finite()
            )
            .drop("_factor_mean", "_label_mean", "_covariance", "_factor_variance")
            .sort(*keys)
        )

    def factor_return_summary(
        self,
        factor_returns: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Average factor return, t-statistic, and positive-period rate."""

        returns = self.factor_returns() if factor_returns is None else factor_returns
        keys = ["factor_name", "label_name", "horizon_bars"]
        result = returns.group_by(keys).agg(
            pl.len().alias("periods"),
            pl.col("factor_return").mean().alias("mean_factor_return"),
            pl.col("factor_return").std().alias("std_factor_return"),
            pl.col("factor_return").median().alias("median_factor_return"),
            (pl.col("factor_return") > 0).mean().alias("positive_rate"),
        )
        return result.with_columns(
            pl.when((pl.col("std_factor_return") > 0) & (pl.col("periods") > 1))
            .then(
                pl.col("mean_factor_return")
                / (pl.col("std_factor_return") / pl.col("periods").sqrt())
            )
            .alias("t_stat")
        ).sort(*keys)

    def quantile_spread(
        self,
        quantile_returns: pl.DataFrame | None = None,
        *,
        quantiles: int = 5,
        scope: QuantileScope = "cross_sectional",
    ) -> pl.DataFrame:
        """Top-minus-bottom return and quantile monotonicity."""

        returns = (
            self.quantile_returns(quantiles=quantiles, scope=scope)
            if quantile_returns is None
            else quantile_returns
        )
        keys = ["factor_name", "label_name", "horizon_bars", "scope"]
        return (
            returns.group_by(keys)
            .agg(
                pl.col("mean_return")
                .filter(pl.col("quantile") == quantiles)
                .first()
                .alias("top_return"),
                pl.col("mean_return")
                .filter(pl.col("quantile") == 1)
                .first()
                .alias("bottom_return"),
                pl.corr("quantile", "mean_return", method="spearman").alias("monotonicity"),
            )
            .with_columns(
                (pl.col("top_return") - pl.col("bottom_return")).alias("long_short_return")
            )
            .sort(*keys)
        )

    def periodic_ic(
        self,
        *,
        every: str = "1mo",
        method: CorrelationMethod = "spearman",
        min_observations: int = 20,
    ) -> pl.DataFrame:
        """IC within calendar periods for stability analysis."""

        _validate_minimum(min_observations)
        keys = ["factor_name", "label_name", "horizon_bars", "period"]
        return (
            self._valid()
            .with_columns(pl.col("timestamp").dt.truncate(every).alias("period"))
            .group_by(keys)
            .agg(
                pl.len().alias("observations"),
                pl.corr("value", "label_value", method=method).alias("ic"),
            )
            .with_columns(pl.lit(method).alias("method"))
            .filter(pl.col("observations") >= min_observations)
            .sort(*keys)
        )

    def decay(
        self,
        *,
        method: CorrelationMethod = "spearman",
        quantiles: int = 5,
        scope: QuantileScope = "cross_sectional",
    ) -> pl.DataFrame:
        """Prediction-strength decay across label horizons."""

        ic = self.overall_ic(method=method, min_observations=2)
        spreads = self.quantile_spread(quantiles=quantiles, scope=scope)
        return ic.join(
            spreads.select(
                "factor_name",
                "label_name",
                "horizon_bars",
                "long_short_return",
                "monotonicity",
            ),
            on=["factor_name", "label_name", "horizon_bars"],
            how="left",
        ).sort("factor_name", "horizon_bars")

    def factor_correlation(self, *, method: CorrelationMethod = "spearman") -> pl.DataFrame:
        """Long-form pairwise correlation matrix between factor values."""

        keys = ["venue", "instrument_id", "timestamp"]
        unique = self.frame.select(*keys, "factor_name", "value").unique(
            subset=[*keys, "factor_name"]
        )
        wide = unique.pivot(on="factor_name", index=keys, values="value")
        factors = sorted(set(wide.columns).difference(keys))
        rows: list[dict[str, object]] = []
        for left, right in combinations_with_replacement(factors, 2):
            correlation = None
            if left == right:
                pair = wide.select(pl.col(left).alias("_left")).drop_nulls()
                if pair.height >= 2 and pair.get_column("_left").n_unique() > 1:
                    correlation = 1.0
            else:
                pair = wide.select(
                    pl.col(left).alias("_left"),
                    pl.col(right).alias("_right"),
                ).drop_nulls()
            if left != right and pair.height >= 2:
                correlation = pair.select(pl.corr("_left", "_right", method=method)).item()
                if correlation is not None and not math.isfinite(float(correlation)):
                    correlation = None
            rows.append(
                {
                    "factor_left": left,
                    "factor_right": right,
                    "observations": pair.height,
                    "correlation": correlation,
                    "method": method,
                }
            )
        return pl.DataFrame(rows, infer_schema_length=None)

    def autocorrelation(self, *, lag_bars: int = 1) -> pl.DataFrame:
        """Per-instrument factor persistence, useful as a turnover proxy."""

        if lag_bars <= 0:
            raise ValueError("lag_bars must be positive")
        keys = ["factor_name", "venue", "instrument_id"]
        unique = (
            self.frame.select(*keys, "timestamp", "value")
            .unique(subset=[*keys, "timestamp"])
            .sort(*keys, "timestamp")
            .with_columns(pl.col("value").shift(lag_bars).over(keys).alias("_lagged"))
        )
        return (
            unique.group_by(keys)
            .agg(
                pl.col("value")
                .is_not_null()
                .and_(pl.col("_lagged").is_not_null())
                .sum()
                .alias("observations"),
                pl.corr("value", "_lagged").alias("autocorrelation"),
            )
            .with_columns(pl.lit(lag_bars).alias("lag_bars"))
            .sort(*keys)
        )

    def vif(
        self,
        *,
        factors: Sequence[str] | None = None,
        min_observations: int = 20,
        ridge: float = 1e-8,
    ) -> pl.DataFrame:
        """Variance inflation factors for multicollinearity diagnosis."""

        from trend_trader.research.redundancy import FactorRedundancyAnalyzer

        return FactorRedundancyAnalyzer(self.frame).vif(
            factors=factors,
            min_observations=min_observations,
            ridge=ridge,
        )

    def unique_contribution(
        self,
        *,
        target_factors: Sequence[str] | None = None,
        control_factors: Sequence[str] | None = None,
        method: CorrelationMethod = "spearman",
        min_observations: int = 5,
        ridge: float = 1e-8,
    ) -> pl.DataFrame:
        """Conditional IC and incremental R² after controlling for other factors."""

        from trend_trader.research.redundancy import FactorRedundancyAnalyzer

        return FactorRedundancyAnalyzer(self.frame).unique_contribution(
            target_factors=target_factors,
            control_factors=control_factors,
            method=method,
            min_observations=min_observations,
            ridge=ridge,
        )

    def redundancy_report(
        self,
        *,
        method: CorrelationMethod = "spearman",
        min_observations: int = 5,
        ridge: float = 1e-8,
        cluster_threshold: float = 0.8,
        target_factors: Sequence[str] | None = None,
        control_factors: Sequence[str] | None = None,
    ) -> RedundancyAnalysisReport:
        """Run the explicit, potentially expensive multivariate redundancy suite."""

        from trend_trader.research.redundancy import FactorRedundancyAnalyzer

        correlations = self.factor_correlation(method=method)
        overall_ic = self.overall_ic(method=method, min_observations=min_observations)
        return FactorRedundancyAnalyzer(self.frame).run(
            correlations,
            overall_ic,
            method=method,
            min_observations=min_observations,
            ridge=ridge,
            cluster_threshold=cluster_threshold,
            target_factors=target_factors,
            control_factors=control_factors,
        )

    def run(
        self,
        *,
        method: CorrelationMethod = "spearman",
        min_cross_section: int = 5,
        quantiles: int = 5,
        quantile_scope: QuantileScope = "cross_sectional",
        stability_period: str = "1mo",
        stability_min_observations: int = 20,
    ) -> FactorAnalysisReport:
        """Run the standard factor-analysis suite."""

        periodic = self.periodic_ic(
            every=stability_period,
            method=method,
            min_observations=stability_min_observations,
        )
        series = (
            periodic.rename({"period": "timestamp"})
            if quantile_scope == "time_series"
            else self.ic_series(method=method, min_observations=min_cross_section)
        )
        factor_returns = self.factor_returns(min_observations=min_cross_section)
        quantile_returns = self.quantile_returns(quantiles=quantiles, scope=quantile_scope)
        return FactorAnalysisReport(
            summary=self.summary(),
            overall_ic=self.overall_ic(
                method=method,
                min_observations=min_cross_section,
            ),
            ic_series=series,
            ic_summary=self.ic_summary(series),
            factor_returns=factor_returns,
            factor_return_summary=self.factor_return_summary(factor_returns),
            quantile_returns=quantile_returns,
            quantile_spread=self.quantile_spread(
                quantile_returns,
                quantiles=quantiles,
                scope=quantile_scope,
            ),
            periodic_ic=periodic,
            decay=self.decay(method=method, quantiles=quantiles, scope=quantile_scope),
            factor_correlation=self.factor_correlation(method=method),
            autocorrelation=self.autocorrelation(),
        )

    def _valid(self) -> pl.DataFrame:
        return self.frame.filter(
            pl.col("is_valid")
            & pl.col("value").is_not_null()
            & pl.col("value").is_finite()
            & pl.col("label_value").is_not_null()
            & pl.col("label_value").is_finite()
        )


def _validate_minimum(value: int) -> None:
    if value < 2:
        raise ValueError("min_observations must be at least 2")
