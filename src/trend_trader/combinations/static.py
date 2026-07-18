"""Rule, linear-score, and cross-sectional-rank factor combinations."""

from __future__ import annotations

import math
from typing import Any

import polars as pl

from trend_trader.combinations.base import (
    KEYS,
    FactorCombinationRequest,
    FactorCombinationResult,
    FactorCombiner,
    assemble_combined_dataset,
    impute_frame,
    prepare_features,
)
from trend_trader.research import ResearchDataset


class RuleCombiner(FactorCombiner):
    method = "rule"

    def combine(
        self, dataset: ResearchDataset, request: FactorCombinationRequest
    ) -> FactorCombinationResult:
        features = prepare_features(dataset, request.factor_names)
        configured_rules = request.params.get("rules")
        if configured_rules is not None:
            if not isinstance(configured_rules, list) or not configured_rules:
                raise ValueError("rules must be a non-empty list")
            score = _score_value(
                request.params,
                request.factor_names,
                score_key="default_score",
                factor_key="default_score_factor",
                default=None,
            )
            for rule in reversed(configured_rules):
                if not isinstance(rule, dict):
                    raise TypeError("each rule must be a mapping")
                passed = _conditions_expression(rule, request.factor_names)
                rule_score = _score_value(
                    rule,
                    request.factor_names,
                    score_key="score",
                    factor_key="score_factor",
                    default=1.0,
                )
                score = pl.when(passed).then(rule_score).otherwise(score)
            diagnostics: dict[str, Any] = {
                "method": self.method,
                "rules": configured_rules,
            }
        else:
            conditions = request.params.get("conditions")
            if not isinstance(conditions, list) or not conditions:
                raise ValueError("rule combination requires conditions or rules")
            passed = _conditions_expression(request.params, request.factor_names)
            pass_score = _score_value(
                request.params,
                request.factor_names,
                score_key="pass_score",
                factor_key="score_factor",
                default=1.0,
            )
            fail_score = _score_value(
                request.params,
                request.factor_names,
                score_key="fail_score",
                factor_key="fail_score_factor",
                default=None,
            )
            score = pl.when(passed).then(pass_score).otherwise(fail_score)
            diagnostics = {
                "method": self.method,
                "conditions": conditions,
                "logic": str(request.params.get("logic", "all")).lower(),
            }
        scores = features.select(
            *KEYS,
            score.alias("raw_value"),
        ).with_columns(
            pl.col("raw_value").is_not_null().alias("factor_is_valid"),
            pl.when(pl.col("raw_value").is_not_null())
            .then(pl.lit(""))
            .otherwise(pl.lit("RULE_NOT_MATCHED"))
            .alias("factor_quality_flags"),
        )
        return FactorCombinationResult(
            assemble_combined_dataset(dataset, scores, request=request, version=self.version),
            diagnostics=diagnostics,
        )


class LinearCombiner(FactorCombiner):
    method = "linear"

    def combine(
        self, dataset: ResearchDataset, request: FactorCombinationRequest
    ) -> FactorCombinationResult:
        weights = _weights(request)
        if bool(request.params.get("normalize_weights", True)):
            denominator = sum(abs(value) for value in weights.values())
            if denominator == 0:
                raise ValueError("linear weights must not all be zero")
            weights = {name: value / denominator for name, value in weights.items()}
        features = impute_frame(
            prepare_features(dataset, request.factor_names),
            request.factor_names,
            str(request.params.get("missing", "drop")),
        )
        score = pl.lit(float(request.params.get("intercept", 0.0)))
        for name, weight in weights.items():
            score += pl.col(name) * weight
        scores = _score_frame(features, score)
        weight_frame = pl.DataFrame(
            {"factor_name": list(weights), "weight": list(weights.values())}
        )
        return FactorCombinationResult(
            assemble_combined_dataset(dataset, scores, request=request, version=self.version),
            weights=weight_frame,
            diagnostics={"method": self.method, "weights": weights},
        )


class RankCombiner(FactorCombiner):
    method = "rank"

    def combine(
        self, dataset: ResearchDataset, request: FactorCombinationRequest
    ) -> FactorCombinationResult:
        weights = _weights(request)
        denominator = sum(abs(value) for value in weights.values())
        if denominator == 0:
            raise ValueError("rank weights must not all be zero")
        weights = {name: value / denominator for name, value in weights.items()}
        features = prepare_features(dataset, request.factor_names)
        groups = ["venue", "bar_type", "timestamp"]
        rank_columns: list[str] = []
        expressions: list[pl.Expr] = []
        for index, name in enumerate(request.factor_names):
            rank_name = f"_rank_{index}"
            count = pl.col(name).count().over(groups)
            rank = pl.col(name).rank(method="average").over(groups)
            expressions.append(
                pl.when(count > 1).then(2 * (rank - 1) / (count - 1) - 1).alias(rank_name)
            )
            rank_columns.append(rank_name)
        ranked = features.with_columns(*expressions)
        ranked = impute_frame(
            ranked,
            tuple(rank_columns),
            str(request.params.get("missing", "drop")),
        )
        score = pl.lit(0.0)
        for rank_name, factor_name in zip(rank_columns, request.factor_names, strict=True):
            score += pl.col(rank_name) * weights[factor_name]
        scores = _score_frame(ranked, score)
        weight_frame = pl.DataFrame(
            {"factor_name": list(weights), "weight": list(weights.values())}
        )
        return FactorCombinationResult(
            assemble_combined_dataset(dataset, scores, request=request, version=self.version),
            weights=weight_frame,
            diagnostics={"method": self.method, "weights": weights},
        )


def _weights(request: FactorCombinationRequest) -> dict[str, float]:
    configured = request.params.get("weights")
    if configured is None:
        return {name: 1.0 for name in request.factor_names}
    if not isinstance(configured, dict):
        raise TypeError("combination weights must be a mapping")
    unknown = sorted(set(configured).difference(request.factor_names))
    missing = sorted(set(request.factor_names).difference(configured))
    if unknown or missing:
        raise ValueError(f"weights mismatch; missing={missing}, unknown={unknown}")
    weights = {name: float(configured[name]) for name in request.factor_names}
    if any(not math.isfinite(value) for value in weights.values()):
        raise ValueError("combination weights must be finite")
    return weights


def _score_frame(features: pl.DataFrame, expression: pl.Expr) -> pl.DataFrame:
    return features.select(*KEYS, expression.alias("raw_value")).with_columns(
        (pl.col("raw_value").is_not_null() & pl.col("raw_value").is_finite()).alias(
            "factor_is_valid"
        ),
        pl.when(pl.col("raw_value").is_not_null() & pl.col("raw_value").is_finite())
        .then(pl.lit(""))
        .otherwise(pl.lit("MISSING_COMPONENT_FACTOR"))
        .alias("factor_quality_flags"),
    )


def _condition_expression(item: Any, factor_names: tuple[str, ...]) -> pl.Expr:
    if not isinstance(item, dict):
        raise TypeError("rule condition must be a mapping")
    factor = str(item.get("factor", ""))
    if factor not in factor_names:
        raise ValueError(f"rule references unknown factor {factor!r}")
    operator = str(item.get("operator", "gt")).lower()
    value = float(item.get("value", 0.0))
    column = pl.col(factor)
    operations = {
        "gt": column > value,
        "ge": column >= value,
        "lt": column < value,
        "le": column <= value,
        "eq": column == value,
        "ne": column != value,
    }
    if operator == "between":
        if "upper" not in item:
            raise ValueError("between rule requires upper")
        return column.is_between(value, float(item["upper"]), closed="both").fill_null(False)
    try:
        return operations[operator].fill_null(False)
    except KeyError as exc:
        raise ValueError(f"unsupported rule operator: {operator}") from exc


def _conditions_expression(config: dict[str, Any], factor_names: tuple[str, ...]) -> pl.Expr:
    conditions = config.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("each rule requires a non-empty conditions list")
    expressions = [_condition_expression(item, factor_names) for item in conditions]
    logic = str(config.get("logic", "all")).lower()
    if logic == "all":
        return pl.all_horizontal(expressions)
    if logic == "any":
        return pl.any_horizontal(expressions)
    raise ValueError("rule logic must be all or any")


def _score_value(
    config: dict[str, Any],
    factor_names: tuple[str, ...],
    *,
    score_key: str,
    factor_key: str,
    default: float | None,
) -> pl.Expr:
    factor = config.get(factor_key)
    if factor is not None:
        factor = str(factor)
        if factor not in factor_names:
            raise ValueError(f"rule score_factor references unknown factor {factor!r}")
        return pl.col(factor)
    value = config.get(score_key, default)
    return pl.lit(None if value is None else float(value), dtype=pl.Float64)
