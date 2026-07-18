"""Factor discovery and construction."""

from __future__ import annotations

from trend_trader.factors.base import Factor


class FactorRegistry:
    def __init__(self) -> None:
        self._factors: dict[str, Factor] = {}

    def register(self, factor: Factor, *, replace: bool = False) -> None:
        name = factor.name.strip().lower()
        if not name:
            raise ValueError("factor name must not be empty")
        if name in self._factors and not replace:
            raise ValueError(f"factor already registered: {name}")
        self._factors[name] = factor

    def get(self, name: str) -> Factor:
        normalized = name.strip().lower()
        try:
            return self._factors[normalized]
        except KeyError as exc:
            available = ", ".join(sorted(self._factors))
            raise KeyError(f"unknown factor {normalized!r}; available: {available}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factors))


default_registry = FactorRegistry()
