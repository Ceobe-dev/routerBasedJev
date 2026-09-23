"""Routing policies for selecting a model from a prepared registry."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from .errors import RoutingError
from .models import ModelProfile, RouteDecision, Tier


@runtime_checkable
class RoutingPolicy(Protocol):
    """Replaceable model-selection policy."""

    def select(
        self,
        requirements: Mapping[str, float],
        models: Sequence[ModelProfile],
        *,
        tier: Tier = "balanced",
        provider: str | None = None,
        providers: Iterable[str] | None = None,
        max_cost: float | None = None,
        max_latency: float | None = None,
    ) -> RouteDecision:
        ...


class ShortfallPolicy:
    """Choose the least expensive model meeting the tier shortfall budget."""

    thresholds: dict[str, float] = {"efficiency": 0.10, "balanced": 0.05, "intelligence": 0.0}

    @staticmethod
    def _cost(model: ModelProfile) -> float | None:
        value = getattr(getattr(model, "price", None), "routing_cost", None)
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) and number >= 0 else None

    @staticmethod
    def _latency(model: ModelProfile) -> float | None:
        value = getattr(model, "latency_p50_seconds", None)
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) and number > 0 else None

    @classmethod
    def _shortfall(
        cls, requirements: Mapping[str, float], model: ModelProfile
    ) -> float | None:
        capabilities = getattr(model, "capabilities", {}) or {}
        total = 0.0
        for name, required in requirements.items():
            if name not in capabilities:
                if float(required) > 0:
                    return None
                continue
            capability = capabilities[name]
            try:
                capability_value = float(capability)
            except (TypeError, ValueError):
                capability_value = 0.0
            total += max(0.0, float(required) - capability_value)
        return total

    @classmethod
    def _filtered(
        cls,
        models: Sequence[ModelProfile],
        *,
        provider: str | None,
        providers: Iterable[str] | None,
        max_cost: float | None,
        max_latency: float | None,
    ) -> list[ModelProfile]:
        if provider is not None and providers is not None:
            raise RoutingError("provider and providers are mutually exclusive")
        if isinstance(providers, str):
            raise RoutingError("providers must be a sequence of strings")
        provider_set = set(providers) if providers is not None else None
        if provider_set is not None and any(not isinstance(item, str) for item in provider_set):
            raise RoutingError("providers must contain strings")
        if provider is not None and not isinstance(provider, str):
            raise RoutingError("provider must be a string")
        result: list[ModelProfile] = []
        for model in models:
            model_provider = getattr(model, "provider", None)
            if provider is not None and model_provider != provider:
                continue
            if provider_set is not None and model_provider not in provider_set:
                continue
            cost = cls._cost(model)
            latency = cls._latency(model)
            # An explicit constraint cannot be satisfied by an unknown value.
            if max_cost is not None and (cost is None or cost > max_cost):
                continue
            if max_latency is not None and (latency is None or latency > max_latency):
                continue
            result.append(model)
        return result

    @classmethod
    def select(
        cls,
        requirements: Mapping[str, float],
        models: Sequence[ModelProfile],
        *,
        tier: Tier = "balanced",
        provider: str | None = None,
        providers: Iterable[str] | None = None,
        max_cost: float | None = None,
        max_latency: float | None = None,
    ) -> RouteDecision:
        if tier not in cls.thresholds:
            raise RoutingError(f"unknown routing tier: {tier}")
        if not isinstance(requirements, Mapping) or not requirements:
            raise RoutingError("requirements must be a non-empty mapping")
        for name, value in requirements.items():
            if not isinstance(name, str) or not isinstance(value, (int, float)) or isinstance(value, bool):
                raise RoutingError("requirements must map capability names to numbers")
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise RoutingError("requirements must be finite scores in [0, 1]")
        if max_cost is not None and (not isinstance(max_cost, (int, float)) or not math.isfinite(float(max_cost)) or float(max_cost) < 0):
            raise RoutingError("max_cost must be a non-negative finite number")
        if max_latency is not None and (not isinstance(max_latency, (int, float)) or not math.isfinite(float(max_latency)) or float(max_latency) <= 0):
            raise RoutingError("max_latency must be a positive finite number")
        if not isinstance(models, Sequence):
            raise RoutingError("registry models must be a sequence")

        candidates = cls._filtered(
            models,
            provider=provider,
            providers=providers,
            max_cost=None if max_cost is None else float(max_cost),
            max_latency=None if max_latency is None else float(max_latency),
        )
        if not candidates:
            raise RoutingError("no candidate models satisfy provider and constraints")
        scored = []
        for model in candidates:
            shortfall = cls._shortfall(requirements, model)
            if shortfall is not None:
                scored.append((model, shortfall, cls._cost(model)))
        if not scored:
            raise RoutingError(
                "no candidate model has data for every required capability",
                details={"required_capabilities": sorted(requirements)},
            )

        threshold = cls.thresholds[tier]
        qualified = [item for item in scored if item[1] <= threshold]
        if qualified:
            # Unknown prices are valid but always lose to known prices.
            chosen, shortfall, _ = min(
                qualified,
                key=lambda item: (
                    item[2] is None,
                    item[2] if item[2] is not None else math.inf,
                    item[0].model_id,
                ),
            )
            fallback = False
            reason = "selected lowest-cost model meeting tier shortfall"
        else:
            # Capability fit is primary for a fallback; cost only breaks equal
            # shortfalls. Unknown costs are retained as a last resort.
            chosen, shortfall, _ = min(
                scored,
                key=lambda item: (
                    item[1],
                    item[2] is None,
                    item[2] if item[2] is not None else math.inf,
                    item[0].model_id,
                ),
            )
            fallback = True
            reason = "no model met tier shortfall; selected closest fallback"
        return RouteDecision(
            model=getattr(chosen, "model_id", None),
            provider=getattr(chosen, "provider", None),
            tier=tier,
            shortfall=shortfall,
            # Relative benchmark positions do not estimate task success rates.
            success_probability=None,
            fallback=fallback,
            reason=reason,
        )

    # Friendly aliases for callers that use the policy outside Router.
    choose = select
    route = select
