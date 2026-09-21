"""The small online orchestration boundary."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import ValidationError

from .errors import ConfigurationError, RegistryError, RouterError
from .models import RequirementPrediction, RouteDecision, Tier
from .policy import RoutingPolicy, ShortfallPolicy
from .predictor import DEFAULT_CAPABILITIES, JevPredictor, RequirementPredictor
from .registry import ModelRegistry


class Router:
    """Predict requirements, then delegate model choice to an injected policy."""

    def __init__(
        self,
        registry: ModelRegistry,
        predictor: RequirementPredictor | None = None,
        policy: RoutingPolicy | None = None,
    ) -> None:
        if not isinstance(registry, ModelRegistry):
            raise ConfigurationError("registry must be a ModelRegistry")
        self.registry = registry
        self._owns_predictor = predictor is None
        if predictor is None:
            capability_names = sorted(
                {name for model in registry.models for name in model.capabilities}
            )
            if not capability_names:
                raise ConfigurationError("registry has no available capabilities")
            descriptions = {
                name: DEFAULT_CAPABILITIES.get(
                    name, f"How much {name} ability does this task require?"
                )
                for name in capability_names
            }
            predictor = JevPredictor.from_env(capability_descriptions=descriptions)
        if not isinstance(predictor, RequirementPredictor):
            raise ConfigurationError("predictor must implement RequirementPredictor")
        if policy is None:
            policy = ShortfallPolicy()
        if not isinstance(policy, RoutingPolicy):
            raise ConfigurationError("policy must implement RoutingPolicy")
        self.predictor = predictor
        self.policy = policy

    def close(self) -> None:
        if self._owns_predictor and hasattr(self.predictor, "close"):
            self.predictor.close()

    def __enter__(self) -> "Router":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _models(self) -> Any:
        try:
            models = self.registry.models
        except AttributeError as exc:
            raise RegistryError("registry must expose a models attribute") from exc
        if models is None:
            raise RegistryError("registry models cannot be null")
        return models

    def route(
        self,
        task: str,
        *,
        tier: Tier = "balanced",
        provider: str | None = None,
        providers: Iterable[str] | None = None,
        max_cost: float | None = None,
        max_latency: float | None = None,
        max_routing_cost: float | None = None,
        max_latency_seconds: float | None = None,
    ) -> RouteDecision:
        """Return one stable decision envelope; never call a target model."""

        try:
            if provider is not None and providers is not None:
                raise ConfigurationError("provider and providers are mutually exclusive")
            if max_routing_cost is not None:
                if max_cost is not None:
                    raise ConfigurationError("max_cost and max_routing_cost are mutually exclusive")
                max_cost = max_routing_cost
            if max_latency_seconds is not None:
                if max_latency is not None:
                    raise ConfigurationError("max_latency and max_latency_seconds are mutually exclusive")
                max_latency = max_latency_seconds

            prediction = self.predictor.predict(task)
            if not isinstance(prediction, RequirementPrediction):
                # Custom predictors may return a plain mapping for convenience,
                # but confidence is still required to preserve the core contract.
                if isinstance(prediction, Mapping) and "requirements" in prediction:
                    try:
                        prediction = RequirementPrediction.model_validate(prediction)
                    except ValidationError as exc:
                        raise ConfigurationError(
                            "predictor returned an invalid requirement prediction",
                            details={"errors": exc.errors(include_url=False)},
                        ) from exc
                else:
                    raise ConfigurationError("predictor must return RequirementPrediction")
            available_capabilities = {
                name for model in self._models() for name in model.capabilities
            }
            unavailable = sorted(
                name
                for name, value in prediction.requirements.items()
                if value > 0 and name not in available_capabilities
            )
            if unavailable:
                raise ConfigurationError(
                    "prediction requires capabilities absent from the registry",
                    details={"capabilities": unavailable},
                )
            decision = self.policy.select(
                prediction.requirements,
                self._models(),
                tier=tier,
                provider=provider,
                providers=providers,
                max_cost=max_cost,
                max_latency=max_latency,
            )
            if not isinstance(decision, RouteDecision):
                if isinstance(decision, Mapping):
                    try:
                        decision = RouteDecision.model_validate(decision)
                    except ValidationError as exc:
                        raise ConfigurationError(
                            "routing policy returned an invalid decision",
                            details={"errors": exc.errors(include_url=False)},
                        ) from exc
                else:
                    raise ConfigurationError("routing policy must return RouteDecision")
            return decision.model_copy(
                update={
                    "requirements": dict(prediction.requirements),
                    "confidence": prediction.confidence,
                }
            )
        except RouterError as exc:
            safe_tier = tier if tier in ("efficiency", "balanced", "intelligence") else "balanced"
            return RouteDecision(
                tier=safe_tier,
                error=exc.as_route_error(),
                reason=str(exc),
            )
