"""Core data contracts shared by the offline builder and online router."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

Score = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
Tier = Literal["efficiency", "balanced", "intelligence"]


class FrozenModel(BaseModel):
    """Immutable, strict base model for public contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Price(FrozenModel):
    """Published token price in USD per one million tokens."""

    input_per_million: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    output_per_million: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    cached_input_per_million: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    source_url: str | None = None

    @computed_field
    @property
    def routing_cost(self) -> float | None:
        """Comparable MVP cost: one million input plus output tokens."""

        if self.input_per_million is None or self.output_per_million is None:
            return None
        return self.input_per_million + self.output_per_million


class BenchmarkObservation(FrozenModel):
    """A traceable raw benchmark result; null values mean unknown, never zero."""

    raw_value: Annotated[float, Field(allow_inf_nan=False)] | None
    normalized_0_1: Score | None = None
    metric: str
    dataset_version: str | None = None
    protocol_id: str
    eval_date: date | None = None
    source_url: str
    notes: str | None = None


class ModelMetadata(FrozenModel):
    """Provider metadata and benchmark facts used by the offline builder."""

    model_id: str
    provider: str
    benchmark_scores: dict[str, BenchmarkObservation] = Field(default_factory=dict)
    price: Price = Field(default_factory=Price)
    latency_p50_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    context_length: Annotated[int, Field(gt=0)] | None = None


class ModelProfile(FrozenModel):
    """Precomputed model capability profile consumed by online routing."""

    model_id: str
    provider: str
    capabilities: dict[str, Score]
    capability_benchmark_counts: dict[str, int] = Field(default_factory=dict)
    price: Price = Field(default_factory=Price)
    latency_p50_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    context_length: Annotated[int, Field(gt=0)] | None = None

    @field_validator("capabilities")
    @classmethod
    def require_capabilities(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("a model profile needs at least one capability")
        return value

    @field_validator("capability_benchmark_counts")
    @classmethod
    def validate_capability_counts(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not isinstance(count, int) or isinstance(count, bool) or count < 1 for count in value.values()):
            raise ValueError("capability benchmark counts must be positive integers")
        return value


class RequirementPrediction(FrozenModel):
    """Task capability requirements and predictor certainty."""

    requirements: dict[str, Score]
    confidence: Score

    @field_validator("requirements")
    @classmethod
    def require_requirements(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("prediction needs at least one requirement")
        return value


class RouteError(FrozenModel):
    """Stable error envelope returned for expected operational failures."""

    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class RouteDecision(FrozenModel):
    """The single success and failure response shape exposed by every entrypoint."""

    model: str | None = None
    provider: str | None = None
    tier: Tier = "balanced"
    requirements: dict[str, Score] = Field(default_factory=dict)
    shortfall: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    success_probability: Score | None = None
    confidence: Score | None = None
    fallback: bool = False
    reason: str = ""
    error: RouteError | None = None
