"""Offline construction of model capability profiles.

The builder deliberately works on :class:`ModelMetadata` (the benchmark
facts) and returns a :class:`ModelProfile` (the cached, routing-facing view).
Keeping the two objects separate prevents an aggregate score from replacing
the raw benchmark evidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from .errors import ConfigurationError, RegistryError
from .models import ModelMetadata, ModelProfile


@runtime_checkable
class CapabilityAggregator(Protocol):
    """Replaceable strategy for combining normalized benchmark scores."""

    def aggregate(self, scores: Sequence[float]) -> float:
        """Return one normalized capability score for non-empty ``scores``."""


class MeanAggregator:
    """Default arithmetic-mean aggregator used by the MVP."""

    def aggregate(self, scores: Sequence[float]) -> float:
        if not scores:
            raise ConfigurationError("cannot aggregate an empty score sequence")
        return sum(scores) / len(scores)


def _definitions_from_mapping(
    definitions: Mapping[str, object],
    *,
    benchmark_ids: Iterable[str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Validate the small capability-definition DSL used by YAML and Python."""

    if not isinstance(definitions, Mapping) or not definitions:
        raise ConfigurationError("capability definitions must be a non-empty mapping")
    known = set(benchmark_ids) if benchmark_ids is not None else None
    result: dict[str, tuple[str, ...]] = {}
    for name, definition in definitions.items():
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError("capability names must be non-empty strings")
        if isinstance(definition, Mapping):
            values = definition.get("benchmarks")
        else:
            values = definition
        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            raise ConfigurationError(f"capability {name!r} must define benchmarks")
        values = tuple(values)
        if not values or any(not isinstance(item, str) or not item.strip() for item in values):
            raise ConfigurationError(f"capability {name!r} has an empty or invalid benchmark")
        if len(set(values)) != len(values):
            raise ConfigurationError(f"capability {name!r} contains duplicate benchmarks")
        if known is not None:
            missing = sorted(set(values) - known)
            if missing:
                raise ConfigurationError(
                    f"capability {name!r} references unknown benchmark(s): {', '.join(missing)}"
                )
        result[name] = values
    return result


class CapabilityBuilder:
    """Build immutable model profiles from benchmark observations.

    ``definitions`` accepts either ``{"reasoning": {"benchmarks": [...]}}``
    (the YAML shape) or the shorter ``{"reasoning": [...]}`` form.  Only
    ``normalized_0_1`` observations are aggregated; a missing value is
    unknown, not zero.
    """

    def __init__(
        self,
        definitions: Mapping[str, object],
        aggregator: CapabilityAggregator | None = None,
        *,
        benchmark_ids: Iterable[str] | None = None,
    ) -> None:
        self._definitions = _definitions_from_mapping(
            definitions, benchmark_ids=benchmark_ids
        )
        self.aggregator = aggregator or MeanAggregator()
        if not isinstance(self.aggregator, CapabilityAggregator):
            raise ConfigurationError("aggregator must implement CapabilityAggregator")

    @property
    def definitions(self) -> dict[str, tuple[str, ...]]:
        return dict(self._definitions)

    @property
    def configured_benchmark_count(self) -> int:
        """Return the number of unique public benchmarks in the configuration."""

        return len(
            {
                benchmark
                for benchmarks in self._definitions.values()
                for benchmark in benchmarks
            }
        )

    @property
    def configured_benchmark_counts(self) -> dict[str, int]:
        """Return configured benchmark counts grouped by capability."""

        return {
            capability: len(benchmarks)
            for capability, benchmarks in self._definitions.items()
        }

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        aggregator: CapabilityAggregator | None = None,
        *,
        benchmark_ids: Iterable[str] | None = None,
    ) -> "CapabilityBuilder":
        """Load and validate a capability-definition YAML file."""

        try:
            from .registry import load_yaml

            data = load_yaml(path)
        except (OSError, ValueError, TypeError) as exc:
            raise RegistryError(f"unable to load capability definitions: {path}") from exc
        if not isinstance(data, Mapping) or "capabilities" not in data:
            raise ConfigurationError("capability YAML must contain a capabilities mapping")
        return cls(data["capabilities"], aggregator, benchmark_ids=benchmark_ids)

    def build_model(self, metadata: ModelMetadata) -> ModelProfile:
        """Build one profile, omitting capabilities with no observations."""

        if not isinstance(metadata, ModelMetadata):
            try:
                metadata = ModelMetadata.model_validate(metadata)
            except Exception as exc:  # pydantic's concrete exception is an implementation detail
                raise ConfigurationError("invalid model metadata") from exc
        capabilities: dict[str, float] = {}
        capability_counts: dict[str, int] = {}
        for capability, benchmark_names in self._definitions.items():
            scores = [
                observation.normalized_0_1
                for benchmark in benchmark_names
                if (observation := metadata.benchmark_scores.get(benchmark)) is not None
                and observation.normalized_0_1 is not None
            ]
            if scores:
                try:
                    value = self.aggregator.aggregate(scores)
                except ConfigurationError:
                    raise
                except Exception as exc:
                    raise ConfigurationError(
                        f"aggregator failed for capability {capability!r}"
                    ) from exc
                if not 0 <= value <= 1:
                    raise ConfigurationError(
                        f"aggregated capability {capability!r} is outside [0, 1]"
                    )
                capabilities[capability] = value
                capability_counts[capability] = len(scores)
        if not capabilities:
            raise ConfigurationError(
                f"model {metadata.model_id!r} has no non-null capability observations"
            )
        try:
            return ModelProfile(
                model_id=metadata.model_id,
                provider=metadata.provider,
                capabilities=capabilities,
                capability_benchmark_counts=capability_counts,
                price=metadata.price,
                latency_p50_seconds=metadata.latency_p50_seconds,
                context_length=metadata.context_length,
            )
        except Exception as exc:
            raise ConfigurationError(f"invalid profile for model {metadata.model_id!r}") from exc

    def build(self, models: Iterable[ModelMetadata]) -> list[ModelProfile]:
        """Build profiles in stable model-id order."""

        profiles = [self.build_model(model) for model in models]
        if len({profile.model_id for profile in profiles}) != len(profiles):
            raise ConfigurationError("duplicate model_id in model metadata")
        return sorted(profiles, key=lambda profile: profile.model_id)

    build_models = build
