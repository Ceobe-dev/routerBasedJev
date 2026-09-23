"""Strict YAML loading and the routing model registry.

Benchmark observations and cached profiles intentionally have separate load
paths.  ``benchmarks.yaml`` is the factual source; ``models.yaml`` can be
regenerated from it and is what the online router normally reads.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError, RegistryError
from .models import BenchmarkObservation, ModelMetadata, ModelProfile, Price


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects duplicate YAML keys."""


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping", node.start_mark, f"duplicate key: {key!r}", key_node.start_mark
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def load_yaml(path: str | Path) -> Any:
    """Read one YAML document, rejecting malformed or duplicate-key input."""

    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.load(handle, Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise RegistryError(f"unable to parse YAML: {path}") from exc
    if data is None:
        raise ConfigurationError(f"YAML document is empty: {path}")
    return data


def _records(data: Any, section: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or section not in data:
        raise ConfigurationError(f"YAML must contain a {section!r} mapping")
    records = data[section]
    if isinstance(records, list):
        result: dict[str, Any] = {}
        for record in records:
            if not isinstance(record, Mapping) or not isinstance(record.get("model_id"), str):
                raise ConfigurationError(f"every {section} list item needs model_id")
            model_id = record["model_id"]
            if model_id in result:
                raise ConfigurationError(f"duplicate model_id: {model_id}")
            result[model_id] = {key: value for key, value in record.items() if key != "model_id"}
        return result
    if not isinstance(records, Mapping) or not records:
        raise ConfigurationError(f"{section} must be a non-empty mapping")
    if any(not isinstance(key, str) or not key.strip() for key in records):
        raise ConfigurationError(f"{section} keys must be non-empty strings")
    return dict(records)


def _benchmark_catalog(data: Any) -> dict[str, Mapping[str, Any]]:
    raw = data.get("benchmarks") if isinstance(data, Mapping) else None
    if not isinstance(raw, Mapping) or not raw:
        raise ConfigurationError("benchmarks YAML must contain a non-empty benchmarks mapping")
    catalog: dict[str, Mapping[str, Any]] = {}
    for name, definition in raw.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(definition, Mapping):
            raise ConfigurationError("invalid benchmark definition")
        for required in ("metric", "protocol_id", "source_url"):
            if not isinstance(definition.get(required), str) or not definition[required].strip():
                raise ConfigurationError(f"benchmark {name!r} needs {required}")
        bounds = (definition.get("raw_min"), definition.get("raw_max"))
        for bound in bounds:
            if bound is not None and (isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound)):
                raise ConfigurationError(f"benchmark {name!r} needs finite numeric raw bounds")
        if all(bound is not None for bound in bounds) and bounds[0] >= bounds[1]:
            raise ConfigurationError(f"benchmark {name!r} needs increasing raw bounds")
        normalization = definition.get("normalization")
        if normalization is not None:
            if not isinstance(normalization, Mapping) or normalization.get("method") != "piecewise_linear":
                raise ConfigurationError(f"benchmark {name!r} needs a piecewise_linear normalization or null")
            if not isinstance(normalization.get("version"), str) or not normalization["version"]:
                raise ConfigurationError(f"benchmark {name!r} needs a calibration version")
            if definition.get("direction") not in ("higher_is_better", "lower_is_better"):
                raise ConfigurationError(f"benchmark {name!r} has an invalid direction")
            anchors = normalization.get("anchors")
            if not isinstance(anchors, list) or len(anchors) < 2:
                raise ConfigurationError(f"benchmark {name!r} needs at least two anchors")
            previous_raw = previous_score = None
            for anchor in anchors:
                if not isinstance(anchor, Mapping) or set(anchor) != {"raw", "score"}:
                    raise ConfigurationError(f"benchmark {name!r} has an invalid anchor")
                x, y = anchor["raw"], anchor["score"]
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (x, y)):
                    raise ConfigurationError(f"benchmark {name!r} anchors must be finite numbers")
                if not 0 <= y <= 1 or (previous_raw is not None and x <= previous_raw):
                    raise ConfigurationError(f"benchmark {name!r} anchors need increasing raw values and scores in [0, 1]")
                if previous_score is not None:
                    increasing = definition["direction"] == "higher_is_better"
                    if (increasing and y <= previous_score) or (not increasing and y >= previous_score):
                        raise ConfigurationError(f"benchmark {name!r} anchor scores must follow its direction strictly")
                previous_raw, previous_score = x, y
            expected = (0, 1) if definition["direction"] == "higher_is_better" else (1, 0)
            if (anchors[0]["score"], anchors[-1]["score"]) != expected:
                raise ConfigurationError(f"benchmark {name!r} anchors must cover the full [0, 1] scale")
        catalog[name] = definition
    return catalog


def _normalize_observation(raw: float | None, definition: Mapping[str, Any]) -> float | None:
    """Apply frozen anchors only; online model/provider pools never refit them."""
    normalization = definition.get("normalization")
    if raw is None or normalization is None:
        return None
    anchors = normalization["anchors"]
    if raw <= anchors[0]["raw"]:
        return float(anchors[0]["score"])
    for left, right in zip(anchors, anchors[1:]):
        if raw <= right["raw"]:
            ratio = (raw - left["raw"]) / (right["raw"] - left["raw"])
            return left["score"] + ratio * (right["score"] - left["score"])
    return float(anchors[-1]["score"])


def load_jev_prompt_config(path: str | Path) -> dict[str, Any]:
    """Validate shared score rubrics and all three rounds in one YAML boundary."""
    data = load_yaml(path)
    if not isinstance(data, Mapping) or not isinstance(data.get("calibration_version"), str) or not data["calibration_version"].strip():
        raise ConfigurationError("Jev prompts need a calibration_version")
    common = data.get("common_instructions")
    capabilities = data.get("capabilities")
    rounds = data.get("rounds")
    if not isinstance(common, str) or not common.strip() or not isinstance(capabilities, Mapping) or not capabilities:
        raise ConfigurationError("Jev prompts need common instructions and capability rubrics")
    if not isinstance(rounds, Mapping):
        raise ConfigurationError("Jev prompts need three rounds")
    for name, definition in capabilities.items():
        if not isinstance(name, str) or not isinstance(definition, Mapping):
            raise ConfigurationError("invalid Jev capability rubric")
        criteria = definition.get("criteria")
        if not isinstance(definition.get("definition"), str) or not definition["definition"].strip():
            raise ConfigurationError(f"Jev capability {name!r} needs a definition")
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10 or any(not isinstance(c, str) or not c.strip() for c in criteria):
            raise ConfigurationError(f"Jev capability {name!r} needs 2 to 10 descriptive criteria")
        if len(set(criteria)) != len(criteria):
            raise ConfigurationError(f"Jev capability {name!r} has duplicate criteria")
    questions, requestion = {}, {}
    for number in (1, 2, 3):
        entry = rounds.get(number, rounds.get(str(number)))
        instructions = entry.get("instructions") if isinstance(entry, Mapping) else None
        if not isinstance(instructions, Mapping) or set(instructions) != set(capabilities):
            raise ConfigurationError(f"Jev round {number} must cover every configured capability")
        composed = {}
        for name, prompt in instructions.items():
            if not isinstance(prompt, str) or not prompt.strip():
                raise ConfigurationError(f"Jev round {number}/{name} needs instructions")
            composed[name] = f"仅评估 {name}：{capabilities[name]['definition']}\n{common}\n{prompt}"
            if number == 1:
                questions[name] = {"type": "score", "instructions": composed[name], "criteria": list(capabilities[name]["criteria"])}
        if number != 1:
            requestion[number] = {"instructions": composed}
    return {"version": data["calibration_version"], "questions": questions, "rounds": requestion}


def load_benchmark_observations(path: str | Path) -> tuple[set[str], dict[str, dict[str, BenchmarkObservation]]]:
    """Load benchmark definitions and per-model observations from YAML."""

    data = load_yaml(path)
    catalog = _benchmark_catalog(data)
    raw_observations = data.get("observations", {})
    if not isinstance(raw_observations, Mapping):
        raise ConfigurationError("observations must be a mapping")
    result: dict[str, dict[str, BenchmarkObservation]] = {}
    for model_id, observations in raw_observations.items():
        if not isinstance(model_id, str) or not model_id.strip() or not isinstance(observations, Mapping):
            raise ConfigurationError("observations must map model ids to mappings")
        model_result: dict[str, BenchmarkObservation] = {}
        for benchmark_id, value in observations.items():
            if benchmark_id not in catalog:
                raise ConfigurationError(f"unknown benchmark in observations: {benchmark_id!r}")
            definition = catalog[benchmark_id]
            if value is None:
                # Null is the explicit unknown form and is intentionally not
                # materialized as a zero-valued BenchmarkObservation.
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                value = {"raw_value": value}
            if not isinstance(value, Mapping):
                raise ConfigurationError(f"invalid observation for {model_id}/{benchmark_id}")
            payload = dict(value)
            # Cached derived scores are never a source of truth.
            payload.pop("normalized_0_1", None)
            payload.setdefault("metric", definition["metric"])
            payload.setdefault("protocol_id", definition["protocol_id"])
            payload.setdefault("source_url", definition["source_url"])
            for optional in ("dataset_version", "eval_date", "notes"):
                if optional in definition:
                    payload.setdefault(optional, definition[optional])
            try:
                observation = BenchmarkObservation.model_validate(payload)
            except Exception as exc:
                raise ConfigurationError(
                    f"invalid observation for {model_id}/{benchmark_id}"
                ) from exc
            for field in ("metric", "protocol_id", "dataset_version"):
                if definition.get("normalization") is not None and getattr(observation, field) != definition.get(field):
                    raise ConfigurationError(f"observation protocol mismatch: {model_id}/{benchmark_id}/{field}")
            if observation.raw_value is not None:
                low, high = definition.get("raw_min"), definition.get("raw_max")
                if (low is not None and observation.raw_value < low) or (high is not None and observation.raw_value > high):
                    raise ConfigurationError(f"raw score outside benchmark range: {model_id}/{benchmark_id}")
            model_result[benchmark_id] = observation.model_copy(update={
                "normalized_0_1": _normalize_observation(observation.raw_value, definition)
            })
        result[model_id] = model_result
    return set(catalog), result


def load_model_metadata(
    models_path: str | Path,
    *,
    benchmarks_path: str | Path | None = None,
) -> list[ModelMetadata]:
    """Load metadata and, when supplied, join benchmark facts by model id."""

    data = load_yaml(models_path)
    model_records = _records(data, "models")
    observations: dict[str, dict[str, BenchmarkObservation]] = {}
    if benchmarks_path is not None:
        _, observations = load_benchmark_observations(benchmarks_path)
    result: list[ModelMetadata] = []
    for model_id, record in model_records.items():
        if not isinstance(record, Mapping):
            raise ConfigurationError(f"model {model_id!r} must be a mapping")
        payload = dict(record)
        payload["model_id"] = model_id
        # A models.yaml may also be the generated profile snapshot.  The
        # benchmark facts still come exclusively from benchmarks.yaml, so the
        # cached capability vector is ignored on the metadata load path.
        payload.pop("capabilities", None)
        payload.pop("capability_benchmark_counts", None)
        payload.pop("calibration_version", None)
        if benchmarks_path is not None or "benchmark_scores" not in payload:
            payload["benchmark_scores"] = observations.get(model_id, {})
        try:
            result.append(ModelMetadata.model_validate(payload))
        except Exception as exc:
            raise ConfigurationError(f"invalid model metadata: {model_id}") from exc
    return sorted(result, key=lambda model: model.model_id)


class ModelRegistry:
    """Stable in-memory registry of routing-facing model profiles."""

    def __init__(self, models: Iterable[ModelProfile] = (), *, excluded_models: Mapping[str, str] | None = None) -> None:
        profiles: dict[str, ModelProfile] = {}
        for model in models:
            try:
                profile = model if isinstance(model, ModelProfile) else ModelProfile.model_validate(model)
            except Exception as exc:
                raise RegistryError("invalid model profile") from exc
            if profile.model_id in profiles:
                raise RegistryError(f"duplicate model_id: {profile.model_id}")
            profiles[profile.model_id] = profile
        self._models = profiles
        self.excluded_models = dict(excluded_models or {})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelRegistry":
        data = load_yaml(path)
        records = _records(data, "models")
        profiles: list[ModelProfile] = []
        for model_id, record in records.items():
            if not isinstance(record, Mapping):
                raise ConfigurationError(f"model {model_id!r} must be a mapping")
            payload = dict(record)
            payload["model_id"] = model_id
            if "capabilities" not in payload:
                raise ConfigurationError(f"cached model {model_id!r} needs capabilities")
            try:
                profiles.append(ModelProfile.model_validate(payload))
            except Exception as exc:
                raise RegistryError(f"invalid model profile: {model_id}") from exc
        excluded = data.get("excluded_models", {})
        if not isinstance(excluded, Mapping) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in excluded.items()):
            raise ConfigurationError("excluded_models must map model ids to reasons")
        return cls(profiles, excluded_models=excluded)

    load = from_yaml

    @classmethod
    def build_from_yaml(
        cls,
        models_path: str | Path,
        benchmarks_path: str | Path,
        capabilities_path: str | Path,
        *,
        aggregator: Any = None,
    ) -> "ModelRegistry":
        from .capability import CapabilityBuilder

        benchmark_ids, _ = load_benchmark_observations(benchmarks_path)
        metadata = load_model_metadata(models_path, benchmarks_path=benchmarks_path)
        builder = CapabilityBuilder.from_yaml(
            capabilities_path, aggregator, benchmark_ids=benchmark_ids
        )
        version = load_yaml(capabilities_path).get("calibration_version")
        if not isinstance(version, str) or not version.strip():
            raise ConfigurationError("capability definitions need a calibration_version")
        catalog = _benchmark_catalog(load_yaml(benchmarks_path))
        active_benchmarks = {name for names in builder.definitions.values() for name in names}
        for name in active_benchmarks:
            normalization = catalog[name].get("normalization")
            if not normalization or normalization["version"] != version:
                raise ConfigurationError(f"active benchmark {name!r} has no matching calibration version")
        eligible, excluded = [], {}
        for model in metadata:
            if any(name in active_benchmarks and score.normalized_0_1 is not None for name, score in model.benchmark_scores.items()):
                eligible.append(model)
            else:
                excluded[model.model_id] = "No observations for the active calibrated benchmarks; original metadata is retained separately."
        if not eligible:
            raise ConfigurationError("no models have active calibrated benchmark observations")
        profiles = [profile.model_copy(update={"calibration_version": version}) for profile in builder.build(eligible)]
        return cls(profiles, excluded_models=excluded)

    def get(self, model_id: str) -> ModelProfile | None:
        return self._models.get(model_id)

    def require(self, model_id: str) -> ModelProfile:
        profile = self.get(model_id)
        if profile is None:
            raise RegistryError(f"unknown model: {model_id}")
        return profile

    def list_models(self) -> list[ModelProfile]:
        return [self._models[key] for key in sorted(self._models)]

    @property
    def models(self) -> tuple[ModelProfile, ...]:
        return tuple(self.list_models())

    def save(self, path: str | Path) -> None:
        """Write deterministic cached profiles suitable for version control."""

        payload: dict[str, Any] = {"models": {}, "excluded_models": self.excluded_models}
        for profile in self.list_models():
            model = profile.model_dump(mode="json", exclude_none=False)
            # Preserve the concise mapping shape used by the checked-in YAML.
            model.pop("model_id", None)
            # ``routing_cost`` is a pydantic computed field, not a constructor
            # input and therefore must not be persisted as an unknown key.
            if isinstance(model.get("price"), dict):
                model["price"].pop("routing_cost", None)
            payload["models"][profile.model_id] = model
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        except OSError as exc:
            raise RegistryError(f"unable to save model registry: {path}") from exc
