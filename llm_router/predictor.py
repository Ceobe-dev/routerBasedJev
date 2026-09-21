"""Online task-requirement predictors.

The predictor deliberately knows only about TypeSafe's HTTP contract.  It does
not know anything about the registry or routing policy, which keeps the online
boundary replaceable in deployments without TypeSafe.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import httpx
from dotenv import load_dotenv

from .errors import ConfigurationError, PredictorError
from .models import RequirementPrediction


@runtime_checkable
class RequirementPredictor(Protocol):
    """Interface used by :class:`llm_router.router.Router`."""

    def predict(self, task: str) -> RequirementPrediction:
        """Predict normalized capability requirements for one task."""


DEFAULT_CAPABILITIES: dict[str, str] = {
    "reasoning": "How much reasoning ability does this task require?",
    "coding": "How much software coding ability does this task require?",
    "debugging": "How much debugging ability does this task require?",
    "tool_use": "How much tool-use ability does this task require?",
}


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictorError(f"{label} must be a number", details={"field": label})
    number = float(value)
    if not math.isfinite(number):
        raise PredictorError(f"{label} must be finite", details={"field": label})
    return number


class JevPredictor:
    """Call TypeSafe's Jev ``systemone`` endpoint and normalize score answers."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.typesafe.ai",
        model: str = "jev-latest",
        timeout: float = 30.0,
        client: Any | None = None,
        capability_descriptions: Mapping[str, Any] | None = None,
        capabilities: Mapping[str, Any] | None = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.0,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ConfigurationError("TYPESAFE_API_KEY is required")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ConfigurationError("JEV_BASE_URL must be a non-empty URL")
        if not isinstance(model, str) or not model.strip():
            raise ConfigurationError("JEV_MODEL must be a non-empty model name")
        try:
            timeout_number = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("timeout must be a number") from exc
        if not math.isfinite(timeout_number):
            raise ConfigurationError("timeout must be finite")
        if timeout_number <= 0:
            raise ConfigurationError("timeout must be greater than zero")
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or not 0 <= max_retries <= 3
        ):
            raise ConfigurationError("max_retries must be an integer between 0 and 3")
        backoff = _finite_number(retry_backoff_seconds, label="retry_backoff_seconds")
        if not 0 <= backoff <= 5:
            raise ConfigurationError("retry_backoff_seconds must be between 0 and 5")
        total_budget = (max_retries + 1) * timeout_number + backoff * (2**max_retries - 1)
        if total_budget > 120:
            raise ConfigurationError("configured Jev retry budget cannot exceed 120 seconds")

        if capability_descriptions is not None and capabilities is not None:
            raise ConfigurationError("capability_descriptions and capabilities are mutually exclusive")
        descriptions = dict(capability_descriptions if capability_descriptions is not None else capabilities or DEFAULT_CAPABILITIES)
        if not descriptions or any(not isinstance(k, str) or not k.strip() for k in descriptions):
            raise ConfigurationError("capability_descriptions must contain named capabilities")

        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.timeout = timeout_number
        self._owns_client = client is None
        self.client = client if client is not None else httpx.Client()
        self.capability_descriptions = descriptions
        self.max_retries = max_retries
        self.retry_backoff_seconds = backoff

    def close(self) -> None:
        """Release an internally-created HTTP connection pool."""

        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "JevPredictor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @classmethod
    def from_env(
        cls,
        *,
        client: Any | None = None,
        capability_descriptions: Mapping[str, Any] | None = None,
    ) -> "JevPredictor":
        """Build a predictor from the documented environment variables."""

        load_dotenv()
        key = os.getenv("TYPESAFE_API_KEY")
        if not key or not key.strip():
            raise ConfigurationError("TYPESAFE_API_KEY is required")
        base_url = os.getenv("JEV_BASE_URL", "https://api.typesafe.ai")
        model = os.getenv("JEV_MODEL", "jev-latest")
        raw_timeout = os.getenv("JEV_TIMEOUT_SECONDS", "30")
        try:
            timeout = float(raw_timeout)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("JEV_TIMEOUT_SECONDS must be a number") from exc
        try:
            max_retries = int(os.getenv("JEV_MAX_RETRIES", "2"))
            retry_backoff_seconds = float(os.getenv("JEV_RETRY_BACKOFF_SECONDS", "0.5"))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                "JEV_MAX_RETRIES and JEV_RETRY_BACKOFF_SECONDS must be numbers"
            ) from exc
        return cls(
            key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            client=client,
            capability_descriptions=capability_descriptions,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )

    def _questions(self) -> dict[str, dict[str, Any]]:
        questions: dict[str, dict[str, Any]] = {}
        for name, description in self.capability_descriptions.items():
            if isinstance(description, Mapping):
                question = dict(description)
                question.setdefault("type", "score")
                question.setdefault("instructions", f"How much {name} ability does this task require?")
                criteria = question.get("criteria")
            else:
                question = {"type": "score", "instructions": str(description)}
                criteria = None
            if question.get("type") != "score":
                raise ConfigurationError(f"capability question {name!r} must have type 'score'")
            if criteria is None:
                # Ten levels are accepted by TypeSafe and make score/(n-1)
                # normalization unambiguous while retaining enough resolution.
                criteria = [f"Requirement level {i}/9" for i in range(10)]
                question["criteria"] = criteria
            if not isinstance(criteria, Sequence) or isinstance(criteria, (str, bytes)):
                raise ConfigurationError(f"criteria for {name!r} must be a sequence")
            if not 2 <= len(criteria) <= 10:
                raise ConfigurationError(f"criteria for {name!r} must contain 2 to 10 levels")
            questions[name] = question
        return questions

    def _post(self, payload: dict[str, Any]) -> Any:
        url = f"{self.base_url}/v1/systemone"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_status: int | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(url, headers=headers, json=payload, timeout=self.timeout)
            except (httpx.HTTPError, OSError, TimeoutError) as exc:
                if attempt < self.max_retries:
                    if self.retry_backoff_seconds:
                        time.sleep(self.retry_backoff_seconds * (2**attempt))
                    continue
                raise PredictorError(
                    "Jev request failed",
                    retryable=True,
                    details={"reason": str(exc)},
                ) from exc
            status = getattr(response, "status_code", None)
            retryable_status = status == 429 or (
                isinstance(status, int) and 500 <= status < 600
            )
            if retryable_status and attempt < self.max_retries:
                last_status = int(status)
                if self.retry_backoff_seconds:
                    time.sleep(self.retry_backoff_seconds * (2**attempt))
                continue
            if status is not None and status >= 400:
                message = f"Jev request returned HTTP {status}"
                raise PredictorError(
                    message,
                    retryable=retryable_status,
                    details={"status_code": status},
                )
            try:
                return response.json()
            except (ValueError, TypeError) as exc:
                raise PredictorError("Jev response was not valid JSON") from exc
        raise PredictorError("Jev request could not be completed", retryable=True, details={"status_code": last_status})

    def predict(self, task: str) -> RequirementPrediction:
        if not isinstance(task, str) or not task.strip():
            raise PredictorError("task must be a non-empty string")
        questions = self._questions()
        response = self._post({"state": task, "model": self.model, "questions": questions})
        if not isinstance(response, Mapping):
            raise PredictorError("Jev response must be an object")
        answers = response.get("answers")
        if not isinstance(answers, Mapping):
            raise PredictorError("Jev response is missing an answers object")
        expected = set(questions)
        actual = set(answers)
        if actual != expected:
            raise PredictorError(
                "Jev answer set does not match requested capabilities",
                details={"missing": sorted(expected - actual), "extra": sorted(actual - expected)},
            )

        requirements: dict[str, float] = {}
        confidences: list[float] = []
        for capability, question in questions.items():
            answer = answers[capability]
            if not isinstance(answer, Mapping) or answer.get("type") != "score":
                raise PredictorError(f"answer {capability!r} must be a score answer")
            criteria = question["criteria"]
            raw_score = _finite_number(answer.get("score"), label=f"answers.{capability}.score")
            maximum = len(criteria) - 1
            if raw_score < 0 or raw_score > maximum:
                raise PredictorError(
                    f"answers.{capability}.score is outside its score levels",
                    details={"min": 0, "max": maximum},
                )
            confidence = _finite_number(answer.get("confidence"), label=f"answers.{capability}.confidence")
            if not 0 <= confidence <= 1:
                raise PredictorError(f"answers.{capability}.confidence must be between 0 and 1")
            requirements[capability] = raw_score / maximum
            confidences.append(confidence)
        return RequirementPrediction(requirements=requirements, confidence=sum(confidences) / len(confidences))
