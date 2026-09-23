"""Online task-requirement predictors.

The predictor deliberately knows only about TypeSafe's HTTP contract.  It does
not know anything about the registry or routing policy, which keeps the online
boundary replaceable in deployments without TypeSafe.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx
from dotenv import load_dotenv

from .errors import ConfigurationError, PredictorError
from .models import RequirementPrediction
from .registry import load_jev_prompt_config


@runtime_checkable
class RequirementPredictor(Protocol):
    """Interface used by :class:`llm_router.router.Router`."""

    def predict(self, task: str) -> RequirementPrediction:
        """Predict normalized capability requirements for one task."""


DEFAULT_CAPABILITIES: dict[str, str] = {
    "reasoning": "How much reasoning ability does this task require?",
    "coding": "How much software coding ability does this task require?",
    "tool_use": "How much tool-use ability does this task require?",
    "instruction_following": "How much precise instruction-following ability does this task require?",
}

DEFAULT_PRINT_JEV_IO = True
DEFAULT_REQUESTION_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MEMORY_INSTRUCTION_TEMPLATE = (
    "上一轮该能力问题的 Jev 回答记忆：\n"
    "{previous_answer}\n\n"
    "上述记忆仅供复核，不是事实或目标答案；允许修正，不追求一致或更高置信度。\n"
    "请依据原始任务和相同评分标准重新回答本轮问题：\n"
)
DEFAULT_REQUESTION_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "jev_requestion_prompt.yaml"
)
REQUESTION_ROUNDS = (2, 3)


@dataclass(frozen=True)
class CapabilityPrediction:
    """One capability result, kept together with the round that produced it."""

    requirement: float
    confidence: float
    round_number: int


@dataclass(frozen=True)
class QuestionRound:
    """One completed semantic question round inside the Jev predictor."""

    number: int
    capabilities: tuple[str, ...]
    predictions: dict[str, CapabilityPrediction]
    answers: dict[str, dict[str, Any]]


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictorError(f"{label} must be a number", details={"field": label})
    number = float(value)
    if not math.isfinite(number):
        raise PredictorError(f"{label} must be finite", details={"field": label})
    return number


def _requestion_threshold(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ConfigurationError(f"{label} must be a number between 0 and 1")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{label} must be a number between 0 and 1") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ConfigurationError(f"{label} must be a number between 0 and 1")
    return number


def _normalize_requestion_prompts(
    rounds: Any,
) -> dict[int, dict[str, str]]:
    if not isinstance(rounds, Mapping):
        raise ConfigurationError("Jev re-question prompt YAML needs a rounds mapping")
    normalized: dict[int, dict[str, str]] = {}
    for round_number in REQUESTION_ROUNDS:
        definition = rounds.get(round_number, rounds.get(str(round_number)))
        if not isinstance(definition, Mapping):
            raise ConfigurationError(
                f"Jev re-question prompt YAML needs round {round_number}"
            )
        instructions = definition.get("instructions")
        if not isinstance(instructions, Mapping):
            raise ConfigurationError(
                f"Jev re-question round {round_number} needs an instructions mapping"
            )
        prompts: dict[str, str] = {}
        for capability, prompt in instructions.items():
            if not isinstance(capability, str) or not capability.strip():
                raise ConfigurationError(
                    f"Jev re-question round {round_number} has an invalid capability name"
                )
            if prompt is None:
                prompt = ""
            if not isinstance(prompt, str):
                raise ConfigurationError(
                    f"Jev re-question prompt for {capability!r} in round {round_number} must be a string"
                )
            prompts[capability] = prompt
        normalized[round_number] = prompts
    return normalized


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
        requestion_confidence_threshold: float = DEFAULT_REQUESTION_CONFIDENCE_THRESHOLD,
        requestion_prompts: Mapping[int | str, Any] | None = None,
        memory_instruction_template: str = DEFAULT_MEMORY_INSTRUCTION_TEMPLATE,
        print_io: bool = DEFAULT_PRINT_JEV_IO,
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
        prompt_config = load_jev_prompt_config(DEFAULT_REQUESTION_PROMPT_PATH)
        self.calibration_version = prompt_config["version"]
        self._default_questions = prompt_config["questions"]
        descriptions = dict(
            capability_descriptions if capability_descriptions is not None
            else capabilities if capabilities is not None
            else self._default_questions
        )
        for name, description in list(descriptions.items()):
            if description is None:
                if name not in self._default_questions:
                    raise ConfigurationError(f"capability {name!r} needs an explicit question and criteria")
                descriptions[name] = self._default_questions[name]
        if not descriptions or any(not isinstance(k, str) or not k.strip() for k in descriptions):
            raise ConfigurationError("capability_descriptions must contain named capabilities")
        threshold = _requestion_threshold(
            requestion_confidence_threshold,
            label="requestion_confidence_threshold",
        )
        if not isinstance(print_io, bool):
            raise ConfigurationError("print_io must be a boolean")
        if (
            not isinstance(memory_instruction_template, str)
            or not memory_instruction_template.strip()
        ):
            raise ConfigurationError("memory_instruction_template must be a non-empty string")
        if "{previous_answer}" not in memory_instruction_template:
            raise ConfigurationError(
                "memory_instruction_template must contain {previous_answer}"
            )
        prompts = (
            _normalize_requestion_prompts(prompt_config["rounds"])
            if requestion_prompts is None
            else _normalize_requestion_prompts(requestion_prompts)
        )

        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.timeout = timeout_number
        self._owns_client = client is None
        self.client = client if client is not None else httpx.Client()
        self.capability_descriptions = descriptions
        self.max_retries = max_retries
        self.retry_backoff_seconds = backoff
        self.requestion_confidence_threshold = threshold
        self.requestion_prompts = prompts
        self.memory_instruction_template = memory_instruction_template
        self.print_io = print_io

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
        requestion_confidence_threshold = _requestion_threshold(
            os.getenv(
                "JEV_REQUESTION_CONFIDENCE_THRESHOLD",
                str(DEFAULT_REQUESTION_CONFIDENCE_THRESHOLD),
            ),
            label="JEV_REQUESTION_CONFIDENCE_THRESHOLD",
        )
        return cls(
            key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            client=client,
            capability_descriptions=capability_descriptions,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            requestion_confidence_threshold=requestion_confidence_threshold,
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
                if name not in self._default_questions:
                    raise ConfigurationError(f"capability {name!r} needs descriptive criteria")
                criteria = list(self._default_questions[name]["criteria"])
                question["criteria"] = criteria
            if not isinstance(criteria, Sequence) or isinstance(criteria, (str, bytes)):
                raise ConfigurationError(f"criteria for {name!r} must be a sequence")
            if not 2 <= len(criteria) <= 10:
                raise ConfigurationError(f"criteria for {name!r} must contain 2 to 10 levels")
            if any(not isinstance(item, str) or not item.strip() for item in criteria):
                raise ConfigurationError(f"criteria for {name!r} must contain descriptive strings")
            questions[name] = question
        return questions

    def _print_event(
        self,
        event: str,
        *,
        round_number: int,
        capabilities: Sequence[str],
        payload: Any,
    ) -> None:
        if not self.print_io:
            return
        print(
            json.dumps(
                {
                    "event": event,
                    "round": round_number,
                    "capabilities": list(capabilities),
                    "payload": payload,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
            flush=True,
        )

    def _post(
        self,
        payload: dict[str, Any],
        *,
        round_number: int,
        capabilities: Sequence[str],
    ) -> Any:
        url = f"{self.base_url}/v1/systemone"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        self._print_event(
            "jev_request",
            round_number=round_number,
            capabilities=capabilities,
            payload=payload,
        )
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
                response_payload = response.json()
            except (ValueError, TypeError) as exc:
                raise PredictorError("Jev response was not valid JSON") from exc
            self._print_event(
                "jev_response",
                round_number=round_number,
                capabilities=capabilities,
                payload=response_payload,
            )
            return response_payload
        raise PredictorError("Jev request could not be completed", retryable=True, details={"status_code": last_status})

    def _round_questions(
        self,
        base_questions: Mapping[str, Mapping[str, Any]],
        capabilities: Sequence[str],
        *,
        round_number: int,
        previous_answers: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        questions: dict[str, dict[str, Any]] = {}
        prompts = self.requestion_prompts.get(round_number, {})
        for capability in capabilities:
            question = dict(base_questions[capability])
            if round_number in REQUESTION_ROUNDS:
                prompt = prompts.get(capability, "")
                if prompt.strip():
                    question["instructions"] = prompt
                if previous_answers is not None and capability in previous_answers:
                    previous_answer = json.dumps(
                        previous_answers[capability],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    memory = self.memory_instruction_template.replace(
                        "{previous_answer}", previous_answer
                    )
                    question["instructions"] = f"{memory}{question['instructions']}"
            questions[capability] = question
        return questions

    def _ask_round(
        self,
        task: str,
        base_questions: Mapping[str, Mapping[str, Any]],
        capabilities: Sequence[str],
        *,
        round_number: int,
        previous_answers: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> QuestionRound:
        questions = self._round_questions(
            base_questions,
            capabilities,
            round_number=round_number,
            previous_answers=previous_answers,
        )
        response = self._post(
            {"state": task, "model": self.model, "questions": questions},
            round_number=round_number,
            capabilities=capabilities,
        )
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
                details={
                    "round": round_number,
                    "missing": sorted(expected - actual),
                    "extra": sorted(actual - expected),
                },
            )

        predictions: dict[str, CapabilityPrediction] = {}
        round_answers: dict[str, dict[str, Any]] = {}
        for capability, question in questions.items():
            answer = answers[capability]
            if not isinstance(answer, Mapping) or answer.get("type") != "score":
                raise PredictorError(f"answer {capability!r} must be a score answer")
            criteria = question["criteria"]
            raw_score = _finite_number(
                answer.get("score"),
                label=f"answers.{capability}.score",
            )
            maximum = len(criteria) - 1
            if raw_score < 0 or raw_score > maximum:
                raise PredictorError(
                    f"answers.{capability}.score is outside its score levels",
                    details={"round": round_number, "min": 0, "max": maximum},
                )
            confidence = _finite_number(
                answer.get("confidence"),
                label=f"answers.{capability}.confidence",
            )
            if not 0 <= confidence <= 1:
                raise PredictorError(
                    f"answers.{capability}.confidence must be between 0 and 1"
                )
            predictions[capability] = CapabilityPrediction(
                requirement=raw_score / maximum,
                confidence=confidence,
                round_number=round_number,
            )
            round_answers[capability] = dict(answer)
        return QuestionRound(
            number=round_number,
            capabilities=tuple(capabilities),
            predictions=predictions,
            answers=round_answers,
        )

    def predict(self, task: str) -> RequirementPrediction:
        if not isinstance(task, str) or not task.strip():
            raise PredictorError("task must be a non-empty string")
        base_questions = self._questions()
        pending = tuple(base_questions)
        rounds: list[QuestionRound] = []

        for round_number in (1, *REQUESTION_ROUNDS):
            if not pending:
                break
            result = self._ask_round(
                task,
                base_questions,
                pending,
                round_number=round_number,
                previous_answers=rounds[-1].answers if rounds else None,
            )
            rounds.append(result)
            if round_number == REQUESTION_ROUNDS[-1]:
                break
            pending = tuple(
                capability
                for capability in result.capabilities
                if result.predictions[capability].confidence
                < self.requestion_confidence_threshold
            )

        selected: dict[str, CapabilityPrediction] = {}
        for result in rounds:
            for capability, candidate in result.predictions.items():
                current = selected.get(capability)
                if current is None or candidate.confidence > current.confidence:
                    selected[capability] = candidate

        self._print_event(
            "jev_prediction_summary",
            round_number=rounds[-1].number,
            capabilities=tuple(base_questions),
            payload={
                "calibration_version": self.calibration_version,
                "rounds": [
                    {"round": result.number, "scores": {
                        name: {"requirement_0_10": item.requirement * 10, "confidence": item.confidence}
                        for name, item in result.predictions.items()
                    }} for result in rounds
                ],
                "selected": {
                    name: {"round": item.round_number, "requirement_0_10": item.requirement * 10, "confidence": item.confidence}
                    for name, item in selected.items()
                },
            },
        )

        return RequirementPrediction(
            requirements={
                capability: selected[capability].requirement
                for capability in base_questions
            },
            confidences={
                capability: selected[capability].confidence
                for capability in base_questions
            },
        )
