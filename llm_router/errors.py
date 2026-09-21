"""Expected domain and operational errors with one public representation."""

from __future__ import annotations

from typing import Any

from .models import RouteError


class RouterError(Exception):
    """Base class for failures callers may safely receive as structured output."""

    code = "router_error"

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.details = details or {}

    def as_route_error(self) -> RouteError:
        return RouteError(
            code=self.code,
            message=str(self),
            retryable=self.retryable,
            details=self.details,
        )


class ConfigurationError(RouterError):
    code = "configuration_error"


class RegistryError(RouterError):
    code = "registry_error"


class PredictorError(RouterError):
    code = "predictor_error"


class RoutingError(RouterError):
    code = "routing_error"
