"""Optional model-provider adapters for intelligent proposals."""

from dataclasses import dataclass
from typing import Any


PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "lever": {"type": "string"},
        "value": {"type": ["string", "number", "boolean", "null"]},
        "hypothesis": {"type": "string"},
        "evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "expected_effect": {"type": "string"},
        "verification": {"type": "string"},
    },
    "required": [
        "lever",
        "value",
        "hypothesis",
        "evidence_ids",
        "expected_effect",
        "verification",
    ],
    "additionalProperties": False,
}


class ProviderError(RuntimeError):
    """Raised when a provider returns no usable proposal output."""


@dataclass(frozen=True)
class ProviderMetadata:
    """Auditable metadata from the most recent provider call."""

    provider: str
    model: str
    response_id: str | None
    status: str | None
    usage: dict[str, Any] | None


class OpenAIResponsesGenerator:
    """Generate strict proposal JSON through the OpenAI Responses API.

    The OpenAI SDK is imported only when no client is supplied. The SDK reads
    ``OPENAI_API_KEY`` through its normal configuration. Tests and compatible
    clients can be injected with ``client``.
    """

    def __init__(
        self,
        model: str,
        *,
        client=None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_output_tokens: int = 1000,
        reasoning_effort: str | None = None,
    ):
        if not model:
            raise ValueError("model is required")
        if timeout <= 0 or max_output_tokens < 1:
            raise ValueError("timeout and max_output_tokens must be positive")
        if reasoning_effort is not None and reasoning_effort not in {
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError("unsupported reasoning effort")
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self._client = client or self._create_client(base_url, timeout)
        self.last_metadata: ProviderMetadata | None = None

    @staticmethod
    def _create_client(base_url, timeout):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI support requires the optional dependency: "
                "pip install 'siai[openai]'"
            ) from exc
        options = {"timeout": timeout}
        if base_url is not None:
            options["base_url"] = base_url
        return OpenAI(**options)

    def __call__(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be nonempty text")
        request = {
            "model": self.model,
            "input": prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "siai_proposal",
                    "description": "One evidence-grounded model improvement proposal",
                    "strict": True,
                    "schema": PROPOSAL_SCHEMA,
                }
            },
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        if self.reasoning_effort is not None:
            request["reasoning"] = {"effort": self.reasoning_effort}

        response = self._client.responses.create(**request)
        self.last_metadata = ProviderMetadata(
            provider="openai",
            model=self.model,
            response_id=getattr(response, "id", None),
            status=getattr(response, "status", None),
            usage=self._usage(getattr(response, "usage", None)),
        )
        output = getattr(response, "output_text", None)
        if not isinstance(output, str) or not output.strip():
            raise ProviderError(
                "OpenAI response contained no usable text; inspect provider metadata "
                "for refusal or incomplete status"
            )
        return output

    @staticmethod
    def _usage(usage) -> dict[str, Any] | None:
        if usage is None:
            return None
        if isinstance(usage, dict):
            return dict(usage)
        model_dump = getattr(usage, "model_dump", None)
        if callable(model_dump):
            return model_dump()
        fields = ("input_tokens", "output_tokens", "total_tokens")
        result = {field: getattr(usage, field) for field in fields if hasattr(usage, field)}
        return result or None
