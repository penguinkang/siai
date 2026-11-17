# Provider API

Import these objects from `siai`. Installing or importing the core package does
not import the OpenAI SDK.

## `OpenAIResponsesGenerator`

```python
OpenAIResponsesGenerator(
    model: str,
    *,
    client=None,
    base_url: str | None = None,
    timeout: float = 60.0,
    max_output_tokens: int = 1000,
    reasoning_effort: str | None = None,
)
```

A callable provider adapter for the OpenAI Responses API. Calling it with a
nonempty prompt returns the structured proposal as JSON text.

When `client` is omitted, the adapter lazily creates `openai.OpenAI`. `base_url`
and `timeout` are passed to that client. Supported reasoning-effort values are
`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, and `max`; actual model
support remains provider-dependent.

### `last_metadata`

`None` before the first completed API response, then a `ProviderMetadata` value.
Metadata is updated before empty-output validation so incomplete response details
remain available after `ProviderError`.

## `ProviderMetadata`

```python
ProviderMetadata(
    provider: str,
    model: str,
    response_id: str | None,
    status: str | None,
    usage: dict | None,
)
```

Auditable, JSON-serializable metadata attached to validated proposals.

## `ProviderError`

A `RuntimeError` subclass raised when the provider returns no usable proposal
text. Transport and authentication errors from the SDK propagate unchanged.

