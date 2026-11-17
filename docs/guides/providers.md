# Connect a model provider

Any callable with the shape `generate(prompt: str) -> str` works with
`IntelligentProposer`. SIAI includes an optional OpenAI Responses API adapter for
strict structured proposals.

## Install

```bash
pip install "siai[openai]"
export OPENAI_API_KEY="..."
```

The OpenAI SDK reads `OPENAI_API_KEY` through its normal configuration. Do not
put credentials in model cards, observations, journals, or source files.

## Configure

```python
from siai import IntelligentProposer, OpenAIResponsesGenerator

generate = OpenAIResponsesGenerator(
    model="your-supported-model",
    max_output_tokens=1000,
    reasoning_effort="low",
)
proposer = IntelligentProposer(generate)
```

The adapter calls the Responses API with `store=False` and a strict JSON Schema
covering the proposal contract. It does not grant the model tools or access to
artifacts. SIAI still validates allowlisted levers, evidence IDs, nonempty text,
and duplicate candidates after schema generation.

The implementation follows the official
[Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create)
and [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)
contracts.

## Auditing

After each call, `generate.last_metadata` contains:

- provider and configured model;
- response ID and status; and
- token-usage fields returned by the SDK.

`IntelligentProposer` attaches this metadata to the `Proposal`, and
`ImprovementLoop` writes it inside the proposal field in `journal.jsonl`.
Raw prompts and provider responses are not duplicated into the journal because
the proposal request and validated output already capture the decision inputs and
result. Applications needing full provider traces should record them in a
restricted artifact store and reference that artifact separately.

## Incomplete output and refusals

Structured Outputs does not remove the need to handle refusals, incomplete
responses, API errors, or application-specific rules. If the SDK returns no
usable `output_text`, the adapter raises `ProviderError` after preserving provider
metadata. `ImprovementLoop` records the proposer error and stops without spending
training budget on an undefined candidate.

## Compatible clients

Pass `client=` to inject a configured SDK client, test double, or compatible
implementation exposing `client.responses.create(...)`. Compatibility requires
the Responses request and response shape used by this adapter; a custom base URL
alone does not guarantee provider compatibility.

